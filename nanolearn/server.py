"""Every address the page talks to.

The shape of a session: **inspect** a spreadsheet → **teach** on it (as a
background job, so the page never freezes) → **try it** on made-up values or a
new file → **download** the report, the predictions and the model.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from . import APP_NAME, __version__, engine, report, store, table
from .httpbase import App, Bytes, Error, Json, Stream

WEB_DIR = Path(__file__).parent / "web"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._\- +]+")

_uploads: dict[str, dict] = {}
_uploads_lock = threading.Lock()

_cache_lock = threading.Lock()
_dataset_cache: dict[str, tuple] = {}

# Notes about the most recent runs, so the report and the examples panel can be
# built without repeating the training work.
_run_notes: dict[str, dict] = {}


def _safe_filename(name: str) -> str:
    cleaned = _SAFE_NAME.sub("_", os.path.basename(name or "")).strip(" .")
    return cleaned[:180] or "data.csv"


# --------------------------------------------------------------------------
# Reading a spreadsheet
# --------------------------------------------------------------------------
def _load_table(path: str, limit: int = table.MAX_ROWS):
    """Read a CSV, remembering the last few so the page feels instant."""
    try:
        stamp = (os.path.getmtime(path), os.path.getsize(path))
    except OSError:
        raise ValueError("I cannot find that file any more. Please drop it in again.")

    key = os.path.abspath(path)
    with _cache_lock:
        cached = _dataset_cache.get(key)
        if cached and cached[0] == stamp:
            return cached[1]

    headers, rows, delimiter = table.read_csv(path, limit=limit)
    columns = table.profile(headers, rows)
    value = (headers, rows, columns, delimiter)
    with _cache_lock:
        _dataset_cache[key] = (stamp, value)
        while len(_dataset_cache) > 4:
            _dataset_cache.pop(next(iter(_dataset_cache)))
    return value


def _describe(path: str, name: str = "", limit: int = table.MAX_ROWS) -> dict:
    """Everything the page shows after a file arrives."""
    headers, rows, columns, delimiter = _load_table(path, limit=limit)
    if not rows:
        raise ValueError("That file has a heading row but no data underneath it. "
                         "I need at least a couple of rows to learn from.")
    if len(headers) < 2:
        raise ValueError("That file only has one column, so there is nothing to learn from. "
                         "I need the column you want predicted plus at least one more that might explain it.")

    target, reason = table.guess_target(columns, len(rows))
    target_column = next((column for column in columns if column["name"] == target), None)
    task = table.guess_task(target_column)

    charts = {}
    if target_column:
        charts[target_column["name"]] = table.chart_for(target_column, rows)
    for column in columns:
        if len(charts) >= 6:
            break
        if column["name"] not in charts and column["kind"] in ("number", "category"):
            charts[column["name"]] = table.chart_for(column, rows)

    return {
        "path": path,
        "name": name or os.path.basename(path),
        "delimiter": "tab" if delimiter == "\t" else delimiter,
        "summary": table.dataset_summary(headers, rows, columns),
        "columns": columns,
        "preview": table.preview(headers, rows),
        "target_guess": {"name": target, "reason": reason, "task": task},
        "charts": charts,
    }


def _encode_rows(rows: list, headers: list, bundle: dict) -> list:
    """Turn rows of a *new* file into numbers, using the trained model's recipe."""
    positions = {name: headers.index(name) for name in bundle["encoders"] if name in headers}
    if not positions:
        # Nothing matched by name — assume the columns are in the same order.
        positions = {name: index for index, name in enumerate(bundle["encoders"])}
    out = []
    for row in rows:
        values = {name: (row[position] if position < len(row) else "") for name, position in positions.items()}
        out.append(table.encode_row(bundle["encoders"], bundle["feature_names"], values))
    return out


def _examples(bundle: dict, notes: dict, limit: int = 6) -> list:
    """A few held-back rows: the truth, the machine's guess, and whether it was right."""
    X_test = notes.get("X_test") or []
    y_test = notes.get("y_test") or []
    if not X_test:
        return []
    X_test, y_test = X_test[:limit], y_test[:limit]
    predictions = engine.predict_with(bundle, X_test)["predictions"]
    rows = []
    for position, item in enumerate(predictions):
        truth = y_test[position]
        if bundle["task"] == "classification":
            label = bundle["classes"][int(truth)] if 0 <= int(truth) < len(bundle["classes"]) else str(truth)
            rows.append({
                "actual": label,
                "predicted": item.get("label"),
                "correct": label == item.get("label"),
                "confidence": item.get("confidence"),
            })
        else:
            guess = float(item.get("value") or 0)
            rows.append({
                "actual": table.format_number(truth),
                "predicted": table.format_number(guess),
                "correct": None,
                "difference": table.format_number(abs(float(truth) - guess)),
            })
    return rows


# --------------------------------------------------------------------------
# Training (runs on a background thread)
# --------------------------------------------------------------------------
def _train_work(payload: dict):
    def work(job):
        path, target, task = payload["path"], payload["target"], payload["task"]
        choice = payload.get("engine", "auto")
        settings = store.load_settings()

        job.log("Reading %s…" % os.path.basename(path))
        headers, rows, columns, _ = _load_table(path)
        job.set_progress(5)
        job.log("Found %d rows and %d columns." % (len(rows), len(headers)))

        job.log("Turning the table into numbers the computer can work with…")
        data = table.build_features(columns, headers, rows, target)
        job.set_progress(12)
        job.log("Using %d rows — the rest had nothing in the “%s” column." % (data["rows_used"], target))
        if data["dropped"]:
            job.log("Leaving out: %s" % ", ".join(data["dropped"]))

        split = table.split(data["X"], data["y"],
                            test_fraction=float(settings.get("test_fraction", 0.2)),
                            seed=int(settings.get("seed", 7)))
        data.update(split)
        job.log("Keeping %d rows back for the exam, practising on %d."
                % (split["test_count"], split["train_count"]))

        use_full = True if choice == "full" else (engine.sklearn_available() if choice == "auto" else False)
        if use_full and not engine.sklearn_available():
            job.log("The full engine is not installed, so I am using the built-in one. "
                    "You can install it later — it is optional.")
            use_full = False

        if use_full:
            job.log("Full engine detected (scikit-learn %s)." % engine.sklearn_version())
            trained = engine.train_full(data, job.log, job.set_progress)
        else:
            trained = engine.train_simple(data, job.log, job.set_progress)

        explanation = engine.explain_score(task, trained["metrics"], trained.get("baseline"))
        job.log(explanation)

        classes = data["classes"] if task == "classification" else []
        bundle = {
            "run_id": job.id,
            "engine": trained["engine"],
            "task": task,
            "target": target,
            "classes": classes,
            "feature_names": data["feature_names"],
            "encoders": data["encoders"],
            "model": trained["model"],
        }
        try:
            engine.save_model(str(store.data_dir()), bundle)
        except Exception as exc:
            job.log("(Could not save the model to disk: %s)" % exc)

        _run_notes[job.id] = {
            "X_test": data["X_test"],
            "y_test": data["y_test"],
            "leaderboard": trained["leaderboard"],
            "baseline": trained.get("baseline"),
            "explanation": explanation,
            "dropped": data["dropped"],
        }
        while len(_run_notes) > 6:
            _run_notes.pop(next(iter(_run_notes)))

        return {
            "engine": trained["engine"],
            "task": task,
            "target": target,
            "classes": classes,
            "feature_names": data["feature_names"],
            "encoders": data["encoders"],
            "metrics": trained["metrics"],
            "baseline": trained.get("baseline"),
            "importance": trained["importance"],
            "leaderboard": trained["leaderboard"],
            "best_model_name": trained.get("best_model_name", "Small forest (built in)"),
            "explanation": explanation,
            "rows_used": data["rows_used"],
            "train_count": split["train_count"],
            "test_count": split["test_count"],
            "dropped": data["dropped"],
            "examples": _examples(bundle, _run_notes[job.id]),
            "model": trained["model"],   # removed by the job runner before sending
        }

    return work


# --------------------------------------------------------------------------
def create_app() -> App:
    app = App(APP_NAME, WEB_DIR, __version__)

    # ------------------------------------------------------------------ status
    @app.get("/api/health")
    def health(_request):
        return Json({"ok": True, "app": APP_NAME, "version": __version__})

    @app.get("/api/status")
    def status(_request):
        return Json({
            "app": APP_NAME,
            "version": __version__,
            "settings": store.load_settings(),
            "full_engine": {
                "available": engine.sklearn_available(),
                "version": engine.sklearn_version(),
                "packages": ["scikit-learn", "pandas", "numpy"],
            },
            "python": sys.version.split()[0],
            "data_folder": str(store.data_dir()),
            "samples": [{"id": key, "title": value["title"], "blurb": value["blurb"]}
                        for key, value in store.SAMPLES.items()],
            "model_ready": engine.current_model() is not None,
        })

    @app.get("/api/settings")
    def get_settings(_request):
        return Json(store.load_settings())

    @app.post("/api/settings")
    def set_settings(request):
        patch = request.json()
        if not isinstance(patch, dict):
            return Error("Settings must be named values.")
        return Json(store.save_settings(patch))

    # ------------------------------------------------------------------ upload
    @app.post("/api/upload/start")
    def upload_start(request):
        name = _safe_filename(str(request.json().get("name", "data.csv")))
        upload_id = uuid.uuid4().hex
        target = store.uploads_dir() / (upload_id + ".part")
        target.write_bytes(b"")
        with _uploads_lock:
            _uploads[upload_id] = {"name": name, "path": target, "received": 0}
        return Json({"id": upload_id, "name": name, "chunk_bytes": 4 * 1024 * 1024})

    @app.post("/api/upload/chunk")
    def upload_chunk(request):
        upload_id = request.q("id", "")
        with _uploads_lock:
            entry = _uploads.get(upload_id)
        if not entry:
            return Error("That upload expired. Start again.")
        try:
            offset = int(request.q("offset", "-1"))
        except ValueError:
            return Error("Bad offset.")
        if offset != entry["received"]:
            return Error("The pieces arrived out of order.", 409, expected=entry["received"])
        with open(entry["path"], "ab") as handle:
            handle.write(request.body)
        entry["received"] += len(request.body)
        return Json({"received": entry["received"]})

    @app.post("/api/upload/finish")
    def upload_finish(request):
        upload_id = str(request.json().get("id", ""))
        with _uploads_lock:
            entry = _uploads.pop(upload_id, None)
        if not entry:
            return Error("That upload expired. Start again.")
        destination = store.uploads_dir() / entry["name"]
        os.replace(entry["path"], destination)
        # Check it really is a readable table before claiming success.
        try:
            described = _describe(str(destination), entry["name"])
        except ValueError as exc:
            return Error(str(exc))
        described["message"] = "Read %s — %d rows, %d columns." % (
            entry["name"], described["summary"]["rows"], described["summary"]["columns"])
        return Json(described)

    # ----------------------------------------------------------------- samples
    @app.get("/api/samples")
    def samples(_request):
        return Json({"samples": [{"id": key, "title": value["title"], "blurb": value["blurb"]}
                                 for key, value in store.SAMPLES.items()]})

    @app.post("/api/samples/load")
    def sample_load(request):
        name = str(request.json().get("id", ""))
        path = store.sample_path(name)
        if not path:
            return Error("I do not have an example called that.")
        described = _describe(str(path), path.name)
        described["message"] = "%s loaded — %d rows." % (
            store.SAMPLES[name]["title"], described["summary"]["rows"])
        return Json(described)

    @app.post("/api/inspect")
    def inspect(request):
        path = str(request.json().get("path", ""))
        if not path or not os.path.isfile(path):
            return Error("I cannot find that file. Please drop it in again.")
        try:
            return Json(_describe(path))
        except ValueError as exc:
            return Error(str(exc))

    @app.get("/api/files")
    def files(_request):
        found = []
        for folder, label in ((store.uploads_dir(), ""), (store.data_dir() / "samples", "  (example)")):
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*.csv")):
                found.append({"name": path.name + label, "path": str(path),
                              "size_mb": table.file_size_mb(str(path))})
        return Json({"files": found})

    # ------------------------------------------------------------------- teach
    @app.post("/api/train")
    def train(request):
        body = request.json()
        path = str(body.get("path", ""))
        target = str(body.get("target", ""))
        task = str(body.get("task", "classification"))
        if not path or not os.path.isfile(path):
            return Error("I cannot find that file any more.")
        if not target:
            return Error("Choose which column you would like predicted.")
        if task not in ("classification", "regression"):
            return Error("The answer must be either a category or a number.")
        job = engine.start_job("train", _train_work({
            "path": path, "target": target, "task": task, "engine": str(body.get("engine", "auto")),
        }))
        return Json({"job_id": job.id, "message": "Teaching started."})

    @app.get("/api/job/{job_id}")
    def job_status(request):
        job_id = request.params["job_id"]
        payload = engine.get_job(job_id)
        if not payload:
            return Error("I lost track of that run. Start it again.", 404)
        bundle = engine.current_model()
        if (payload["state"] == "done" and payload.get("result")
                and bundle and bundle["run_id"] == job_id):
            payload["result"]["examples"] = _examples(bundle, _run_notes.get(job_id, {}))
        return Json(payload)

    @app.get("/api/model")
    def model_info(_request):
        bundle = engine.current_model()
        if not bundle:
            return Json({"ready": False, "message": "Nothing has been taught yet."})
        return Json({
            "ready": True,
            "run_id": bundle["run_id"],
            "target": bundle["target"],
            "task": bundle["task"],
            "engine": bundle["engine"],
            "classes": bundle["classes"],
            "metrics": bundle["metrics"],
            "importance": bundle["importance"],
            "fields": [
                {"name": name, "type": value["type"],
                 "categories": value.get("categories", [])[:40],
                 "min": value.get("min"), "max": value.get("max")}
                for name, value in bundle["encoders"].items()
            ],
        })

    # ----------------------------------------------------------------- predict
    @app.post("/api/predict")
    def predict(request):
        bundle = engine.current_model()
        if not bundle:
            return Error("Teach it something first — then it can answer questions.")
        values = request.json().get("values") or {}
        if not isinstance(values, dict):
            return Error("The values should be column names and answers.")
        row = table.encode_row(bundle["encoders"], bundle["feature_names"], values)
        result = engine.predict_with(bundle, [row])
        answer = result["predictions"][0] if result["predictions"] else {}
        return Json({
            "task": result["task"],
            "target": bundle["target"],
            "answer": answer,
            "sentence": _sentence(bundle, answer),
        })

    @app.post("/api/predict/file")
    def predict_file(request):
        bundle = engine.current_model()
        if not bundle:
            return Error("Teach it something first — then it can go through a whole file.")
        path = str(request.json().get("path", ""))
        if not path or not os.path.isfile(path):
            return Error("I cannot find that file.")
        headers, rows, _columns, _ = _load_table(path)
        try:
            X = _encode_rows(rows, headers, bundle)
            result = engine.predict_with(bundle, X)
        except Exception as exc:
            return Error("I could not read that file: %s" % exc)

        extra = (["nanoLearn_guess"] if result["task"] == "regression"
                 else ["nanoLearn_guess", "nanoLearn_confidence_pct"])
        out_headers = list(headers) + extra
        out_rows = []
        for position, row in enumerate(rows):
            prediction = result["predictions"][position] if position < len(result["predictions"]) else {}
            if result["task"] == "classification":
                cells = [prediction.get("label", ""), prediction.get("confidence", "")]
            else:
                cells = [prediction.get("value", "")]
            out_rows.append(list(row) + cells)

        folder = store.data_dir() / "predictions"
        folder.mkdir(parents=True, exist_ok=True)
        name = _safe_filename("guesses-" + os.path.basename(path))
        table.write_csv(str(folder / name), out_headers, out_rows)
        return Json({
            "name": name,
            "download": "/api/download/predictions/" + name,
            "preview": {"headers": out_headers, "rows": out_rows[:8]},
            "count": len(out_rows),
            "sentence": "Added a guess for all %d rows — the new columns are at the right-hand end."
                        % len(out_rows),
        })

    # ------------------------------------------------------------------ report
    @app.post("/api/report")
    def make_report(request):
        bundle = engine.current_model()
        if not bundle:
            return Error("Teach it something first — then there is a report to write.")
        path = str(request.json().get("path", ""))
        try:
            dataset = _describe(path) if path and os.path.isfile(path) else {
                "name": bundle["target"], "summary": {}, "columns": []}
        except ValueError:
            dataset = {"name": bundle["target"], "summary": {}, "columns": []}

        notes = _run_notes.get(bundle["run_id"], {})
        html = report.build_report(dataset, {
            "task": bundle["task"],
            "target": bundle["target"],
            "metrics": bundle["metrics"],
            "importance": bundle["importance"],
            "engine": bundle["engine"],
            "leaderboard": notes.get("leaderboard", []),
            "baseline": notes.get("baseline"),
            "explanation": notes.get("explanation", ""),
            "dropped": notes.get("dropped", []),
        }, store.load_settings())

        name = _safe_filename("report-%s.html" % bundle["run_id"])
        (store.reports_dir() / name).write_text(html, encoding="utf-8")
        return Json({"name": name, "download": "/api/download/reports/" + name,
                     "message": "Report written."})

    @app.get("/api/download/{kind}/{name}")
    def download(request):
        folders = {
            "models": store.models_dir(),
            "reports": store.reports_dir(),
            "predictions": store.data_dir() / "predictions",
            "uploads": store.uploads_dir(),
        }
        folder = folders.get(request.params["kind"])
        if folder is None:
            return Error("There is nothing to download there.", 404)
        name = _safe_filename(request.params["name"])
        path = (folder / name).resolve()
        try:
            path.relative_to(folder.resolve())
        except ValueError:
            return Error("Not allowed.", 403)
        if not path.is_file():
            return Error("That file is gone.", 404)
        if name.endswith(".html"):
            content_type = "text/html; charset=utf-8"
        elif name.endswith(".csv"):
            content_type = "text/csv; charset=utf-8"
        else:
            content_type = "application/octet-stream"
        return Bytes(path.read_bytes(), content_type)

    # --------------------------------------------------------- optional engine
    @app.post("/api/install")
    def install(_request):
        """Install the optional full engine, with the output visible as it runs."""
        def events():
            if engine.sklearn_available():
                yield {"type": "done", "message": "The full engine is already installed "
                                                  "(scikit-learn %s) — there is nothing to do."
                                                  % engine.sklearn_version()}
                return
            command = [sys.executable, "-m", "pip", "install", "--upgrade",
                       "scikit-learn", "pandas", "numpy"]
            yield {"type": "log", "message": "Running: %s" % " ".join(command)}
            yield {"type": "log", "message": "This downloads roughly 100 MB. It only adds Python "
                                             "packages; nothing else on your computer changes."}
            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, bufsize=1, errors="replace")
            except OSError as exc:
                yield {"type": "error", "message": "The installer would not start: %s" % exc}
                return
            assert process.stdout is not None
            for line in iter(process.stdout.readline, ""):
                line = line.rstrip()
                if line:
                    yield {"type": "log", "message": line}
            process.wait()
            if process.returncode == 0:
                yield {"type": "done", "message": "Installed. Close this window and start nanoLearn again "
                                                  "so it picks up the new engine."}
            else:
                yield {"type": "error", "message": "The install stopped with code %d. The built-in engine "
                                                   "still works, so you lose nothing." % process.returncode}

        return Stream.sse(events())

    @app.get("/api/about")
    def about(_request):
        return Json({
            "app": APP_NAME,
            "version": __version__,
            "data_folder": str(store.data_dir()),
            "offline": True,
            "siblings": [
                {"name": "nanolaama", "url": "https://github.com/Agarwalrishu13/nanolaama",
                 "what": "talk to a local AI"},
                {"name": "nanobuild", "url": "https://github.com/Agarwalrishu13/nanobuild",
                 "what": "make a project without coding"},
                {"name": "nanollama.c", "url": "https://github.com/Agarwalrishu13/nanollama.c",
                 "what": "the C inference engine"},
                {"name": "nanobrain", "url": "https://github.com/Agarwalrishu13/nanobrain",
                 "what": "the from-scratch trainer"},
            ],
        })

    return app


def _sentence(bundle: dict, answer: dict) -> str:
    """The one-line answer a person reads first."""
    target = bundle["target"]
    if bundle["task"] == "classification":
        label = answer.get("label", "—")
        confidence = answer.get("confidence")
        if confidence is None:
            return "It thinks “%s” is %s." % (target, label)
        if confidence >= 80:
            return "It is fairly sure: %s = %s (%s%% confident)." % (target, label, confidence)
        if confidence >= 60:
            return "It leans towards %s = %s (%s%% confident)." % (target, label, confidence)
        return ("It is not sure — the best it can say is %s = %s, and it is only %s%% confident. "
                "The other columns may not carry enough information." % (target, label, confidence))
    value = answer.get("value")
    return "Its guess for %s is %s." % (target, table.format_number(value))
