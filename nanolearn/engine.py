"""Teaching a computer from a table, and checking whether it learned anything.

Two engines live here:

* **Simple** — a small random forest written in plain Python. No install, works
  on any machine, good enough to be genuinely useful on ordinary spreadsheets.
* **Full** — scikit-learn, if it is present (nanolearn can install it for you
  with one click). Cross-validated, several model families, better scores.

Both report the same things: a plain-language score, a leaderboard, what
mattered most, and an honest comparison against guessing.
"""

from __future__ import annotations

import math
import os
import pickle
import random
import threading
import time
import uuid
from collections import Counter

# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------
def accuracy(y_true, y_pred) -> float:
    if len(y_true) == 0:
        return 0.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def confusion_matrix(y_true, y_pred, n_classes: int) -> list[list[int]]:
    matrix = [[0] * n_classes for _ in range(n_classes)]
    for actual, predicted in zip(y_true, y_pred):
        if 0 <= int(actual) < n_classes and 0 <= int(predicted) < n_classes:
            matrix[int(actual)][int(predicted)] += 1
    return matrix


def macro_f1(y_true, y_pred, n_classes: int) -> float:
    scores = []
    for label in range(n_classes):
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == label and b == label)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != label and b == label)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == label and b != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def mae(y_true, y_pred) -> float:
    if len(y_true) == 0:
        return 0.0
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true, y_pred) -> float:
    if len(y_true) == 0:
        return 0.0
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))


def r_squared(y_true, y_pred) -> float:
    if len(y_true) == 0:
        return 0.0
    mean = sum(y_true) / len(y_true)
    total = sum((value - mean) ** 2 for value in y_true)
    if total < 1e-12:
        return 0.0
    residual = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    return 1.0 - residual / total


def score_all(task: str, y_true, y_pred, classes: list) -> dict:
    # Callers may hand us plain lists or numpy arrays; make them uniform first.
    y_true = [float(value) for value in y_true]
    y_pred = [float(value) for value in y_pred]
    if task == "classification":
        return {
            "correct_pct": round(100.0 * accuracy(y_true, y_pred), 1),
            "accuracy": accuracy(y_true, y_pred),
            "macro_f1": round(macro_f1(y_true, y_pred, max(len(classes), 1)), 3),
            "confusion": confusion_matrix(y_true, y_pred, max(len(classes), 1)),
        }
    return {
        "mae": round(mae(y_true, y_pred), 4),
        "rmse": round(rmse(y_true, y_pred), 4),
        "r2": round(r_squared(y_true, y_pred), 3),
    }


# --------------------------------------------------------------------------
# A small decision tree, in plain Python
# --------------------------------------------------------------------------
class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "count")

    def __init__(self, value=None):
        self.feature = -1
        self.threshold = 0.0
        self.left: "_Node | None" = None
        self.right: "_Node | None" = None
        self.value = value
        self.count = 0


def _impurity(y, indices, task: str) -> float:
    """How mixed-up a group of rows is. 0 means perfectly pure."""
    if not indices:
        return 0.0
    values = [y[i] for i in indices]
    if task == "classification":
        counts = Counter(values)
        total = len(values)
        return 1.0 - sum((count / total) ** 2 for count in counts.values())  # Gini
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)        # variance


def _leaf_value(y, indices, task: str) -> float:
    values = [y[i] for i in indices]
    if task == "regression":
        return sum(values) / len(values)
    return Counter(values).most_common(1)[0][0]


class DecisionTree:
    """A single small tree. Not clever on its own — it is the forest that works."""

    def __init__(self, max_depth: int = 6, min_leaf: int = 3, feature_fraction: float = 0.7, seed: int = 0):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.feature_fraction = feature_fraction
        self.rng = random.Random(seed)
        self.root: _Node | None = None
        self.task = "classification"

    def fit(self, X, y, task: str, indices=None):
        self.task = task
        n_features = len(X[0]) if X else 0
        indices = list(range(len(X))) if indices is None else list(indices)
        picked = max(1, int(n_features * self.feature_fraction))
        self.features_per_node = max(1, min(n_features, picked))
        self.root = self._grow(X, y, indices, 0, n_features)
        return self

    def _grow(self, X, y, indices, depth, n_features) -> _Node:
        node = _Node(value=_leaf_value(y, indices, self.task))
        node.count = len(indices)
        if depth >= self.max_depth or len(indices) < 2 * self.min_leaf:
            return node
        if _impurity(y, indices, self.task) <= 1e-9:
            return node  # already pure

        features = self.rng.sample(range(n_features), min(self.features_per_node, n_features))
        best = None
        base = _impurity(y, indices, self.task)
        for feature in features:
            ordered = sorted((X[i][feature], i) for i in indices)
            thresholds = []
            for position in range(1, len(ordered)):
                if ordered[position][0] != ordered[position - 1][0]:
                    thresholds.append((ordered[position][0] + ordered[position - 1][0]) / 2.0)
            if not thresholds:
                continue
            # A handful of candidate cut points is plenty and keeps this fast.
            if len(thresholds) > 12:
                step = len(thresholds) / 12.0
                thresholds = [thresholds[min(len(thresholds) - 1, int(index * step))] for index in range(12)]
            for threshold in thresholds:
                left = [i for i in indices if X[i][feature] <= threshold]
                if len(left) < self.min_leaf or len(indices) - len(left) < self.min_leaf:
                    continue
                right = [i for i in indices if X[i][feature] > threshold]
                gain = base - (len(left) * _impurity(y, left, self.task)
                               + len(right) * _impurity(y, right, self.task)) / len(indices)
                if best is None or gain > best[0]:
                    best = (gain, feature, threshold, left, right)
        if best is None or best[0] <= 1e-12:
            return node

        _, feature, threshold, left, right = best
        node.feature = feature
        node.threshold = threshold
        node.left = self._grow(X, y, left, depth + 1, n_features)
        node.right = self._grow(X, y, right, depth + 1, n_features)
        node.value = None
        return node

    def predict_one(self, row) -> float:
        node = self.root
        while node is not None and node.value is None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value if node is not None else 0.0

    def predict(self, X) -> list:
        return [self.predict_one(row) for row in X]

    def importance(self, n_features: int) -> list:
        """How often each column was used to split, weighted by how many rows it split."""
        counts = [0.0] * n_features
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node is None or node.value is not None:
                continue
            counts[node.feature] += float(node.count)
            stack.append(node.left)
            stack.append(node.right)
        return counts


class SimpleForest:
    """A handful of small trees voting together — the 'Simple' engine.

    Written from scratch so the app never needs an install to be useful.
    """

    def __init__(self, n_trees: int = 15, max_depth: int = 6, seed: int = 0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed
        self.trees: list[DecisionTree] = []
        self.task = "classification"
        self.n_features = 0

    def fit(self, X, y, task: str):
        self.task = task
        self.n_features = len(X[0]) if X else 0
        rng = random.Random(self.seed)
        self.trees = []
        for number in range(self.n_trees):
            # Each tree sees a random sample of the rows, which is what makes
            # a crowd of weak trees better than any one of them.
            sample = [rng.randrange(len(X)) for _ in range(len(X))]
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_leaf=max(2, len(X) // 200),
                feature_fraction=0.6 if task == "classification" else 0.8,
                seed=self.seed + number * 17,
            )
            tree.fit(X, y, task, indices=sample)
            self.trees.append(tree)
        return self

    def predict(self, X) -> list:
        if not self.trees:
            return [0.0] * len(X)
        out = []
        for row in X:
            votes = [tree.predict_one(row) for tree in self.trees]
            if self.task == "regression":
                out.append(sum(votes) / len(votes))
            else:
                out.append(Counter(votes).most_common(1)[0][0])
        return out

    def predict_proba(self, X, n_classes: int) -> list:
        """How confident the crowd is, as a share of the vote."""
        out = []
        for row in X:
            votes = Counter(tree.predict_one(row) for tree in self.trees)
            total = sum(votes.values()) or 1
            out.append([votes.get(float(label), 0) / total for label in range(n_classes)])
        return out

    def importance(self, n_features: int) -> list:
        totals = [0.0] * n_features
        for tree in self.trees:
            for index, value in enumerate(tree.importance(n_features)):
                totals[index] += value
        return totals


# --------------------------------------------------------------------------
# Training, the simple way
# --------------------------------------------------------------------------
def _baseline_score(task: str, y_train, y_test, classes) -> float:
    """What you would get by always answering the same thing. The bar to beat."""
    if len(y_test) == 0:
        return 0.0
    if task == "classification":
        most_common = Counter(y_train).most_common(1)[0][0] if y_train else 0.0
        return round(100.0 * accuracy(y_test, [most_common] * len(y_test)), 1)
    mean = sum(y_train) / len(y_train) if y_train else 0.0
    return round(mae(y_test, [mean] * len(y_test)), 4)


def train_simple(data: dict, log, progress) -> dict:
    """The no-install route: build and score a small forest."""
    task = data["task"]
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    log("Using the built-in engine — nothing to install, works everywhere.")
    limit = 4000
    if len(X_train) > limit:
        log("That is a lot of rows for the built-in engine, so I am practising on %d of them."
            % limit)
        picked = random.Random(0).sample(range(len(X_train)), limit)
        X_train = [X_train[i] for i in picked]
        y_train = [y_train[i] for i in picked]

    baseline = _baseline_score(task, y_train, y_test, data["classes"])
    log("First, the dumb guess, so you have something to compare against: %s"
        % ("%.1f%% correct" % baseline if task == "classification" else "average error %s" % baseline))

    progress(15)
    log("Growing %d small trees…" % 15)
    started = time.time()
    forest = SimpleForest(n_trees=15, max_depth=6, seed=11).fit(X_train, y_train, task)
    progress(70)

    predictions = forest.predict(X_test)
    scores = score_all(task, y_test, predictions, data["classes"])
    log("Finished in %.1f seconds." % (time.time() - started))
    progress(90)

    importances = forest.importance(len(data["feature_names"]))
    total = sum(importances) or 1.0
    importance = sorted(
        ({"name": name, "weight": round(value / total, 4)}
         for name, value in zip(data["feature_names"], importances)),
        key=lambda item: -item["weight"],
    )[:20]

    return {
        "engine": "simple",
        "model": forest,
        "metrics": scores,
        "baseline": baseline,
        "importance": importance,
        "leaderboard": [{"name": "Small forest (built in)", "score": scores, "is_best": True}],
        "seconds": round(time.time() - started, 2),
    }


# --------------------------------------------------------------------------
# Training, the scikit-learn way
# --------------------------------------------------------------------------
def sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


def sklearn_version() -> str:
    try:
        import sklearn
        return sklearn.__version__
    except ImportError:
        return ""


def _sklearn_models(task: str, big: bool):
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    if task == "classification":
        models = [
            ("Logistic regression (a straight line)", LogisticRegression(max_iter=2000)),
            ("Decision tree", DecisionTreeClassifier(random_state=0, max_depth=8)),
            ("Random forest (200 trees)", RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)),
            ("Nearest neighbours", KNeighborsClassifier(n_neighbors=7)),
        ]
        if not big:
            models.insert(3, ("Boosted trees", GradientBoostingClassifier(random_state=0)))
        models.append(("Always guess the same", DummyClassifier(strategy="most_frequent")))
    else:
        models = [
            ("Straight-line fit", Ridge()),
            ("Decision tree", DecisionTreeRegressor(random_state=0, max_depth=8)),
            ("Random forest (200 trees)", RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)),
            ("Nearest neighbours", KNeighborsRegressor(n_neighbors=7)),
        ]
        if not big:
            models.insert(3, ("Boosted trees", GradientBoostingRegressor(random_state=0)))
        models.append(("Always guess the average", DummyRegressor(strategy="mean")))
    return models


def train_full(data: dict, log, progress) -> dict:
    """The scikit-learn route: try several families, keep the best."""
    import numpy as np
    from sklearn.model_selection import cross_val_score

    task = data["task"]
    X_train = np.asarray(data["X_train"], dtype=float)
    y_train = np.asarray(data["y_train"], dtype=float)
    X_test = np.asarray(data["X_test"], dtype=float)
    y_test = np.asarray(data["y_test"], dtype=float)

    big = len(X_train) > 5000
    if big:
        log("Big table (%d rows) — using the faster models." % len(X_train))

    scoring = "accuracy" if task == "classification" else "neg_mean_absolute_error"
    folds = 3 if (big or len(X_train) < 100) else 5
    log("Checking every model %d times on data it has not seen, so the scores are honest." % folds)

    models = _sklearn_models(task, big)
    leaderboard = []
    best_name, best_model, best_score, best_key = "", None, None, -1e18

    for position, (name, model) in enumerate(models):
        try:
            values = cross_val_score(model, X_train, y_train, cv=folds, scoring=scoring, n_jobs=1)
            average = float(values.mean())
            spread = float(values.std())
        except Exception as exc:  # a model that cannot handle this data should not stop the run
            log("· %s — skipped (%s)" % (name, str(exc)[:90]))
            continue
        key = average if task == "classification" else -average
        entry = {
            "name": name,
            "score": round(average, 4),
            "spread": round(spread, 4),
            "plain": ("%.1f%% correct" % (average * 100)) if task == "classification"
                     else ("average error %s" % abs(round(average, 3))),
            "is_best": False,
        }
        leaderboard.append(entry)
        log("· %s — %s" % (name, entry["plain"]))
        if key > best_key:
            best_key, best_name, best_model = key, name, model
        progress(20 + int(55 * (position + 1) / len(models)))

    if best_model is None:
        raise ValueError("None of the models could handle this table. Try a different column to predict.")

    log("Best so far: %s. Training it properly on all the practice data…" % best_name)
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)
    scores = score_all(task, y_test, [float(value) for value in predictions], data["classes"])
    for entry in leaderboard:
        entry["is_best"] = entry["name"] == best_name

    baseline = _baseline_score(task, list(y_train), list(y_test), data["classes"])

    # What mattered most: ask the model, then fall back to shuffling columns.
    importance = []
    names = data["feature_names"]
    try:
        if hasattr(best_model, "feature_importances_"):
            weights = list(best_model.feature_importances_)
        elif hasattr(best_model, "coef_"):
            coef = np.asarray(best_model.coef_)
            weights = list(np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef))
        else:
            weights = None
        if weights is not None and len(weights) == len(names):
            total = float(sum(weights)) or 1.0
            importance = sorted(
                ({"name": name, "weight": round(float(value) / total, 4)}
                 for name, value in zip(names, weights)),
                key=lambda item: -item["weight"],
            )[:20]
    except Exception:
        importance = []

    if not importance:
        log("Working out which columns matter by shuffling them one at a time…")
        try:
            from sklearn.inspection import permutation_importance
            result = permutation_importance(best_model, X_test, y_test, n_repeats=5,
                                            random_state=0, n_jobs=1)
            weights = [max(0.0, float(value)) for value in result.importances_mean]
            total = sum(weights) or 1.0
            importance = sorted(
                ({"name": name, "weight": round(value / total, 4)}
                 for name, value in zip(names, weights)),
                key=lambda item: -item["weight"],
            )[:20]
        except Exception as exc:
            log("(Could not work that out: %s)" % str(exc)[:80])

    progress(95)
    log("Done. Score below is on rows the model has never seen.")
    return {
        "engine": "scikit-learn",
        "model": best_model,
        "metrics": scores,
        "baseline": baseline,
        "importance": importance,
        "leaderboard": leaderboard,
        "best_model_name": best_name,
        "seconds": None,
    }


# --------------------------------------------------------------------------
# Jobs — training happens in the background so the page stays alive
# --------------------------------------------------------------------------
_jobs: dict[str, dict] = {}
_current: dict | None = None
MODEL_DIR_NAME = "models"


class Job:
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.state = "running"
        self.progress = 0
        self.logs: list[str] = []
        self.result: dict | None = None
        self.error = ""
        self.started = time.time()
        self.finished: float | None = None
        self._lock = threading.Lock()

    def log(self, message: str) -> None:
        with self._lock:
            self.logs.append(str(message))
            del self.logs[:-400]

    def set_progress(self, value: int) -> None:
        with self._lock:
            self.progress = max(self.progress, min(100, int(value)))

    def to_dict(self) -> dict:
        with self._lock:
            payload = {
                "id": self.id,
                "kind": self.kind,
                "state": self.state,
                "progress": self.progress,
                "logs": list(self.logs),
                "error": self.error,
                "seconds": round((self.finished or time.time()) - self.started, 1),
            }
            if self.result:
                payload["result"] = self.result
            return payload


def start_job(kind: str, work) -> Job:
    """Run ``work(job)`` in the background and return the job to poll."""
    job = Job(kind)
    _jobs[job.id] = job
    while len(_jobs) > 12:  # keep the last few only
        _jobs.pop(next(iter(_jobs)))

    def runner():
        global _current
        try:
            result = work(job)
            bundle = result.pop("model", None) if isinstance(result, dict) else None
            job.result = result
            if bundle is not None:
                _current = {
                    "run_id": job.id,
                    "model": bundle,
                    "task": result["task"],
                    "target": result["target"],
                    "classes": result["classes"],
                    "feature_names": result["feature_names"],
                    "encoders": result["encoders"],
                    "engine": result["engine"],
                    "metrics": result["metrics"],
                    "importance": result["importance"],
                }
                result["run_id"] = job.id
            job.state = "done"
            job.progress = 100
        except Exception as exc:
            job.state = "error"
            job.error = str(exc)
            job.log("Something went wrong: %s" % exc)
        finally:
            job.finished = time.time()

    threading.Thread(target=runner, daemon=True).start()
    return job


def get_job(job_id: str) -> dict | None:
    job = _jobs.get(job_id)
    return job.to_dict() if job else None


def current_model() -> dict | None:
    return _current


def set_current_model(bundle: dict | None) -> None:
    global _current
    _current = bundle


def model_path(directory: str, run_id: str) -> str:
    return os.path.join(directory, MODEL_DIR_NAME, "run-%s.model" % run_id)


def save_model(directory: str, bundle: dict) -> str:
    """Write the trained model to disk so it outlives this session."""
    folder = os.path.join(directory, MODEL_DIR_NAME)
    os.makedirs(folder, exist_ok=True)
    path = model_path(directory, bundle["run_id"])
    with open(path, "wb") as handle:
        pickle.dump(bundle, handle, protocol=4)
    return path


def load_model(path: str) -> dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def predict_with(bundle: dict, X: list) -> dict:
    """Predict, with the confidence numbers a person actually wants to see."""
    model = bundle["model"]
    task = bundle["task"]
    classes = bundle.get("classes") or []

    if bundle.get("engine") == "simple":
        raw = model.predict(X)
        if task == "classification":
            probabilities = model.predict_proba(X, max(len(classes), 1))
            out = []
            for index, value in enumerate(raw):
                label = classes[int(value)] if 0 <= int(value) < len(classes) else str(value)
                out.append({
                    "label": label,
                    "confidence": round(100.0 * max(probabilities[index]), 1) if probabilities[index] else None,
                    "probabilities": [
                        {"label": classes[position], "percent": round(100.0 * value, 1)}
                        for position, value in enumerate(probabilities[index])
                    ] if classes else [],
                })
            return {"task": task, "predictions": out}
        return {"task": task, "predictions": [{"value": round(float(value), 4)} for value in raw]}

    predictions = model.predict(X)
    if task == "classification":
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(X)
            except Exception:
                probabilities = None
        out = []
        for index, value in enumerate(predictions):
            label_index = int(value)
            entry = {
                "label": classes[label_index] if 0 <= label_index < len(classes) else str(value),
                "confidence": None,
                "probabilities": [],
            }
            if probabilities is not None and probabilities[index] is not None:
                row = list(probabilities[index])
                entry["confidence"] = round(100.0 * max(row), 1)
                entry["probabilities"] = [
                    {"label": classes[position] if position < len(classes) else str(position),
                     "percent": round(100.0 * float(value), 1)}
                    for position, value in enumerate(row)
                ]
            out.append(entry)
        return {"task": task, "predictions": out}
    return {"task": task, "predictions": [{"value": round(float(value), 4)} for value in predictions]}


def explain_score(task: str, metrics: dict, baseline) -> str:
    """One sentence a person can act on."""
    if task == "classification":
        correct = metrics.get("correct_pct", 0)
        if baseline is None:
            return "It gets %s%% of the test rows right." % correct
        difference = correct - baseline
        if difference >= 15:
            return ("It gets %s%% right, where always guessing the same answer would only get %s%%. "
                    "That is a real, useful signal." % (correct, baseline))
        if difference >= 5:
            return ("It gets %s%% right against %s%% for always guessing — better than chance, "
                    "but not enormously." % (correct, baseline))
        if difference > -1:
            return ("It gets %s%% right and always guessing gets %s%%. The columns in this file "
                    "do not really predict “%s” — that is a real finding, not a failure."
                    % (correct, baseline, "the target"))
        return "It did worse than guessing, which means this column cannot be predicted from the others."
    error = metrics.get("mae")
    r2 = metrics.get("r2", 0)
    if r2 >= 0.7:
        return "A strong fit: it explains about %d%% of the variation in the answer." % round(r2 * 100)
    if r2 >= 0.3:
        return "A useful fit: it explains about %d%% of the variation, with an average error of %s." % (
            round(r2 * 100), error)
    if r2 >= 0.05:
        return ("A weak fit: it explains only %d%% of the variation, with an average error of %s."
                % (round(r2 * 100), error))
    return ("Hardly any signal here: the other columns do not explain the answer, with an average "
            "error of %s." % error)
