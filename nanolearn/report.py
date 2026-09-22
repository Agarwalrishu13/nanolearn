"""A written report you can keep, print, or send to someone.

The point is that the result does not live only in a browser tab: after a run
nanoLearn can write a single HTML file containing the score, the leaderboard,
what mattered most, and what to do next — in plain words.
"""

from __future__ import annotations

import html
import time

STYLE = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0b0d12;
color:#e9edf6;margin:0;padding:48px 20px;line-height:1.6}
.page{max-width:820px;margin:0 auto}
h1{font-size:30px;margin:0 0 4px}h2{font-size:19px;margin:34px 0 10px}
.sub{color:#99a5bd;margin:0 0 30px}
.card{background:#141821;border:1px solid #262f42;border-radius:14px;padding:20px 22px;margin-bottom:16px}
.big{font-size:44px;font-weight:700;color:#ffb454;line-height:1.1}
.label{font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:#67718a}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #1e2536}
th{color:#67718a;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.06em}
tr.best td{color:#ffb454;font-weight:600}
.bar{background:#222a3c;border-radius:6px;height:9px;overflow:hidden;margin-top:3px}
.bar span{display:block;height:100%;background:linear-gradient(90deg,#ffb454,#f59243)}
code{font-family:ui-monospace,Menlo,Consolas,monospace;background:#222a3c;padding:2px 6px;border-radius:5px}
footer{color:#67718a;font-size:13px;margin-top:40px;border-top:1px solid #1e2536;padding-top:16px}
ul{padding-left:20px}
"""


def _e(value) -> str:
    return html.escape(str(value))


def _metric_cards(task: str, metrics: dict, baseline) -> str:
    if task == "classification":
        cards = [
            ("Correct on unseen data", "%s%%" % metrics.get("correct_pct", 0)),
            ("Always guessing would give", "%s%%" % baseline if baseline is not None else "—"),
            ("Balanced quality score", metrics.get("macro_f1", "—")),
            ("Answers to choose from", len(metrics.get("confusion") or []) or "—"),
        ]
    else:
        cards = [
            ("Typical error", metrics.get("mae", "—")),
            ("Worst-case swing", metrics.get("rmse", "—")),
            ("Explained", "%d%%" % round(100 * metrics.get("r2", 0))),
            ("Always guessing would give", baseline if baseline is not None else "—"),
        ]
    out = []
    for label, value in cards:
        out.append('<div class="card"><div class="label">%s</div><div class="big">%s</div></div>'
                   % (_e(label), _e(value)))
    return '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">%s</div>' % "".join(out)


def _leaderboard(rows: list) -> str:
    if not rows:
        return "<p>No comparison table for this run.</p>"
    body = []
    for row in rows:
        score = row.get("plain") or ", ".join("%s: %s" % (k, v) for k, v in (row.get("score") or {}).items())
        body.append(
            '<tr class="%s"><td>%s</td><td>%s</td></tr>'
            % ("best" if row.get("is_best") else "", _e(row.get("name", "")), _e(score))
        )
    return ("<table><thead><tr><th>Approach</th><th>Score</th></tr></thead><tbody>%s</tbody></table>"
            % "".join(body))


def _importance(rows: list) -> str:
    if not rows:
        return "<p>This model does not report which columns mattered most.</p>"
    biggest = max((row.get("weight") or 0) for row in rows) or 1.0
    out = []
    for row in rows[:10]:
        percent = round(100 * (row.get("weight") or 0) / biggest)
        out.append(
            '<div style="margin-bottom:10px"><div style="font-size:14px">%s <span style="color:#67718a">'
            '(%s%% of the influence)</span></div><div class="bar"><span style="width:%d%%"></span></div></div>'
            % (_e(row.get("name", "")), round(100 * (row.get("weight") or 0)), percent)
        )
    return "".join(out)


def build_report(dataset: dict, result: dict, settings: dict) -> str:
    """Return a complete, self-contained HTML report as a string."""
    task = result.get("task", "classification")
    target = result.get("target", "")
    metrics = result.get("metrics") or {}
    summary = dataset.get("summary") or {}
    columns = dataset.get("columns") or []

    description = result.get("explanation") or ""
    columns_rows = "".join(
        "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s%%</td></tr>"
        % (_e(column["name"]), _e(column["kind"]),
           _e(column["unique"]), _e(column["missing_pct"]))
        for column in columns[:30]
    )

    notes = result.get("dropped") or []
    notes_html = ""
    if notes:
        notes_html = ("<h2>Columns left out</h2><ul>%s</ul>"
                      "<p style='color:#99a5bd'>These were excluded because they cannot help predict "
                      "anything: identifiers are unique per row, and free text has no fixed set of values. "
                      "Leaving them in would make the score look better than it really is.</p>"
                      % "".join("<li>%s</li>" % _e(item) for item in notes))

    return """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>nanoLearn report — %(target)s</title><style>%(style)s</style></head>
<body><div class="page">
<h1>What nanoLearn found</h1>
<p class="sub">%(when)s · data file: %(file)s · engine: %(engine)s</p>

<div class="card">
  <div class="label">The question</div>
  <p style="font-size:17px;margin:6px 0 0">Predict <code>%(target)s</code> — %(task_words)s</p>
  <p style="color:#99a5bd;margin:8px 0 0">%(description)s</p>
</div>

<h2>The score</h2>
%(cards)s

<h2>What it tried</h2>
%(leaderboard)s

<h2>What matters most</h2>
%(importance)s

<h2>The data</h2>
<p>%(rows)s rows, %(columns)s columns · %(missing_pct)s%% of cells are empty</p>
<table><thead><tr><th>Column</th><th>Type</th><th>Different values</th><th>Empty</th></tr></thead>
<tbody>%(columns_rows)s</tbody></table>

%(notes)s

<h2>How to use this</h2>
<ul>
  <li>The score above is measured on rows that were held back while learning, so it is an honest estimate
      — not a promise. Real-world results are usually a little worse.</li>
  <li>If the score is close to “always guessing”, this file does not contain the information needed to
      predict <code>%(target)s</code>. That is a useful answer.</li>
  <li>The model is saved next to this report and can be reused without training again.</li>
</ul>

<footer>Made by nanoLearn — a no-code front end for ordinary machine learning.
Settings at the time: practiced on %(train_pct)s%% of the rows, tested on the rest.</footer>
</div></body></html>""" % {
        "style": STYLE,
        "when": time.strftime("%d %B %Y, %H:%M"),
        "file": _e(dataset.get("name", "—")),
        "engine": _e(result.get("engine", "—")),
        "target": _e(target),
        "task_words": "guessing a category" if task == "classification" else "predicting a number",
        "description": _e(description),
        "cards": _metric_cards(task, metrics, result.get("baseline")),
        "leaderboard": _leaderboard(result.get("leaderboard") or []),
        "importance": _importance(result.get("importance") or []),
        "rows": _e(summary.get("rows", "—")),
        "columns": _e(summary.get("columns", "—")),
        "missing_pct": _e(summary.get("missing_pct", 0)),
        "columns_rows": columns_rows,
        "notes": notes_html,
        "train_pct": _e(round(100 * (1 - float(settings.get("test_fraction", 0.2))))),
    }
