"""Reading and understanding a spreadsheet — with no libraries at all.

This module answers the questions a person would ask about an unfamiliar table:

* What is in here?          → :func:`profile`
* Which column should we predict?  → :func:`guess_target`
* Is that a number or a category?  → :func:`infer_kind`
* Turn it into something a learner can eat → :func:`build_features`

Being standard-library-only matters: the app has to work on a fresh Python
install, so there is no pandas here. Everything is plain lists and dicts.
"""

from __future__ import annotations

import csv
import io
import math
import os
import random
import re
from collections import Counter

MAX_ROWS = 200_000

# Column names that usually mean "this is the answer we want to predict".
TARGET_HINTS = (
    "target", "label", "class", "outcome", "result", "output", "answer", "y",
    "prediction", "price", "cost", "amount", "sales", "revenue", "profit",
    "churn", "survived", "survival", "bought", "purchased", "clicked", "spam",
    "fraud", "default", "risk", "score", "rating", "quality", "grade", "value",
    "salary", "wage", "weight", "height", "mpg", "temperature", "count", "demand",
)

# Column names that never are: identifiers, timestamps, free text.
IGNORE_HINTS = (
    "id", "uuid", "guid", "index", "idx", "row", "rowid", "key", "code",
    "name", "email", "phone", "address", "street", "url", "link", "note",
    "comment", "description", "text", "date", "time", "timestamp", "created",
    "updated", "datetime", "month", "year",
)

_ID_LIKE = re.compile(r"(^|[_\-\s])(id|uuid|guid|index|idx|no|number|code)($|[_\-\s])", re.I)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def read_csv_text(text: str, limit: int = MAX_ROWS) -> tuple[list[str], list[list[str]], str]:
    """Parse CSV text into ``(headers, rows, delimiter)``.

    Handles the usual mess: a byte-order mark, semicolons or tabs instead of
    commas, quoted fields, and rows that are shorter than the header.
    """
    text = text.lstrip("\ufeff")
    sample = text[:8192]
    delimiter = ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        # Count which separator appears most on the first line.
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        counts = {sep: first_line.count(sep) for sep in ",;\t|"}
        delimiter = max(counts, key=counts.get) if any(counts.values()) else ","

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = []
    headers: list[str] = []
    for position, raw in enumerate(reader):
        if position == 0:
            headers = [cell.strip() or "column_%d" % (index + 1) for index, cell in enumerate(raw)]
            # Drop a leading unnamed index column that spreadsheets love to add.
            if headers and headers[0].lower() in ("", "unnamed: 0", "index"):
                headers[0] = headers[0] or "index"
            continue
        if not any(cell.strip() for cell in raw):
            continue  # blank line
        rows.append([cell.strip() for cell in raw] + [""] * max(0, len(headers) - len(raw)))
        if len(rows) >= limit:
            break

    if not headers:
        raise ValueError("That file has no header row. The first line should be the column names.")
    return headers, rows, delimiter


def read_csv(path: str, limit: int = MAX_ROWS):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return read_csv_text(handle.read(), limit=limit)


def write_csv(path: str, headers: list[str], rows: list[list]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


# --------------------------------------------------------------------------
# Understanding
# --------------------------------------------------------------------------
def to_number(value) -> float | None:
    """Parse a number the way a spreadsheet writes one. None means 'not a number'."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None if isinstance(value, float) and math.isnan(value) else float(value)
    text = str(value).strip()
    if not text:
        return None
    # 1,234.5 / 1 234,5 / $42 / 12% / (3.5) for negatives
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = text.replace("$", "").replace("£", "").replace("€", "").replace("%", "").replace(" ", "")
    if text.count(",") and text.count(".") == 0:
        # Could be a thousands separator or a decimal comma: 1,5 means 1.5.
        if len(text.split(",")[-1]) == 3 and len(text.split(",")) > 1:
            text = text.replace(",", "")
        else:
            text = text.replace(",", ".")
    else:
        text = text.replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return None
    return -number if negative else number


def infer_kind(values: list[str], unique_count: int) -> str:
    """Classify a column as number / category / text / empty."""
    filled = [value for value in values if str(value).strip() != ""]
    if not filled:
        return "empty"
    numeric = sum(1 for value in filled if to_number(value) is not None)
    if numeric / len(filled) >= 0.9:
        return "number"
    # A handful of repeating values is something we can count and predict.
    if unique_count <= max(2, min(40, len(filled) // 2 or 2)):
        return "category"
    return "text"


def profile(headers: list[str], rows: list[list[str]]) -> list[dict]:
    """Describe every column: its type, its gaps, and what is in it."""
    columns = []
    total = max(len(rows), 1)
    for index, name in enumerate(headers):
        values = [row[index] if index < len(row) else "" for row in rows]
        filled = [value for value in values if str(value).strip() != ""]
        counts = Counter(filled)
        unique_count = len(counts)
        kind = infer_kind(values, unique_count)

        entry = {
            "name": name,
            "index": index,
            "kind": kind,
            "missing": len(values) - len(filled),
            "missing_pct": round(100.0 * (len(values) - len(filled)) / total, 1),
            "unique": unique_count,
            "sample": [value for value, _ in counts.most_common(5)],
            "top": [[value, count] for value, count in counts.most_common(8)],
            "looks_like_id": bool(_ID_LIKE.search(name)) or (unique_count == len(filled) and len(filled) > 20
                                                             and kind != "number"),
        }
        if kind == "number":
            numbers = [number for number in (to_number(value) for value in filled) if number is not None]
            if numbers:
                numbers.sort()
                entry.update({
                    "min": numbers[0],
                    "max": numbers[-1],
                    "mean": sum(numbers) / len(numbers),
                    "median": numbers[len(numbers) // 2],
                })
                ones = sum(1 for number in numbers if float(number).is_integer())
                entry["whole_numbers"] = ones == len(numbers)
        if kind == "category":
            entry["is_two_way"] = unique_count == 2
        columns.append(entry)
    return columns


def guess_target(columns: list[dict], row_count: int) -> tuple[str, str]:
    """Pick the column a person most likely wants predicted, and say why.

    Returns ``(name, reason)``. The reason is shown in the interface, because a
    guess that explains itself is a guess people will trust or correct.
    """
    best, best_score, best_reason = "", -999.0, ""
    for position, column in enumerate(columns):
        name = column["name"].lower().strip()
        score = 0.0
        reasons = []

        if column["kind"] == "empty":
            continue
        if column["missing_pct"] > 50:
            continue

        for hint in TARGET_HINTS:
            if name == hint:
                score += 6
                reasons.append("its name is “%s”" % column["name"])
                break
            if name.endswith("_" + hint) or name.startswith(hint + "_") or hint in name.split():
                score += 4
                reasons.append("its name contains “%s”" % hint)
                break
            if hint in name:
                score += 2
                reasons.append("its name mentions “%s”" % hint)
                break

        if any(re.search(r"(^|[_\-\s])%s($|[_\-\s])" % re.escape(hint), name) for hint in IGNORE_HINTS):
            score -= 8
            reasons.append("but it looks like an identifier")

        if column["kind"] == "category":
            if 2 <= column["unique"] <= 6:
                score += 3
                reasons.append("it has %d possible answers" % column["unique"])
            elif column["unique"] <= 20:
                score += 1.5
                reasons.append("it has %d possible answers" % column["unique"])
            else:
                score -= 2
        elif column["kind"] == "number":
            if column["unique"] > 20:
                score += 2.5
                reasons.append("it is a number with many different values")
            elif column["unique"] <= 5 and column.get("whole_numbers"):
                score += 2.5
                reasons.append("it is a small set of whole numbers")
            else:
                score += 1
        elif column["kind"] == "text":
            score -= 6
            reasons.append("it is free text, which is hard to predict")

        if column.get("looks_like_id"):
            score -= 6
        # The last column is the answer more often than chance.
        if position == len(columns) - 1:
            score += 1
        if column["missing_pct"] > 20:
            score -= 1

        if score > best_score:
            best, best_score, best_reason = column["name"], score, ", ".join(reasons)

    if not best:
        return "", "I could not work out which column you want to predict — please choose one."
    reason = "I picked “%s” because %s." % (best, best_reason) if best_reason else \
             "I picked “%s” — it is the last column." % best
    return best, reason


def guess_task(column: dict) -> str:
    """'classification' (pick a label) or 'regression' (predict a number)."""
    if column is None:
        return "classification"
    if column["kind"] == "category":
        return "classification"
    if column["kind"] == "number":
        # A whole-number column with few distinct values behaves like labels.
        if column.get("whole_numbers") and column["unique"] <= 12:
            return "classification"
        return "regression"
    return "classification"


# --------------------------------------------------------------------------
# Turning a table into features
# --------------------------------------------------------------------------
def build_features(columns: list[dict], headers: list[str], rows: list[list[str]], target: str) -> dict:
    """Convert the table into numbers a learner can use.

    Numbers are filled in and standardised; small sets of words become
    yes/no columns; identifiers and free text are left out, and the caller is
    told what was dropped so it can be explained to the user.
    """
    target_column = next((column for column in columns if column["name"] == target), None)
    if target_column is None:
        raise ValueError("I could not find a column called “%s”." % target)

    dropped: list[str] = []
    feature_columns = []
    for column in columns:
        if column["name"] == target:
            continue
        if column["kind"] == "empty":
            dropped.append("%s (empty)" % column["name"])
            continue
        if column["kind"] == "text":
            dropped.append("%s (free text)" % column["name"])
            continue
        if column.get("looks_like_id"):
            dropped.append("%s (identifier)" % column["name"])
            continue
        if column["kind"] == "category" and column["unique"] > 30:
            dropped.append("%s (too many different values)" % column["name"])
            continue
        if column["missing_pct"] > 40:
            dropped.append("%s (mostly empty)" % column["name"])
            continue
        feature_columns.append(column)

    target_index = target_column["index"]
    usable_rows = []
    y_raw = []
    for row in rows:
        value = row[target_index] if target_index < len(row) else ""
        if str(value).strip() == "":
            continue
        usable_rows.append(row)
        y_raw.append(value)
    if not usable_rows:
        raise ValueError("Every row is missing a value in “%s”, so there is nothing to learn from." % target)

    task = guess_task(target_column)

    # --- target
    classes: list[str] = []
    if task == "classification":
        counts = Counter(str(value) for value in y_raw)
        classes = [value for value, _ in counts.most_common()]
        mapping = {label: index for index, label in enumerate(classes)}
        y = [float(mapping[str(value)]) for value in y_raw]
    else:
        y = []
        for value in y_raw:
            number = to_number(value)
            if number is None:
                raise ValueError("“%s” mostly looks like numbers, but I found “%s” in it, which is not one."
                                 % (target, value))
            y.append(number)

    # --- features
    feature_names: list[str] = []
    matrix: list[list[float]] = [[] for _ in usable_rows]
    encoders: dict[str, dict] = {}

    for column in feature_columns:
        index = column["index"]
        values = [(row[index] if index < len(row) else "") for row in usable_rows]

        if column["kind"] == "number":
            numbers = [to_number(value) for value in values]
            present = [number for number in numbers if number is not None]
            mean = sum(present) / len(present) if present else 0.0
            filled = [mean if number is None else number for number in numbers]
            spread = (sum((value - mean) ** 2 for value in present) / len(present)) ** 0.5 if len(present) > 1 else 0.0
            scale = spread if spread > 1e-9 else 1.0
            for row_index, value in enumerate(filled):
                matrix[row_index].append((value - mean) / scale)
            feature_names.append(column["name"])
            encoders[column["name"]] = {"type": "number", "mean": mean, "scale": scale,
                                        "min": column.get("min"), "max": column.get("max")}
        else:
            categories = [value for value, _ in Counter(values).most_common(30)]
            for category in categories:
                feature_names.append("%s = %s" % (column["name"], category))
                for row_index, value in enumerate(values):
                    matrix[row_index].append(1.0 if value == category else 0.0)
            encoders[column["name"]] = {"type": "category", "categories": categories}

    if not feature_names:
        raise ValueError("There is nothing left to learn from — every other column is an identifier or empty.")

    return {
        "feature_names": feature_names,
        "X": matrix,
        "y": y,
        "y_raw": [str(value) for value in y_raw],
        "task": task,
        "classes": classes,
        "encoders": encoders,
        "dropped": dropped,
        "target": target,
        "rows_used": len(usable_rows),
        "row_indexes": list(range(len(usable_rows))),
        "usable_rows": usable_rows,
    }


def split(X: list, y: list, test_fraction: float = 0.2, seed: int = 7) -> dict:
    """Split into a practice set and an unseen test set.

    The test set is what makes the reported score honest — it is data the model
    never saw while learning.
    """
    indices = list(range(len(X)))
    random.Random(seed).shuffle(indices)
    cut = max(1, min(len(indices) - 1, int(len(indices) * (1.0 - test_fraction)))) if len(indices) > 3 else len(indices)
    train_index, test_index = indices[:cut], indices[cut:]
    return {
        "X_train": [X[i] for i in train_index],
        "y_train": [y[i] for i in train_index],
        "X_test": [X[i] for i in test_index],
        "y_test": [y[i] for i in test_index],
        "test_rows": test_index,
        "train_count": len(train_index),
        "test_count": len(test_index),
    }


def encode_row(encoders: dict, feature_names: list[str], values: dict) -> list[float]:
    """Turn one person's filled-in form into the same numbers the model saw."""
    row = []
    for name in feature_names:
        if " = " in name:
            column_name, category = name.split(" = ", 1)
            encoder = encoders.get(column_name, {})
            row.append(1.0 if str(values.get(column_name, "")).strip() == category else 0.0)
        else:
            encoder = encoders.get(name, {})
            number = to_number(values.get(name, ""))
            if number is None:
                number = encoder.get("mean", 0.0)
            scale = encoder.get("scale") or 1.0
            row.append((number - encoder.get("mean", 0.0)) / scale)
    return row


def chart_for(column: dict, rows: list[list[str]], buckets: int = 12) -> dict:
    """A small chart description the page draws as SVG."""
    index = column["index"]
    values = [row[index] if index < len(row) else "" for row in rows]
    filled = [value for value in values if str(value).strip() != ""]

    if column["kind"] == "number":
        numbers = [number for number in (to_number(value) for value in filled) if number is not None]
        if not numbers:
            return {"kind": "empty"}
        low, high = min(numbers), max(numbers)
        if high - low < 1e-9:
            return {"kind": "bar", "labels": [format_number(low)], "values": [len(numbers)],
                    "caption": "Every value is the same"}
        step = (high - low) / buckets
        counts = [0] * buckets
        for number in numbers:
            position = min(buckets - 1, int((number - low) / step))
            counts[position] += 1
        labels = [format_number(low + step * i) for i in range(buckets)]
        return {"kind": "bar", "labels": labels, "values": counts,
                "caption": "%d values from %s to %s" % (len(numbers), format_number(low), format_number(high))}

    pairs = Counter(filled).most_common(10)
    return {
        "kind": "bar",
        "labels": [value for value, _ in pairs],
        "values": [count for _, count in pairs],
        "caption": "%d different values" % column["unique"],
    }


def format_number(value) -> str:
    """Write a number the way a person would say it."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1000:
        return "{:,.0f}".format(number) if abs(number) >= 10_000 else "{:,.1f}".format(number)
    if number == int(number):
        return str(int(number))
    return ("%.2f" % number).rstrip("0").rstrip(".")


def dataset_summary(headers: list[str], rows: list[list[str]], columns: list[dict]) -> dict:
    """A friendly paragraph about the file."""
    countable = [column for column in columns if column["kind"] not in ("empty",)]
    gaps = sum(column["missing"] for column in countable)
    total_cells = len(rows) * max(len(headers), 1)
    return {
        "rows": len(rows),
        "columns": len(headers),
        "numbers": sum(1 for column in countable if column["kind"] == "number"),
        "categories": sum(1 for column in countable if column["kind"] == "category"),
        "text": sum(1 for column in countable if column["kind"] == "text"),
        "empty_columns": sum(1 for column in columns if column["kind"] == "empty"),
        "missing_cells": gaps,
        "missing_pct": round(100.0 * gaps / total_cells, 1) if total_cells else 0.0,
    }


def preview(headers: list[str], rows: list[list[str]], limit: int = 8) -> dict:
    return {"headers": headers, "rows": rows[:limit], "shown": min(limit, len(rows))}


def file_size_mb(path: str) -> float:
    try:
        return round(os.path.getsize(path) / (1024 * 1024), 2)
    except OSError:
        return 0.0
