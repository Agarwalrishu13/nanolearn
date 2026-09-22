"""Where nanoLearn keeps uploads, trained models and sample data.

One folder in the home directory, plain files, no database. Delete it and the
app forgets everything.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import threading
from pathlib import Path

APP_DIR_NAME = ".nanolearn"
_lock = threading.Lock()

DEFAULT_SETTINGS = {
    "engine": "auto",        # "auto" | "simple" | "full"
    "test_fraction": 0.2,
    "seed": 7,
    "completed_steps": 0,
}


def data_dir() -> Path:
    override = os.environ.get("NANOLEARN_HOME")
    path = Path(override) if override else Path.home() / APP_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def uploads_dir() -> Path:
    path = data_dir() / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def reports_dir() -> Path:
    path = data_dir() / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ------------------------------------------------------------------- settings
def _settings_path() -> Path:
    return data_dir() / "settings.json"


def load_settings() -> dict:
    with _lock:
        try:
            with open(_settings_path(), "r", encoding="utf-8") as handle:
                stored = json.load(handle)
        except (OSError, json.JSONDecodeError):
            stored = {}
    merged = dict(DEFAULT_SETTINGS)
    merged.update({key: value for key, value in stored.items() if key in DEFAULT_SETTINGS})
    return merged


def save_settings(patch: dict) -> dict:
    with _lock:
        current = dict(DEFAULT_SETTINGS)
        try:
            with open(_settings_path(), "r", encoding="utf-8") as handle:
                current.update({k: v for k, v in json.load(handle).items() if k in DEFAULT_SETTINGS})
        except (OSError, json.JSONDecodeError):
            pass
        for key, value in patch.items():
            if key in DEFAULT_SETTINGS:
                current[key] = value
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=str(_settings_path().parent),
            prefix="settings.", suffix=".tmp", delete=False,
        )
        try:
            with handle:
                json.dump(current, handle, indent=2)
            os.replace(handle.name, _settings_path())
        except Exception:
            try:
                os.unlink(handle.name)
            except OSError:
                pass
            raise
    return current


# -------------------------------------------------------------- example data
# Fisher's iris measurements — the classic beginner dataset, public domain.
IRIS = """sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
4.7,3.2,1.3,0.2,setosa
4.6,3.1,1.5,0.2,setosa
5.0,3.6,1.4,0.2,setosa
5.4,3.9,1.7,0.4,setosa
4.6,3.4,1.4,0.3,setosa
5.0,3.4,1.5,0.2,setosa
4.4,2.9,1.4,0.2,setosa
4.9,3.1,1.5,0.1,setosa
5.4,3.7,1.5,0.2,setosa
4.8,3.4,1.6,0.2,setosa
4.8,3.0,1.4,0.1,setosa
4.3,3.0,1.1,0.1,setosa
5.8,4.0,1.2,0.2,setosa
5.7,4.4,1.5,0.4,setosa
5.4,3.9,1.3,0.4,setosa
5.1,3.5,1.4,0.3,setosa
5.7,3.8,1.7,0.3,setosa
5.1,3.8,1.5,0.3,setosa
5.4,3.4,1.7,0.2,setosa
5.1,3.7,1.5,0.4,setosa
4.6,3.6,1.0,0.2,setosa
5.1,3.3,1.7,0.5,setosa
4.8,3.4,1.9,0.2,setosa
5.0,3.0,1.6,0.2,setosa
5.0,3.4,1.6,0.4,setosa
5.2,3.5,1.5,0.2,setosa
5.2,3.4,1.4,0.2,setosa
4.7,3.2,1.6,0.2,setosa
4.8,3.1,1.6,0.2,setosa
5.4,3.4,1.5,0.4,setosa
5.2,4.1,1.5,0.1,setosa
5.5,4.2,1.4,0.2,setosa
4.9,3.1,1.5,0.2,setosa
5.0,3.2,1.2,0.2,setosa
5.5,3.5,1.3,0.2,setosa
4.9,3.6,1.4,0.1,setosa
4.4,3.0,1.3,0.2,setosa
5.1,3.4,1.5,0.2,setosa
5.0,3.5,1.3,0.3,setosa
4.5,2.3,1.3,0.3,setosa
4.4,3.2,1.3,0.2,setosa
5.0,3.5,1.6,0.6,setosa
5.1,3.8,1.9,0.4,setosa
4.8,3.0,1.4,0.3,setosa
5.1,3.8,1.6,0.2,setosa
4.6,3.2,1.4,0.2,setosa
5.3,3.7,1.5,0.2,setosa
5.0,3.3,1.4,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
6.4,3.2,4.5,1.5,versicolor
6.9,3.1,4.9,1.5,versicolor
5.5,2.3,4.0,1.3,versicolor
6.5,2.8,4.6,1.5,versicolor
5.7,2.8,4.5,1.3,versicolor
6.3,3.3,4.7,1.6,versicolor
4.9,2.4,3.3,1.0,versicolor
6.6,2.9,4.6,1.3,versicolor
5.2,2.7,3.9,1.4,versicolor
5.0,2.0,3.5,1.0,versicolor
5.9,3.0,4.2,1.5,versicolor
6.0,2.2,4.0,1.0,versicolor
6.1,2.9,4.7,1.4,versicolor
5.6,2.9,3.6,1.3,versicolor
6.7,3.1,4.4,1.4,versicolor
5.6,3.0,4.5,1.5,versicolor
5.8,2.7,4.1,1.0,versicolor
6.2,2.2,4.5,1.5,versicolor
5.6,2.5,3.9,1.1,versicolor
5.9,3.2,4.8,1.8,versicolor
6.1,2.8,4.0,1.3,versicolor
6.3,2.5,4.9,1.5,versicolor
6.1,2.8,4.7,1.2,versicolor
6.4,2.9,4.3,1.3,versicolor
6.6,3.0,4.4,1.4,versicolor
6.8,2.8,4.8,1.4,versicolor
6.7,3.0,5.0,1.7,versicolor
6.0,2.9,4.5,1.5,versicolor
5.7,2.6,3.5,1.0,versicolor
5.5,2.4,3.8,1.1,versicolor
5.5,2.4,3.7,1.0,versicolor
5.8,2.7,3.9,1.2,versicolor
6.0,2.7,5.1,1.6,versicolor
5.4,3.0,4.5,1.5,versicolor
6.0,3.4,4.5,1.6,versicolor
6.7,3.1,4.7,1.5,versicolor
6.3,2.3,4.4,1.3,versicolor
5.6,3.0,4.1,1.3,versicolor
5.5,2.5,4.0,1.3,versicolor
5.5,2.6,4.4,1.2,versicolor
6.1,3.0,4.6,1.4,versicolor
5.8,2.6,4.0,1.2,versicolor
5.0,2.3,3.3,1.0,versicolor
5.6,2.7,4.2,1.3,versicolor
5.7,3.0,4.2,1.2,versicolor
5.7,2.9,4.2,1.3,versicolor
6.2,2.9,4.3,1.3,versicolor
5.1,2.5,3.0,1.1,versicolor
5.7,2.8,4.1,1.3,versicolor
6.3,3.3,6.0,2.5,virginica
5.8,2.7,5.1,1.9,virginica
7.1,3.0,5.9,2.1,virginica
6.3,2.9,5.6,1.8,virginica
6.5,3.0,5.8,2.2,virginica
7.6,3.0,6.6,2.1,virginica
4.9,2.5,4.5,1.7,virginica
7.3,2.9,6.3,1.8,virginica
6.7,2.5,5.8,1.8,virginica
7.2,3.6,6.1,2.5,virginica
6.5,3.2,5.1,2.0,virginica
6.4,2.7,5.3,1.9,virginica
6.8,3.0,5.5,2.1,virginica
5.7,2.5,5.0,2.0,virginica
5.8,2.8,5.1,2.4,virginica
6.4,3.2,5.3,2.3,virginica
6.5,3.0,5.5,1.8,virginica
7.7,3.8,6.7,2.2,virginica
7.7,2.6,6.9,2.3,virginica
6.0,2.2,5.0,1.5,virginica
6.9,3.2,5.7,2.3,virginica
5.6,2.8,4.9,2.0,virginica
7.7,2.8,6.7,2.0,virginica
6.3,2.7,4.9,1.8,virginica
6.7,3.3,5.7,2.1,virginica
7.2,3.2,6.0,1.8,virginica
6.2,2.8,4.8,1.8,virginica
6.1,3.0,4.9,1.8,virginica
6.4,2.8,5.6,2.1,virginica
7.2,3.0,5.8,1.6,virginica
7.4,2.8,6.1,1.9,virginica
7.9,3.8,6.4,2.0,virginica
6.4,2.8,5.6,2.2,virginica
6.3,2.8,5.1,1.5,virginica
6.1,2.6,5.6,1.4,virginica
7.7,3.0,6.1,2.3,virginica
6.3,3.4,5.6,2.4,virginica
6.4,3.1,5.5,1.8,virginica
6.0,3.0,4.8,1.8,virginica
6.9,3.1,5.4,2.1,virginica
6.7,3.1,5.6,2.4,virginica
6.9,3.1,5.1,2.3,virginica
5.8,2.7,5.1,1.9,virginica
6.8,3.2,5.9,2.3,virginica
6.7,3.3,5.7,2.5,virginica
6.7,3.0,5.2,2.3,virginica
6.3,2.5,5.0,1.9,virginica
6.5,3.0,5.2,2.0,virginica
6.2,3.4,5.4,2.3,virginica
5.9,3.0,5.1,1.8,virginica
"""


def sample_iris() -> str:
    return IRIS


def sample_houses() -> str:
    """A made-up house-price table, so there is a numbers example to try.

    Generated with a fixed seed and a known formula, so the numbers are stable
    and the "what matters most" panel has an honest right answer (size).
    """
    rng = random.Random(1234)
    lines = ["size_sqm,bedrooms,age_years,distance_to_centre_km,garage,price"]
    for _ in range(120):
        size = round(rng.uniform(45, 220), 1)
        bedrooms = max(1, min(6, int(size / 35) + rng.randint(-1, 1)))
        age = rng.randint(0, 70)
        distance = round(rng.uniform(0.5, 25), 1)
        garage = rng.choice(["yes", "no"])
        price = (1200 * size
                 + 9000 * bedrooms
                 - 700 * age
                 - 4200 * distance
                 + (18000 if garage == "yes" else 0)
                 + rng.gauss(0, 14000))
        lines.append("%s,%d,%d,%s,%s,%d" % (size, bedrooms, age, distance, garage, max(30000, price)))
    return "\n".join(lines) + "\n"


SAMPLES = {
    "flowers": {
        "file": "flowers.csv",
        "title": "Flower measurements",
        "blurb": "150 flowers, measured. Guess the species from the measurements.",
        "text": sample_iris,
    },
    "houses": {
        "file": "house-prices.csv",
        "title": "House prices",
        "blurb": "120 made-up houses. Guess the price from size, age and distance.",
        "text": sample_houses,
    },
}


def sample_path(name: str) -> Path | None:
    entry = SAMPLES.get(name)
    if not entry:
        return None
    path = data_dir() / "samples" / entry["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_text(entry["text"](), encoding="utf-8")
    return path
