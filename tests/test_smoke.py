"""Tests for nanoLearn.

The interesting ones are not "does the endpoint return 200" but "does the model
actually learn something". The iris and house examples have known structure, so
the tests can assert real accuracy thresholds rather than just shapes.
"""

import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TMP = tempfile.mkdtemp(prefix="nanolearn-tests-")
os.environ["NANOLEARN_HOME"] = _TMP
os.environ["NANOLEARN_TEST"] = "1"

from nanolearn import engine, store, table  # noqa: E402
from nanolearn.httpbase import free_port  # noqa: E402
from nanolearn.server import create_app  # noqa: E402


# ==========================================================================
# Reading numbers and tables
# ==========================================================================
class TestParsing(unittest.TestCase):
    def test_numbers_as_people_write_them(self):
        cases = {
            "42": 42.0, "3.5": 3.5, "1,234.5": 1234.5, "1,5": 1.5, "$42": 42.0,
            "12%": 12.0, "(3.5)": -3.5, " 7 ": 7.0, "1 234": 1234.0, "": None,
            "n/a": None, "seven": None, None: None, "0": 0.0, "-2.5": -2.5,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(table.to_number(text), expected)

    def test_reading_a_british_style_file(self):
        headers, rows, delimiter = table.read_csv_text("a;b;c\n1;2;3\n4;5;6\n")
        self.assertEqual(delimiter, ";")
        self.assertEqual(headers, ["a", "b", "c"])
        self.assertEqual(rows[1], ["4", "5", "6"])

    def test_reading_a_file_with_a_byte_order_mark_and_ragged_rows(self):
        headers, rows, _ = table.read_csv_text("\ufeffname,score\nalice,10\nbob\n")
        self.assertEqual(headers, ["name", "score"])
        self.assertEqual(rows[1], ["bob", ""])      # the short row is padded, not lost

    def test_a_file_with_only_a_header_is_an_error(self):
        with self.assertRaises(ValueError):
            table.read_csv_text("")

    def test_types_are_worked_out(self):
        self.assertEqual(table.infer_kind(["1", "2", "3"], 3), "number")
        self.assertEqual(table.infer_kind(["yes", "no", "yes"], 2), "category")
        self.assertEqual(table.infer_kind(["", "", ""], 0), "empty")
        long_texts = ["sentence number %d with words" % i for i in range(50)]
        self.assertEqual(table.infer_kind(long_texts, 50), "text")


class TestGuessing(unittest.TestCase):
    def setUp(self):
        headers, rows, _ = table.read_csv_text(store.sample_iris())
        self.columns = table.profile(headers, rows)
        self.headers, self.rows = headers, rows

    def test_it_finds_the_species_column(self):
        target, reason = table.guess_target(self.columns, len(self.rows))
        self.assertEqual(target, "species")
        self.assertIn("species", reason)

    def test_it_knows_that_is_a_category(self):
        species = next(column for column in self.columns if column["name"] == "species")
        self.assertEqual(table.guess_task(species), "classification")

    def test_it_knows_a_price_is_a_number(self):
        headers, rows, _ = table.read_csv_text(store.sample_houses())
        columns = table.profile(headers, rows)
        target, _ = table.guess_target(columns, len(rows))
        self.assertEqual(target, "price")
        price = next(column for column in columns if column["name"] == "price")
        self.assertEqual(table.guess_task(price), "regression")

    def test_features_are_standardised_and_one_hot(self):
        data = table.build_features(self.columns, self.headers, self.rows, "species")
        self.assertEqual(data["task"], "classification")
        self.assertEqual(data["classes"], ["setosa", "versicolor", "virginica"])
        self.assertEqual(len(data["X"]), len(data["y"]))
        self.assertEqual(len(data["X"][0]), len(data["feature_names"]))
        # Numbers arrive standardised: near zero mean, near one spread.
        first = [row[0] for row in data["X"]]
        self.assertLess(abs(sum(first) / len(first)), 0.2)

    def test_identifiers_and_free_text_are_left_out(self):
        text = "id,notes,visits,label\n" + "\n".join(
            "%d,this is a long free text note %d,%d,yes" % (i, i, i) for i in range(30))
        headers, rows, _ = table.read_csv_text(text)
        columns = table.profile(headers, rows)
        data = table.build_features(columns, headers, rows, "label")
        dropped = " ".join(data["dropped"])
        self.assertIn("id", dropped)
        self.assertIn("notes", dropped)
        self.assertIn("visits", data["feature_names"])       # the useful column survives
        self.assertNotIn("id", data["feature_names"])


# ==========================================================================
# Learning — the built-in engine
# ==========================================================================
class TestSimpleEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.headers, cls.rows, _delimiter = table.read_csv(str(store.sample_path("flowers")))
        cls.columns = table.profile(cls.headers, cls.rows)
        cls.iris = table.build_features(cls.columns, cls.headers, cls.rows, "species")
        cls.iris.update(table.split(cls.iris["X"], cls.iris["y"], 0.25, seed=3))

        cls.h_headers, cls.h_rows, _delimiter = table.read_csv(str(store.sample_path("houses")))
        cls.h_columns = table.profile(cls.h_headers, cls.h_rows)
        cls.houses = table.build_features(cls.h_columns, cls.h_headers, cls.h_rows, "price")
        cls.houses.update(table.split(cls.houses["X"], cls.houses["y"], 0.25, seed=3))

    def _train(self, data):
        logs = []
        result = engine.train_simple(data, logs.append, lambda value: None)
        return result, logs

    def test_it_classifies_flowers_well(self):
        result, logs = self._train(self.iris)
        accuracy = result["metrics"]["accuracy"]
        self.assertGreater(accuracy, 0.8, "iris should be easy: got %s\n%s" % (accuracy, "\n".join(logs)))
        self.assertIn("correct_pct", result["metrics"])
        self.assertEqual(result["engine"], "simple")

    def test_it_beats_guessing(self):
        result, _ = self._train(self.iris)
        self.assertGreater(result["metrics"]["correct_pct"], result["baseline"])

    def test_it_predicts_house_prices(self):
        result, logs = self._train(self.houses)
        self.assertGreater(result["metrics"]["r2"], 0.6,
                           "house prices are generated from a formula, so this should fit well:\n"
                           + "\n".join(logs))

    def test_it_says_which_columns_matter(self):
        result, _ = self._train(self.houses)
        top = [row["name"] for row in result["importance"][:3]]
        self.assertTrue(any("size" in name for name in top),
                        "house size should be the strongest signal, got %s" % top)

    def test_it_reports_probabilities_that_sum_to_about_one(self):
        result, _ = self._train(self.iris)
        check = engine.predict_with({
            "engine": "simple", "task": "classification", "classes": self.iris["classes"],
            "model": result["model"],
        }, self.iris["X_test"][:3])
        self.assertEqual(len(check["predictions"]), 3)
        for prediction in check["predictions"]:
            total = sum(item["percent"] for item in prediction["probabilities"])
            self.assertAlmostEqual(total, 100.0, delta=1.0)

    def test_a_column_with_no_signal_is_reported_as_such(self):
        """A random label must not be presented as a discovery."""
        import random
        rng = random.Random(5)
        headers = ["a", "b", "c", "coin"]
        rows = [[str(rng.random()), str(rng.random()), str(rng.randint(0, 9)),
                 rng.choice(["heads", "tails"])] for _ in range(200)]
        columns = table.profile(headers, rows)
        data = table.build_features(columns, headers, rows, "coin")
        data.update(table.split(data["X"], data["y"], 0.25, seed=1))
        result = engine.train_simple(data, lambda message: None, lambda value: None)
        sentence = engine.explain_score("classification", result["metrics"], result["baseline"])
        self.assertTrue(sentence)
        # Any edge over guessing should be tiny on pure noise.
        self.assertLess(result["metrics"]["correct_pct"] - result["baseline"], 25)


# ==========================================================================
# Learning — the full engine, when it exists
# ==========================================================================
@unittest.skipUnless(engine.sklearn_available(), "scikit-learn is not installed")
class TestFullEngine(unittest.TestCase):
    def test_it_beats_the_baseline_on_iris(self):
        headers, rows, _delimiter = table.read_csv(str(store.sample_path("flowers")))
        columns = table.profile(headers, rows)
        data = table.build_features(columns, headers, rows, "species")
        data.update(table.split(data["X"], data["y"], 0.25, seed=3))
        result = engine.train_full(data, lambda message: None, lambda value: None)
        self.assertGreaterEqual(result["metrics"]["accuracy"], 0.9)
        self.assertGreater(len(result["leaderboard"]), 2)
        self.assertTrue(any(row["is_best"] for row in result["leaderboard"]))
        self.assertTrue(result["importance"])


# ==========================================================================
# The app itself
# ==========================================================================
class ServerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.port = free_port(8792)
        cls.base = "http://127.0.0.1:%d" % cls.port
        threading.Thread(target=cls.app.serve, kwargs={
            "host": "127.0.0.1", "port": cls.port, "open_browser": False, "quiet": True,
        }, daemon=True).start()
        for _ in range(80):
            try:
                cls.get("/api/health")
                return
            except Exception:
                time.sleep(0.05)
        raise RuntimeError("the test server never came up")

    @classmethod
    def tearDownClass(cls):
        cls.app.shutdown()

    @classmethod
    def get(cls, path, raw=False):
        try:
            with urllib.request.urlopen(cls.base + path, timeout=60) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read()          # so tests can read the error body too
        return body if raw else json.loads(body.decode("utf-8"))

    @classmethod
    def post(cls, path, payload=None, raw=False):
        request = urllib.request.Request(
            cls.base + path, data=json.dumps(payload or {}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read()
        return body if raw else json.loads(body.decode("utf-8"))

    @classmethod
    def upload(cls, text, name="table.csv"):
        start = cls.post("/api/upload/start", {"name": name})
        request = urllib.request.Request(
            cls.base + "/api/upload/chunk?id=%s&offset=0" % start["id"],
            data=text.encode("utf-8"), method="POST")
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
        return cls.post("/api/upload/finish", {"id": start["id"]})

    @classmethod
    def train(cls, path, target, task, engine_choice="simple", timeout=180):
        started = cls.post("/api/train", {"path": path, "target": target, "task": task,
                                          "engine": engine_choice})
        deadline = time.time() + timeout
        while time.time() < deadline:
            payload = cls.get("/api/job/" + started["job_id"])
            if payload["state"] == "done":
                return payload["result"]
            if payload["state"] == "error":
                raise AssertionError("training failed: %s\n%s" % (payload["error"], "\n".join(payload["logs"])))
            time.sleep(0.25)
        raise AssertionError("training did not finish in time")


class TestEndpoints(ServerTestCase):
    def test_health_and_status(self):
        self.assertTrue(self.get("/api/health")["ok"])
        status = self.get("/api/status")
        self.assertIn("full_engine", status)
        self.assertGreaterEqual(len(status["samples"]), 2)

    def test_index_and_assets_are_served(self):
        self.assertIn("nanoLearn", self.get("/", raw=True).decode("utf-8"))
        self.assertIn("--accent", self.get("/style.css", raw=True).decode("utf-8"))
        self.assertIn("renderChart", self.get("/app.js", raw=True).decode("utf-8"))

    def test_unknown_api_path_is_a_json_404(self):
        data = self.get("/api/nope")
        self.assertIn("error", data)

    def test_loading_an_example(self):
        data = self.post("/api/samples/load", {"id": "flowers"})
        self.assertEqual(data["target_guess"]["name"], "species")
        self.assertEqual(data["summary"]["rows"], 150)
        self.assertTrue(data["charts"])

    def test_uploading_a_spreadsheet(self):
        data = self.upload("colour,size,liked\nred,3,yes\nblue,5,no\nred,4,yes\nblue,6,no\n")
        self.assertEqual(data["message"].split("—")[0].strip(), "Read table.csv")
        self.assertEqual(data["summary"]["rows"], 4)
        self.assertEqual(data["columns"][0]["name"], "colour")

    def test_uploading_something_that_is_not_a_table(self):
        data = self.upload("this is just a sentence with no commas at all")
        self.assertIn("error", data)

    def test_settings_round_trip(self):
        saved = self.post("/api/settings", {"test_fraction": 0.3})
        self.assertEqual(saved["test_fraction"], 0.3)
        self.post("/api/settings", {"test_fraction": 0.2})


class TestTeachingFlow(ServerTestCase):
    """The whole journey, exactly as the page does it."""

    def test_flowers_from_drop_to_prediction(self):
        dataset = self.post("/api/samples/load", {"id": "flowers"})
        result = self.train(dataset["path"], "species", "classification")

        self.assertGreater(result["metrics"]["accuracy"], 0.8)
        self.assertTrue(result["leaderboard"])
        self.assertTrue(result["examples"])
        self.assertTrue(result["explanation"])

        model = self.get("/api/model")
        self.assertTrue(model["ready"])
        self.assertEqual(model["target"], "species")
        self.assertTrue(any(field["name"] == "petal_length" for field in model["fields"]))

        # A setosa-sized flower should be recognised as setosa.
        answer = self.post("/api/predict", {"values": {
            "sepal_length": "5.1", "sepal_width": "3.5",
            "petal_length": "1.4", "petal_width": "0.2",
        }})
        self.assertEqual(answer["answer"]["label"], "setosa")
        self.assertIn("setosa", answer["sentence"])

    def test_house_prices_as_numbers(self):
        dataset = self.post("/api/samples/load", {"id": "houses"})
        result = self.train(dataset["path"], "price", "regression")
        self.assertIn("mae", result["metrics"])
        self.assertGreater(result["metrics"]["r2"], 0.6)

        answer = self.post("/api/predict", {"values": {
            "size_sqm": "150", "bedrooms": "4", "age_years": "10",
            "distance_to_centre_km": "3", "garage": "yes",
        }})
        self.assertIn("value", answer["answer"])
        self.assertGreater(answer["answer"]["value"], 0)

    def test_a_whole_file_of_guesses(self):
        dataset = self.post("/api/samples/load", {"id": "flowers"})
        self.train(dataset["path"], "species", "classification")
        new_file = self.upload(
            "sepal_length,sepal_width,petal_length,petal_width\n"
            "5.0,3.4,1.5,0.2\n6.5,3.0,5.5,1.8\n", name="new-flowers.csv")
        guesses = self.post("/api/predict/file", {"path": new_file["path"]})
        self.assertEqual(guesses["count"], 2)
        self.assertIn("nanoLearn_guess", guesses["preview"]["headers"])
        # Downloading what it produced
        body = self.get(guesses["download"], raw=True).decode("utf-8")
        self.assertIn("nanoLearn_guess", body)

    def test_the_report_is_written_and_readable(self):
        dataset = self.post("/api/samples/load", {"id": "flowers"})
        self.train(dataset["path"], "species", "classification")
        report = self.post("/api/report", {"path": dataset["path"]})
        self.assertTrue(report["name"].endswith(".html"))
        html = self.get(report["download"], raw=True).decode("utf-8")
        self.assertIn("What nanoLearn found", html)
        self.assertIn("species", html)
        self.assertIn("What matters most", html)

    def test_it_refuses_to_train_on_a_missing_column(self):
        dataset = self.post("/api/samples/load", {"id": "flowers"})
        started = self.post("/api/train", {"path": dataset["path"], "target": "not_a_column",
                                          "task": "classification", "engine": "simple"})
        deadline = time.time() + 30
        while time.time() < deadline:
            payload = self.get("/api/job/" + started["job_id"])
            if payload["state"] == "error":
                self.assertIn("not_a_column", payload["error"])
                return
            if payload["state"] == "done":
                self.fail("it should not have been able to train on that column")
            time.sleep(0.2)
        self.fail("the job never finished")

    def test_predicting_before_teaching_says_so_clearly(self):
        engine.set_current_model(None)
        answer = self.post("/api/predict", {"values": {"a": "1"}})
        self.assertIn("error", answer)
        self.assertIn("Teach", answer["error"])


class TestDownloadSafety(ServerTestCase):
    def test_it_will_not_hand_out_files_outside_its_folder(self):
        request = urllib.request.Request(self.base + "/api/download/reports/..%2F..%2Fsettings.json")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                body = response.read().decode("utf-8")
                self.assertNotIn("test_fraction", body)
        except urllib.error.HTTPError as exc:
            self.assertIn(exc.code, (403, 404))

    def test_a_missing_download_is_a_clean_404(self):
        data = self.get("/api/download/reports/nothing-here.html")
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
