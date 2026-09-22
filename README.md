<div align="center">

# nanoLearn

**Drop a spreadsheet. Get an answer machine.** No code, no maths, no jargon.

Point it at a CSV, tell it which column you care about, and it teaches itself —
then shows you how well it did, which columns actually mattered, and lets you
ask it about new cases.

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.9+-58a6ff.svg)]()
[![dependencies](https://img.shields.io/badge/required%20deps-0-f0883e.svg)]()
[![tests](https://img.shields.io/badge/tests-32%20passing-3ddc97.svg)]()

</div>

---

## What this is, in one paragraph

Machine learning is normally taught as a programming exercise. It is not one —
it is a *question about your data*: which of these columns tells me the thing I
want to know? nanoLearn is the front door to that question. Drop in a
spreadsheet, and it works out what kind of table it is, guesses which column you
want predicted, teaches itself on part of the data, checks itself on the part it
has never seen, and tells you in plain words whether there is a real signal
there or not. **It works with nothing installed**: the built-in engine is a
small random forest written in plain Python. If `scikit-learn` is available it
uses that instead, and it can install it for you with one click.

---

## Use it

1. Install Python if you do not have it — [python.org/downloads](https://www.python.org/downloads/).
2. Download this repo and unzip it.
3. **Windows:** double-click `run.bat`. **macOS / Linux:** `./run.sh`.
4. Your browser opens at `http://127.0.0.1:8761`.

No file handy? Press **Flower measurements** or **House prices** — both ship with
the app, and both are small enough to read.

<details>
<summary>Prefer the command line? (you do not need to)</summary>

```bash
python start.py                  # start and open the browser
python -m nanolearn doctor       # show which engines this computer has
python -m nanolearn --port 9000 --no-browser
```

</details>

---

## The five steps it walks you through

| step | what happens |
|---|---|
| **1. Give it a spreadsheet** | Drag a `.csv` in, or pick one you used before. Files are copied in 4 MB slices and stay on your machine. |
| **2. Say what to predict** | It *guesses* the column you want and tells you **why** — "I picked `species` because its name is species and it has 3 possible answers". Change it if it guessed wrong. |
| **3. Let it practise** | It holds rows back, tries several approaches, and shows its working as a live log. |
| **4. See how it did** | A score in plain language, a comparison table, what mattered most, and real test rows with its guess next to the truth. |
| **5. Use it** | Fill in a form to ask about a new case, or drop a whole spreadsheet and get a file of guesses back. |

### Things it does that you would not expect from a toy

- **It compares itself to guessing.** "It gets 96% right, where always answering
  the same thing would get 33%." If there is no signal in your file, it says so
  instead of dressing up noise as a discovery.
- **It explains what it threw away.** Identifiers, free text and mostly-empty
  columns are excluded *and listed*, because leaving them in would inflate the
  score.
- **Unreadable numbers are handled.** `$1,234.50`, `12%`, `(3.5)` for negatives,
  `1,5` as a European decimal, `1 234` with a space — all parsed correctly.
- **It writes a report.** One HTML file with the score, the leaderboard, the
  important columns and what to do next, that you can keep or send to someone.

---

## The two engines

| engine | needs | what it is |
|---|---|---|
| **Built-in** | nothing | A small random forest — decision trees written from scratch in plain Python, bagged and voting. Uses the standard library only. |
| **Full** | `scikit-learn` (one-click install) | Cross-validated comparison of logistic/linear regression, decision trees, random forests, boosted trees, k-nearest neighbours, and an always-guess baseline. |

The built-in engine is not a placeholder. On the shipped examples it gets
**above 90% on the flower data** and explains **more than 60% of the variation in
the house prices** — with no third-party code at all. The full engine is
better-tuned and cross-validated; it is an upgrade, not a requirement.

Both paths are covered by tests that assert real accuracy thresholds, not just
that the code ran.

---

## How it works

```
browser ──► httpbase.py    routing, static files, SSE        (standard library)
            server.py      /api/* addresses
            table.py       CSV in, features out — parsing, types,
                           target guessing, one-hot, standardising
            engine.py      the two engines, metrics, jobs, prediction
            report.py      the written report
            store.py       your folder, settings, example data
```

The pipeline, in the order it runs:

1. **Read** the CSV (any delimiter, BOM, ragged rows, blank lines).
2. **Profile** every column: number / category / text / empty, gaps, spread.
3. **Guess the target** by name and shape, and say why.
4. **Build features**: standardise numbers (filling gaps with the column mean),
   one-hot encode small category sets, drop identifiers and free text.
5. **Split** into practice and unseen exam rows.
6. **Train** several approaches; keep the best by cross-validated score.
7. **Score** the winner on the held-back rows, against a always-guess baseline.
8. **Explain**: weigh feature importance, then phrase it for a human.

```bash
python -m unittest discover tests -v     # 32 tests
```

## Honest limitations

- It is for **tables**, not images, audio or free text. A column of sentences is
  ignored, and it tells you so.
- The score is an **estimate from your own data**, measured on rows it did not
  see. Real-world performance is usually a little worse, never better.
- The **built-in engine's score is a honest one** but it is not tuned: expect the
  full engine to win by a few points on harder tables.
- Large files (over ~200,000 rows) are truncated, and the built-in engine
  practises on a 4,000-row sample so a click never turns into a hang. Both are
  said out loud in the log.
- Saved models are Python pickles. Open only the ones **nanoLearn wrote** — that
  format can run code, so it is not safe to accept from strangers.

## Family

| repo | role |
|---|---|
| [nanolaama](https://github.com/Agarwalrishu13/nanolaama) | **talk to an AI** — a friendly window onto local engines |
| **nanolearn** (this repo) | **teach an AI** — spreadsheets into working models |
| [nonoForge](https://github.com/Agarwalrishu13/nonoforge) | **build an app** — pick a card, press one button, it exists |
| [nanollama.c](https://github.com/Agarwalrishu13/nanollama.c) | the from-scratch C inference engine |
| [nanobrain](https://github.com/Agarwalrishu13/nanobrain) | the from-scratch trainer and tokenizer |
| [nanoforge](https://github.com/Agarwalrishu13/nanoforge) | the offline studio for building tiny models |

## License

MIT © Priyanshu Agarwal
