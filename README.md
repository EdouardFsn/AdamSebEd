# AdamSebEd — CS-433 Project 1

> Predicting coronary heart disease (**MICHD**) risk from BRFSS survey data.

**Team:** Edouard Fousson · Adam Cohen · Sebastien Nyiri

**Deadline:** Thursday, 29 October 2026 · 16:00 — submit at [mlcourse.epfl.ch](http://mlcourse.epfl.ch)

---

## Contents

1. [First-time setup](#first-time-setup)
2. [Running the tests](#running-the-tests)
3. [Working together](#working-together)
4. [Repo structure](#repo-structure)
5. [Required functions](#required-functions)
6. [Project constraints](#project-constraints)

---

## First-time setup

Everything in this section is a **one-time** setup per machine. Ping the group chat if anything fails.

> Paths below use Windows (`C:\Users\...`). On macOS/Linux the commands are the same; paths look like `/Users/you/...`.

### 1. Install the tools

Check what you already have:

```bash
git --version
conda --version
code --version
```

Install anything that's missing, then re-check.

### 2. Configure git (once per machine)

```bash
git config --global user.name "Your Name"
git config --global user.email "your.github.email@example.com"
```

Use the email tied to your GitHub account — otherwise your commits won't be linked to your profile.

### 3. Clone both repos

Clone them **side by side, never one inside the other**:

```bash
cd C:\Users\YOU\Documents
git clone https://github.com/epfml/ML_course.git
git clone https://github.com/EdouardFsn/AdamSebEd.git
```

The course repo (`ML_course`) is needed for the grading tests. We never edit it — we only pull updates from it.

### 4. Get the data

The `data/` folder is **not versioned** — the CSVs are too large for git, so everyone downloads their own copy.

1. Create an AIcrowd account with your **@epfl.ch** address.
2. Join the challenge: <https://www.aicrowd.com/challenges/epfl-machine-learning-project-1>
3. Download `x_train.csv`, `y_train.csv` and `x_test.csv`.
4. Create a `data/` folder at the repo root and put the three files in it.

> **Do not rename them** — `load_csv_data` expects these exact names.
> 
> **Sanity check:** `git status` should show no `.csv` files.

### 5. Create the grading environment

```bash
cd C:\Users\YOU\ML_course\projects\project1\grading_tests
conda env create --file=environment.yml --name=project1-grading
conda activate project1-grading
```

Then point VS Code at it: open the **AdamSebEd** folder → `Ctrl+Shift+P` → **Python: Select Interpreter** → pick `project1-grading`.

> This environment ships scikit-learn and pandas **for running the tests only**. We are **not** allowed to use them in our code.

### 6. Enable auto-formatting

We all use [**black**](https://github.com/psf/black) so git never flags conflicts on lines that differ only by spacing. This only pays off if **all three of us** do it.

1. Install the **Black Formatter** extension (`ms-python.black-formatter`) from the Extensions panel.
2. `Ctrl+Shift+P` → **Preferences: Open User Settings (JSON)**, then add (mind the comma on the line above):

```jsonc
"[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true
}
```

**Check it works:** open a `.py` file, type `x=1+2`, save — it should snap to `x = 1 + 2`.

> **If nothing happens on save:** the formatter's language server needs Python ≥ 3.10, but `project1-grading` runs 3.9. Fix by giving black its own interpreter — create one (`conda create --name fmt python=3.12`) and add `"black-formatter.interpreter": ["C:\\Users\\YOU\\anaconda3\\envs\\fmt\\python.exe"]` to your settings. This is separate from the interpreter your project uses.

---

## Running the tests

```bash
cd C:\Users\YOU\ML_course\projects\project1\grading_tests
conda activate project1-grading
pytest --github_link C:\Users\YOU\Documents\AdamSebEd .
```

> The trailing **`.`** matters — it tells pytest to load this folder's `conftest.py`, which defines the custom `--github_link` flag. Drop the dot and you get `unrecognized arguments: --github_link`.

Target a single function with `-k`:

```bash
pytest --github_link C:\Users\YOU\Documents\AdamSebEd . -k least_squares
```

**Two failures are expected and harmless during development:**

| Failure | Why |
|---|---|
| value tests failing with `NotImplementedError` | that function isn't written yet |
| `test_github_link_format` | we pass a local path instead of a URL |

The TAs may update the tests, so pull the course repo now and then (we'll also get `helpers.py` from there later):

```bash
cd C:\Users\YOU\ML_course
git pull
```

---

## Working together

A few conventions so the three of us never step on each other. None of this is bureaucracy — it's what keeps `main` always working and our history readable.

### The golden rule

**`main` is protected by convention: all work happens on a branch, and `main` only changes through a reviewed pull request.**

### Daily workflow

**Start of session** — always branch off an up-to-date `main`:

```bash
git checkout main
git pull
git checkout -b feature/my-thing
```

**During work** — commit in small, meaningful steps:

```bash
git add implementations.py
git commit -m "implement ridge_regression"
```

**End of session** — push your branch and open a PR:

```bash
git push -u origin feature/my-thing
```

Commit messages: **English, imperative mood, short** (`implement ridge_regression`, not `added ridge`).

### Branch naming

| Prefix | For | Example |
|---|---|---|
| `feature/` | new functionality | `feature/ridge-regression` |
| `fix/` | bug fixes | `fix/sigmoid-overflow` |

Lowercase, hyphen-separated, short.

### Pull requests

- **Open a PR for every change** — however small.
- **Write a description detailed enough** that a teammate understands *what* changed and *why* without reading every line. Mention the function(s) touched, any helpers you added (so nobody duplicates them), and anything a reviewer should watch for.
- **Assign the most relevant reviewer**, then **drop a message in the WhatsApp group** so they actually see it.

### Reviews & merging

- **No merge before someone else has approved.** Self-merging defeats the point of review.
- The reviewer checks the code reads well **and** the grading tests pass, then approves.
- Once approved, merge — and delete the branch to keep the branch list clean.

### Notebooks

Shared code lives in **`.py` files**. Notebooks are for personal exploration and each of us keeps their own (`explo_adam.ipynb`, etc.) — `.ipynb` files are JSON under the hood and merge terribly, so we never share them on `main`.

---

## Repo structure

```
AdamSebEd/
├── README.md              this file
├── implementations.py     the six required functions
├── run.py                 reproduces our best AIcrowd submission
├── helpers.py             data loading and submission writing
├── data/                  not versioned — see setup step 4
└── .gitignore
```

> `README.md`, `implementations.py` and `run.py` must stay at the **root** — the grading tests check for them there.

---

## Required functions

All six go in `implementations.py`, with these exact signatures:

| Function | Signature |
|---|---|
| `mean_squared_error_gd` | `(y, tx, initial_w, max_iters, gamma)` |
| `mean_squared_error_sgd` | `(y, tx, initial_w, max_iters, gamma)` |
| `least_squares` | `(y, tx)` |
| `ridge_regression` | `(y, tx, lambda_)` |
| `logistic_regression` | `(y, tx, initial_w, max_iters, gamma)` |
| `reg_logistic_regression` | `(y, tx, lambda_, initial_w, max_iters, gamma)` |

**Conventions enforced by the tests:**

- return `(w, loss)` — only the **last** `w`, not the whole history
- `loss` must be a **scalar** (`loss.ndim == 0`), not a size-1 array
- `w` must have shape `(D,)`, never `(D, 1)`
- MSE carries a factor **0.5**
- ridge and regularized logistic return the loss **without** the penalty term
- SGD uses **mini-batch size 1**
- logistic regression expects **y ∈ {0, 1}**, not {−1, 1}
- `least_squares` may use anything from `numpy.linalg` **except** `lstsq`
- `max_iters=0` must work and return `initial_w` with its loss
- every function needs a **docstring**
- the word `TODO` must not appear in any `.py` file

---

## Project constraints

- **Allowed:** Python standard library, NumPy. matplotlib/seaborn **for plots only**.
- **Not allowed:** pandas, scikit-learn, PyTorch, TensorFlow, any external library or dataset.
- The repo must stay **public**.
- AIcrowd allows **5 submissions per day**. Always validate locally too — the leaderboard score is not a substitute for cross-validation.
- Project 1 does **not** count toward the final grade; it's preparation for **Project 2 (30%)**.
