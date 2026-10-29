# AdamSebEd — CS-433 Project 1

> Predicting coronary heart disease (**MICHD**) risk from BRFSS survey data.

**Team:** Edouard Fousson · Adam Cohen · Sebastien Nyiri

---

## Contents

1. [Reproducing our best submission](#reproducing-our-best-submission)
2. [Pipeline](#pipeline)
3. [Model and validation](#model-and-validation)
4. [Setup](#setup)
5. [Repo structure](#repo-structure)
6. [Required functions](#required-functions)
7. [Running the tests](#running-the-tests)
8. [Project constraints](#project-constraints)

---

## Reproducing our best submission

1. Put `x_train.csv`, `y_train.csv` and `x_test.csv` in `data/` (see [Setup](#setup)).
2. Run:

```bash
python run.py
```

It prints the 5-fold cross-validation F1 of the final model, then trains it on the whole training set and writes `submission.csv`, the exact file of our best AIcrowd submission (**F1 = 0.449** on the leaderboard). Runtime: about **2–3 minutes** on a laptop.

---

## Pipeline

All steps are learned on the training data only (`preprocess_fit`) and replayed unchanged on validation or test data (`preprocess_apply`), so no statistic ever leaks from held-out rows. Feature names and answer codes follow the [BRFSS 2015 codebook](https://www.cdc.gov/brfss/annual_data/annual_2015.html). Everything lives in `preprocessing.py`.

1. **Code mapping.** Answers meaning "none" (88, 888, 98) become 0 (`NONE_VALUES`). Answers meaning "don't know / refused" (7, 9, 77, 99, …) become NaN (`MISSING_VALUES`). Some calculated variables are rescaled in the course data, so their codes are adjusted (e.g. 99000 → 99).
2. **Skip-logic fill.** When a question was not asked because of an earlier answer, the blank is filled from that answer, e.g. non-drinkers get 0 drinks (`DEPENDENT_RULES`, 14 rules, each checked on `x_train`).
3. **Column drops.** Survey metadata, raw answers that pack a unit and a value into one number (BRFSS provides a decoded version, which we keep), and columns duplicated elsewhere (`DROP_FEATURES`). Then columns with more than 99.9% missing values.
4. **One-hot encoding** of 14 nominal or order-breaking variables (`CATEGORICAL_FEATURES`), with the most frequent level as reference.
5. **Missing indicators.** A 0/1 "was missing" column for every column with more than 5% missing values, so that the model can tell a blank from an imputed value.
6. **Median imputation**, **standardization**, and a **bias column**.

The final design matrix has about **514 columns** (including the bias).

---

## Model and validation

- **Model:** logistic regression (`reg_logistic_regression`), λ = 0, γ = 0.5, 500 gradient steps (converged). Labels are mapped from {-1, 1} to {0, 1}.
- **Validation:** 5-fold cross-validation (seed 0). The preprocessing is re-fitted inside each fold.
- **Decision threshold:** chosen on the training folds' predictions to maximize F1, then applied to the held-out fold. It is stable across folds (0.20–0.21), and the submission uses t = 0.21.
- **Final CV F1:** **0.439 ± 0.004**.

Every experiment, including the ones that did not help, is logged in `results.md` (ablation study, Table I of the report); the rows are obtained by switching the settings at the top of `run.py`. `tune.py` reruns the hyperparameter sweeps (set `SWEEP_MISSING = False` for the λ / γ and class-weighting sweeps), and `implementations_comparaison.py` produces the method-comparison figure of the report (`methods_comparison.pdf`).

---

## Setup

### Get the data

The `data/` folder is **not versioned**: the CSVs are too large for git.

1. Create an AIcrowd account with an **@epfl.ch** address.
2. Join the challenge: <https://www.aicrowd.com/challenges/epfl-machine-learning-project-1>
3. Download `x_train.csv`, `y_train.csv` and `x_test.csv`.
4. Create a `data/` folder at the repo root and put the three files in it.

> **Do not rename them**: `load_csv_data` expects these exact names.

### Environment

Only Python and NumPy are needed (matplotlib for the report figure). We used the course's grading environment:

```bash
cd ML_course/projects/project1/grading_tests
conda env create --file=environment.yml --name=project1-grading
conda activate project1-grading
```

> This environment ships scikit-learn and pandas **for running the tests only**. Our code does not use them.

---

## Repo structure

```
AdamSebEd/
├── README.md                        this file
├── implementations.py               the six required functions
├── run.py                           reproduces our best AIcrowd submission
├── preprocessing.py                 feature lists and the fit / apply cleaning pipeline
├── evaluation.py                    F1, cross-validation, trainers
├── tune.py                          hyperparameter sweeps (results in results.md)
├── implementations_comparaison.py   method comparison figure for the report
├── methods_comparison.pdf           figure for the report
├── results.md                       log of every experiment (ablation study)
├── helpers.py                       data loading and submission writing (from the course)
├── data/                            not versioned, see Setup
└── .gitignore
```

---

## Required functions

All six are in `implementations.py`, with these exact signatures:

| Function | Signature |
|---|---|
| `mean_squared_error_gd` | `(y, tx, initial_w, max_iters, gamma)` |
| `mean_squared_error_sgd` | `(y, tx, initial_w, max_iters, gamma)` |
| `least_squares` | `(y, tx)` |
| `ridge_regression` | `(y, tx, lambda_)` |
| `logistic_regression` | `(y, tx, initial_w, max_iters, gamma)` |
| `reg_logistic_regression` | `(y, tx, lambda_, initial_w, max_iters, gamma)` |

Each returns `(w, loss)`: the last weight vector, of shape `(D,)`, and its loss as a scalar. The MSE carries a factor 0.5, the regularized methods return the loss **without** the penalty term, SGD uses mini-batches of size 1, and the logistic methods expect labels in {0, 1}.

---

## Running the tests

```bash
cd ML_course/projects/project1/grading_tests
conda activate project1-grading
pytest --github_link path/to/AdamSebEd .
```

> The trailing **`.`** matters: it tells pytest to load this folder's `conftest.py`, which defines the `--github_link` flag.

---

## Project constraints

- **Allowed:** Python standard library, NumPy. matplotlib **for plots only**.
- **Not used:** pandas, scikit-learn, PyTorch, TensorFlow, any external library or dataset.
- Model choices were made with local cross-validation, not with the leaderboard score.