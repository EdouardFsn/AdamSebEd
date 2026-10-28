"""Compare methods across preprocessing stages, for the report figure.

Runs 5-fold CV F1 for each (method, pipeline stage), caches the numbers to a
small CSV, and draws a line plot from it. Delete the CSV to recompute.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from helpers import load_csv_data
from preprocessing import preprocess_fit, preprocess_apply
from evaluation import k_fold_indices, f1_score
from implementations import (
    ridge_regression,
    mean_squared_error_gd,
    reg_logistic_regression,
)

RESULTS_CSV = "comparison_results.csv"
K, SEED = 5, 0

# pipeline stages (cumulative), in plot order
CONFIGS = [
    (
        "Baseline (80% cut)",
        dict(
            use_categoricals=False,
            use_dependent=False,
            max_missing=0.8,
            use_missing_indicators=False,
        ),
    ),
    (
        "+ one-hot",
        dict(
            use_categoricals=True,
            use_dependent=False,
            max_missing=0.8,
            use_missing_indicators=False,
        ),
    ),
    (
        "+ dependent",
        dict(
            use_categoricals=True,
            use_dependent=True,
            max_missing=0.8,
            use_missing_indicators=False,
        ),
    ),
    (
        "+ keep cols\n (99% cut)",
        dict(
            use_categoricals=True,
            use_dependent=True,
            max_missing=0.999,
            use_missing_indicators=True,
        ),
    ),
]

GAMMA, MAX_ITERS = 0.5, 500
GAMMA_MSE, MAX_ITERS_MSE = 0.01, 1000


def t_ridge(xtr, ytr):
    w, _ = ridge_regression(ytr, xtr, 1e-2)
    return w


def t_logistic(xtr, ytr):
    w0 = np.zeros(xtr.shape[1])
    w, _ = reg_logistic_regression(ytr, xtr, 0.0, w0, MAX_ITERS, GAMMA)
    return w


def t_mse_gd(xtr, ytr):
    w0 = np.zeros(xtr.shape[1])
    w, _ = mean_squared_error_gd(ytr, xtr, w0, MAX_ITERS_MSE, GAMMA_MSE)
    return w


METHODS = [
    ("Logistic", t_logistic),
    ("Ridge", t_ridge),
    ("MSE-GD", t_mse_gd),
]


def best_threshold_on_scores(y, scores, n=99):
    grid = np.quantile(scores, np.linspace(0.01, 0.99, n))
    best_t, best_f1 = grid[0], -1.0
    for t in grid:
        f1 = f1_score(y, (scores >= t).astype(float))
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


def cv_f1(x, y, names, train_model, cfg):
    """5-fold CV F1 (mean, std) for one (method, config); threshold on train scores."""
    folds = k_fold_indices(len(y), k=K, seed=SEED)
    scores = []
    for i in range(K):
        val_idx = folds[i]
        tr_idx = np.concatenate([folds[j] for j in range(K) if j != i])
        xtr, state = preprocess_fit(x[tr_idx], names, **cfg)
        xval = preprocess_apply(x[val_idx], names, state)
        w = train_model(xtr, y[tr_idx])
        s_tr, s_val = xtr @ w, xval @ w
        if not (np.all(np.isfinite(s_tr)) and np.all(np.isfinite(s_val))):
            scores.append(np.nan)  # diverged / singular
            continue
        t = best_threshold_on_scores(y[tr_idx], s_tr)
        scores.append(f1_score(y[val_idx], (s_val >= t).astype(float)))
    scores = np.array(scores)
    return float(np.nanmean(scores)), float(np.nanstd(scores))


def load_results():
    """Load cached means/stds POSITIONALLY (rows in METHODS x CONFIGS order).
    Labels are ignored, so renaming a stage never triggers a recompute.
    Returns None if the grid shape no longer matches (caller recomputes)."""
    with open(RESULTS_CSV) as f:
        rows = list(csv.DictReader(f))
    n_m, n_s = len(METHODS), len(CONFIGS)
    if len(rows) != n_m * n_s:
        return None
    means = np.array([float(r["f1_mean"]) for r in rows]).reshape(n_m, n_s)
    stds = np.array([float(r["f1_std"]) for r in rows]).reshape(n_m, n_s)
    return means, stds


def save_results(means, stds):
    """Write the grid atomically; the old CSV survives until this succeeds."""
    tmp = RESULTS_CSV + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "stage", "f1_mean", "f1_std"])
        for mi, (m_label, _) in enumerate(METHODS):
            for si, (s_label, _) in enumerate(CONFIGS):
                w.writerow(
                    [m_label, s_label, f"{means[mi, si]:.4f}", f"{stds[mi, si]:.4f}"]
                )
    os.replace(tmp, RESULTS_CSV)


def compute_all():
    x_train, _, y_train, _, _ = load_csv_data("data/")
    with open("data/x_train.csv") as f:
        names = f.readline().strip().split(",")[1:]
    y01 = (y_train + 1) / 2
    n_m, n_s = len(METHODS), len(CONFIGS)
    means, stds = np.zeros((n_m, n_s)), np.zeros((n_m, n_s))
    for mi, (m_label, trainer) in enumerate(METHODS):
        for si, (s_label, cfg) in enumerate(CONFIGS):
            mean, std = cv_f1(x_train, y01, names, trainer, cfg)
            means[mi, si], stds[mi, si] = mean, std
            print(f"{m_label:<10} | {s_label:<22} | {mean:.4f} ± {std:.4f}")
    save_results(means, stds)
    return means, stds


def get_results():
    if os.path.exists(RESULTS_CSV):
        loaded = load_results()
        if loaded is not None:
            return loaded  # file found -> just use it, no recompute
    return compute_all()  # not found (or shape changed) -> compute


def plot(outfile="methods_comparison.pdf"):
    means, stds = get_results()
    stages = [s for s, _ in CONFIGS]
    xpos = np.arange(len(stages))

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for mi, (m_label, _) in enumerate(METHODS):
        ax.errorbar(
            xpos, means[mi], yerr=stds[mi], marker="o", capsize=3, label=m_label
        )

    lo, hi = np.nanmin(means - stds), np.nanmax(means + stds)
    pad = 0.1 * (hi - lo) + 1e-3
    ax.set_ylim(lo - pad, hi + pad)
    ax.yaxis.set_major_locator(MultipleLocator(0.005))
    ax.set_xticks(xpos)
    ax.set_xticklabels(stages)
    ax.set_ylabel("5-fold CV F1")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile)
    print(f"saved {outfile}")


if __name__ == "__main__":
    plot()
