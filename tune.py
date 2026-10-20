"""this is the hyperparameter sweep for regularized logistic regression

cross-validates the L2 strength lambda_ and the gradient descent settings
(gamma, max_iters) on the same 5 folds and preprocessing as run.py, and prints
one line per setting, and gives results in results.md
"""

import numpy as np

from helpers import load_csv_data
from evaluation import (
    cross_validate_auto_threshold,
    hyperparameter_sweep,
    make_logistic_trainer,
    make_weighted_trainer,
    summarize_cv,
)

# same preprocessing switches as run.py
USE_CAT = True
USE_DEP = True

# (label, lambda_, gamma, max_iters). The first line is the previous setting
# of run.py
# gamma=0.5 reaches the minimum of the loss within 500 steps.
CONFIGS = [
    ("lambda=0     gamma=0.1", 0.0, 0.1, 500),
    ("lambda=0     gamma=0.5", 0.0, 0.5, 500),
    ("lambda=1e-4  gamma=0.5", 1e-4, 0.5, 500),
    ("lambda=1e-3  gamma=0.5", 1e-3, 0.5, 500),
    ("lambda=1e-2  gamma=0.5", 1e-2, 0.5, 500),
    ("lambda=1e-1  gamma=0.5", 1e-1, 0.5, 500),
]

THRESHOLDS = np.round(np.arange(0.10, 0.51, 0.01), 2)
REFERENCE_THRESHOLD = 0.2  # the fixed threshold used by run.py


def main():
    x_train, _, y_train, _, _ = load_csv_data("data/")
    with open("data/x_train.csv") as f:
        feature_names = f.readline().strip().split(",")[1:]
    y01 = (y_train + 1) / 2

    trainers = [make_logistic_trainer(lam, g, it) for _, lam, g, it in CONFIGS]
    f1, train_loss = hyperparameter_sweep(
        x_train,
        y01,
        feature_names,
        trainers,
        THRESHOLDS,
        k=5,
        seed=0,
        use_categoricals=USE_CAT,
        use_dependent=USE_DEP,
    )

    ref = list(THRESHOLDS).index(REFERENCE_THRESHOLD)
    for m, (label, _, _, _) in enumerate(CONFIGS):
        mean_by_t = f1[m].mean(axis=0)  # mean over folds, per threshold
        best = int(np.argmax(mean_by_t))
        print(
            f"{label} | train loss {train_loss[m].mean():.4f} | "
            f"F1@{REFERENCE_THRESHOLD}: {f1[m, :, ref].mean():.4f} ± {f1[m, :, ref].std():.4f} | "
            f"best t={THRESHOLDS[best]:.2f}: {mean_by_t[best]:.4f} ± {f1[m, :, best].std():.4f} | "
            f"folds@{REFERENCE_THRESHOLD}: {np.round(f1[m, :, ref], 3)}"
        )

    for pw in [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
        trainer = make_weighted_trainer(pos_weight=pw, gamma=0.5, max_iters=500)
        scores, chosen = cross_validate_auto_threshold(
            x_train,
            y01,
            feature_names,
            trainer,
            k=5,
            seed=0,
            use_categoricals=True,
            use_dependent=True,
        )
        mean, std = summarize_cv(scores)
        print(
            f"pos_weight={pw:5.1f} | F1 {mean:.3f} ± {std:.3f} | t {np.round(chosen,2)}"
        )


if __name__ == "__main__":
    main()
