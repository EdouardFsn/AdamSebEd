import numpy as np
from helpers import load_csv_data
from preprocessing import preprocess_fit, preprocess_apply
from evaluation import cross_validate, summarize_cv
from implementations import reg_logistic_regression


def train_baseline(xtr, ytr):
    """Train logistic regression (lambda=0) and return the weights."""
    w0 = np.zeros(xtr.shape[1])
    w, _ = reg_logistic_regression(ytr, xtr, 0.0, w0, max_iters=500, gamma=0.1)
    return w


def main():
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/")

    with open("data/x_train.csv") as f:
        feature_names = f.readline().strip().split(",")[1:]

    y01 = (y_train + 1) / 2  # logistic needs {0,1}, not {-1,1}

    scores = cross_validate(
        x_train,
        y01,
        feature_names,
        train_baseline,
        k=5,
        threshold=0.2,
        seed=0,
        use_categoricals=True,
        use_dependent=False,
    )

    mean, std = summarize_cv(scores)
    print(
        f"categorical | CV F1: {mean:.3f} ± {std:.3f} | folds: {np.round(scores, 3)}"
    )  # baseline for now, replace when you add true to cat and dep


if __name__ == "__main__":
    main()
