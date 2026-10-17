import numpy as np
from helpers import load_csv_data, create_csv_submission
from preprocessing import preprocess_fit, preprocess_apply
from evaluation import (
    predict_logistic,
    predict_proba,
    train_val_split,
    f1_score,
    threshold_sweep,
)
from implementations import reg_logistic_regression


def make_submission(x_train, y01, x_test, names, test_ids, threshold=0.2):
    """Train on all data, predict on test, write an AIcrowd-ready CSV."""
    xtr, state = preprocess_fit(x_train, names)  # fit recipe on ALL train
    xte = preprocess_apply(x_test, names, state)  # same recipe on test

    w0 = np.zeros(xtr.shape[1])
    w, _ = reg_logistic_regression(y01, xtr, 0.0, w0, max_iters=500, gamma=0.1)

    pred01 = predict_logistic(xte, w, threshold=threshold)  # {0,1}
    pred_pm1 = (2 * pred01 - 1).astype(int)  # back to {-1, 1}

    create_csv_submission(test_ids, pred_pm1, "submission.csv")
    print(f"wrote submission.csv | predicted positive: {pred01.mean():.3f}")


def main():
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/")

    with open("data/x_train.csv") as f:
        feature_names = f.readline().strip().split(",")[1:]

    y01 = (y_train + 1) / 2  # logistic needs {0,1}, not {-1,1}

    make_submission(x_train, y01, x_test, feature_names, test_ids, threshold=0.2)


if __name__ == "__main__":
    main()
