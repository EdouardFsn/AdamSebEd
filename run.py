import numpy as np
from helpers import load_csv_data, create_csv_submission
from preprocessing import preprocess_fit, preprocess_apply
from evaluation import cross_validate, summarize_cv, predict_logistic
from implementations import reg_logistic_regression


def train_baseline(xtr, ytr):
    """Train logistic regression (lambda=0) and return the weights."""
    w0 = np.zeros(xtr.shape[1])
    w, _ = reg_logistic_regression(ytr, xtr, 0.0, w0, max_iters=500, gamma=0.1)
    return w


def make_submission(
    x_train,
    y01,
    x_test,
    names,
    test_ids,
    threshold=0.2,
    use_categoricals=True,
    use_dependent=False,
):
    """Train on all data, predict on test, write an AIcrowd-ready CSV."""
    xtr, state = preprocess_fit(x_train, names, use_categoricals=use_categoricals, use_dependent=use_dependent)
    xte = preprocess_apply(x_test, names, state)

    w = train_baseline(xtr, y01)  # same trainer CV uses

    pred01 = predict_logistic(xte, w, threshold=threshold)  # {0,1}
    pred_pm1 = (2 * pred01 - 1).astype(int)  # back to {-1, 1}

    create_csv_submission(test_ids, pred_pm1, "submission.csv")
    print(f"wrote submission.csv | predicted positive: {pred01.mean():.3f}")


def main():
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/")

    with open("data/x_train.csv") as f:
        feature_names = f.readline().strip().split(",")[1:]

    y01 = (y_train + 1) / 2  # logistic needs {0,1}, not {-1,1}

    USE_CAT = True
    USE_DEP = False
    THRESHOLD = 0.2
    MAKE_SUBMISSION = False  # NEW: set to True only to write submission.csv

    scores = cross_validate(
        x_train,
        y01,
        feature_names,
        train_baseline,
        k=5,
        threshold=THRESHOLD,
        seed=0,
        use_categoricals=USE_CAT,
        use_dependent=USE_DEP,
    )
    mean, std = summarize_cv(scores)
    print(f"CV F1: {mean:.3f} ± {std:.3f} | folds: {np.round(scores, 3)}")

    MAKE_SUBMISSION = False  # NEW: set to True only to write submission.csv
        make_submission(
            x_train,
            y01,
            x_test,
            feature_names,
            test_ids,
            threshold=THRESHOLD,
            use_categoricals=USE_CAT,
            use_dependent=USE_DEP,
        )


if __name__ == "__main__":
    main()
