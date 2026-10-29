"""Reproduce our final model and AIcrowd submission.
 
Runs a 5-fold cross-validation of the final configuration and prints its F1
score. If MAKE_SUBMISSION is True, also trains on all the training data and
writes submission.csv for the test set.
"""

import numpy as np
from helpers import load_csv_data, create_csv_submission
from preprocessing import preprocess_fit, preprocess_apply
from evaluation import (
    cross_validate_auto_threshold,
    summarize_cv,
    predict_logistic,
    make_logistic_trainer,
)


def make_submission(
    x_train,
    y01,
    x_test,
    names,
    test_ids,
    train_model,
    threshold=0.2,
    use_categoricals=True,
    use_dependent=False,
    max_missing=0.8, 
    use_missing_indicators=False,
):
    """Train on all the training data, predict the test set, write the CSV.
 
    Args:
        x_train: numpy array of shape (N, D), the raw training features.
        y01: numpy array of shape (N,), the training labels in {0, 1}.
        x_test: numpy array of shape (M, D), the raw test features.
        names: list of D feature names, aligned with the columns of x_train.
        test_ids: numpy array of shape (M,), the test ids.
        train_model: function (xtr, ytr) -> w, the same trainer as in CV.
        threshold: float, decision threshold.
        use_categoricals, use_dependent, max_missing, use_missing_indicators:
            preprocessing options, passed unchanged to preprocess_fit.
 
    Returns:
        None. Writes submission.csv with labels in {-1, 1}.
    """
    xtr, state = preprocess_fit(
        x_train, names, use_categoricals=use_categoricals, use_dependent=use_dependent, max_missing=max_missing, use_missing_indicators=use_missing_indicators
    )
    xte = preprocess_apply(x_test, names, state)
    assert not np.isnan(xte).any(), "NaN left in test after preprocessing"

    w = train_model(xtr, y01)  # same trainer CV uses

    pred01 = predict_logistic(xte, w, threshold=threshold)  # {0,1}
    pred_pm1 = (2 * pred01 - 1).astype(int)  # back to {-1, 1}

    create_csv_submission(test_ids, pred_pm1, "submission.csv")
    print(f"wrote submission.csv | predicted positive: {pred01.mean():.3f}")


def main():
    """Cross-validate the final configuration and optionally write the submission."""
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/")

    with open("data/x_train.csv") as f:
        feature_names = f.readline().strip().split(",")[1:]

    y01 = (y_train + 1) / 2  # logistic needs {0,1}, not {-1,1}

    # Settings, chosen with cross-validation (see tune.py and results.md)
    USE_CAT = True  # one-hot encode CATEGORICAL_FEATURES
    USE_DEP = True  # fill skip-logic blanks (DEPENDENT_RULES)
    MAX_MISSING = 0.999  # drop columns more empty than this
    USE_MI = True  # add a "was missing" column for columns more than 5% empty
    THRESHOLD = 0.21  # decision threshold of the submission
    MAKE_SUBMISSION = True  # True to write submission.csv
 
    LAMBDA = 0.0
    GAMMA = 0.5
    MAX_ITERS = 500
    train_model = make_logistic_trainer(LAMBDA, GAMMA, MAX_ITERS)

    # Column count after preprocessing (info only: CV re-does it on each fold)
    x_check, _ = preprocess_fit(
        x_train, feature_names, use_categoricals=USE_CAT, use_dependent=USE_DEP, max_missing=MAX_MISSING, use_missing_indicators=USE_MI,
    )
    print(f"cat={USE_CAT} dep={USE_DEP} | columns: {x_check.shape[1]} (incl. bias)")

    scores, chosen = cross_validate_auto_threshold(
        x_train,
        y01,
        feature_names,
        train_model,
        k=5,
        seed=0,
        use_categoricals=USE_CAT,
        use_dependent=USE_DEP,
        max_missing=MAX_MISSING,
        use_missing_indicators=USE_MI,
    )
    mean, std = summarize_cv(scores)
    print(f"CV F1 (auto-threshold): {mean:.3f} ± {std:.3f}")
    print(f"thresholds chosen per fold: {np.round(chosen, 2)}")

    if MAKE_SUBMISSION:
        make_submission(
            x_train,
            y01,
            x_test,
            feature_names,
            test_ids,
            train_model,
            threshold=THRESHOLD,
            use_categoricals=USE_CAT,
            use_dependent=USE_DEP,
            max_missing=MAX_MISSING,
            use_missing_indicators=USE_MI,
        )


if __name__ == "__main__":
    main()
