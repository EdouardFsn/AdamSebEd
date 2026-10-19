import numpy as np
from helpers import load_csv_data, create_csv_submission
from preprocessing import preprocess_fit, preprocess_apply
from evaluation import (
    cross_validate,
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
):
    """Train on all data, predict on test, write an AIcrowd-ready CSV."""
    xtr, state = preprocess_fit(
        x_train, names, use_categoricals=use_categoricals, use_dependent=use_dependent
    )
    xte = preprocess_apply(x_test, names, state)
    assert not np.isnan(xte).any(), "NaN left in test after preprocessing"

    w = train_model(xtr, y01)  # same trainer CV uses

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
    USE_DEP = True  # fill skip-logic blanks (DEPENDENT_RULES)

    THRESHOLD = 0.2  # select after CV for aicrowd submission
    MAKE_SUBMISSION = False  # set to True only to write submission.csv

    # Logistic regression training, chosen by the CV sweep in tune.py
    LAMBDA = 0.0
    GAMMA = 0.5
    MAX_ITERS = 500
    train_model = make_logistic_trainer(LAMBDA, GAMMA, MAX_ITERS)

    # Column count after preprocessing (info only: CV re-does it on each fold)
    x_check, _ = preprocess_fit(
        x_train, feature_names, use_categoricals=USE_CAT, use_dependent=USE_DEP
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
        )


if __name__ == "__main__":
    main()
