import numpy as np
from helpers import load_csv_data
from preprocessing import (
    drop_named,
    apply_none_map,
    apply_missing_map,
    drop_high_missing,
    fit_impute,
    apply_impute,
    fit_standardize,
    apply_standardize,
    add_bias,
)
from implementations import reg_logistic_regression

x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/")

with open("data/x_train.csv") as f:
    feature_names = f.readline().strip().split(",")[1:]


def preprocess_fit(x, names):
    """Clean training data, and RETURN the stats learned, so the exact same
    transform can be replayed on validation/test.

    Returns:
        x_clean: the processed training matrix (with bias column).
        state: dict of everything learned on train (which columns to keep,
               medians, mean, std) — the "recipe" to replay elsewhere.
    """
    x = apply_none_map(x, names)  # none-codes -> 0   (while values are raw)
    x = apply_missing_map(x, names)  # sentinels  -> nan

    x, names, keep_named = drop_named(x, names)  # drop metadata
    x, names, keep_missing = drop_high_missing(x, names)  # drop near-empty cols

    medians = fit_impute(x)  # learn medians on train
    x = apply_impute(x, medians)

    mean, std = fit_standardize(x)  # learn scale on train
    x = apply_standardize(x, mean, std)

    x = add_bias(x)

    state = {
        "keep_named": keep_named,
        "keep_missing": keep_missing,
        "medians": medians,
        "mean": mean,
        "std": std,
    }
    return x, state


def preprocess_apply(x, names, state):
    """Replay the training recipe on new data (validation or test).
    Uses the stats in `state` — never recomputes anything from this data.
    """
    x = apply_none_map(x, names)
    x = apply_missing_map(x, names)

    x = x[:, state["keep_named"]]  # same column drops as train
    x = x[:, state["keep_missing"]]

    x = apply_impute(x, state["medians"])  # train medians
    x = apply_standardize(x, state["mean"], state["std"])  # train scale
    x = add_bias(x)
    return x


def train_val_split(x, y, val_ratio=0.2, seed=0):
    """Shuffle and split rows into train / validation."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_val = int(len(y) * val_ratio)
    val, tr = idx[:n_val], idx[n_val:]
    return x[tr], y[tr], x[val], y[val]


def f1_score(y_true, y_pred):
    """F1 for labels in {0, 1}. Punishes ignoring the rare positive class."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def predict_logistic(tx, w, threshold=0.5):
    """Probability >= threshold -> class 1, else class 0."""
    prob = 1 / (1 + np.exp(-tx @ w))
    return (prob >= threshold).astype(float)


def main():
    y01 = (y_train + 1) / 2  # logistic needs {0,1}, not {-1,1}

    xtr_raw, ytr, xval_raw, yval = train_val_split(x_train, y01, val_ratio=0.2, seed=0)

    # split names stay aligned with columns (same for every row)
    xtr, state = preprocess_fit(xtr_raw, feature_names)
    xval = preprocess_apply(xval_raw, feature_names, state)

    print("xtr:", xtr.shape, "| xval:", xval.shape)

    w0 = np.zeros(xtr.shape[1])
    w, loss = reg_logistic_regression(
        ytr, xtr, lambda_=0.0, initial_w=w0, max_iters=500, gamma=0.1
    )

    pred = predict_logistic(xval, w)
    print("train loss:   ", loss)
    print("validation F1:", f1_score(yval, pred))
    print("validation acc:", np.mean(pred == yval))
    print("fraction predicted positive:", np.mean(pred))


if __name__ == "__main__":
    main()
