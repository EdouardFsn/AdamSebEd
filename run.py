import numpy as np
from helpers import load_csv_data
from preprocessing import preprocess_fit, preprocess_apply
from evaluation import predict_proba, train_val_split, f1_score, threshold_sweep
from implementations import reg_logistic_regression


def main():
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/")

    with open("data/x_train.csv") as f:
        feature_names = f.readline().strip().split(",")[1:]

    y01 = (y_train + 1) / 2  # logistic needs {0,1}, not {-1,1}

    xtr_raw, ytr, xval_raw, yval = train_val_split(x_train, y01, val_ratio=0.2, seed=0)

    xtr, state = preprocess_fit(xtr_raw, feature_names)
    xval = preprocess_apply(xval_raw, feature_names, state)
    print("xtr:", xtr.shape, "| xval:", xval.shape)

    w0 = np.zeros(xtr.shape[1])
    w, loss = reg_logistic_regression(
        ytr, xtr, lambda_=0.0, initial_w=w0, max_iters=500, gamma=0.1
    )

    prob = predict_proba(xval, w)

    # Sanity checks: preprocessing must leave no NaN, and train/val must match
    assert not np.isnan(xtr).any() and not np.isnan(xval).any(), "NaN left after preprocessing"
    assert xtr.shape[1] == xval.shape[1], "train / val column mismatch"

    # Fine threshold sweep, best threshold picked automatically
    thresholds = np.arange(0.05, 0.51, 0.01)
    results = threshold_sweep(yval, prob, thresholds)
    best_t, best_f1, best_frac = max(results, key=lambda r: r[1])

    print("\nthreshold sweep (every 0.01):")
    for t, f1, frac in results:
        print(f"  t={t:.2f} | F1={f1:.3f} | frac_pos={frac:.3f}")

    print(f"\ntrain loss: {loss:.4f}")
    print(f"val positive rate: {yval.mean():.3f}")
    print(f"BEST: t={best_t:.2f} | F1={best_f1:.3f} | frac_pos={best_frac:.3f}")


if __name__ == "__main__":
    main()
