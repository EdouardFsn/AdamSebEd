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

    print("\nthreshold sweep:")
    for t, f1, frac in threshold_sweep(yval, prob, [0.3, 0.2, 0.15, 0.1]):
        print(f"  t={t:.3f} | F1={f1:.3f} | frac_pos={frac:.3f}")

    best_t = 0.2
    print(f"\ntrain loss: {loss:.4f}")
    print(
        f"validation F1 (t={best_t}): {f1_score(yval, (prob >= best_t).astype(float)):.3f}"
    )


if __name__ == "__main__":
    main()
