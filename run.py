import numpy as np
from helpers import load_csv_data
from preprocessing import preprocess_fit, preprocess_apply
from evaluation import train_val_split, f1_score, predict_logistic
from implementations import reg_logistic_regression


def main():
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/")

    with open("data/x_train.csv") as f:
        feature_names = f.readline().strip().split(",")[1:]

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

    print("\nthreshold sweep:")
    for t in [0.3, 0.2, 0.15, 0.1]:
        pred_t = predict_logistic(xval, w, threshold=t)
        print(
            f"  t={t:.3f} | F1={f1_score(yval, pred_t):.3f} | "
            f"frac_pos={np.mean(pred_t):.3f}"
        )

    pred = predict_logistic(xval, w)
    print("\ntrain loss:   ", loss)
    print("validation F1:", f1_score(yval, pred))
    print("validation acc:", np.mean(pred == yval))
    print("fraction predicted positive:", np.mean(pred))


if __name__ == "__main__":
    main()
