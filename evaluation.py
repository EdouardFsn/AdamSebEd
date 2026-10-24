import numpy as np

from preprocessing import preprocess_fit, preprocess_apply
from implementations import reg_logistic_regression, compute_logistic_loss
from implementations import sigmoid


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


def predict_proba(tx, w):
    """Model's predicted probability of class 1."""
    return 1 / (1 + np.exp(-tx @ w))


def predict_logistic(tx, w, threshold=0.5):
    """Probability >= threshold -> class 1, else class 0."""
    return (predict_proba(tx, w) >= threshold).astype(float)


def threshold_sweep(y_true, prob, thresholds):
    """F1 and positive-fraction at each threshold. Returns list of (t, f1, frac)."""
    results = []
    for t in thresholds:
        pred = (prob >= t).astype(float)
        results.append((t, f1_score(y_true, pred), float(np.mean(pred))))
    return results


def k_fold_indices(n, k=5, seed=0):
    """Split row indices 0..n-1 into k shuffled folds.

    Returns a list of k arrays of indices; each is used once as validation.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return np.array_split(idx, k)


def cross_validate(
    x,
    y,
    names,
    train_model,
    k=5,
    threshold=0.2,
    seed=0,
    use_categoricals=False,
    use_dependent=False,
    max_missing=0.8,
    use_missing_indicators=False,   
):
    """Run k-fold CV. For each fold: preprocess (fit on train folds, apply to
    val fold), train the model, score F1 on the val fold.

    Args:
        x, y: raw features and {0,1} labels (before any preprocessing).
        names: feature names aligned with x's columns.
        train_model: function (xtr, ytr) -> w, the trained weight vector.
        threshold: decision threshold for F1.

    Returns:
        scores: numpy array of k F1 scores, one per fold.
    """
    folds = k_fold_indices(len(y), k=k, seed=seed)
    scores = []
    for i in range(k):
        val_idx = folds[i]
        tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        xtr, state = preprocess_fit(
            x[tr_idx],
            names,
            use_categoricals=use_categoricals,
            use_dependent=use_dependent,
            max_missing=max_missing,
            use_missing_indicators=use_missing_indicators,
        )
        xval = preprocess_apply(x[val_idx], names, state)

        w = train_model(xtr, y[tr_idx])
        pred = (predict_proba(xval, w) >= threshold).astype(float)
        scores.append(f1_score(y[val_idx], pred))
    return np.array(scores)


def summarize_cv(scores):
    """Mean and std of fold scores, as a (mean, std) tuple."""
    return float(scores.mean()), float(scores.std())


def best_threshold(y_true, prob, thresholds):
    """Return the threshold (from `thresholds`) with the highest F1 on these predictions."""
    scored = [(t, f1_score(y_true, (prob >= t).astype(float))) for t in thresholds]
    return max(scored, key=lambda tf: tf[1])[0]


def cross_validate_auto_threshold(
    x,
    y,
    names,
    train_model,
    k=5,
    seed=0,
    use_categoricals=False,
    use_dependent=False,
    max_missing=0.8,
    use_missing_indicators=False,
    thresholds=np.arange(0.1, 0.4, 0.01),
):
    """k-fold cross validation that selects the threshold on each fold's training data and measures F1 on the held-out fold with that threshold.

    Honest: the held-out fold never influences its own threshold.

    Returns:
        scores: k F1 scores (one per fold, at each fold's chosen threshold).
        chosen: the k thresholds chosen, so we can see how stable they are.
    """
    folds = k_fold_indices(len(y), k=k, seed=seed)
    scores, chosen = [], []
    for i in range(k):
        val_idx = folds[i]
        tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        xtr, state = preprocess_fit(
            x[tr_idx],
            names,
            use_categoricals=use_categoricals,
            use_dependent=use_dependent,
            max_missing=max_missing,
            use_missing_indicators=use_missing_indicators,
        )
        xval = preprocess_apply(x[val_idx], names, state)

        w = train_model(xtr, y[tr_idx])

        # choose threshold on the training predictions, never on the held-out fold
        t = best_threshold(y[tr_idx], predict_proba(xtr, w), thresholds)
        pred = (predict_proba(xval, w) >= t).astype(float)

        scores.append(f1_score(y[val_idx], pred))
        chosen.append(t)
    return np.array(scores), np.array(chosen)


def make_logistic_trainer(lambda_=0.0, gamma=0.1, max_iters=500):
    """Build a train_model(xtr, ytr) -> w function, as used by cross_validate.

    Args:
        lambda_: float, L2 regularization strength of reg_logistic_regression.
        gamma: float, gradient descent step size.
        max_iters: int, number of gradient descent steps.

    Returns:
        train_model: function (xtr, ytr) -> w, trained from w = 0.
    """

    def train_model(xtr, ytr):
        w0 = np.zeros(xtr.shape[1])
        w, _ = reg_logistic_regression(ytr, xtr, lambda_, w0, max_iters, gamma)
        return w

    return train_model


def hyperparameter_sweep(
    x,
    y,
    names,
    train_models,
    thresholds,
    k=5,
    seed=0,
    use_categoricals=False,
    use_dependent=False,
    max_missing=0.8,
    use_missing_indicators=False,
):
    """Cross-validate several trainers on the same k folds.

    Args:
        x, y: raw features and {0,1} labels (before any preprocessing).
        names: feature names aligned with x's columns.
        train_models: list of functions (xtr, ytr) -> w, e.g. from
            make_logistic_trainer.
        thresholds: 1D array of decision thresholds to score.

    Returns:
        f1: numpy array (n_models, k, n_thresholds), val F1 per fold/threshold.
        train_loss: numpy array (n_models, k), logistic loss on the train folds
            (without penalty), to check that gradient descent converged.
    """
    folds = k_fold_indices(len(y), k=k, seed=seed)
    f1 = np.zeros((len(train_models), k, len(thresholds)))
    train_loss = np.zeros((len(train_models), k))
    for i in range(k):
        val_idx = folds[i]
        tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        xtr, state = preprocess_fit(
            x[tr_idx],
            names,
            use_categoricals=use_categoricals,
            use_dependent=use_dependent,
            max_missing=max_missing,
            use_missing_indicators=use_missing_indicators,
        )
        xval = preprocess_apply(x[val_idx], names, state)

        for m, train_model in enumerate(train_models):
            w = train_model(xtr, y[tr_idx])
            train_loss[m, i] = compute_logistic_loss(y[tr_idx], xtr, w)
            prob = predict_proba(xval, w)
            for t, threshold in enumerate(thresholds):
                pred = (prob >= threshold).astype(float)
                f1[m, i, t] = f1_score(y[val_idx], pred)
    return f1, train_loss


def compute_weighted_logistic_gradient(y, tx, w, sample_weights):
    """Logistic gradient with a per-sample weight on each error term."""
    pred = sigmoid(tx @ w)
    return tx.T @ (sample_weights * (pred - y)) / len(y)


def make_weighted_trainer(pos_weight=1.0, gamma=0.5, max_iters=500):
    """Build a train_model(xtr, ytr) -> w that up-weights the positive class.

    pos_weight = 1.0 recovers ordinary logistic regression. A value near the
    imbalance ratio (~10) makes a missed positive cost ~10x a missed negative.
    """

    def train_model(xtr, ytr):
        w = np.zeros(xtr.shape[1])
        # weight: pos_weight for y=1 rows, 1.0 for y=0 rows
        s = np.where(ytr == 1, pos_weight, 1.0)
        for _ in range(max_iters):
            w -= gamma * compute_weighted_logistic_gradient(ytr, xtr, w, s)
        return w

    return train_model
