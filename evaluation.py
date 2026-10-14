import numpy as np


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
