"""Feature filtering and preprocessing for the MICHD classification project."""

import numpy as np

# Columns that carry no health signal: survey metadata, dates, IDs, phone bookkeeping...
METADATA_FEATURES = [
    "_STATE",
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
]


def drop_named(x, names, to_drop=METADATA_FEATURES):
    """Drop columns whose name is in `to_drop`. Returns reduced x, names, and the mask."""
    keep = np.array([n not in to_drop for n in names])
    return x[:, keep], [n for n, k in zip(names, keep) if k], keep


def drop_high_missing(x, names, max_missing=0.8):
    """Keep only columns whose missing fraction is <= max_missing.

    Args:
        x: numpy array of shape (N, D), the features.
        names: list of D feature names, aligned with the columns of x.
        max_missing: float, maximum allowed fraction of missing values.

    Returns:
        x_kept: numpy array of shape (N, D_kept), the kept columns.
        names_kept: list of D_kept kept feature names.
        keep: boolean numpy array of shape (D,), the kept-column mask.
    """
    missing_frac = np.isnan(x).mean(axis=0)
    keep = missing_frac <= max_missing
    return x[:, keep], [n for n, k in zip(names, keep) if k], keep


def fit_impute(x):
    """Compute the per-column median, ignoring missing values.

    Args:
        x: numpy array of shape (N, D), the features.

    Returns:
        medians: numpy array of shape (D,), the per-column medians.
    """
    return np.nanmedian(x, axis=0)


def apply_impute(x, medians):
    """Replace missing values by the provided per-column medians.

    Args:
        x: numpy array of shape (N, D), the features.
        medians: numpy array of shape (D,), the per-column medians.

    Returns:
        out: numpy array of shape (N, D), with no missing values.
    """
    out = x.copy()
    rows, cols = np.where(np.isnan(out))
    out[rows, cols] = np.take(medians, cols)
    return out


def fit_standardize(x):
    """Compute the per-column mean and std (std of 0 is replaced by 1).

    Args:
        x: numpy array of shape (N, D), the features.

    Returns:
        mean: numpy array of shape (D,), the per-column means.
        std: numpy array of shape (D,), the per-column stds (zeros -> 1).
    """
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_standardize(x, mean, std):
    """Standardize using the provided per-column mean and std.

    Args:
        x: numpy array of shape (N, D), the features.
        mean: numpy array of shape (D,), the per-column means.
        std: numpy array of shape (D,), the per-column stds.

    Returns:
        numpy array of shape (N, D), standardized.
    """
    return (x - mean) / std


def add_bias(x):
    """Prepend a column of ones for the bias/intercept term.

    Args:
        x: numpy array of shape (N, D), the features.

    Returns:
        numpy array of shape (N, D + 1), with a leading column of ones.
    """
    return np.c_[np.ones(x.shape[0]), x]
