"""Feature filtering and preprocessing for the MICHD classification project."""

import numpy as np


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
