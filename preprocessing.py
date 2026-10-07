"""Feature filtering and preprocessing for the MICHD classification project."""

import numpy as np

# Columns that carry no health signal: survey metadata, dates, IDs, phone bookkeeping.
METADATA_FEATURES = [
    # Survey administration / telephone screening
    "IDATE",
    "IDAY",
    "DISPCODE",
    "SEQNO",
    "_PSU",
    "CTELENUM",
    "PVTRESD1",
    "COLGHOUS",
    "STATERES",
    "CELLFON3",
    "LADULT",
    "CTELNUM1",
    "CELLFON2",
    "CADULT",
    "PVTRESD2",
    "CSTATE",
    # Questionnaire / language administration
    "QSTVER",
    "QSTLANG",
    # Free-text administrative fields
    "EXACTOT1",
    "EXACTOT2",
    # Survey weighting / sampling machinery
    "_STSTR",
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_DUALCOR",
    "_LLCPWT",
    "_CLLCPWT",
    # Purely administrative / derived sampling variables
    "MSCODE",
]

# Codes meaning "don't know / not sure / refused" -> become NaN
MISSING_VALUES = {
    # Single-digit categoricals: 7 = don't know, 9 = refused
    "MARITAL": [9],
    "EDUCA": [9],
    "EMPLOY1": [9],
    "GENHLTH": [7, 9],
    "HLTHPLN1": [7, 9],
    "PERSDOC2": [7, 9],
    "MEDCOST": [7, 9],
    "CHECKUP1": [7, 9],
    "BPHIGH4": [7, 9],
    "BPMEDS": [7, 9],
    "BLOODCHO": [7, 9],
    "CHOLCHK": [7, 9],
    "TOLDHI2": [7, 9],
    "CVDSTRK3": [7, 9],
    "ASTHMA3": [7, 9],
    "ASTHNOW": [7, 9],
    "CHCSCNCR": [7, 9],
    "CHCOCNCR": [7, 9],
    "CHCCOPD1": [7, 9],
    "HAVARTH3": [7, 9],
    "ADDEPEV2": [7, 9],
    "CHCKIDNY": [7, 9],
    "DIABETE3": [7, 9],
    "SMOKE100": [7, 9],
    "SMOKDAY2": [7, 9],
    "STOPSMK2": [7, 9],
    "USENOW3": [7, 9],
    "EXERANY2": [7, 9],
    "FLUSHOT6": [7, 9],
    "PNEUVAC3": [7, 9],
    "HIVTST6": [7, 9],
    "CAREGIV1": [7, 9],
    "SXORIENT": [7, 9],
    "TRNSGNDR": [7, 9],
    "CASTHDX2": [7, 9],
    "CASTHNO2": [7, 9],
    "EMTSUPRT": [7, 9],
    "LSATISFY": [7, 9],
    "MISTMNT": [7, 9],
    "ADANXEV": [7, 9],
    # Two-digit numeric quantities: 77 = don't know, 99 = refused
    "PHYSHLTH": [77, 99],
    "MENTHLTH": [77, 99],
    "POORHLTH": [77, 99],
    "LASTSMK2": [77, 99],
    "AVEDRNK2": [77, 99],
    "DRNK3GE5": [77, 99],
    "MAXDRNKS": [77, 99],
    "CHILDREN": [99],
    # Three-digit frequency variables: 777 = don't know, 999 = refused
    "FRUITJU1": [777, 999],
    "FRUIT1": [777, 999],
    "FVBEANS": [777, 999],
    "FVGREEN": [777, 999],
    "FVORANG": [777, 999],
    "VEGETAB1": [777, 999],
    "ALCDAY5": [777, 999],
    # Calculated-variable "unknown" codes (moved here from SPECIAL_VALUES)
    "DROCDY3_": [900],
    "_DRNKWEK": [99900],
    "_AGEG5YR": [14],
    "_PACAT1": [9],
    "_PAINDX1": [9],
    "_RFHLTH": [9],
    "_RFHYPE5": [9],
    "_LTASTH1": [9],
    "_CASTHM1": [9],
    "_ASTHMS1": [9],
    "_RFBMI5": [9],
    "_RFSEAT2": [9],
    "_RFSEAT3": [9],
    "_SMOKER3": [9],
    "_CHLDCNT": [9],
    "_FLSHOT6": [9],
    "_PNEUMO2": [9],
    "_AIDTST3": [9],
    # Physical-activity calculated quantities
    "PAFREQ1_": [99000],
    "PAFREQ2_": [99000],
    "MAXVO2_": [99900],
    "FC60_": [99900],
}

# Codes that encode "none / zero" -> become 0
NONE_VALUES = {
    # PHQ-style "number of days" items: 88 = none / 0 days
    "ADPLEASR": [88],
    "ADDOWN": [88],
    "ADSLEEP": [88],
    "ADENERGY": [88],
    "ADEAT1": [88],
    "ADFAIL": [88],
    "ADTHINK": [88],
    "ADMOVE": [88],
    # Alcohol: did not drink in past 30 days
    "ALCDAY5": [888],
    "CHILDREN": [88],
}

# Nominal categoricals: integer codes carry NO natural order
CATEGORICAL_FEATURES = [
    "MARITAL",
    "EMPLOY1",
    "_SMOKER3",
    "ACTIN11_",
    "ACTIN21_",
    "_ASTHMS1",
]

# Conditional ("skip-logic") features: a blank means the question was NOT asked because a precursor answer made it irrelevant
DEPENDENT_FEATURES = {
    # To fill later with features that are derived from other features, e.g.
    # "SMOKDAY2": {"precursor": "SMOKE100", "precursor_value": 2, "fill": 3},
    #   never smoked 100 cigs (SMOKE100 == 2) -> "not at all" (SMOKDAY2 = 3)
}


def drop_named(x, names, to_drop=METADATA_FEATURES):
    """Drop columns whose name is in `to_drop`. Returns reduced x, names, and the mask."""
    keep = np.array([n not in to_drop for n in names])
    return x[:, keep], [n for n, k in zip(names, keep) if k], keep


def apply_missing_map(x, names, missing_map=MISSING_VALUES):
    """Replace each column's sentinel codes (per missing_map) with nan.

    Args:
        x: numpy array of shape (N, D), the features.
        names: list of D feature names, aligned with the columns of x.
        missing_map: dict {feature_name: [sentinel values]}.

    Returns:
        out: numpy array of shape (N, D), with sentinels replaced by nan.
    """
    out = x.copy()
    for j, name in enumerate(names):
        if name in missing_map:
            out[np.isin(out[:, j], missing_map[name]), j] = np.nan
    return out


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
