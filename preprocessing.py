"""Feature cleaning and preprocessing for the MICHD classification project.
 
The pipeline is split into a fit step (preprocess_fit, learns everything on the
training data) and an apply step (preprocess_apply, replays it on validation or
test data), so no statistic is ever computed on held-out rows.
 
All feature names and answer codes follow the BRFSS 2015 codebook. Some
calculated variables are rescaled in the course data (see MISSING_VALUES).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------

# Columns with no health signal, or with information that is invalid or
# duplicated elsewhere. Removed by drop_named.
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
    "QSTVER",
    "QSTLANG",
    "EXACTOT1",
    "EXACTOT2",
    # Survey weighting and sampling
    "_STSTR",
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_DUALCOR",
    "_LLCPWT",
    "_CLLCPWT",
    "MSCODE",
    # Interview date: when the survey was taken, not who was surveyed
    "FMONTH",
    "IMONTH",
    "IYEAR",
    # Household and phone bookkeeping (landline only, 40-60% blank)
    "NUMADULT",
    "NUMMEN",
    "NUMWOMEN",
    "HHADULT",
    "LANDLINE",
    "_DUALUSE",
    "CCLGHOUS",
    # About a child in the household, not the respondent
    "_CHISPNC",
    "_CRACE1",
    "_CPRACE",
    # Dates coded as MMYYYY, meaningless as magnitudes
    "FLSHTMY2",
    "HIVTSTD3",
    # Nominal codes with no order: vaccination place, activity types
    "IMFVPLAC",
    "EXRACT11",
    "EXRACT21",
    # Data-quality flags (was an answer missing), not health answers
    "_MISFRTN",
    "_MISVEGN",
    "_FRTRESP",
    "_VEGRESP",
    "_FRUITEX",
    "_VEGETEX",
    "PAMISS1_",
    # Only asked to people aged 65+, covered by FLUSHOT6 / PNEUVAC3
    "_FLSHOT6",
    "_PNEUMO2",
    # Same information as _RACE
    "_PRACE1",
    "_MRACE1",
    "_RACEGR3", 
    "_RACE_G1",
    "_RACEG21",
    "_HISPANC",
]

# Raw answers that pack a unit and a value into one number (e.g. ALCDAY5:
# 1xx = days per week, 2xx = days per month), so their size is meaningless.
# BRFSS provides a decoded version of each (shown after "->"), which we keep.
PACKED_FEATURES = [
    # Alcohol -> DROCDY3_, _DRNKWEK
    "ALCDAY5",
    # Fruit / vegetables -> FTJUDA1_, FRUTDA1_, BEANDAY_, GRENDAY_, ORNGDAY_,
    # VEGEDA1_, _FRUTSUM, _VEGESUM
    "FRUITJU1",
    "FRUIT1",
    "FVBEANS",
    "FVGREEN",
    "FVORANG",
    "VEGETAB1",
    # Exercise -> PAFREQ1_, PAFREQ2_, PADUR1_, PADUR2_
    "EXEROFT1",
    "EXEROFT2",
    "EXERHMM1",
    "EXERHMM2",
    # Strength training -> STRFREQ_
    "STRENGTH",
    # Weight and height (mixed units) -> WTKG3, HTM4, _BMI5
    "WEIGHT2",
    "HEIGHT3",
    # Diabetes self-care: no decoded version, 93% blank, covered by DIABETE3
    "BLDSUGAR",
    "FEETCHK2",
    # Not packed: code 9 mixes "aged 65+" with "don't know", and the variable
    # is rebuilt from HLTHPLN1 and age anyway
    "_HCVU651",
]

# Everything drop_named removes by default
DROP_FEATURES = METADATA_FEATURES + PACKED_FEATURES

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
    "DIFFWALK": [7, 9],
    "USEEQUIP": [7, 9],
    "QLACTLM2": [7, 9],
    "BLIND": [7, 9],
    "DECIDE": [7, 9],
    "DIFFDRES": [7, 9],
    "DIFFALON": [7, 9],
    "VETERAN3": [7, 9],
    "INTERNET": [7, 9],
    "RENTHOM1": [7, 9],
    "NUMHHOL2": [7, 9],
    "CPDEMO1": [7, 9],
    "LMTJOIN3": [7, 9],
    "ARTHDIS2": [7, 9],
    "ARTHSOCL": [7, 9],
    "CIMEMLOS": [7, 9],
    # 8 = never drives or rides in a car
    "SEATBELT": [7, 8, 9],
    # Two-digit numeric quantities: 77 = don't know, 99 = refused
    "PHYSHLTH": [77, 99],
    "MENTHLTH": [77, 99],
    "POORHLTH": [77, 99],
    "LASTSMK2": [77, 99],
    "AVEDRNK2": [77, 99],
    "DRNK3GE5": [77, 99],
    "MAXDRNKS": [77, 99],
    "CHILDREN": [99],
    "INCOME2": [77, 99],
    "DOCTDIAB": [77, 99],
    "JOINPAIN": [77, 99],
    "WHRTST10": [77, 99],
    "ADPLEASR": [77, 99],
    "ADDOWN": [77, 99], 
    "ADSLEEP": [77, 99],
    "ADENERGY": [77, 99],
    "ADEAT1": [77, 99],
    "ADFAIL": [77, 99],
    "ADTHINK": [77, 99],
    "ADMOVE": [77, 99],
    # Other two-digit codes (97 / 98 = don't know, 99 = refused)
    "SCNTLWK1": [97, 99],
    # 98 = never heard of the test
    "CHKHEMO3": [77, 98, 99],
    "ASERVIST": [98],
    "ASDRVIST": [98],
    "ASRCHKUP": [98, 99],
    "DIABAGE2": [98, 99],
    # Calculated variables: "unknown" code of each
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
    "_AIDTST3": [9],
    "_INCOMG": [9],
    "_EDUCAG": [9],
    "_RACE": [9],
    "_RACEG21": [9],
    "_RACEGR3": [9],
    "_HISPANC": [9],
    "_RFCHOL": [9],
    "_CHOLCHK": [9],
    "_RFSMOK3": [9],
    "_RFBING5": [9],
    "_RFDRHV5": [9],
    "_FRTLT1": [9],
    "_VEGLT1": [9],
    "_TOTINDA": [9],
    "_PA150R2": [9],
    "_PA300R2": [9],
    "_PA30021": [9],
    "_PASTRNG": [9],
    "_PAREC1": [9],
    "_PASTAE1": [9],
    "_LMTACT1": [9],
    "_LMTWRK1": [9],
    "_LMTSCL1": [9],
    "_AGE65YR": [3],
    "_PRACE1": [77, 99],
    "_MRACE1": [77, 99],
    "DRNKANY5": [7, 9],
    # Rescaled in the course data (divided by 1000 or 100), so the codebook's
    # 99000 / 99900 "unknown" codes appear as 99 / 999
    "PAFREQ1_": [99],
    "PAFREQ2_": [99],
    "STRFREQ_": [99],
    "MAXVO2_": [999],
    "FC60_": [999],
    "ASACTLIM": [777, 999],
}

# Codes meaning "none", replaced by 0 (88 / 888 = none, 98 = zero hours)
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
    "PHYSHLTH": [88],
    "MENTHLTH": [88],
    "POORHLTH": [88],
    # Counts (binge-drinking occasions, children, visits, tests, days)
    "DRNK3GE5": [88],
    "CHILDREN": [88],
    "DOCTDIAB": [88],
    "CHKHEMO3": [88],
    "ASERVIST": [88],
    "ASDRVIST": [88],
    "ASRCHKUP": [88],
    "ASACTLIM": [888],
    # Zero hours worked
    "SCNTLWK1": [98],
}

# Nominal or order-breaking categoricals, one-hot encoded by fit_onehot
CATEGORICAL_FEATURES = [
    "MARITAL",  # nominal: marital status
    "EMPLOY1",  # nominal: employment status
    "_RACE",  # nominal: race / ethnicity group
    "_STATE",  # nominal: US state code
    "RENTHOM1",  # nominal: own / rent / other
    "PERSDOC2",  # order broken: one doctor / several / none
    "BPHIGH4",  # order broken: yes / pregnancy only / no / borderline
    "DIABETE3",  # order broken: yes / pregnancy only / no / pre-diabetes
    "CHECKUP1",  # order broken: years since checkup, 8 = never
    "_PAREC1",  # order broken: aerobic only vs strength only
    "SXORIENT",  # nominal: sexual orientation
    "WHRTST10",  # nominal: place of HIV test
    "_SMOKER3",  # non-linear effect: smoking status
    "GENHLTH",  # non-linear effect: self-rated health
]

# Skip logic: a blank means the question was not asked because of an earlier
# answer. Each rule (target, {precursor: [values]}, fill) fills the target
# where it is blank and every precursor condition holds. Applied after the
# code mappings above, so "none" codes are already 0.
DEPENDENT_RULES = [
    # Non-drinkers (no drink-occasions) were not asked how much they drink
    ("AVEDRNK2", {"DROCDY3_": [0]}, 0),
    ("DRNK3GE5", {"DROCDY3_": [0]}, 0),
    ("MAXDRNKS", {"DROCDY3_": [0]}, 0),
    # Not hypertensive -> no blood-pressure medication (2 = No)
    ("BPMEDS", {"BPHIGH4": [2, 3, 4]}, 2),
    # No bad physical AND no bad mental health days -> 0 days limited
    ("POORHLTH", {"PHYSHLTH": [0], "MENTHLTH": [0]}, 0),
    # Never smoked 100 cigarettes -> smokes "not at all" (3)
    ("SMOKDAY2", {"SMOKE100": [2]}, 3),
    # Time since quitting: never smoked -> 8 (never), current smoker -> 0 (now)
    ("LASTSMK2", {"SMOKE100": [2]}, 8),
    ("LASTSMK2", {"SMOKDAY2": [1, 2]}, 0),
    # Cholesterol never checked -> after "5 or more years ago" (4)
    ("CHOLCHK", {"BLOODCHO": [2]}, 5),
    # No arthritis -> not limited, no joint pain
    ("LMTJOIN3", {"HAVARTH3": [2]}, 2),
    ("ARTHDIS2", {"HAVARTH3": [2]}, 2),
    ("ARTHSOCL", {"HAVARTH3": [2]}, 3),
    ("JOINPAIN", {"HAVARTH3": [2]}, 0),
    # Never had asthma -> does not have asthma now (2 = No)
    ("ASTHNOW", {"ASTHMA3": [2]}, 2),
]


# ---------------------------------------------------------------------------
# Cleaning steps: code mappings and column drops
# ---------------------------------------------------------------------------


def drop_named(x, names, to_drop=DROP_FEATURES):
    """Drop the columns whose name is in to_drop.
 
    Args:
        x: numpy array of shape (N, D), the features.
        names: list of D feature names, aligned with the columns of x.
        to_drop: list of feature names to remove.
 
    Returns:
        x_kept: numpy array of shape (N, D_kept), the kept columns.
        names_kept: list of D_kept kept feature names.
        keep: boolean numpy array of shape (D,), the kept-column mask.
    """
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


def apply_none_map(x, names, none_map=NONE_VALUES):
    """Replace each column's "none / zero" codes (per none_map) with 0.

    Args:
        x: numpy array of shape (N, D), the features.
        names: list of D feature names, aligned with the columns of x.
        none_map: dict {feature_name: [codes that mean zero]}.

    Returns:
        out: numpy array of shape (N, D), with "none" codes replaced by 0.
    """
    out = x.copy()
    for j, name in enumerate(names):
        if name in none_map:
            out[np.isin(out[:, j], none_map[name]), j] = 0
    return out


def apply_dependent(x, names, rules=DEPENDENT_RULES):
    """Fill blanks caused by skip logic, using the answers to earlier questions.
 
    For each rule (target, conditions, fill), the rows where the target is NaN
    and every precursor value is in its allowed list get target = fill.
 
    Args:
        x: numpy array of shape (N, D), the features (missing values as NaN).
        names: list of D feature names, aligned with the columns of x.
        rules: list of (target, {precursor: [values]}, fill).
 
    Returns:
        out: numpy array of shape (N, D), with skip-logic blanks filled.
    """
    out = x.copy()
    for target, conditions, fill in rules:
        if target not in names or any(p not in names for p in conditions):
            continue
        mask = np.isnan(out[:, names.index(target)])
        for precursor, values in conditions.items():
            mask &= np.isin(out[:, names.index(precursor)], values)
        out[mask, names.index(target)] = fill
    return out


def drop_high_missing(x, names, max_missing=0.8):
    """Keep only the columns whose fraction of NaN is at most max_missing.
 
    Args:
        x: numpy array of shape (N, D), the features (missing values as NaN).
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


# ---------------------------------------------------------------------------
# Fit / apply pairs: fit learns on train, apply replays on any data
# ---------------------------------------------------------------------------


def fit_onehot(x, names, categorical=CATEGORICAL_FEATURES):
    """Learn how to one-hot encode each nominal categorical column.

    The most frequent value of each column is the reference level and gets no
    indicator column (it is encoded as all zeros). Otherwise the indicators of
    one feature would sum to the bias column and make least_squares singular.
    If the column has missing values, "missing" gets its own indicator, so that
    it is not confused with the reference level.

    Args:
        x: numpy array of shape (N, D), the features (missing values as nan).
        names: list of D feature names, aligned with the columns of x.
        categorical: list of names of the columns to encode.

    Returns:
        encoding: list of (name, levels, has_missing) tuples, one per encoded
            column. levels are the non-reference values that get an indicator
            column, has_missing says whether a "missing" indicator is added.
    """
    encoding = []
    for j, name in enumerate(names):
        if name not in categorical:
            continue
        col = x[:, j]
        values, counts = np.unique(col[~np.isnan(col)], return_counts=True)
        reference = values[np.argmax(counts)]
        levels = [v for v in values if v != reference]
        encoding.append((name, levels, bool(np.isnan(col).any())))
    return encoding


def apply_onehot(x, names, encoding):
    """Replace each encoded column by its indicator columns (per encoding).

    The reference level, and any value not seen when fitting, become all zeros.

    Args:
        x: numpy array of shape (N, D), the features (missing values as nan).
        names: list of D feature names, aligned with the columns of x.
        encoding: list of (name, levels, has_missing), from fit_onehot.

    Returns:
        out: numpy array of shape (N, D_out), the non-encoded columns followed
            by the indicator columns.
        names_out: list of D_out feature names, indicators are named like
            "EMPLOY1=7" or "EMPLOY1=missing".
    """
    encoded = {name for name, _, _ in encoding}
    keep = [j for j, name in enumerate(names) if name not in encoded]
    columns = [x[:, keep]]
    names_out = [names[j] for j in keep]
    for name, levels, has_missing in encoding:
        col = x[:, names.index(name)]
        for value in levels:
            columns.append((col == value).astype(float)[:, None])
            names_out.append(f"{name}={value:g}")
        if has_missing:
            columns.append(np.isnan(col).astype(float)[:, None])
            names_out.append(f"{name}=missing")
    return np.hstack(columns), names_out


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

def fit_missing_indicators(x, min_frac=0.05):
    """Find the columns that deserve a "was missing" indicator.
 
    Args:
        x: numpy array of shape (N, D), the features (missing values as NaN).
        min_frac: float, minimum fraction of NaN for a column to be selected.
 
    Returns:
        cols: numpy array of the indices of the selected columns.
    """
    return np.where(np.isnan(x).mean(axis=0) > min_frac)[0]


def apply_missing_indicators(x, names, cols):
    """Append a 0/1 "was missing" column for each selected column.
 
    Args:
        x: numpy array of shape (N, D), the features (missing values as NaN).
        names: list of D feature names, aligned with the columns of x.
        cols: indices of the columns to flag, from fit_missing_indicators.
 
    Returns:
        out: numpy array of shape (N, D + len(cols)), x followed by the flags.
        names_out: list of feature names, flags are named like "BPMEDS=nan".
    """
    ind = np.isnan(x[:, cols]).astype(float)
    return np.hstack([x, ind]), names + [names[j] + "=nan" for j in cols]


def fit_standardize(x):
    """Compute the mean and standard deviation of each column.
 
    Args:
        x: numpy array of shape (N, D), the features.
 
    Returns:
        mean: numpy array of shape (D,), the per-column means.
        std: numpy array of shape (D,), the per-column stds (0 replaced by 1).
    """
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_standardize(x, mean, std):
    """Standardize each column with the given mean and standard deviation.
 
    Args:
        x: numpy array of shape (N, D), the features.
        mean: numpy array of shape (D,), from fit_standardize.
        std: numpy array of shape (D,), from fit_standardize.
 
    Returns:
        out: numpy array of shape (N, D), the standardized features.
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


# ---------------------------------------------------------------------------
# Full pipeline: fit on train, replay on validation / test
# ---------------------------------------------------------------------------


def preprocess_fit(x, names, use_categoricals=False, use_dependent=False, max_missing =0.8, use_missing_indicators=False):
    """Clean the training data and return everything learned on it.
 
    Steps, in order: code mappings, skip-logic fill, column drops, one-hot
    encoding, missing indicators, median imputation, standardization, bias.
 
    Args:
        x: numpy array of shape (N, D), the raw training features.
        names: list of D feature names, aligned with the columns of x.
        use_categoricals: bool, one-hot encode CATEGORICAL_FEATURES.
        use_dependent: bool, fill skip-logic blanks with DEPENDENT_RULES.
        max_missing: float, drop columns with a larger fraction of NaN.
        use_missing_indicators: bool, add a "was missing" column for columns
            with more than 5% NaN.
 
    Returns:
        x_clean: numpy array of shape (N, D_out), the processed training data.
        state: dict of everything learned on train (kept columns, encoding,
            indicator columns, medians, mean, std, final feature names), to be
            replayed by preprocess_apply.
    """
    x = apply_none_map(x, names)  
    x = apply_missing_map(x, names)  
    if use_dependent:
        x = apply_dependent(x, names)    # before dropping, so filled columns count

    x, names, keep_named = drop_named(x, names)  
    x, names, keep_missing = drop_high_missing(x, names, max_missing)  

    # before imputing, so that missing values get their own indicator column
    encoding = fit_onehot(x, names) if use_categoricals else []
    x, names = apply_onehot(x, names, encoding)

    # before imputing, so we still know which values were missing
    mi_cols = fit_missing_indicators(x) if use_missing_indicators else []
    x, names = apply_missing_indicators(x, names, mi_cols)

    medians = fit_impute(x)  
    x = apply_impute(x, medians)

    mean, std = fit_standardize(x)  
    x = apply_standardize(x, mean, std)

    x = add_bias(x)

    state = {
        "keep_named": keep_named,
        "keep_missing": keep_missing,
        "encoding": encoding,
        "mi_cols": mi_cols,
        "use_dependent": use_dependent,  
        "medians": medians,
        "mean": mean,
        "std": std,
        "names": ["bias"] + names,  # aligned with the columns of x_clean / w
    }
    return x, state


def preprocess_apply(x, names, state):
    """Replay the training recipe on new data (validation or test).
 
    Uses only what is stored in state and never recomputes anything from this
    data, so the output has exactly the same columns as the training matrix.
 
    Args:
        x: numpy array of shape (N, D), the raw features.
        names: list of D feature names, aligned with the columns of x.
        state: dict returned by preprocess_fit.
 
    Returns:
        x_clean: numpy array of shape (N, D_out), the processed data.
    """
    x = apply_none_map(x, names)
    x = apply_missing_map(x, names)

    if state["use_dependent"]:
        x = apply_dependent(x, names)

    x = x[:, state["keep_named"]]  # same column drops as train
    x = x[:, state["keep_missing"]]
    names = [n for n, k in zip(names, state["keep_named"]) if k]
    names = [n for n, k in zip(names, state["keep_missing"]) if k]

    x, names = apply_onehot(x, names, state["encoding"])  # train levels
    x, names = apply_missing_indicators(x, names, state["mi_cols"])  # train columns

    x = apply_impute(x, state["medians"])  # train medians
    x = apply_standardize(x, state["mean"], state["std"])  # train scale
    x = add_bias(x)
    return x
