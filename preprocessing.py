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
    # Interview month / year: when the survey was taken, not who was surveyed
    "FMONTH",
    "IMONTH",
    "IYEAR",
    # Household / phone bookkeeping (landline-only, ~40-60% blank by design)
    "NUMADULT",
    "NUMMEN",
    "NUMWOMEN",
    "HHADULT",
    "LANDLINE",
    "_DUALUSE",
    "CCLGHOUS",
    # Child-survey variables (about a child in the household, not the respondent)
    "_CHISPNC",
    "_CRACE1",
    "_CPRACE",
    # Raw dates encoded as MMYYYY: a linear model would read them as magnitudes
    "FLSHTMY2",
    "HIVTSTD3",
    # Place of vaccination: nominal, 57% blank
    "IMFVPLAC",
    "EXRACT11",
    "EXRACT21",
    # Data-quality flags: whether an answer was missing, not a health answer
    "_MISFRTN",
    "_MISVEGN",
    "_FRTRESP",
    "_VEGRESP",
    "_FRUITEX",
    "_VEGETEX",
    "PAMISS1_",
    # Only asked to 65+ (64% blank); FLUSHOT6 / PNEUVAC3 cover everyone
    "_FLSHOT6",
    "_PNEUMO2",
    # Race / ethnicity: same information as _RACE (one-hot encoded)
    "_PRACE1",
    "_MRACE1",
    "_RACEGR3", 
    "_RACE_G1",
    "_RACEG21",
    "_HISPANC",
]

# Raw answers that pack a unit and a value into one number, e.g. ALCDAY5:
# 1xx = days per week, 2xx = days in past 30, 888 = none. A linear model would
# read 230 (30 days a month) as more than 107 (7 days a week), which is wrong.
# BRFSS already provides a decoded version of each one, so we drop the raw
# column. Checked on x_train: the decoded column is present for 100% of valid
# raw answers and blank where the raw answer is don't know / refused.
PACKED_FEATURES = [
    # Alcohol -> DROCDY3_ (drink-days per 100 days), _DRNKWEK (drinks per week)
    "ALCDAY5",
    # Fruit / vegetables (1xx per day, 2xx per week, 3xx per month,
    # 555 = never) -> FTJUDA1_, FRUTDA1_, BEANDAY_, GRENDAY_, ORNGDAY_,
    # VEGEDA1_ (times per day) and the sums _FRUTSUM, _VEGESUM
    "FRUITJU1",
    "FRUIT1",
    "FVBEANS",
    "FVGREEN",
    "FVORANG",
    "VEGETAB1",
    # Exercise (1xx per week, 2xx per month; durations as H:MM, so 130 = 1h30)
    # -> PAFREQ1_, PAFREQ2_ (times per week), PADUR1_, PADUR2_ (minutes)
    "EXEROFT1",
    "EXEROFT2",
    "EXERHMM1",
    "EXERHMM2",
    # Strength training (1xx per week, 2xx per month, 888 = never) -> STRFREQ_
    "STRENGTH",
    # Weight mixes pounds and kg (9xxx), height mixes ft-in (507 = 5'7") and
    # cm (9xxx), both with 7777 / 9999 codes -> WTKG3 (kg), HTM4 (m), _BMI5
    "WEIGHT2",
    "HEIGHT3",
    # Diabetes self-care frequencies (same 1xx/2xx/3xx scheme) have no decoded
    # version, but only diabetics were asked (93% blank) and DIABETE3 already
    # carries that information.
    "BLDSUGAR",
    "FEETCHK2",
    # 9 mixes "aged 65+, not asked" (36% of rows) with "don't know";
    # rebuilt from HLTHPLN1 + age anyway
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
    "SCNTLWK1": [97, 99],
    "CHKHEMO3": [77, 98, 99],
    "ASERVIST": [98],
    "ASDRVIST": [98],
    "ASRCHKUP": [98, 99],
    # Age at diabetes diagnosis: 98 = don't know, 99 = refused
    "DIABAGE2": [98, 99],
    # (The 777 / 999 codes of the packed frequency variables are not listed:
    # those columns are dropped, see PACKED_FEATURES.)
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
    # Physical-activity calculated quantities. The course data divided these
    # by 1000 (PAFREQ*, STRFREQ_) or 100 (MAXVO2_, FC60_) compared with the
    # BRFSS codebook, so the codebook's 99000 / 99900 appear here as 99 / 999.
    "PAFREQ1_": [99],
    "PAFREQ2_": [99],
    "STRFREQ_": [99],
    "MAXVO2_": [999],
    "FC60_": [999],
    "ASACTLIM": [777, 999],
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
    # Days of bad physical / mental health in the past 30 days: 88 = none
    "PHYSHLTH": [88],
    "MENTHLTH": [88],
    "POORHLTH": [88],
    # Alcohol: no binge-drinking occasion in past 30 days
    "DRNK3GE5": [88],
    "CHILDREN": [88],
    # Doctor visits for diabetes in past 12 months: 88 = none
    "DOCTDIAB": [88],
    # Diabetes and asthma visit/test counts
    "CHKHEMO3": [88],
    "ASERVIST": [88],
    "ASDRVIST": [88],
    "ASRCHKUP": [88],
    "ASACTLIM": [888],
    # Zero hours worked
    "SCNTLWK1": [98],
}

# Nominal or order-breaking categoricals -> one-hot encoded
CATEGORICAL_FEATURES = [
    "MARITAL",   # married / divorced / widowed / ... : no order
    "EMPLOY1",   # employed / student / retired / unable to work : no order
    "_RACE",     # race / ethnicity groups : no order
    "_STATE",    # US state FIPS code : no order (53 values)
    "RENTHOM1",  # own / rent / other : no order
    "PERSDOC2",  # 1 = one doctor, 2 = more than one, 3 = none : "none" is not "more"
    "BPHIGH4",   # 1 yes, 2 yes only in pregnancy, 3 no, 4 borderline : order broken
    "DIABETE3",  # 1 yes, 2 pregnancy only, 3 no, 4 pre-diabetes : order broken
    "CHECKUP1",  # 1-4 = years since checkup, 8 = never : order broken
    "_PAREC1",   # met both / aerobic only / strength only / neither : 2 vs 3 not ordered
    "SXORIENT",  # straight / gay / bisexual / other : no order
    "WHRTST10",  # place of HIV test : no order
    "_SMOKER3",
    "GENHLTH"
]

# Skip-logic: a blank means the question was NOT asked because of an earlier
# answer. Each rule: (target, {precursor: [values]}, fill). The target is filled
# only where it is blank AND every precursor condition holds.
# Applied after apply_none_map / apply_missing_map, so "none" codes are already 0.
# Checked on x_train: every row matching a condition has a blank target.
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


def drop_named(x, names, to_drop=DROP_FEATURES):
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

    For each rule (target, conditions, fill), rows where the target is nan and
    every precursor value is in its allowed list get target = fill. Fixed rules
    from the codebook: nothing is learned from the data.

    Args:
        x: numpy array of shape (N, D), the features (missing values as nan).
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


# The two following functions are used in run.py, they call all the above helpers in the correct order to clean and standardize the training data, and then replay the same transformations on validation/test data.


def preprocess_fit(x, names, use_categoricals=False, use_dependent=False):
    """Clean training data, and RETURN the stats learned, so the exact same
    transform can be replayed on validation/test.

    Args:
        use_categoricals: if True, one-hot encode CATEGORICAL_FEATURES instead
            of feeding their raw integer codes to the model.

    Returns:
        x_clean: the processed training matrix (with bias column).
        state: dict of everything learned on train (which columns to keep,
               one-hot encoding, medians, mean, std, final feature names)
               — the "recipe" to replay elsewhere.
    """
    x = apply_none_map(x, names)  # none-codes -> 0   (while values are raw)
    x = apply_missing_map(x, names)  # sentinels  -> nan
    if use_dependent:
        x = apply_dependent(x, names)    # fill skip-logic blanks before dropping

    x, names, keep_named = drop_named(x, names)  # drop metadata
    x, names, keep_missing = drop_high_missing(x, names)  # drop near-empty cols

    # before imputing, so that missing values get their own indicator column
    encoding = fit_onehot(x, names) if use_categoricals else []
    x, names = apply_onehot(x, names, encoding)

    medians = fit_impute(x)  # learn medians on train
    x = apply_impute(x, medians)

    mean, std = fit_standardize(x)  # learn scale on train
    x = apply_standardize(x, mean, std)

    x = add_bias(x)

    state = {
        "keep_named": keep_named,
        "keep_missing": keep_missing,
        "encoding": encoding,
        "use_dependent": use_dependent,  # replayed by preprocess_apply
        "medians": medians,
        "mean": mean,
        "std": std,
        "names": ["bias"] + names,  # aligned with the columns of x_clean / w
    }
    return x, state


def preprocess_apply(x, names, state):
    """Replay the training recipe on new data (validation or test).
    Uses the stats in `state` — never recomputes anything from this data.
    Whether categoricals are one-hot encoded is also read from `state`, so
    validation/test are always encoded the same way as train.
    """
    x = apply_none_map(x, names)
    x = apply_missing_map(x, names)

    if state["use_dependent"]:
        x = apply_dependent(x, names)

    x = x[:, state["keep_named"]]  # same column drops as train
    x = x[:, state["keep_missing"]]
    names = [n for n, k in zip(names, state["keep_named"]) if k]
    names = [n for n, k in zip(names, state["keep_missing"]) if k]

    x, _ = apply_onehot(x, names, state["encoding"])  # train levels

    x = apply_impute(x, state["medians"])  # train medians
    x = apply_standardize(x, state["mean"], state["std"])  # train scale
    x = add_bias(x)
    return x
