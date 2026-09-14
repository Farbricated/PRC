"""Robust feature engineering for PRC 2026 taxi-time prediction."""
import pandas as pd
import numpy as np

CAT_COLS = ['ADEP_mvt', 'RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'WK_TBL_CAT_flt', 'AIRCRAFT_OPERATOR_flt']

def fit_categorical_types(train_df, test_df):
    """Build unified CategoricalDtype for categorical columns across train and test sets."""
    cat_types = {}
    for c in CAT_COLS:
        train_vals = train_df[c].dropna().astype(str).unique() if c in train_df.columns else []
        test_vals = test_df[c].dropna().astype(str).unique() if c in test_df.columns else []
        all_vals = sorted(list(set(train_vals) | set(test_vals)))
        cat_types[c] = pd.CategoricalDtype(categories=all_vals)
    
    tr_pfx = train_df['STAND_mvt'].dropna().astype(str).str[:2].unique() if 'STAND_mvt' in train_df.columns else []
    te_pfx = test_df['STAND_mvt'].dropna().astype(str).str[:2].unique() if 'STAND_mvt' in test_df.columns else []
    all_pfx = sorted(list(set(tr_pfx) | set(te_pfx)))
    cat_types['stand_prefix'] = pd.CategoricalDtype(categories=all_pfx)
    return cat_types

def build_features(df, cat_types=None):
    """Extract temporal, operational, and flight-plan timing features."""
    f_df = pd.DataFrame(index=df.index)
    
    if 'AOBT_3_flt' in df.columns:
        f_df['diff_aobt'] = (df['MVT_TIME_UTC_mvt'] - df['AOBT_3_flt']).dt.total_seconds()
        f_df['diff_aobt_clipped'] = f_df['diff_aobt'].clip(0, 7200)
        f_df['has_aobt'] = df['AOBT_3_flt'].notna().astype(int)
        f_df['aobt_is_valid'] = ((f_df['diff_aobt'] > 0) & (f_df['diff_aobt'] < 7200)).astype(int)
    else:
        f_df['diff_aobt'] = np.nan
        f_df['diff_aobt_clipped'] = np.nan
        f_df['has_aobt'] = 0
        f_df['aobt_is_valid'] = 0

    if 'EOBT_1_flt' in df.columns:
        f_df['diff_eobt'] = (df['MVT_TIME_UTC_mvt'] - df['EOBT_1_flt']).dt.total_seconds().clip(-3600, 7200)
        f_df['has_eobt'] = df['EOBT_1_flt'].notna().astype(int)
    else:
        f_df['diff_eobt'] = np.nan
        f_df['has_eobt'] = 0

    if 'LOBT_flt' in df.columns:
        f_df['diff_lobt'] = (df['MVT_TIME_UTC_mvt'] - df['LOBT_flt']).dt.total_seconds().clip(-3600, 7200)
    else:
        f_df['diff_lobt'] = np.nan

    if 'IOBT_flt' in df.columns:
        f_df['diff_iobt'] = (df['MVT_TIME_UTC_mvt'] - df['IOBT_flt']).dt.total_seconds().clip(-3600, 7200)
    else:
        f_df['diff_iobt'] = np.nan

    if 'SCHED_TIME_UTC_mvt' in df.columns:
        f_df['diff_sched'] = (df['MVT_TIME_UTC_mvt'] - df['SCHED_TIME_UTC_mvt']).dt.total_seconds().clip(-3600, 7200)
    else:
        f_df['diff_sched'] = np.nan

    if 'AOBT_3_flt' in df.columns and 'EOBT_1_flt' in df.columns:
        f_df['aobt_minus_eobt'] = (df['AOBT_3_flt'] - df['EOBT_1_flt']).dt.total_seconds().clip(-3600, 7200)
    if 'AOBT_3_flt' in df.columns and 'LOBT_flt' in df.columns:
        f_df['aobt_minus_lobt'] = (df['AOBT_3_flt'] - df['LOBT_flt']).dt.total_seconds().clip(-3600, 7200)
    if 'AOBT_3_flt' in df.columns and 'SCHED_TIME_UTC_mvt' in df.columns:
        f_df['aobt_minus_sched'] = (df['AOBT_3_flt'] - df['SCHED_TIME_UTC_mvt']).dt.total_seconds().clip(-3600, 7200)
    if 'LOBT_flt' in df.columns and 'EOBT_1_flt' in df.columns:
        f_df['lobt_minus_eobt'] = (df['LOBT_flt'] - df['EOBT_1_flt']).dt.total_seconds().clip(-3600, 7200)

    mvt_dt = df['MVT_TIME_UTC_mvt'].dt
    f_df['hour'] = mvt_dt.hour
    f_df['minute'] = mvt_dt.minute
    f_df['dow'] = mvt_dt.dayofweek
    f_df['month'] = mvt_dt.month
    f_df['day'] = mvt_dt.day
    f_df['hour_sin'] = np.sin(2 * np.pi * f_df['hour'] / 24.0)
    f_df['hour_cos'] = np.cos(2 * np.pi * f_df['hour'] / 24.0)
    f_df['month_sin'] = np.sin(2 * np.pi * f_df['month'] / 12.0)
    f_df['month_cos'] = np.cos(2 * np.pi * f_df['month'] / 12.0)

    if 'STAND_mvt' in df.columns:
        f_df['stand_prefix'] = df['STAND_mvt'].dropna().astype(str).str[:2]
        if cat_types and 'stand_prefix' in cat_types:
            f_df['stand_prefix'] = f_df['stand_prefix'].astype(cat_types['stand_prefix'])

    for c in CAT_COLS:
        if c in df.columns:
            if cat_types and c in cat_types:
                f_df[c] = df[c].astype(str).astype(cat_types[c])
            else:
                f_df[c] = df[c].astype('category')

    return f_df
