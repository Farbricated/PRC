"""PRC 2026 - merry-iguana v8: Advanced Delta Formulation with Target Encoding
Key improvements:
1. Predict delta = BLOCK_TIME - AOBT_3 (the offset between surveillance and actual block)
2. Airport+Runway+Stand target encoding with K-fold out-of-fold predictions
3. Direct TAXITIME prediction as ensemble backup
4. Better handling of extreme outliers
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
import glob
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚀 merry-iguana v8: Delta Formulation + Target Encoding + Ensemble")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")
train_files = sorted(glob.glob('data/raw/training_*.parquet'))
test_ranking = pd.read_parquet('data/raw/ranking.parquet')
test_submit = pd.read_parquet('data/raw/submitting.parquet')

# Filter to departures
test_dep = test_ranking[test_ranking['PHASE_mvt'] == 'DEP'].copy()
test_submit_dep = test_submit[test_submit['MVT_ID_mvt'].isin(test_dep['MVT_ID_mvt'])].copy()

print(f"Test set: {len(test_dep):,} DEP flights, {test_submit_dep.shape[0]:,} to submit")

# Load training data - all files, sampled for memory efficiency
train_list = []
for i, f in enumerate(train_files):
    df = pd.read_parquet(f)
    dep = df[df['PHASE_mvt'] == 'DEP'].copy()
    # Keep only valid taxi times
    dep = dep[(dep['TAXITIME_SEC_mvt'].notna()) & (dep['TAXITIME_SEC_mvt'] > 0)].copy()
    # Sample 20% from each month (reduced for memory)
    if len(dep) > 15000:
        dep = dep.sample(frac=0.20, random_state=42+i)
    train_list.append(dep)
    print(f"  Loaded {i+1}/12: {len(dep):,} rows")
    del df, dep  # Free memory

train = pd.concat(train_list, ignore_index=True)
print(f"Total training: {len(train):,} DEP flights")

# Free memory
del train_list
import gc
gc.collect()

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
print("\n[2/6] Engineering features...")

def engineer_features(df, is_train=True):
    """Extract temporal, operational, and flight-plan timing features."""
    f_df = pd.DataFrame(index=df.index)
    
    # Flight plan timing deltas (critical features!)
    if 'AOBT_3_flt' in df.columns:
        f_df['diff_aobt'] = (df['MVT_TIME_UTC_mvt'] - df['AOBT_3_flt']).dt.total_seconds()
        f_df['has_aobt'] = df['AOBT_3_flt'].notna().astype(int)
        # Clip to reasonable range but keep extremes for outlier detection
        f_df['diff_aobt_clipped'] = f_df['diff_aobt'].clip(-600, 7200)
    else:
        f_df['diff_aobt'] = np.nan
        f_df['has_aobt'] = 0
        f_df['diff_aobt_clipped'] = np.nan
    
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
    
    # Time-based features
    mvt_dt = df['MVT_TIME_UTC_mvt'].dt
    f_df['hour'] = mvt_dt.hour
    f_df['minute'] = mvt_dt.minute
    f_df['dow'] = mvt_dt.dayofweek
    f_df['month'] = mvt_dt.month
    f_df['day'] = mvt_dt.day
    
    # Cyclical encodings
    f_df['hour_sin'] = np.sin(2 * np.pi * f_df['hour'] / 24.0)
    f_df['hour_cos'] = np.cos(2 * np.pi * f_df['hour'] / 24.0)
    f_df['dow_sin'] = np.sin(2 * np.pi * f_df['dow'] / 7.0)
    f_df['dow_cos'] = np.cos(2 * np.pi * f_df['dow'] / 7.0)
    f_df['month_sin'] = np.sin(2 * np.pi * f_df['month'] / 12.0)
    f_df['month_cos'] = np.cos(2 * np.pi * f_df['month'] / 12.0)
    
    # Is peak hour?
    f_df['is_peak'] = ((f_df['hour'] >= 6) & (f_df['hour'] <= 9) | 
                       (f_df['hour'] >= 17) & (f_df['hour'] <= 20)).astype(int)
    
    # Is night?
    f_df['is_night'] = ((f_df['hour'] < 6) | (f_df['hour'] > 21)).astype(int)
    
    # Stand prefix (very important!)
    if 'STAND_mvt' in df.columns:
        f_df['stand_prefix'] = df['STAND_mvt'].dropna().astype(str).str[:2]
    else:
        f_df['stand_prefix'] = 'XX'
    
    # Categorical columns for target encoding
    for c in ['ADEP_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt', 'WK_TBL_CAT_flt', 'AIRCRAFT_OPERATOR_flt']:
        if c in df.columns:
            f_df[c] = df[c].astype(str)
    
    return f_df

X_train = engineer_features(train, is_train=True)
X_test = engineer_features(test_dep, is_train=False)

# ============================================================================
# 3. TARGET ENGINEERING - DELTA FORMULATION
# ============================================================================
print("\n[3/6] Computing delta targets...")

# For training data: delta = diff_aobt - TAXITIME
# Because: TAXITIME = MVT - BLOCK, diff_aobt = MVT - AOBT
# So: delta = BLOCK - AOBT = diff_aobt - TAXITIME
train_with_feat = train.copy()
train_with_feat['diff_aobt'] = (train_with_feat['MVT_TIME_UTC_mvt'] - train_with_feat['AOBT_3_flt']).dt.total_seconds()

# Calculate delta
y_delta = train_with_feat['diff_aobt'] - train_with_feat['TAXITIME_SEC_mvt']

# Filter to valid delta range for training
valid_mask = (
    (train_with_feat['AOBT_3_flt'].notna()) & 
    (y_delta >= -300) & (y_delta <= 1200) &
    (train_with_feat['TAXITIME_SEC_mvt'] > 0) &
    (train_with_feat['TAXITIME_SEC_mvt'] < 10000)
)
print(f"Valid delta samples: {valid_mask.sum():,} / {len(train):,}")

# Also keep direct taxi time target for ensemble
y_taxi = train_with_feat['TAXITIME_SEC_mvt'].clip(upper=5000)  # Cap extreme outliers

# ============================================================================
# 4. TARGET ENCODING WITH K-FOLD (Prevent Leakage!)
# ============================================================================
print("\n[4/6] Building target-encoded features (5-fold OOF)...")

cat_cols_for_te = ['ADEP_mvt', 'RUNWAY_mvt', 'stand_prefix', 'AIRCRAFT_TYPE_mvt']

# Create combined features
X_train['adeP_rwy'] = X_train['ADEP_mvt'] + '_' + X_train['RUNWAY_mvt']
X_train['adeP_stand'] = X_train['ADEP_mvt'] + '_' + X_train['stand_prefix']
X_test['adeP_rwy'] = X_test['ADEP_mvt'] + '_' + X_test['RUNWAY_mvt']
X_test['adeP_stand'] = X_test['ADEP_mvt'] + '_' + X_test['stand_prefix']

cat_cols_for_te.extend(['adeP_rwy', 'adeP_stand'])

def target_encode_oof(train_df, test_df, col, target, n_splits=5):
    """Out-of-fold target encoding to prevent leakage."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    train_encoded = np.zeros(len(train_df))
    test_encoded = np.zeros(len(test_df))
    
    global_mean = target.mean()
    smoothing = 100  # Larger smoothing for stability
    
    for fold_idx, (trn_idx, val_idx) in enumerate(kf.split(train_df)):
        # Train on fold, encode validation
        trn_target = target.iloc[trn_idx]
        trn_col = train_df[col].iloc[trn_idx]
        
        # Compute mean and count for each category
        agg = pd.DataFrame({
            'mean': trn_target.groupby(trn_col).mean(),
            'count': trn_target.groupby(trn_col).count()
        })
        
        # Smoothed encoding
        smoothed = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
        
        # Encode validation set
        val_col = train_df[col].iloc[val_idx]
        train_encoded[val_idx] = val_col.map(smoothed).fillna(global_mean)
        
        # Accumulate test encoding
        test_col = test_df[col]
        test_encoded += test_col.map(smoothed).fillna(global_mean) / n_splits
    
    # Full training encoding for final model
    agg_full = target.groupby(train_df[col]).agg(['mean', 'count'])
    smoothed_full = (agg_full['count'] * agg_full['mean'] + smoothing * global_mean) / (agg_full['count'] + smoothing)
    
    # Override test with full encoding (more stable)
    test_encoded = test_df[col].map(smoothed_full).fillna(global_mean).values
    
    return train_encoded, test_encoded

# Apply target encoding
te_features = []
for col in cat_cols_for_te:
    if col in X_train.columns and col in X_test.columns:
        tr_enc, te_enc = target_encode_oof(X_train, X_test, col, y_taxi, n_splits=5)
        X_train[f'{col}_te'] = tr_enc
        X_test[f'{col}_te'] = te_enc
        te_features.append(f'{col}_te')
        print(f"  Encoded {col}: {len(pd.Index(X_train[col].unique()).intersection(pd.Index(X_test[col].unique())))} common categories")

# ============================================================================
# 5. DEFINE FINAL FEATURE SETS
# ============================================================================
print("\n[5/6] Preparing feature sets...")

base_features = [
    'hour', 'minute', 'dow', 'month',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'is_peak', 'is_night',
    'has_aobt', 'has_eobt',
    'diff_aobt_clipped', 'diff_eobt', 'diff_lobt'
]

# Add target-encoded features
all_features = base_features + te_features

# Remove any missing columns
all_features = [f for f in all_features if f in X_train.columns and f in X_test.columns]
print(f"Using {len(all_features)} features: {all_features[:10]}...")

X_train_final = X_train[all_features].fillna(0)
X_test_final = X_test[all_features].fillna(0)

# ============================================================================
# 6. TRAIN MODELS
# ============================================================================
print("\n[6/6] Training models...")

# Model 1: Delta prediction (primary)
print("\n  🎯 Model 1: Delta formulation (BLOCK - AOBT)")
delta_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 200,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'max_depth': 8,
    'verbose': -1,
    'seed': 42,
    'force_col_wise': True,
    'n_jobs': -1
}

# Use smaller sample for training delta model
delta_train_idx = valid_mask[valid_mask].index
if len(delta_train_idx) > 200000:
    import numpy as np
    delta_train_idx = np.random.RandomState(42).choice(delta_train_idx, 200000, replace=False)

delta_model = lgb.LGBMRegressor(**delta_params, n_estimators=800)
delta_model.fit(
    X_train_final.loc[delta_train_idx], 
    y_delta.loc[delta_train_idx],
    eval_set=[(X_train_final.loc[delta_train_idx], y_delta.loc[delta_train_idx])],
    callbacks=[lgb.early_stopping(100, verbose=False)]
)

pred_delta = delta_model.predict(X_test_final)
# Bound delta predictions to realistic range [-3min, +12min]
pred_delta = np.clip(pred_delta, -180, 720)
print(f"    Delta predictions: mean={pred_delta.mean():.1f}, std={pred_delta.std():.1f}")

# Model 2: Direct taxi time prediction (ensemble backup)
print("\n  🎯 Model 2: Direct TAXITIME prediction")
taxi_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 150,
    'reg_alpha': 0.3,
    'reg_lambda': 0.3,
    'max_depth': 8,
    'verbose': -1,
    'seed': 42,
    'force_col_wise': True,
    'n_jobs': -1
}

# Sample for taxi model too
if len(X_train_final) > 250000:
    taxi_idx = np.random.RandomState(42).choice(len(X_train_final), 250000, replace=False)
else:
    taxi_idx = slice(None)

taxi_model = lgb.LGBMRegressor(**taxi_params, n_estimators=600)
taxi_model.fit(
    X_train_final.iloc[taxi_idx], y_taxi.iloc[taxi_idx] if hasattr(y_taxi, 'iloc') else y_taxi[taxi_idx],
    eval_set=[(X_train_final.iloc[taxi_idx], y_taxi.iloc[taxi_idx] if hasattr(y_taxi, 'iloc') else y_taxi[taxi_idx])],
    callbacks=[lgb.early_stopping(100, verbose=False)]
)

pred_taxi_direct = taxi_model.predict(X_test_final)
print(f"    Direct predictions: mean={pred_taxi_direct.mean():.1f}, std={pred_taxi_direct.std():.1f}")

# ============================================================================
# 7. GENERATE FINAL PREDICTIONS
# ============================================================================
print("\n[7/7] Generating final predictions...")

# Method 1: Delta formulation (for flights with AOBT)
test_diff_aobt = X_test['diff_aobt'].values
test_has_aobt = X_test['has_aobt'].values == 1

final_preds = np.zeros(len(X_test))

# Rule 1: Flights with valid AOBT - use delta formulation (98%+ of flights)
valid_aobt_mask = test_has_aobt & (test_diff_aobt > 0) & (test_diff_aobt < 7000)
final_preds[valid_aobt_mask] = test_diff_aobt[valid_aobt_mask] - pred_delta[valid_aobt_mask]
print(f"  Delta method: {valid_aobt_mask.sum():,} flights")

# Rule 2: Extreme outliers in ground truth (multi-hour taxi times)
extreme_mask = test_has_aobt & (test_diff_aobt >= 7000.0)
final_preds[extreme_mask] = test_diff_aobt[extreme_mask] - 60.0
print(f"  Extreme outliers: {extreme_mask.sum():,} flights")

# Rule 3: Missing or invalid AOBT - blend of direct prediction and fallback (~2% of flights)
invalid_aobt_mask = (~test_has_aobt) | (test_diff_aobt <= 0.0)
# Use weighted blend: 70% direct model, 30% airport median fallback
fallback = 900  # Global median
final_preds[invalid_aobt_mask] = 0.7 * pred_taxi_direct[invalid_aobt_mask] + 0.3 * fallback
print(f"  Fallback method: {invalid_aobt_mask.sum():,} flights")

# Final clipping - preserve realistic range but allow some extremes
final_preds = np.clip(final_preds, 60.0, 90000.0)

# Map to submission format
submission_map = pd.Series(final_preds, index=test_dep.index)
sub_out = test_submit_dep[['MVT_ID_mvt']].copy()
sub_out['TAXITIME_SEC_mvt'] = sub_out['MVT_ID_mvt'].map(
    pd.Series(final_preds, index=test_dep['MVT_ID_mvt'].values)
)

# Fill any remaining nulls
if sub_out['TAXITIME_SEC_mvt'].isna().any():
    sub_out['TAXITIME_SEC_mvt'] = sub_out['TAXITIME_SEC_mvt'].fillna(900)

# ============================================================================
# 8. SAVE SUBMISSION
# ============================================================================
out_path = "outputs/merry-iguana_v8.parquet"
sub_out.to_parquet(out_path, index=False)

print("\n" + "=" * 80)
print(f"✅ SUBMISSION SAVED: {out_path}")
print(f"   Rows: {len(sub_out):,}")
print(f"   Mean: {sub_out['TAXITIME_SEC_mvt'].mean():.1f}s")
print(f"   Median: {sub_out['TAXITIME_SEC_mvt'].median():.1f}s")
print(f"   Std: {sub_out['TAXITIME_SEC_mvt'].std():.1f}s")
print(f"   Min: {sub_out['TAXITIME_SEC_mvt'].min():.1f}s")
print(f"   Max: {sub_out['TAXITIME_SEC_mvt'].max():.1f}s")
print("=" * 80)
print("\n🎯 Expected improvement: Delta formulation should reduce RMSE by 50-100 seconds!")
print("   Current best: 532.5s → Target: <450s")
