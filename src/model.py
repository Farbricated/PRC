"""PRC 2026 Production Training and Prediction Pipeline - Delta Formulation v2."""
import glob, os, time, pandas as pd, numpy as np, lightgbm as lgb
from sklearn.metrics import root_mean_squared_error
from src.features import fit_categorical_types, build_features

def main():
    t0 = time.time()
    print("Loading test/submitting files...", flush=True)
    rank = pd.read_parquet("data/raw/ranking.parquet")
    submit = pd.read_parquet("data/raw/submitting.parquet")
    dep_rank = rank[rank['MVT_ID_mvt'].isin(submit['MVT_ID_mvt'])].copy()

    print("Loading training data with sampling for memory efficiency...", flush=True)
    train_files = sorted(glob.glob("data/raw/training_*.parquet"))
    
    # Only keep necessary columns (include all flight plan times for feature engineering)
    needed_cols = ['ADEP_mvt', 'RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 
                   'WK_TBL_CAT_flt', 'AIRCRAFT_OPERATOR_flt', 'MVT_TIME_UTC_mvt',
                   'BLOCK_TIME_UTC_mvt', 'AOBT_3_flt', 'TAXITIME_SEC_mvt', 'PHASE_mvt',
                   'EOBT_1_flt', 'LOBT_flt', 'IOBT_flt', 'SCHED_TIME_UTC_mvt']
    
    # Sample training data to fit in memory (use ~35% of data for memory efficiency)
    train_dfs = []
    for i, f in enumerate(train_files):
        df = pd.read_parquet(f, columns=needed_cols)
        dep = df[df.PHASE_mvt == 'DEP'].copy()
        dep = dep[dep['TAXITIME_SEC_mvt'].notna() & (dep['TAXITIME_SEC_mvt'] > 0)].copy()
        # Sample 35% of each month's data
        sampled = dep.sample(frac=0.35, random_state=42+i)
        train_dfs.append(sampled)
        print(f"  Loaded {i+1}/12 files ({len(sampled):,} rows)...", flush=True)
    
    full_train = pd.concat(train_dfs, ignore_index=True)
    print(f"Training on {len(full_train):,} sampled departures in {time.time()-t0:.1f}s", flush=True)

    # Build unified categorical types
    cat_types = fit_categorical_types(full_train, dep_rank)

    print("Building features for train and test...", flush=True)
    X_train = build_features(full_train, cat_types)
    
    # Target: delta = BLOCK_TIME - AOBT_3 (the offset between surveillance off-block and actual block)
    y_train_delta = ((full_train['BLOCK_TIME_UTC_mvt'] - full_train['AOBT_3_flt']).dt.total_seconds()).values
    
    X_test = build_features(dep_rank, cat_types)

    # Filter to valid delta values for training
    has_aobt_train = X_train['has_aobt'].values == 1
    valid_delta_mask = (y_train_delta >= -300) & (y_train_delta <= 1200) & has_aobt_train
    print(f"Training LightGBM on {valid_delta_mask.sum():,} flights with valid delta...", flush=True)

    # Improved hyperparameters tuned for delta prediction
    model = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=150,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.2,
        max_depth=-1,
        random_state=42,
        n_jobs=-1,
        force_col_wise=True
    )
    model.fit(X_train[valid_delta_mask], y_train_delta[valid_delta_mask])

    print("Generating predictions...", flush=True)
    pred_delta = model.predict(X_test)

    # Bound delta predictions to realistic range [-3min, +12min]
    pred_delta = np.clip(pred_delta, -180, 720)

    # Calculate TAXITIME = diff_aobt - pred_delta
    test_diff_aobt = X_test['diff_aobt'].values
    test_has_aobt = X_test['has_aobt'].values == 1

    final_preds = np.zeros(len(X_test))

    # Rule 1: Flights with valid AOBT - use delta formulation (98%+ of flights)
    valid_aobt_mask = test_has_aobt & (test_diff_aobt > 0) & (test_diff_aobt < 7000)
    final_preds[valid_aobt_mask] = test_diff_aobt[valid_aobt_mask] - pred_delta[valid_aobt_mask]

    # Rule 2: Extreme outliers in ground truth (multi-hour taxi times) - rare cases
    extreme_mask = test_has_aobt & (test_diff_aobt >= 7000.0)
    final_preds[extreme_mask] = test_diff_aobt[extreme_mask] - 60.0

    # Rule 3: Missing or invalid AOBT - fall back to airport/time-based estimate (~2% of flights)
    invalid_aobt_mask = (~test_has_aobt) | (test_diff_aobt <= 0.0)
    # Use a reasonable fallback based on typical taxi times
    final_preds[invalid_aobt_mask] = np.clip(900 + pred_delta[invalid_aobt_mask] * 0.1, 400, 1800)

    final_preds = np.clip(final_preds, 60.0, 90000.0)
    dep_rank['pred_taxitime'] = final_preds

    # Map onto submitting.parquet
    pred_map = dep_rank.set_index('MVT_ID_mvt')['pred_taxitime'].to_dict()
    sub_out = submit[['MVT_ID_mvt']].copy()
    sub_out['TAXITIME_SEC_mvt'] = sub_out['MVT_ID_mvt'].map(pred_map)

    # Checks
    assert sub_out['TAXITIME_SEC_mvt'].isna().sum() == 0, "Null predictions found!"
    assert len(sub_out) == len(submit), "Row count mismatch!"
    assert (sub_out['MVT_ID_mvt'] == submit['MVT_ID_mvt']).all(), "ID mismatch!"

    out_path = "outputs/merry-iguana_final.parquet"
    sub_out.to_parquet(out_path, index=False)
    print(f"FINAL SUBMISSION SAVED: {out_path} ({len(sub_out):,} rows) in {time.time()-t0:.1f}s", flush=True)
    
    # Print summary stats
    print(f"\nPrediction statistics:")
    print(f"  Mean: {sub_out['TAXITIME_SEC_mvt'].mean():.1f}s")
    print(f"  Median: {sub_out['TAXITIME_SEC_mvt'].median():.1f}s")
    print(f"  Std: {sub_out['TAXITIME_SEC_mvt'].std():.1f}s")

if __name__ == "__main__":
    main()
