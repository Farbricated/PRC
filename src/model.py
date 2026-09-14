"""PRC 2026 Production Training and Prediction Pipeline."""
import glob, os, time, pandas as pd, numpy as np, lightgbm as lgb
from sklearn.metrics import root_mean_squared_error
from src.features import fit_categorical_types, build_features

def main():
    t0 = time.time()
    print("Loading test/submitting files...", flush=True)
    rank = pd.read_parquet("data/raw/ranking.parquet")
    submit = pd.read_parquet("data/raw/submitting.parquet")
    dep_rank = rank[rank['MVT_ID_mvt'].isin(submit['MVT_ID_mvt'])].copy()

    print("Loading all 12 training parquet files...", flush=True)
    train_files = sorted(glob.glob("data/raw/training_*.parquet"))
    train_dfs = []
    for f in train_files:
        df = pd.read_parquet(f)
        dep = df[df.PHASE_mvt == 'DEP'].copy()
        dep = dep[dep['TAXITIME_SEC_mvt'].notna() & (dep['TAXITIME_SEC_mvt'] > 0)].copy()
        train_dfs.append(dep)
    full_train = pd.concat(train_dfs, ignore_index=True)
    print(f"Loaded {len(full_train):,} training departures in {time.time()-t0:.1f}s", flush=True)

    # Build unified categorical types
    cat_types = fit_categorical_types(full_train, dep_rank)

    print("Building features for train and test...", flush=True)
    X_train = build_features(full_train, cat_types)
    y_train = full_train['TAXITIME_SEC_mvt'].values
    X_test = build_features(dep_rank, cat_types)

    # Train production model on normal taxi times
    train_norm_mask = y_train < 7200
    print(f"Training LightGBM on {train_norm_mask.sum():,} flights...", flush=True)
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.04,
        num_leaves=127,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train[train_norm_mask], y_train[train_norm_mask])

    print("Generating predictions...", flush=True)
    base_preds = model.predict(X_test)
    final_preds = base_preds.copy()

    test_diff_aobt = X_test['diff_aobt'].values
    test_has_aobt = X_test['has_aobt'].values == 1

    # Rule 1: Flights with diff_aobt >= 7000 (multi-hour/24-hour outliers in ground truth)
    extreme_mask = test_has_aobt & (test_diff_aobt >= 7000.0)
    final_preds[extreme_mask] = test_diff_aobt[extreme_mask] - 60.0

    # Rule 2: Flights with missing or invalid AOBT
    invalid_aobt_mask = (~test_has_aobt) | (test_diff_aobt <= 0.0)
    final_preds[invalid_aobt_mask] = np.clip(final_preds[invalid_aobt_mask], 300.0, 2800.0)

    # Rule 3: Normal flights with valid AOBT
    normal_aobt_mask = test_has_aobt & (test_diff_aobt > 0.0) & (test_diff_aobt < 7000.0)
    lower_bound = np.maximum(60.0, test_diff_aobt[normal_aobt_mask] - 600.0)
    upper_bound = np.minimum(5400.0, test_diff_aobt[normal_aobt_mask] + 1200.0)
    final_preds[normal_aobt_mask] = np.clip(final_preds[normal_aobt_mask], lower_bound, upper_bound)

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

if __name__ == "__main__":
    main()
