# Upgrade Plan — v4 to v5 (Comprehensive)

Current: RMSE ~656 (v3, real predictions, 1.88MiB, rank 114/115)
Target: <350 to reach mid-tier; <300 competitive; ~245 to win

## 1. Data / Feature Upgrades
- Add sinusoidal hour/dow (capture cyclic patterns)
- Add aircraft-type delay profile (group mean taxi by AIRCRAFT_TYPE_mvt)
- Add runway congestion (count of same RUNWAY_mvt in 1h window)
- Add delay propagation from _flt fields (LOBT_flt, IOBT_flt, ARVT_3_flt offset vs MVT)
- Add holiday / weekday flag from month/dot
- Use all 14 files fully, not just January split for validation

## 2. Model / Hyperparam
- LightGBM: n_estimators 500-1000, learning_rate 0.03-0.05, num_leaves 31-63, max_depth -1
- Add XGBoost base + Ridge on residuals (ensemble)
- Time-based CV with airport-stratified split
- Track RMSE by airport/hour to find weak spots

## 3. Post-process
- Per-airport rolling median (not just global)
- Outlier clip per airport-percentile (not fixed 7200)
- Residual correction (model error by hour)

## 4. Validation / Quality
- Ensure ranking predictions use correct feature build (fixed in v3)
- Verify every submit ID mapped; no nulls
- Check submission file naming (v4, v5...)

## 5. Documentation / Open Science
- Update RESEARCH.md with v3 RMSE / improvement note
- Update PROGRESS.md with v3 upload and next iteration
- JOAS-style writeup: feature importance, validation design, error analysis
