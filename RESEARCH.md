# Comprehensive PRC Research — This Project & 24/25 Winners

## What is PRC
- PRC: Performance Review Commission (EUROCONTROL)
- Challenge: open aviation data, reproducible ML, JOAS publication encouraged for top teams
- Data papers published per edition (e.g., Aircraft Fuel Burn / Takeoff Weight)
- Team repos: github.com/prc-data-challenge-2024, github.com/prc-data-challenge-2025

## Previous Editions (2024 / 2025)
- 2024: Aircraft Takeoff Weight Estimation (data paper + repos)
- 2025: Aircraft Fuel Burn Estimation (data paper + repos)
- Pattern: open-source models, physics/domain-guided gradient boosting, reproducible validation splits, per-group error analysis
- Winners submit JOAS papers describing inner workings, feature importance, validation design

## This Dataset (Repo Evidence)
- 14 monthly parquet (2025-01 to 2026-01): ~4.17M total rows, 30 cols each
- Schema: MVT_ID, FLIGHT_ID, ADEP/ADES, PHASE (DEP/ARR), MVT_TIME_UTC, BLOCK_TIME_UTC, SCHED_TIME_UTC, AIRCRAFT_TYPE, RUNWAY, STAND, TAXITIME_SEC (target), + 14 `_flt` delay/features
- Submitting: 344,841 rows, 2 cols (ID + target)
- Ranking: 689,534 rows, 30 cols (full feature evaluation set)
- Null TAXITIME: only 1 in Nov; negative taxi: ~36/mo (drop); zero: ~2/mo (keep)
- Airports: 10 present; spec mentions 11th missing — known gap

## Work Completed (PROGRESS.md + Code)
- Ingest: 14 .parquet moved to data/raw/; schema matches dictionary
- Clean: drop TAXITIME_SEC < 0; balanced DEP/ARR; 2,541 nulls in `_flt` (expected unmatched flights)
- Features (v1): hour, dow, month, schedule_delta, adep_code
- Features (cascading): origin_congestion_rolling, pair_flow_3h
- Model v1: LightGBM, RMSE ~700 sec (Jan time split)
- Model cascading: RMSE ~366 sec (same split) — ~48% improvement
- Submission v1: outputs/merry-iguana_v1.parquet (2 cols, 344,841 rows, no null predictions)
- Broken: ranking predict fails (empty feature set on ranking file) — needs fix

## What Winners Do (From 24/25 Repos + JOAS)
1. **Reproducible pipeline** — ingest, clean, feature, train, predict, submit all in one script/repo
2. **Domain-informed features** — physics or congestion dynamics, not just raw tabular dump
3. **Strong validation** — time-based + airport-stratified; per-airport, per-hour RMSE tracking
4. **Ensemble + smoothing** — base GBM + residual correction + rolling temporal smoothing
5. **Clean submittable format** — exact ID match, no nulls, outlier-clipped predictions, versioned filename
6. **Open documentation** — approach paper (JOAS-style) with feature descriptions, error analysis, data paper reference

## Strategic Plan to Win / Submittable / Documented
1. Fix ranking/predict pipeline (ensure feature columns exist for ranking)
2. Train ensemble on all 14 months (not only Jan) with cascading features
3. Add domain features: runway congestion proxy, aircraft-type delay profile, time sinusoidal, delay propagation from `_flt` fields
4. Ensemble: LightGBM + XGBoost + Ridge on residuals; weighted blend
5. Validation: airport-stratified + temporal CV; track RMSE by airport/hour
6. Post-process: rolling median by airport-hour + outlier clip (0–7200 sec)
7. Final output: `outputs/merry-iguana_final.parquet` (2 col, exact IDs, no nulls)
8. Documentation: `OUTREACH.md` (already updated with PRC context + repo references) + JOAS-style approach summary

## Key Files in This Repo
- `PROGRESS.md`: timeline of work completed
- `src/features.py`: cascading congestion feature engineering
- `src/model.py`: training, validation, prediction pipeline
- `OUTREACH.md`: PRC context, team repos, data papers, winning strategy
- `outputs/merry-iguana_v1.parquet`: v1 submittable (valid format)
- `data/raw/*.parquet`: 14 training + ranking + submitting

=== CURRENT RESULTS (11 Sep 2026) ===
v2 RMSE 688.5 (median) -> v3 656.6 (real) -> v4 validation 460.4 (hidden ~656). Rank 114/115. Gap: overfit to Jan patterns; need v5 with regularization + July CV + ensemble.
HANDOFF.md + UPGRADES.md + PROGRESS.md updated. Antigravity prompt ready.
