# HANDOFF / ANTIGRAVITY PROMPT — Continue Work

## Project
PRC 2026 — Team merry-iguana (competitionId bb3693e1-26bc-4a9e-8619-4fe78b4eab0c)
Goal: predict TAXITIME_SEC_mvt (taxi-out time) for DEP flights at European airports.

## Current Status (as of 11 Sep 2026, 17:51 UTC)
- **Current Rank:** 91 / 116 (jumped 23 places from 114)
- **Best Score (RMSE):** 532.5355 sec (down from 656.5970 sec, -124.06 sec improvement!)
- **Submissions Today:** 5 / 5 (v2, v3, v4, v5, v6) — Daily submission quota reached for today!
- **Target for Tomorrow's Submissions:** < 270 sec to enter Top 10; ~245 sec to challenge #1 (youthful-giraffe: 245.0207).

## What Was Discovered & Fixed Today
1. **Critical Data Structure Flaw Identified:**
   - Previous models used `(BLOCK_TIME_UTC_mvt - SCHED_TIME_UTC_mvt)`. In `ranking.parquet` for DEP flights, `BLOCK_TIME_UTC_mvt` is 100% `NaT`. All delay features collapsed to 0 at test time!
   - `adep_code` used arbitrary category codes per monthly file, scrambling airport representations between training and test sets.
2. **Surveillance & Flight Plan Breakthrough:**
   - `TAXITIME_SEC_mvt = MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt` exactly to the second.
   - For 98.47% of test flights, `AOBT_3_flt` (Actual Off-Block Time) is available.
   - `diff_aobt = MVT_TIME_UTC_mvt - AOBT_3_flt = TAXITIME + (BLOCK_TIME - AOBT_3)`.
   - The offset `delta = BLOCK_TIME - AOBT_3` has a median of +52s and standard deviation ~370s.
3. **Outlier Mechanism Uncovered:**
   - In ground truth, a few flights have multi-hour or 24-hour taxi times (e.g. Rome LIRF July 17 ground truth `TAXITIME = 86,340s` because `BLOCK_TIME` was recorded on the previous date).
   - In v5, clipping to 5400s caused `(86340 - 5400)^2` which added 19,249 to the test set MSE (~138 seconds to RMSE).
   - In v6, extreme flights (`diff_aobt >= 7000s`) were preserved (`diff_aobt - 60s`), dropping the score to 532.5355.

## Architecture & Codebase Upgrades
- `src/features.py`: Full feature engineering with unified categorical dtypes across train and test, stand prefix extraction, cyclical sinusoids, and flight-plan timing deltas.
- `src/model.py`: Production pipeline trained on all 2.08 million departures across 2025 with bounded predictions and physical outlier handling.
- `outputs/merry-iguana_v6.parquet`: Best submission (532.5355 score, rank 91).
- `outputs/merry-iguana_final.parquet`: Generated submission ready for the next iteration.

## Next High-Impact Steps (for Day 2 / Tomorrow)
1. **Delta Formulation Model:**
   - Train an ensemble predicting `delta = BLOCK_TIME - AOBT_3` directly instead of predicting `TAXITIME`.
   - Reconstruct `TAXITIME = diff_aobt - pred_delta`.
   - Bound `pred_delta` to `[-300, +500]`, preventing the trees from drifting away from the surveillance time.
2. **Surface Congestion Dynamics:**
   - Use arrival movements from `ranking.parquet` (all ARR flights have exact `MVT_TIME_UTC_mvt`).
   - Compute rolling counts of active departures and arrivals in the 15-minute and 30-minute windows prior to each takeoff.
3. **Stand-to-Runway Target Encoding:**
   - Build out-of-fold smoothed target encodings for `(ADEP_mvt, RUNWAY_mvt)` and `(ADEP_mvt, stand_prefix)`.
4. **Ensemble Blend:**
   - Weighted blend of LightGBM + CatBoost on `delta` + Direct model.
   - Target CV RMSE: ~220 - 240 seconds.

## Commands Reference
- Run pipeline: `python src/model.py`
- Check leaderboard: `python scratch/check_lb.py`
- Upload submission: `"C:/Users/akars/mc.exe" cp outputs/<file>.parquet prc/prc-2026-merry-iguana/`
- Check bucket: `"C:/Users/akars/mc.exe" ls prc/prc-2026-merry-iguana/`

