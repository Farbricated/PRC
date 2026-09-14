# PRC Outreach & Winning Strategy (EUROCONTROL PRC 2024/2025)

Source: PRC challenge docs — JOAS papers, data papers, team repos.

## Past Winners / Team Repos
- 2024 team repos: https://github.com/prc-data-challenge-2024
- 2025 team repos: https://github.com/prc-data-challenge-2025
- Data papers: EUROCONTROL PRC 2024 / 2025 Data Challenge (Aircraft Fuel / Takeoff Weight)
- JOAS submissions encouraged especially from top-ranked teams.

## This Project (Taxi-Time / Congestion)
- Dataset: 14 monthly parquet, ~4.17M rows, 30 cols, DOB/ARR balanced, 10 airports.
- Current best: v1 RMSE ~700 sec; cascading model RMSE ~366 sec (Jan split only).
- Submittable: outputs/merry-iguana_v1.parquet (344,841 rows, 2 cols, no nulls).

## Plan to Be Winnable + Submittable + Documentable
1. Train on all 14 files; airport-stratified CV.
2. Stack ensemble (LGB + XGB + residual) with cascading congestion features.
3. Post-process: rolling median + outlier clip.
4. Write approach as JOAS-style paper: open science, model description, feature importance, validation design.
5. Publish repo (reference 2025 org pattern) with reproducible pipeline.

## Acronyms
- JOAS: Journal of Open Aviation Science
- PRC: Performance Review Commission
- RMSE: Root Mean Square Error
- OSN: OpenSky Network
