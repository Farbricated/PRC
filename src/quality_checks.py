"""Stage 1 — Data quality checks across all 12 monthly files."""
import pandas as pd, glob, sys
files = sorted(glob.glob("PRC/data/raw/training_*.parquet"))
print(f"Files found: {len(files)}")
for f in files:
    df = pd.read_parquet(f)
    dep = df[df.PHASE_mvt == "DEP"]
    arr = df[df.PHASE_mvt == "ARR"]
    neg = (dep.TAXITIME_SEC_mvt < 0).sum()
    zero = (dep.TAXITIME_SEC_mvt == 0).sum()
    null_taxi = dep.TAXITIME_SEC_mvt.isnull().sum()
    airports = sorted(dep.ADEP_mvt.dropna().unique())
    print(f"{f.split('/')[-1]}: rows={len(df)} DEP={len(dep)} ARR={len(arr)} | taxi<0={neg} taxi==0={zero} taxi_null={null_taxi} | ADEP_codes({len(airports)}): {airports}")
