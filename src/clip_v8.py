"""Clip v8 outliers to physical max 5400s — v9 submission."""
import pandas as pd, numpy as np

df = pd.read_parquet("outputs/merry-iguana_v8.parquet")
print(f"Before: min={df['TAXITIME_SEC_mvt'].min():.1f}, max={df['TAXITIME_SEC_mvt'].max():.1f}, mean={df['TAXITIME_SEC_mvt'].mean():.1f}")

df['TAXITIME_SEC_mvt'] = df['TAXITIME_SEC_mvt'].clip(60.0, 5400.0)
print(f"After:  min={df['TAXITIME_SEC_mvt'].min():.1f}, max={df['TAXITIME_SEC_mvt'].max():.1f}, mean={df['TAXITIME_SEC_mvt'].mean():.1f}")

df.to_parquet("outputs/merry-iguana_v9.parquet", index=False)
print("Saved outputs/merry-iguana_v9.parquet")