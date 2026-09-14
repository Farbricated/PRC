import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
import glob
import warnings
warnings.filterwarnings('ignore')

print("🚀 merry-iguana v8: Target Encoding & Direct Prediction (Memory Optimized)")

# 1. Load Data - Sample to fit in memory
print("Loading data with sampling...")
train_files = sorted(glob.glob('data/raw/training_*.parquet'))

# Read and sample each file to stay within memory
train_list = []
for f in train_files:
    df = pd.read_parquet(f)
    df_dep = df[df['Type'] == 'D']
    # Sample 40% from each month to maintain temporal distribution
    if len(df_dep) > 50000:
        df_dep = df_dep.sample(frac=0.4, random_state=42)
    train_list.append(df_dep)

train = pd.concat(train_list, ignore_index=True)
test = pd.read_parquet('data/raw/submitting.parquet')
test = test[test['Type'] == 'D'].copy()

print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# 2. Target Engineering
target_cap = train['TAXITIME'].quantile(0.99)
print(f"Clipping target at 99th percentile: {target_cap:.2f}s")
train['TAXITIME_CLIP'] = train['TAXITIME'].clip(upper=target_cap)

# 3. Feature Engineering
def engineer_features(df):
    df = df.copy()
    df['SCHED_TIME'] = pd.to_datetime(df['SCHED_TIME'])
    df['Hour'] = df['SCHED_TIME'].dt.hour
    df['DayOfWeek'] = df['SCHED_TIME'].dt.dayofweek
    df['Month'] = df['SCHED_TIME'].dt.month
    df['MinuteOfDay'] = df['Hour'] * 60 + df['SCHED_TIME'].dt.minute
    
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['DOW_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DOW_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    
    df['Stand_Rwy'] = df['DEP_STAND'].astype(str) + '_' + df['RwyConfig'].astype(str)
    df['Airline_Stand'] = df['Airline'].astype(str) + '_' + df['DEP_STAND'].astype(str)
    
    return df

train = engineer_features(train)
test = engineer_features(test)

# 4. Target Encoding
cat_cols = ['DEP_STAND', 'RwyConfig', 'Airline', 'Stand_Rwy', 'Airline_Stand']

for col in cat_cols:
    global_mean = train['TAXITIME_CLIP'].mean()
    agg = train.groupby(col)['TAXITIME_CLIP'].agg(['mean', 'count']).reset_index()
    agg.columns = [col, f'{col}_mean', f'{col}_count']
    
    smoothing = 20
    agg[f'{col}_te'] = (agg[f'{col}_count'] * agg[f'{col}_mean'] + smoothing * global_mean) / (agg[f'{col}_count'] + smoothing)
    
    map_dict = agg.set_index(col)[f'{col}_te'].to_dict()
    train[f'{col}_te'] = train[col].map(map_dict).fillna(global_mean)
    test[f'{col}_te'] = test[col].map(map_dict).fillna(global_mean)
    
    print(f"Encoded {col}: {len(map_dict)} categories")

# 5. Define Features
feature_cols = [
    'Hour', 'DayOfWeek', 'Month', 'MinuteOfDay',
    'Hour_sin', 'Hour_cos', 'DOW_sin', 'DOW_cos',
    'DEP_STAND_te', 'RwyConfig_te', 'Airline_te', 
    'Stand_Rwy_te', 'Airline_Stand_te'
]

X_train = train[feature_cols]
y_train = train['TAXITIME_CLIP']
X_test = test[feature_cols]

# Clean up memory
del train
import gc
gc.collect()

# 6. Model Training
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 50,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'verbose': -1,
    'seed': 42,
    'force_col_wise': True
}

print("Training LightGBM with 5-Fold CV...")
oof_pred = np.zeros(len(X_train))
test_pred = np.zeros(len(X_test))

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]
    
    model = lgb.LGBMRegressor(**params, n_estimators=1000)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    
    oof_pred[val_idx] = model.predict(X_val)
    test_pred += model.predict(X_test) / 5
    
    fold_rmse = np.sqrt(np.mean((oof_pred[val_idx] - y_val)**2))
    print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")

cv_rmse = np.sqrt(np.mean((oof_pred - y_train)**2))
print(f"\n✅ Final CV RMSE: {cv_rmse:.4f}")

# 7. Post-Processing
test_pred = np.maximum(test_pred, 0)

# Create Submission
submission = test[['UUID']].copy()
submission['TAXITIME'] = test_pred
submission.to_parquet('outputs/merry-iguana_v8.parquet', index=False)

print(f"\n💾 Submission saved: outputs/merry-iguana_v8.parquet")
print(f"Rows: {len(submission)}, Mean: {submission['TAXITIME'].mean():.2f}, Median: {submission['TAXITIME'].median():.2f}")
print("Ready for upload!")
