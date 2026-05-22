# 02_models/xgboost/train_loyo.py
"""
XGBoost LOYO (Leave-One-Year-Out) cross-validation.
For each year Y in training period (1950–2014):
  train on all years ≠ Y → predict year Y
Output: results/xgboost_loyo_metrics.csv + xgboost_loyo_predictions.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
from config import TABULAR_CSV, SELECTED_VARS_JSON, RESULT_DIR, XGB_PARAMS, \
                   TRAIN_YEAR_MIN, TRAIN_YEAR_MAX
import xgboost as xgb

print("=== XGBoost LOYO ===")

df = pd.read_csv(TABULAR_CSV)
df_train = df[(df['annual_balance_m'].notna()) &
              (df['year'] >= TRAIN_YEAR_MIN) &
              (df['year'] <= TRAIN_YEAR_MAX)].copy()

with open(SELECTED_VARS_JSON) as f:
    sel = json.load(f)
feature_cols = sel['selected_features']

X_all = df_train[feature_cols].fillna(df_train[feature_cols].median()).values
y_all = df_train['annual_balance_m'].values
years_all = df_train['year'].values
gids_all = df_train['glacier_id'].values

fold_years = sorted(df_train['year'].unique())
print(f"Samples: {len(y_all)}, Folds: {len(fold_years)}")

preds, obs, fold_r2 = [], [], []
for yr in fold_years:
    tr = years_all != yr
    te = years_all == yr
    if te.sum() == 0:
        continue
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_all[tr], y_all[tr])
    p = model.predict(X_all[te])
    r2 = r2_score(y_all[te], p)
    fold_r2.append({'year': yr, 'n': int(te.sum()), 'r2': r2,
                    'rmse': float(np.sqrt(mean_squared_error(y_all[te], p)))})
    preds.extend(p.tolist())
    obs.extend(y_all[te].tolist())

r2_global = r2_score(obs, preds)
pearson_r, _ = pearsonr(obs, preds)
rmse_global = np.sqrt(mean_squared_error(obs, preds))
bias = np.mean(np.array(preds) - np.array(obs))
print(f"LOYO R2={r2_global:.4f}  R={pearson_r:.4f}  RMSE={rmse_global*1000:.1f}mm  Bias={bias*1000:.1f}mm")

os.makedirs(RESULT_DIR, exist_ok=True)
pd.DataFrame(fold_r2).to_csv(os.path.join(RESULT_DIR, 'xgboost_loyo_metrics.csv'), index=False)
pd.DataFrame({'obs': obs, 'pred': preds}).to_csv(
    os.path.join(RESULT_DIR, 'xgboost_loyo_predictions.csv'), index=False)
pd.DataFrame([{'model': 'xgboost', 'cv': 'LOYO',
               'r2': r2_global, 'pearson_r': pearson_r,
               'rmse_mm': rmse_global*1000, 'bias_mm': bias*1000}]).to_csv(
    os.path.join(RESULT_DIR, 'xgboost_loyo_summary.csv'), index=False)
print(f"Saved → {RESULT_DIR}/xgboost_loyo_*.csv")
