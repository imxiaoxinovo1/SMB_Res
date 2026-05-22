# 03_evaluation/eval_holdout.py
"""
Hold-out evaluation for XGBoost.
仅在 HOLDOUT_YEAR_MIN / HOLDOUT_YEAR_MAX 非 None 时运行。
当前配置已将全部观测并入训练集（TRAIN_YEAR_MAX=2023），本脚本将跳过。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from config import (TABULAR_CSV, SELECTED_VARS_JSON, RESULT_DIR,
                    TRAIN_YEAR_MIN, TRAIN_YEAR_MAX,
                    HOLDOUT_YEAR_MIN, HOLDOUT_YEAR_MAX, XGB_PARAMS)
import xgboost as xgb

if HOLDOUT_YEAR_MIN is None or HOLDOUT_YEAR_MAX is None:
    print("Hold-out 集已禁用（HOLDOUT_YEAR_MIN/MAX=None）。跳过。")
    sys.exit(0)

print(f"=== Hold-out Evaluation ({HOLDOUT_YEAR_MIN}–{HOLDOUT_YEAR_MAX}) ===")

df = pd.read_csv(TABULAR_CSV)
df_labeled = df[df['annual_balance_m'].notna()].copy()

with open(SELECTED_VARS_JSON) as f:
    feature_cols = json.load(f)['selected_features']

X_all = df_labeled[feature_cols].fillna(df_labeled[feature_cols].median()).values
y_all = df_labeled['annual_balance_m'].values
years = df_labeled['year'].values

tr_mask = (years >= TRAIN_YEAR_MIN) & (years <= TRAIN_YEAR_MAX)
te_mask = (years >= HOLDOUT_YEAR_MIN) & (years <= HOLDOUT_YEAR_MAX)

print(f"Train: {tr_mask.sum()} samples | Hold-out: {te_mask.sum()} samples")

model = xgb.XGBRegressor(**XGB_PARAMS)
model.fit(X_all[tr_mask], y_all[tr_mask])
preds = model.predict(X_all[te_mask])
obs   = y_all[te_mask]

r2   = r2_score(obs, preds)
rmse = np.sqrt(mean_squared_error(obs, preds))
bias = np.mean(preds - obs)

print(f"XGBoost Hold-out R²={r2:.4f}  RMSE={rmse*1000:.1f}mm  Bias={bias*1000:.1f}mm")
os.makedirs(RESULT_DIR, exist_ok=True)
pd.DataFrame([{'model': 'xgboost', 'cv': 'holdout',
               'r2': r2, 'rmse_mm': rmse*1000, 'bias_mm': bias*1000,
               'n': int(te_mask.sum())}]).to_csv(
    os.path.join(RESULT_DIR, 'holdout_metrics.csv'), index=False)

# Save scatter data for fig1
pd.DataFrame({'obs': obs, 'pred': preds}).to_csv(
    os.path.join(RESULT_DIR, 'xgboost_holdout_predictions.csv'), index=False)
print(f"Saved → {RESULT_DIR}/holdout_metrics.csv")
