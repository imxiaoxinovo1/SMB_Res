# 02_models/xgboost/feature_selection.py
"""
Two-stage feature selection:
  Stage 1: XGBoost importance — drop features with 0 importance
  Stage 2: RFE with XGBoost estimator — keep top N features
Output: data/selected_vars.json  {feature_name: importance_score}
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import RFE
from config import TABULAR_CSV, SELECTED_VARS_JSON, XGB_PARAMS, TRAIN_YEAR_MAX

print("=== Feature Selection ===")

df = pd.read_csv(TABULAR_CSV)
df_train = df[(df['annual_balance_m'].notna()) & (df['year'] <= TRAIN_YEAR_MAX)].copy()

EXCLUDE = ['glacier_id', 'year', 'annual_balance_m']
feature_cols = [c for c in df_train.columns if c not in EXCLUDE]
X = df_train[feature_cols].fillna(df_train[feature_cols].median()).values
y = df_train['annual_balance_m'].values

print(f"Training samples: {len(y)}, features: {len(feature_cols)}")

# Stage 1: fit full XGBoost, drop zero-importance features
model_full = xgb.XGBRegressor(**XGB_PARAMS)
model_full.fit(X, y)
importances = model_full.feature_importances_
nonzero_mask = importances > 0
feature_cols_nz = [f for f, m in zip(feature_cols, nonzero_mask) if m]
X_nz = X[:, nonzero_mask]
print(f"After zero-importance drop: {len(feature_cols_nz)} features")

# Stage 2: RFE — keep top 30 features
N_SELECT = min(30, len(feature_cols_nz))
estimator = xgb.XGBRegressor(**{**XGB_PARAMS, 'n_estimators': 100})
rfe = RFE(estimator, n_features_to_select=N_SELECT, step=5)
rfe.fit(X_nz, y)
selected = [f for f, s in zip(feature_cols_nz, rfe.support_) if s]
print(f"After RFE: {len(selected)} features")

# Save with importance scores
sel_importances = dict(zip(feature_cols, importances))
selected_dict = {k: float(sel_importances.get(k, 0)) for k in selected}
selected_dict = dict(sorted(selected_dict.items(), key=lambda x: -x[1]))

os.makedirs(os.path.dirname(SELECTED_VARS_JSON), exist_ok=True)
with open(SELECTED_VARS_JSON, 'w') as f:
    json.dump({'selected_features': list(selected_dict.keys()),
               'importances': selected_dict}, f, indent=2)
print(f"Saved → {SELECTED_VARS_JSON}")
print("Top 10:", list(selected_dict.keys())[:10])
