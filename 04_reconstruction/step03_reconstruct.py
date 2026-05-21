# 04_reconstruction/step03_reconstruct.py
"""
使用 XGBoost（最终全量训练）对全部 RGI02 目标冰川进行 1950–2024 重建。
Output: results/RGI02_SMB_reconstruction.csv
  columns: rgi_id, year, predicted_smb_m, area_km2, cenlat, cenlon
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from config import (TABULAR_CSV, ERA5_RGI02_CSV, RGI02_TARGET_CSV,
                    SELECTED_VARS_JSON, RESULT_DIR,
                    XGB_PARAMS, STATIC_FEATURES, MONTHLY_CLIMATE_VARS,
                    RECON_YEAR_MIN, RECON_YEAR_MAX,
                    CAL_SUMMER_MONTHS, CAL_WINTER_MONTHS,
                    HYD_ACCUM_MONTHS, HYD_ABLAT_MONTHS)

print("=== Reconstruction Step 03: Reconstruct 1950–2024 ===")

TEMP_LIKE = {'t2m', 'skt', 'd2m', 'sd', 'asn'}

def seasonal_agg(df_sub, prefix):
    feats = {}
    for v in MONTHLY_CLIMATE_VARS:
        if v not in df_sub.columns:
            continue
        vals = df_sub[v]
        if v in TEMP_LIKE:
            feats[f'{prefix}_{v}_mean'] = vals.mean() if len(vals) > 0 else np.nan
        else:
            feats[f'{prefix}_{v}_sum'] = vals.sum() if len(vals) > 0 else np.nan
    return feats

# ── 1. Train final XGBoost on ALL labeled data ───────────────────────────────
df_tab = pd.read_csv(TABULAR_CSV)
df_labeled = df_tab[df_tab['annual_balance_m'].notna()].copy()
with open(SELECTED_VARS_JSON) as f:
    feature_cols = json.load(f)['selected_features']

X_all = df_labeled[feature_cols].fillna(df_labeled[feature_cols].median()).values
y_all = df_labeled['annual_balance_m'].values
feat_medians = df_labeled[feature_cols].median()

model = xgb.XGBRegressor(**XGB_PARAMS)
model.fit(X_all, y_all)
print(f"Final model trained on {len(y_all)} labeled samples")

# ── 2. Build tabular features for all RGI02 target glaciers ─────────────────
df_era5    = pd.read_csv(ERA5_RGI02_CSV)
df_terrain = pd.read_csv(RGI02_TARGET_CSV)

rows = []
for rid, g_era5 in df_era5.groupby('rgi_id'):
    for year, y_era5 in g_era5.groupby('year'):
        if len(y_era5) < 12:
            continue
        row = {'rgi_id': rid, 'year': year}
        # Annual means/sums
        for v in MONTHLY_CLIMATE_VARS:
            if v not in y_era5.columns:
                continue
            if v in TEMP_LIKE:
                row[f'ann_{v}_mean'] = y_era5[v].mean()
            else:
                row[f'ann_{v}_sum'] = y_era5[v].sum()
        # Cal/hyd seasonal
        row.update(seasonal_agg(y_era5[y_era5['month'].isin(CAL_SUMMER_MONTHS)], 'cal_summer'))
        row.update(seasonal_agg(y_era5[y_era5['month'].isin(CAL_WINTER_MONTHS)], 'cal_winter'))
        row.update(seasonal_agg(y_era5[y_era5['month'].isin(HYD_ABLAT_MONTHS)], 'hyd_ablat'))
        # hyd_accum: prev year Oct-Dec + cur year Jan-Apr
        prev = g_era5[g_era5['year'] == year - 1]
        hyd_accum = pd.concat([prev[prev['month'].isin([10,11,12])],
                                y_era5[y_era5['month'].isin([1,2,3,4])]], ignore_index=True)
        row.update(seasonal_agg(hyd_accum, 'hyd_accum'))
        rows.append(row)

df_feat = pd.DataFrame(rows)
# Merge static terrain features
terrain_sta = [c for c in STATIC_FEATURES if c in df_terrain.columns]
df_feat = df_feat.merge(df_terrain[['rgi_id'] + terrain_sta], on='rgi_id', how='left')
print(f"Reconstruction feature rows: {len(df_feat):,}")

# ── 3. Predict ───────────────────────────────────────────────────────────────
X_recon = df_feat[feature_cols].fillna(feat_medians).values
df_feat['predicted_smb_m'] = model.predict(X_recon)

df_out = df_feat[['rgi_id', 'year', 'predicted_smb_m']].merge(
    df_terrain[['rgi_id', 'area_km2', 'cenlat', 'cenlon']], on='rgi_id', how='left'
)
os.makedirs(RESULT_DIR, exist_ok=True)
out_path = os.path.join(RESULT_DIR, 'RGI02_SMB_reconstruction.csv')
df_out.to_csv(out_path, index=False)
print(f"Saved {len(df_out):,} rows → {out_path}")
print(df_out['predicted_smb_m'].describe())
