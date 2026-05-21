# 04_reconstruction/step04_regional_stats.py
"""
计算区域面积加权年均 SMB 统计，与 Hugonnet 2021 比较。
Output: results/regional_stats.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
from config import RESULT_DIR

print("=== Reconstruction Step 04: Regional Stats ===")

df = pd.read_csv(os.path.join(RESULT_DIR, 'RGI02_SMB_reconstruction.csv'))

def area_weighted_mean(group):
    w = group['area_km2']
    s = group['predicted_smb_m']
    valid = w.notna() & s.notna()
    if valid.sum() == 0:
        return pd.Series({'smb_weighted_mean': np.nan, 'smb_equal_mean': np.nan,
                          'smb_std': np.nan, 'n_glaciers': 0})
    wm = np.average(s[valid], weights=w[valid])
    return pd.Series({'smb_weighted_mean': wm,
                      'smb_equal_mean': s[valid].mean(),
                      'smb_std': s[valid].std(),
                      'n_glaciers': int(valid.sum())})

annual = df.groupby('year').apply(area_weighted_mean, include_groups=False).reset_index()
out = os.path.join(RESULT_DIR, 'regional_stats.csv')
annual.to_csv(out, index=False)
print(f"Saved → {out}")
print(annual[['year', 'smb_weighted_mean', 'n_glaciers']].tail(10).to_string())
