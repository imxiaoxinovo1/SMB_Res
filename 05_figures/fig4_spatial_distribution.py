# 05_figures/fig4_spatial_distribution.py
"""Fig 4: Spatial distribution of mean SMB and trend (2-panel map)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from config import RESULT_DIR, FIG_DIR

os.makedirs(FIG_DIR, exist_ok=True)
df = pd.read_csv(os.path.join(RESULT_DIR, 'RGI02_SMB_reconstruction.csv'))

# Per-glacier mean SMB and trend
def glacier_stats(g):
    g_s = g.sort_values('year')
    mean_smb = g_s['predicted_smb_m'].mean()
    if len(g_s) >= 5:
        sl, _, _, _, _ = linregress(g_s['year'], g_s['predicted_smb_m'])
        trend = sl * 1000  # mm/yr per yr
    else:
        trend = np.nan
    return pd.Series({'mean_smb': mean_smb, 'trend_mm_yr2': trend,
                      'lat': g_s['cenlat'].iloc[0], 'lon': g_s['cenlon'].iloc[0]})

gstats = df.groupby('rgi_id').apply(glacier_stats, include_groups=False).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel A: mean SMB
sc1 = axes[0].scatter(gstats['lon'], gstats['lat'],
                      c=gstats['mean_smb'], cmap='RdBu', vmin=-3, vmax=1,
                      s=3, alpha=0.6, rasterized=True)
plt.colorbar(sc1, ax=axes[0], label='Mean SMB (m w.e. yr⁻¹)')
axes[0].set_title('Mean SMB 1950–2024')
axes[0].set_xlabel('Longitude'); axes[0].set_ylabel('Latitude')

# Panel B: trend
sc2 = axes[1].scatter(gstats['lon'], gstats['lat'],
                      c=gstats['trend_mm_yr2'], cmap='RdBu_r', vmin=-20, vmax=20,
                      s=3, alpha=0.6, rasterized=True)
plt.colorbar(sc2, ax=axes[1], label='SMB trend (mm w.e. yr⁻²)')
axes[1].set_title('SMB Trend 1950–2024')
axes[1].set_xlabel('Longitude'); axes[1].set_ylabel('Latitude')

plt.tight_layout()
out = os.path.join(FIG_DIR, 'fig4_spatial_distribution.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved → {out}")
plt.close()
