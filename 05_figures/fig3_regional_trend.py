# 05_figures/fig3_regional_trend.py
"""Fig 3: Area-weighted regional SMB trend 1950–2024."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from config import RESULT_DIR, FIG_DIR

os.makedirs(FIG_DIR, exist_ok=True)
df = pd.read_csv(os.path.join(RESULT_DIR, 'regional_stats.csv'))
years = df['year'].values
smb   = df['smb_weighted_mean'].values

slope, intercept, r, p, _ = linregress(years, smb)
trend = slope * years + intercept

fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#d73027' if v < 0 else '#4575b4' for v in smb]
ax.bar(years, smb, color=colors, alpha=0.75, width=0.8, label='Annual area-weighted SMB')
ax.plot(years, trend, 'k-', lw=1.5,
        label=f'Trend: {slope*1000:.1f} mm w.e. yr⁻²  (p={p:.3f})')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('SMB (m w.e. yr⁻¹)')
ax.set_title('RGI02 Area-weighted Mean SMB 1950–2024 (XGBoost reconstruction)')
ax.legend()
plt.tight_layout()
out = os.path.join(FIG_DIR, 'fig3_regional_trend.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved → {out}")
plt.close()
