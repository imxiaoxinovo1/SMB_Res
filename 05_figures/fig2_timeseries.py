# 05_figures/fig2_timeseries.py
"""Fig 2: Observed vs reconstructed time series for 5 example glaciers."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import RESULT_DIR, FIG_DIR, MASSBAL_RGI02_CSV

os.makedirs(FIG_DIR, exist_ok=True)

df_recon = pd.read_csv(os.path.join(RESULT_DIR, 'RGI02_SMB_reconstruction.csv'))
df_obs   = pd.read_csv(MASSBAL_RGI02_CSV)

# Pick 5 glaciers with the most observations
top5 = (df_obs.groupby('glacier_id')['year'].count()
        .nlargest(5).index.tolist())

fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
for ax, gid in zip(axes, top5):
    obs_g = df_obs[df_obs['glacier_id'] == gid].sort_values('year')
    # Try matching by glacier_id in reconstruction (training glaciers have rgi_id)
    # Use glacier_id directly if present, else skip
    ax.plot(obs_g['year'], obs_g['annual_balance'], 'o', ms=4,
            color='black', label='Observed', zorder=3)
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_ylabel('SMB (m w.e.)')
    ax.set_title(f'Glacier ID {gid}')
    ax.legend(loc='upper right', fontsize=8)

axes[-1].set_xlabel('Year')
plt.tight_layout()
out = os.path.join(FIG_DIR, 'fig2_timeseries.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved → {out}")
plt.close()
