# 05_figures/fig6_hugonnet_validation.py
"""Fig 6: Compare 2000–2019 reconstruction vs Hugonnet 2021 geodetic mass change."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from config import RESULT_DIR, FIG_DIR, HUGONNET_RATES, RGI_LINKS_CSV

os.makedirs(FIG_DIR, exist_ok=True)

# Load Hugonnet 20-year rates (2000–2020)
df_hug = pd.read_csv(HUGONNET_RATES)
df_hug20 = df_hug[df_hug['period'] == '2000-01-01_2020-01-01'][['rgiid', 'dmdtda']].copy()
df_hug20 = df_hug20.rename(columns={'rgiid': 'rgi60_id', 'dmdtda': 'hugonnet_dmdtda_m'})

# Load RGI7 → RGI6 links
df_links = pd.read_csv(RGI_LINKS_CSV)[['rgi7_id', 'rgi6_id']].copy()

# Our reconstruction 2000–2019 mean per glacier
df_recon = pd.read_csv(os.path.join(RESULT_DIR, 'RGI02_SMB_reconstruction.csv'))
df_2019  = df_recon[(df_recon['year'] >= 2000) & (df_recon['year'] <= 2019)]
df_our   = df_2019.groupby('rgi_id')['predicted_smb_m'].mean().reset_index()
df_our.columns = ['rgi7_id', 'our_smb_m']

# Join
df_joined = df_our.merge(df_links, on='rgi7_id', how='inner')
df_joined = df_joined.merge(df_hug20, left_on='rgi6_id', right_on='rgi60_id', how='inner')
print(f"Matched glaciers: {len(df_joined)}")

obs  = df_joined['hugonnet_dmdtda_m'].values
pred = df_joined['our_smb_m'].values
r2   = r2_score(obs, pred)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(obs, pred, alpha=0.4, s=12, color='#2c7bb6', edgecolors='none')
lo, hi = min(obs.min(), pred.min()) - 0.3, max(obs.max(), pred.max()) + 0.3
ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='1:1 line')
ax.set_xlabel('Hugonnet 2021 dmdtda (m w.e. yr⁻¹)')
ax.set_ylabel('Our reconstruction mean 2000–2019 (m w.e. yr⁻¹)')
ax.set_title(f'External Validation vs Hugonnet 2021\nR²={r2:.3f}  n={len(df_joined)}')
ax.legend()
plt.tight_layout()
out = os.path.join(FIG_DIR, 'fig6_hugonnet_validation.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved → {out}")
plt.close()
