# 05_figures/fig1_validation_scatter.py
"""Fig 1: LOYO / LOGO / hold-out validation scatter (3-panel)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from config import RESULT_DIR, FIG_DIR

os.makedirs(FIG_DIR, exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for fname, label, ax in [
    ('xgboost_loyo_predictions.csv', 'LOYO', axes[0]),
    ('xgboost_logo_predictions.csv', 'LOGO', axes[1]),
    ('xgboost_holdout_predictions.csv', 'Hold-out 2015–2024', axes[2]),
]:
    fpath = os.path.join(RESULT_DIR, fname)
    if not os.path.exists(fpath):
        ax.set_title(f'{label}\n(not run yet)')
        continue
    df = pd.read_csv(fpath)
    obs, pred = df['obs'].values, df['pred'].values
    r2 = r2_score(obs, pred)
    rmse = np.sqrt(mean_squared_error(obs, pred)) * 1000
    ax.scatter(obs, pred, alpha=0.4, s=20, color='steelblue', edgecolors='none')
    lo = min(obs.min(), pred.min()) - 0.3
    hi = max(obs.max(), pred.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='1:1')
    ax.set_xlabel('Observed (m w.e.)')
    ax.set_ylabel('Predicted (m w.e.)')
    ax.set_title(f'{label}\nR²={r2:.3f}  RMSE={rmse:.0f} mm')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

plt.tight_layout()
out = os.path.join(FIG_DIR, 'fig1_validation_scatter.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved → {out}")
plt.close()
