# 05_figures/fig_logo_4models.py
"""4-panel LOGO scatter plot: RF / XGBoost / LSTM / GlacioFormer"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde, pearsonr
from sklearn.metrics import mean_squared_error
from config import RESULT_DIR, FIG_DIR

MODELS = [
    ('rf',              'RF'),
    ('xgboost',         'XGBoost'),
    ('lstm',            'LSTM'),
    ('glacioformer_full', 'GlacioFormer'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
fig.suptitle(
    'Evaluation of modelled annual glacier-wide SMB against ground truth SMB\n'
    'using LOGO cross validation',
    fontsize=13,
)

for ax, (model_key, model_label) in zip(axes, MODELS):
    pred_csv = os.path.join(RESULT_DIR, f'{model_key}_logo_predictions.csv')
    if not os.path.exists(pred_csv):
        ax.set_title(f'{model_label}\n(no results)')
        ax.text(0.5, 0.5, 'Run training first', transform=ax.transAxes,
                ha='center', va='center', color='gray')
        continue

    df = pd.read_csv(pred_csv).dropna()
    obs  = df['obs'].values
    pred = df['pred'].values

    pearson_r, _ = pearsonr(obs, pred)
    rmse = np.sqrt(mean_squared_error(obs, pred))
    mae  = np.mean(np.abs(pred - obs))
    bias = np.mean(pred - obs)

    # Kernel density for color
    xy = np.vstack([obs, pred])
    density = gaussian_kde(xy)(xy)
    density_norm = density / density.max()

    sc = ax.scatter(obs, pred, c=density_norm, cmap='plasma',
                    s=15, alpha=0.75, linewidths=0,
                    norm=Normalize(vmin=0, vmax=1))
    plt.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=density.max()),
                                cmap='plasma'), ax=ax, label='Density')

    # 1:1 line
    lim = max(abs(obs).max(), abs(pred).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k-', linewidth=1.2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')

    stats_txt = (f'R = {pearson_r:.2f}\n'
                 f'RMSE = {rmse:.2f}\n'
                 f'MAE = {mae:.2f}\n'
                 f'Bias = {bias:+.2f}')
    ax.text(0.04, 0.96, stats_txt, transform=ax.transAxes,
            va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    ax.set_title(model_label, fontsize=12)
    ax.set_xlabel('Ground truth SMB (m w.e. a⁻¹)')
    ax.set_ylabel('Predicted SMB (m w.e. a⁻¹)')
    ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
os.makedirs(FIG_DIR, exist_ok=True)
out_path = os.path.join(FIG_DIR, 'fig_logo_4models.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved → {out_path}")
plt.close()
