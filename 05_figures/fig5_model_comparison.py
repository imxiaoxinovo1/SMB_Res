# 05_figures/fig5_model_comparison.py
"""Fig 5: Model performance comparison (LOYO/LOGO R² bar chart)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import matplotlib.pyplot as plt
from config import RESULT_DIR, FIG_DIR

os.makedirs(FIG_DIR, exist_ok=True)
fpath = os.path.join(RESULT_DIR, 'model_comparison.csv')
if not os.path.exists(fpath):
    print("model_comparison.csv not found — run compare_models.py first")
    exit(0)

df = pd.read_csv(fpath)
# Show R² grouped by model and CV type
pivot = df.pivot_table(index='model', columns='cv', values='r2', aggfunc='mean')

fig, ax = plt.subplots(figsize=(9, 5))
pivot.plot(kind='bar', ax=ax, colormap='Set2', edgecolor='white', width=0.6)
ax.set_ylabel('R²')
ax.set_title('Model Performance: LOYO / LOGO / Hold-out R²')
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
ax.legend(title='CV type')
ax.axhline(0, color='k', lw=0.5, ls=':')
plt.tight_layout()
out = os.path.join(FIG_DIR, 'fig5_model_comparison.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved → {out}")
plt.close()
