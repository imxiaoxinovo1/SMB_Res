# 03_evaluation/compare_models.py
"""
Aggregate all LOYO/LOGO/holdout metrics into one comparison table.
Reads *_loyo_summary.csv, *_logo_summary.csv, holdout_metrics.csv from RESULT_DIR.
Output: results/model_comparison.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import glob
from config import RESULT_DIR

print("=== Model Comparison ===")

dfs = []
patterns = (
    glob.glob(os.path.join(RESULT_DIR, '*_loyo_summary.csv')) +
    glob.glob(os.path.join(RESULT_DIR, '*_logo_summary.csv')) +
    [os.path.join(RESULT_DIR, 'holdout_metrics.csv')]
)
for f in patterns:
    if os.path.exists(f):
        df = pd.read_csv(f)
        df['source_file'] = os.path.basename(f)
        dfs.append(df)

if not dfs:
    print("No result files found. Run model training first.")
else:
    df_all = pd.concat(dfs, ignore_index=True)
    out = os.path.join(RESULT_DIR, 'model_comparison.csv')
    df_all.to_csv(out, index=False)
    print(f"Saved → {out}")
    print(df_all.to_string())
