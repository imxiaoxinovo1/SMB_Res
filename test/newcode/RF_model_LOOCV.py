"""
Random Forest Model with Leave-One-Year-Out Cross-Validation
Using the new processed dataset from newcode pipeline

This script implements:
1. RFECV feature selection
2. Leave-One-Year-Out cross-validation (LOOCV)
3. RandomForest regression
4. Feature importance visualization

Input:
    - H:/Code/SMB/test/result_data/wgms_era5_merged_final.csv

Output:
    - Model performance metrics (R^2, RMSE, MAE)
    - Feature importance plot
    - Prediction results CSV

Author: Claude Code
Date: 2025-12-29
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

# ================= Configuration =================

DATA_PATH = r"H:\Code\SMB\test\result_data\wgms_era5_merged_final.csv"
OUTPUT_DIR = r"H:\Code\SMB\test\result_data"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "rf_loocv_predictions.csv")
METRICS_FILE = os.path.join(OUTPUT_DIR, "rf_loocv_metrics.txt")

# ================= 1. Data Loading and Preparation =================

print("=" * 70)
print("RANDOM FOREST MODEL - LEAVE-ONE-YEAR-OUT CROSS-VALIDATION")
print("=" * 70)

print("\n>>> Step 1: Loading data...")
print(f"  Path: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"  Loaded: {len(df):,} records")
print(f"  Glaciers: {df['WGMS_ID'].nunique()}")
print(f"  Years: {df['YEAR'].min()}-{df['YEAR'].max()}")

# Remove records with missing target variable
df_clean = df.dropna(subset=['ANNUAL_BALANCE']).copy()
print(f"  After removing missing ANNUAL_BALANCE: {len(df_clean):,} records")

# ================= 2. Feature Engineering =================

print("\n>>> Step 2: Building feature pool...")

# 1. Climate features (annual)
climate_annual = [col for col in df_clean.columns if col.endswith('_year')]

# 2. Climate features (summer)
climate_summer = [col for col in df_clean.columns if col.endswith('_summer')]

# 3. Geographic features
geography_features = ['LATITUDE', 'LONGITUDE']

# 4. Glacier geometry features (if available)
geometry_features = []
potential_geom = ['AREA', 'ELA', 'AAR', 'LOWER_BOUND', 'UPPER_BOUND']
for feat in potential_geom:
    if feat in df_clean.columns:
        geometry_features.append(feat)

# 5. CRITICAL: Include YEAR as feature
time_feature = ['YEAR']

# Combine all features
all_features = geography_features + geometry_features + climate_annual + climate_summer + time_feature

print(f"\n  Feature categories:")
print(f"    Geography        : {len(geography_features)} features")
print(f"    Glacier geometry : {len(geometry_features)} features")
print(f"    Climate (annual) : {len(climate_annual)} features")
print(f"    Climate (summer) : {len(climate_summer)} features")
print(f"    Time             : {len(time_feature)} features")
print(f"    TOTAL            : {len(all_features)} features")

# Fill missing values in features with median
print("\n  Handling missing values...")
for col in all_features:
    if col in df_clean.columns:
        missing_count = df_clean[col].isna().sum()
        if missing_count > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            print(f"    Filled {missing_count} missing values in {col}")

# Prepare feature matrix and target
X = df_clean[all_features].copy()
y = df_clean['ANNUAL_BALANCE'].copy()
years = df_clean['YEAR'].copy()
glacier_ids = df_clean['WGMS_ID'].copy()

print(f"\n  Final dataset shape: X={X.shape}, y={y.shape}")

# ================= 3. RFECV Feature Selection =================

print("\n>>> Step 3: Running RFECV feature selection...")
print("  This may take a few minutes...")

# Use RandomForest with RFECV to select optimal features
rf_selector = RandomForestRegressor(
    n_estimators=100,
    max_depth=50,
    random_state=42,
    n_jobs=-1
)

rfecv = RFECV(
    estimator=rf_selector,
    step=1,
    cv=KFold(5, shuffle=True, random_state=42),
    scoring='neg_mean_squared_error',
    min_features_to_select=10,
    n_jobs=-1
)

rfecv.fit(X, y)

# Get selected features
selected_features = np.array(all_features)[rfecv.support_]
n_selected = len(selected_features)

print(f"\n  Optimal number of features: {n_selected}")
print(f"\n  Selected features:")
for i, feat in enumerate(selected_features, 1):
    print(f"    {i:2d}. {feat}")

# ================= 4. Leave-One-Year-Out Cross-Validation =================

print("\n>>> Step 4: Running Leave-One-Year-Out Cross-Validation...")
print("  Training model for each test year...")

results = {
    'y_true': [],
    'y_pred': [],
    'year': [],
    'glacier_id': []
}

yearly_metrics = []
unique_years = sorted(years.unique())

for test_year in unique_years:
    # Split data: current year for testing, all other years for training
    train_mask = (years != test_year)
    test_mask = (years == test_year)

    # Skip if too few test samples
    n_test = test_mask.sum()
    if n_test < 2:
        print(f"    Year {test_year}: SKIPPED (only {n_test} samples)")
        continue

    # Prepare train/test data using selected features
    X_train = X.loc[train_mask, selected_features]
    y_train = y[train_mask]
    X_test = X.loc[test_mask, selected_features]
    y_test = y[test_mask]

    # Train RandomForest model
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Predict on test year
    y_pred = rf_model.predict(X_test)

    # Store results
    results['y_true'].extend(y_test.values)
    results['y_pred'].extend(y_pred)
    results['year'].extend([test_year] * n_test)
    results['glacier_id'].extend(glacier_ids[test_mask].values)

    # Calculate year-specific metrics
    year_r2 = r2_score(y_test, y_pred)
    year_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    year_mae = mean_absolute_error(y_test, y_pred)

    yearly_metrics.append({
        'year': test_year,
        'n_samples': n_test,
        'r2': year_r2,
        'rmse': year_rmse,
        'mae': year_mae
    })

    print(f"    Year {test_year}: n={n_test:3d}, R^2={year_r2:6.3f}, RMSE={year_rmse:6.3f}, MAE={year_mae:6.3f}")

# ================= 5. Overall Performance Evaluation =================

print("\n>>> Step 5: Overall performance evaluation...")

y_true = np.array(results['y_true'])
y_pred = np.array(results['y_pred'])

# Calculate overall metrics
overall_r2 = r2_score(y_true, y_pred)
overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
overall_mae = mean_absolute_error(y_true, y_pred)
overall_r = np.corrcoef(y_true, y_pred)[0, 1]
bias = np.mean(y_pred - y_true)

print("\n" + "=" * 70)
print("OVERALL MODEL PERFORMANCE (Leave-One-Year-Out CV)")
print("=" * 70)
print(f"  R^2 (Coefficient of Determination) : {overall_r2:7.4f}")
print(f"  R (Correlation Coefficient)       : {overall_r:7.4f}")
print(f"  RMSE (Root Mean Squared Error)    : {overall_rmse:7.4f} m w.e.")
print(f"  MAE (Mean Absolute Error)         : {overall_mae:7.4f} m w.e.")
print(f"  Bias (Mean Prediction Error)      : {bias:7.4f} m w.e.")
print(f"  Total test samples                : {len(y_true):7d}")
print("=" * 70)

# ================= 6. Feature Importance Analysis =================

print("\n>>> Step 6: Feature importance analysis...")

# Train model on full dataset to get feature importances
rf_final = RandomForestRegressor(n_estimators=200, max_depth=100, random_state=42, n_jobs=-1)
rf_final.fit(X[selected_features], y)

importances = rf_final.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': selected_features,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n  Top 15 most important features:")
for i, row in feature_importance_df.head(15).iterrows():
    print(f"    {row['feature']:40s}: {row['importance']:.4f}")

# ================= 7. Save Results =================

print("\n>>> Step 7: Saving results...")

# Save predictions
pred_df = pd.DataFrame({
    'YEAR': results['year'],
    'WGMS_ID': results['glacier_id'],
    'ANNUAL_BALANCE_TRUE': y_true,
    'ANNUAL_BALANCE_PRED': y_pred,
    'RESIDUAL': y_true - y_pred
})
pred_df.to_csv(RESULTS_FILE, index=False, encoding='utf-8-sig')
print(f"  Predictions saved: {RESULTS_FILE}")

# Save metrics
with open(METRICS_FILE, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("RANDOM FOREST MODEL - LEAVE-ONE-YEAR-OUT CV RESULTS\n")
    f.write("=" * 70 + "\n\n")

    f.write("OVERALL METRICS:\n")
    f.write(f"  R^2   : {overall_r2:.4f}\n")
    f.write(f"  R     : {overall_r:.4f}\n")
    f.write(f"  RMSE : {overall_rmse:.4f} m w.e.\n")
    f.write(f"  MAE  : {overall_mae:.4f} m w.e.\n")
    f.write(f"  Bias : {bias:.4f} m w.e.\n")
    f.write(f"  N    : {len(y_true)}\n\n")

    f.write(f"NUMBER OF FEATURES SELECTED: {n_selected}\n\n")

    f.write("SELECTED FEATURES:\n")
    for i, feat in enumerate(selected_features, 1):
        f.write(f"  {i:2d}. {feat}\n")

    f.write("\n" + "=" * 70 + "\n")
    f.write("FEATURE IMPORTANCE (Top 20):\n")
    f.write("=" * 70 + "\n")
    for i, row in feature_importance_df.head(20).iterrows():
        f.write(f"  {row['feature']:40s}: {row['importance']:.4f}\n")

    f.write("\n" + "=" * 70 + "\n")
    f.write("YEARLY PERFORMANCE:\n")
    f.write("=" * 70 + "\n")
    for metric in yearly_metrics:
        f.write(f"  Year {metric['year']}: n={metric['n_samples']:3d}, "
                f"R^2={metric['r2']:6.3f}, RMSE={metric['rmse']:6.3f}, MAE={metric['mae']:6.3f}\n")

print(f"  Metrics saved: {METRICS_FILE}")

# ================= 8. Visualization =================

print("\n>>> Step 8: Creating visualizations...")

# Figure 1: Feature Importance
plt.figure(figsize=(12, 8))

top_n = min(20, len(feature_importance_df))
top_features = feature_importance_df.head(top_n)

# Color code by feature type
colors = []
for feat in top_features['feature']:
    if feat in geography_features or feat in geometry_features:
        colors.append('red')
    elif '_summer' in feat:
        colors.append('orange')
    elif feat == 'YEAR':
        colors.append('green')
    else:
        colors.append('skyblue')

sns.barplot(
    data=top_features,
    y='feature',
    x='importance',
    hue='feature',
    palette=colors,
    legend=False
)

plt.title(f'Feature Importance (Random Forest LOOCV)\nR^2 = {overall_r2:.4f}, RMSE = {overall_rmse:.3f} m w.e.',
          fontsize=14, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()

# Save figure
fig1_path = os.path.join(OUTPUT_DIR, 'rf_loocv_feature_importance.png')
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
print(f"  Feature importance plot saved: {fig1_path}")
plt.show()

# Figure 2: Predicted vs Observed
plt.figure(figsize=(10, 10))

plt.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

# Add 1:1 line
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')

# Add regression line
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)
x_line = np.array([min_val, max_val])
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, 'b-', linewidth=2, label=f'Regression line (slope={slope:.2f})')

plt.xlabel('Observed Annual Balance (m w.e.)', fontsize=12)
plt.ylabel('Predicted Annual Balance (m w.e.)', fontsize=12)
plt.title(f'Random Forest LOOCV Performance\nR^2 = {overall_r2:.4f}, RMSE = {overall_rmse:.3f} m w.e., n = {len(y_true)}',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()

# Save figure
fig2_path = os.path.join(OUTPUT_DIR, 'rf_loocv_scatter.png')
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
print(f"  Scatter plot saved: {fig2_path}")
plt.show()

# Figure 3: Residual plot
plt.figure(figsize=(12, 6))

residuals = y_true - y_pred

plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Annual Balance (m w.e.)', fontsize=12)
plt.ylabel('Residual (Observed - Predicted)', fontsize=12)
plt.title('Residual Plot', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Residual (m w.e.)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title(f'Residual Distribution\nMean = {residuals.mean():.3f}, Std = {residuals.std():.3f}',
          fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
fig3_path = os.path.join(OUTPUT_DIR, 'rf_loocv_residuals.png')
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
print(f"  Residual plots saved: {fig3_path}")
plt.show()

# ================= Final Summary =================

print("\n" + "=" * 70)
print("MODEL EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)
print("\nOutput files:")
print(f"  📄 Predictions  : {RESULTS_FILE}")
print(f"  📄 Metrics      : {METRICS_FILE}")
print(f"  📊 Feature Imp. : {fig1_path}")
print(f"  📊 Scatter Plot : {fig2_path}")
print(f"  📊 Residuals    : {fig3_path}")
print("\n" + "=" * 70)
print("Key Results:")
print(f"  R^2   = {overall_r2:.4f}")
print(f"  RMSE = {overall_rmse:.3f} m w.e.")
print(f"  MAE  = {overall_mae:.3f} m w.e.")
print(f"  Bias = {bias:.3f} m w.e.")
print("=" * 70)
