"""直接复现 test_rf_fixed.py 的逻辑，比较 R2"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

csv_file = r"H:\Code\SMB\test\data\study_test\data_glacier_era5_fixed.csv"
df = pd.read_csv(csv_file)
df.columns = [c.replace('.1', '') for c in df.columns]
df = df.loc[:, ~df.columns.duplicated()]

# 年份筛选
df = df[(df['YEAR'] >= 1980) & (df['YEAR'] <= 2024)].copy()
print(f"Year-filtered rows: {len(df)}")

features_columns = [
    "LOWER_BOUND", "UPPER_BOUND", "AREA", "LATITUDE", "LONGITUDE",
    "temperature_2m_year", "temperature_2m_summer",
    "skin_temperature_year", "skin_temperature_summer",
    "dewpoint_temperature_2m_summer",
    "total_precipitation_sum_year", "total_precipitation_sum_summer",
    "snowfall_sum_year", "snowfall_sum_summer",
    "snow_depth_year", "snow_depth_summer",
    "snow_density_summer", "snow_albedo_summer",
    "surface_net_solar_radiation_sum_summer",
    "surface_net_thermal_radiation_sum_summer",
    "surface_solar_radiation_downwards_sum_summer",
    "surface_thermal_radiation_downwards_sum_summer",
    "surface_sensible_heat_flux_sum_summer",
    "surface_latent_heat_flux_sum_summer",
    "total_evaporation_sum_year", "total_evaporation_sum_summer",
    "snow_evaporation_sum_summer",
    "runoff_sum_summer",
    "snowmelt_sum_year", "snowmelt_sum_summer",
    "YEAR"
]
valid_features = [c for c in features_columns if c in df.columns]
target_column = 'ANNUAL_BALANCE'
print(f"Features: {len(valid_features)}")

min_year = int(df['YEAR'].min())
max_year = int(df['YEAR'].max())

y_test_all, y_pred_all = [], []

for test_year in range(min_year, max_year + 1):
    train_data = df[df['YEAR'] != test_year]
    test_data = df[(df['YEAR'] == test_year) & (df['TAG'] == 9999)]
    if len(test_data) == 0:
        continue
    model = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    model.fit(train_data[valid_features], train_data[target_column])
    y_pred = model.predict(test_data[valid_features]) / 1000
    y_test = test_data[target_column].values / 1000
    y_test_all.extend(y_test)
    y_pred_all.extend(y_pred)

y_true = np.array(y_test_all)
y_pred = np.array(y_pred_all)
r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r = np.corrcoef(y_true, y_pred)[0, 1]
bias = np.mean(y_pred - y_true)
print(f"\n=== Replicated test_rf_fixed.py ===")
print(f"n predictions: {len(y_true)}")
print(f"R2   = {r2:.4f}")
print(f"R    = {r:.4f}")
print(f"RMSE = {rmse:.4f} m w.e.")
print(f"Bias = {bias:.4f} m w.e.")
