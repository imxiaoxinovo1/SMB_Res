"""Check if sklearn RF can train with NaN in AREA (exactly like user's script)"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv(r'H:\Code\SMB\test\data\study_test\data_glacier_era5_fixed.csv')
df.columns = [c.replace('.1', '') for c in df.columns]
df = df.loc[:, ~df.columns.duplicated()]
df = df[(df['YEAR'] >= 1980) & (df['YEAR'] <= 2024)].copy()

features = [
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
valid_features = [c for c in features if c in df.columns]

# Test just one year - does sklearn raise error?
test_year = 2000
train = df[df['YEAR'] != test_year]
test = df[(df['YEAR'] == test_year) & (df['TAG'] == 9999)]
print(f"Training NaN in AREA: {train['AREA'].isna().sum()}")
print(f"Test NaN in AREA: {test['AREA'].isna().sum()}")
print(f"Training rows: {len(train)}, Test rows: {len(test)}")

try:
    model = RandomForestRegressor(n_estimators=5, max_depth=None, random_state=42, n_jobs=1)
    model.fit(train[valid_features], train['ANNUAL_BALANCE'])
    print("SUCCESS: sklearn fit with NaN AREA!")
    y_pred = model.predict(test[valid_features])
    print("SUCCESS: sklearn predict with NaN AREA!")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
