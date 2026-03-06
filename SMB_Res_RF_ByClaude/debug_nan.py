import pandas as pd
import numpy as np

df = pd.read_csv(r'H:\Code\SMB\test\data\study_test\data_glacier_era5_fixed.csv')
df.columns = [c.replace('.1', '') for c in df.columns]
df = df.loc[:, ~df.columns.duplicated()]

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

df_filtered = df[(df['YEAR'] >= 1980) & (df['YEAR'] <= 2024)].copy()
print(f"Year-filtered: {len(df_filtered)} rows")

# Per-feature NaN count
print("\nNaN per feature:")
for f in features:
    n = df_filtered[f].isna().sum()
    if n > 0:
        print(f"  {f}: {n} NaN")

# Rows with any NaN
has_nan = df_filtered[features].isna().any(axis=1)
print(f"\nRows with NaN in any feature: {has_nan.sum()}")
print(f"  TAG=9999: {(has_nan & (df_filtered['TAG'] == 9999)).sum()}")
print(f"  TAG!=9999 (band): {(has_nan & (df_filtered['TAG'] != 9999)).sum()}")

# Year distribution of NaN rows
nan_rows = df_filtered[has_nan]
print(f"\nYear range of NaN rows: {nan_rows['YEAR'].min()} - {nan_rows['YEAR'].max()}")
print(f"Unique years with NaN: {sorted(nan_rows['YEAR'].unique())}")
