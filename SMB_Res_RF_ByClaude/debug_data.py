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

print(f"Total rows: {len(df)}")
print(f"Year range: {df['YEAR'].min()} - {df['YEAR'].max()}")

# After year filter (user's approach)
df_filtered = df[(df['YEAR'] >= 1980) & (df['YEAR'] <= 2024)].copy()
print(f"\nAfter year filter (1980-2024): {len(df_filtered)} rows")

# NaN in features within year filter
nan_count = df_filtered[features].isna().sum().sum()
target_nan = df_filtered['ANNUAL_BALANCE'].isna().sum()
print(f"NaN in 31 features (year-filtered): {nan_count}")
print(f"NaN in ANNUAL_BALANCE (year-filtered): {target_nan}")

# After dropna within year filter
df_clean = df_filtered.dropna(subset=features + ['ANNUAL_BALANCE'])
print(f"After dropna (within year filter): {len(df_clean)} rows")

# Before year filter, after dropna (my script's approach)
df_dropna_first = df.dropna(subset=features + ['ANNUAL_BALANCE'])
df_then_year = df_dropna_first[(df_dropna_first['YEAR'] >= 1980) & (df_dropna_first['YEAR'] <= 2024)]
print(f"\nMy script order (dropna->year filter): {len(df_then_year)} rows")

# TAG=9999 test set size
print(f"\nTAG=9999 rows in 1980-2024 (before dropna): {len(df_filtered[df_filtered['TAG']==9999])}")
print(f"TAG=9999 rows in 1980-2024 (after dropna): {len(df_clean[df_clean['TAG']==9999])}")
