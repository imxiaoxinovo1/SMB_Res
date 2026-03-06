import pandas as pd
df = pd.read_csv(r'H:\Code\SMB\test\data\study_test\data_glacier_era5_fixed.csv')
df.columns = [c.replace('.1', '') for c in df.columns]
df = df.loc[:, ~df.columns.duplicated()]

mask_9999 = df['TAG'] == 9999
print('TAG=9999 rows BEFORE Fix_9999:')
print(f'  LOWER_BOUND mean: {df.loc[mask_9999, "LOWER_BOUND"].mean():.1f}')
print(f'  UPPER_BOUND mean: {df.loc[mask_9999, "UPPER_BOUND"].mean():.1f}')

df_bands = df[df['TAG'] != 9999]
glacier_bounds = df_bands.groupby('WGMS_ID').agg({'LOWER_BOUND': 'min', 'UPPER_BOUND': 'max'}).reset_index()
glacier_bounds.rename(columns={'LOWER_BOUND': 'REAL_LOWER', 'UPPER_BOUND': 'REAL_UPPER'}, inplace=True)
df2 = pd.merge(df.copy(), glacier_bounds, on='WGMS_ID', how='left')
mask2 = df2['TAG'] == 9999
df2.loc[mask2, 'LOWER_BOUND'] = df2.loc[mask2, 'REAL_LOWER'].fillna(df2.loc[mask2, 'LOWER_BOUND'])
df2.loc[mask2, 'UPPER_BOUND'] = df2.loc[mask2, 'REAL_UPPER'].fillna(df2.loc[mask2, 'UPPER_BOUND'])

print('TAG=9999 rows AFTER Fix_9999:')
print(f'  LOWER_BOUND mean: {df2.loc[mask2, "LOWER_BOUND"].mean():.1f}')
print(f'  UPPER_BOUND mean: {df2.loc[mask2, "UPPER_BOUND"].mean():.1f}')

orig_lb = df.loc[mask_9999, 'LOWER_BOUND'].values
orig_ub = df.loc[mask_9999, 'UPPER_BOUND'].values
new_lb = df2.loc[mask2, 'LOWER_BOUND'].values
new_ub = df2.loc[mask2, 'UPPER_BOUND'].values
changed = ((orig_lb != new_lb) | (orig_ub != new_ub)).sum()
print(f'  Changed rows: {changed} / {mask_9999.sum()}')
