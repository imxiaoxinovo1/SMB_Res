# 04_reconstruction/step02_extract_era5_all.py
"""
ERA5 月度数据提取到全部 ~4,999 个重建目标冰川（1950–2024）。
逐批处理（每批 500 冰川）以控制内存。
Output: data/era5_monthly_rgi02.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import xarray as xr
from config import (RGI02_TARGET_CSV, ERA5_NC, ERA5_RGI02_CSV,
                    MONTHLY_CLIMATE_VARS, RECON_YEAR_MIN, RECON_YEAR_MAX)

print("=== Reconstruction Step 02: ERA5 Extraction for All RGI02 ===")

df_target = pd.read_csv(RGI02_TARGET_CSV)
lats = df_target['cenlat'].values
lons = df_target['cenlon'].values
rgi_ids = df_target['rgi_id'].values
print(f"Target glaciers: {len(df_target):,}")

print(f"Opening ERA5: {ERA5_NC}")
ds = xr.open_dataset(ERA5_NC, chunks={'valid_time': 120})
if 'valid_time' in ds.dims:
    ds = ds.rename({'valid_time': 'time'})
if 'expver' in ds.dims:
    ds = ds.sel(expver=1, drop=True).combine_first(ds.sel(expver=5, drop=True))
ds = ds.sel(time=slice(f'{RECON_YEAR_MIN}-01', f'{RECON_YEAR_MAX}-12'))
times = pd.to_datetime(ds.time.values)
print(f"ERA5 time steps: {len(times)}")

TEMP_VARS  = {'t2m', 'skt', 'd2m'}
ACCUM_VARS = {'tp', 'sf', 'smlt', 'ssrd', 'strd', 'ssr', 'str', 'slhf', 'sshf', 'ro'}

# ERA5 lat bounds
era5_lat_min = float(ds.latitude.values.min())
era5_lat_max = float(ds.latitude.values.max())
lats_clamped = np.clip(lats, era5_lat_min, era5_lat_max)
n_clipped = int((lats != lats_clamped).sum())
if n_clipped > 0:
    print(f"WARNING: {n_clipped} glacier(s) lat clamped to ERA5 bounds")

BATCH_SIZE = 500
all_dfs = []

for batch_start in range(0, len(rgi_ids), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(rgi_ids))
    b_rgi = rgi_ids[batch_start:batch_end]
    b_lat = lats_clamped[batch_start:batch_end]
    b_lon = lons[batch_start:batch_end]

    lats_da = xr.DataArray(b_lat, dims='glacier')
    lons_da = xr.DataArray(b_lon, dims='glacier')

    records = []
    for t in times:
        for gi in range(len(b_rgi)):
            records.append({'rgi_id': b_rgi[gi], 'year': t.year, 'month': t.month})

    df_batch = pd.DataFrame(records)

    for var in MONTHLY_CLIMATE_VARS:
        if var not in ds.data_vars:
            continue
        interp = ds[var].interp(latitude=lats_da, longitude=lons_da,
                                method='linear').values  # (T, B)
        if var in TEMP_VARS:
            interp = interp - 273.15
        elif var in ACCUM_VARS:
            interp = interp * 1000.0
        vals = []
        for ti in range(len(times)):
            for gi in range(len(b_rgi)):
                vals.append(float(interp[ti, gi]))
        df_batch[var] = vals

    all_dfs.append(df_batch)
    print(f"  Batch {batch_start//BATCH_SIZE + 1}: glaciers {batch_start}–{batch_end-1} done")

df_out = pd.concat(all_dfs, ignore_index=True)
df_out.to_csv(ERA5_RGI02_CSV, index=False)
print(f"\nSaved {len(df_out):,} rows → {ERA5_RGI02_CSV}")
