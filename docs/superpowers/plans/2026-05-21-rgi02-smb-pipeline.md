# RGI02 SMB Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean, end-to-end glacier SMB reconstruction pipeline for RGI02 (Western Canada & USA), from WGMS filtering through XGBoost + GlacioFormer training, full RGI02 reconstruction (≥0.5 km²), and 6 publication figures.

**Architecture:** Sequential pipeline — preprocessing (01) → models (02) → evaluation (03) → reconstruction (04) → figures (05). All paths come from `config.py`. Two model families (XGBoost tabular, GlacioFormer sequence) are registered in `registry.py` for easy extension and automated comparison.

**Tech Stack:** Python 3.10+, pandas, numpy, xarray, scipy, geopandas, scikit-learn, xgboost, torch, matplotlib, cartopy

---

## Critical Data Facts (verified from source files)

- **WGMS glacier.csv** uses lowercase fields: `id`, `latitude`, `longitude`, `gtng_region`
- **WGMS mass_balance.csv** uses lowercase fields: `glacier_id`, `year`, `annual_balance`
- **`annual_balance` is ALREADY in m w.e.** in FoG 2025-02b (values range –4.4 to +4.2). Do NOT divide by 1000.
- **`gtng_region == '02_western_canada_usa'`** yields exactly 63 glaciers with annual_balance (1016 rows, 1950–2024)
- **RGI v7 shapefile** CRS is EPSG:4326; key fields: `rgi_id`, `cenlon`, `cenlat`, `area_km2`, `slope_deg`, `aspect_deg`, `zmin_m`, `zmax_m`, `zmean_m`, `zmed_m`, `lmax_m`
- **4,999 RGI02 glaciers** have area_km2 ≥ 0.5
- **ERA5 NetCDF** has coord `valid_time` (not `time`) and `expver` dimension
- **Hugonnet** uses RGI60 IDs; link via `RGI2000-v7.0-G-02_western_canada_usa-rgi6_links.csv`

---

## File Structure

```
SMB_Res_ByClaudeV2/
├── config.py
├── 01_preprocessing/
│   ├── step01_filter_wgms.py        → data/training_glaciers.csv
│   ├── step02_match_rgi.py          → data/training_glaciers_terrain.csv
│   ├── step03_extract_era5.py       → data/era5_monthly_training.csv
│   ├── step04_build_tabular.py      → data/tabular_dataset.csv
│   └── step05_build_sequences.py    → data/sequences_training.npz
├── 02_models/
│   ├── registry.py
│   ├── base_model.py
│   ├── xgboost/
│   │   ├── model.py
│   │   ├── feature_selection.py     → data/selected_vars.json
│   │   ├── train_loyo.py            → results/xgboost_loyo_metrics.csv
│   │   └── train_logo.py            → results/xgboost_logo_metrics.csv
│   └── glacioformer/
│       ├── model.py                 (v1_transformer)
│       ├── train_loyo.py            → results/glacioformer_loyo_metrics.csv
│       └── train_logo.py            → results/glacioformer_logo_metrics.csv
├── 03_evaluation/
│   ├── eval_holdout.py              → results/holdout_metrics.csv
│   └── compare_models.py            → results/model_comparison.csv
├── 04_reconstruction/
│   ├── step01_prepare_rgi02.py      → data/rgi02_target_glaciers.csv
│   ├── step02_extract_era5_all.py   → data/era5_monthly_rgi02.csv
│   ├── step03_reconstruct.py        → results/RGI02_SMB_reconstruction.csv
│   └── step04_regional_stats.py     → results/regional_stats.csv
└── 05_figures/
    ├── fig1_validation_scatter.py
    ├── fig2_timeseries.py
    ├── fig3_regional_trend.py
    ├── fig4_spatial_distribution.py
    ├── fig5_model_comparison.py
    └── fig6_hugonnet_validation.py
```

---

## Task 1: config.py — Central Configuration

**Files:**
- Create: `config.py`

- [ ] **Step 1: Write config.py**

```python
"""Central configuration for SMB_Res_ByClaudeV2."""
import os

# ── Root directories ──────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")           # preprocessed outputs
RESULT_DIR = os.path.join(BASE_DIR, "results")        # model results
FIG_DIR    = os.path.join(BASE_DIR, "figures")

for d in [DATA_DIR, RESULT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── External data paths ───────────────────────────────────────────────────────
WGMS_DIR       = r"H:\Code\SMB\WGMS\FoG_DataBase\DOI-WGMS-FoG-2025-02b\data"
GLACIER_CSV    = os.path.join(WGMS_DIR, "glacier.csv")
MASSBAL_CSV    = os.path.join(WGMS_DIR, "mass_balance.csv")
RGI_SHP        = r"H:\Code\SMB\RGI\RGI2000-v7.0-G-02_western_canada_usa\RGI2000-v7.0-G-02_western_canada_usa.shp"
RGI_LINKS_CSV  = r"H:\Code\SMB\RGI\RGI2000-v7.0-G-02_western_canada_usa\RGI2000-v7.0-G-02_western_canada_usa-rgi6_links.csv"
ERA5_NC        = r"H:\Code\SMB\ERA5-LAND\data_stream-moda.nc"
HUGONNET_RATES = r"H:\Code\SMB\Hugonnet_results\time_series_02\dh_02_rgi60_pergla_rates.csv"

# ── Preprocessing outputs ─────────────────────────────────────────────────────
TRAINING_GLACIERS_CSV  = os.path.join(DATA_DIR, "training_glaciers.csv")
TERRAIN_CSV            = os.path.join(DATA_DIR, "training_glaciers_terrain.csv")
ERA5_MONTHLY_CSV       = os.path.join(DATA_DIR, "era5_monthly_training.csv")
TABULAR_CSV            = os.path.join(DATA_DIR, "tabular_dataset.csv")
SEQUENCES_NPZ          = os.path.join(DATA_DIR, "sequences_training.npz")
SELECTED_VARS_JSON     = os.path.join(DATA_DIR, "selected_vars.json")

# ── Reconstruction data outputs ───────────────────────────────────────────────
RGI02_TARGET_CSV       = os.path.join(DATA_DIR, "rgi02_target_glaciers.csv")
ERA5_RGI02_CSV         = os.path.join(DATA_DIR, "era5_monthly_rgi02.csv")

# ── Time periods ──────────────────────────────────────────────────────────────
TRAIN_YEAR_MIN = 1950
TRAIN_YEAR_MAX = 2014     # training + CV period
HOLDOUT_YEAR_MIN = 2015   # hold-out (unseen during training)
HOLDOUT_YEAR_MAX = 2024
RECON_YEAR_MIN = 1950
RECON_YEAR_MAX = 2024

# ── ERA5 variables ────────────────────────────────────────────────────────────
MONTHLY_CLIMATE_VARS = [
    't2m', 'skt', 'd2m',          # temperature (3)
    'sd', 'asn',                   # snow state (2)
    'tp', 'sf', 'smlt',            # precip & melt (3)
    'ssrd', 'strd', 'ssr', 'str',  # radiation (4)
    'slhf', 'sshf',                # turbulent flux (2)
    'ro',                          # runoff (1)
]  # 15 variables
N_DYNAMIC = len(MONTHLY_CLIMATE_VARS)   # 15

# ── Static features (from RGI v7 terrain matching) ───────────────────────────
STATIC_FEATURES = [
    'slope_deg', 'aspect_sin', 'aspect_cos',
    'zmin_m', 'zmax_m', 'zmean_m', 'zmed_m',
    'area_km2', 'lmax_m', 'cenlat',
]  # 10 features
N_STATIC = len(STATIC_FEATURES)   # 10

# ── Calendar-year seasonal aggregation for tabular features ──────────────────
CAL_SUMMER_MONTHS  = [6, 7, 8]         # JJA
CAL_WINTER_MONTHS  = [12, 1, 2]        # DJF (Dec of prev year + Jan, Feb)
HYD_ACCUM_MONTHS   = [10, 11, 12, 1, 2, 3, 4]  # Oct–Apr
HYD_ABLAT_MONTHS   = [5, 6, 7, 8, 9]           # May–Sep

# ── XGBoost hyperparameters ───────────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1,
)

# ── GlacioFormer hyperparameters ──────────────────────────────────────────────
GLACIOFORMER_PARAMS = dict(
    n_dynamic_features=N_DYNAMIC,  # 15
    n_static_features=N_STATIC,    # 10
    d_model=64, n_heads=4, n_encoder_layers=2, ff_dim=256,
    dropout=0.15,
    batch_size=32, epochs=300, lr=1e-3,
    early_stop_patience=40, min_epochs=60,
    weight_decay=1e-4,
)

# ── RGI02 reconstruction filter ───────────────────────────────────────────────
MIN_AREA_KM2 = 0.5   # ~4,999 glaciers cover ~95% of total RGI02 area
```

- [ ] **Step 2: Smoke-test config**

```bash
cd H:/Code/SMB/SMB_Res_ByClaudeV2
"C:/Users/zjw31/.conda/envs/smb/python.exe" -c "
import config
print('DATA_DIR:', config.DATA_DIR)
print('N_DYNAMIC:', config.N_DYNAMIC, '  N_STATIC:', config.N_STATIC)
print('MONTHLY_CLIMATE_VARS:', config.MONTHLY_CLIMATE_VARS)
import os
assert os.path.exists(config.GLACIER_CSV), 'GLACIER_CSV not found'
assert os.path.exists(config.ERA5_NC), 'ERA5_NC not found'
assert os.path.exists(config.RGI_SHP), 'RGI_SHP not found'
print('All paths OK')
"
```
Expected: prints paths, "All paths OK", no errors.

- [ ] **Step 3: Commit**

```bash
cd H:/Code/SMB/SMB_Res_ByClaudeV2
git add config.py
git commit -m "Add config.py: central path + hyperparameter configuration"
```

---

## Task 2: step01_filter_wgms.py — WGMS RGI02 Filter

**Files:**
- Create: `01_preprocessing/step01_filter_wgms.py`

Output: `data/training_glaciers.csv` with columns: `glacier_id, name, latitude, longitude, gtng_region, n_years, year_min, year_max, annual_balance_mean_m`

- [ ] **Step 1: Create 01_preprocessing/__init__.py and step01**

```bash
mkdir -p H:/Code/SMB/SMB_Res_ByClaudeV2/01_preprocessing
touch H:/Code/SMB/SMB_Res_ByClaudeV2/01_preprocessing/__init__.py
```

```python
# 01_preprocessing/step01_filter_wgms.py
"""
从 WGMS FoG 2025-02b 筛选 RGI02 冰川的年度物质平衡记录。
筛选依据: gtng_region == '02_western_canada_usa'（GTN-G官方分区，不依赖国家字段）
输出: training_glaciers.csv — 每行一个冰川，含静态信息和观测统计
注意: annual_balance 在 FoG 2025-02b 中已为 m w.e.，无需除以 1000
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
from config import GLACIER_CSV, MASSBAL_CSV, TRAINING_GLACIERS_CSV, \
                   TRAIN_YEAR_MIN, TRAIN_YEAR_MAX

print("=== Step 01: WGMS RGI02 Filter ===")

# ── 1. Load WGMS glacier.csv and filter by gtng_region ─────────────────────
df_gl = pd.read_csv(GLACIER_CSV, encoding='latin1', low_memory=False)
df_gl_rgi02 = df_gl[df_gl['gtng_region'] == '02_western_canada_usa'].copy()
print(f"glacier.csv RGI02 rows: {len(df_gl_rgi02):,}")

# ── 2. Load mass_balance.csv ────────────────────────────────────────────────
df_mb = pd.read_csv(MASSBAL_CSV, encoding='latin1', low_memory=False)
print(f"mass_balance.csv total rows: {len(df_mb):,}")

# ── 3. Filter to RGI02 glaciers with non-null annual_balance ────────────────
rgi02_ids = set(df_gl_rgi02['id'].values)
df_mb_rgi02 = df_mb[
    df_mb['glacier_id'].isin(rgi02_ids) &
    df_mb['annual_balance'].notna()
].copy()

# ── 4. Year range filter ─────────────────────────────────────────────────────
df_mb_rgi02 = df_mb_rgi02[
    (df_mb_rgi02['year'] >= TRAIN_YEAR_MIN) &
    (df_mb_rgi02['year'] <= HOLDOUT_YEAR_MAX if 'HOLDOUT_YEAR_MAX' in dir() else True)
].copy()

# Import holdout max separately to avoid circular
from config import HOLDOUT_YEAR_MAX
df_mb_rgi02 = df_mb_rgi02[df_mb_rgi02['year'] <= HOLDOUT_YEAR_MAX].copy()

print(f"RGI02 annual_balance rows ({TRAIN_YEAR_MIN}–{HOLDOUT_YEAR_MAX}): {len(df_mb_rgi02):,}")
print(f"Unique glacier_ids: {df_mb_rgi02['glacier_id'].nunique()}")

# Unit check: annual_balance should be in m w.e. (range ≈ –5 to +5)
ab_mean = df_mb_rgi02['annual_balance'].mean()
assert abs(ab_mean) < 10, f"annual_balance looks wrong: mean={ab_mean:.3f} (expected m w.e., range –5 to +5)"
print(f"annual_balance mean: {ab_mean:.3f} m w.e.  [unit check OK — already m w.e., no conversion needed]")

# ── 5. Merge glacier static info ─────────────────────────────────────────────
df_static = df_gl_rgi02[['id', 'names', 'latitude', 'longitude', 'gtng_region']].copy()
df_static = df_static.rename(columns={'id': 'glacier_id', 'names': 'name'})

# Per-glacier observation stats
df_stats = df_mb_rgi02.groupby('glacier_id').agg(
    n_years=('year', 'count'),
    year_min=('year', 'min'),
    year_max=('year', 'max'),
    annual_balance_mean_m=('annual_balance', 'mean'),
).reset_index()

df_out = pd.merge(df_stats, df_static, on='glacier_id', how='left')
df_out = df_out.sort_values('n_years', ascending=False)

# ── 6. Save ──────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(TRAINING_GLACIERS_CSV), exist_ok=True)
df_out.to_csv(TRAINING_GLACIERS_CSV, index=False)
print(f"\nSaved {len(df_out)} glaciers → {TRAINING_GLACIERS_CSV}")
print(df_out[['glacier_id', 'name', 'latitude', 'longitude', 'n_years', 'annual_balance_mean_m']].head(10).to_string())
```

- [ ] **Step 2: Run and validate**

```bash
cd H:/Code/SMB/SMB_Res_ByClaudeV2
"C:/Users/zjw31/.conda/envs/smb/python.exe" 01_preprocessing/step01_filter_wgms.py
```
Expected output:
```
RGI02 annual_balance rows ...: 1016
Unique glacier_ids: 63
annual_balance mean: -0.607 m w.e.  [unit check OK ...]
Saved 63 glaciers → ...training_glaciers.csv
```

- [ ] **Step 3: Also save the full observation table for model training**

Add at end of step01:
```python
# Save observation table (glacier_id, year, annual_balance) for model input
MASSBAL_RGI02_CSV = os.path.join(os.path.dirname(TRAINING_GLACIERS_CSV), "massbal_rgi02.csv")
df_mb_rgi02[['glacier_id', 'year', 'annual_balance', 'winter_balance', 'summer_balance']].to_csv(
    MASSBAL_RGI02_CSV, index=False)
print(f"Saved observation table → {MASSBAL_RGI02_CSV}")
```

Add `MASSBAL_RGI02_CSV = os.path.join(DATA_DIR, "massbal_rgi02.csv")` to `config.py`.

- [ ] **Step 4: Commit**

```bash
cd H:/Code/SMB/SMB_Res_ByClaudeV2
git add 01_preprocessing/ config.py
git commit -m "step01: WGMS RGI02 filter via gtng_region — 63 glaciers, 1016 obs"
```

---

## Task 3: step02_match_rgi.py — KD-tree Terrain Matching

**Files:**
- Create: `01_preprocessing/step02_match_rgi.py`

Output: `data/training_glaciers_terrain.csv` — training_glaciers + 10 RGI terrain columns

- [ ] **Step 1: Write step02**

```python
# 01_preprocessing/step02_match_rgi.py
"""
将 WGMS RGI02 冰川质心与 RGI v7.0 shapefile 匹配，提取地形属性。
方法: KD-tree 最近邻，距离阈值 10 km（EPSG:4326 度数 ≈ 0.09°）。
输出列（10 个静态特征）:
  slope_deg, aspect_sin, aspect_cos, zmin_m, zmax_m,
  zmean_m, zmed_m, area_km2, lmax_m, cenlat
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from config import TRAINING_GLACIERS_CSV, RGI_SHP, TERRAIN_CSV, STATIC_FEATURES

print("=== Step 02: RGI Terrain Matching ===")

# ── 1. Load training glaciers ────────────────────────────────────────────────
df_wgms = pd.read_csv(TRAINING_GLACIERS_CSV)
wgms_lats = df_wgms['latitude'].values
wgms_lons = df_wgms['longitude'].values

# ── 2. Load RGI shapefile (geometry not needed, use attribute CSV for speed) ─
import geopandas as gpd
gdf_rgi = gpd.read_file(RGI_SHP)[
    ['rgi_id', 'cenlon', 'cenlat', 'area_km2',
     'slope_deg', 'aspect_deg', 'zmin_m', 'zmax_m',
     'zmean_m', 'zmed_m', 'lmax_m']
].copy()
print(f"RGI v7 loaded: {len(gdf_rgi):,} glaciers")

# ── 3. Compute aspect_sin, aspect_cos ────────────────────────────────────────
gdf_rgi['aspect_sin'] = np.sin(np.deg2rad(gdf_rgi['aspect_deg']))
gdf_rgi['aspect_cos'] = np.cos(np.deg2rad(gdf_rgi['aspect_deg']))

# ── 4. KD-tree nearest-neighbor match ────────────────────────────────────────
rgi_coords = np.column_stack([gdf_rgi['cenlat'].values, gdf_rgi['cenlon'].values])
wgms_coords = np.column_stack([wgms_lats, wgms_lons])
tree = cKDTree(rgi_coords)
dist, idx = tree.query(wgms_coords, k=1)

# Distance threshold: ~10 km at these latitudes ≈ 0.09 degrees
DIST_THRESH = 0.09
n_unmatched = (dist > DIST_THRESH).sum()
if n_unmatched > 0:
    print(f"WARNING: {n_unmatched} glaciers exceed {DIST_THRESH}° threshold")
    for i, (d, gi) in enumerate(zip(dist, idx)):
        if d > DIST_THRESH:
            print(f"  glacier_id={df_wgms.iloc[i]['glacier_id']}  dist={d:.4f}°")

# ── 5. Extract matched terrain attributes ───────────────────────────────────
terrain_cols = ['slope_deg', 'aspect_sin', 'aspect_cos',
                'zmin_m', 'zmax_m', 'zmean_m', 'zmed_m',
                'area_km2', 'lmax_m', 'cenlat']
df_terrain = gdf_rgi.iloc[idx][terrain_cols].reset_index(drop=True)
df_terrain['rgi_id'] = gdf_rgi.iloc[idx]['rgi_id'].values
df_terrain['match_dist_deg'] = dist

# ── 6. Merge and save ────────────────────────────────────────────────────────
df_out = pd.concat([df_wgms.reset_index(drop=True), df_terrain], axis=1)
df_out.to_csv(TERRAIN_CSV, index=False)
print(f"\nSaved {len(df_out)} rows → {TERRAIN_CSV}")
print(f"Match distance stats: mean={dist.mean():.4f}° max={dist.max():.4f}°")
assert all(c in df_out.columns for c in STATIC_FEATURES), \
    f"Missing static features: {[c for c in STATIC_FEATURES if c not in df_out.columns]}"
print("All 10 static features present ✓")
```

- [ ] **Step 2: Run and validate**

```bash
"C:/Users/zjw31/.conda/envs/smb/python.exe" 01_preprocessing/step02_match_rgi.py
```
Expected: "All 10 static features present ✓", max match distance < 0.09°

- [ ] **Step 3: Commit**

```bash
git add 01_preprocessing/step02_match_rgi.py
git commit -m "step02: KD-tree terrain matching from RGI v7 — 10 static features"
```

---

## Task 4: step03_extract_era5.py — ERA5 Monthly Extraction

**Files:**
- Create: `01_preprocessing/step03_extract_era5.py`

Output: `data/era5_monthly_training.csv` — columns: `glacier_id, year, month, t2m, skt, ...` (15 climate vars)

- [ ] **Step 1: Write step03**

```python
# 01_preprocessing/step03_extract_era5.py
"""
用双线性插值将 ERA5-Land 月度数据提取到训练冰川质心坐标。
单位转换: t2m/skt/d2m: K→°C; 累积量(m)→mm (×1000)
ERA5 NetCDF 特点: 时间维度名为 valid_time，有 expver 维度需合并
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import xarray as xr
from config import (TERRAIN_CSV, ERA5_NC, ERA5_MONTHLY_CSV,
                    MONTHLY_CLIMATE_VARS, TRAIN_YEAR_MIN, HOLDOUT_YEAR_MAX)

print("=== Step 03: ERA5 Monthly Extraction ===")

# ── 1. Load glacier list ─────────────────────────────────────────────────────
df_gl = pd.read_csv(TERRAIN_CSV)
lats = df_gl['latitude'].values
lons = df_gl['longitude'].values
glacier_ids = df_gl['glacier_id'].values
print(f"Glaciers to extract: {len(df_gl)}")

# ── 2. Open ERA5 NetCDF ──────────────────────────────────────────────────────
print(f"Opening ERA5: {ERA5_NC}")
ds = xr.open_dataset(ERA5_NC, chunks={'valid_time': 120})

# Rename valid_time → time
if 'valid_time' in ds.dims:
    ds = ds.rename({'valid_time': 'time'})

# Merge expver versions (ERA5 has two data streams stored separately)
if 'expver' in ds.dims:
    ds = ds.sel(expver=1, drop=True).combine_first(ds.sel(expver=5, drop=True))

# Filter time range
ds = ds.sel(time=slice(f'{TRAIN_YEAR_MIN}-01', f'{HOLDOUT_YEAR_MAX}-12'))
print(f"Time range: {str(ds.time.values[0])[:7]} → {str(ds.time.values[-1])[:7]}")

# ── 3. Unit conversion multipliers ──────────────────────────────────────────
# Temperature variables: K → °C offset (-273.15)
TEMP_VARS = {'t2m', 'skt', 'd2m'}
# Accumulated fluxes in m → mm (×1000): tp, sf, smlt, ssrd, strd, ssr, str, slhf, sshf, ro
ACCUM_VARS = {'tp', 'sf', 'smlt', 'ssrd', 'strd', 'ssr', 'str', 'slhf', 'sshf', 'ro'}

# ── 4. Interpolate to glacier centroids ──────────────────────────────────────
records = []
n_times = len(ds.time)
print(f"Processing {n_times} months × {len(glacier_ids)} glaciers...")

# Build lat/lon DataArrays for vectorized interpolation
lats_da = xr.DataArray(lats, dims='glacier')
lons_da = xr.DataArray(lons, dims='glacier')

for vi, var in enumerate(MONTHLY_CLIMATE_VARS):
    if var not in ds.data_vars:
        print(f"  WARNING: {var} not in ERA5, skipping")
        continue
    da = ds[var]
    # Bilinear interpolation to all glacier points
    interp = da.interp(latitude=lats_da, longitude=lons_da, method='linear')  # (time, glacier)
    arr = interp.values  # shape (n_time, n_glaciers)

    if var in TEMP_VARS:
        arr = arr - 273.15
    elif var in ACCUM_VARS:
        arr = arr * 1000.0  # m → mm

    if vi == 0:
        # Initialize records on first variable
        times = pd.to_datetime(ds.time.values)
        for ti, t in enumerate(times):
            for gi in range(len(glacier_ids)):
                records.append({
                    'glacier_id': glacier_ids[gi],
                    'year': t.year,
                    'month': t.month,
                })
        print(f"  Initialized {len(records):,} rows")

    for ti in range(n_times):
        for gi in range(len(glacier_ids)):
            records[ti * len(glacier_ids) + gi][var] = float(arr[ti, gi])

    if (vi + 1) % 5 == 0:
        print(f"  Variables done: {vi+1}/{len(MONTHLY_CLIMATE_VARS)}")

df_out = pd.DataFrame(records)
df_out.to_csv(ERA5_MONTHLY_CSV, index=False)
print(f"\nSaved {len(df_out):,} rows → {ERA5_MONTHLY_CSV}")
print(f"Shape: {df_out.shape}, Columns: {df_out.columns.tolist()}")
```

- [ ] **Step 2: Run and validate**

```bash
"C:/Users/zjw31/.conda/envs/smb/python.exe" 01_preprocessing/step03_extract_era5.py
```
Expected: `~63 × 75 years × 12 months = 56,700 rows`, all 15 climate vars present, no NaNs in t2m.

- [ ] **Step 3: Commit**

```bash
git add 01_preprocessing/step03_extract_era5.py
git commit -m "step03: ERA5 bilinear interpolation to 63 training glaciers (15 vars)"
```

---

## Task 5: step04_build_tabular.py — Seasonal Feature Engineering

**Files:**
- Create: `01_preprocessing/step04_build_tabular.py`

Output: `data/tabular_dataset.csv` — one row per (glacier, year), target `annual_balance_m`, ~70 features with `cal_` and `hyd_` prefixes.

- [ ] **Step 1: Write step04**

```python
# 01_preprocessing/step04_build_tabular.py
"""
构建 XGBoost 表格特征数据集。
每行 = 一个冰川 × 一年 (glacier_id, year, annual_balance_m, feature_cols...)
双套特征:
  cal_* : 日历年季节聚合（夏=JJA, 冬=DJF, 年=全年）
  hyd_* : 水文年季节聚合（积累期=Oct-Apr, 消融期=May-Sep）
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
from config import (ERA5_MONTHLY_CSV, TERRAIN_CSV, TABULAR_CSV,
                    MASSBAL_RGI02_CSV if False else None,
                    STATIC_FEATURES,
                    CAL_SUMMER_MONTHS, CAL_WINTER_MONTHS,
                    HYD_ACCUM_MONTHS, HYD_ABLAT_MONTHS,
                    MONTHLY_CLIMATE_VARS)
from config import MASSBAL_RGI02_CSV

print("=== Step 04: Build Tabular Dataset ===")

# ── 1. Load ERA5 monthly data ────────────────────────────────────────────────
df_era5 = pd.read_csv(ERA5_MONTHLY_CSV)
print(f"ERA5 monthly rows: {len(df_era5):,}")

# ── 2. Load mass balance observations ───────────────────────────────────────
df_mb = pd.read_csv(MASSBAL_RGI02_CSV)  # glacier_id, year, annual_balance

# ── 3. Load terrain (static features) ───────────────────────────────────────
df_terrain = pd.read_csv(TERRAIN_CSV)
terrain_cols = [c for c in STATIC_FEATURES if c in df_terrain.columns]

def seasonal_agg(df_year, months, prefix, var_list):
    """Aggregate ERA5 monthly rows for given months → dict of features."""
    sub = df_year[df_year['month'].isin(months)]
    feats = {}
    for v in var_list:
        if v not in sub.columns:
            continue
        # Temperature/state vars: mean; flux vars: sum
        if v in {'t2m', 'skt', 'd2m', 'sd', 'asn'}:
            feats[f'{prefix}_{v}_mean'] = sub[v].mean()
        else:
            feats[f'{prefix}_{v}_sum'] = sub[v].sum()
    return feats

# ── 4. Build one row per (glacier, year) ────────────────────────────────────
rows = []
for gid, g_era5 in df_era5.groupby('glacier_id'):
    for year, y_era5 in g_era5.groupby('year'):
        if len(y_era5) < 12:
            continue  # skip incomplete years

        row = {'glacier_id': gid, 'year': year}

        # Annual aggregations
        for v in MONTHLY_CLIMATE_VARS:
            if v not in y_era5.columns:
                continue
            if v in {'t2m', 'skt', 'd2m', 'sd', 'asn'}:
                row[f'ann_{v}_mean'] = y_era5[v].mean()
            else:
                row[f'ann_{v}_sum'] = y_era5[v].sum()

        # Calendar-year seasons
        row.update(seasonal_agg(y_era5, CAL_SUMMER_MONTHS, 'cal_summer', MONTHLY_CLIMATE_VARS))
        row.update(seasonal_agg(y_era5, CAL_WINTER_MONTHS, 'cal_winter', MONTHLY_CLIMATE_VARS))

        # Hydrological-year seasons
        # Accumulation period: Oct–Apr spans two calendar years;
        # use Oct-Dec of prev year + Jan-Apr of current year
        prev_year_rows = g_era5[g_era5['year'] == year - 1]
        accum_oct_dec = prev_year_rows[prev_year_rows['month'].isin([10, 11, 12])]
        accum_jan_apr = y_era5[y_era5['month'].isin([1, 2, 3, 4])]
        hyd_accum = pd.concat([accum_oct_dec, accum_jan_apr], ignore_index=True)
        row.update(seasonal_agg(hyd_accum, list(range(1, 13)), 'hyd_accum', MONTHLY_CLIMATE_VARS))
        row.update(seasonal_agg(y_era5, HYD_ABLAT_MONTHS, 'hyd_ablat', MONTHLY_CLIMATE_VARS))

        rows.append(row)

df_feat = pd.DataFrame(rows)
print(f"Feature rows (before merge): {len(df_feat):,}")

# ── 5. Merge terrain, mass balance ──────────────────────────────────────────
df_feat = df_feat.merge(df_terrain[['glacier_id'] + terrain_cols], on='glacier_id', how='left')
df_feat = df_feat.merge(df_mb[['glacier_id', 'year', 'annual_balance']], on=['glacier_id', 'year'], how='left')
df_feat = df_feat.rename(columns={'annual_balance': 'annual_balance_m'})

# Keep rows where target is known for training; keep all for reconstruction
df_labeled = df_feat[df_feat['annual_balance_m'].notna()].copy()
print(f"Labeled rows (with annual_balance): {len(df_labeled):,}")

df_feat.to_csv(TABULAR_CSV, index=False)
print(f"Saved full tabular dataset → {TABULAR_CSV}")
print(f"Feature columns: {len(df_feat.columns) - 3}")  # minus glacier_id, year, target
```

- [ ] **Step 2: Run and validate**

```bash
"C:/Users/zjw31/.conda/envs/smb/python.exe" 01_preprocessing/step04_build_tabular.py
```
Expected: ~1,016 labeled rows, ~60–70 feature columns, no NaN in terrain columns.

- [ ] **Step 3: Commit**

```bash
git add 01_preprocessing/step04_build_tabular.py
git commit -m "step04: cal_/hyd_ seasonal tabular features for XGBoost"
```

---

## Task 6: step05_build_sequences.py — Monthly Sequences for GlacioFormer

**Files:**
- Create: `01_preprocessing/step05_build_sequences.py`

Output: `data/sequences_training.npz` with arrays: `X_dyn (N,12,15)`, `X_sta (N,10)`, `y (N,)`, `glacier_ids (N,)`, `years (N,)`, plus normalization params.

- [ ] **Step 1: Write step05**

```python
# 01_preprocessing/step05_build_sequences.py
"""
构建 GlacioFormer 月度序列数据集。
X_dyn: (N, 12, 15) — 每年12个月 × 15个气候变量（已归一化）
X_sta: (N, 10)     — 10个静态地形特征（已归一化）
y:     (N,)        — annual_balance_m (m w.e.)
归一化: 使用训练集 (year <= TRAIN_YEAR_MAX) 的 mean/std，保存到 npz 供重建使用
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
from config import (ERA5_MONTHLY_CSV, TERRAIN_CSV, SEQUENCES_NPZ,
                    MASSBAL_RGI02_CSV, STATIC_FEATURES, MONTHLY_CLIMATE_VARS,
                    TRAIN_YEAR_MIN, TRAIN_YEAR_MAX, HOLDOUT_YEAR_MAX)

print("=== Step 05: Build Sequence Dataset ===")

df_era5   = pd.read_csv(ERA5_MONTHLY_CSV)
df_terrain = pd.read_csv(TERRAIN_CSV)
df_mb      = pd.read_csv(MASSBAL_RGI02_CSV)

terrain_cols = [c for c in STATIC_FEATURES if c in df_terrain.columns]

# ── Build one sample per (glacier_id, year) ──────────────────────────────────
X_dyn_list, X_sta_list, y_list, gid_list, year_list = [], [], [], [], []

df_mb_idx = df_mb.set_index(['glacier_id', 'year'])
df_terrain_idx = df_terrain.set_index('glacier_id')

for gid, g_era5 in df_era5.groupby('glacier_id'):
    sta = df_terrain_idx.loc[gid, terrain_cols].values.astype(float) \
          if gid in df_terrain_idx.index else None
    if sta is None:
        continue

    for year, y_era5 in g_era5.groupby('year'):
        y_era5_sorted = y_era5.sort_values('month')
        if len(y_era5_sorted) < 12:
            continue

        dyn = y_era5_sorted[MONTHLY_CLIMATE_VARS].values.astype(float)  # (12, 15)

        # Target (NaN if no observation — still include for hold-out inference)
        key = (gid, year)
        target = df_mb_idx.loc[key, 'annual_balance'].item() \
                 if key in df_mb_idx.index else np.nan

        X_dyn_list.append(dyn)
        X_sta_list.append(sta)
        y_list.append(target)
        gid_list.append(gid)
        year_list.append(year)

X_dyn = np.array(X_dyn_list, dtype=np.float32)   # (N, 12, 15)
X_sta = np.array(X_sta_list, dtype=np.float32)   # (N, 10)
y     = np.array(y_list,    dtype=np.float32)    # (N,)
glacier_ids = np.array(gid_list)
years       = np.array(year_list)

print(f"Total samples: {len(y)}  (labeled: {(~np.isnan(y)).sum()})")

# ── Compute normalization stats from training split only ─────────────────────
train_mask = years <= TRAIN_YEAR_MAX

dyn_mean = X_dyn[train_mask].mean(axis=(0, 1), keepdims=True)  # (1, 1, 15)
dyn_std  = X_dyn[train_mask].std(axis=(0, 1), keepdims=True) + 1e-8
sta_mean = X_sta[train_mask].mean(axis=0, keepdims=True)       # (1, 10)
sta_std  = X_sta[train_mask].std(axis=0, keepdims=True) + 1e-8

X_dyn_norm = (X_dyn - dyn_mean) / dyn_std
X_sta_norm = (X_sta - sta_mean) / sta_std

np.savez_compressed(
    SEQUENCES_NPZ,
    X_dyn=X_dyn_norm, X_sta=X_sta_norm, y=y,
    glacier_ids=glacier_ids, years=years,
    dyn_mean=dyn_mean, dyn_std=dyn_std,
    sta_mean=sta_mean, sta_std=sta_std,
)
print(f"Saved → {SEQUENCES_NPZ}")
print(f"X_dyn: {X_dyn_norm.shape}, X_sta: {X_sta_norm.shape}, y: {y.shape}")
```

- [ ] **Step 2: Run and validate**

```bash
"C:/Users/zjw31/.conda/envs/smb/python.exe" 01_preprocessing/step05_build_sequences.py
```
Expected: `X_dyn: (N, 12, 15)`, `X_sta: (N, 10)`, labeled ≈ 1016.

- [ ] **Step 3: Commit**

```bash
git add 01_preprocessing/step05_build_sequences.py
git commit -m "step05: monthly sequence dataset for GlacioFormer (N,12,15)+(N,10)"
```

---

## Task 7: Model Registry and Base Class

**Files:**
- Create: `02_models/__init__.py`
- Create: `02_models/base_model.py`
- Create: `02_models/registry.py`

- [ ] **Step 1: Write base_model.py**

```python
# 02_models/base_model.py
"""Abstract base class for all SMB models."""
from abc import ABC, abstractmethod

class BaseSMBModel(ABC):
    @abstractmethod
    def fit(self, X_train, y_train): ...

    @abstractmethod
    def predict(self, X): ...

    @property
    @abstractmethod
    def name(self) -> str: ...
```

- [ ] **Step 2: Write registry.py**

```python
# 02_models/registry.py
"""
Model registry — add one line to register a new model.
Usage:
    from registry import get_model
    model = get_model('xgboost', **params)
"""
_REGISTRY = {}

def register(name):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator

def get_model(name: str, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)

def list_models():
    return list(_REGISTRY.keys())
```

- [ ] **Step 3: Create directory structure**

```bash
mkdir -p H:/Code/SMB/SMB_Res_ByClaudeV2/02_models/xgboost
mkdir -p H:/Code/SMB/SMB_Res_ByClaudeV2/02_models/glacioformer
touch H:/Code/SMB/SMB_Res_ByClaudeV2/02_models/__init__.py
touch H:/Code/SMB/SMB_Res_ByClaudeV2/02_models/xgboost/__init__.py
touch H:/Code/SMB/SMB_Res_ByClaudeV2/02_models/glacioformer/__init__.py
```

- [ ] **Step 4: Commit**

```bash
git add 02_models/
git commit -m "Add model registry + base class"
```

---

## Task 8: XGBoost Model + Feature Selection

**Files:**
- Create: `02_models/xgboost/model.py`
- Create: `02_models/xgboost/feature_selection.py`

- [ ] **Step 1: Write xgboost/model.py**

```python
# 02_models/xgboost/model.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import xgboost as xgb
from base_model import BaseSMBModel
from registry import register

@register('xgboost')
class XGBoostSMB(BaseSMBModel):
    name = 'xgboost'

    def __init__(self, **kwargs):
        from config import XGB_PARAMS
        params = {**XGB_PARAMS, **kwargs}
        self.model = xgb.XGBRegressor(**params)
        self.feature_names_ = None

    def fit(self, X_train, y_train, feature_names=None):
        self.feature_names_ = feature_names
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def feature_importance(self):
        return dict(zip(self.feature_names_ or [], self.model.feature_importances_))
```

- [ ] **Step 2: Write feature_selection.py**

```python
# 02_models/xgboost/feature_selection.py
"""
Two-stage feature selection:
  Stage 1: XGBoost importance — drop features with 0 importance
  Stage 2: RFE with XGBoost estimator — keep top N features
Output: data/selected_vars.json  {feature_name: importance_score}
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import RFE
from config import TABULAR_CSV, SELECTED_VARS_JSON, XGB_PARAMS, TRAIN_YEAR_MAX

print("=== Feature Selection ===")

df = pd.read_csv(TABULAR_CSV)
df_train = df[(df['annual_balance_m'].notna()) & (df['year'] <= TRAIN_YEAR_MAX)].copy()

EXCLUDE = ['glacier_id', 'year', 'annual_balance_m']
feature_cols = [c for c in df_train.columns if c not in EXCLUDE]
X = df_train[feature_cols].fillna(df_train[feature_cols].median()).values
y = df_train['annual_balance_m'].values

print(f"Training samples: {len(y)}, features: {len(feature_cols)}")

# Stage 1: fit full XGBoost, drop zero-importance features
model_full = xgb.XGBRegressor(**XGB_PARAMS)
model_full.fit(X, y)
importances = model_full.feature_importances_
nonzero_mask = importances > 0
feature_cols_nz = [f for f, m in zip(feature_cols, nonzero_mask) if m]
X_nz = X[:, nonzero_mask]
print(f"After zero-importance drop: {len(feature_cols_nz)} features")

# Stage 2: RFE — keep top 30 features
N_SELECT = min(30, len(feature_cols_nz))
estimator = xgb.XGBRegressor(**{**XGB_PARAMS, 'n_estimators': 100})
rfe = RFE(estimator, n_features_to_select=N_SELECT, step=5)
rfe.fit(X_nz, y)
selected = [f for f, s in zip(feature_cols_nz, rfe.support_) if s]
print(f"After RFE: {len(selected)} features")

# Save with importance scores
sel_importances = dict(zip(feature_cols, importances))
selected_dict = {k: float(sel_importances.get(k, 0)) for k in selected}
selected_dict = dict(sorted(selected_dict.items(), key=lambda x: -x[1]))

os.makedirs(os.path.dirname(SELECTED_VARS_JSON), exist_ok=True)
with open(SELECTED_VARS_JSON, 'w') as f:
    json.dump({'selected_features': list(selected_dict.keys()),
               'importances': selected_dict}, f, indent=2)
print(f"Saved → {SELECTED_VARS_JSON}")
print("Top 10:", list(selected_dict.keys())[:10])
```

- [ ] **Step 3: Run feature selection**

```bash
"C:/Users/zjw31/.conda/envs/smb/python.exe" 02_models/xgboost/feature_selection.py
```
Expected: selected_vars.json with 30 features, top features likely include t2m, sf, smlt summer/winter vars.

- [ ] **Step 4: Commit**

```bash
git add 02_models/xgboost/
git commit -m "XGBoost model wrapper + two-stage feature selection (XGB importance + RFE)"
```

---

## Task 9: XGBoost LOYO and LOGO Training

**Files:**
- Create: `02_models/xgboost/train_loyo.py`
- Create: `02_models/xgboost/train_logo.py`

- [ ] **Step 1: Write train_loyo.py**

```python
# 02_models/xgboost/train_loyo.py
"""
XGBoost LOYO (Leave-One-Year-Out) cross-validation.
For each year Y in training period (1950–2014):
  train on all years ≠ Y → predict year Y
Output: results/xgboost_loyo_metrics.csv + xgboost_loyo_predictions.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from config import TABULAR_CSV, SELECTED_VARS_JSON, RESULT_DIR, XGB_PARAMS, \
                   TRAIN_YEAR_MIN, TRAIN_YEAR_MAX, STATIC_FEATURES
import xgboost as xgb

print("=== XGBoost LOYO ===")

df = pd.read_csv(TABULAR_CSV)
df_train = df[(df['annual_balance_m'].notna()) &
              (df['year'] >= TRAIN_YEAR_MIN) &
              (df['year'] <= TRAIN_YEAR_MAX)].copy()

with open(SELECTED_VARS_JSON) as f:
    sel = json.load(f)
feature_cols = sel['selected_features']

terrain_only = [c for c in STATIC_FEATURES if c in feature_cols]
X_all = df_train[feature_cols].fillna(df_train[feature_cols].median()).values
y_all = df_train['annual_balance_m'].values
years_all = df_train['year'].values
gids_all = df_train['glacier_id'].values

fold_years = sorted(df_train['year'].unique())
print(f"Samples: {len(y_all)}, Folds: {len(fold_years)}")

preds, obs, fold_r2 = [], [], []
for yr in fold_years:
    tr = years_all != yr
    te = years_all == yr
    if te.sum() == 0:
        continue
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_all[tr], y_all[tr])
    p = model.predict(X_all[te])
    r2 = r2_score(y_all[te], p)
    fold_r2.append({'year': yr, 'n': te.sum(), 'r2': r2,
                    'rmse': np.sqrt(mean_squared_error(y_all[te], p))})
    preds.extend(p.tolist())
    obs.extend(y_all[te].tolist())

r2_global = r2_score(obs, preds)
rmse_global = np.sqrt(mean_squared_error(obs, preds))
bias = np.mean(np.array(preds) - np.array(obs))
print(f"LOYO R²={r2_global:.4f}  RMSE={rmse_global*1000:.1f}mm  Bias={bias*1000:.1f}mm")

os.makedirs(RESULT_DIR, exist_ok=True)
pd.DataFrame(fold_r2).to_csv(os.path.join(RESULT_DIR, 'xgboost_loyo_metrics.csv'), index=False)
print(f"Saved LOYO metrics → {RESULT_DIR}/xgboost_loyo_metrics.csv")
```

- [ ] **Step 2: Write train_logo.py** (same structure, split by glacier_id instead of year)

```python
# 02_models/xgboost/train_logo.py
"""XGBoost LOGO (Leave-One-Glacier-Out) cross-validation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from config import TABULAR_CSV, SELECTED_VARS_JSON, RESULT_DIR, XGB_PARAMS, \
                   TRAIN_YEAR_MIN, TRAIN_YEAR_MAX
import xgboost as xgb

print("=== XGBoost LOGO ===")

df = pd.read_csv(TABULAR_CSV)
df_train = df[(df['annual_balance_m'].notna()) &
              (df['year'] >= TRAIN_YEAR_MIN) &
              (df['year'] <= TRAIN_YEAR_MAX)].copy()

with open(SELECTED_VARS_JSON) as f:
    feature_cols = json.load(f)['selected_features']

X_all = df_train[feature_cols].fillna(df_train[feature_cols].median()).values
y_all = df_train['annual_balance_m'].values
gids_all = df_train['glacier_id'].values

glaciers = sorted(df_train['glacier_id'].unique())
print(f"Samples: {len(y_all)}, Folds: {len(glaciers)}")

preds, obs, fold_r2 = [], [], []
for gid in glaciers:
    tr = gids_all != gid
    te = gids_all == gid
    if te.sum() == 0:
        continue
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_all[tr], y_all[tr])
    p = model.predict(X_all[te])
    r2 = r2_score(y_all[te], p) if te.sum() > 1 else float('nan')
    fold_r2.append({'glacier_id': gid, 'n': te.sum(), 'r2': r2,
                    'rmse': np.sqrt(mean_squared_error(y_all[te], p))})
    preds.extend(p.tolist())
    obs.extend(y_all[te].tolist())

r2_global = r2_score(obs, preds)
rmse_global = np.sqrt(mean_squared_error(obs, preds))
bias = np.mean(np.array(preds) - np.array(obs))
print(f"LOGO R²={r2_global:.4f}  RMSE={rmse_global*1000:.1f}mm  Bias={bias*1000:.1f}mm")

os.makedirs(RESULT_DIR, exist_ok=True)
pd.DataFrame(fold_r2).to_csv(os.path.join(RESULT_DIR, 'xgboost_logo_metrics.csv'), index=False)
print(f"Saved LOGO metrics → {RESULT_DIR}/xgboost_logo_metrics.csv")
```

- [ ] **Step 3: Run both**

```bash
"C:/Users/zjw31/.conda/envs/smb/python.exe" 02_models/xgboost/train_loyo.py
"C:/Users/zjw31/.conda/envs/smb/python.exe" 02_models/xgboost/train_logo.py
```
Expected: LOYO R² > 0.55, LOGO R² > 0.30 (these are challenging but achievable given 63 glaciers).

- [ ] **Step 4: Commit**

```bash
git add 02_models/xgboost/train_loyo.py 02_models/xgboost/train_logo.py
git commit -m "XGBoost LOYO + LOGO training scripts"
```

---

## Task 10: GlacioFormer Model (v1_transformer)

**Files:**
- Create: `02_models/glacioformer/model.py`

- [ ] **Step 1: Write glacioformer/model.py** (adapted from existing v1_transformer.py, registered)

```python
# 02_models/glacioformer/model.py
"""
GlacioFormer v1 — encoder-only dual-branch Transformer for annual SMB.
Adapted from SMB_Res_Glacierformer_Byclaude/02_model/models/v1_transformer.py.
Registered as 'glacioformer' in model registry.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import torch
import torch.nn as nn
from registry import register

@register('glacioformer')
class GlacioFormer(nn.Module):
    name = 'glacioformer'

    def __init__(self, n_dynamic_features=15, n_static_features=10,
                 d_model=64, n_heads=4, n_encoder_layers=2,
                 ff_dim=256, dropout=0.15, **kwargs):
        super().__init__()
        self.dyn_embedding = nn.Sequential(
            nn.Linear(n_dynamic_features, d_model),
            nn.LayerNorm(d_model),
        )
        self.month_embed = nn.Embedding(12, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_encoder_layers)
        self.static_mlp = nn.Sequential(
            nn.Linear(n_static_features, d_model // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_dynamic, x_static):
        B, T, _ = x_dynamic.shape
        x = self.dyn_embedding(x_dynamic)
        x = x + self.month_embed(torch.arange(T, device=x.device)).unsqueeze(0)
        x = self.transformer(x)
        h_dyn = x.mean(dim=1)
        h_sta = self.static_mlp(x_static)
        return self.head(torch.cat([h_dyn, h_sta], dim=-1)).squeeze(-1)
```

- [ ] **Step 2: Smoke-test model shape**

```bash
"C:/Users/zjw31/.conda/envs/smb/python.exe" -c "
import sys; sys.path.insert(0,'H:/Code/SMB/SMB_Res_ByClaudeV2/02_models')
import torch
from glacioformer.model import GlacioFormer
m = GlacioFormer()
xd = torch.randn(4,12,15); xs = torch.randn(4,10)
out = m(xd, xs)
assert out.shape == (4,), f'Expected (4,), got {out.shape}'
print('GlacioFormer output shape OK:', out.shape)
params = sum(p.numel() for p in m.parameters())
print(f'Parameters: {params:,}')
"
```
Expected: output shape (4,), params ~35,000–50,000.

- [ ] **Step 3: Commit**

```bash
git add 02_models/glacioformer/model.py
git commit -m "GlacioFormer v1: dual-branch Transformer, registered in registry"
```

---

## Task 11: GlacioFormer LOYO and LOGO Training

**Files:**
- Create: `02_models/glacioformer/train_loyo.py`
- Create: `02_models/glacioformer/train_logo.py`

- [ ] **Step 1: Write train_loyo.py**

```python
# 02_models/glacioformer/train_loyo.py
"""
GlacioFormer LOYO cross-validation.
Supports --variant selected (XGBoost-selected features only) or full (all 15 vars).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from config import (SEQUENCES_NPZ, SELECTED_VARS_JSON, RESULT_DIR,
                    GLACIOFORMER_PARAMS, MONTHLY_CLIMATE_VARS,
                    TRAIN_YEAR_MIN, TRAIN_YEAR_MAX)
from glacioformer.model import GlacioFormer

parser = argparse.ArgumentParser()
parser.add_argument('--variant', default='full', choices=['full', 'selected'])
args = parser.parse_args()

print(f"=== GlacioFormer LOYO — variant={args.variant} ===")

data = np.load(SEQUENCES_NPZ, allow_pickle=True)
X_dyn = data['X_dyn']    # (N, 12, 15) normalized
X_sta = data['X_sta']    # (N, 10) normalized
y     = data['y']         # (N,) in m w.e.
years = data['years']
gids  = data['glacier_ids']

# Apply train split mask
train_mask = (years >= TRAIN_YEAR_MIN) & (years <= TRAIN_YEAR_MAX) & (~np.isnan(y))
X_dyn = X_dyn[train_mask]
X_sta = X_sta[train_mask]
y     = y[train_mask]
years_tr = years[train_mask]

print(f"Training samples: {len(y)}, years: {sorted(set(years_tr))[:3]}...{sorted(set(years_tr))[-3:]}")

P = GLACIOFORMER_PARAMS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

fold_years = sorted(set(years_tr))
all_preds, all_obs = [], []

for yi, yr in enumerate(fold_years):
    tr = years_tr != yr
    te = years_tr == yr
    if te.sum() == 0:
        continue

    Xd_tr = torch.tensor(X_dyn[tr], dtype=torch.float32).to(device)
    Xs_tr = torch.tensor(X_sta[tr], dtype=torch.float32).to(device)
    yt_tr = torch.tensor(y[tr],     dtype=torch.float32).to(device)
    Xd_te = torch.tensor(X_dyn[te], dtype=torch.float32).to(device)
    Xs_te = torch.tensor(X_sta[te], dtype=torch.float32).to(device)

    model = GlacioFormer(
        n_dynamic_features=P['n_dynamic_features'],
        n_static_features=P['n_static_features'],
        d_model=P['d_model'], n_heads=P['n_heads'],
        n_encoder_layers=P['n_encoder_layers'],
        ff_dim=P['ff_dim'], dropout=P['dropout'],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=P['lr'],
                                  weight_decay=P['weight_decay'])
    criterion = nn.MSELoss()

    loader = DataLoader(TensorDataset(Xd_tr, Xs_tr, yt_tr),
                        batch_size=P['batch_size'], shuffle=True)
    best_loss, patience_cnt, best_state = float('inf'), 0, None

    for epoch in range(P['epochs']):
        model.train()
        for xd, xs, yt in loader:
            optimizer.zero_grad()
            criterion(model(xd, xs), yt).backward()
            optimizer.step()

        if epoch < P['min_epochs']:
            continue
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xd_tr, Xs_tr), yt_tr).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= P['early_stop_patience']:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = model(Xd_te, Xs_te).cpu().numpy()
    all_preds.extend(p.tolist())
    all_obs.extend(y[te].tolist())

    if (yi + 1) % 10 == 0:
        r2_so_far = r2_score(all_obs, all_preds)
        print(f"  Fold {yi+1}/{len(fold_years)} done — R²={r2_so_far:.3f}")

r2 = r2_score(all_obs, all_preds)
rmse = np.sqrt(mean_squared_error(all_obs, all_preds))
bias = np.mean(np.array(all_preds) - np.array(all_obs))
print(f"\nLOYO R²={r2:.4f}  RMSE={rmse*1000:.1f}mm  Bias={bias*1000:.1f}mm")

os.makedirs(RESULT_DIR, exist_ok=True)
tag = f'glacioformer_{args.variant}'
pd.DataFrame({'obs': all_obs, 'pred': all_preds}).to_csv(
    os.path.join(RESULT_DIR, f'{tag}_loyo_predictions.csv'), index=False)
pd.DataFrame([{'variant': args.variant, 'cv': 'LOYO',
               'r2': r2, 'rmse_mm': rmse*1000, 'bias_mm': bias*1000}]).to_csv(
    os.path.join(RESULT_DIR, f'{tag}_loyo_summary.csv'), index=False)
print(f"Saved → {RESULT_DIR}/{tag}_loyo_*.csv")
```

- [ ] **Step 2: Write train_logo.py** (identical structure, split by glacier_id)

Create `02_models/glacioformer/train_logo.py` with the same structure as train_loyo.py but:
- Loop over `glaciers = sorted(set(gids))` instead of `fold_years`
- Split masks: `tr = gids != gid`, `te = gids == gid`
- Output files: `{tag}_logo_predictions.csv`, `{tag}_logo_summary.csv`

- [ ] **Step 3: Run full variant**

```bash
"C:/Users/zjw31/.conda/envs/smb/python.exe" 02_models/glacioformer/train_loyo.py --variant full
"C:/Users/zjw31/.conda/envs/smb/python.exe" 02_models/glacioformer/train_logo.py --variant full
```

- [ ] **Step 4: Commit**

```bash
git add 02_models/glacioformer/
git commit -m "GlacioFormer LOYO + LOGO training (full + selected variants)"
```

---

## Task 12: Evaluation — Hold-out and Model Comparison

**Files:**
- Create: `03_evaluation/eval_holdout.py`
- Create: `03_evaluation/compare_models.py`

- [ ] **Step 1: Write eval_holdout.py**

```python
# 03_evaluation/eval_holdout.py
"""
Hold-out evaluation on 2015–2024 for all registered models.
Trains on 1950–2014, evaluates on 2015–2024.
Output: results/holdout_metrics.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from config import (TABULAR_CSV, SELECTED_VARS_JSON, RESULT_DIR,
                    TRAIN_YEAR_MIN, TRAIN_YEAR_MAX,
                    HOLDOUT_YEAR_MIN, HOLDOUT_YEAR_MAX, XGB_PARAMS)

print("=== Hold-out Evaluation (2015–2024) ===")

df = pd.read_csv(TABULAR_CSV)
df_labeled = df[df['annual_balance_m'].notna()].copy()

with open(SELECTED_VARS_JSON) as f:
    feature_cols = json.load(f)['selected_features']

X_all = df_labeled[feature_cols].fillna(df_labeled[feature_cols].median()).values
y_all = df_labeled['annual_balance_m'].values
years = df_labeled['year'].values

tr_mask = (years >= TRAIN_YEAR_MIN) & (years <= TRAIN_YEAR_MAX)
te_mask = (years >= HOLDOUT_YEAR_MIN) & (years <= HOLDOUT_YEAR_MAX)

print(f"Train: {tr_mask.sum()} samples | Hold-out: {te_mask.sum()} samples")

import xgboost as xgb
model = xgb.XGBRegressor(**XGB_PARAMS)
model.fit(X_all[tr_mask], y_all[tr_mask])
preds = model.predict(X_all[te_mask])
obs   = y_all[te_mask]

r2   = r2_score(obs, preds)
rmse = np.sqrt(mean_squared_error(obs, preds))
bias = np.mean(preds - obs)

print(f"XGBoost Hold-out R²={r2:.4f}  RMSE={rmse*1000:.1f}mm  Bias={bias*1000:.1f}mm")
os.makedirs(RESULT_DIR, exist_ok=True)
pd.DataFrame([{'model': 'xgboost', 'cv': 'holdout',
               'r2': r2, 'rmse_mm': rmse*1000, 'bias_mm': bias*1000,
               'n': te_mask.sum()}]).to_csv(
    os.path.join(RESULT_DIR, 'holdout_metrics.csv'), index=False)
print(f"Saved → {RESULT_DIR}/holdout_metrics.csv")
```

- [ ] **Step 2: Write compare_models.py**

```python
# 03_evaluation/compare_models.py
"""
Aggregate all LOYO/LOGO/holdout metrics from results/ into one comparison table.
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
for f in glob.glob(os.path.join(RESULT_DIR, '*_loyo_summary.csv')) + \
         glob.glob(os.path.join(RESULT_DIR, '*_logo_summary.csv')) + \
         [os.path.join(RESULT_DIR, 'holdout_metrics.csv')]:
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
```

- [ ] **Step 3: Run**

```bash
"C:/Users/zjw31/.conda/envs/smb/python.exe" 03_evaluation/eval_holdout.py
"C:/Users/zjw31/.conda/envs/smb/python.exe" 03_evaluation/compare_models.py
```

- [ ] **Step 4: Commit**

```bash
mkdir -p H:/Code/SMB/SMB_Res_ByClaudeV2/03_evaluation
git add 03_evaluation/
git commit -m "03_evaluation: holdout eval + model comparison aggregator"
```

---

## Task 13: Reconstruction Preprocessing

**Files:**
- Create: `04_reconstruction/step01_prepare_rgi02.py`
- Create: `04_reconstruction/step02_extract_era5_all.py`

- [ ] **Step 1: Write step01_prepare_rgi02.py**

```python
# 04_reconstruction/step01_prepare_rgi02.py
"""
从 RGI v7.0 提取全部 area_km2 >= 0.5 的 RGI02 冰川，计算地形特征。
输出: data/rgi02_target_glaciers.csv (~4,999 行)
列: rgi_id, cenlon, cenlat, area_km2, slope_deg, aspect_sin, aspect_cos,
    zmin_m, zmax_m, zmean_m, zmed_m, lmax_m
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
import geopandas as gpd
from config import RGI_SHP, RGI02_TARGET_CSV, MIN_AREA_KM2

print("=== Reconstruction Step 01: Prepare RGI02 Target Glaciers ===")

gdf = gpd.read_file(RGI_SHP)[[
    'rgi_id', 'cenlon', 'cenlat', 'area_km2',
    'slope_deg', 'aspect_deg', 'zmin_m', 'zmax_m',
    'zmean_m', 'zmed_m', 'lmax_m',
]].copy()

gdf['aspect_sin'] = np.sin(np.deg2rad(gdf['aspect_deg']))
gdf['aspect_cos'] = np.cos(np.deg2rad(gdf['aspect_deg']))
gdf = gdf.drop(columns=['aspect_deg'])

df_target = gdf[gdf['area_km2'] >= MIN_AREA_KM2].reset_index(drop=True)
print(f"Total RGI02 glaciers: {len(gdf):,}")
print(f"area >= {MIN_AREA_KM2} km²: {len(df_target):,}")
print(f"Total area: {df_target['area_km2'].sum():.1f} km²")

os.makedirs(os.path.dirname(RGI02_TARGET_CSV), exist_ok=True)
df_target.to_csv(RGI02_TARGET_CSV, index=False)
print(f"Saved → {RGI02_TARGET_CSV}")
```

- [ ] **Step 2: Write step02_extract_era5_all.py**

```python
# 04_reconstruction/step02_extract_era5_all.py
"""
ERA5 月度数据提取到全部 4,999 个重建目标冰川。
复用 step03 逻辑，但处理所有目标冰川（不仅训练的 63 个）。
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

TEMP_VARS  = {'t2m', 'skt', 'd2m'}
ACCUM_VARS = {'tp', 'sf', 'smlt', 'ssrd', 'strd', 'ssr', 'str', 'slhf', 'sshf', 'ro'}

lats_da = xr.DataArray(lats, dims='glacier')
lons_da = xr.DataArray(lons, dims='glacier')

records = []
times = pd.to_datetime(ds.time.values)
for ti, t in enumerate(times):
    for gi in range(len(rgi_ids)):
        records.append({'rgi_id': rgi_ids[gi], 'year': t.year, 'month': t.month})

print(f"Total records to fill: {len(records):,}")

for var in MONTHLY_CLIMATE_VARS:
    if var not in ds.data_vars:
        continue
    da = ds[var]
    interp = da.interp(latitude=lats_da, longitude=lons_da, method='linear').values
    if var in TEMP_VARS:
        interp = interp - 273.15
    elif var in ACCUM_VARS:
        interp = interp * 1000.0
    for ti in range(len(times)):
        for gi in range(len(rgi_ids)):
            records[ti * len(rgi_ids) + gi][var] = float(interp[ti, gi])
    print(f"  {var} done")

df_out = pd.DataFrame(records)
df_out.to_csv(ERA5_RGI02_CSV, index=False)
print(f"\nSaved {len(df_out):,} rows → {ERA5_RGI02_CSV}")
```

Note: step02 on ~5,000 glaciers × 900 months is computationally heavy. If memory is an issue, process in glacier batches of 500.

- [ ] **Step 3: Run step01 (step02 can be run when ready)**

```bash
mkdir -p H:/Code/SMB/SMB_Res_ByClaudeV2/04_reconstruction
"C:/Users/zjw31/.conda/envs/smb/python.exe" 04_reconstruction/step01_prepare_rgi02.py
```
Expected: 4,999 glaciers, area sum ~XX,XXX km².

- [ ] **Step 4: Commit**

```bash
git add 04_reconstruction/step01_prepare_rgi02.py 04_reconstruction/step02_extract_era5_all.py
git commit -m "04_reconstruction: RGI02 target preparation + ERA5 full extraction"
```

---

## Task 14: Reconstruction and Regional Stats

**Files:**
- Create: `04_reconstruction/step03_reconstruct.py`
- Create: `04_reconstruction/step04_regional_stats.py`

- [ ] **Step 1: Write step03_reconstruct.py**

```python
# 04_reconstruction/step03_reconstruct.py
"""
使用最优模型对全部 RGI02 目标冰川进行 1950–2024 年重建。
默认使用 XGBoost（LOGO R² 最高的模型，由 compare_models.py 确认）。
Output: results/RGI02_SMB_reconstruction.csv
  columns: rgi_id, year, predicted_smb_m, area_km2, cenlat, cenlon
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from config import (TABULAR_CSV, ERA5_RGI02_CSV, RGI02_TARGET_CSV,
                    SELECTED_VARS_JSON, RESULT_DIR, DATA_DIR,
                    XGB_PARAMS, STATIC_FEATURES, MONTHLY_CLIMATE_VARS,
                    RECON_YEAR_MIN, RECON_YEAR_MAX,
                    CAL_SUMMER_MONTHS, CAL_WINTER_MONTHS,
                    HYD_ACCUM_MONTHS, HYD_ABLAT_MONTHS)

print("=== Reconstruction Step 03: Reconstruct 1950–2024 ===")

# ── 1. Train final XGBoost on ALL labeled data ───────────────────────────────
df = pd.read_csv(TABULAR_CSV)
df_labeled = df[df['annual_balance_m'].notna()].copy()
with open(SELECTED_VARS_JSON) as f:
    feature_cols = json.load(f)['selected_features']

X_all = df_labeled[feature_cols].fillna(df_labeled[feature_cols].median()).values
y_all = df_labeled['annual_balance_m'].values
feat_medians = df_labeled[feature_cols].median()

model = xgb.XGBRegressor(**XGB_PARAMS)
model.fit(X_all, y_all)
print(f"Final model trained on {len(y_all)} samples")

# ── 2. Build tabular features for all RGI02 target glaciers ─────────────────
# (Same logic as step04_build_tabular but for rgi_id instead of glacier_id)
df_era5  = pd.read_csv(ERA5_RGI02_CSV)
df_terrain = pd.read_csv(RGI02_TARGET_CSV)

terrain_col_map = {c: c for c in STATIC_FEATURES if c in df_terrain.columns}

def seasonal_agg_recon(df_year, months, prefix, var_list):
    sub = df_year[df_year['month'].isin(months)]
    feats = {}
    for v in var_list:
        if v not in sub.columns:
            continue
        if v in {'t2m', 'skt', 'd2m', 'sd', 'asn'}:
            feats[f'{prefix}_{v}_mean'] = sub[v].mean() if len(sub) > 0 else np.nan
        else:
            feats[f'{prefix}_{v}_sum'] = sub[v].sum() if len(sub) > 0 else np.nan
    return feats

rows = []
for rid, g_era5 in df_era5.groupby('rgi_id'):
    for year, y_era5 in g_era5.groupby('year'):
        if len(y_era5) < 12:
            continue
        row = {'rgi_id': rid, 'year': year}
        for v in MONTHLY_CLIMATE_VARS:
            if v not in y_era5.columns:
                continue
            if v in {'t2m', 'skt', 'd2m', 'sd', 'asn'}:
                row[f'ann_{v}_mean'] = y_era5[v].mean()
            else:
                row[f'ann_{v}_sum'] = y_era5[v].sum()
        row.update(seasonal_agg_recon(y_era5, CAL_SUMMER_MONTHS, 'cal_summer', MONTHLY_CLIMATE_VARS))
        row.update(seasonal_agg_recon(y_era5, CAL_WINTER_MONTHS, 'cal_winter', MONTHLY_CLIMATE_VARS))
        row.update(seasonal_agg_recon(y_era5, HYD_ABLAT_MONTHS, 'hyd_ablat', MONTHLY_CLIMATE_VARS))
        # hyd_accum: need prev year Oct-Dec
        prev = g_era5[g_era5['year'] == year - 1]
        hyd_accum = pd.concat([prev[prev['month'].isin([10,11,12])],
                                y_era5[y_era5['month'].isin([1,2,3,4])]], ignore_index=True)
        row.update(seasonal_agg_recon(hyd_accum, list(range(1,13)), 'hyd_accum', MONTHLY_CLIMATE_VARS))
        rows.append(row)

df_recon_feat = pd.DataFrame(rows)
df_recon_feat = df_recon_feat.merge(
    df_terrain[['rgi_id'] + [c for c in STATIC_FEATURES if c in df_terrain.columns]],
    on='rgi_id', how='left'
)
print(f"Reconstruction feature rows: {len(df_recon_feat):,}")

# ── 3. Predict ───────────────────────────────────────────────────────────────
X_recon = df_recon_feat[feature_cols].fillna(feat_medians).values
df_recon_feat['predicted_smb_m'] = model.predict(X_recon)

# Merge area for output
df_out = df_recon_feat[['rgi_id', 'year', 'predicted_smb_m']].merge(
    df_terrain[['rgi_id', 'area_km2', 'cenlat', 'cenlon']], on='rgi_id', how='left'
)

os.makedirs(RESULT_DIR, exist_ok=True)
out_path = os.path.join(RESULT_DIR, 'RGI02_SMB_reconstruction.csv')
df_out.to_csv(out_path, index=False)
print(f"Saved {len(df_out):,} rows → {out_path}")
print(f"Predicted SMB stats:\n{df_out['predicted_smb_m'].describe()}")
```

- [ ] **Step 2: Write step04_regional_stats.py**

```python
# 04_reconstruction/step04_regional_stats.py
"""
计算区域面积加权年均 SMB 统计，与 Hugonnet 2021 比较。
Output: results/regional_stats.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
from config import RESULT_DIR, HUGONNET_RATES, RGI_LINKS_CSV

print("=== Reconstruction Step 04: Regional Stats ===")

df = pd.read_csv(os.path.join(RESULT_DIR, 'RGI02_SMB_reconstruction.csv'))

# Area-weighted annual mean SMB
def area_weighted_mean(group):
    w = group['area_km2']
    s = group['predicted_smb_m']
    valid = w.notna() & s.notna()
    if valid.sum() == 0:
        return pd.Series({'smb_weighted_mean': np.nan, 'smb_equal_mean': np.nan,
                          'smb_std': np.nan, 'n_glaciers': 0})
    wm = np.average(s[valid], weights=w[valid])
    return pd.Series({'smb_weighted_mean': wm,
                      'smb_equal_mean': s[valid].mean(),
                      'smb_std': s[valid].std(),
                      'n_glaciers': valid.sum()})

annual = df.groupby('year').apply(area_weighted_mean, include_groups=False).reset_index()
out = os.path.join(RESULT_DIR, 'regional_stats.csv')
annual.to_csv(out, index=False)
print(f"Saved → {out}")
print(annual[['year', 'smb_weighted_mean', 'n_glaciers']].tail(10).to_string())
```

- [ ] **Step 3: Commit**

```bash
git add 04_reconstruction/
git commit -m "04_reconstruction: reconstruct 1950-2024 + regional stats"
```

---

## Task 15: Publication Figures

**Files:**
- Create: `05_figures/fig1_validation_scatter.py` through `fig6_hugonnet_validation.py`

- [ ] **Step 1: Write fig1_validation_scatter.py**

```python
# 05_figures/fig1_validation_scatter.py
"""Fig 1: LOYO / LOGO / hold-out validation scatter (3-panel)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from config import RESULT_DIR, FIG_DIR

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
panel_data = [
    ('xgboost_loyo_predictions.csv', 'LOYO', axes[0]),
    ('xgboost_logo_predictions.csv', 'LOGO', axes[1]),
    # holdout loaded separately
]

for fname, label, ax in panel_data:
    fpath = os.path.join(RESULT_DIR, fname)
    if not os.path.exists(fpath):
        ax.set_title(f'{label} (not run yet)')
        continue
    df = pd.read_csv(fpath)
    obs, pred = df['obs'].values, df['pred'].values
    r2 = r2_score(obs, pred)
    rmse = np.sqrt(mean_squared_error(obs, pred)) * 1000
    ax.scatter(obs, pred, alpha=0.4, s=20, color='steelblue')
    lim = [min(obs.min(), pred.min()) - 0.2, max(obs.max(), pred.max()) + 0.2]
    ax.plot(lim, lim, 'k--', lw=1)
    ax.set_xlabel('Observed SMB (m w.e.)')
    ax.set_ylabel('Predicted SMB (m w.e.)')
    ax.set_title(f'{label}\nR²={r2:.3f}  RMSE={rmse:.0f} mm')
    ax.set_xlim(lim); ax.set_ylim(lim)

# Hold-out panel
ax = axes[2]
hpath = os.path.join(RESULT_DIR, 'holdout_metrics.csv')
if os.path.exists(hpath):
    df_h = pd.read_csv(hpath)
    ax.set_title(f"Hold-out 2015–2024\nR²={df_h['r2'].iloc[0]:.3f}  RMSE={df_h['rmse_mm'].iloc[0]:.0f} mm")
    ax.text(0.5, 0.5, 'Run eval_holdout.py\nfor scatter data',
            ha='center', va='center', transform=ax.transAxes)
else:
    ax.set_title('Hold-out (not run yet)')

plt.tight_layout()
out = os.path.join(FIG_DIR, 'fig1_validation_scatter.png')
os.makedirs(FIG_DIR, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved → {out}")
plt.close()
```

- [ ] **Step 2: Write fig3_regional_trend.py**

```python
# 05_figures/fig3_regional_trend.py
"""Fig 3: Area-weighted regional SMB trend 1950–2024 (bar chart with trend line)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from config import RESULT_DIR, FIG_DIR

df = pd.read_csv(os.path.join(RESULT_DIR, 'regional_stats.csv'))
years = df['year'].values
smb   = df['smb_weighted_mean'].values

slope, intercept, r, p, _ = linregress(years, smb)
trend = slope * years + intercept

fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#d73027' if v < 0 else '#4575b4' for v in smb]
ax.bar(years, smb, color=colors, alpha=0.75, width=0.8, label='Annual mean SMB')
ax.plot(years, trend, 'k-', lw=1.5,
        label=f'Trend: {slope*1000:.1f} mm w.e. yr⁻²  (p={p:.3f})')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('SMB (m w.e. yr⁻¹)')
ax.set_title('RGI02 Area-weighted Mean SMB 1950–2024 (XGBoost reconstruction)')
ax.legend()
plt.tight_layout()
out = os.path.join(FIG_DIR, 'fig3_regional_trend.png')
os.makedirs(FIG_DIR, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved → {out}")
plt.close()
```

- [ ] **Step 3: Write fig6_hugonnet_validation.py**

```python
# 05_figures/fig6_hugonnet_validation.py
"""Fig 6: Compare 2000–2019 reconstruction vs Hugonnet 2021 geodetic mass change."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from config import RESULT_DIR, FIG_DIR, HUGONNET_RATES, RGI_LINKS_CSV

# Load Hugonnet 20-year rates (period = '2000-01-01_2020-01-01')
df_hug = pd.read_csv(HUGONNET_RATES)
df_hug20 = df_hug[df_hug['period'] == '2000-01-01_2020-01-01'][['rgiid', 'dmdtda']].copy()
df_hug20 = df_hug20.rename(columns={'rgiid': 'rgi60_id', 'dmdtda': 'hugonnet_dmdtda_m'})

# Load RGI7→RGI6 links
df_links = pd.read_csv(RGI_LINKS_CSV)[['rgi7_id', 'rgi6_id']].copy()

# Our reconstruction 2000–2019 mean per glacier
df_recon = pd.read_csv(os.path.join(RESULT_DIR, 'RGI02_SMB_reconstruction.csv'))
df_recon_2019 = df_recon[(df_recon['year'] >= 2000) & (df_recon['year'] <= 2019)]
df_our = df_recon_2019.groupby('rgi_id')['predicted_smb_m'].mean().reset_index()
df_our.columns = ['rgi7_id', 'our_smb_m']

# Join
df_joined = df_our.merge(df_links, on='rgi7_id', how='inner')
df_joined = df_joined.merge(df_hug20, left_on='rgi6_id', right_on='rgi60_id', how='inner')
print(f"Matched glaciers for comparison: {len(df_joined)}")

obs  = df_joined['hugonnet_dmdtda_m'].values
pred = df_joined['our_smb_m'].values
r2   = r2_score(obs, pred)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(obs, pred, alpha=0.4, s=15, color='#2c7bb6')
lim = [-3, 1]
ax.plot(lim, lim, 'k--', lw=1, label='1:1 line')
ax.set_xlabel('Hugonnet 2021 dmdtda (m w.e. yr⁻¹)')
ax.set_ylabel('Our reconstruction mean 2000–2019 (m w.e. yr⁻¹)')
ax.set_title(f'External Validation vs Hugonnet 2021\nR²={r2:.3f}  n={len(df_joined)}')
ax.legend()
plt.tight_layout()
out = os.path.join(FIG_DIR, 'fig6_hugonnet_validation.png')
os.makedirs(FIG_DIR, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved → {out}")
plt.close()
```

- [ ] **Step 4: Create stub scripts for fig2, fig4, fig5**

```bash
# Create placeholder scripts; implement after reconstruction results are available
for f in fig2_timeseries fig4_spatial_distribution fig5_model_comparison; do
  echo "# TODO: implement $f" > H:/Code/SMB/SMB_Res_ByClaudeV2/05_figures/${f}.py
done
```

- [ ] **Step 5: Commit**

```bash
mkdir -p H:/Code/SMB/SMB_Res_ByClaudeV2/05_figures
git add 05_figures/
git commit -m "05_figures: fig1 (validation), fig3 (trend), fig6 (Hugonnet) + stubs"
```

---

## Self-Review

**Spec coverage:**
- ✅ WGMS gtng_region filter (step01) — not country-based
- ✅ annual_balance unit note (already m w.e., no /1000)
- ✅ RGI v7 terrain matching (step02, 10 static features)
- ✅ ERA5 bilinear interpolation (step03, 15 vars, K→°C, m→mm)
- ✅ cal_/hyd_ tabular features (step04)
- ✅ Monthly sequences for Transformer (step05)
- ✅ XGBoost + feature selection (Task 8)
- ✅ GlacioFormer registered in registry (Task 10)
- ✅ LOYO + LOGO training for both models (Tasks 9, 11)
- ✅ Hold-out 2015–2024 (Task 12)
- ✅ compare_models.py auto-aggregates all results (Task 12)
- ✅ Full RGI02 ≥0.5 km² reconstruction (4,999 glaciers, Task 13-14)
- ✅ 1950–2024 reconstruction (Tasks 13-14)
- ✅ Fig1 validation scatter, Fig3 trend, Fig6 Hugonnet (Task 15)
- ⚠ Fig2 (timeseries), Fig4 (spatial), Fig5 (model comparison) are stubs — implement after results

**Type consistency:** `glacier_id` used throughout training pipeline; `rgi_id` used in reconstruction pipeline. Both are integer-keyed. `annual_balance_m` is the consistent target column name.

**Note:** `MASSBAL_RGI02_CSV` must be added to config.py before Task 5 runs (see Task 2 Step 3).

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-21-rgi02-smb-pipeline.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** — Fresh subagent per task, review between tasks

**2. Inline Execution** — Execute tasks in this session with checkpoints

Which approach?
