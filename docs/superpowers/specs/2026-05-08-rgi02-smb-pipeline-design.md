# RGI02 冰川物质平衡重建流程设计文档

**日期**: 2026-05-08  
**项目**: SMB_Res_ByClaudeV2  
**目标区域**: RGI Region 02 (Western Canada & USA)

---

## 1. 核心目标

- **精度提升**: 正确筛选 RGI02 冰川（排除 Alaska 等越界冰川），统一单位，规范验证
- **覆盖扩展**: 从 63 个有观测冰川 → 重建全部 area ≥ 0.5 km² 的 RGI02 冰川（~3,500 个）

---

## 2. 数据源

| 数据 | 路径 | 用途 |
|---|---|---|
| WGMS FoG 2025-02b | `H:/Code/SMB/WGMS/FoG_DataBase/DOI-WGMS-FoG-2025-02b/` | 训练目标（annual balance） |
| RGI v7.0 | `H:/Code/SMB/RGI/RGI2000-v7.0-G-02_western_canada_usa/` | 地形属性 + 空间验证 + 重建目标冰川 |
| ERA5-LAND | `H:/Code/SMB/ERA5-LAND/data_stream-moda.nc` | 月度气候强迫（1950–2025） |

---

## 3. 目录结构

```
SMB_Res_ByClaudeV2/
├── config.py
├── 01_preprocessing/
│   ├── step01_filter_wgms.py        # WGMS gtng_region筛选 + RGI v7 空间验证
│   ├── step02_match_rgi.py          # 地形属性匹配（最近邻 <5 km）
│   ├── step03_extract_era5.py       # ERA5 双线插值到冰川质心
│   ├── step04_build_tabular.py      # 构建 tabular 特征（cal_ + hyd_ 双套）
│   └── step05_build_sequences.py    # 构建 Transformer 月度序列
├── 02_models/
│   ├── registry.py                  # 可扩展模型注册表
│   ├── base_model.py                # 统一接口
│   ├── xgboost/
│   │   ├── model.py
│   │   ├── feature_selection.py     # XGBoost importance + RFE → selected_vars.json
│   │   ├── train_loyo.py
│   │   └── train_logo.py
│   └── glacioformer/
│       ├── model.py                 # v1_transformer
│       ├── train_loyo.py            # --variant selected|full
│       └── train_logo.py
├── 03_evaluation/
│   ├── eval_holdout.py
│   └── compare_models.py            # 自动遍历 registry，输出对比报告
├── 04_reconstruction/
│   ├── step01_prepare_rgi02.py      # area ≥ 0.5 km² 过滤
│   ├── step02_extract_era5_all.py
│   ├── step03_reconstruct.py
│   └── step04_regional_stats.py
└── 05_figures/
    ├── fig1_validation_scatter.py
    ├── fig2_timeseries.py
    ├── fig3_regional_trend.py
    ├── fig4_spatial_distribution.py
    ├── fig5_model_comparison.py
    └── fig6_hugonnet_validation.py
```

---

## 4. 数据处理规范

### 4.1 RGI02 冰川筛选（训练集）
1. `WGMS glacier.csv` → `gtng_region == '02_western_canada_usa'` → 63 个冰川
2. 空间验证：冰川质心与 RGI v7 polygon union（5 km buffer，UTM 10N）做相交检验
3. 输出：`data/training_glaciers.csv`（glacier_id, latitude, longitude, rgi60_id）

### 4.2 地形属性匹配
- 方法：KD-tree 最近邻，距离阈值 5 km
- 来源：RGI v7 字段：`slope_deg, aspect_deg, zmin_m, zmax_m, zmean_m, zmed_m, area_km2, lmax_m`
- aspect 变换：`aspect_sin = sin(aspect_deg × π/180)`，`aspect_cos = cos(aspect_deg × π/180)`
- 输出：`data/training_glaciers_terrain.csv`

### 4.3 ERA5 提取
- 方法：双线性插值到冰川质心坐标
- 单位转换：T(K) → °C（-273.15）；累积量(m) → mm（×1000）
- 输出：`data/era5_monthly_training.csv`（glacier_id × year × month × 变量）

### 4.4 特征构建

**Tabular 特征（XGBoost 用）**

| 前缀 | 季节定义 | 示例特征 |
|---|---|---|
| `cal_` | 日历年（夏=JJA，冬=DJF） | `cal_summer_t2m_mean`, `cal_winter_sf_sum` |
| `hyd_` | 水文年（积累=Oct-Apr，消融=May-Sep） | `hyd_accum_sf_sum`, `hyd_ablat_smlt_sum` |

每行 = 一个冰川 × 一年，目标列 = `annual_balance_m`（m w.e.）

**月度序列（Transformer 用）**
- 形状：`(N, 12, n_vars)`，静态特征：`(N, n_static)`
- 归一化参数随 `.npz` 一起保存，供重建时使用

### 4.5 单位规范
- **全流程统一使用 m w.e.**
- WGMS `annual_balance`（mm w.e.）在 step01 读入时除以 1000
- 模型输出、评估指标、图表均为 m w.e. yr⁻¹

---

## 5. 模型方案

| 模型 key | 类型 | 输入特征 |
|---|---|---|
| `xgboost` | XGBoost | tabular（cal_ + hyd_ 全套） |
| `glacioformer_selected` | Transformer v1 | 月度序列（XGBoost/RFE 筛选后的变量） |
| `glacioformer_full` | Transformer v1 | 月度序列（全部月度 ERA5 变量） |

后续可在 `registry.py` 中一行注册新模型，`compare_models.py` 自动纳入对比。

---

## 6. 验证方案

```
训练+交叉验证期: 1950–2014（65年）
├── LOYO: 逐年留出，65折
└── LOGO: 逐冰川留出，63折

Hold-out 测试集: 2015–2024（训练期间完全不可见）
```

**最优模型选择标准**: 以 LOGO R² 为主指标（空间外推能力），LOYO 和 hold-out 为辅助验证。

---

## 7. 重建规范

- 目标冰川：RGI v7，`area_km2 >= 0.5`（~3,500 个）
- 重建时段：1950–2024（75年）
- 输出：`RGI02_SMB_reconstruction.csv`（glacier_id, year, predicted_smb_m, area_km2, lat, lon）
- 区域统计：面积加权年均 SMB，与 Hugonnet 2021 比对（2000–2019）

---

## 8. 论文图表

| 图 | 内容 |
|---|---|
| Fig 1 | LOYO / LOGO / hold-out 三重验证散点图 |
| Fig 2 | 典型冰川观测 vs 重建时间序列 |
| Fig 3 | 区域面积加权 SMB 趋势 1950–2024 |
| Fig 4 | 空间分布图（mean SMB + trend，双面板） |
| Fig 5 | 三模型性能对比 |
| Fig 6 | 与 Hugonnet 2021 外部验证 |
