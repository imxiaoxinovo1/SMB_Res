# RGI02 SMB Reconstruction

This project develops leakage-safe annual glacier-wide surface mass-balance (SMB) models for RGI Region 02. It is isolated from the legacy pipelines in `SMB_Res_ByClaudeV2` and does not overwrite their outputs.

## Current Scientific Position

The corrected XGBoost model is currently the primary annual-SMB model. On 993 WGMS glacier-years from 58 unique RGI v7 glaciers, it outperforms the compact PhysGlacierFormer under strict cross-validation. PhysGlacierFormer remains a deep-learning and seasonal-process benchmark; it is not presented as superior without evidence.

Key corrections in the `phys_v2` workflow are:

- official RGI6 polygon-overlap matching before nearest-outline fallback;
- observation-aligned 12-month mass-balance windows rather than calendar years;
- correct ERA5-Land `moda` accumulation units;
- 1981-2010 glacier-month climatology and climate anomalies;
- fold-local preprocessing and RGI v7 glacier grouping;
- LOGO, 50 km buffered LOGO, LOYO, +/-1-year buffered LOYO, rolling-origin validation, LOSO, nested tuning, and cluster-bootstrap intervals.

## Layout

```text
01_preprocessing/   WGMS-RGI matching and corrected ERA5/data construction
02_models/          GlacierFormer model definitions
03_training/        CV, nested tuning, final fitting, and evaluation
04_reconstruction/  Raw reconstruction and Hugonnet calibration checks
05_figures/         Figure-generation scripts
tests/              Fast scientific invariants
docs/               Experiment plan and publication-readiness audit
data/                Generated intermediates (git-ignored)
results/             Models and numerical outputs (git-ignored)
figures/             Rendered figures (git-ignored)
```

Scripts predating the `phys_v2` workflow, including preprocessing `step01`-`step10`,
legacy GlacierFormer trainers, and ensemble utilities, are retained only for experiment
provenance. They are not publication-reproduction entry points and must not be mixed
with corrected `phys_v2` metrics.

## Reproduce The Corrected Dataset

Run from this directory:

```powershell
& C:\Users\zjw31\.conda\envs\smb\python.exe 01_preprocessing\step11_match_wgms_rgi_v2.py
& C:\Users\zjw31\.conda\envs\smb\python.exe 01_preprocessing\step12_extract_era5_v2.py
& C:\Users\zjw31\.conda\envs\smb\python.exe 01_preprocessing\step13_build_phys_v2_dataset.py
& C:\Users\zjw31\.conda\envs\smb\python.exe -m unittest tests.test_phys_v2_pipeline -v
```

## Formal Validation

```powershell
& C:\Users\zjw31\.conda\envs\smb\python.exe 03_training\train_tree_v2_cv.py --cv logo --model xgboost --feature-set compact --representation monthly --static-set all --xgb-profile regularized
& C:\Users\zjw31\.conda\envs\smb\python.exe 03_training\train_tree_v2_cv.py --cv logo_buffered --model xgboost --feature-set compact --representation monthly --static-set all --xgb-profile regularized
& C:\Users\zjw31\.conda\envs\smb\python.exe 03_training\train_tree_v2_cv.py --cv loyo --model xgboost --feature-set compact --representation monthly --static-set all --xgb-profile regularized
& C:\Users\zjw31\.conda\envs\smb\python.exe 03_training\train_tree_v2_cv.py --cv loyo_buffered --model xgboost --feature-set compact --representation monthly --static-set all --xgb-profile regularized
& C:\Users\zjw31\.conda\envs\smb\python.exe 03_training\train_tree_v2_cv.py --cv forward --model xgboost --feature-set compact --representation monthly --static-set all --xgb-profile regularized
& C:\Users\zjw31\.conda\envs\smb\python.exe 03_training\train_tree_v2_cv.py --cv loso --model xgboost --feature-set compact --representation monthly --static-set all --xgb-profile regularized
& C:\Users\zjw31\.conda\envs\smb\python.exe 03_training\train_xgboost_v2_nested_cv.py --cv logo
& C:\Users\zjw31\.conda\envs\smb\python.exe 03_training\train_xgboost_v2_nested_cv.py --cv loyo
& C:\Users\zjw31\.conda\envs\smb\python.exe 03_training\evaluate_phys_v2_results.py
& C:\Users\zjw31\.conda\envs\smb\python.exe 03_training\calibrate_prediction_intervals.py
& C:\Users\zjw31\.conda\envs\smb\python.exe 05_figures\plot_phys_v2_validation_performance.py
& C:\Users\zjw31\.conda\envs\smb\python.exe 05_figures\plot_xgboost_feature_contributions.py
```

## Final Model And Reconstruction

```powershell
& C:\Users\zjw31\.conda\envs\smb\python.exe 03_training\train_xgboost_v2_final.py
& C:\Users\zjw31\.conda\envs\smb\python.exe 04_reconstruction\reconstruct_xgboost_v2.py
& C:\Users\zjw31\.conda\envs\smb\python.exe 04_reconstruction\evaluate_hugonnet_temporal_transfer.py --reconstruction results\reconstruction\RGI02_SMB_xgboost_v2_raw_all_glaciers.csv --output results\reconstruction\hugonnet_temporal_transfer_xgboost_v2.csv
& C:\Users\zjw31\.conda\envs\smb\python.exe 04_reconstruction\step03_calibrate_hugonnet_conservative.py --reconstruction results\reconstruction\RGI02_SMB_xgboost_v2_raw_all_glaciers.csv --output results\reconstruction\RGI02_SMB_xgboost_v2_hugonnet_conservative.csv --regional-output results\reconstruction\xgboost_v2_regional_timeseries.csv --qc-output results\reconstruction\xgboost_v2_hugonnet_calibration_qc.csv
& C:\Users\zjw31\.conda\envs\smb\python.exe 04_reconstruction\analyze_xgboost_v2_reconstruction.py
```

The corrected hydrological reconstruction begins in 1951 because ERA5-Land begins in January 1950 and the 1950 balance year requires October-December 1949 forcing. Outputs include all RGI02 glaciers plus a `recommended_area_domain` flag for the historically used >=0.5 km2 subset.

## Interpretation Rules

- Report out-of-fold metrics, never training-set metrics.
- Treat Hugonnet residual reduction after fitting as calibration consistency, not independent validation.
- Treat Malles & Marzeion as an external model comparison, not observations.
- Describe the reconstruction as reference-geometry SMB unless evolving glacier geometry is explicitly modeled.
- Do not claim seasonal skill from annual skill; winter and summer errors can cancel.
- Treat SHAP contributions as descriptive model interpretation, not causal attribution.
