# Formal Experiment Design

## Objective

Reconstruct annual glacier-wide SMB for all 18,730 RGI02 glaciers while separating interpolation skill, spatial transfer, temporal extrapolation, and external-product consistency. Historical pipelines remain unchanged.

## Primary Model

The publication-facing annual model is regularized XGBoost using observation-aligned monthly ERA5-Land climate, 1981-2010 glacier-month anomalies, static terrain and climate normals, and elevation-band hypsometry. PhysGlacierFormer is a seasonal-process benchmark, not the headline model.

## Required Comparisons

All methods use the corrected `phys_v2` dataset and identical outer folds:

1. Mean, Ridge, Random Forest, Extra Trees, LightGBM, XGBoost, and PhysGlacierFormer baselines.
2. Raw-climate, climatology-anomaly, physical-index, hypsometry, and snow-albedo ablations.
3. Nested LOGO/LOYO for unbiased hyperparameter-selection estimates.
4. 50 km buffered LOGO, +/-1-year buffered LOYO, rolling-origin validation, and LOSO stress tests.
5. Cluster-bootstrap confidence intervals and extreme-tail diagnostics.

## Reconstruction Evaluation

Publish raw and conservatively Hugonnet-calibrated products. Treat Hugonnet transfer as sensitivity evidence, Malles & Marzeion as external model intercomparison, and Zemp et al. as a non-independent consistency check. Report out-of-domain inventory coverage and fixed-reference-geometry limitations.

## Remaining Data Experiment

The next justified model-input experiment is observed glacier-surface information from MODIS albedo or Sentinel-2 snowline/snow cover. Additional attention modules are not justified without new information.
