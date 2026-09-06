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

The cross-fitted two-stage spatial-mean/anomaly experiment is retained as a negative-result ablation: it improves LOGO but reduces LOYO and is not eligible as the primary model without a nested selection result that reverses this trade-off.

Season-selective monotonic constraints are retained as a physics-guided sensitivity experiment. Their 1-4 mm changes in LOGO, LOYO, and forward RMSE are not statistically decisive, so they do not replace the unconstrained primary model.

Nested linear amplitude calibration is also rejected: it nearly removes LOYO mean bias but its RMSE improvement is small and its paired bootstrap interval crosses zero.

## Reconstruction Evaluation

Publish raw and conservatively Hugonnet-calibrated products. Treat Hugonnet transfer as sensitivity evidence, Malles & Marzeion as external model intercomparison, and Zemp et al. and GlaMBIE as shared-information consistency checks. Official GlaMBIE annual hydrological data now cover 2000-2023; the 2013-2022 altimetry component is also compared where it provides its own annual variability. Assess the amplitude deficit and recent loss underestimation without tuning to this benchmark. Report out-of-domain inventory coverage across all reconstruction years and fixed-reference-geometry limitations.

## Remaining Data Experiment

Before acquiring new predictors, test one predeclared training-loss sensitivity: regularized XGBoost with `--year-coherence-weight 0.5`. Add half the sample-count-weighted squared mean residual within training year/end-month groups containing at least three glaciers. This addresses coherent temporal errors without using GlaMBIE targets. Run LOYO and LOGO; continue to forward validation only if temporal gains do not trade against substantial spatial degradation. Evaluate paired uncertainty and within-glacier amplitude, not regional curve matching. This is exploratory model development, not an untouched outer-CV confirmation experiment.

Outcome: rejected. LOGO RMSE=642.5 mm versus 641.3 mm; LOYO RMSE=685.0 mm versus 677.0 mm. LOYO most-negative-decile RMSE also worsens from 1154.7 to 1181.1 mm. The fixed 0.5 weight fails the temporal-improvement criterion, so forward validation and further weight searches are not pursued.

Reproduce from the project directory:

```powershell
python 03_training/train_tree_v2_cv.py --cv loyo --model xgboost --xgb-profile regularized --year-coherence-weight 0.5
python 03_training/train_tree_v2_cv.py --cv logo --model xgboost --xgb-profile regularized --year-coherence-weight 0.5
python 03_training/evaluate_phys_v2_results.py
```

The next justified model-input experiment is observed glacier-surface information from MODIS albedo or Sentinel-2 snowline/snow cover. Additional attention modules are not justified without new information.
