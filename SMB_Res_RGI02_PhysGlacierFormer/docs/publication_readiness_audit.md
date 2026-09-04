# Publication-Readiness Audit

## Executive Assessment

The legacy workflow was not publication-ready because WGMS-RGI matching, ERA5-Land accumulation units, annual time windows, and fold preprocessing were inconsistent. The corrected `phys_v2` workflow substantially improves both validity and accuracy. The strongest current annual model is XGBoost with monthly climate and glacier-month anomaly features; the Transformer is not the best model for this 993-sample tabular-sequence problem.

## Verified Data Corrections

- **RGI linkage:** 54 WGMS series use official RGI6 IDs mapped to RGI v7 by polygon overlap; nine require nearest-outline fallback. Fourteen official mappings differed from the old centroid-nearest result. South Cascade changed from an implausible 0.012 km2 match to a 2.924 km2 glacier.
- **Time support:** records with credible dates use the 12 months ending at the reported observation month. Missing or invalid dates default to September. The corrected dataset contains 993 samples, 59 WGMS series, 58 unique RGI glaciers, and 70 years.
- **ERA5-Land units:** `moda` hydrological accumulations are integrated from m d-1 to mm month-1; energy accumulations are converted from J m-2 d-1 to W m-2. The implementation follows [ECMWF ERA5-Land documentation](https://confluence.ecmwf.int/pages/viewpage.action?pageId=505384848).
- **Leakage control:** scaling and imputation are fitted inside each fold; LOGO groups by RGI v7 glacier, not WGMS series; early stopping uses grouped inner validation.
- **Scientific invariants:** seven automated tests verify month completeness, plausible units, seasonal-label conservation, corrected South Cascade mapping, model output conservation, final-model feature-schema consistency, and the complete 18,730-glacier reconstruction grid.

## Current Results

| Model / representation | LOGO R2 | LOGO RMSE | LOYO R2 | LOYO RMSE |
|---|---:|---:|---:|---:|
| Legacy Hypsometry GlacierFormer QC | 0.432 | 787 mm | 0.323 | 859 mm |
| Corrected mean-only baseline | -0.007 | 1050 mm | -0.024 | 1059 mm |
| Corrected Ridge baseline | 0.539 | 711 mm | 0.488 | 749 mm |
| Corrected Random Forest baseline | 0.634 | 633 mm | 0.472 | 761 mm |
| Corrected XGBoost, raw climate only | 0.433 | 788 mm | 0.446 | 779 mm |
| Corrected XGBoost, climate + anomalies | 0.622 | 643 mm | 0.579 | 679 mm |
| XGBoost + explicit physical indices | 0.626 | 640 mm | 0.582 | 676 mm |
| Nested-tuned XGBoost | 0.624 | 642 mm | 0.574 | 683 mm |
| Compact PhysGlacierFormer, seasonal weight 2 | 0.562 | 692 mm | 0.533 | 715 mm |

The anomaly decomposition is the defensible main gain: relative to corrected raw climate, it adds about 0.19 LOGO R2 and 0.13 LOYO R2. Random Forest is marginally better than the fixed XGBoost profile under LOGO but substantially worse under LOYO, so XGBoost is selected for balanced spatial-temporal robustness rather than universal dominance. Explicit physical indices provide only a small gain whose paired bootstrap interval overlaps zero for LOGO and LOYO; it should be an ablation, not the headline contribution. Nested XGBoost results are the unbiased hyperparameter-selection estimate; fixed-profile scores are sensitivity and selected-model results.

Robustness tests for the fixed regularized XGBoost yield 50 km buffered LOGO R2=0.596, +/-1-year buffered LOYO R2=0.572, four-observed-subregion LOSO R2=0.507, and rolling-origin forward-validation R2=0.553 for 1980-2023. These drops are moderate but show that ordinary LOGO can benefit from nearby glaciers and ordinary LOYO is temporal interpolation, not future forecasting.

## Negative Results That Must Be Reported Honestly

- Fixed lapse-rate correction using pressure-derived ERA5 grid elevation reduced LOGO and LOYO skill. A proper static ERA5 geopotential field and calibrated precipitation gradient would be required before revisiting this route.
- Glacier-balanced and uncertainty-weighted objectives did not improve overall RMSE.
- Coarse seasonal aggregation underperformed monthly features.
- Hugonnet weak-label pretraining caused unstable or negative transfer. Geodetic information is better used as a long-term post-hoc constraint.
- PhysGlacierFormer seasonal heads initially exhibited strong winter and summer error cancellation. Stronger seasonal supervision improved seasonal R2 but did not beat XGBoost annually.
- Target-magnitude weighting reduced tail errors slightly but worsened overall LOGO skill (R2 0.624 to 0.613), so it is not used in the final model.
- Removing ERA5-Land snow albedo slightly improved LOGO but reduced forward-validation R2 from 0.553 to 0.505. The final model retains it as a reanalysis proxy, not as observed glacier-surface albedo.

Final-model SHAP magnitudes assign 64.4% of total absolute contribution to glacier-month climate anomalies. July ERA5-Land snow-albedo anomaly is the largest individual monthly contribution and is strongly associated with annual SMB, but its sign is not straightforward physically. It is therefore treated as a modeled snow-state proxy and a priority for MODIS/Sentinel-2 verification, not as causal evidence of glacier-albedo feedback.

## Hugonnet Calibration Evidence

Calibration residuals cannot be reused as independent validation. With the corrected all-glacier reconstruction, a temporal-transfer test estimates offsets from 2000-2010 and evaluates 2010-2020 for 9,869 glaciers. Early and late residuals correlate at r=0.293. With a +/-1 m w.e. yr-1 clip, shrink=0.5 reduces later-period RMSE from 0.491 to 0.398 m w.e. yr-1, whereas full correction worsens it to 0.507. Because the later period was also used to compare shrink factors, this is sensitivity evidence for conservative shrinkage rather than an unbiased final performance estimate.

The final reconstruction contains exactly 1,386,020 glacier-years (18,730 glaciers x 74 years, 1951-2024), no duplicate keys, and no non-finite predictions. The raw regional series has a 1980-2024 trend of -0.141 m w.e. decade-1 (p=0.023). Against Malles & Marzeion over 1951-2018, the raw reconstruction gives r=0.822 and RMSE=4.55 Gt; conservative calibration keeps the same correlation and RMSE=4.54 Gt while reversing the mean bias from -0.66 to +0.57 Gt. A complementary specific-balance comparison, which reduces but does not eliminate the mismatch between fixed RGI v7 area and evolving Malles area, gives raw/calibrated r=0.851 and RMSE=0.279/0.271 m w.e. yr-1. Against Zemp et al. over 1953-2016, raw RMSE is 0.300 m w.e. yr-1 and calibrated RMSE is 0.327. Zemp shares WGMS source information and is not independent. These comparisons require publishing both raw and calibrated products rather than labeling the calibrated series universally superior.

Cluster-excluded residual calibration gives empirical 90% coverage of 0.899 for both LOGO and LOYO. The corresponding global half-widths are 1.03 and 1.07 m w.e. yr-1. These are validation-distribution intervals and do not guarantee coverage for inventory glaciers outside the WGMS covariate domain.

## Literature Alignment

- [Sjursen et al. (2025)](https://tc.copernicus.org/articles/19/5801/2025/) show that XGBoost is competitive for medium-sized glacier tabular data and stress non-random validation and reanalysis-to-glacier elevation differences.
- [van der Meer et al. (2025)](https://tc.copernicus.org/articles/19/805/2025/) find that parsimonious temperature and precipitation predictors can outperform larger sets and document failures in extreme years.
- [Guidicelli et al. (2023)](https://tc.copernicus.org/articles/17/977/2023/) identify glacier-reanalysis elevation difference as important for winter-balance downscaling.
- [Draeger et al. (2026)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025JF008740) show for western Canada that precipitation correction, elevation gradients, MODIS-informed albedo, and glacier-specific geodetic calibration are high-value additions; they also demonstrate why annual agreement can hide opposing seasonal errors.
- [Hugonnet et al. (2021)](https://www.nature.com/articles/s41586-021-03436-z) provide remote-sensing geodetic mass-change constraints, not annual in-situ SMB observations.

## Remaining Publication Gates

1. Validate recent extremes and long-term regional totals against genuinely independent products. Malles & Marzeion is a model intercomparison; Zemp shares WGMS information.
2. Add MODIS albedo or Sentinel-2 snowline/snow-cover products before claiming physically resolved ablation. ERA5 snow albedo is not a substitute for glacier-surface albedo.
3. Obtain pressure-level temperature and precipitation-gradient information before retrying elevation downscaling; the pressure-derived fixed lapse-rate ablation failed.
4. State that fixed RGI v7 geometry yields reference-geometry SMB; it is not a coupled glacier-evolution or dynamic mass-change reconstruction.
5. Treat 2024 as provisional extrapolation because the latest WGMS training target is 2023 and available external regional products end earlier.

## Recommended Paper Framing

The strongest defensible paper is currently a regional machine-learning reconstruction built around rigorous data harmonization, climate climatology-anomaly decomposition, spatial and temporal robustness tests, and conservative geodetic calibration. Calling the work primarily a novel Transformer paper is not supported by the results.
