# Publication-Readiness Audit

## Executive Assessment

The legacy workflow was not publication-ready because WGMS-RGI matching, ERA5-Land accumulation units, annual time windows, and fold preprocessing were inconsistent. The corrected `phys_v2` workflow substantially improves both validity and accuracy. The strongest current annual model is XGBoost with monthly climate and glacier-month anomaly features; the Transformer is not the best model for this 993-sample tabular-sequence problem.

## Verified Data Corrections

- **RGI linkage:** 54 WGMS series use official RGI6 IDs mapped to RGI v7 by polygon overlap; nine require nearest-outline fallback. Fourteen official mappings differed from the old centroid-nearest result. South Cascade changed from an implausible 0.012 km2 match to a 2.924 km2 glacier.
- **Time support:** records with credible dates use the 12 months ending at the reported observation month. Missing or invalid dates default to September. The corrected dataset contains 993 samples, 59 WGMS series, 58 unique RGI glaciers, and 70 years.
- **ERA5-Land units:** `moda` hydrological accumulations are integrated from m d-1 to mm month-1; energy accumulations are converted from J m-2 d-1 to W m-2. The implementation follows [ECMWF ERA5-Land documentation](https://confluence.ecmwf.int/pages/viewpage.action?pageId=505384848).
- **Leakage control:** scaling and imputation are fitted inside each fold; LOGO groups by RGI v7 glacier, not WGMS series; early stopping uses grouped inner validation.
- **Scientific invariants:** sixteen automated tests cover month completeness, plausible units, seasonal-label conservation, corrected glacier mapping, model/schema consistency, reconstruction completeness, weighted fitting, constraints, amplitude calibration, GlaMBIE time support, cumulative ensemble trajectories, bootstrap indexing, identical paired targets, undefined constant-target R2, and two-stage prediction invariance to outer-test targets with missing spatial features.

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

The anomaly decomposition is the defensible main gain; use matched terrain/hypsometry settings for the one-factor ablation. Random Forest is marginally better than the fixed XGBoost profile under LOGO but substantially worse under LOYO, so XGBoost is selected for balanced spatial-temporal robustness rather than universal dominance. Explicit physical indices provide only a small gain whose paired bootstrap interval overlaps zero for LOGO and LOYO; it should be an ablation, not the headline contribution. Nested XGBoost evaluates hyperparameter selection within the specified model and feature family. It does not remove selection bias from repeatedly comparing architectures or feature families on the same outer folds.

Robustness tests for the fixed regularized XGBoost yield 50 km buffered LOGO R2=0.596, +/-1-year buffered LOYO R2=0.572, four-observed-subregion LOSO R2=0.507, and rolling-origin forward-validation R2=0.553 for 1980-2023. These drops are moderate but show that ordinary LOGO can benefit from nearby glaciers and ordinary LOYO is temporal interpolation, not future forecasting.

## Negative Results That Must Be Reported Honestly

- Fixed lapse-rate correction using pressure-derived ERA5 grid elevation reduced LOGO and LOYO skill. A proper static ERA5 geopotential field and calibrated precipitation gradient would be required before revisiting this route.
- Glacier-balanced and uncertainty-weighted objectives did not improve overall RMSE.
- Coarse seasonal aggregation underperformed monthly features.
- Hugonnet weak-label pretraining caused unstable or negative transfer. Geodetic information is better used as a long-term post-hoc constraint.
- PhysGlacierFormer seasonal heads initially exhibited strong winter and summer error cancellation. Stronger seasonal supervision improved seasonal R2 but did not beat XGBoost annually.
- Target-magnitude weighting reduced tail errors slightly but worsened overall LOGO skill (R2 0.624 to 0.613), so it is not used in the final model.
- Removing ERA5-Land snow albedo slightly improved LOGO but reduced forward-validation R2 from 0.553 to 0.505. The final model retains it as a reanalysis proxy, not as observed glacier-surface albedo.
- A leakage-safe two-stage model first cross-fitted glacier climatological means and then modeled annual residuals. The anomaly-only version improved LOGO to R2=0.644 and RMSE=624 mm, but the paired RMSE difference versus the main XGBoost was -16 mm with a 95% glacier-bootstrap interval of -59 to +21 mm. It degraded LOYO to R2=0.551 and RMSE=701 mm. Supplying the full compact monthly climate to the residual stage only recovered LOYO to R2=0.560 and RMSE=694 mm; its +18 mm paired degradation has a 95% year-bootstrap interval of +5 to +31 mm. The spatial gain is uncertain and trades against temporal skill, so the method is not selected.
- Season-selective monotonic XGBoost constraints leave ambiguous precipitation, flux and reanalysis-albedo signs unconstrained. They change LOGO RMSE from 641.3 to 642.2 mm, LOYO RMSE from 677.0 to 673.1 mm, and forward RMSE from 700.2 to 696.3 mm. All three paired bootstrap intervals cross zero (forward difference -3.8 mm, 95% CI -11.3 to +4.4 mm). The constrained model is a physics-guided sensitivity test, not a demonstrated accuracy improvement, and the unconstrained regularized XGBoost remains primary.
- Nested amplitude calibration fits an observation-on-prediction slope from inner grouped OOF predictions only. It improves LOYO RMSE from 677.0 to 673.2 mm and bias from -15.1 to -1.5 mm, but the 10,000-resample paired year-bootstrap RMSE difference is -3.6 mm with a 95% interval of -20.1 to +12.5 mm. It is rejected for the final reconstruction because the apparent error gain is not reliable.

Final-model SHAP magnitudes assign 64.4% of total absolute contribution to glacier-month climate anomalies. July ERA5-Land snow-albedo anomaly is the largest individual monthly contribution and is strongly associated with annual SMB, but its sign is not straightforward physically. It is therefore treated as a modeled snow-state proxy and a priority for MODIS/Sentinel-2 verification, not as causal evidence of glacier-albedo feedback.

## Hugonnet Calibration Evidence

Calibration residuals cannot be reused as independent validation. With the corrected all-glacier reconstruction, a temporal-transfer test estimates offsets from 2000-2010 and evaluates 2010-2020 for 9,869 glaciers. Early and late residuals correlate at r=0.293. With a +/-1 m w.e. yr-1 clip, shrink=0.5 reduces later-period RMSE from 0.491 to 0.398 m w.e. yr-1, whereas full correction worsens it to 0.507. Five-fold glacier cross-fitting selects shrink=0.5 in every fold and gives held-glacier RMSE=0.398 (95% CI 0.391-0.405). This removes reuse of the same glaciers for shrink selection and evaluation, but it remains evidence for 2000-2010 to 2010-2020 transfer rather than proof for all historical periods.

The final reconstruction contains exactly 1,386,020 glacier-years (18,730 glaciers x 74 years, 1951-2024), no duplicate keys, and no non-finite predictions. The raw regional series has a 1980-2024 trend of -0.141 m w.e. decade-1 (p=0.023). Against Malles & Marzeion over 1951-2018, the raw reconstruction gives r=0.822 and RMSE=4.55 Gt; conservative calibration keeps the same correlation and RMSE=4.54 Gt while reversing the mean bias from -0.66 to +0.57 Gt. A complementary specific-balance comparison, which reduces but does not eliminate the mismatch between fixed RGI v7 area and evolving Malles area, gives raw/calibrated r=0.851 and RMSE=0.279/0.271 m w.e. yr-1. Against Zemp et al. over 1953-2016, raw RMSE is 0.300 m w.e. yr-1 and calibrated RMSE is 0.327. Zemp shares WGMS source information and is not independent. These comparisons require publishing both raw and calibrated products rather than labeling the calibrated series universally superior.

Against the GlaMBIE 2000-2023 RGI02 period mean, raw/calibrated specific balances are -0.760/-0.675 m w.e. yr-1 versus -0.68 +/- 0.06 m w.e. yr-1. Corresponding fixed-area mass changes are -11.03/-9.81 Gt yr-1 versus -9.0 +/- 0.9 Gt yr-1. The calibrated period mean is closely consistent, but GlaMBIE combines glaciological and geodetic inputs and therefore shares information with WGMS and Hugonnet. This is a period-mean consistency check, not independent annual or glacier-scale validation.

Cluster-excluded residual calibration gives empirical 90% coverage of 0.899 for both LOGO and LOYO. The corresponding global half-widths are 1.03 and 1.07 m w.e. yr-1. These are validation-distribution intervals and do not guarantee coverage for inventory glaciers outside the WGMS covariate domain.

### Annual GlaMBIE Check (2026-09-06)

The official WGMS GlaMBIE Dataset 1.0.0 is now available locally. The 24 northern hydrological periods 1999.75-2000.75 through 2022.75-2023.75 map to model end years 2000-2023. The reference average is -0.6781 m w.e. yr-1. Its combined series shares glaciological/geodetic sources; no model or calibration parameter was selected using these new comparisons. Source CSV SHA256: `4683C41B8480762D01D9BC8E1D0130CF907BC29E9A5B08EFAF209F1E1CB90EEE`.

| Reference / years | Raw RMSE | Calibrated RMSE | Raw bias | Calibrated bias | r |
|---|---:|---:|---:|---:|---:|
| Combined, 2000-2023 | 0.459 | 0.451 | -0.082 | +0.003 | 0.913 |
| Combined, 2020-2023 | 0.396 | 0.472 | +0.347 | +0.431 | 0.978 |
| Altimetry component, 2013-2022 | 0.382 | 0.395 | +0.021 | +0.105 | 0.752 |

Errors and biases above are in m w.e. yr-1. The four-year recent subset is descriptive and receives no five-year block-bootstrap CI. The altimetry rows have their own annual-variability flag set to one, but remain a processed GlaMBIE component rather than raw independent satellite measurements. Model annual standard deviation is only 63.3% of the combined reference over 2000-2023 and 58.3% of the altimetry component over 2013-2022. Constant offsets correct the mean but cannot correct this amplitude deficit. In 2023, GlaMBIE is -2.802 versus raw -2.448 and calibrated -2.364 m w.e. yr-1. Retain both products and investigate temporal variability and recent underestimation before claiming uniform improvement.

The external figure now accumulates individual Malles forcing trajectories before taking 5th/95th percentiles. This band represents forcing ensemble spread, not total reconstruction uncertainty. Domain diagnostics now summarize each glacier across all 74 reconstructed years; the previous first-row summary only represented 1951. Cluster bootstrap uses positional indices and paired target consistency checks; paired differences report the observed point difference separately from the bootstrap mean. Two-stage spatial imputation was moved inside its inner glacier folds; the current 993 x 21 spatial matrix has zero missing values, so this prevents a future leakage path without changing the present experiment inputs. Spatial-stage targets remain glacier means over differing observation periods, not common-period climatologies.

## Literature Alignment

### Amplitude And Sampling Diagnosis

For matched October-September support over 2000-2023, there are 342 observations from 30 RGI glaciers. Annual coverage varies from 2 to 25 glaciers and represents only 0.019%-0.606% of RGI02 area. The 2020 sample has two glaciers and the 2022 sample three; annual diagnostics therefore report both all years and years with at least five observed glaciers.

Within-glacier centered series (318 records; sites with at least five years) have predicted/observed standard-deviation ratios of 0.883 under LOGO and 0.675 under LOYO. This is evidence of temporal generalization shrinkage at observed glaciers. Centering is performed after OOF prediction for diagnosis, not during model fitting.

Sampling and weighting also matter: the standard deviation of annual observed WGMS means relative to GlaMBIE is 0.933 with equal glacier weights but 0.664 with area weights. Holding the same 30 modeled sites fixed for all 24 years gives modeled/reference ratios of 0.851 with equal weights and 0.657 with area weights, versus 0.633 for all RGI02 area-weighted predictions. These are representativeness diagnostics, not evidence that equal weights estimate total regional mass change better. The final-model fit at training glaciers is explicitly labeled in-sample and excluded from validation claims. A blanket amplification of the regional curve is not justified by these comparisons.

A predeclared year/end-month coherence penalty (weight 0.5, groups with at least three training samples) was tested using only outer-fold training targets. Its gradient passed finite-difference checks and its diagonal Hessian majorant passed a positive-semidefinite bound check. LOGO RMSE becomes 642.5 mm and LOYO 685.0 mm versus 641.3/677.0 mm without the penalty. Most-negative-decile LOYO RMSE worsens from 1154.7 to 1181.1 mm, and the full-sample predicted/observed standard-deviation ratio decreases from 0.692 to 0.683. This route is rejected. The test suite now contains 17 tests. Variance ratios are descriptive: an MSE-optimal conditional-mean predictor can have lower variance than noisy observations, so matching observed variance is not itself a valid training objective.

The paired 2,000-resample differences for coherence weight 0.5 are +1.15 mm (LOGO; 95% CI -3.89 to +6.50) and +7.95 mm (LOYO; -2.01 to +19.76). They show no demonstrated improvement, without proving a significant degradation. Bootstrap groups are now sorted and RNGs reset per comparison so adding experiments does not change another run's intervals. The current `publication_evaluation` CSVs are authoritative for updated intervals; earlier ablation paragraphs record the original resampling runs.

Malles ensemble membership needs explicit interpretation: only forcing members 1, 9 and 10 fully span 1951-2018; all ten coexist only in 1981-2010. Accumulating the annual available-member mean gives -379.97 Gt over 1951-2018, while the mean of three complete cumulative trajectories gives -335.32 Gt. The updated figure shows both curves and shades only the complete-trajectory 5th-95th percentile range in the cumulative panel. This difference is an ensemble-support effect, not a change in this study's predictions. The official model uncertainty field is not represented by that forcing-spread band.

- [Sjursen et al. (2025)](https://tc.copernicus.org/articles/19/5801/2025/) show that XGBoost is competitive for medium-sized glacier tabular data and stress non-random validation and reanalysis-to-glacier elevation differences.
- [van der Meer et al. (2025)](https://tc.copernicus.org/articles/19/805/2025/) find that parsimonious temperature and precipitation predictors can outperform larger sets and document failures in extreme years.
- [Guidicelli et al. (2023)](https://tc.copernicus.org/articles/17/977/2023/) identify glacier-reanalysis elevation difference as important for winter-balance downscaling.
- [Draeger et al. (2026)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025JF008740) show for western Canada that precipitation correction, elevation gradients, MODIS-informed albedo, and glacier-specific geodetic calibration are high-value additions; they also demonstrate why annual agreement can hide opposing seasonal errors.
- [Hugonnet et al. (2021)](https://www.nature.com/articles/s41586-021-03436-z) provide remote-sensing geodetic mass-change constraints, not annual in-situ SMB observations.
- [The GlaMBIE Team (2025)](https://www.nature.com/articles/s41586-024-08545-z) provide a reconciled annual regional observational baseline for 2000-2023. RGI02 has a reported period mean of -0.68 +/- 0.06 m w.e. yr-1, but the product is not statistically independent of all inputs used here.

## Remaining Publication Gates

1. Explain and test the regional amplitude deficit and 2020-2023 loss underestimation now identified by the annual GlaMBIE comparison. Preserve this external benchmark as evaluation evidence; any future tuning against it must be disclosed and evaluated elsewhere.
2. Add MODIS albedo or Sentinel-2 snowline/snow-cover products before claiming physically resolved ablation. ERA5 snow albedo is not a substitute for glacier-surface albedo.
3. Obtain pressure-level temperature and precipitation-gradient information before retrying elevation downscaling; the pressure-derived fixed lapse-rate ablation failed.
4. State that fixed RGI v7 geometry yields reference-geometry SMB; it is not a coupled glacier-evolution or dynamic mass-change reconstruction.
5. Treat 2024 as provisional extrapolation because the latest WGMS training target is 2023 and available external regional products end earlier.

## Recommended Paper Framing

The strongest defensible paper is currently a regional machine-learning reconstruction built around rigorous data harmonization, climate climatology-anomaly decomposition, spatial and temporal robustness tests, and conservative geodetic calibration. Calling the work primarily a novel Transformer paper is not supported by the results.
