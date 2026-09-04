# Literature-to-Experiment Matrix

| Study | Transferable method | Decision in this project |
|---|---|---|
| Sjursen et al. (2025), *The Cryosphere* | XGBoost for medium-sized glacier tables; elevation-band representation; domain-aware validation | Adopted XGBoost, hypsometry, LOGO/LOYO, 50 km buffered LOGO and subregional holdout |
| van der Meer et al. (2025), *The Cryosphere* | Parsimonious temperature/precipitation predictors; explicit warning about unseen extremes | Tested minimal and compact feature sets; added rolling-origin validation and extreme-tail diagnostics |
| Guidicelli et al. (2023), *The Cryosphere* | Reanalysis-to-glacier elevation difference for winter-balance downscaling | Tested a pressure-derived grid elevation and fixed lapse rate; rejected after lower LOGO/LOYO skill |
| Draeger et al. (2026), *JGR Earth Surface* | Western Canada ERA5 forcing, elevation-dependent precipitation correction, MODIS-informed albedo, glacier-specific geodetic calibration | Adopted conservative geodetic calibration with early-to-late transfer and glacier-cross-fitted shrink selection; MODIS albedo and pressure-level lapse rates remain the highest-value new inputs |
| Hugonnet et al. (2021), *Nature* | Glacier-scale multi-period geodetic mass-change constraints | Weak-label pretraining was rejected; post-hoc calibration is retained, with 2000-2010 to 2010-2020 transfer testing |
| Zemp et al. (2019), *Nature* | Regional interpolation of glaciological and geodetic records | Used only as a consistency benchmark because it shares WGMS source information |
| Malles and Marzeion (2021), *The Cryosphere* | Regional ensemble reconstruction under multiple climate forcings | Used as external model intercomparison, not observational validation |

## Resulting Methodological Position

The publishable contribution is not a claim that a Transformer outperforms established methods. It is a rigorously harmonized regional reconstruction showing that climate climatology-anomaly decomposition, hypsometry, domain-aware validation, and conservatively validated geodetic calibration provide measurable value under sparse WGMS supervision. The Transformer supplies a physically constrained seasonal benchmark; the corrected XGBoost model supplies the strongest annual predictions.

## Next Data Acquisition Priority

1. MODIS glacier-surface albedo or Sentinel-2 snowline/snow-cover metrics for melt-season state.
2. ERA5 pressure-level temperature for time-varying lapse rates.
3. Orographic precipitation gradients or independent snow-accumulation constraints.

These additions address identified structural errors. Further attention modules or deeper encoders are not justified by the present sample size or validation results.
