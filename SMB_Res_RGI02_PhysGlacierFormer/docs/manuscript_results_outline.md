# Manuscript Results Outline

## Central Claim

A leakage-controlled regional reconstruction can recover transferable annual glacier-wide SMB skill from sparse WGMS observations when monthly ERA5-Land conditions are expressed as local climatology and anomaly components. Conservative geodetic calibration improves long-term regional consistency but does not improve interannual correlation.

Do not claim that GlacierFormer is the best model, that Hugonnet is independent validation, or that fixed-geometry SMB is glacier-volume evolution.

## Main Results

1. **Data correction and leakage control:** report official RGI6-to-RGI7 remapping, corrected ERA5-Land accumulation units, observation-aligned annual windows, and fold-local preprocessing.
2. **Model selection:** use nested XGBoost as the unbiased selection result (LOGO R2=0.624, LOYO R2=0.574). Report the fixed regularized profile separately (LOGO R2=0.624, LOYO R2=0.581).
3. **Method contribution:** climate-anomaly decomposition improves R2 by about 0.19 under LOGO and 0.13 under LOYO relative to raw monthly climate.
4. **Robustness:** report buffered LOGO/LOYO, rolling-origin, and LOSO together with glacier/year cluster-bootstrap intervals.
5. **Reconstruction:** provide raw and conservative-calibrated products for 18,730 glaciers over 1951-2024, plus covariate-domain flags and validation-distribution prediction intervals.
6. **External consistency:** separate Malles model intercomparison, Hugonnet temporal-transfer evidence, Zemp shared-WGMS comparison, and GlaMBIE period-mean comparison.

## Required Figures

1. Data and method workflow, including fixed-reference geometry and calibration boundaries.
2. LOGO/LOYO validation, benchmark models, and stress tests.
3. Feature-family and month-resolved SHAP contributions with a non-causal interpretation.
4. Regional annual SMB, spatial mean/trend, and period-distribution shift.
5. Malles annual comparison, cumulative mass change, Hugonnet temporal transfer, and calibration offsets.
6. Extreme-year residual diagnostics or a dedicated tail-performance panel.

## Required Tables

1. Dataset coverage, glacier/year counts, units, and mapping methods.
2. Nested and fixed-profile model performance with 95% cluster-bootstrap intervals.
3. Ablation and negative-result matrix.
4. External-product comparison with scale, period, geometry, shared inputs, and valid interpretation.

## Submission Gates

- Obtain annual GlaMBIE RGI02 data before claiming annual observational agreement for 2000-2023.
- Add observed snowline or albedo information before interpreting ERA5 snow albedo physically.
- Explain weak performance in the most negative 10% of years and retain the extreme-tail diagnostic.
- Label 2024 as provisional and all regional totals as fixed-RGI-v7 reference-geometry estimates.
