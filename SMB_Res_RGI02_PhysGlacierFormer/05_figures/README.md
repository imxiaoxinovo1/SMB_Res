# Figures

Figure scripts should read from `results/` and write rendered files to `figures/`.

Publication-facing outputs:

1. `fig_phys_v2_validation_performance.png`: LOGO/LOYO scatter, corrected benchmarks, and stress tests.
2. `fig_xgboost_feature_contributions.png`: grouped and month-resolved SHAP contributions.
3. `fig_regional_reconstruction_results.png`: annual SMB, spatial mean/trend, and period distributions.
4. `fig_reconstruction_external_comparison.png`: Malles intercomparison, Hugonnet transfer, and calibration offsets.

Figures are rendered as PNG by default. Numerical values must come from the CSV outputs under `results/`; do not manually type metrics into plotting scripts.
