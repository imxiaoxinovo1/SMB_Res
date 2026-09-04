# Training

Training scripts should write metrics to method-specific result folders:

```text
results/baseline/
results/hypsometry/
results/remote_sensing/
results/ensemble/
```

Required validation protocols:

1. LOGO: leave one glacier out for spatial generalization.
2. LOYO: leave one year out for temporal generalization.
3. Buffered LOGO/LOYO and rolling-origin validation: spatial and temporal stress tests.
4. LOSO: broad subregional transfer, interpreted cautiously because WGMS targets cover only four O2 subregions.

The publication-facing annual model is the corrected regularized XGBoost model. PhysGlacierFormer is retained as a seasonal-process benchmark. All preprocessing, imputation, sample weighting, and hyperparameter selection must be fitted without held-out labels.

Corrected-pipeline mean, Ridge, Random Forest, Extra Trees, and LightGBM baselines are available through `train_tree_v2_cv.py`. Use identical feature sets and outer folds when comparing algorithms.

Do not save metrics directly into the project root.
