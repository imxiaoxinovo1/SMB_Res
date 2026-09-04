# Reconstruction

This folder contains the final all-RGI02 reconstruction, early-to-late Hugonnet temporal-transfer sensitivity test, conservative calibration, and regional quality-control analysis.

Reconstruction outputs should be written under:

```text
results/reconstruction/
```

The stable baseline reconstruction in `SMB_Res_ByClaudeV2` remains unchanged. Publish both raw and conservatively calibrated products: calibration improves transfer to a later Hugonnet decade but can shift the regional mean relative to other products. Regional mass conversion uses fixed RGI v7 reference areas, so the output is reference-geometry SMB rather than a dynamic glacier-evolution simulation.
