"""Fast invariant checks for the corrected PhysGlacierFormer v2 pipeline."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "02_models"))
sys.path.insert(0, os.path.join(PROJECT_DIR, "03_training"))

from config import (  # noqa: E402
    PHYS_GLACIERFORMER_V2_PARAMS,
    PHYS_V2_ERA5_CSV,
    PHYS_V2_FINAL_PREPROCESSOR,
    PHYS_V2_RECONSTRUCTION_CALIBRATED_CSV,
    PHYS_V2_SEQUENCES_NPZ,
    PHYS_V2_TERRAIN_CSV,
)
from phys_glacierformer_v2 import PhysGlacierFormerV2  # noqa: E402
from train_tree_v2_cv import (  # noqa: E402
    fit_model,
    make_model,
    monthly_physical_monotonic_constraints,
)
from calibrate_xgboost_v2_amplitude import fit_amplitude_calibrator  # noqa: E402


class PhysV2PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = np.load(PHYS_V2_SEQUENCES_NPZ, allow_pickle=True)

    def test_month_windows_are_complete_and_unique(self) -> None:
        months = self.data["month_ids"]
        self.assertEqual(months.shape[1], 12)
        for row in months:
            self.assertEqual(set(row.tolist()), set(range(1, 13)))

    def test_dynamic_units_are_physically_plausible(self) -> None:
        frame = pd.read_csv(PHYS_V2_ERA5_CSV, usecols=["t2m", "tp", "sf", "ssrd", "sp"])
        self.assertTrue(frame.t2m.between(-60.0, 45.0).all())
        self.assertTrue(frame.tp.between(0.0, 2500.0).all())
        self.assertTrue(frame.sf.between(0.0, 2500.0).all())
        self.assertTrue(frame.ssrd.between(0.0, 500.0).all())
        self.assertTrue(frame.sp.between(40000.0, 110000.0).all())

    def test_seasonal_labels_conserve_annual_balance(self) -> None:
        annual = self.data["y_annual"]
        winter = self.data["y_winter"]
        summer = self.data["y_summer"]
        valid = np.isfinite(winter) & np.isfinite(summer)
        self.assertGreater(int(valid.sum()), 300)
        agreement = np.abs(annual[valid] - winter[valid] - summer[valid]) <= 0.10
        self.assertGreaterEqual(float(np.mean(agreement)), 0.98)

    def test_official_mapping_corrects_south_cascade(self) -> None:
        terrain = pd.read_csv(PHYS_V2_TERRAIN_CSV)
        row = terrain.loc[terrain["name"].str.contains("South Cascade", case=False, na=False)]
        self.assertEqual(len(row), 1)
        self.assertEqual(row.iloc[0]["mapping_method"], "rgi6_polygon_overlap")
        self.assertGreater(float(row.iloc[0]["area_km2"]), 2.0)

    def test_model_enforces_annual_seasonal_sum(self) -> None:
        params = dict(PHYS_GLACIERFORMER_V2_PARAMS)
        params["n_dynamic_features"] = self.data["X_dyn"].shape[-1]
        params["n_static_features"] = self.data["X_sta"].shape[-1]
        model = PhysGlacierFormerV2(**params).eval()
        indices = slice(0, 4)
        with torch.no_grad():
            annual, winter, summer = model(
                torch.tensor(self.data["X_dyn"][indices]),
                torch.tensor(self.data["X_sta"][indices]),
                torch.tensor(self.data["X_hyp"][indices]),
                torch.tensor(self.data["month_ids"][indices], dtype=torch.long),
            )
        torch.testing.assert_close(annual, winter + summer)

    def test_final_model_feature_schema_is_explicit(self) -> None:
        preprocessing = np.load(PHYS_V2_FINAL_PREPROCESSOR, allow_pickle=True)
        names = [str(value) for value in preprocessing["feature_names"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), len(preprocessing["medians"]))
        self.assertTrue(all(f"t2m_m{month:02d}" in names for month in range(1, 13)))
        self.assertEqual(
            names[-5:],
            ["hyp_q10_m", "hyp_q25_m", "hyp_q50_m", "hyp_q75_m", "hyp_q90_m"],
        )

    def test_final_reconstruction_grid_is_complete(self) -> None:
        frame = pd.read_csv(
            PHYS_V2_RECONSTRUCTION_CALIBRATED_CSV,
            usecols=[
                "rgi_id",
                "year",
                "predicted_smb_m",
                "predicted_smb_conservative_m",
            ],
        )
        self.assertEqual(len(frame), 18_730 * 74)
        self.assertEqual(frame["rgi_id"].nunique(), 18_730)
        self.assertEqual((int(frame["year"].min()), int(frame["year"].max())), (1951, 2024))
        self.assertFalse(frame.duplicated(["rgi_id", "year"]).any())
        self.assertTrue(
            np.isfinite(
                frame[["predicted_smb_m", "predicted_smb_conservative_m"]].to_numpy()
            ).all()
        )

    def test_ridge_accepts_fold_local_sample_weights(self) -> None:
        x = np.arange(24, dtype=np.float64).reshape(8, 3)
        y = np.linspace(-1.0, 1.0, 8)
        weights = np.linspace(0.5, 1.5, 8)
        model = make_model("ridge", seed=42)
        fit_model(model, "ridge", x, y, weights)
        prediction = model.predict(x)
        self.assertTrue(np.isfinite(prediction).all())

    def test_monotonic_constraints_encode_only_robust_seasonal_signs(self) -> None:
        constraints = monthly_physical_monotonic_constraints(
            ["t2m", "sf", "tp"], n_static=2, n_hypsometry=1
        )
        self.assertEqual(len(constraints), 39)
        self.assertEqual(constraints[4:9], (-1, -1, -1, -1, -1))
        self.assertEqual(constraints[12:16], (1, 1, 1, 1))
        self.assertEqual(constraints[21:24], (1, 1, 1))
        self.assertTrue(all(value == 0 for value in constraints[24:]))

    def test_amplitude_calibration_clips_unstable_slopes(self) -> None:
        observed = np.array([-2.0, -1.0, 0.0, 1.0])
        predicted = observed * 0.1
        intercept, slope, raw_slope = fit_amplitude_calibrator(
            observed, predicted, slope_min=0.75, slope_max=1.50
        )
        self.assertGreater(raw_slope, 1.50)
        self.assertEqual(slope, 1.50)
        self.assertAlmostEqual(intercept, float(observed.mean() - slope * predicted.mean()))


if __name__ == "__main__":
    unittest.main()
