"""Fast invariant checks for the corrected PhysGlacierFormer v2 pipeline."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "02_models"))
sys.path.insert(0, os.path.join(PROJECT_DIR, "03_training"))
sys.path.insert(0, os.path.join(PROJECT_DIR, "04_reconstruction"))
sys.path.insert(0, os.path.join(PROJECT_DIR, "01_preprocessing"))

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
    year_coherence_objective,
)
from calibrate_xgboost_v2_amplitude import fit_amplitude_calibrator  # noqa: E402
from analyze_xgboost_v2_reconstruction import (  # noqa: E402
    cumulative_ensemble_summary,
    glambie_region02,
)
from evaluate_phys_v2_results import (  # noqa: E402
    bootstrap_intervals,
    paired_bootstrap_difference,
    regression_metrics,
)
from train_twostage_xgboost_v2 import cross_fitted_spatial_mean  # noqa: E402
from step14_probe_sentinel2_surface import (  # noqa: E402
    check_radiometry, ndsi_with_quality, scaled_reflectance,
)


class ExternalComparisonTests(unittest.TestCase):
    def test_sentinel_rejects_ambiguous_legacy_cog_scaling(self) -> None:
        asset = {"raster:bands": [{"nodata": 0, "scale": 0.0001, "offset": -0.1}]}
        check_radiometry(asset, 0.0001, -0.1, 0)
        with self.assertRaises(ValueError):
            check_radiometry(asset, 1.0, 0.0, 0)
        with self.assertRaises(ValueError):
            check_radiometry(asset, 0.0001, -0.1, 65535)

    def test_sentinel_scaling_excludes_nodata_before_applying_offset(self) -> None:
        asset = {"raster:bands": [{"nodata": 0, "scale": 0.0001, "offset": -0.1}]}
        values = scaled_reflectance(np.array([0, 1500, 2000]), asset)
        self.assertTrue(np.isnan(values[0]))
        np.testing.assert_allclose(values[1:], [0.05, 0.1])

    def test_ndsi_masks_cloud_shadow_water_and_pixels_outside_outline(self) -> None:
        green = np.full((2, 3), 0.5)
        swir = np.full((2, 3), 0.1)
        scl = np.array([[11, 9, 3], [6, 5, 11]])
        inside = np.ones((2, 3), dtype=bool)
        inside[1, 2] = False
        ndsi, valid = ndsi_with_quality(green, swir, scl, inside)
        np.testing.assert_array_equal(valid, [[True, False, False], [False, True, False]])
        np.testing.assert_allclose(ndsi[valid], 2 / 3)
        self.assertTrue(np.isnan(ndsi[~valid]).all())

    def test_year_coherence_gradient_and_hessian_bound(self) -> None:
        groups = np.array([200009, 200009, 200009, 200010])
        observed = np.array([-2., -1., 0., 1.])
        predicted = np.array([-1., -0.5, 0.2, 0.8])
        weight = 0.5
        gradient, majorant = year_coherence_objective(groups, weight)(observed, predicted)

        def loss(pred):
            residual = pred - observed
            return 0.5 * (residual @ residual) + 0.5 * weight * 3 * residual[:3].mean()**2

        step = 1e-5
        finite_difference = []
        for index in range(4):
            delta = np.eye(4)[index] * step
            finite_difference.append((loss(predicted + delta) - loss(predicted - delta)) / (2 * step))
        np.testing.assert_allclose(gradient, finite_difference, atol=1e-8)
        hessian = np.eye(4)
        hessian[:3, :3] += weight / 3
        self.assertGreaterEqual(np.linalg.eigvalsh(np.diag(majorant) - hessian).min(), -1e-10)
        self.assertAlmostEqual(gradient[-1], predicted[-1] - observed[-1])
        zero_gradient, zero_hessian = year_coherence_objective(groups, 0)(observed, predicted)
        np.testing.assert_allclose(zero_gradient, predicted - observed)
        np.testing.assert_allclose(zero_hessian, np.ones(4))

    def test_two_stage_spatial_fit_handles_missing_data_without_test_targets(self) -> None:
        spatial = np.repeat(np.array([[0, 1], [2, np.nan], [3, 4], [5, np.nan]]), 2, axis=0)
        target = np.linspace(-2, 1, 8)
        ids = np.repeat(["A", "B", "C", "D"], 2)
        train = ids != "D"
        first = cross_fitted_spatial_mean(spatial, target, ids, train, ~train, "ridge", 2, 42)
        changed = target.copy()
        changed[~train] = 1000
        second = cross_fitted_spatial_mean(spatial, changed, ids, train, ~train, "ridge", 2, 42)
        self.assertTrue(np.isfinite(first[0][train]).all())
        self.assertTrue(np.isfinite(first[1]).all())
        np.testing.assert_array_equal(first[0][train], second[0][train])
        np.testing.assert_array_equal(first[1], second[1])

    def test_cluster_bootstrap_uses_positions_not_dataframe_labels(self) -> None:
        frame = pd.DataFrame({
            "rgi_id": ["A", "A", "B", "B"],
            "obs_annual": [-2, -1, 0, 1], "pred_annual": [-1.5, -0.8, 0.1, 0.6],
        })
        relabeled = frame.set_axis([80, 12, 17, 90])
        first = bootstrap_intervals(frame, "rgi_id", 30, np.random.default_rng(42))
        second = bootstrap_intervals(relabeled, "rgi_id", 30, np.random.default_rng(42))
        self.assertEqual(first, second)

    def test_paired_metrics_reject_target_mismatch_and_keep_point_estimate(self) -> None:
        reference = pd.DataFrame({
            "rgi_id": ["A", "A", "B", "B"], "year": [2000, 2001, 2000, 2001],
            "obs_annual": [-2, -1, 0, 1], "pred_annual": [-1.5, -0.8, 0.1, 0.6],
        })
        candidate = reference.copy()
        candidate["pred_annual"] = candidate["obs_annual"]
        values = paired_bootstrap_difference(candidate, reference, "rgi_id", 30, np.random.default_rng(42))
        expected = -np.sqrt(np.mean((reference.pred_annual - reference.obs_annual)**2)) * 1000
        self.assertAlmostEqual(values["rmse_delta_mm"], expected)
        candidate["obs_annual"] = candidate["obs_annual"].astype(float)
        candidate.loc[0, "obs_annual"] += 0.01
        with self.assertRaisesRegex(ValueError, "identical observation"):
            paired_bootstrap_difference(candidate, reference, "rgi_id", 30, np.random.default_rng(42))

    def test_constant_observations_have_undefined_r2_but_valid_rmse(self) -> None:
        values = regression_metrics(np.ones(4), np.zeros(4))
        self.assertTrue(np.isnan(values["r2"]))
        self.assertTrue(np.isnan(values["pearson_r"]))
        self.assertEqual(values["rmse_mm"], 1000.0)

    def test_glambie_end_year_and_rejection_of_wrong_time_support(self) -> None:
        frame = pd.DataFrame({
            "start_dates": [1999.75, 2000.75], "end_dates": [2000.75, 2001.75],
            "region": ["western_canada_us"] * 2, "glacier_area": [14000, 13900],
            "combined_gt": [-10, -12], "combined_gt_errors": [1, 1],
            "combined_mwe": [-0.7, -0.8], "combined_mwe_errors": [0.1, 0.1],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "reference.csv")
            frame.to_csv(path, index=False)
            self.assertEqual(glambie_region02(path).year.tolist(), [2000, 2001])
            calendar = frame.copy()
            calendar[["start_dates", "end_dates"]] -= 0.75
            calendar.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Oct-Sep"):
                glambie_region02(path)
            duplicate = pd.concat([frame, frame.iloc[[0]]])
            duplicate.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate or missing"):
                glambie_region02(path)
            gapped = frame.copy()
            gapped.loc[1, ["start_dates", "end_dates"]] += 1
            gapped.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate or missing"):
                glambie_region02(path)

    def test_cumulative_band_preserves_member_trajectories(self) -> None:
        # Crossing trajectories cancel at year two; annual quantiles do not.
        values = np.array([[-10.0, 10.0], [10.0, -10.0]])
        result = cumulative_ensemble_summary(values)
        self.assertEqual(result["malles_p05_cumulative_gt"][-1], 0.0)
        self.assertEqual(result["malles_p95_cumulative_gt"][-1], 0.0)
        self.assertEqual(result["malles_cumulative_gt"][-1], 0.0)
        incomplete = np.vstack([values, [np.nan, 100]])
        np.testing.assert_array_equal(
            cumulative_ensemble_summary(incomplete)["malles_cumulative_gt"], [0, 0]
        )


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
