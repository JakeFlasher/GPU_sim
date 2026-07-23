"""Producer-side test for ``observation_builder.emit_canonical_pre_calibration_features``.

Pins the producer contract: when the walker's launch summary carries
``unique_sector_count`` + the cache model spec carries
``cache_model_capacity_bytes``, the helper emits all 6 canonical
features without dropping any to null/missing.
"""

from __future__ import annotations

import unittest

from tools.hrd_profiler.observation_builder import (
    emit_canonical_pre_calibration_features,
)


class EmitCanonicalFeaturesTest(unittest.TestCase):
    def test_emits_all_six_keys(self) -> None:
        launch_summary = {"unique_sector_count": 10_000}
        sharing_locality = {
            "grounded": [{"class_name": "intra_lane", "fraction": 0.5}],
        }
        hrd_profile = {
            "bins": [
                {
                    "hierarchy_layer": "sector",
                    "rd_sector_units_min": 4096,
                    "observation_count": 100,
                },
                {
                    "hierarchy_layer": "sector",
                    "rd_sector_units_min": 0,
                    "observation_count": 900,
                },
            ],
        }
        features = emit_canonical_pre_calibration_features(
            launch_summary=launch_summary,
            cache_model_capacity_bytes=33554432,
            raw_prediction=0.7,
            sharing_locality=sharing_locality,
            hrd_profile=hrd_profile,
            sector_stream_records=[
                {"memory_op_class": "GLOBAL"},
                {"memory_op_class": "SHARED"},
            ],
            grid_volume=200,
            device="NVIDIA GeForce RTX 5060 Laptop GPU",
        )
        for k in (
            "raw_prediction", "footprint_ratio", "long_rd_tail_mass",
            "intra_lane_reuse", "global_share", "gridDim_density",
        ):
            self.assertIn(k, features)
            self.assertIsNotNone(features[k])
        self.assertAlmostEqual(features["raw_prediction"], 0.7)
        self.assertAlmostEqual(features["long_rd_tail_mass"], 0.1)
        self.assertAlmostEqual(features["intra_lane_reuse"], 0.5)
        self.assertAlmostEqual(features["global_share"], 0.5)
        self.assertGreater(features["footprint_ratio"], 0.0)
        self.assertGreater(features["gridDim_density"], 0.0)

    def test_fail_closed_on_missing_raw_prediction(self) -> None:
        # Pass raw_prediction=None and expect fail-closed exit (ValueError
        # OR TypeError; the conversion to float raises before validation).
        with self.assertRaises((ValueError, TypeError)):
            emit_canonical_pre_calibration_features(
                launch_summary={"unique_sector_count": 10},
                cache_model_capacity_bytes=1024,
                raw_prediction=None,  # type: ignore[arg-type]
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
