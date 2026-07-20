from __future__ import annotations

import unittest

from experimental_adaptive_server import (
    EXPERIMENTAL_GRID,
    calibrate_probability,
    calibrated_standard_deviation,
    layout_from_roles,
)


class ExperimentalAdaptiveServerTest(unittest.TestCase):
    def test_experimental_grid_does_not_change_production_defaults(self) -> None:
        self.assertEqual((EXPERIMENTAL_GRID.base_columns, EXPERIMENTAL_GRID.base_rows), (4, 2))
        self.assertEqual(EXPERIMENTAL_GRID.minimum_samples_for_split, 3)
        self.assertEqual(EXPERIMENTAL_GRID.minimum_child_samples, 1)

    def test_layout_from_normalized_roles(self) -> None:
        layout = layout_from_roles(
            {"cue": [0.25, 0.5], "object1": [0.5, 0.25], "object2": [0.75, 0.8]}
        )
        self.assertEqual(layout.cue.x_mm, 711.0)
        self.assertEqual(layout.cue.y_mm, 711.0)
        self.assertEqual(layout.object2.x_mm, 2133.0)
        self.assertAlmostEqual(layout.object2.y_mm, 1137.6)

    def test_calibration_is_monotonic_and_shrinks_extremes(self) -> None:
        low = calibrate_probability(0.1)
        middle = calibrate_probability(0.5)
        high = calibrate_probability(0.9)
        self.assertLess(low, middle)
        self.assertLess(middle, high)
        self.assertGreater(low, 0.1)
        self.assertLess(high, 0.9)
        self.assertGreaterEqual(calibrated_standard_deviation(0.5, 0.1), 0.0)

    def test_invalid_role_coordinate_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            layout_from_roles(
                {"cue": [1.1, 0.5], "object1": [0.5, 0.25], "object2": [0.75, 0.8]}
            )


if __name__ == "__main__":
    unittest.main()
