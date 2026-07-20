from __future__ import annotations

import unittest

from symmetric_adaptive_server import layout_from_roles


class SymmetricAdaptiveServerTest(unittest.TestCase):
    def test_layout_from_roles_converts_normalized_coordinates(self) -> None:
        layout = layout_from_roles(
            {"cue": [0.25, 0.5], "object1": [0.5, 0.25], "object2": [0.75, 0.8]}
        )
        self.assertEqual(layout.cue.x_mm, 711.0)
        self.assertEqual(layout.cue.y_mm, 711.0)
        self.assertEqual(layout.object2.x_mm, 2133.0)
        self.assertAlmostEqual(layout.object2.y_mm, 1137.6)

    def test_layout_from_roles_rejects_out_of_range_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            layout_from_roles(
                {"cue": [-0.1, 0.5], "object1": [0.5, 0.25], "object2": [0.75, 0.8]}
            )


if __name__ == "__main__":
    unittest.main()
