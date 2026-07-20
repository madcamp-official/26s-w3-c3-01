from __future__ import annotations

import unittest

from cuecast_yolo.shot_probability import (
    GridConfig,
    LayoutMm,
    NeighborConfig,
    PointMm,
    ShotRecord,
    TABLE_WIDTH_MM,
)
from cuecast_yolo.symmetric_probability import (
    CanonicalAdaptiveGridIndex,
    PlattCalibration,
    SymmetricCatBoostCoordinateModel,
    SymmetricHybridShotProbabilityEngine,
    SymmetricNeighborIndex,
    augment_shot_records,
    canonical_layout,
    symmetric_layouts,
)


def layout() -> LayoutMm:
    return LayoutMm(
        PointMm(400.0, 300.0),
        PointMm(1200.0, 500.0),
        PointMm(2300.0, 1100.0),
    )


def record(index: int, before: LayoutMm | None = None, success: bool = True) -> ShotRecord:
    return ShotRecord(
        shot_id=f"shot-{index}",
        cue_ball="yellow",
        before=before or layout(),
        after=None,
        success=success,
    )


class FakeBaseModel:
    version = "fake-base"

    def predict_probability(self, value: LayoutMm, _cue_ball: str) -> float:
        points = value.as_array()
        return float(0.2 + 0.6 * points[0, 0] / TABLE_WIDTH_MM)


class SymmetryTransformTest(unittest.TestCase):
    def test_eight_unique_layouts_are_created(self) -> None:
        transformed = symmetric_layouts(layout())
        self.assertEqual(len(transformed), 8)
        arrays = {tuple(item.as_array().ravel()) for item in transformed}
        self.assertEqual(len(arrays), 8)

    def test_canonical_layout_is_identical_for_every_symmetry(self) -> None:
        expected = canonical_layout(layout())
        for transformed in symmetric_layouts(layout()):
            self.assertEqual(canonical_layout(transformed), expected)

    def test_training_augmentation_does_not_change_original_records(self) -> None:
        original = [record(1), record(2, success=False)]
        augmented = augment_shot_records(original)
        self.assertEqual(len(augmented), 16)
        self.assertEqual(len(original), 2)


class SymmetricModelTest(unittest.TestCase):
    def test_prediction_is_invariant(self) -> None:
        model = SymmetricCatBoostCoordinateModel(FakeBaseModel())  # type: ignore[arg-type]
        expected = model.predict_probability(layout(), "yellow")
        for transformed in symmetric_layouts(layout()):
            self.assertAlmostEqual(
                model.predict_probability(transformed, "yellow"), expected
            )


class SymmetricDataIndexTest(unittest.TestCase):
    def test_neighbor_counts_each_original_record_once(self) -> None:
        index = SymmetricNeighborIndex([record(1)], NeighborConfig())
        estimate = index.estimate(layout())
        self.assertEqual(estimate.raw_samples, 1)
        self.assertAlmostEqual(estimate.effective_samples, 1.0)

    def test_grid_counts_each_original_record_once(self) -> None:
        records = [record(1), record(2, success=False)]
        grid = CanonicalAdaptiveGridIndex(
            records,
            GridConfig(base_columns=4, base_rows=2),
        )
        for level in grid.levels:
            self.assertEqual(sum(stat.attempts for stat in level.values()), 2)

    def test_grid_key_is_invariant(self) -> None:
        grid = CanonicalAdaptiveGridIndex(
            [record(1)], GridConfig(base_columns=4, base_rows=2)
        )
        expected = grid.state_key(layout(), 0)
        for transformed in symmetric_layouts(layout()):
            self.assertEqual(grid.state_key(transformed, 0), expected)


class SymmetricEngineTest(unittest.TestCase):
    def test_full_prediction_is_invariant_and_does_not_inflate_samples(self) -> None:
        model = SymmetricCatBoostCoordinateModel(FakeBaseModel())  # type: ignore[arg-type]
        engine = SymmetricHybridShotProbabilityEngine(
            [record(1)], model, calibration=PlattCalibration()
        )
        expected = engine.predict(layout(), "yellow", prediction_id="same")
        for transformed in symmetric_layouts(layout()):
            result = engine.predict(transformed, "yellow", prediction_id="same")
            self.assertAlmostEqual(
                result["successProbability"], expected["successProbability"]
            )
            self.assertEqual(result["components"]["neighborRawSamples"], 1)
            self.assertEqual(result["components"]["gridSamples"], 1)


if __name__ == "__main__":
    unittest.main()
