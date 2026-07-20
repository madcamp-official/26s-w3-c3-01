from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from cuecast_yolo.shot_probability import (
    AdaptiveGridIndex,
    BootstrapProbabilityModel,
    CatBoostCoordinateModel,
    GridConfig,
    HybridConfig,
    HybridShotProbabilityEngine,
    LayoutMm,
    LogisticCoordinateModel,
    NeighborConfig,
    NeighborIndex,
    PointMm,
    ShotRecord,
    UncertaintyConfig,
    layout_from_normalized_colors,
    layout_features,
    load_continuous_model,
    load_shot_records,
)


def layout(offset: float = 0.0) -> LayoutMm:
    return LayoutMm(
        cue=PointMm(600.0 + offset, 1000.0),
        object1=PointMm(1800.0 + offset, 380.0),
        object2=PointMm(2380.0 + offset, 830.0),
    )


def record(index: int, success: bool, offset: float = 0.0) -> ShotRecord:
    return ShotRecord(
        shot_id=f"shot-{index}",
        cue_ball="white",
        before=layout(offset),
        after=None,
        success=success,
        points=int(success),
        position_error_mm=10.0,
        detection_confidence=0.95,
    )


class ConstantModel:
    version = "constant-test-v1"

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_probability(self, _layout: LayoutMm, _cue_ball: str) -> float:
        return self.probability


class ShotRecordTest(unittest.TestCase):
    def test_round_trip_uses_database_json_shape(self) -> None:
        original = record(1, True)
        restored = ShotRecord.from_dict(original.to_dict())
        self.assertEqual(restored.shot_id, original.shot_id)
        self.assertEqual(restored.before, original.before)
        self.assertTrue(restored.success)

    def test_feature_vector_is_finite(self) -> None:
        features = layout_features(layout(), "white")
        self.assertEqual(features.shape, (30,))
        self.assertTrue(np.isfinite(features).all())

    def test_reads_color_keyed_normalized_db_record(self) -> None:
        db_row = {
            "video_id": "screenrec2",
            "turn": 4,
            "epoch": 3,
            "shooter": "yellow",
            "before": {
                "white": [0.25, 0.50],
                "yellow": [0.50, 0.75],
                "red": [0.75, 0.25],
            },
            "after": {
                "white": [0.30, 0.50],
                "yellow": [0.55, 0.70],
                "red": [0.70, 0.30],
            },
            "success": True,
            "success_detail": {"coverage": 0.92},
        }
        restored = ShotRecord.from_db_dict(db_row)
        self.assertEqual(restored.shot_id, "screenrec2:turn:4:epoch:3")
        self.assertEqual(restored.cue_ball, "yellow")
        self.assertEqual(restored.before.cue, PointMm(1420.0, 1065.0))
        self.assertEqual(restored.before.object1, PointMm(2130.0, 355.0))
        self.assertEqual(restored.before.object2, PointMm(710.0, 710.0))
        self.assertAlmostEqual(restored.detection_confidence, 0.92)

    def test_jsonl_loader_ignores_separators_and_deduplicates(self) -> None:
        line = (
            '{"video_id":"v1","turn":1,"epoch":0,"shooter":"white",'
            '"before":{"white":[0.1,0.2],"yellow":[0.3,0.4],"red":[0.5,0.6]},'
            '"after":{"white":[0.2,0.2],"yellow":[0.3,0.4],"red":[0.5,0.6]},'
            '"success":false,"success_detail":{"coverage":1.0}}'
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "data"
            path.write_text(f"{line}\n---\n{line}\n", encoding="utf-8")
            records = load_shot_records(path)
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0].success)

    def test_db_export_pos_keys_and_top_level_coverage(self) -> None:
        db_row = {
            "video_id": "export",
            "turn": 1,
            "epoch": 0,
            "shooter": "white",
            "before_pos": {
                "white": [0.25, 0.50],
                "yellow": [0.50, 0.75],
                "red": [0.75, 0.25],
            },
            "after_pos": {
                "white": [0.30, 0.50],
                "yellow": [0.55, 0.70],
                "red": [0.70, 0.30],
            },
            "success": True,
            "coverage": 0.87,
        }
        restored = ShotRecord.from_db_dict(db_row)
        self.assertEqual(restored.before.cue, PointMm(710.0, 710.0))
        self.assertIsNotNone(restored.after)
        self.assertAlmostEqual(restored.detection_confidence, 0.87)

    def test_layout_roles_are_deterministic(self) -> None:
        converted, roles = layout_from_normalized_colors(
            {"white": [0.1, 0.2], "yellow": [0.3, 0.4], "red": [0.5, 0.6]},
            "white",
        )
        self.assertEqual(roles, ("white", "red", "yellow"))
        self.assertEqual(converted.object1, PointMm(1420.0, 852.0))


class NeighborIndexTest(unittest.TestCase):
    def test_identical_neighbors_have_full_effective_sample_count(self) -> None:
        records = [record(i, i < 7) for i in range(10)]
        estimate = NeighborIndex(records, NeighborConfig()).estimate(layout())
        self.assertAlmostEqual(estimate.effective_samples, 10.0)
        self.assertAlmostEqual(estimate.probability or 0.0, 0.7)


class AdaptiveGridTest(unittest.TestCase):
    def test_single_observation_is_shrunk_to_model_prior(self) -> None:
        grid = AdaptiveGridIndex([record(1, True)], GridConfig())
        probability, samples, level, _ = grid.smoothed_probability(layout(), 0.72)
        self.assertEqual(samples, 1)
        self.assertEqual(level, 0)
        self.assertAlmostEqual(probability, (1.0 + 30.0 * 0.72) / 31.0)

    def test_validation_logloss_can_enable_local_split(self) -> None:
        records: list[ShotRecord] = []
        for index in range(200):
            left_half = index % 2 == 0
            before = LayoutMm(
                cue=PointMm(610.0 if left_half else 690.0, 1000.0),
                object1=PointMm(1800.0, 380.0),
                object2=PointMm(2380.0, 830.0),
            )
            records.append(
                ShotRecord(
                    shot_id=f"split-{index}",
                    cue_ball="white",
                    before=before,
                    after=None,
                    success=left_half,
                )
            )
        config = GridConfig(
            maximum_level=1,
            minimum_samples_for_split=150,
            minimum_child_samples=25,
            minimum_probability_difference=1.1,
            minimum_validation_logloss_improvement=0.01,
        )
        grid = AdaptiveGridIndex(records, config, ConstantModel(0.5))
        parent_key = grid.state_key(records[0].before, 0)
        diagnostics = grid.split_diagnostics(parent_key, 1)
        self.assertGreater(diagnostics["validationLoglossImprovement"], 0.01)
        self.assertTrue(diagnostics["splitWorthy"])
        self.assertEqual(grid.selected_level(records[0].before), 1)


class LogisticCoordinateModelTest(unittest.TestCase):
    def test_fit_and_json_round_trip(self) -> None:
        records = [
            record(i, success=(i >= 10), offset=float(i * 20)) for i in range(20)
        ]
        model = LogisticCoordinateModel.fit(records, epochs=50)
        restored = LogisticCoordinateModel.from_dict(model.to_dict())
        self.assertAlmostEqual(
            restored.predict_probability(layout(300), "white"),
            model.predict_probability(layout(300), "white"),
        )

    @unittest.skipUnless(CatBoostCoordinateModel.is_available(), "CatBoost not installed")
    def test_catboost_fit_and_artifact_round_trip(self) -> None:
        records = [
            record(i, success=(i % 3 == 0), offset=float(i * 20)) for i in range(12)
        ]
        model = CatBoostCoordinateModel.fit(
            records, iterations=5, depth=2, learning_rate=0.1
        )
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "model.json"
            model.save(manifest)
            restored = load_continuous_model(manifest)
            probability = restored.predict_probability(layout(60), "white")
        self.assertTrue(0.0 <= probability <= 1.0)


class HybridEngineTest(unittest.TestCase):
    def test_prediction_is_database_ready_json(self) -> None:
        config = HybridConfig(
            uncertainty=UncertaintyConfig(monte_carlo_samples=4, random_seed=7)
        )
        engine = HybridShotProbabilityEngine(
            [record(i, i % 2 == 0) for i in range(12)],
            ConstantModel(0.60),
            config,
        )
        result = engine.predict(
            layout(), "white", position_error_mm=18.0, prediction_id="prediction-1"
        )
        self.assertEqual(result["schemaVersion"], "1.0")
        self.assertEqual(result["predictionId"], "prediction-1")
        self.assertTrue(0.0 <= result["successProbability"] <= 1.0)
        self.assertEqual(result["uncertainty"]["samples"], 4)
        self.assertIn(result["confidence"]["level"], ("low", "medium", "high"))

    def test_bootstrap_prior_supports_zero_records(self) -> None:
        engine = HybridShotProbabilityEngine(
            [],
            BootstrapProbabilityModel(0.37),
            HybridConfig(uncertainty=UncertaintyConfig(monte_carlo_samples=2)),
        )
        result = engine.predict(layout(), "white")
        self.assertAlmostEqual(result["successProbability"], 0.37)
        self.assertIn("bootstrap_prior", result["flags"])
        self.assertIn("sparse_neighbors", result["flags"])

    def test_near_cushion_ball_uses_larger_position_error(self) -> None:
        near = LayoutMm(
            cue=PointMm(50.0, 700.0),
            object1=PointMm(1400.0, 700.0),
            object2=PointMm(2200.0, 700.0),
        )
        engine = HybridShotProbabilityEngine(
            [],
            BootstrapProbabilityModel(0.5),
            HybridConfig(uncertainty=UncertaintyConfig(monte_carlo_samples=2)),
        )
        result = engine.predict(near, "white", position_error_mm=25.0)
        errors = result["uncertainty"]["perBallPositionErrorMm"]
        self.assertEqual(errors["cue"], 35.0)
        self.assertEqual(errors["object1"], 25.0)

    def test_bootstrap_model_json_round_trip(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            BootstrapProbabilityModel(0.42).save(path)
            restored = load_continuous_model(path)
        self.assertAlmostEqual(restored.predict_probability(layout(), "white"), 0.42)


if __name__ == "__main__":
    unittest.main()
