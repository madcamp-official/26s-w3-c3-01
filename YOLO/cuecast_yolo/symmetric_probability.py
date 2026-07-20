from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .shot_probability import (
    AdaptiveGridIndex,
    CatBoostCoordinateModel,
    GridConfig,
    GridStat,
    HybridConfig,
    HybridShotProbabilityEngine,
    LayoutMm,
    NeighborEstimate,
    PointMm,
    ShotRecord,
    TABLE_HEIGHT_MM,
    TABLE_WIDTH_MM,
    confidence_label,
    confidence_score,
    layout_features,
)


SYMMETRY_COUNT = 8


def symmetric_layouts(layout: LayoutMm) -> tuple[LayoutMm, ...]:
    """Return table reflections crossed with the two object-ball orders."""
    source = layout.as_array()
    converted: list[LayoutMm] = []
    for reflect_x in (False, True):
        for reflect_y in (False, True):
            reflected = source.copy()
            if reflect_x:
                reflected[:, 0] = TABLE_WIDTH_MM - reflected[:, 0]
            if reflect_y:
                reflected[:, 1] = TABLE_HEIGHT_MM - reflected[:, 1]
            for swap_objects in (False, True):
                points = reflected.copy()
                if swap_objects:
                    points[[1, 2]] = points[[2, 1]]
                converted.append(
                    LayoutMm(
                        *[
                            PointMm(float(x_mm), float(y_mm))
                            for x_mm, y_mm in points
                        ]
                    )
                )
    return tuple(converted)


def canonical_layout(layout: LayoutMm) -> LayoutMm:
    """Choose one deterministic representative for an eight-layout orbit."""
    return min(
        symmetric_layouts(layout),
        key=lambda candidate: tuple(float(value) for value in candidate.as_array().ravel()),
    )


def augment_shot_records(records: Sequence[ShotRecord]) -> list[ShotRecord]:
    """Augment model training rows only; callers must not use these as sample counts."""
    augmented: list[ShotRecord] = []
    for record in records:
        for index, layout in enumerate(symmetric_layouts(record.before)):
            augmented.append(
                ShotRecord(
                    shot_id=f"{record.shot_id}:symmetry:{index}",
                    cue_ball=record.cue_ball,
                    before=layout,
                    after=None,
                    success=record.success,
                    points=record.points,
                    player_id=record.player_id,
                    player_avg=record.player_avg,
                    position_error_mm=record.position_error_mm,
                    detection_confidence=record.detection_confidence,
                )
            )
    return augmented


class SymmetricCatBoostCoordinateModel:
    def __init__(self, base: CatBoostCoordinateModel) -> None:
        self.base = base
        self.version = f"symmetric-v2-{base.version}"

    @classmethod
    def fit(
        cls,
        records: Sequence[ShotRecord],
        *,
        iterations: int = 75,
        depth: int = 3,
        learning_rate: float = 0.03,
        l2_leaf_reg: float = 30.0,
        random_strength: float = 1.0,
    ) -> "SymmetricCatBoostCoordinateModel":
        augmented = augment_shot_records(records)
        try:
            from catboost import CatBoostClassifier
        except ImportError as error:
            raise RuntimeError("CatBoost is required for Symmetric Hybrid v2") from error
        features = np.stack(
            [layout_features(record.before, record.cue_ball) for record in augmented]
        )
        outcomes = np.asarray([int(record.success) for record in augmented])
        sample_weights = np.asarray(
            [
                max(0.05, min(1.0, record.detection_confidence))
                * max(0.05, 1.0 - record.position_error_mm / 100.0)
                for record in augmented
            ]
        )
        model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            random_strength=random_strength,
            loss_function="Logloss",
            random_seed=20260718,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(features, outcomes, sample_weight=sample_weights)
        digest = hashlib.sha256(
            np.asarray(model.predict_proba(features), dtype=np.float64).tobytes()
            + features.tobytes()
        ).hexdigest()[:12]
        base = CatBoostCoordinateModel(
            model, version=f"catboost-coordinate-v1-{digest}"
        )
        return cls(base)

    def predict_probability(self, layout: LayoutMm, cue_ball: str) -> float:
        transformed_layouts = symmetric_layouts(layout)
        if not hasattr(self.base, "model"):
            return float(
                np.mean(
                    [
                        self.base.predict_probability(transformed, cue_ball)
                        for transformed in transformed_layouts
                    ]
                )
            )
        features = np.stack(
            [
                layout_features(transformed, cue_ball)
                for transformed in transformed_layouts
            ]
        )
        probabilities = self.base.model.predict_proba(features)  # type: ignore[attr-defined]
        return float(np.mean(np.asarray(probabilities, dtype=np.float64)[:, 1]))

    def save(self, path: str | Path) -> None:
        manifest = Path(path)
        base_manifest = manifest.with_name(f"{manifest.stem}.base.json")
        self.base.save(base_manifest)
        payload = {
            "type": "symmetric_catboost_coordinate_v2",
            "version": self.version,
            "symmetryCount": SYMMETRY_COUNT,
            "baseManifest": base_manifest.name,
            "training": {
                "iterations": 75,
                "depth": 3,
                "learningRate": 0.03,
                "l2LeafReg": 30.0,
                "randomStrength": 1.0,
                "autoClassWeights": None,
            },
        }
        manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "SymmetricCatBoostCoordinateModel":
        manifest = Path(path)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("type") != "symmetric_catboost_coordinate_v2":
            raise ValueError("Not a symmetric CatBoost v2 model")
        if int(payload.get("symmetryCount", 0)) != SYMMETRY_COUNT:
            raise ValueError("Symmetry definition does not match this engine")
        base = CatBoostCoordinateModel.load(
            manifest.with_name(str(payload["baseManifest"]))
        )
        model = cls(base)
        model.version = str(payload.get("version", model.version))
        return model


class SymmetricNeighborIndex:
    """Search symmetry-equivalent queries without duplicating historical records."""

    def __init__(self, records: Sequence[ShotRecord], config: object) -> None:
        self.records = list(records)
        self.config = config
        self.positions = (
            np.stack([record.before.as_array() for record in records])
            if records
            else np.empty((0, 3, 2), dtype=np.float64)
        )
        self.outcomes = np.asarray([float(record.success) for record in records])

    def estimate(self, layout: LayoutMm) -> NeighborEstimate:
        if not self.records:
            return NeighborEstimate(
                None, 0.0, 0, float(self.config.maximum_radius_mm)  # type: ignore[attr-defined]
            )
        queries = np.stack(
            [transformed.as_array() for transformed in symmetric_layouts(layout)]
        )
        difference = queries[:, None, :, :] - self.positions[None, :, :, :]
        distances = np.min(
            np.sqrt(np.mean(np.sum(difference**2, axis=3), axis=2)), axis=0
        )
        radius = float(self.config.initial_radius_mm)  # type: ignore[attr-defined]
        maximum = float(self.config.maximum_radius_mm)  # type: ignore[attr-defined]
        step = float(self.config.radius_step_mm)  # type: ignore[attr-defined]
        sigma = float(self.config.gaussian_sigma_mm)  # type: ignore[attr-defined]
        target = float(self.config.target_effective_samples)  # type: ignore[attr-defined]
        best: NeighborEstimate | None = None
        while radius <= maximum + 1e-9:
            mask = distances <= radius
            selected = distances[mask]
            if selected.size:
                weights = np.exp(-(selected**2) / (2.0 * sigma**2))
                weight_sum = float(weights.sum())
                effective = weight_sum**2 / max(
                    float(np.square(weights).sum()), 1e-12
                )
                probability = float(
                    np.dot(weights, self.outcomes[mask]) / weight_sum
                )
                best = NeighborEstimate(
                    probability, effective, int(mask.sum()), radius
                )
                if effective >= target:
                    break
            radius += step
        return best or NeighborEstimate(None, 0.0, 0, maximum)


class CanonicalAdaptiveGridIndex(AdaptiveGridIndex):
    """Aggregate symmetry-equivalent state keys once per original shot."""

    def state_key(self, layout: LayoutMm, level: int) -> tuple[int, int, int]:
        keys = [
            tuple(
                self.cell_id(point, level)
                for point in (
                    transformed.cue,
                    transformed.object1,
                    transformed.object2,
                )
            )
            for transformed in symmetric_layouts(layout)
        ]
        return min(keys)


@dataclass(frozen=True)
class PlattCalibration:
    intercept: float = 0.0
    slope: float = 1.0

    def apply(self, probability: float) -> float:
        clipped = min(1.0 - 1e-7, max(1e-7, probability))
        logit = math.log(clipped / (1.0 - clipped))
        value = self.intercept + self.slope * logit
        return 1.0 / (1.0 + math.exp(-value))

    def to_dict(self) -> dict[str, float]:
        return {"intercept": self.intercept, "slope": self.slope}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PlattCalibration":
        return cls(float(value["intercept"]), float(value["slope"]))


def fit_platt_calibration(
    probabilities: Sequence[float], outcomes: Sequence[bool | float]
) -> PlattCalibration:
    probabilities_array = np.clip(
        np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7
    )
    outcomes_array = np.asarray(outcomes, dtype=np.float64)
    logits = np.log(probabilities_array / (1.0 - probabilities_array))
    design = np.column_stack([np.ones_like(logits), logits])
    parameters = np.asarray([0.0, 1.0], dtype=np.float64)
    for _ in range(100):
        values = np.clip(design @ parameters, -35.0, 35.0)
        fitted = 1.0 / (1.0 + np.exp(-values))
        gradient = design.T @ (fitted - outcomes_array)
        variance = np.maximum(fitted * (1.0 - fitted), 1e-8)
        hessian = design.T @ (design * variance[:, None])
        step = np.linalg.solve(hessian + np.eye(2) * 1e-8, gradient)
        parameters -= step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return PlattCalibration(float(parameters[0]), float(parameters[1]))


DEFAULT_SYMMETRIC_GRID = GridConfig(
    base_columns=4,
    base_rows=2,
    maximum_level=2,
    minimum_samples_for_split=6,
    preferred_samples_for_split=12,
    minimum_child_samples=2,
    minimum_probability_difference=0.20,
    minimum_validation_logloss_improvement=0.02,
    prior_strengths=(16.0, 10.0, 7.0),
)


class SymmetricHybridShotProbabilityEngine(HybridShotProbabilityEngine):
    def __init__(
        self,
        records: Sequence[ShotRecord],
        model: SymmetricCatBoostCoordinateModel,
        *,
        grid_config: GridConfig = DEFAULT_SYMMETRIC_GRID,
        calibration: PlattCalibration | None = None,
    ) -> None:
        config = HybridConfig(grid=grid_config)
        self.records = list(records)
        self.model = model
        self.config = config
        self.calibration = calibration or PlattCalibration()
        self.grid = CanonicalAdaptiveGridIndex(records, grid_config, model)
        self.neighbors = SymmetricNeighborIndex(records, config.neighbors)

    @staticmethod
    def _weights(
        effective_samples: float, grid_samples: int
    ) -> tuple[float, float, float]:
        neighbor_weight = (
            0.0 if effective_samples < 5.0 else 0.05 if effective_samples < 10.0 else 0.10
        )
        grid_weight = 0.05 if grid_samples >= 3 else 0.0
        return 1.0 - neighbor_weight - grid_weight, neighbor_weight, grid_weight

    def _predict_once(self, layout: LayoutMm, cue_ball: str) -> dict[str, object]:
        model_probability = float(
            np.clip(self.model.predict_probability(layout, cue_ball), 0.0, 1.0)
        )
        neighbor = self.neighbors.estimate(layout)
        grid_probability, grid_samples, grid_level, grid_key = (
            self.grid.smoothed_probability(layout, model_probability)
        )
        neighbor_probability = (
            neighbor.probability
            if neighbor.probability is not None
            else model_probability
        )
        weights = self._weights(neighbor.effective_samples, grid_samples)
        raw_probability = (
            weights[0] * model_probability
            + weights[1] * neighbor_probability
            + weights[2] * grid_probability
        )
        probability = self.calibration.apply(raw_probability)
        components = [model_probability, neighbor_probability, grid_probability]
        next_level = grid_level + 1
        split_diagnostics = None
        if next_level <= self.config.grid.maximum_level:
            parent_key = self.grid.state_key(layout, grid_level)
            split_diagnostics = self.grid.split_diagnostics(parent_key, next_level)
        return {
            "probability": probability,
            "rawProbability": raw_probability,
            "modelProbability": model_probability,
            "neighborProbability": neighbor.probability,
            "gridProbability": grid_probability,
            "weights": {
                "model": weights[0],
                "neighbor": weights[1],
                "grid": weights[2],
            },
            "symmetryPredictions": SYMMETRY_COUNT,
            "neighborEffectiveSamples": neighbor.effective_samples,
            "neighborRawSamples": neighbor.raw_samples,
            "neighborRadiusMm": neighbor.radius_mm,
            "gridSamples": grid_samples,
            "gridLevel": grid_level,
            "gridStateKey": list(grid_key),
            "nextGridSplit": split_diagnostics,
            "modelDisagreement": max(components) - min(components),
        }

    def predict(
        self,
        layout: LayoutMm,
        cue_ball: str,
        *,
        position_error_mm: float | None = None,
        prediction_id: str | None = None,
    ) -> dict[str, object]:
        canonical = canonical_layout(layout)
        result = super().predict(
            canonical,
            cue_ball,
            position_error_mm=position_error_mm,
            prediction_id=prediction_id,
        )
        result["before"] = layout.to_dict()
        result["canonicalBefore"] = canonical.to_dict()
        result["engineVersion"] = "symmetric-hybrid-v2"
        result["calibration"] = self.calibration.to_dict()
        return result


def save_calibration(calibration: PlattCalibration, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_calibration(path: str | Path) -> PlattCalibration:
    return PlattCalibration.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
