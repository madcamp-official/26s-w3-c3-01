from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


TABLE_WIDTH_MM = 2840.0
TABLE_HEIGHT_MM = 1420.0
BALL_COLORS = ("white", "yellow", "red")


@dataclass(frozen=True)
class PointMm:
    x_mm: float
    y_mm: float

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PointMm":
        return cls(float(value["xMm"]), float(value["yMm"]))

    def to_dict(self) -> dict[str, float]:
        return {"xMm": self.x_mm, "yMm": self.y_mm}


@dataclass(frozen=True)
class LayoutMm:
    cue: PointMm
    object1: PointMm
    object2: PointMm

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "LayoutMm":
        return cls(
            cue=PointMm.from_dict(value["cue"]),  # type: ignore[arg-type]
            object1=PointMm.from_dict(value["object1"]),  # type: ignore[arg-type]
            object2=PointMm.from_dict(value["object2"]),  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            "cue": self.cue.to_dict(),
            "object1": self.object1.to_dict(),
            "object2": self.object2.to_dict(),
        }

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                [self.cue.x_mm, self.cue.y_mm],
                [self.object1.x_mm, self.object1.y_mm],
                [self.object2.x_mm, self.object2.y_mm],
            ],
            dtype=np.float64,
        )


def ordered_object_colors(shooter: str) -> tuple[str, str]:
    if shooter not in BALL_COLORS:
        raise ValueError(f"Unknown shooter color: {shooter}")
    objects = sorted(color for color in BALL_COLORS if color != shooter)
    return objects[0], objects[1]


def layout_from_normalized_colors(
    positions: dict[str, object], shooter: str
) -> tuple[LayoutMm, tuple[str, str, str]]:
    """Convert the DB's color-keyed 0..1 coordinates to model roles in mm."""
    object1, object2 = ordered_object_colors(shooter)
    roles = (shooter, object1, object2)
    points: list[PointMm] = []
    for color in roles:
        raw = positions.get(color)
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError(f"Missing or invalid {color} coordinate")
        x, y = float(raw[0]), float(raw[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"{color} coordinate must be normalized to 0..1")
        points.append(PointMm(x * TABLE_WIDTH_MM, y * TABLE_HEIGHT_MM))
    return LayoutMm(*points), roles


@dataclass(frozen=True)
class ShotRecord:
    shot_id: str
    cue_ball: str
    before: LayoutMm
    after: LayoutMm | None
    success: bool
    points: int = 0
    player_id: str | None = None
    player_avg: float | None = None
    position_error_mm: float = 25.0
    detection_confidence: float = 1.0

    @classmethod
    def from_db_dict(cls, value: dict[str, object]) -> "ShotRecord":
        """Read the newline-delimited DB record used by ``data``."""
        shooter = str(value["shooter"])
        before_value = value.get("before", value.get("before_pos"))
        if isinstance(before_value, str):
            before_value = json.loads(before_value)
        if not isinstance(before_value, dict):
            raise ValueError("DB shot record requires before or before_pos coordinates")
        before, _ = layout_from_normalized_colors(
            before_value, shooter
        )
        after_value = value.get("after", value.get("after_pos"))
        if isinstance(after_value, str):
            after_value = json.loads(after_value)
        after = (
            layout_from_normalized_colors(after_value, shooter)[0]  # type: ignore[arg-type]
            if isinstance(after_value, dict)
            else None
        )
        detail = value.get("success_detail") or {}
        coverage = float(
            detail.get("coverage", value.get("coverage", 1.0))  # type: ignore[union-attr]
        )
        shot_id = (
            f"{value['video_id']}:turn:{int(value['turn'])}:epoch:{int(value['epoch'])}"
        )
        return cls(
            shot_id=shot_id,
            cue_ball=shooter,
            before=before,
            after=after,
            success=bool(value["success"]),
            points=int(bool(value["success"])),
            position_error_mm=25.0,
            detection_confidence=max(0.0, min(1.0, coverage)),
        )

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ShotRecord":
        player = value.get("player") or {}
        quality = value.get("quality") or {}
        return cls(
            shot_id=str(value["shotId"]),
            cue_ball=str(value["cueBall"]),
            before=LayoutMm.from_dict(value["before"]),  # type: ignore[arg-type]
            after=(
                LayoutMm.from_dict(value["after"])  # type: ignore[arg-type]
                if value.get("after") is not None
                else None
            ),
            success=bool(value["success"]),
            points=int(value.get("points", int(bool(value["success"])))),
            player_id=player.get("playerId"),  # type: ignore[union-attr]
            player_avg=(
                float(player["avg"])  # type: ignore[index]
                if player.get("avg") is not None  # type: ignore[union-attr]
                else None
            ),
            position_error_mm=float(quality.get("positionErrorMm", 25.0)),  # type: ignore[union-attr]
            detection_confidence=float(quality.get("detectionConfidence", 1.0)),  # type: ignore[union-attr]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "shotId": self.shot_id,
            "cueBall": self.cue_ball,
            "before": self.before.to_dict(),
            "after": self.after.to_dict() if self.after else None,
            "success": self.success,
            "points": self.points,
            "player": {"playerId": self.player_id, "avg": self.player_avg},
            "quality": {
                "positionErrorMm": self.position_error_mm,
                "detectionConfidence": self.detection_confidence,
            },
        }


def load_shot_records(path: str | Path) -> list[ShotRecord]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [
            json.loads(line)
            for line in text.splitlines()
            if line.lstrip().startswith("{")
        ]
    if isinstance(payload, dict):
        payload = payload.get("shots", [payload])

    unique: dict[str, ShotRecord] = {}
    for item in payload:
        record = (
            ShotRecord.from_db_dict(item)
            if "video_id" in item and "shooter" in item
            else ShotRecord.from_dict(item)
        )
        previous = unique.get(record.shot_id)
        if previous is not None and previous != record:
            raise ValueError(f"Conflicting duplicate shot id: {record.shot_id}")
        unique[record.shot_id] = record
    return list(unique.values())


class ContinuousProbabilityModel(Protocol):
    version: str

    def predict_probability(self, layout: LayoutMm, cue_ball: str) -> float:
        ...

    def save(self, path: str | Path) -> None:
        ...


FEATURE_NAMES = (
    "cue_x",
    "cue_y",
    "object1_x",
    "object1_y",
    "object2_x",
    "object2_y",
    "cue_object1_distance",
    "cue_object2_distance",
    "object1_object2_distance",
    "object_angle",
    "triangle_area",
    "cue_left",
    "cue_right",
    "cue_top",
    "cue_bottom",
    "object1_left",
    "object1_right",
    "object1_top",
    "object1_bottom",
    "object2_left",
    "object2_right",
    "object2_top",
    "object2_bottom",
    "cue_near_cushion",
    "object1_near_cushion",
    "object2_near_cushion",
    "cue_object1_close",
    "cue_object2_close",
    "object1_object2_close",
    "cue_is_white",
)


def layout_features(layout: LayoutMm, cue_ball: str) -> np.ndarray:
    points = layout.as_array()
    normalized = points / np.asarray([TABLE_WIDTH_MM, TABLE_HEIGHT_MM])
    cue, object1, object2 = points
    d01 = float(np.linalg.norm(cue - object1))
    d02 = float(np.linalg.norm(cue - object2))
    d12 = float(np.linalg.norm(object1 - object2))
    vector1, vector2 = object1 - cue, object2 - cue
    denominator = max(d01 * d02, 1e-9)
    cosine = float(np.clip(np.dot(vector1, vector2) / denominator, -1.0, 1.0))
    angle = math.acos(cosine) / math.pi
    cross_2d = vector1[0] * vector2[1] - vector1[1] * vector2[0]
    triangle_area = abs(float(cross_2d)) / (
        TABLE_WIDTH_MM * TABLE_HEIGHT_MM
    )
    cushion_distances: list[float] = []
    near_cushion: list[float] = []
    for x_mm, y_mm in points:
        distances = (
            x_mm,
            TABLE_WIDTH_MM - x_mm,
            y_mm,
            TABLE_HEIGHT_MM - y_mm,
        )
        cushion_distances.extend(
            (
                distances[0] / TABLE_WIDTH_MM,
                distances[1] / TABLE_WIDTH_MM,
                distances[2] / TABLE_HEIGHT_MM,
                distances[3] / TABLE_HEIGHT_MM,
            )
        )
        near_cushion.append(float(min(distances) <= 120.0))
    close = [float(distance <= 150.0) for distance in (d01, d02, d12)]
    return np.asarray(
        [
            *normalized.ravel(),
            d01 / TABLE_WIDTH_MM,
            d02 / TABLE_WIDTH_MM,
            d12 / TABLE_WIDTH_MM,
            angle,
            triangle_area,
            *cushion_distances,
            *near_cushion,
            *close,
            float(cue_ball.lower() == "white"),
        ],
        dtype=np.float64,
    )


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(value, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class BootstrapProbabilityModel:
    """Cold-start prior used before a trainable coordinate model exists."""

    probability: float = 0.5
    version: str = "bootstrap-prior-v1"

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("Bootstrap probability must be between 0 and 1")
        self.version = f"bootstrap-prior-v1-p{self.probability:.4f}"

    def predict_probability(self, _layout: LayoutMm, _cue_ball: str) -> float:
        return self.probability

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "bootstrap_prior",
            "version": self.version,
            "probability": self.probability,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "BootstrapProbabilityModel":
        return cls(probability=float(value["probability"]))


@dataclass
class LogisticCoordinateModel:
    """Dependency-free baseline continuous model.

    The JSON interface is intentionally small so this model can later be replaced
    with a CatBoost/LightGBM adapter implementing ``predict_probability``.
    """

    weights: np.ndarray
    bias: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    version: str = "logistic-coordinate-v1"

    @classmethod
    def fit(
        cls,
        records: Sequence[ShotRecord],
        *,
        epochs: int = 2500,
        learning_rate: float = 0.05,
        l2: float = 0.002,
    ) -> "LogisticCoordinateModel":
        if not records:
            raise ValueError("At least one shot record is required")
        x = np.stack([layout_features(r.before, r.cue_ball) for r in records])
        y = np.asarray([float(r.success) for r in records])
        sample_weights = np.asarray(
            [
                max(0.05, min(1.0, r.detection_confidence))
                * max(0.05, 1.0 - r.position_error_mm / 100.0)
                for r in records
            ]
        )
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-8] = 1.0
        z = (x - mean) / scale
        weights = np.zeros(z.shape[1], dtype=np.float64)
        weighted_rate = float(np.average(y, weights=sample_weights))
        bias = math.log((weighted_rate + 1e-4) / (1.0001 - weighted_rate))
        normalizer = float(sample_weights.sum())
        for _ in range(epochs):
            prediction = _sigmoid(z @ weights + bias)
            residual = (prediction - y) * sample_weights
            weights -= learning_rate * ((z.T @ residual) / normalizer + l2 * weights)
            bias -= learning_rate * float(residual.sum() / normalizer)
        signature = hashlib.sha256(
            b"".join(array.tobytes() for array in (weights, mean, scale))
            + np.float64(bias).tobytes()
        ).hexdigest()[:12]
        return cls(
            weights,
            bias,
            mean,
            scale,
            version=f"logistic-coordinate-v1-{signature}",
        )

    def predict_probability(self, layout: LayoutMm, cue_ball: str) -> float:
        features = (layout_features(layout, cue_ball) - self.feature_mean) / self.feature_scale
        return float(_sigmoid(float(features @ self.weights + self.bias)))

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "logistic_coordinate",
            "version": self.version,
            "featureNames": list(FEATURE_NAMES),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "featureMean": self.feature_mean.tolist(),
            "featureScale": self.feature_scale.tolist(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "LogisticCoordinateModel":
        if value.get("featureNames") != list(FEATURE_NAMES):
            raise ValueError("Model feature definition does not match this code version")
        return cls(
            weights=np.asarray(value["weights"], dtype=np.float64),
            bias=float(value["bias"]),
            feature_mean=np.asarray(value["featureMean"], dtype=np.float64),
            feature_scale=np.asarray(value["featureScale"], dtype=np.float64),
            version=str(value.get("version", "logistic-coordinate-v1")),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "LogisticCoordinateModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class CatBoostCoordinateModel:
    """CatBoost-backed continuous model with a small JSON manifest."""

    def __init__(self, model: object, version: str = "catboost-coordinate-v1") -> None:
        self.model = model
        self.version = version

    @staticmethod
    def is_available() -> bool:
        try:
            import catboost  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def fit(
        cls,
        records: Sequence[ShotRecord],
        *,
        iterations: int = 400,
        depth: int = 6,
        learning_rate: float = 0.05,
    ) -> "CatBoostCoordinateModel":
        if not records:
            raise ValueError("At least one shot record is required")
        outcomes = [int(record.success) for record in records]
        if len(set(outcomes)) < 2:
            raise ValueError("CatBoost requires both success and failure records")
        try:
            from catboost import CatBoostClassifier
        except ImportError as error:
            raise RuntimeError(
                "CatBoost is not installed. Run: pip install catboost"
            ) from error
        features = np.stack(
            [layout_features(record.before, record.cue_ball) for record in records]
        )
        sample_weights = np.asarray(
            [
                max(0.05, min(1.0, record.detection_confidence))
                * max(0.05, 1.0 - record.position_error_mm / 100.0)
                for record in records
            ]
        )
        model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            random_seed=20260718,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(features, outcomes, sample_weight=sample_weights)
        digest = hashlib.sha256(
            np.asarray(model.predict_proba(features), dtype=np.float64).tobytes()
            + features.tobytes()
        ).hexdigest()[:12]
        return cls(model, version=f"catboost-coordinate-v1-{digest}")

    def predict_probability(self, layout: LayoutMm, cue_ball: str) -> float:
        features = layout_features(layout, cue_ball).reshape(1, -1)
        probabilities = self.model.predict_proba(features)  # type: ignore[attr-defined]
        return float(probabilities[0][1])

    def save(self, path: str | Path) -> None:
        manifest = Path(path)
        artifact = manifest.with_suffix(".cbm")
        self.model.save_model(str(artifact))  # type: ignore[attr-defined]
        payload = {
            "type": "catboost_coordinate",
            "version": self.version,
            "featureNames": list(FEATURE_NAMES),
            "artifact": artifact.name,
        }
        manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "CatBoostCoordinateModel":
        manifest = Path(path)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("featureNames") != list(FEATURE_NAMES):
            raise ValueError("Model feature definition does not match this code version")
        try:
            from catboost import CatBoostClassifier
        except ImportError as error:
            raise RuntimeError("CatBoost is required to load this model") from error
        model = CatBoostClassifier()
        model.load_model(str(manifest.with_name(payload["artifact"])))
        return cls(model, version=str(payload["version"]))


def load_continuous_model(path: str | Path) -> ContinuousProbabilityModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    model_type = payload.get("type")
    if model_type == "bootstrap_prior":
        return BootstrapProbabilityModel.from_dict(payload)
    if model_type == "logistic_coordinate":
        return LogisticCoordinateModel.from_dict(payload)
    if model_type == "catboost_coordinate":
        return CatBoostCoordinateModel.load(path)
    raise ValueError(f"Unsupported model type: {model_type}")


@dataclass(frozen=True)
class GridConfig:
    base_columns: int = 24
    base_rows: int = 12
    maximum_level: int = 2
    minimum_samples_for_split: int = 150
    preferred_samples_for_split: int = 300
    minimum_child_samples: int = 25
    minimum_probability_difference: float = 0.15
    minimum_validation_logloss_improvement: float = 0.02
    prior_strengths: tuple[float, float, float] = (30.0, 20.0, 15.0)


@dataclass(frozen=True)
class NeighborConfig:
    initial_radius_mm: float = 30.0
    maximum_radius_mm: float = 120.0
    gaussian_sigma_mm: float = 45.0
    minimum_effective_samples: float = 10.0
    target_effective_samples: float = 50.0
    radius_step_mm: float = 15.0


@dataclass(frozen=True)
class UncertaintyConfig:
    monte_carlo_samples: int = 32
    default_position_error_mm: float = 25.0
    near_cushion_error_mm: float = 35.0
    near_cushion_threshold_mm: float = 120.0
    reject_position_error_mm: float = 50.0
    random_seed: int = 20260718


@dataclass(frozen=True)
class HybridConfig:
    grid: GridConfig = field(default_factory=GridConfig)
    neighbors: NeighborConfig = field(default_factory=NeighborConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)


@dataclass
class GridStat:
    attempts: int = 0
    successes: int = 0

    @property
    def rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0


class AdaptiveGridIndex:
    def __init__(
        self,
        records: Sequence[ShotRecord],
        config: GridConfig,
        model: ContinuousProbabilityModel | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.levels: list[dict[tuple[int, int, int], GridStat]] = [
            {} for _ in range(config.maximum_level + 1)
        ]
        self.training_levels: list[dict[tuple[int, int, int], GridStat]] = [
            {} for _ in range(config.maximum_level + 1)
        ]
        self.validation_records: list[ShotRecord] = []
        for record in records:
            validation = self._is_validation_record(record)
            if validation:
                self.validation_records.append(record)
            for level in range(config.maximum_level + 1):
                key = self.state_key(record.before, level)
                stat = self.levels[level].setdefault(key, GridStat())
                stat.attempts += 1
                stat.successes += int(record.success)
                if not validation:
                    training_stat = self.training_levels[level].setdefault(key, GridStat())
                    training_stat.attempts += 1
                    training_stat.successes += int(record.success)
        self._validation_improvements: dict[
            tuple[int, tuple[int, int, int]], float
        ] = {}

    @staticmethod
    def _is_validation_record(record: ShotRecord) -> bool:
        digest = hashlib.sha256(record.shot_id.encode("utf-8")).digest()
        return digest[0] % 5 == 0

    def dimensions(self, level: int) -> tuple[int, int]:
        factor = 2**level
        return self.config.base_columns * factor, self.config.base_rows * factor

    def cell_id(self, point: PointMm, level: int) -> int:
        columns, rows = self.dimensions(level)
        column = min(columns - 1, max(0, int(point.x_mm / TABLE_WIDTH_MM * columns)))
        row = min(rows - 1, max(0, int(point.y_mm / TABLE_HEIGHT_MM * rows)))
        return row * columns + column

    def state_key(self, layout: LayoutMm, level: int) -> tuple[int, int, int]:
        return tuple(
            self.cell_id(point, level)
            for point in (layout.cue, layout.object1, layout.object2)
        )  # type: ignore[return-value]

    def _parent_cell(self, cell: int, child_level: int) -> int:
        child_columns, _ = self.dimensions(child_level)
        parent_columns, _ = self.dimensions(child_level - 1)
        row, column = divmod(cell, child_columns)
        return (row // 2) * parent_columns + column // 2

    def _observed_children(
        self, parent_key: tuple[int, int, int], child_level: int
    ) -> list[GridStat]:
        result: list[GridStat] = []
        for key, stat in self.levels[child_level].items():
            parents = tuple(self._parent_cell(cell, child_level) for cell in key)
            if parents == parent_key and stat.attempts >= self.config.minimum_child_samples:
                result.append(stat)
        return result

    @staticmethod
    def _logloss(outcome: bool, probability: float) -> float:
        clipped = min(1.0 - 1e-7, max(1e-7, probability))
        return -math.log(clipped if outcome else 1.0 - clipped)

    def _training_probability(
        self, layout: LayoutMm, model_probability: float, through_level: int
    ) -> float:
        probability = model_probability
        for level in range(through_level + 1):
            key = self.state_key(layout, level)
            stat = self.training_levels[level].get(key, GridStat())
            strength = self.config.prior_strengths[level]
            probability = (stat.successes + strength * probability) / (
                stat.attempts + strength
            )
        return probability

    def validation_logloss_improvement(
        self, parent_key: tuple[int, int, int], child_level: int
    ) -> float:
        cache_key = (child_level, parent_key)
        if cache_key in self._validation_improvements:
            return self._validation_improvements[cache_key]
        if self.model is None:
            self._validation_improvements[cache_key] = 0.0
            return 0.0
        relevant = [
            record
            for record in self.validation_records
            if self.state_key(record.before, child_level - 1) == parent_key
        ]
        if len(relevant) < 5:
            self._validation_improvements[cache_key] = 0.0
            return 0.0
        parent_losses: list[float] = []
        child_losses: list[float] = []
        for record in relevant:
            p_model = self.model.predict_probability(record.before, record.cue_ball)
            p_parent = self._training_probability(
                record.before, p_model, child_level - 1
            )
            child_key = self.state_key(record.before, child_level)
            child = self.training_levels[child_level].get(child_key, GridStat())
            strength = self.config.prior_strengths[child_level]
            p_child = (child.successes + strength * p_parent) / (
                child.attempts + strength
            )
            parent_losses.append(self._logloss(record.success, p_parent))
            child_losses.append(self._logloss(record.success, p_child))
        improvement = float(np.mean(parent_losses) - np.mean(child_losses))
        self._validation_improvements[cache_key] = improvement
        return improvement

    def split_diagnostics(
        self, parent_key: tuple[int, int, int], child_level: int
    ) -> dict[str, float | int | bool]:
        parent = self.levels[child_level - 1].get(parent_key, GridStat())
        children = self._observed_children(parent_key, child_level)
        probability_difference = (
            max(stat.rate for stat in children) - min(stat.rate for stat in children)
            if len(children) >= 2
            else 0.0
        )
        logloss_improvement = self.validation_logloss_improvement(
            parent_key, child_level
        )
        enough_samples = parent.attempts >= self.config.minimum_samples_for_split
        worthwhile = (
            probability_difference >= self.config.minimum_probability_difference
            or logloss_improvement
            >= self.config.minimum_validation_logloss_improvement
        )
        return {
            "parentSamples": parent.attempts,
            "preferredSamplesReached": (
                parent.attempts >= self.config.preferred_samples_for_split
            ),
            "eligibleChildren": len(children),
            "probabilityDifference": probability_difference,
            "validationLoglossImprovement": logloss_improvement,
            "splitWorthy": enough_samples and len(children) >= 2 and worthwhile,
        }

    def selected_level(self, layout: LayoutMm) -> int:
        selected = 0
        for child_level in range(1, self.config.maximum_level + 1):
            parent_key = self.state_key(layout, child_level - 1)
            child_key = self.state_key(layout, child_level)
            parent = self.levels[child_level - 1].get(parent_key, GridStat())
            child = self.levels[child_level].get(child_key, GridStat())
            diagnostics = self.split_diagnostics(parent_key, child_level)
            if (
                parent.attempts < self.config.minimum_samples_for_split
                or child.attempts < self.config.minimum_child_samples
                or not bool(diagnostics["splitWorthy"])
            ):
                break
            selected = child_level
        return selected

    def smoothed_probability(
        self, layout: LayoutMm, model_probability: float
    ) -> tuple[float, int, int, tuple[int, int, int]]:
        selected = self.selected_level(layout)
        probability = model_probability
        samples = 0
        key = self.state_key(layout, 0)
        for level in range(selected + 1):
            key = self.state_key(layout, level)
            stat = self.levels[level].get(key, GridStat())
            strength = self.config.prior_strengths[level]
            probability = (stat.successes + strength * probability) / (
                stat.attempts + strength
            )
            samples = stat.attempts
        return probability, samples, selected, key


@dataclass(frozen=True)
class NeighborEstimate:
    probability: float | None
    effective_samples: float
    raw_samples: int
    radius_mm: float


class NeighborIndex:
    def __init__(self, records: Sequence[ShotRecord], config: NeighborConfig) -> None:
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
            return NeighborEstimate(None, 0.0, 0, self.config.maximum_radius_mm)
        difference = self.positions - layout.as_array()[None, :, :]
        distances = np.sqrt(np.mean(np.sum(difference**2, axis=2), axis=1))
        radius = self.config.initial_radius_mm
        best: NeighborEstimate | None = None
        while radius <= self.config.maximum_radius_mm + 1e-9:
            mask = distances <= radius
            selected = distances[mask]
            if selected.size:
                weights = np.exp(
                    -(selected**2) / (2.0 * self.config.gaussian_sigma_mm**2)
                )
                weight_sum = float(weights.sum())
                effective = weight_sum**2 / max(float(np.square(weights).sum()), 1e-12)
                probability = float(np.dot(weights, self.outcomes[mask]) / weight_sum)
                best = NeighborEstimate(probability, effective, int(mask.sum()), radius)
                if effective >= self.config.target_effective_samples:
                    break
            radius += self.config.radius_step_mm
        return best or NeighborEstimate(None, 0.0, 0, self.config.maximum_radius_mm)


def select_weights(effective_samples: float, grid_samples: int) -> tuple[float, float, float]:
    if effective_samples < 5:
        return 0.85, 0.10, 0.05
    if effective_samples < 20:
        return 0.65, 0.25, 0.10
    if effective_samples < 80:
        return 0.45, 0.40, 0.15
    if grid_samples >= 100:
        return 0.25, 0.50, 0.25
    return 0.35, 0.50, 0.15


def confidence_score(
    effective_samples: float,
    grid_samples: int,
    position_error_mm: float,
    model_disagreement: float,
) -> float:
    sample_score = min(effective_samples / 80.0, 1.0)
    grid_score = min(grid_samples / 100.0, 1.0)
    position_score = max(0.0, 1.0 - position_error_mm / 50.0)
    agreement_score = max(0.0, 1.0 - model_disagreement / 0.30)
    return float(
        0.35 * sample_score
        + 0.20 * grid_score
        + 0.25 * position_score
        + 0.20 * agreement_score
    )


def confidence_label(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


class HybridShotProbabilityEngine:
    def __init__(
        self,
        records: Sequence[ShotRecord],
        model: ContinuousProbabilityModel,
        config: HybridConfig | None = None,
    ) -> None:
        self.records = list(records)
        self.model = model
        self.config = config or HybridConfig()
        self.grid = AdaptiveGridIndex(records, self.config.grid, model)
        self.neighbors = NeighborIndex(records, self.config.neighbors)

    def _predict_once(self, layout: LayoutMm, cue_ball: str) -> dict[str, object]:
        model_probability = float(
            np.clip(self.model.predict_probability(layout, cue_ball), 0.0, 1.0)
        )
        neighbor = self.neighbors.estimate(layout)
        grid_probability, grid_samples, grid_level, grid_key = (
            self.grid.smoothed_probability(layout, model_probability)
        )
        neighbor_probability = (
            neighbor.probability if neighbor.probability is not None else model_probability
        )
        weights = select_weights(neighbor.effective_samples, grid_samples)
        final = (
            weights[0] * model_probability
            + weights[1] * neighbor_probability
            + weights[2] * grid_probability
        )
        components = [model_probability, neighbor_probability, grid_probability]
        next_level = grid_level + 1
        split_diagnostics = None
        if next_level <= self.config.grid.maximum_level:
            parent_key = self.grid.state_key(layout, grid_level)
            split_diagnostics = self.grid.split_diagnostics(parent_key, next_level)
        return {
            "probability": float(np.clip(final, 0.0, 1.0)),
            "modelProbability": model_probability,
            "neighborProbability": neighbor.probability,
            "gridProbability": grid_probability,
            "weights": {
                "model": weights[0],
                "neighbor": weights[1],
                "grid": weights[2],
            },
            "neighborEffectiveSamples": neighbor.effective_samples,
            "neighborRawSamples": neighbor.raw_samples,
            "neighborRadiusMm": neighbor.radius_mm,
            "gridSamples": grid_samples,
            "gridLevel": grid_level,
            "gridStateKey": list(grid_key),
            "nextGridSplit": split_diagnostics,
            "modelDisagreement": max(components) - min(components),
        }

    def _position_errors(
        self, layout: LayoutMm, position_error_mm: float
    ) -> np.ndarray:
        errors = np.full(3, position_error_mm, dtype=np.float64)
        for index, (x_mm, y_mm) in enumerate(layout.as_array()):
            cushion_distance = min(
                x_mm,
                TABLE_WIDTH_MM - x_mm,
                y_mm,
                TABLE_HEIGHT_MM - y_mm,
            )
            if cushion_distance <= self.config.uncertainty.near_cushion_threshold_mm:
                errors[index] = max(
                    errors[index], self.config.uncertainty.near_cushion_error_mm
                )
        return errors

    @staticmethod
    def _perturbed_layout(
        layout: LayoutMm, position_errors_mm: np.ndarray, rng: np.random.Generator
    ) -> LayoutMm:
        noise = rng.normal(0.0, 1.0, (3, 2)) * position_errors_mm[:, None]
        points = layout.as_array() + noise
        points[:, 0] = np.clip(points[:, 0], 0.0, TABLE_WIDTH_MM)
        points[:, 1] = np.clip(points[:, 1], 0.0, TABLE_HEIGHT_MM)
        converted = [PointMm(float(x), float(y)) for x, y in points]
        return LayoutMm(*converted)

    def predict(
        self,
        layout: LayoutMm,
        cue_ball: str,
        *,
        position_error_mm: float | None = None,
        prediction_id: str | None = None,
    ) -> dict[str, object]:
        error = (
            self.config.uncertainty.default_position_error_mm
            if position_error_mm is None
            else max(0.0, position_error_mm)
        )
        base = self._predict_once(layout, cue_ball)
        position_errors = self._position_errors(layout, error)
        effective_error = float(position_errors.max())
        rng = np.random.default_rng(self.config.uncertainty.random_seed)
        simulations = [
            float(
                self._predict_once(
                    self._perturbed_layout(layout, position_errors, rng), cue_ball
                )["probability"]
            )
            for _ in range(self.config.uncertainty.monte_carlo_samples)
        ]
        mean = float(np.mean(simulations)) if simulations else float(base["probability"])
        standard_deviation = float(np.std(simulations)) if simulations else 0.0
        confidence = confidence_score(
            float(base["neighborEffectiveSamples"]),
            int(base["gridSamples"]),
            effective_error,
            float(base["modelDisagreement"]),
        )
        flags: list[str] = []
        if effective_error >= self.config.uncertainty.reject_position_error_mm:
            flags.append("position_error_too_high")
        if float(base["modelDisagreement"]) >= 0.30:
            flags.append("model_data_conflict")
        if float(base["neighborEffectiveSamples"]) < 5:
            flags.append("sparse_neighbors")
        if isinstance(self.model, BootstrapProbabilityModel):
            flags.append("bootstrap_prior")
        return {
            "schemaVersion": "1.0",
            "predictionId": prediction_id,
            "modelVersion": self.model.version,
            "cueBall": cue_ball,
            "before": layout.to_dict(),
            "successProbability": mean,
            "difficulty": 1.0 - mean,
            "uncertainty": {
                "standardDeviation": standard_deviation,
                "positionErrorMm": error,
                "perBallPositionErrorMm": {
                    "cue": float(position_errors[0]),
                    "object1": float(position_errors[1]),
                    "object2": float(position_errors[2]),
                },
                "samples": self.config.uncertainty.monte_carlo_samples,
            },
            "confidence": {
                "score": confidence,
                "level": confidence_label(confidence),
            },
            "components": base,
            "flags": flags,
        }
