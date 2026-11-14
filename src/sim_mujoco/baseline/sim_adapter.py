from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import combine_feature_dicts, hw_to_dataset_features


@dataclass(frozen=True)
class FeatureValueTransform:
    input_min: float
    input_max: float
    output_min: float
    output_max: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FeatureValueTransform:
        transform = cls(
            input_min=float(raw["input_min"]),
            input_max=float(raw["input_max"]),
            output_min=float(raw["output_min"]),
            output_max=float(raw["output_max"]),
        )
        if not transform.input_max > transform.input_min:
            raise ValueError("feature transform input_max must be greater than input_min.")
        if not transform.output_max > transform.output_min:
            raise ValueError("feature transform output_max must be greater than output_min.")
        return transform

    def apply(self, value: float) -> float:
        ratio = (value - self.input_min) / (self.input_max - self.input_min)
        mapped = self.output_min + ratio * (self.output_max - self.output_min)
        return float(np.clip(mapped, self.output_min, self.output_max))


@dataclass
class SimAdapterConfig:
    state_key: str = "agent_pos"
    image_keys: dict[str, str] = field(
        default_factory=lambda: {
            "top": "pixels.top",
            "gripper": "pixels.gripper",
        }
    )
    expected_image_hw: tuple[int, int] | None = None
    state_names: list[str] | None = None
    action_names: list[str] | None = None
    dataset_angle_unit: str = "radian"
    feature_value_transforms: dict[str, FeatureValueTransform] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> SimAdapterConfig:
        raw = raw or {}
        expected_image_hw = raw.get("expected_image_hw")
        image_keys = raw.get("image_keys")
        if image_keys is None:
            # Backward compatibility with the original top/gripper adapter config.
            image_keys = {
                "top": raw.get("top_image_key", "pixels.top"),
                "gripper": raw.get("gripper_image_key", "pixels.gripper"),
            }
        if not isinstance(image_keys, dict) or not image_keys:
            raise ValueError("adapter.image_keys must be a non-empty object.")
        resolved_image_keys = {
            str(dataset_name).strip(): str(observation_path).strip()
            for dataset_name, observation_path in image_keys.items()
        }
        if any(
            not dataset_name or not observation_path
            for dataset_name, observation_path in resolved_image_keys.items()
        ):
            raise ValueError("adapter.image_keys keys and values must be non-empty strings.")

        raw_transforms = raw.get("feature_value_transforms", {})
        if not isinstance(raw_transforms, dict):
            raise ValueError("adapter.feature_value_transforms must be an object.")

        return cls(
            state_key=str(raw.get("state_key", "agent_pos")),
            image_keys=resolved_image_keys,
            expected_image_hw=tuple(expected_image_hw) if expected_image_hw else None,
            state_names=raw.get("state_names"),
            action_names=raw.get("action_names"),
            dataset_angle_unit=_normalize_angle_unit(raw.get("dataset_angle_unit", "radian")),
            feature_value_transforms={
                str(feature_name): FeatureValueTransform.from_dict(transform)
                for feature_name, transform in raw_transforms.items()
            },
        )


class SimDatasetAdapter:
    def __init__(self, config: SimAdapterConfig):
        self.config = config
        self.angle_scale = 180.0 / np.pi if config.dataset_angle_unit == "degree" else 1.0
        self.state_names: list[str] | None = None
        self.action_names: list[str] | None = None
        self.image_shapes_hwc: dict[str, tuple[int, int, int]] = {}

    @property
    def image_names(self) -> tuple[str, ...]:
        return tuple(self.config.image_keys)

    def prepare(self, observation: Any, sample_action: np.ndarray) -> None:
        state = _to_float_vector(_get_by_path(observation, self.config.state_key), "state")
        if self.config.state_names is None:
            self.state_names = [f"state_{i}" for i in range(state.shape[0])]
        else:
            if len(self.config.state_names) != state.shape[0]:
                raise ValueError(
                    "Configured `state_names` length does not match state dimension: "
                    f"{len(self.config.state_names)} != {state.shape[0]}"
                )
            self.state_names = list(self.config.state_names)

        if sample_action.ndim != 1:
            raise ValueError(f"Action must be 1D, got shape={sample_action.shape}.")
        if self.config.action_names is None:
            self.action_names = [f"action_{i}" for i in range(sample_action.shape[0])]
        else:
            if len(self.config.action_names) != sample_action.shape[0]:
                raise ValueError(
                    "Configured `action_names` length does not match action dimension: "
                    f"{len(self.config.action_names)} != {sample_action.shape[0]}"
                )
            self.action_names = list(self.config.action_names)

        known_value_names = set(self.state_names or []) | set(self.action_names or [])
        unknown_transforms = set(self.config.feature_value_transforms) - known_value_names
        if unknown_transforms:
            raise ValueError(
                "Feature transforms refer to unknown state/action names: "
                f"{sorted(unknown_transforms)}"
            )

        self.image_shapes_hwc = {}
        for dataset_name, observation_path in self.config.image_keys.items():
            image = _normalize_image(_get_by_path(observation, observation_path), dataset_name)
            self._validate_expected_hw(image, dataset_name)
            self.image_shapes_hwc[dataset_name] = tuple(image.shape)

    def build_dataset_features(self, use_videos: bool) -> dict[str, dict]:
        self._ensure_prepared()
        observation_hw_features: dict[str, type | tuple[int, int, int]] = dict.fromkeys(
            self.state_names or [], float
        )
        observation_hw_features.update(self.image_shapes_hwc)
        action_hw_features: dict[str, type | tuple[int, int, int]] = dict.fromkeys(
            self.action_names or [], float
        )
        return combine_feature_dicts(
            hw_to_dataset_features(observation_hw_features, OBS_STR, use_video=use_videos),
            hw_to_dataset_features(action_hw_features, ACTION, use_video=use_videos),
        )

    def observation_to_values(self, observation: Any) -> dict[str, Any]:
        self._ensure_prepared()
        state = _to_float_vector(_get_by_path(observation, self.config.state_key), "state")
        if len(self.state_names or []) != state.shape[0]:
            raise ValueError("State vector dimension changed during collection.")
        values: dict[str, Any] = {
            name: self._convert_value(name, float(state[idx]))
            for idx, name in enumerate(self.state_names or [])
        }

        for dataset_name, observation_path in self.config.image_keys.items():
            image = _normalize_image(_get_by_path(observation, observation_path), dataset_name)
            self._validate_image_shape(
                image,
                self.image_shapes_hwc.get(dataset_name),
                dataset_name,
            )
            values[dataset_name] = image
        return values

    def action_to_values(self, action: np.ndarray) -> dict[str, float]:
        self._ensure_prepared()
        arr = _to_float_vector(action, "action")
        if len(self.action_names or []) != arr.shape[0]:
            raise ValueError("Action dimension changed during collection.")
        return {
            name: self._convert_value(name, float(arr[idx]))
            for idx, name in enumerate(self.action_names or [])
        }

    def _convert_value(self, feature_name: str, raw_value: float) -> float:
        transform = self.config.feature_value_transforms.get(feature_name)
        if transform is not None:
            return transform.apply(raw_value)
        return raw_value * self.angle_scale

    def _validate_expected_hw(self, image: np.ndarray, camera_name: str) -> None:
        if self.config.expected_image_hw is None:
            return
        expected_h, expected_w = self.config.expected_image_hw
        h, w, _ = image.shape
        if (h, w) != (expected_h, expected_w):
            raise ValueError(
                f"Camera '{camera_name}' shape mismatch: expected {(expected_h, expected_w)} got {(h, w)}"
            )

    @staticmethod
    def _validate_image_shape(
        image: np.ndarray,
        expected_shape: tuple[int, int, int] | None,
        camera_name: str,
    ) -> None:
        if expected_shape is None:
            return
        if tuple(image.shape) != expected_shape:
            raise ValueError(
                f"Camera '{camera_name}' shape changed during recording: "
                f"expected {expected_shape}, got {tuple(image.shape)}."
            )

    def _ensure_prepared(self) -> None:
        if self.state_names is None or self.action_names is None:
            raise RuntimeError("Adapter is not prepared. Call `prepare` first.")
        if set(self.image_shapes_hwc) != set(self.config.image_keys):
            raise RuntimeError("Adapter image features are not prepared.")


def _get_by_path(container: Any, path: str) -> Any:
    if isinstance(container, dict) and path in container:
        return container[path]

    value = container
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise KeyError(f"Could not resolve path '{path}'. Missing key part '{part}'.")
    return value


def _to_float_vector(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf values.")
    return arr


def _normalize_angle_unit(value: Any) -> str:
    unit = str(value).strip().lower()
    if unit in {"rad", "radian", "radians"}:
        return "radian"
    if unit in {"deg", "degree", "degrees"}:
        return "degree"
    raise ValueError("adapter.dataset_angle_unit must be 'radian' or 'degree'.")


def _normalize_image(value: Any, camera_name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim != 3:
        raise ValueError(f"Camera '{camera_name}' image must be HWC or CHW, got shape={arr.shape}.")

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr
