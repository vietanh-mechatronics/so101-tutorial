"""Pure per-joint delay and own-angle shift adapter for MuJoCo controls.

All public inputs and outputs are native MuJoCo radians.  Shift anchors are
stored in degrees in the S3 JSON and converted internally.  The adapter never
changes the policy action that is recorded in a dataset; it only shapes the
actuator reference sent to physics.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

RANDOM_SHIFT_MODE = "lag_plus_shift_random"
MODES = ("off", "lag_only", "lag_plus_shift", RANDOM_SHIFT_MODE)
SHIFT_PROFILE_LEVELS = ("q10", "center", "q90")
BALANCED_PROFILE_SAMPLING = "seeded_balanced_cartesian_cycle"


@dataclass(frozen=True)
class ShiftRandomizationConfig:
    seed: int
    levels: tuple[str, ...]
    sampling: str = BALANCED_PROFILE_SAMPLING

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> ShiftRandomizationConfig:
        if not isinstance(raw, dict):
            raise TypeError("action_reference.randomization must be an object")
        seed = int(raw.get("seed", 36))
        if seed < 0:
            raise ValueError("action_reference.randomization.seed must be non-negative")
        levels = tuple(str(value).strip().lower() for value in raw.get("levels", ()))
        if not levels or len(set(levels)) != len(levels):
            raise ValueError(
                "action_reference.randomization.levels must be unique and non-empty"
            )
        unknown = [value for value in levels if value not in SHIFT_PROFILE_LEVELS]
        if unknown:
            raise ValueError(
                "Unsupported random shift levels "
                f"{unknown}; choose from {SHIFT_PROFILE_LEVELS}"
            )
        sampling = str(raw.get("sampling", BALANCED_PROFILE_SAMPLING)).strip().lower()
        if sampling != BALANCED_PROFILE_SAMPLING:
            raise ValueError(
                "Only seeded_balanced_cartesian_cycle shift sampling is supported"
            )
        return cls(seed=seed, levels=levels, sampling=sampling)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "levels": list(self.levels),
            "sampling": self.sampling,
        }


@dataclass(frozen=True)
class ActionReferenceConfig:
    mode: str = "off"
    spec_path: str | None = None
    randomization: ShiftRandomizationConfig | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> ActionReferenceConfig:
        raw = raw or {}
        mode = str(raw.get("mode", "off")).strip().lower()
        if mode not in MODES:
            raise ValueError(
                f"Unsupported action-reference mode {mode!r}; use one of {MODES}"
            )
        spec_raw = raw.get("spec_path")
        spec_path = str(spec_raw).strip() if spec_raw is not None else None
        if mode != "off" and not spec_path:
            raise ValueError(
                f"action_reference.spec_path is required for mode={mode!r}"
            )
        randomization_raw = raw.get("randomization")
        if mode == RANDOM_SHIFT_MODE:
            randomization = ShiftRandomizationConfig.from_dict(randomization_raw)
        else:
            if randomization_raw is not None:
                raise ValueError(
                    "action_reference.randomization is only valid for "
                    f"mode={RANDOM_SHIFT_MODE!r}"
                )
            randomization = None
        return cls(
            mode=mode,
            spec_path=spec_path or None,
            randomization=randomization,
        )


@dataclass(frozen=True)
class ActionReferenceResult:
    mode: str
    episode_profile_id: str
    episode_profile_levels: dict[str, str]
    input_action_rad: np.ndarray
    delayed_action_rad: np.ndarray
    shift_bias_rad: np.ndarray
    requested_action_rad: np.ndarray

    def diagnostics(self, applied_action_rad: Sequence[float]) -> dict[str, Any]:
        applied = np.asarray(applied_action_rad, dtype=np.float64).reshape(-1)
        if applied.shape != self.requested_action_rad.shape:
            raise ValueError("applied action shape differs from adapter output")
        clip_delta = applied - self.requested_action_rad
        return {
            "mode": self.mode,
            "episode_profile_id": self.episode_profile_id,
            "episode_profile_levels": dict(self.episode_profile_levels),
            "input_action_rad": self.input_action_rad.tolist(),
            "delayed_action_rad": self.delayed_action_rad.tolist(),
            "shift_bias_rad": self.shift_bias_rad.tolist(),
            "requested_action_rad": self.requested_action_rad.tolist(),
            "applied_action_rad": applied.tolist(),
            "clip_delta_rad": clip_delta.tolist(),
            "clipped_joint_count": int(np.count_nonzero(np.abs(clip_delta) > 1e-12)),
        }


class PerJointDelayBuffer:
    """Return action[max(0,t-D[j]),j] and repeat the first episode action."""

    def __init__(self, delay_frames: Sequence[int]):
        values = np.asarray(delay_frames, dtype=np.int64)
        if values.ndim != 1 or len(values) == 0 or np.any(values < 0):
            raise ValueError("delay_frames must be a non-empty non-negative vector")
        self.delay_frames = values
        self._history: list[np.ndarray] = []
        self._first_push_pending = False

    def reset(self, first_action: Sequence[float] | None = None) -> None:
        self._history = []
        self._first_push_pending = False
        if first_action is not None:
            first = self._validate_action(first_action)
            self._history = [first.copy()]
            self._first_push_pending = True

    def push(self, action: Sequence[float]) -> np.ndarray:
        value = self._validate_action(action)
        if not self._history:
            self.reset(value)
        if self._first_push_pending:
            self._history[0] = value.copy()
            self._first_push_pending = False
        else:
            self._history.append(value.copy())
        frame = len(self._history) - 1
        return np.asarray(
            [
                self._history[max(0, frame - int(delay))][joint]
                for joint, delay in enumerate(self.delay_frames)
            ],
            dtype=np.float64,
        )

    def _validate_action(self, action: Sequence[float]) -> np.ndarray:
        value = np.asarray(action, dtype=np.float64).reshape(-1)
        if len(value) != len(self.delay_frames):
            raise ValueError("action width does not match delay vector")
        if not np.all(np.isfinite(value)):
            raise ValueError("action contains NaN or Inf")
        return value


class LinearClampedCurve:
    def __init__(self, angle_deg: Sequence[float], bias_deg: Sequence[float]):
        self.angle_deg = np.asarray(angle_deg, dtype=np.float64)
        self.bias_deg = np.asarray(bias_deg, dtype=np.float64)
        if (
            self.angle_deg.ndim != 1
            or self.bias_deg.shape != self.angle_deg.shape
            or len(self.angle_deg) < 2
        ):
            raise ValueError("curve anchors must be equal one-dimensional arrays")
        if not np.all(np.isfinite(self.angle_deg)) or not np.all(
            np.isfinite(self.bias_deg)
        ):
            raise ValueError("curve anchors must be finite")
        if not np.all(np.diff(self.angle_deg) > 0.0):
            raise ValueError("curve angle anchors must be strictly increasing")

    def evaluate_degree(self, angle_deg: float | np.ndarray) -> float | np.ndarray:
        values = np.asarray(angle_deg, dtype=np.float64)
        result = np.interp(
            values,
            self.angle_deg,
            self.bias_deg,
            left=self.bias_deg[0],
            right=self.bias_deg[-1],
        )
        return float(result) if values.ndim == 0 else result


class ActionReferenceAdapter:
    def __init__(
        self,
        config: ActionReferenceConfig,
        joint_names: Sequence[str],
    ):
        self.config = config
        self.joint_names = tuple(str(value) for value in joint_names)
        if len(self.joint_names) == 0 or len(set(self.joint_names)) != len(
            self.joint_names
        ):
            raise ValueError("joint_names must be unique and non-empty")
        self.spec_path: Path | None = None
        self.spec_sha256: str | None = None
        self.spec: dict[str, Any] | None = None
        self.curves: dict[int, LinearClampedCurve] = {}
        self.curve_bank: dict[int, dict[str, LinearClampedCurve]] = {}
        self.shift_joint_indices: tuple[int, ...] = ()
        self.episode_index = 0
        self.episode_profile_id = config.mode
        self.episode_profile_levels: dict[str, str] = {}
        delay = np.zeros(len(self.joint_names), dtype=np.int64)
        if config.mode != "off":
            self.spec_path = Path(config.spec_path or "").expanduser().resolve()
            self.spec = self._load_and_validate_spec(self.spec_path)
            self.spec_sha256 = _sha256(self.spec_path)
            delay = np.asarray(
                self.spec["timing"]["delay_buffer_frames"], dtype=np.int64
            )
            if config.mode in {"lag_plus_shift", RANDOM_SHIFT_MODE}:
                self.shift_joint_indices = tuple(
                    index
                    for index, joint in enumerate(self.joint_names)
                    if joint in self.spec["curves"]
                )
                levels = (
                    config.randomization.levels
                    if config.mode == RANDOM_SHIFT_MODE
                    and config.randomization is not None
                    else ("center",)
                )
                for joint_index in self.shift_joint_indices:
                    joint = self.joint_names[joint_index]
                    curve_spec = self.spec["curves"][joint]
                    self.curve_bank[joint_index] = {
                        level: self._build_curve(curve_spec, level) for level in levels
                    }
        self.buffer = PerJointDelayBuffer(delay)
        self.reset()

    @property
    def enabled(self) -> bool:
        return self.config.mode != "off"

    def reset(self, episode_index: int = 0) -> dict[str, Any]:
        if episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        self.buffer.reset()
        self.episode_index = int(episode_index)
        self.curves = {}
        self.episode_profile_levels = {}
        if self.config.mode == "lag_plus_shift":
            self.episode_profile_id = "deterministic_center"
            self.episode_profile_levels = {
                self.joint_names[index]: "center" for index in self.shift_joint_indices
            }
        elif self.config.mode == RANDOM_SHIFT_MODE:
            self.episode_profile_levels = self._select_random_profile(
                self.episode_index
            )
            self.episode_profile_id = "__".join(
                f"{joint}_{self.episode_profile_levels[joint]}"
                for joint in (
                    self.joint_names[index] for index in self.shift_joint_indices
                )
            )
        else:
            self.episode_profile_id = self.config.mode

        for joint_index in self.shift_joint_indices:
            joint = self.joint_names[joint_index]
            level = self.episode_profile_levels[joint]
            self.curves[joint_index] = self.curve_bank[joint_index][level]
        return self.episode_metadata()

    def episode_metadata(self) -> dict[str, Any]:
        randomization = self.config.randomization
        return {
            "mode": self.config.mode,
            "episode_index": self.episode_index,
            "profile_id": self.episode_profile_id,
            "profile_levels": dict(self.episode_profile_levels),
            "shift_joints": [
                self.joint_names[index] for index in self.shift_joint_indices
            ],
            "delay_buffer_frames": self.buffer.delay_frames.tolist(),
            "shift_spec_sha256": self.spec_sha256,
            "randomized_per_episode": self.config.mode == RANDOM_SHIFT_MODE,
            "randomization": randomization.as_dict()
            if randomization is not None
            else None,
        }

    def _select_random_profile(self, episode_index: int) -> dict[str, str]:
        randomization = self.config.randomization
        if randomization is None:
            raise RuntimeError("random shift mode has no randomization config")
        profiles = tuple(
            product(randomization.levels, repeat=len(self.shift_joint_indices))
        )
        cycle, slot = divmod(episode_index, len(profiles))
        priority_levels = tuple(
            level
            for level in ("center", *randomization.levels)
            if level in randomization.levels
        )
        priority_levels = tuple(dict.fromkeys(priority_levels))
        priority_profiles = tuple(
            (level,) * len(self.shift_joint_indices) for level in priority_levels
        )
        remaining_profiles = tuple(
            profile for profile in profiles if profile not in priority_profiles
        )
        rng = np.random.default_rng(np.random.SeedSequence((randomization.seed, cycle)))
        remaining_order = rng.permutation(len(remaining_profiles))
        ordered_profiles = priority_profiles + tuple(
            remaining_profiles[int(index)] for index in remaining_order
        )
        selected = ordered_profiles[slot]
        return {
            self.joint_names[joint_index]: level
            for joint_index, level in zip(
                self.shift_joint_indices, selected, strict=True
            )
        }

    @staticmethod
    def _build_curve(curve_spec: dict[str, Any], level: str) -> LinearClampedCurve:
        anchors = curve_spec["anchors"]
        if level == "center":
            bias = [float(row["bias_deg"]) for row in anchors]
        else:
            key = f"bias_{level}_deg"
            bias = [float(row["episode_variability"][key]) for row in anchors]
        return LinearClampedCurve(
            [float(row["angle_deg"]) for row in anchors],
            bias,
        )

    def transform(
        self,
        action_rad: Sequence[float],
        current_qpos_rad: Sequence[float],
    ) -> ActionReferenceResult:
        action = self._validate_vector(action_rad, "action")
        qpos = self._validate_vector(current_qpos_rad, "current_qpos")
        if self.config.mode == "off":
            delayed = action.copy()
        else:
            delayed = self.buffer.push(action)
        shift = np.zeros_like(action)
        if self.config.mode in {"lag_plus_shift", RANDOM_SHIFT_MODE}:
            qpos_deg = np.rad2deg(qpos)
            for joint_index, curve in self.curves.items():
                shift[joint_index] = np.deg2rad(
                    curve.evaluate_degree(float(qpos_deg[joint_index]))
                )
        requested = delayed - shift
        return ActionReferenceResult(
            mode=self.config.mode,
            episode_profile_id=self.episode_profile_id,
            episode_profile_levels=dict(self.episode_profile_levels),
            input_action_rad=action.copy(),
            delayed_action_rad=delayed,
            shift_bias_rad=shift,
            requested_action_rad=requested,
        )

    def _validate_vector(self, values: Sequence[float], label: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64).reshape(-1)
        if result.shape != (len(self.joint_names),):
            raise ValueError(
                f"{label} shape mismatch: expected {(len(self.joint_names),)}, got {result.shape}"
            )
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{label} contains NaN or Inf")
        return result

    def _load_and_validate_spec(self, path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("action-reference spec must be a JSON object")
        if value.get("status") != "S3_COMPLETE":
            raise ValueError("action-reference spec is not S3_COMPLETE")
        if tuple(value.get("joint_order", ())) != self.joint_names:
            raise ValueError(
                "action-reference spec joint order differs from MuJoCo actuator order"
            )
        timing = value.get("timing", {})
        delay = np.asarray(timing.get("delay_buffer_frames", ()), dtype=np.int64)
        if delay.shape != (len(self.joint_names),) or np.any(delay < 0):
            raise ValueError("action-reference delay vector is invalid")
        if timing.get("randomized") is not False:
            raise ValueError("S4 deterministic center requires randomized=false")
        curves = value.get("curves", {})
        if set(curves) != {"shoulder_lift", "elbow_flex"}:
            raise ValueError("S4 spec must contain exactly lift and elbow curves")
        for joint, curve_spec in curves.items():
            if joint not in self.joint_names:
                raise ValueError(f"Unknown curve joint: {joint}")
            if curve_spec.get("interpolation") != "piecewise_linear":
                raise ValueError("Only piecewise_linear shift is supported")
            if curve_spec.get("extrapolation") != "clamp_to_nearest_endpoint":
                raise ValueError("Only endpoint clamp extrapolation is supported")
            if curve_spec.get("state_memory") is not False:
                raise ValueError("S4 shift curve must be stateless")
            anchors = curve_spec.get("anchors", ())
            LinearClampedCurve(
                [float(row["angle_deg"]) for row in anchors],
                [float(row["bias_deg"]) for row in anchors],
            )
            if self.config.mode == RANDOM_SHIFT_MODE:
                randomization = self.config.randomization
                if randomization is None:
                    raise ValueError("random shift mode requires randomization config")
                for level in randomization.levels:
                    self._build_curve(curve_spec, level)
        return value


def benchmark_canonical_to_mujoco_degree(action: np.ndarray) -> np.ndarray:
    """Convert [five arm degrees, gripper 0..100] to six MuJoCo raw degrees."""
    values = np.asarray(action, dtype=np.float64)
    if values.ndim not in {1, 2} or values.shape[-1] != 6:
        raise ValueError("benchmark action must end with the six SO-101 joints")
    if not np.all(np.isfinite(values)):
        raise ValueError("benchmark action contains NaN or Inf")
    result = values.copy()
    result[..., 5] = -10.0 + 1.1 * result[..., 5]
    return result


def mujoco_degree_to_benchmark_canonical_state(state: np.ndarray) -> np.ndarray:
    """Convert six MuJoCo raw degrees to [five arm degrees, gripper 0..100]."""
    values = np.asarray(state, dtype=np.float64)
    if values.ndim not in {1, 2} or values.shape[-1] != 6:
        raise ValueError("MuJoCo state must end with the six SO-101 joints")
    if not np.all(np.isfinite(values)):
        raise ValueError("MuJoCo state contains NaN or Inf")
    result = values.copy()
    result[..., 5] = (result[..., 5] + 10.0) * (100.0 / 110.0)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
