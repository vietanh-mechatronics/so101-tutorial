from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

try:
    from .action_reference_adapter import (
        ActionReferenceAdapter,
        ActionReferenceConfig,
    )
    from .top_cam_grid_5x9 import GRID_CELLS, GridRegion
except ImportError:
    from action_reference_adapter import ActionReferenceAdapter, ActionReferenceConfig
    from top_cam_grid_5x9 import GRID_CELLS, GridRegion


@dataclass
class GridObjectPlacementConfig:
    cells: tuple[str, ...]
    z: float | None = None

    @classmethod
    def from_dict(cls, raw: str | list[str] | dict[str, Any]) -> GridObjectPlacementConfig:
        if isinstance(raw, str):
            return cls(cells=(raw,))
        if isinstance(raw, list):
            return cls(cells=tuple(str(cell) for cell in raw))

        cell = raw.get("cell")
        cells = raw.get("cells")
        if cell is not None and cells is not None:
            raise ValueError("Use either `cell` or `cells` for a grid object placement, not both.")
        if cell is not None:
            resolved_cells = (str(cell),)
        elif cells is not None:
            resolved_cells = tuple(str(cell) for cell in cells)
        else:
            raise ValueError("Grid object placement requires `cell` or `cells`.")

        if not resolved_cells:
            raise ValueError("Grid object placement `cells` must not be empty.")
        z = raw.get("z")
        return cls(cells=resolved_cells, z=float(z) if z is not None else None)


@dataclass
class GridPlacementConfig:
    rows: int = 5
    cols: int = 9
    plane_z: float = 0.0
    camera_name: str | None = None
    objects: dict[str, GridObjectPlacementConfig] | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> GridPlacementConfig | None:
        if not raw:
            return None

        rows = int(raw.get("rows", 5))
        cols = int(raw.get("cols", 9))
        if rows <= 0 or cols <= 0:
            raise ValueError("env.grid_placements rows and cols must be > 0.")

        raw_objects = raw.get("objects", {})
        if not isinstance(raw_objects, dict) or not raw_objects:
            raise ValueError("env.grid_placements.objects must be a non-empty object.")

        return cls(
            rows=rows,
            cols=cols,
            plane_z=float(raw.get("plane_z", 0.0)),
            camera_name=raw.get("camera_name"),
            objects={
                str(body_name): GridObjectPlacementConfig.from_dict(body_raw)
                for body_name, body_raw in raw_objects.items()
            },
        )


@dataclass
class MatrixRandomPlacementConfig:
    table_path: str = "matrix_5x9_real_rule_selection_table.html"
    red_box_body: str = "pick_box"
    container_body: str = "container_box"
    red_box_z: float = 0.017
    container_z: float = 0.0
    red_box_region_margin: float = 0.0
    red_box_reach_center_x: float = 0.015
    red_box_reach_center_y: float = 0.0
    red_box_reach_radius: float = 0.44
    red_box_max_sample_attempts: int = 100
    red_box_random_yaw: bool = True
    container_base_center_x: float = 0.015
    container_base_center_y: float = 0.0
    container_min_base_distance: float = 0.18
    container_clearance_samples_per_axis: int = 101

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | bool | None) -> MatrixRandomPlacementConfig | None:
        if not raw:
            return None
        if isinstance(raw, bool):
            return cls()

        return cls(
            table_path=str(raw.get("table_path", cls.table_path)),
            red_box_body=str(raw.get("red_box_body", cls.red_box_body)),
            container_body=str(raw.get("container_body", cls.container_body)),
            red_box_z=float(raw.get("red_box_z", cls.red_box_z)),
            container_z=float(raw.get("container_z", cls.container_z)),
            red_box_region_margin=float(raw.get("red_box_region_margin", cls.red_box_region_margin)),
            red_box_reach_center_x=float(raw.get("red_box_reach_center_x", cls.red_box_reach_center_x)),
            red_box_reach_center_y=float(raw.get("red_box_reach_center_y", cls.red_box_reach_center_y)),
            red_box_reach_radius=float(raw.get("red_box_reach_radius", cls.red_box_reach_radius)),
            red_box_max_sample_attempts=int(
                raw.get("red_box_max_sample_attempts", cls.red_box_max_sample_attempts)
            ),
            red_box_random_yaw=bool(raw.get("red_box_random_yaw", cls.red_box_random_yaw)),
            container_base_center_x=float(raw.get("container_base_center_x", cls.container_base_center_x)),
            container_base_center_y=float(raw.get("container_base_center_y", cls.container_base_center_y)),
            container_min_base_distance=float(
                raw.get("container_min_base_distance", cls.container_min_base_distance)
            ),
            container_clearance_samples_per_axis=int(
                raw.get(
                    "container_clearance_samples_per_axis",
                    cls.container_clearance_samples_per_axis,
                )
            ),
        )


@dataclass
class CameraBrightnessConfig:
    gain: float = 1.0
    offset: float = 0.0

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        *,
        camera_name: str,
    ) -> CameraBrightnessConfig:
        if not isinstance(raw, dict):
            raise TypeError(
                f"env.camera_brightness.{camera_name} must be an object."
            )
        gain = float(raw.get("gain", 1.0))
        offset = float(raw.get("offset", 0.0))
        if not np.isfinite(gain) or gain <= 0.0:
            raise ValueError(
                f"env.camera_brightness.{camera_name}.gain must be finite and > 0."
            )
        if not np.isfinite(offset) or not -255.0 <= offset <= 255.0:
            raise ValueError(
                f"env.camera_brightness.{camera_name}.offset must be finite and within [-255,255]."
            )
        return cls(gain=gain, offset=offset)

    def as_dict(self) -> dict[str, float]:
        return {"gain": self.gain, "offset": self.offset}


@dataclass
class MujocoXmlEnvConfig:
    xml_path: str
    camera_names: dict[str, str] = field(default_factory=lambda: {"top": "cam_top", "gripper": "gripper_cam"})
    camera_brightness: dict[str, CameraBrightnessConfig] = field(default_factory=dict)
    image_height: int = 480
    image_width: int = 640
    frame_skip: int = 1
    state_source: str = "qpos"
    clip_actions_to_ctrlrange: bool = True
    qpos_reset_noise_std: float = 0.0
    auto_resize_offscreen_framebuffer: bool = True
    grid_placements: GridPlacementConfig | None = None
    matrix_random_placements: MatrixRandomPlacementConfig | None = None
    action_reference: ActionReferenceConfig = field(default_factory=ActionReferenceConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MujocoXmlEnvConfig:
        xml_path = str(raw.get("xml_path", "")).strip()
        if not xml_path:
            raise ValueError("env.xml_path is required when env.backend='mujoco_xml'.")

        state_source = str(raw.get("state_source", "qpos")).strip().lower()
        if state_source not in {"qpos", "qpos_qvel"}:
            raise ValueError(f"Unsupported env.state_source='{state_source}'. Use 'qpos' or 'qpos_qvel'.")

        camera_names = raw.get("camera_names")
        if camera_names is None:
            # Backward compatibility with the original two-camera collector configs.
            camera_names = {
                "top": raw.get("top_camera_name", "cam_top"),
                "gripper": raw.get("gripper_camera_name", "gripper_cam"),
            }
        if not isinstance(camera_names, dict) or not camera_names:
            raise ValueError("env.camera_names must be a non-empty object.")
        resolved_camera_names = {
            str(dataset_name).strip(): str(mujoco_name).strip()
            for dataset_name, mujoco_name in camera_names.items()
        }
        if any(
            not dataset_name or not mujoco_name for dataset_name, mujoco_name in resolved_camera_names.items()
        ):
            raise ValueError("env.camera_names keys and values must be non-empty strings.")

        camera_brightness_raw = raw.get("camera_brightness", {})
        if not isinstance(camera_brightness_raw, dict):
            raise TypeError("env.camera_brightness must be an object.")
        unknown_brightness_cameras = set(camera_brightness_raw) - set(resolved_camera_names)
        if unknown_brightness_cameras:
            raise ValueError(
                "env.camera_brightness contains cameras not present in env.camera_names: "
                f"{sorted(unknown_brightness_cameras)}"
            )
        camera_brightness = {
            str(dataset_name): CameraBrightnessConfig.from_dict(
                value,
                camera_name=str(dataset_name),
            )
            for dataset_name, value in camera_brightness_raw.items()
        }

        return cls(
            xml_path=xml_path,
            camera_names=resolved_camera_names,
            camera_brightness=camera_brightness,
            image_height=int(raw.get("image_height", 480)),
            image_width=int(raw.get("image_width", 640)),
            frame_skip=int(raw.get("frame_skip", 1)),
            state_source=state_source,
            clip_actions_to_ctrlrange=bool(raw.get("clip_actions_to_ctrlrange", True)),
            qpos_reset_noise_std=float(raw.get("qpos_reset_noise_std", 0.0)),
            auto_resize_offscreen_framebuffer=bool(raw.get("auto_resize_offscreen_framebuffer", True)),
            grid_placements=GridPlacementConfig.from_dict(raw.get("grid_placements")),
            matrix_random_placements=MatrixRandomPlacementConfig.from_dict(
                raw.get("matrix_random_placements")
            ),
            action_reference=ActionReferenceConfig.from_dict(raw.get("action_reference")),
        )


class MujocoXmlEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: MujocoXmlEnvConfig):
        super().__init__()
        self.cfg = cfg
        xml_path = Path(cfg.xml_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"MuJoCo xml not found: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        if cfg.auto_resize_offscreen_framebuffer:
            self.model.vis.global_.offwidth = max(
                int(self.model.vis.global_.offwidth),
                int(cfg.image_width),
            )
            self.model.vis.global_.offheight = max(
                int(self.model.vis.global_.offheight),
                int(cfg.image_height),
            )
        self.data = mujoco.MjData(self.model)
        self._renderer = mujoco.Renderer(
            self.model,
            height=cfg.image_height,
            width=cfg.image_width,
        )

        self._action_low, self._action_high = _build_action_bounds(self.model)
        actuator_names = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            for index in range(self.model.nu)
        )
        if any(name is None for name in actuator_names):
            raise ValueError("Every MuJoCo actuator must have a name for action-reference shaping")
        self.action_names = tuple(f"{name}.pos" for name in actuator_names)
        self._action_reference = ActionReferenceAdapter(
            cfg.action_reference,
            [str(name) for name in actuator_names],
        )
        self._last_grid_placements: dict[str, dict[str, float | str]] = {}
        self._last_matrix_random_placements: dict[str, dict[str, float | int | str]] = {}
        self._matrix_selection_plan = _load_matrix_selection_plan(
            Path(cfg.matrix_random_placements.table_path)
            if cfg.matrix_random_placements is not None
            else None
        )
        if cfg.matrix_random_placements is not None:
            _validate_reachable_red_cells(
                selection_plan=self._matrix_selection_plan,
                reach_center=(
                    cfg.matrix_random_placements.red_box_reach_center_x,
                    cfg.matrix_random_placements.red_box_reach_center_y,
                ),
                reach_radius=cfg.matrix_random_placements.red_box_reach_radius,
            )
            _validate_container_base_clearance_cells(
                selection_plan=self._matrix_selection_plan,
                base_center=(
                    cfg.matrix_random_placements.container_base_center_x,
                    cfg.matrix_random_placements.container_base_center_y,
                ),
                min_base_distance=cfg.matrix_random_placements.container_min_base_distance,
            )
        self.action_space = spaces.Box(
            low=self._action_low,
            high=self._action_high,
            dtype=np.float32,
        )

        state_dim = self.model.nu if cfg.state_source == "qpos" else self.model.nu * 2
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(state_dim,),
                    dtype=np.float32,
                ),
                "pixels": spaces.Dict(
                    {
                        dataset_name: spaces.Box(
                            low=0,
                            high=255,
                            shape=(cfg.image_height, cfg.image_width, 3),
                            dtype=np.uint8,
                        )
                        for dataset_name in cfg.camera_names
                    }
                ),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        episode_index = int((options or {}).get("episode_index", 0))
        mujoco.mj_resetData(self.model, self.data)
        action_reference_episode = self._action_reference.reset(
            episode_index=episode_index
        )

        if self.cfg.qpos_reset_noise_std > 0:
            noise = self.np_random.normal(
                0.0,
                self.cfg.qpos_reset_noise_std,
                size=self.model.nq,
            )
            self.data.qpos[:] = self.data.qpos[:] + noise.astype(np.float64)

        mujoco.mj_forward(self.model, self.data)
        self._apply_matrix_random_placements(episode_index)
        self._apply_grid_placements(episode_index)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {
            "grid_placements": dict(self._last_grid_placements),
            "matrix_random_placements": dict(self._last_matrix_random_placements),
            "action_reference": action_reference_episode,
        }

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        expected_dim = self.model.nu
        if action.shape != (expected_dim,):
            raise ValueError(f"Action shape mismatch: expected {(expected_dim,)}, got {tuple(action.shape)}.")

        reference = self._action_reference.transform(
            action,
            self.data.qpos[:expected_dim],
        )
        requested_action = reference.requested_action_rad
        action = (
            np.clip(requested_action, self._action_low, self._action_high)
            if self.cfg.clip_actions_to_ctrlrange
            else requested_action
        )

        self.data.ctrl[:] = action.astype(np.float64)
        mujoco.mj_step(self.model, self.data, nstep=max(self.cfg.frame_skip, 1))

        obs = self._get_obs()
        terminated = False
        truncated = False
        reward = 0.0
        info = {
            "sim_time": float(self.data.time),
            "grid_placements": dict(self._last_grid_placements),
            "matrix_random_placements": dict(self._last_matrix_random_placements),
            "action_reference": reference.diagnostics(action),
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._renderer.close()

    def _get_obs(self) -> dict[str, Any]:
        if self.cfg.state_source == "qpos":
            state = self.data.qpos[: self.model.nu].copy().astype(np.float32)
        else:
            state = np.concatenate(
                [self.data.qpos[: self.model.nu], self.data.qvel[: self.model.nu]],
                axis=0,
            ).astype(np.float32)

        return {
            "agent_pos": state,
            "pixels": {
                dataset_name: self._render_camera(
                    mujoco_name,
                    self.cfg.camera_brightness.get(dataset_name),
                )
                for dataset_name, mujoco_name in self.cfg.camera_names.items()
            },
        }

    def _render_camera(
        self,
        camera_name: str,
        brightness: CameraBrightnessConfig | None = None,
    ) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=camera_name)
        rgb = np.asarray(self._renderer.render(), dtype=np.uint8)
        if brightness is None or (
            brightness.gain == 1.0 and brightness.offset == 0.0
        ):
            return rgb
        adjusted = rgb.astype(np.float32) * brightness.gain + brightness.offset
        return np.clip(adjusted, 0.0, 255.0).astype(np.uint8)

    def _apply_matrix_random_placements(self, episode_index: int) -> None:
        matrix_cfg = self.cfg.matrix_random_placements
        self._last_matrix_random_placements = {}
        if matrix_cfg is None:
            return
        if not self._matrix_selection_plan:
            raise ValueError("matrix_random_placements table has no valid rows.")

        row_index, red_cell, container_index, container_cell = _select_matrix_placement(
            self._matrix_selection_plan,
            episode_index,
        )

        container_region = GRID_CELLS[container_cell]
        container_nominal_xy = np.asarray(container_region.center[:2], dtype=np.float64)
        container_xy, container_safety_adjusted = _container_xy_with_base_clearance(
            region=container_region,
            nominal_xy=container_nominal_xy,
            base_center=(
                matrix_cfg.container_base_center_x,
                matrix_cfg.container_base_center_y,
            ),
            min_base_distance=matrix_cfg.container_min_base_distance,
            samples_per_axis=matrix_cfg.container_clearance_samples_per_axis,
        )
        container_pos = self._body_position_with_xy(
            body_name=matrix_cfg.container_body,
            xy=container_xy,
            z=matrix_cfg.container_z,
        )
        self._place_body(matrix_cfg.container_body, container_pos)
        mujoco.mj_forward(self.model, self.data)
        container_base_distance = float(
            np.hypot(
                container_pos[0] - matrix_cfg.container_base_center_x,
                container_pos[1] - matrix_cfg.container_base_center_y,
            )
        )

        red_region = GRID_CELLS[red_cell]
        red_pos: np.ndarray | None = None
        red_yaw = 0.0
        placement_attempts = 0
        max_attempts = max(int(matrix_cfg.red_box_max_sample_attempts), 1)
        for placement_attempts in range(1, max_attempts + 1):
            try:
                red_xy = _sample_region_xy(
                    rng=self.np_random,
                    region=red_region,
                    margin=matrix_cfg.red_box_region_margin,
                    reach_center=(
                        matrix_cfg.red_box_reach_center_x,
                        matrix_cfg.red_box_reach_center_y,
                    ),
                    reach_radius=matrix_cfg.red_box_reach_radius,
                    max_attempts=1,
                )
            except ValueError:
                # This single draw was outside the reachable disk. Keep the
                # same table cells and try another point inside the red cell.
                continue

            candidate_pos = self._body_position_with_xy(
                body_name=matrix_cfg.red_box_body,
                xy=red_xy,
                z=matrix_cfg.red_box_z,
            )
            candidate_yaw = (
                float(self.np_random.uniform(-np.pi, np.pi))
                if matrix_cfg.red_box_random_yaw
                else 0.0
            )
            self._place_body(
                matrix_cfg.red_box_body,
                candidate_pos,
                quat=_yaw_to_quat(candidate_yaw),
            )
            mujoco.mj_forward(self.model, self.data)
            if not _bodies_have_contact(
                self.model,
                self.data,
                matrix_cfg.red_box_body,
                matrix_cfg.container_body,
            ):
                red_pos = candidate_pos
                red_yaw = candidate_yaw
                break

        if red_pos is None:
            raise ValueError(
                f"Could not place '{matrix_cfg.red_box_body}' in cell '{red_cell}' "
                f"without touching '{matrix_cfg.container_body}' in cell "
                f"'{container_cell}' after {max_attempts} attempts."
            )

        red_reach_distance = float(
            np.hypot(
                red_pos[0] - matrix_cfg.red_box_reach_center_x,
                red_pos[1] - matrix_cfg.red_box_reach_center_y,
            )
        )

        self._last_matrix_random_placements = {
            matrix_cfg.red_box_body: {
                "cell": red_cell,
                "episode_index": episode_index,
                "table_row_index": row_index,
                "table_row_number": row_index + 1,
                "x": float(red_pos[0]),
                "y": float(red_pos[1]),
                "z": float(red_pos[2]),
                "yaw": red_yaw,
                "reach_distance": red_reach_distance,
                "reach_radius": matrix_cfg.red_box_reach_radius,
                "placement_attempts": placement_attempts,
                "container_contact_free": True,
            },
            matrix_cfg.container_body: {
                "cell": container_cell,
                "source_row": red_cell,
                "episode_index": episode_index,
                "container_index": container_index,
                "container_number": container_index + 1,
                "x": float(container_pos[0]),
                "y": float(container_pos[1]),
                "z": float(container_pos[2]),
                "nominal_x": float(container_nominal_xy[0]),
                "nominal_y": float(container_nominal_xy[1]),
                "base_distance": container_base_distance,
                "min_base_distance": matrix_cfg.container_min_base_distance,
                "safety_adjusted": container_safety_adjusted,
            },
        }

    def _apply_grid_placements(self, episode_index: int) -> None:
        grid_cfg = self.cfg.grid_placements
        self._last_grid_placements = {}
        if grid_cfg is None or not grid_cfg.objects:
            return

        default_camera_name = self.cfg.camera_names.get(
            "top",
            next(iter(self.cfg.camera_names.values())),
        )
        camera_name = grid_cfg.camera_name or default_camera_name
        for body_name, placement in grid_cfg.objects.items():
            cell = placement.cells[episode_index % len(placement.cells)]
            xy = self._grid_cell_to_floor_xy(
                cell=cell,
                camera_name=camera_name,
                rows=grid_cfg.rows,
                cols=grid_cfg.cols,
                plane_z=grid_cfg.plane_z,
            )
            pos = self._body_position_with_xy(body_name, xy, placement.z)
            self._place_body(body_name, pos)
            self._last_grid_placements[body_name] = {
                "cell": cell,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
            }

    def _grid_cell_to_floor_xy(
        self,
        cell: str,
        camera_name: str,
        rows: int,
        cols: int,
        plane_z: float,
    ) -> np.ndarray:
        row, col = _parse_grid_cell(cell, rows, cols)
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            raise ValueError(f"Camera '{camera_name}' was not found in MuJoCo model.")

        cam_pos = self.data.cam_xpos[camera_id]
        cam_xmat = self.data.cam_xmat[camera_id].reshape(3, 3)
        # MuJoCo stores the camera's local axes in the columns of cam_xmat.
        # Reading rows mirrors/rotates the floor intersection and is especially
        # wrong for the outer cells of a tilted camera.
        x_axis = cam_xmat[:, 0]
        y_axis = cam_xmat[:, 1]
        z_axis = cam_xmat[:, 2]

        fovy = np.deg2rad(float(self.model.cam_fovy[camera_id]))
        aspect = float(self.cfg.image_width) / float(self.cfg.image_height)
        half_height = np.tan(fovy / 2.0)
        half_width = half_height * aspect

        u = (((col + 0.5) / cols) * 2.0 - 1.0) * half_width
        v = (1.0 - ((row + 0.5) / rows) * 2.0) * half_height
        ray = -z_axis + u * x_axis + v * y_axis

        if abs(float(ray[2])) < 1e-9:
            raise ValueError(f"Camera ray for cell '{cell}' is parallel to plane z={plane_z}.")

        t = (plane_z - float(cam_pos[2])) / float(ray[2])
        point = cam_pos + t * ray
        return point[:2].astype(np.float64)

    def _body_position_with_xy(
        self,
        body_name: str,
        xy: np.ndarray,
        z: float | None,
    ) -> np.ndarray:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' was not found in MuJoCo model.")

        pos = self._current_body_position(body_id)
        pos[:2] = xy
        if z is not None:
            pos[2] = z
        return pos

    def _current_body_position(self, body_id: int) -> np.ndarray:
        free_joint_id = _free_joint_id_for_body(self.model, body_id)
        if free_joint_id is None:
            return self.model.body_pos[body_id].copy()

        qpos_adr = self.model.jnt_qposadr[free_joint_id]
        return self.data.qpos[qpos_adr : qpos_adr + 3].copy()

    def _place_body(
        self,
        body_name: str,
        pos: np.ndarray,
        quat: np.ndarray | None = None,
    ) -> None:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        free_joint_id = _free_joint_id_for_body(self.model, body_id)
        if free_joint_id is None:
            self.model.body_pos[body_id, :] = pos
            return

        qpos_adr = self.model.jnt_qposadr[free_joint_id]
        qvel_adr = self.model.jnt_dofadr[free_joint_id]
        self.data.qpos[qpos_adr : qpos_adr + 3] = pos
        if quat is None:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat
        self.data.qvel[qvel_adr : qvel_adr + 6] = 0.0


def _build_action_bounds(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    if model.nu <= 0:
        raise ValueError("Model has zero actuators (nu=0). Cannot build an action space.")

    ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=np.float32)
    ctrllimited = np.asarray(model.actuator_ctrllimited, dtype=np.int32).astype(bool)

    low = np.full((model.nu,), -1.0, dtype=np.float32)
    high = np.full((model.nu,), 1.0, dtype=np.float32)

    if ctrlrange.shape == (model.nu, 2):
        low = np.where(ctrllimited, ctrlrange[:, 0], low).astype(np.float32)
        high = np.where(ctrllimited, ctrlrange[:, 1], high).astype(np.float32)

    eps = np.float32(1e-6)
    invalid = ~np.isfinite(low) | ~np.isfinite(high) | (high <= low)
    low = np.where(invalid, -1.0, low)
    high = np.where(invalid, 1.0, high + eps)
    return low, high


class _SelectionTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple[str, ...]] = []
        self._in_selection_table = False
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "table" and attrs_dict.get("id") == "selection-table":
            self._in_selection_table = True
            return
        if not self._in_selection_table:
            return
        if tag == "tr":
            self._current_row = []
        elif tag == "td" and self._current_row is not None:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._current_cell is not None and self._current_row is not None:
            cell = "".join(self._current_cell).strip().lower()
            if cell:
                self._current_row.append(cell)
            self._current_cell = None
            return
        if tag == "tr" and self._current_row is not None:
            if self._current_row:
                self.rows.append(tuple(self._current_row))
            self._current_row = None
            return
        if tag == "table" and self._in_selection_table:
            self._in_selection_table = False


def _load_matrix_selection_plan(
    table_path: Path | None,
) -> list[tuple[str, tuple[str, ...]]]:
    if table_path is None:
        return []
    if not table_path.exists():
        raise FileNotFoundError(f"Matrix selection table not found: {table_path}")

    parser = _SelectionTableParser()
    parser.feed(table_path.read_text(encoding="utf-8"))

    plan: list[tuple[str, tuple[str, ...]]] = []
    for row in parser.rows:
        grid_cells = tuple(cell.strip().lower() for cell in row if _is_grid_cell_text(cell))
        if len(grid_cells) < 2:
            continue
        red_cell = grid_cells[0]
        container_cells = grid_cells[1:]
        _validate_matrix_cell(red_cell, table_path)
        for cell in container_cells:
            _validate_matrix_cell(cell, table_path)
        plan.append((red_cell, container_cells))

    if not plan:
        raise ValueError(f"No placement rows found in matrix selection table: {table_path}")
    return plan


def _validate_matrix_cell(cell: str, table_path: Path) -> None:
    if cell not in GRID_CELLS:
        raise ValueError(f"Cell '{cell}' in {table_path} is not part of the 5x9 grid.")


def _is_grid_cell_text(value: str) -> bool:
    return value.strip().lower() in GRID_CELLS


def _select_matrix_placement(
    selection_plan: list[tuple[str, tuple[str, ...]]],
    episode_index: int,
) -> tuple[int, str, int, str]:
    if episode_index < 0:
        raise ValueError("episode_index must be >= 0.")

    total_slots = sum(len(container_cells) for _, container_cells in selection_plan)
    if total_slots <= 0:
        raise ValueError("matrix_random_placements table has no container cells.")
    if episode_index >= total_slots:
        raise IndexError(f"Episode {episode_index} exceeds the {total_slots}-episode matrix placement table.")

    slot_index = episode_index
    for row_index, (red_cell, container_cells) in enumerate(selection_plan):
        if slot_index < len(container_cells):
            container_index = int(slot_index)
            return row_index, red_cell, container_index, container_cells[container_index]
        slot_index -= len(container_cells)

    raise RuntimeError("Could not resolve matrix placement slot.")


def _validate_reachable_red_cells(
    selection_plan: list[tuple[str, tuple[str, ...]]],
    reach_center: tuple[float, float],
    reach_radius: float,
) -> None:
    unreachable: list[str] = []
    seen: set[str] = set()
    for red_cell, _ in selection_plan:
        if red_cell in seen:
            continue
        seen.add(red_cell)
        region = GRID_CELLS[red_cell]
        min_distance = _min_distance_to_region_xy(region, reach_center)
        if min_distance > reach_radius:
            unreachable.append(f"{red_cell} min_dist={min_distance:.6f}")

    if unreachable:
        raise ValueError(
            "Some red-box cells in matrix_random_placements are outside the robot reach "
            f"radius {reach_radius:.6f} from center {reach_center}: "
            f"{', '.join(unreachable)}. Update the HTML table, markers, or reach radius."
        )


def _validate_container_base_clearance_cells(
    selection_plan: list[tuple[str, tuple[str, ...]]],
    base_center: tuple[float, float],
    min_base_distance: float,
) -> None:
    if min_base_distance < 0:
        raise ValueError("container_min_base_distance must be >= 0.")

    impossible: list[str] = []
    seen: set[str] = set()
    base_xy = np.asarray(base_center, dtype=np.float64)
    for _, container_cells in selection_plan:
        for container_cell in container_cells:
            if container_cell in seen:
                continue
            seen.add(container_cell)
            region = GRID_CELLS[container_cell]
            max_distance = max(
                float(np.linalg.norm(np.asarray(corner[:2], dtype=np.float64) - base_xy))
                for corner in region.corners
            )
            if max_distance + 1e-12 < min_base_distance:
                impossible.append(f"{container_cell} max_dist={max_distance:.6f}")

    if impossible:
        raise ValueError(
            "Some container cells cannot satisfy the required robot-base clearance "
            f"{min_base_distance:.6f} from center {base_center}: "
            f"{', '.join(impossible)}."
        )


def _container_xy_with_base_clearance(
    region: GridRegion,
    nominal_xy: np.ndarray,
    base_center: tuple[float, float],
    min_base_distance: float,
    samples_per_axis: int,
) -> tuple[np.ndarray, bool]:
    nominal_xy = np.asarray(nominal_xy, dtype=np.float64)
    base_xy = np.asarray(base_center, dtype=np.float64)
    min_base_distance = max(float(min_base_distance), 0.0)
    nominal_distance = float(np.linalg.norm(nominal_xy - base_xy))
    if nominal_distance + 1e-12 >= min_base_distance:
        return nominal_xy.copy(), False

    samples_per_axis = int(samples_per_axis)
    if samples_per_axis < 2:
        raise ValueError("container_clearance_samples_per_axis must be >= 2.")

    best_xy: np.ndarray | None = None
    best_key: tuple[float, float, float, float] | None = None
    for row_t in np.linspace(0.0, 1.0, samples_per_axis):
        for col_t in np.linspace(0.0, 1.0, samples_per_axis):
            candidate = np.asarray(region.point_at(float(row_t), float(col_t))[:2])
            base_distance = float(np.linalg.norm(candidate - base_xy))
            if base_distance + 1e-12 < min_base_distance:
                continue
            displacement = float(np.linalg.norm(candidate - nominal_xy))
            # Once a nominal cell center is unsafe, use the safest point that
            # still belongs to that same rule cell. The container is rigid and
            # can block the arm, so maximizing clearance is preferable to only
            # nudging it to the threshold.
            key = (-base_distance, displacement, float(row_t), float(col_t))
            if best_key is None or key < best_key:
                best_xy = candidate
                best_key = key

    if best_xy is None:
        raise ValueError(
            f"Container cell '{region.name}' cannot provide base clearance "
            f"{min_base_distance:.6f} from center {base_center}."
        )
    return best_xy, True


def _min_distance_to_region_xy(
    region: GridRegion,
    reach_center: tuple[float, float],
) -> float:
    x = min(max(float(reach_center[0]), region.x_min), region.x_max)
    y = min(max(float(reach_center[1]), region.y_min), region.y_max)
    return float(np.hypot(x - float(reach_center[0]), y - float(reach_center[1])))


def _sample_region_xy(
    rng: np.random.Generator,
    region: GridRegion,
    margin: float,
    reach_center: tuple[float, float] | None = None,
    reach_radius: float | None = None,
    max_attempts: int = 100,
) -> np.ndarray:
    margin = max(float(margin), 0.0)
    width = max(
        _xy_distance(region.top_left, region.top_right),
        _xy_distance(region.bottom_left, region.bottom_right),
    )
    height = max(
        _xy_distance(region.top_left, region.bottom_left),
        _xy_distance(region.top_right, region.bottom_right),
    )
    row_margin_t = margin / height if height > 0 else 0.0
    col_margin_t = margin / width if width > 0 else 0.0
    if row_margin_t >= 0.5 or col_margin_t >= 0.5:
        raise ValueError(f"Margin {margin} is too large for grid region '{region.name}'.")

    attempts = max(int(max_attempts), 1)
    last_xy: np.ndarray | None = None
    for _ in range(attempts):
        point = region.point_at(
            row_t=float(rng.uniform(row_margin_t, 1.0 - row_margin_t)),
            col_t=float(rng.uniform(col_margin_t, 1.0 - col_margin_t)),
        )
        xy = np.asarray([point[0], point[1]], dtype=np.float64)
        last_xy = xy
        if reach_center is None or reach_radius is None:
            return xy
        if _is_within_reach(xy, reach_center, reach_radius):
            return xy

    raise ValueError(
        f"Could not sample a reachable point in grid region '{region.name}' "
        f"after {attempts} attempts. Last sampled xy={last_xy}, "
        f"reach_center={reach_center}, reach_radius={reach_radius}."
    )


def _xy_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _is_within_reach(
    xy: np.ndarray,
    reach_center: tuple[float, float],
    reach_radius: float,
) -> bool:
    dx = float(xy[0]) - float(reach_center[0])
    dy = float(xy[1]) - float(reach_center[1])
    return float(np.hypot(dx, dy)) <= float(reach_radius)


def _bodies_have_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    first_body_name: str,
    second_body_name: str,
) -> bool:
    """Return whether MuJoCo reports contact between two named bodies."""
    first_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, first_body_name)
    second_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, second_body_name)
    if first_body_id < 0:
        raise ValueError(f"Body '{first_body_name}' was not found in MuJoCo model.")
    if second_body_id < 0:
        raise ValueError(f"Body '{second_body_name}' was not found in MuJoCo model.")

    target_pair = {int(first_body_id), int(second_body_id)}
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1_body = int(model.geom_bodyid[int(contact.geom1)])
        geom2_body = int(model.geom_bodyid[int(contact.geom2)])
        if {geom1_body, geom2_body} == target_pair:
            return True
    return False


def _yaw_to_quat(yaw: float) -> np.ndarray:
    half = float(yaw) / 2.0
    return np.asarray([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def _parse_grid_cell(cell: str, rows: int, cols: int) -> tuple[int, int]:
    normalized = cell.strip().lower()
    if len(normalized) < 2 or not normalized[0].isalpha() or not normalized[1:].isdigit():
        raise ValueError(f"Invalid grid cell '{cell}'. Expected format like 'c4'.")

    row = ord(normalized[0]) - ord("a")
    col = int(normalized[1:]) - 1
    if row < 0 or row >= rows or col < 0 or col >= cols:
        raise ValueError(f"Grid cell '{cell}' is outside a {rows}x{cols} grid.")
    return row, col


def _free_joint_id_for_body(model: mujoco.MjModel, body_id: int) -> int | None:
    joint_start = int(model.body_jntadr[body_id])
    joint_count = int(model.body_jntnum[body_id])
    for joint_id in range(joint_start, joint_start + joint_count):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return joint_id
    return None
