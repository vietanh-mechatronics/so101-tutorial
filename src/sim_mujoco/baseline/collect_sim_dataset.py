from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os

# Select the known-good renderer before Gym/LeRobot can import MuJoCo. The
# desktop GL backend can spin forever while compiling shaders on this machine.
# Explicit caller settings continue to take precedence.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame

PREVIEW_CONTINUE = "continue"
PREVIEW_FINISH_EPISODE = "finish_episode"
PREVIEW_RERECORD_EPISODE = "rerecord_episode"
PREVIEW_STOP_SESSION = "stop_session"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SIM_TELEOP_MANIFEST_NAME = "sim_teleop_provenance.json"
ACTION_REFERENCE_PROFILE_LOG_NAME = "action_reference_profiles.jsonl"
DEFAULT_GRID_MARGIN_LEFT = 16
DEFAULT_GRID_MARGIN_TOP = 32
DEFAULT_GRID_MARGIN_RIGHT = 16
DEFAULT_GRID_MARGIN_BOTTOM = 0

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from action_reference_adapter import RANDOM_SHIFT_MODE
    from action_source import ActionSourceConfig, create_action_source
    from mujoco_xml_env import MujocoXmlEnv, MujocoXmlEnvConfig
    from sim_adapter import SimAdapterConfig, SimDatasetAdapter
else:
    from .action_reference_adapter import RANDOM_SHIFT_MODE
    from .action_source import ActionSourceConfig, create_action_source
    from .mujoco_xml_env import MujocoXmlEnv, MujocoXmlEnvConfig
    from .sim_adapter import SimAdapterConfig, SimDatasetAdapter


@dataclass
class EnvCollectConfig:
    backend: str
    id: str | None
    xml: MujocoXmlEnvConfig | None
    kwargs: dict[str, Any]
    seed: int | None
    reset_each_episode_seed: bool
    max_steps_per_episode: int | None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EnvCollectConfig:
        backend = str(raw.get("backend", "gym")).strip().lower()
        if backend not in {"gym", "mujoco_xml"}:
            raise ValueError(f"Unsupported env.backend='{backend}'. Use 'gym' or 'mujoco_xml'.")

        env_id = raw.get("id")
        xml_cfg: MujocoXmlEnvConfig | None = None
        if backend == "mujoco_xml":
            xml_cfg = MujocoXmlEnvConfig.from_dict(raw)
            env_id = None
        else:
            if not env_id:
                raise ValueError("env.id is required when env.backend='gym'.")

        return cls(
            backend=backend,
            id=str(env_id) if env_id else None,
            xml=xml_cfg,
            kwargs=raw.get("kwargs", {}),
            seed=raw.get("seed"),
            reset_each_episode_seed=bool(raw.get("reset_each_episode_seed", True)),
            max_steps_per_episode=raw.get("max_steps_per_episode"),
        )


@dataclass
class DatasetCollectConfig:
    repo_id: str
    root: str
    robot_type: str | None
    single_task: str
    fps: int
    num_episodes: int
    episode_count_mode: str
    episode_time_s: float | None
    use_videos: bool
    video_backend: str | None
    push_to_hub: bool
    private: bool
    tags: list[str] | None
    streaming_encoding: bool
    encoder_threads: int | None
    encoder_queue_maxsize: int
    image_writer_threads: int
    auto_resume_if_root_exists: bool

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DatasetCollectConfig:
        episode_time_raw = raw.get("episode_time_s", 60.0)
        episode_count_mode = str(raw.get("episode_count_mode", "additional")).strip().lower()
        if episode_count_mode not in {"additional", "target_total"}:
            raise ValueError(
                "dataset.episode_count_mode must be 'additional' or 'target_total'."
            )
        return cls(
            repo_id=str(raw["repo_id"]),
            root=str(raw["root"]),
            robot_type=str(raw["robot_type"]) if raw.get("robot_type") is not None else None,
            single_task=str(raw.get("single_task", "")),
            fps=int(raw.get("fps", 30)),
            num_episodes=int(raw.get("num_episodes", 50)),
            episode_count_mode=episode_count_mode,
            episode_time_s=(
                float(episode_time_raw) if episode_time_raw is not None else None
            ),
            use_videos=bool(raw.get("use_videos", True)),
            video_backend=(
                str(raw["video_backend"]) if raw.get("video_backend") is not None else None
            ),
            push_to_hub=bool(raw.get("push_to_hub", True)),
            private=bool(raw.get("private", False)),
            tags=raw.get("tags"),
            streaming_encoding=bool(raw.get("streaming_encoding", True)),
            encoder_threads=raw.get("encoder_threads"),
            encoder_queue_maxsize=int(raw.get("encoder_queue_maxsize", 30)),
            image_writer_threads=int(raw.get("image_writer_threads", 0)),
            auto_resume_if_root_exists=bool(raw.get("auto_resume_if_root_exists", True)),
        )


@dataclass
class TopGridConfig:
    enabled: bool
    rows: int
    cols: int
    labels: bool
    margin_left: int
    margin_top: int
    margin_right: int
    margin_bottom: int

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | bool | None) -> TopGridConfig:
        if isinstance(raw, bool):
            return cls(
                enabled=raw,
                rows=5,
                cols=9,
                labels=True,
                margin_left=DEFAULT_GRID_MARGIN_LEFT,
                margin_top=DEFAULT_GRID_MARGIN_TOP,
                margin_right=DEFAULT_GRID_MARGIN_RIGHT,
                margin_bottom=DEFAULT_GRID_MARGIN_BOTTOM,
            )

        raw = raw or {}
        rows = int(raw.get("rows", 5))
        cols = int(raw.get("cols", 9))
        if rows <= 0 or cols <= 0:
            raise ValueError("recording.top_grid rows and cols must be > 0.")

        margins = (
            int(raw.get("margin_left", DEFAULT_GRID_MARGIN_LEFT)),
            int(raw.get("margin_top", DEFAULT_GRID_MARGIN_TOP)),
            int(raw.get("margin_right", DEFAULT_GRID_MARGIN_RIGHT)),
            int(raw.get("margin_bottom", DEFAULT_GRID_MARGIN_BOTTOM)),
        )
        if min(margins) < 0:
            raise ValueError("recording.top_grid margins must be >= 0.")

        return cls(
            enabled=bool(raw.get("enabled", False)),
            rows=rows,
            cols=cols,
            labels=bool(raw.get("labels", True)),
            margin_left=margins[0],
            margin_top=margins[1],
            margin_right=margins[2],
            margin_bottom=margins[3],
        )


@dataclass
class RecordingConfig:
    sleep_to_fps: bool
    control_fps: int | None
    prepare_time_s: float
    rest_time_s: float
    teleop_during_prepare: bool
    teleop_during_rest: bool
    require_enter_before_episode: bool
    save_partial_episode_on_interrupt: bool
    display_camera_preview: bool
    preview_window_name: str
    preview_scale: float
    top_grid: TopGridConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> RecordingConfig:
        raw = raw or {}
        control_fps = raw.get("control_fps")
        return cls(
            sleep_to_fps=bool(raw.get("sleep_to_fps", True)),
            control_fps=int(control_fps) if control_fps is not None else None,
            prepare_time_s=max(float(raw.get("prepare_time_s", 0.0)), 0.0),
            rest_time_s=max(float(raw.get("rest_time_s", 0.0)), 0.0),
            teleop_during_prepare=bool(raw.get("teleop_during_prepare", False)),
            teleop_during_rest=bool(raw.get("teleop_during_rest", False)),
            require_enter_before_episode=bool(raw.get("require_enter_before_episode", False)),
            save_partial_episode_on_interrupt=bool(
                raw.get("save_partial_episode_on_interrupt", True)
            ),
            display_camera_preview=bool(raw.get("display_camera_preview", False)),
            preview_window_name=str(raw.get("preview_window_name", "sim_collect_preview")),
            preview_scale=float(raw.get("preview_scale", 1.0)),
            top_grid=TopGridConfig.from_dict(raw.get("top_grid")),
        )


@dataclass
class SimCollectConfig:
    env: EnvCollectConfig
    dataset: DatasetCollectConfig
    adapter: SimAdapterConfig
    action_source: ActionSourceConfig
    recording: RecordingConfig
    resume: bool

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SimCollectConfig:
        return cls(
            env=EnvCollectConfig.from_dict(raw["env"]),
            dataset=DatasetCollectConfig.from_dict(raw["dataset"]),
            adapter=SimAdapterConfig.from_dict(raw.get("adapter")),
            action_source=ActionSourceConfig.from_dict(raw.get("action_source")),
            recording=RecordingConfig.from_dict(raw.get("recording")),
            resume=bool(raw.get("resume", False)),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect LeRobot dataset from a simulation environment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/simulation/delay_shift_v2.json",
        help="Path to JSON config file.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume an existing dataset.")
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Do not push to Hugging Face Hub after collection.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help=(
            "Start fresh when dataset.root exists. This never deletes immediately: "
            "an exact confirmation token is required first."
        ),
    )
    parser.add_argument("--teleop.type", dest="teleop_type")
    parser.add_argument("--teleop.port", dest="teleop_port")
    parser.add_argument("--teleop.id", dest="teleop_id")
    parser.add_argument("--teleop.calibration_dir", dest="teleop_calibration_dir")
    parser.add_argument(
        "--teleop.use_degrees",
        dest="teleop_use_degrees",
        type=_parse_bool,
    )
    parser.add_argument("--dataset.num_episodes", dest="dataset_num_episodes", type=int)
    parser.add_argument("--dataset.root", dest="dataset_root", type=Path)
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id")
    parser.add_argument(
        "--dataset.episode_time_s",
        dest="dataset_episode_time_s",
        type=_parse_episode_duration,
        default=argparse.SUPPRESS,
        help="Positive seconds, or 'manual'/'none' for user-controlled episode duration.",
    )
    parser.add_argument(
        "--recording.prepare_time_s",
        dest="recording_prepare_time_s",
        type=_parse_nonnegative_duration,
    )
    parser.add_argument(
        "--recording.rest_time_s",
        dest="recording_rest_time_s",
        type=_parse_nonnegative_duration,
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    cfg = SimCollectConfig.from_dict(_load_json(args.config))
    _apply_teleop_cli_overrides(args)
    _apply_collection_cli_overrides(cfg, args)
    if args.resume:
        cfg.resume = True
    if args.no_push:
        cfg.dataset.push_to_hub = False
    if args.replace_existing:
        if cfg.resume:
            raise ValueError("--replace-existing cannot be combined with --resume.")
        dataset_root_exists = Path(cfg.dataset.root).expanduser().exists()
        deleted = _confirm_and_delete_existing_dataset_root(cfg.dataset.root)
        if dataset_root_exists and not deleted:
            logging.info("Collection cancelled because the existing dataset was kept.")
            return

    _collect(cfg, config_path=args.config)


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def _parse_episode_duration(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized in {"manual", "none", "null", "unlimited"}:
        return None
    try:
        duration = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Episode duration must be positive seconds or 'manual'."
        ) from exc
    if duration <= 0:
        raise argparse.ArgumentTypeError(
            "Episode duration must be positive seconds or 'manual'."
        )
    return duration


def _parse_nonnegative_duration(value: str) -> float:
    try:
        duration = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Duration must be a non-negative number.") from exc
    if duration < 0:
        raise argparse.ArgumentTypeError("Duration must be a non-negative number.")
    return duration


def _apply_teleop_cli_overrides(args: argparse.Namespace) -> None:
    if args.teleop_type is not None and args.teleop_type != "so101_leader":
        raise ValueError(
            f"Unsupported --teleop.type={args.teleop_type!r}; use 'so101_leader'."
        )
    if args.teleop_port is not None:
        os.environ["REAL_TELEOP_PORT"] = args.teleop_port
    if args.teleop_id is not None:
        os.environ["REAL_TELEOP_ID"] = args.teleop_id
    if args.teleop_calibration_dir is not None:
        os.environ["REAL_TELEOP_CALIBRATION_DIR"] = args.teleop_calibration_dir
    if args.teleop_use_degrees is not None:
        os.environ["REAL_TELEOP_USE_DEGREES"] = str(args.teleop_use_degrees).lower()


def _apply_collection_cli_overrides(
    cfg: SimCollectConfig,
    args: argparse.Namespace,
) -> None:
    if getattr(args, "dataset_num_episodes", None) is not None:
        cfg.dataset.num_episodes = int(args.dataset_num_episodes)
    if getattr(args, "dataset_root", None) is not None:
        cfg.dataset.root = str(args.dataset_root)
    if getattr(args, "dataset_repo_id", None) is not None:
        cfg.dataset.repo_id = str(args.dataset_repo_id)
    if hasattr(args, "dataset_episode_time_s"):
        cfg.dataset.episode_time_s = args.dataset_episode_time_s
    if getattr(args, "recording_prepare_time_s", None) is not None:
        cfg.recording.prepare_time_s = float(args.recording_prepare_time_s)
    if getattr(args, "recording_rest_time_s", None) is not None:
        cfg.recording.rest_time_s = float(args.recording_rest_time_s)


def _confirm_and_delete_existing_dataset_root(
    dataset_root: str | Path,
    *,
    input_fn: Any | None = None,
    workspace: Path | None = None,
) -> bool:
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        return False
    if not root.is_dir():
        raise ValueError(f"Dataset root exists but is not a directory: {root}")

    workspace = (workspace or Path.cwd()).resolve()
    datasets_dir = (workspace / "datasets").resolve()
    try:
        root.relative_to(datasets_dir)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to delete dataset root outside {datasets_dir}: {root}"
        ) from exc
    if root == datasets_dir:
        raise ValueError(f"Refusing to delete the datasets directory itself: {root}")

    token = f"DELETE {root.name}"
    prompt = (
        f"\nDataset trial already exists:\n  {root}\n"
        "This will permanently remove that directory before recording.\n"
        f"Type exactly '{token}' to continue, or press ENTER to cancel: "
    )
    response = (input_fn or input)(prompt).strip()
    if response != token:
        logging.info("Dataset deletion cancelled. Nothing was removed.")
        return False

    shutil.rmtree(root)
    logging.info("Deleted existing dataset root after confirmation: %s", root)
    return True


def _collect(cfg: SimCollectConfig, *, config_path: Path) -> None:
    if cfg.dataset.fps <= 0:
        raise ValueError("dataset.fps must be > 0.")
    control_fps = cfg.recording.control_fps or cfg.dataset.fps
    if control_fps <= 0:
        raise ValueError("recording.control_fps must be > 0.")
    if control_fps < cfg.dataset.fps:
        raise ValueError("recording.control_fps must be >= dataset.fps.")
    if control_fps % cfg.dataset.fps != 0:
        logging.warning(
            "control_fps=%d is not an integer multiple of dataset.fps=%d. "
            "Recorded frame cadence will be approximate.",
            control_fps,
            cfg.dataset.fps,
        )
    if cfg.dataset.num_episodes <= 0:
        raise ValueError("dataset.num_episodes must be > 0.")
    if cfg.resume and not cfg.dataset.root:
        raise ValueError("`dataset.root` is required when resume=true.")

    logging.info(
        "Creating simulation environment (MUJOCO_GL=%s)...",
        os.environ.get("MUJOCO_GL", "<unset>"),
    )
    env = _make_env(cfg.env)
    logging.info("Simulation environment is ready.")
    _log_mujoco_timing(cfg.env, env, control_fps)
    preview = None
    if cfg.recording.display_camera_preview:
        preview = _create_camera_preview(cfg.recording.preview_window_name)
    preview_enabled = preview is not None
    action_source = create_action_source(cfg.action_source, env)
    adapter = SimDatasetAdapter(cfg.adapter)
    dataset: LeRobotDataset | None = None

    try:
        seed = cfg.env.seed
        logging.info("Resetting simulation and connecting the action source...")
        init_obs, init_info = env.reset(seed=seed, options={"episode_index": 0})
        action_source.reset(episode_index=0, observation=init_obs, info=init_info)
        sample_action = action_source.next_action(
            observation=init_obs,
            info=init_info,
            step_index=0,
            episode_index=0,
        )
        _validate_action_shape_against_env(sample_action, env)
        adapter.prepare(init_obs, sample_action)

        dataset_features = adapter.build_dataset_features(use_videos=cfg.dataset.use_videos)
        dataset = _create_or_resume_dataset(cfg, dataset_features)
        _write_or_validate_sim_teleop_manifest(
            dataset=dataset,
            cfg=cfg,
            config_path=config_path,
        )
        episodes_to_record = _resolve_episodes_to_record(
            cfg.dataset,
            existing_episodes=dataset.num_episodes,
        )
        step_limit = _resolve_step_limit(cfg, env, control_fps=control_fps)
        if step_limit is None:
            logging.info(
                "Episode duration is manual (`episode_time_s=null`, no environment step limit)."
            )
            if preview is None:
                logging.warning(
                    "Manual-duration mode has no preview controls. Use Ctrl+C to finish safely."
                )
        control_interval = 1.0 / control_fps
        record_every_steps = max(1, round(control_fps / cfg.dataset.fps))

        logging.info("Dataset root: %s", dataset.root)
        logging.info("Existing saved episodes: %d", dataset.num_episodes)
        logging.info("Dataset action/state angle unit: %s", cfg.adapter.dataset_angle_unit)
        logging.info(
            "Matrix placement log: %s",
            Path(dataset.root) / "meta" / "matrix_random_placements.jsonl",
        )
        logging.info(
            "Starting collection with %d remaining episodes "
            "(target=%d, episode_count_mode=%s, control_fps=%d, "
            "dataset_fps=%d, record_every=%d control steps)",
            episodes_to_record,
            cfg.dataset.num_episodes,
            cfg.dataset.episode_count_mode,
            control_fps,
            cfg.dataset.fps,
            record_every_steps,
        )
        if episodes_to_record == 0:
            logging.info("Dataset already reached the configured episode target.")
            return
        if preview_enabled:
            logging.info(
                "Preview controls while recording: Right/Enter/Space saves the episode; "
                "Left/R/Backspace requests a confirmed re-record; Q/Esc saves the "
                "current episode and stops collection."
            )

        stop_requested = False
        recorded_episodes = 0
        while recorded_episodes < episodes_to_record:
            episode_index = dataset.num_episodes
            display_episode_number = (
                episode_index + 1
                if cfg.dataset.episode_count_mode == "target_total"
                else recorded_episodes + 1
            )
            episode_seed = None
            if cfg.env.reset_each_episode_seed and cfg.env.seed is not None:
                episode_seed = cfg.env.seed + episode_index

            obs, info = env.reset(seed=episode_seed, options={"episode_index": episode_index})
            episode_reset_info = dict(info or {})
            action_source.reset(episode_index=episode_index, observation=obs, info=info)
            matrix_placements = episode_reset_info.get("matrix_random_placements")
            placement_status = _matrix_placement_status(episode_reset_info)
            action_reference_episode = episode_reset_info.get("action_reference") or {}
            if action_reference_episode.get("randomized_per_episode"):
                logging.info(
                    "Episode %d action-reference profile: %s %s",
                    episode_index,
                    action_reference_episode.get("profile_id"),
                    action_reference_episode.get("profile_levels"),
                )
            if matrix_placements:
                logging.info(
                    "Episode %d matrix placement: %s",
                    episode_index,
                    matrix_placements,
                )
            wait_event, obs, info = _wait_before_episode(
                cfg=cfg,
                adapter=adapter,
                observation=obs,
                info=info,
                episode_index=episode_index,
                run_episode_number=display_episode_number,
                preview=preview,
                env=env,
                action_source=action_source,
                control_fps=control_fps,
            )
            if wait_event == PREVIEW_STOP_SESSION:
                stop_requested = True
                logging.info("Stop requested during episode preparation. Exiting collection loop.")
                break

            done = False
            interrupted = False
            rerecord_episode = False
            step = 0
            recorded_frames = 0
            episode_start_t = time.perf_counter()

            try:
                while (step_limit is None or step < step_limit) and not done:
                    loop_start_t = time.perf_counter()

                    action = action_source.next_action(
                        observation=obs,
                        info=info,
                        step_index=step,
                        episode_index=episode_index,
                    )
                    obs_values = adapter.observation_to_values(obs)

                    if step % record_every_steps == 0:
                        action_values = adapter.action_to_values(action)
                        obs_frame = build_dataset_frame(dataset.features, obs_values, prefix=OBS_STR)
                        action_frame = build_dataset_frame(
                            dataset.features,
                            action_values,
                            prefix=ACTION,
                        )
                        frame = {**obs_frame, **action_frame, "task": cfg.dataset.single_task}
                        dataset.add_frame(frame)
                        recorded_frames += 1

                    if preview is not None:
                        if step_limit is None:
                            duration_status = (
                                f"manual finish | {time.perf_counter() - episode_start_t:.1f}s elapsed"
                            )
                        else:
                            remaining_record_s = max(
                                0.0,
                                (step_limit - step) / control_fps,
                            )
                            duration_status = f"{remaining_record_s:.1f}s left"
                        preview_event = preview.show(
                            images={
                                image_name: obs_values[image_name]
                                for image_name in adapter.image_names
                            },
                            window_name=cfg.recording.preview_window_name,
                            scale=cfg.recording.preview_scale,
                            top_grid=cfg.recording.top_grid,
                            status_text=(
                                f"RECORDING {display_episode_number}/{cfg.dataset.num_episodes} "
                                f"| {placement_status} | {duration_status}"
                            ),
                            help_text=(
                                "Enter/Space/Right: save episode | "
                                "R/Left: re-record | Q/Esc: save & stop"
                            ),
                        )
                        if preview_event == PREVIEW_FINISH_EPISODE:
                            logging.info("Early finish requested for episode %d.", episode_index)
                            done = True
                        elif preview_event == PREVIEW_RERECORD_EPISODE:
                            if _confirm_rerecord_episode(preview, episode_index):
                                logging.info("Confirmed re-record for episode %d.", episode_index)
                                rerecord_episode = True
                                done = True
                            else:
                                logging.info(
                                    "Re-record cancelled for episode %d. Recording continues.",
                                    episode_index,
                                )
                        elif preview_event == PREVIEW_STOP_SESSION:
                            stop_requested = True
                            done = True

                    obs, _, terminated, truncated, info = env.step(action)
                    done = done or bool(terminated or truncated)
                    step += 1

                    if cfg.recording.sleep_to_fps:
                        elapsed_s = time.perf_counter() - loop_start_t
                        sleep_s = control_interval - elapsed_s
                        if sleep_s > 0:
                            time.sleep(sleep_s)
            except KeyboardInterrupt:
                interrupted = True
                logging.info("Interrupted during episode %d.", episode_index)

            if rerecord_episode:
                dataset.clear_episode_buffer()
                logging.info("Discarded episode %d. Recording it again.", episode_index)
                continue

            if recorded_frames == 0:
                dataset.clear_episode_buffer()
                if interrupted or stop_requested:
                    logging.info("No frames recorded for interrupted episode %d.", episode_index)
                    break
                raise RuntimeError("Episode has zero recorded frames. Check fps settings and action source.")

            if interrupted and not cfg.recording.save_partial_episode_on_interrupt:
                dataset.clear_episode_buffer()
                logging.info("Discarded interrupted episode %d.", episode_index)
                break

            if not interrupted and not stop_requested:
                rest_event = _wait_after_episode(
                    cfg=cfg,
                    adapter=adapter,
                    observation=obs,
                    info=info,
                    episode_index=episode_index,
                    run_episode_number=display_episode_number,
                    preview=preview,
                    env=env,
                    action_source=action_source,
                    control_fps=control_fps,
                )
                if rest_event == PREVIEW_RERECORD_EPISODE:
                    dataset.clear_episode_buffer()
                    logging.info(
                        "Discarded unsaved episode %d after confirmation. Recording it again.",
                        episode_index,
                    )
                    continue
                if rest_event == PREVIEW_STOP_SESSION:
                    stop_requested = True
                    logging.info(
                        "Stop requested during review for episode %d. Saving it before exit.",
                        episode_index,
                    )

            dataset.save_episode()
            _append_matrix_placement_log(
                dataset_root=Path(dataset.root),
                episode_index=episode_index,
                info=episode_reset_info,
            )
            _append_action_reference_profile_log(
                dataset_root=Path(dataset.root),
                episode_index=episode_index,
                info=episode_reset_info,
            )
            recorded_episodes += 1
            duration_s = time.perf_counter() - episode_start_t
            logging.info(
                "Saved episode %d (%d/%d in this run), recorded_frames=%d, "
                "control_steps=%d, duration=%.2fs",
                episode_index,
                recorded_episodes,
                episodes_to_record,
                recorded_frames,
                step,
                duration_s,
            )
            if stop_requested:
                logging.info("Stop requested from preview window. Exiting collection loop.")
                break
            if interrupted:
                logging.info("Interrupted run saved partial episode and will finalize.")
                break
    finally:
        if preview is not None:
            with contextlib.suppress(Exception):
                preview.close()
        if dataset is not None:
            dataset.finalize()
            if cfg.dataset.push_to_hub:
                logging.info("Pushing dataset to Hub: %s", cfg.dataset.repo_id)
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
                logging.info("Push completed.")
            else:
                logging.info("Push skipped (`push_to_hub=false`).")
        env.close()


def _write_or_validate_sim_teleop_manifest(
    *,
    dataset: Any,
    cfg: SimCollectConfig,
    config_path: Path,
) -> Path:
    """Pin the control path without adding nonstandard LeRobot training features."""
    expected = _build_sim_teleop_manifest(cfg, config_path=config_path)
    manifest_path = Path(dataset.root) / "meta" / SIM_TELEOP_MANIFEST_NAME

    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("control_fingerprint_sha256") != expected["control_fingerprint_sha256"]:
            raise ValueError(
                "Refusing to mix episodes collected with different MuJoCo/delay/shift "
                f"settings in {dataset.root}. Start a new dataset root."
            )
        return manifest_path

    if int(dataset.num_episodes) > 0:
        raise ValueError(
            "Existing dataset has episodes but no sim teleop provenance manifest. "
            "Start a new dataset root instead of silently mixing collection settings."
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(expected, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)
    logging.info("Pinned sim teleop provenance: %s", manifest_path)
    return manifest_path


def _build_sim_teleop_manifest(
    cfg: SimCollectConfig,
    *,
    config_path: Path,
) -> dict[str, Any]:
    if cfg.env.backend != "mujoco_xml" or cfg.env.xml is None:
        raise ValueError("Sim teleop provenance requires env.backend='mujoco_xml'.")

    xml_record = _file_record(cfg.env.xml.xml_path)
    matrix_cfg = cfg.env.xml.matrix_random_placements
    placement_table_record = (
        _file_record(matrix_cfg.table_path) if matrix_cfg is not None else None
    )
    reference_cfg = cfg.env.xml.action_reference
    spec_record: dict[str, Any] | None = None
    timing: dict[str, Any] | None = None
    if reference_cfg.spec_path is not None:
        spec_record = _file_record(reference_cfg.spec_path)
        spec_payload = json.loads(
            _resolve_project_path(reference_cfg.spec_path).read_text(encoding="utf-8")
        )
        timing_raw = spec_payload.get("timing", {})
        timing = {
            "delay_buffer_frames": timing_raw.get("delay_buffer_frames"),
            "target_output_lag_frames": timing_raw.get("target_output_lag_frames"),
            "randomized": timing_raw.get("randomized"),
        }

    control_identity = {
        "xml_sha256": xml_record["sha256"],
        "action_reference_mode": reference_cfg.mode,
        "action_reference_spec_sha256": spec_record["sha256"] if spec_record else None,
        "action_reference_randomization": (
            reference_cfg.randomization.as_dict()
            if reference_cfg.randomization is not None
            else None
        ),
        "camera_names": dict(cfg.env.xml.camera_names),
        "camera_brightness": {
            name: adjustment.as_dict()
            for name, adjustment in cfg.env.xml.camera_brightness.items()
        },
        "matrix_placement_table_sha256": (
            placement_table_record["sha256"] if placement_table_record else None
        ),
        "timing": timing,
        "control_fps": cfg.recording.control_fps or cfg.dataset.fps,
        "dataset_fps": cfg.dataset.fps,
        "dataset_angle_unit": cfg.adapter.dataset_angle_unit,
        "state_names": cfg.adapter.state_names,
        "action_names": cfg.adapter.action_names,
        "leader_keys": [
            key.strip()
            for key in os.getenv(
                "REAL_TELEOP_KEYS",
                "shoulder_pan.pos,shoulder_lift.pos,elbow_flex.pos,"
                "wrist_flex.pos,wrist_roll.pos,gripper.pos",
            ).split(",")
            if key.strip()
        ],
        "leader_preclip_enabled": os.getenv("REAL_TELEOP_CLIP", "false").strip().lower()
        in {"1", "true", "yes", "y", "on"},
    }
    fingerprint = hashlib.sha256(
        json.dumps(control_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return {
        "schema_version": 1,
        "collection_kind": "real_leader_to_mujoco_follower_teleop",
        "training_semantics": {
            "recorded_action": (
                "leader command after unit mapping, before MuJoCo delay/shift/final clipping"
            ),
            "recorded_observation": "MuJoCo follower state and rendered cameras",
            "applied_action_is_training_feature": False,
            "delay_history_before_first_recorded_frame": (
                "live leader actions from the unrecorded prepare phase"
                if cfg.recording.teleop_during_prepare and cfg.recording.prepare_time_s > 0
                else "first available episode action"
            ),
            "policy_rollout_used_for_collection": False,
        },
        "source_config": _file_record(config_path),
        "rendering": {
            "camera_names": dict(cfg.env.xml.camera_names),
            "camera_brightness": {
                name: adjustment.as_dict()
                for name, adjustment in cfg.env.xml.camera_brightness.items()
            },
        },
        "matrix_random_placements": {
            "table": placement_table_record,
        },
        "mujoco_xml": xml_record,
        "action_reference": {
            "mode": reference_cfg.mode,
            "delay_applied": reference_cfg.mode != "off",
            "shift_applied": reference_cfg.mode
            in {"lag_plus_shift", RANDOM_SHIFT_MODE},
            "shift_randomized_per_episode": reference_cfg.mode == RANDOM_SHIFT_MODE,
            "randomization": (
                reference_cfg.randomization.as_dict()
                if reference_cfg.randomization is not None
                else None
            ),
            "episode_profile_log": (
                ACTION_REFERENCE_PROFILE_LOG_NAME
                if reference_cfg.mode == RANDOM_SHIFT_MODE
                else None
            ),
            "spec": spec_record,
            "timing": timing,
        },
        "control_identity": control_identity,
        "control_fingerprint_sha256": fingerprint,
    }


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _file_record(value: str | Path) -> dict[str, Any]:
    path = _resolve_project_path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    try:
        display_path = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        display_path = str(path)
    return {
        "path": display_path,
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _make_env(cfg: EnvCollectConfig) -> Any:
    if cfg.backend == "mujoco_xml":
        if cfg.xml is None:
            raise ValueError("env.xml is required for env.backend='mujoco_xml'.")
        return MujocoXmlEnv(cfg.xml)
    if not cfg.id:
        raise ValueError("env.id is required for env.backend='gym'.")
    return gym.make(cfg.id, **cfg.kwargs)


def _log_mujoco_timing(cfg: EnvCollectConfig, env: Any, control_fps: int) -> None:
    if cfg.backend != "mujoco_xml" or cfg.xml is None:
        return

    physics_timestep = float(env.model.opt.timestep)
    frame_skip = max(cfg.xml.frame_skip, 1)
    simulated_seconds_per_wall_second = physics_timestep * frame_skip * control_fps
    logging.info(
        "MuJoCo timing: timestep=%.6fs, frame_skip=%d, control_fps=%d, "
        "sim/wall ratio=%.4f",
        physics_timestep,
        frame_skip,
        control_fps,
        simulated_seconds_per_wall_second,
    )
    if not np.isclose(simulated_seconds_per_wall_second, 1.0, rtol=0.02, atol=0.0):
        logging.warning(
            "MuJoCo is not running near real time: sim/wall ratio=%.4f. "
            "Adjust frame_skip or control_fps.",
            simulated_seconds_per_wall_second,
        )


def _append_matrix_placement_log(
    dataset_root: Path,
    episode_index: int,
    info: dict[str, Any],
) -> None:
    matrix_placements = info.get("matrix_random_placements")
    if not matrix_placements:
        return

    log_path = dataset_root / "meta" / "matrix_random_placements.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "episode_index": episode_index,
        "matrix_random_placements": matrix_placements,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _append_action_reference_profile_log(
    dataset_root: Path,
    episode_index: int,
    info: dict[str, Any],
) -> None:
    action_reference = info.get("action_reference") or {}
    if not action_reference.get("randomized_per_episode"):
        return

    log_path = dataset_root / "meta" / ACTION_REFERENCE_PROFILE_LOG_NAME
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "episode_index": episode_index,
        "profile_id": action_reference.get("profile_id"),
        "profile_levels": action_reference.get("profile_levels"),
        "shift_joints": action_reference.get("shift_joints"),
        "delay_buffer_frames": action_reference.get("delay_buffer_frames"),
        "shift_spec_sha256": action_reference.get("shift_spec_sha256"),
        "randomization": action_reference.get("randomization"),
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _matrix_placement_status(info: dict[str, Any] | None) -> str:
    placements = (info or {}).get("matrix_random_placements") or {}
    object_cell = None
    container_cell = None
    for placement in placements.values():
        if "yaw" in placement:
            object_cell = placement.get("cell")
        elif "source_row" in placement:
            container_cell = placement.get("cell")

    if object_cell and container_cell:
        return f"OBJECT {str(object_cell).upper()} | BOX {str(container_cell).upper()}"
    return "OBJECT/BOX fixed"


def _create_or_resume_dataset(
    cfg: SimCollectConfig,
    dataset_features: dict[str, dict],
) -> LeRobotDataset:
    should_resume = cfg.resume
    if (
        not should_resume
        and cfg.dataset.auto_resume_if_root_exists
        and Path(cfg.dataset.root).exists()
    ):
        logging.info(
            "Dataset root already exists and auto_resume_if_root_exists=true. Switching to resume mode."
        )
        should_resume = True

    if should_resume:
        logging.info("Resuming dataset: %s", cfg.dataset.repo_id)
        dataset = LeRobotDataset.resume(
            repo_id=cfg.dataset.repo_id,
            root=cfg.dataset.root,
            video_backend=cfg.dataset.video_backend,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_threads=cfg.dataset.encoder_threads,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            image_writer_threads=cfg.dataset.image_writer_threads,
        )
        if dataset.meta.fps != cfg.dataset.fps:
            raise ValueError(
                f"Dataset FPS mismatch on resume: expected {cfg.dataset.fps}, "
                f"got {dataset.meta.fps}."
            )
        if (
            cfg.dataset.robot_type is not None
            and dataset.meta.robot_type != cfg.dataset.robot_type
        ):
            raise ValueError(
                "Dataset robot_type mismatch on resume: "
                f"expected {cfg.dataset.robot_type!r}, got {dataset.meta.robot_type!r}."
            )
        _assert_feature_compatibility(dataset.features, dataset_features)
        return dataset

    logging.info("Creating dataset: %s", cfg.dataset.repo_id)
    return LeRobotDataset.create(
        repo_id=cfg.dataset.repo_id,
        fps=cfg.dataset.fps,
        features=dataset_features,
        root=cfg.dataset.root,
        robot_type=cfg.dataset.robot_type,
        use_videos=cfg.dataset.use_videos,
        video_backend=cfg.dataset.video_backend,
        streaming_encoding=cfg.dataset.streaming_encoding,
        encoder_threads=cfg.dataset.encoder_threads,
        encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
        image_writer_threads=cfg.dataset.image_writer_threads,
    )


def _assert_feature_compatibility(existing: dict[str, dict], expected: dict[str, dict]) -> None:
    missing = sorted(set(expected.keys()) - set(existing.keys()))
    if missing:
        raise ValueError(
            "Dataset feature keys mismatch on resume.\n"
            f"Missing in existing dataset: {missing}"
        )

    for key in expected:
        exp = expected[key]
        got = existing[key]
        if exp.get("dtype") != got.get("dtype") or tuple(exp.get("shape", ())) != tuple(
            got.get("shape", ())
        ):
            raise ValueError(
                f"Feature mismatch for '{key}' on resume. "
                f"Expected dtype/shape={exp.get('dtype')}/{tuple(exp.get('shape', ()))}, "
                f"got {got.get('dtype')}/{tuple(got.get('shape', ()))}."
            )


def _resolve_episodes_to_record(
    dataset_cfg: DatasetCollectConfig,
    existing_episodes: int,
) -> int:
    if existing_episodes < 0:
        raise ValueError("existing_episodes must be >= 0.")
    if dataset_cfg.episode_count_mode == "additional":
        return dataset_cfg.num_episodes
    if existing_episodes > dataset_cfg.num_episodes:
        raise ValueError(
            f"Existing dataset has {existing_episodes} episodes, exceeding the configured "
            f"target_total={dataset_cfg.num_episodes}."
        )
    return dataset_cfg.num_episodes - existing_episodes


def _resolve_step_limit(
    cfg: SimCollectConfig,
    env: Any,
    control_fps: int,
) -> int | None:
    by_time: int | None = None
    if cfg.dataset.episode_time_s is not None:
        by_time = int(round(cfg.dataset.episode_time_s * control_fps))
        if by_time <= 0:
            raise ValueError("episode_time_s * fps must be > 0, or episode_time_s must be null.")

    by_env = cfg.env.max_steps_per_episode
    if by_env is None:
        env_spec_max_steps = getattr(getattr(env, "spec", None), "max_episode_steps", None)
        if env_spec_max_steps is not None:
            by_env = int(env_spec_max_steps)

    if by_env is None:
        return by_time
    if by_env <= 0:
        raise ValueError("max_steps_per_episode must be > 0.")
    if by_time is None:
        return by_env
    return min(by_time, by_env)


def _validate_action_shape_against_env(action: np.ndarray, env: Any) -> None:
    action_shape = tuple(action.shape)
    space_shape = tuple(getattr(getattr(env, "action_space", None), "shape", ()) or ())
    if not space_shape:
        return
    if action_shape != space_shape:
        raise ValueError(
            f"Action source shape {action_shape} does not match env.action_space.shape {space_shape}."
        )


def _wait_before_episode(
    cfg: SimCollectConfig,
    adapter: SimDatasetAdapter,
    observation: Any,
    info: dict[str, Any] | None,
    episode_index: int,
    run_episode_number: int,
    preview: Any,
    env: Any,
    action_source: Any,
    control_fps: int,
) -> tuple[str, Any, dict[str, Any] | None]:
    if cfg.recording.require_enter_before_episode:
        input(f"Prepare episode {episode_index}, then press ENTER to start recording...")
        return PREVIEW_CONTINUE, observation, info

    wait_s = cfg.recording.prepare_time_s
    if wait_s <= 0:
        return PREVIEW_CONTINUE, observation, info

    live_teleop = cfg.recording.teleop_during_prepare
    logging.info(
        "Prepare episode %d. Recording starts in %.1fs (live_teleop=%s).",
        episode_index,
        wait_s,
        live_teleop,
    )
    deadline = time.perf_counter() + wait_s
    last_logged_second: int | None = None
    current_observation = observation
    current_info = info
    unrecorded_step = 0

    while True:
        loop_start_t = time.perf_counter()
        remaining_s = deadline - time.perf_counter()
        if remaining_s <= 0:
            return PREVIEW_CONTINUE, current_observation, current_info

        remaining_second = int(remaining_s) + 1
        if remaining_second != last_logged_second:
            logging.info("Episode %d starts in %ds.", episode_index, remaining_second)
            last_logged_second = remaining_second

        if live_teleop:
            current_observation, current_info, env_done = _advance_unrecorded_teleop(
                action_source=action_source,
                env=env,
                observation=current_observation,
                info=current_info,
                step_index=unrecorded_step,
                episode_index=episode_index,
            )
            unrecorded_step += 1
            if env_done:
                logging.warning(
                    "Environment ended during preparation for episode %d.",
                    episode_index,
                )
                return PREVIEW_CONTINUE, current_observation, current_info

        if preview is not None:
            obs_values = adapter.observation_to_values(current_observation)
            live_label = " | LIVE TELEOP" if live_teleop else ""
            placement_status = _matrix_placement_status(current_info)
            preview_event = preview.show(
                images={
                    image_name: obs_values[image_name]
                    for image_name in adapter.image_names
                },
                window_name=cfg.recording.preview_window_name,
                scale=cfg.recording.preview_scale,
                top_grid=cfg.recording.top_grid,
                status_text=(
                    f"PREPARE {run_episode_number}/{cfg.dataset.num_episodes} "
                    f"| {placement_status} | starts in {remaining_s:.1f}s{live_label}"
                ),
                help_text="Enter/Space/Right: start now | Q/Esc: stop",
            )
            if preview_event == PREVIEW_FINISH_EPISODE:
                logging.info("Preparation skipped for episode %d.", episode_index)
                return PREVIEW_CONTINUE, current_observation, current_info
            if preview_event == PREVIEW_STOP_SESSION:
                return PREVIEW_STOP_SESSION, current_observation, current_info

        _sleep_during_wait(
            loop_start_t=loop_start_t,
            remaining_s=remaining_s,
            control_fps=control_fps,
            live_teleop=live_teleop,
            preview_enabled=preview is not None,
            sleep_to_fps=cfg.recording.sleep_to_fps,
        )


def _wait_after_episode(
    cfg: SimCollectConfig,
    adapter: SimDatasetAdapter,
    observation: Any,
    info: dict[str, Any] | None,
    episode_index: int,
    run_episode_number: int,
    preview: Any,
    env: Any,
    action_source: Any,
    control_fps: int,
) -> str:
    wait_s = cfg.recording.rest_time_s
    if wait_s <= 0:
        return PREVIEW_CONTINUE

    live_teleop = cfg.recording.teleop_during_rest
    logging.info(
        "Episode %d is waiting %.1fs for review before it is saved (live_teleop=%s).",
        episode_index,
        wait_s,
        live_teleop,
    )
    deadline = time.perf_counter() + wait_s
    current_observation = observation
    current_info = info
    unrecorded_step = 0

    while True:
        loop_start_t = time.perf_counter()
        remaining_s = deadline - time.perf_counter()
        if remaining_s <= 0:
            logging.info("Review time ended for episode %d. Saving it.", episode_index)
            return PREVIEW_CONTINUE

        if live_teleop:
            current_observation, current_info, env_done = _advance_unrecorded_teleop(
                action_source=action_source,
                env=env,
                observation=current_observation,
                info=current_info,
                step_index=unrecorded_step,
                episode_index=episode_index,
            )
            unrecorded_step += 1
            if env_done:
                logging.warning(
                    "Environment ended during review for episode %d.",
                    episode_index,
                )
                return PREVIEW_CONTINUE

        if preview is not None:
            obs_values = adapter.observation_to_values(current_observation)
            live_label = " | LIVE TELEOP" if live_teleop else ""
            placement_status = _matrix_placement_status(current_info)
            preview_event = preview.show(
                images={
                    image_name: obs_values[image_name]
                    for image_name in adapter.image_names
                },
                window_name=cfg.recording.preview_window_name,
                scale=cfg.recording.preview_scale,
                top_grid=cfg.recording.top_grid,
                status_text=(
                    f"REST / REVIEW {run_episode_number}/{cfg.dataset.num_episodes} "
                    f"| {placement_status} | auto-save in {remaining_s:.1f}s{live_label}"
                ),
                help_text=(
                    "Enter/Space/Right: save + next | "
                    "R/Left: confirm re-record | Q/Esc: save + stop"
                ),
            )
            if preview_event == PREVIEW_FINISH_EPISODE:
                logging.info("Review ended early for episode %d. Saving it.", episode_index)
                return PREVIEW_CONTINUE
            if preview_event == PREVIEW_RERECORD_EPISODE:
                if _confirm_rerecord_episode(preview, episode_index):
                    return PREVIEW_RERECORD_EPISODE
                logging.info(
                    "Re-record cancelled for episode %d. Review continues.",
                    episode_index,
                )
            if preview_event == PREVIEW_STOP_SESSION:
                return PREVIEW_STOP_SESSION

        _sleep_during_wait(
            loop_start_t=loop_start_t,
            remaining_s=remaining_s,
            control_fps=control_fps,
            live_teleop=live_teleop,
            preview_enabled=preview is not None,
            sleep_to_fps=cfg.recording.sleep_to_fps,
        )


def _advance_unrecorded_teleop(
    action_source: Any,
    env: Any,
    observation: Any,
    info: dict[str, Any] | None,
    step_index: int,
    episode_index: int,
) -> tuple[Any, dict[str, Any] | None, bool]:
    action = action_source.next_action(
        observation=observation,
        info=info,
        step_index=step_index,
        episode_index=episode_index,
    )
    next_observation, _, terminated, truncated, next_info = env.step(action)
    return next_observation, next_info, bool(terminated or truncated)


def _sleep_during_wait(
    loop_start_t: float,
    remaining_s: float,
    control_fps: int,
    live_teleop: bool,
    preview_enabled: bool,
    sleep_to_fps: bool,
) -> None:
    if live_teleop:
        if not sleep_to_fps:
            return
        elapsed_s = time.perf_counter() - loop_start_t
        sleep_s = min(max((1.0 / control_fps) - elapsed_s, 0.0), remaining_s)
    else:
        sleep_s = min(0.05 if preview_enabled else 0.25, remaining_s)
    if sleep_s > 0:
        time.sleep(sleep_s)


def _confirm_rerecord_episode(preview: Any, episode_index: int) -> bool:
    if preview is not None and hasattr(preview, "confirm_rerecord"):
        return bool(preview.confirm_rerecord(episode_index))

    prompt = (
        f"Discard all unsaved frames and temporary video data for episode "
        f"{episode_index}, then record it again? [y/N]: "
    )
    try:
        return input(prompt).strip().lower() in {"y", "yes"}
    except EOFError:
        logging.warning("Cannot request re-record confirmation without interactive input.")
        return False


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _create_camera_preview(window_name: str) -> Any | None:
    if os.environ.get("MUJOCO_GL", "").strip().lower() == "egl":
        try:
            preview = _TkCameraPreview(window_name)
            logging.info("Camera preview backend: Tkinter (EGL-safe)")
            return preview
        except Exception as exc:
            logging.warning(
                "Camera preview disabled: Tkinter is unavailable and OpenCV is "
                "not attempted because its GUI conflicts with MuJoCo EGL (%s)",
                exc,
            )
            return None

    try:
        import cv2

        preview = _OpenCvCameraPreview(cv2, window_name)
        logging.info("Camera preview backend: OpenCV")
        return preview
    except Exception as exc:
        logging.info("OpenCV GUI preview unavailable (%s). Trying Tkinter.", exc)

    try:
        preview = _TkCameraPreview(window_name)
        logging.info("Camera preview backend: Tkinter")
        return preview
    except Exception as exc:
        logging.warning("Camera preview disabled: no GUI backend is available (%s)", exc)
        return None


class _OpenCvCameraPreview:
    def __init__(self, cv2: Any, window_name: str):
        self.cv2 = cv2
        self.window_name = window_name
        self.cv2.namedWindow(window_name, self.cv2.WINDOW_NORMAL)

    def show(
        self,
        images: dict[str, np.ndarray],
        window_name: str,
        scale: float,
        top_grid: TopGridConfig,
        status_text: str = "",
        help_text: str = "",
    ) -> str:
        _ = window_name
        return _show_camera_preview(
            cv2=self.cv2,
            images=images,
            window_name=self.window_name,
            scale=scale,
            top_grid=top_grid,
            status_text=status_text,
            help_text=help_text,
        )

    def confirm_rerecord(self, episode_index: int) -> bool:
        prompt = (
            f"Discard unsaved episode {episode_index} and record it again? "
            "Type y to confirm [y/N]: "
        )
        try:
            return input(prompt).strip().lower() in {"y", "yes"}
        except EOFError:
            logging.warning("Re-record cancelled because confirmation input is unavailable.")
            return False

    def close(self) -> None:
        self.cv2.destroyWindow(self.window_name)


class _TkCameraPreview:
    def __init__(self, window_name: str):
        import tkinter as tk
        from tkinter import messagebox, ttk

        from PIL import Image, ImageDraw, ImageTk

        self.tk = tk
        self.messagebox = messagebox
        self.ttk = ttk
        self.Image = Image
        self.ImageDraw = ImageDraw
        self.ImageTk = ImageTk
        self.pending_event = PREVIEW_CONTINUE
        self.running = True
        self.photo = None

        self.root = tk.Tk()
        self.root.title(window_name)
        self.root.protocol("WM_DELETE_WINDOW", self._request_stop)
        self.root.bind("<Escape>", lambda _event: self._request_stop())
        self.root.bind("q", lambda _event: self._request_stop())
        self.root.bind("Q", lambda _event: self._request_stop())
        for key in ("<Right>", "<Return>", "<space>"):
            self.root.bind(key, lambda _event: self._set_event(PREVIEW_FINISH_EPISODE))
        for key in ("<Left>", "<BackSpace>", "r", "R"):
            self.root.bind(key, lambda _event: self._set_event(PREVIEW_RERECORD_EPISODE))

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="center",
        )
        self.status_label.pack(fill="x", padx=8, pady=(6, 2))

        self.label = tk.Label(self.root)
        self.label.pack()

        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=8, pady=(4, 8))
        self.finish_button = ttk.Button(
            controls,
            text="Tiếp tục",
            command=lambda: self._set_event(PREVIEW_FINISH_EPISODE),
        )
        self.finish_button.pack(side="left", expand=True, fill="x", padx=3)
        ttk.Button(
            controls,
            text="Quay lại / Thu lại",
            command=lambda: self._set_event(PREVIEW_RERECORD_EPISODE),
        ).pack(side="left", expand=True, fill="x", padx=3)
        ttk.Button(
            controls,
            text="Dừng",
            command=self._request_stop,
        ).pack(side="left", expand=True, fill="x", padx=3)

    def _set_event(self, event: str) -> None:
        self.pending_event = event

    def _request_stop(self) -> None:
        self.running = False
        self.pending_event = PREVIEW_STOP_SESSION

    def show(
        self,
        images: dict[str, np.ndarray],
        window_name: str,
        scale: float,
        top_grid: TopGridConfig,
        status_text: str = "",
        help_text: str = "",
    ) -> str:
        _ = window_name
        if not images:
            raise ValueError("Camera preview requires at least one image.")
        if not self.running:
            return PREVIEW_STOP_SESSION

        self.finish_button.configure(
            text=(
                "Kết thúc & lưu episode"
                if status_text.startswith("RECORDING")
                else "Tiếp tục"
            )
        )

        camera_images = []
        for image_name, image_array in images.items():
            image = self.Image.fromarray(np.asarray(image_array, dtype=np.uint8))
            if image_name == "top" and top_grid.enabled:
                self._draw_top_grid(image, top_grid)
            draw = self.ImageDraw.Draw(image)
            camera_label_y = 30 if status_text else 8
            draw.text(
                (10, camera_label_y),
                image_name,
                fill=(0, 0, 0),
                stroke_width=3,
                stroke_fill=(0, 0, 0),
            )
            draw.text((10, camera_label_y), image_name, fill=(255, 255, 0))
            camera_images.append(image)

        width = sum(image.width for image in camera_images)
        height = max(image.height for image in camera_images)
        frame = self.Image.new("RGB", (width, height))
        x_offset = 0
        for image in camera_images:
            frame.paste(image, (x_offset, 0))
            x_offset += image.width

        draw = self.ImageDraw.Draw(frame)
        if status_text:
            draw.text(
                (10, 8),
                status_text,
                fill=(255, 255, 0),
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
        if not help_text:
            help_text = "Enter/Space/Right: finish | R/Left: re-record | Q/Esc: stop"
        draw.text(
            (10, max(frame.height - 22, 0)),
            help_text,
            fill=(255, 255, 0),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

        resolved_scale = max(float(scale), 0.1)
        if abs(resolved_scale - 1.0) > 1e-6:
            frame = frame.resize(
                (
                    max(1, int(frame.width * resolved_scale)),
                    max(1, int(frame.height * resolved_scale)),
                ),
                self.Image.Resampling.LANCZOS,
            )

        self.photo = self.ImageTk.PhotoImage(frame)
        self.label.configure(image=self.photo)
        self.status_var.set(status_text or "Ready")
        try:
            self.root.update_idletasks()
            self.root.update()
        except self.tk.TclError:
            self._request_stop()

        event = self.pending_event
        if event != PREVIEW_STOP_SESSION:
            self.pending_event = PREVIEW_CONTINUE
        return event

    def confirm_rerecord(self, episode_index: int) -> bool:
        return bool(
            self.messagebox.askyesno(
                "Xác nhận thu lại",
                (
                    f"Xóa toàn bộ frame và video tạm CHƯA LƯU của episode "
                    f"{episode_index}, rồi thu lại episode này?"
                ),
                parent=self.root,
            )
        )

    def _draw_top_grid(self, image: Any, top_grid: TopGridConfig) -> None:
        draw = self.ImageDraw.Draw(image)
        width, height = image.size
        x_min, y_min, x_max, y_max = _grid_bounds(
            width=width,
            height=height,
            margin_left=top_grid.margin_left,
            margin_top=top_grid.margin_top,
            margin_right=top_grid.margin_right,
            margin_bottom=top_grid.margin_bottom,
        )
        grid_width = x_max - x_min
        grid_height = y_max - y_min
        color = (255, 255, 0)
        draw.rectangle((x_min, y_min, x_max - 1, y_max - 1), outline=color, width=1)
        for col in range(1, top_grid.cols):
            x = x_min + round(col * grid_width / top_grid.cols)
            draw.line((x, y_min, x, y_max - 1), fill=color, width=1)
        for row in range(1, top_grid.rows):
            y = y_min + round(row * grid_height / top_grid.rows)
            draw.line((x_min, y, x_max - 1, y), fill=color, width=1)

        if not top_grid.labels:
            return
        for row in range(top_grid.rows):
            for col in range(top_grid.cols):
                label = _top_grid_label(row, col)
                x = x_min + round((col + 0.5) * grid_width / top_grid.cols)
                y = y_min + round((row + 0.5) * grid_height / top_grid.rows)
                draw.text(
                    (x, y),
                    label,
                    anchor="mm",
                    fill=color,
                    stroke_width=2,
                    stroke_fill=(0, 0, 0),
                )

    def close(self) -> None:
        self.running = False
        with contextlib.suppress(self.tk.TclError):
            self.root.destroy()


def _show_camera_preview(
    cv2: Any,
    images: dict[str, np.ndarray],
    window_name: str,
    scale: float,
    top_grid: TopGridConfig,
    status_text: str = "",
    help_text: str = "",
) -> str:
    if not images:
        raise ValueError("Camera preview requires at least one image.")

    camera_frames: list[tuple[str, np.ndarray]] = []
    for image_name, image in images.items():
        camera_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if image_name == "top" and top_grid.enabled:
            _draw_top_grid(
                cv2=cv2,
                image=camera_bgr,
                rows=top_grid.rows,
                cols=top_grid.cols,
                labels=top_grid.labels,
                margin_left=top_grid.margin_left,
                margin_top=top_grid.margin_top,
                margin_right=top_grid.margin_right,
                margin_bottom=top_grid.margin_bottom,
            )
        camera_frames.append((image_name, camera_bgr))

    frame = np.concatenate([camera_frame for _, camera_frame in camera_frames], axis=1)
    source_widths = [camera_frame.shape[1] for _, camera_frame in camera_frames]

    scale = max(float(scale), 0.1)
    if abs(scale - 1.0) > 1e-6:
        new_w = int(frame.shape[1] * scale)
        new_h = int(frame.shape[0] * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x_offset = 0
    for (image_name, _), source_width in zip(camera_frames, source_widths, strict=True):
        label_x = int(x_offset * scale) + 10
        label_y = 48 if status_text else 24
        cv2.putText(
            frame,
            image_name,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        x_offset += source_width
    if status_text:
        cv2.putText(
            frame,
            status_text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            status_text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if not help_text:
        help_text = "Enter/Space/Right: finish | R/Left: re-record | Q/Esc: stop"
    cv2.putText(
        frame,
        help_text,
        (10, max(frame.shape[0] - 12, 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        help_text,
        (10, max(frame.shape[0] - 12, 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imshow(window_name, frame)
    wait_key = getattr(cv2, "waitKeyEx", cv2.waitKey)
    key = wait_key(1)
    return _preview_key_to_event(key)


def _preview_key_to_event(key: int) -> str:
    if key < 0:
        return PREVIEW_CONTINUE

    if key in (27, ord("q"), ord("Q")):
        return PREVIEW_STOP_SESSION

    # OpenCV waitKeyEx key codes: Windows arrows use 0x25/0x27 << 16; X11 uses 65361/65363.
    if key in (13, 10, 32, 2555904, 65363):
        return PREVIEW_FINISH_EPISODE

    if key in (8, 127, ord("r"), ord("R"), 2424832, 65361):
        return PREVIEW_RERECORD_EPISODE

    return PREVIEW_CONTINUE


def _draw_top_grid(
    cv2: Any,
    image: np.ndarray,
    rows: int,
    cols: int,
    labels: bool,
    margin_left: int = DEFAULT_GRID_MARGIN_LEFT,
    margin_top: int = DEFAULT_GRID_MARGIN_TOP,
    margin_right: int = DEFAULT_GRID_MARGIN_RIGHT,
    margin_bottom: int = DEFAULT_GRID_MARGIN_BOTTOM,
) -> None:
    h, w = image.shape[:2]
    x_min, y_min, x_max, y_max = _grid_bounds(
        width=w,
        height=h,
        margin_left=margin_left,
        margin_top=margin_top,
        margin_right=margin_right,
        margin_bottom=margin_bottom,
    )
    grid_width = x_max - x_min
    grid_height = y_max - y_min
    line_color = (0, 255, 255)
    text_color = (0, 255, 255)
    shadow_color = (0, 0, 0)
    thickness = max(1, round(min(h, w) / 360))

    cv2.rectangle(
        image,
        (x_min, y_min),
        (x_max - 1, y_max - 1),
        line_color,
        thickness,
        cv2.LINE_AA,
    )
    for col in range(1, cols):
        x = x_min + round(col * grid_width / cols)
        cv2.line(image, (x, y_min), (x, y_max - 1), line_color, thickness, cv2.LINE_AA)
    for row in range(1, rows):
        y = y_min + round(row * grid_height / rows)
        cv2.line(image, (x_min, y), (x_max - 1, y), line_color, thickness, cv2.LINE_AA)

    if not labels:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(w / 640.0, h / 480.0) * 0.55)
    text_thickness = max(1, round(thickness))
    for row in range(rows):
        for col in range(cols):
            label = _top_grid_label(row, col)
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
            cx = x_min + round((col + 0.5) * grid_width / cols)
            cy = y_min + round((row + 0.5) * grid_height / rows)
            origin = (cx - tw // 2, cy + (th - baseline) // 2)
            cv2.putText(
                image,
                label,
                origin,
                font,
                font_scale,
                shadow_color,
                text_thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                label,
                origin,
                font,
                font_scale,
                text_color,
                text_thickness,
                cv2.LINE_AA,
            )


def _grid_bounds(
    width: int,
    height: int,
    margin_left: int,
    margin_top: int,
    margin_right: int,
    margin_bottom: int,
) -> tuple[int, int, int, int]:
    x_min = int(margin_left)
    y_min = int(margin_top)
    x_max = int(width) - int(margin_right)
    y_max = int(height) - int(margin_bottom)
    if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
        raise ValueError("Grid margins must stay inside the image.")
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Grid margins leave no drawable area.")
    return x_min, y_min, x_max, y_max


def _top_grid_label(row: int, col: int) -> str:
    row_name = chr(ord("a") + row) if row < 26 else f"r{row + 1}"
    return f"{row_name}{col + 1}"


if __name__ == "__main__":
    main()
