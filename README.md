# SO-101 MuJoCo + LeRobot tutorial

Run a real SO-101 leader against the **act_policy_sim2real Original MuJoCo
scene**, record demonstrations, inspect structured robot data, and run a
pretrained ACT policy. Includes the scene, robot meshes and simulator source;
no checkout of the parent project is required.

**Tiếng Việt:** Bộ hướng dẫn dùng scene MuJoCo của `act_policy_sim2real`,
không dùng scene tabletop của LeRobot. Có teleop, record, visualize, rollout
kèm hai camera và upload LeRobotDataset lên Hugging Face. Không training tại chỗ.

## Requirements

- Tested on Ubuntu/Linux x86-64 with Python 3.12 and a working EGL/OpenGL driver.
- [uv](https://docs.astral.sh/uv/) and Git installed. Internet for initial setup
  and model download. Hardware is **not required** for simulation/policy rollout.
- SO-101 leader + USB adapter and calibration only for teleop/record.
- Desktop display and GUI-capable OpenCV for camera preview; Rerun for dataset
  viewing. Use `--headless` on servers. Other operating systems are unverified;
  these launchers require Bash (Windows users need a suitable Linux/WSL setup).

## Install and first run

```bash
git clone https://github.com/vietanh-mechatronics/so101-tutorial.git
cd so101-tutorial
uv sync --frozen
bash scripts/01_check_setup.sh
bash scripts/02_start_sim.sh --headless --seconds 2
```

The lockfile fixes Python dependencies and the exact LeRobot Git commit.
The default PyTorch wheels are CPU-only, so no NVIDIA GPU is required.
MuJoCo assets are in this repository, not fetched from a LeRobot environment.
The `.venv` is created locally; an existing Conda environment is not modified.
If the system lacks EGL/FFmpeg libraries, on Ubuntu install the relevant system
packages, for example `libegl1 libgl1 libglib2.0-0 ffmpeg`. Motor calibration and
interactive previews also need a desktop session and USB device permissions.

For desktop camera preview:

```bash
uv sync --frozen --extra gui
# If cv2 reports "function is not implemented", select the GUI wheel last:
uv pip install --python .venv/bin/python --force-reinstall opencv-python==4.13.0.92
bash scripts/02_start_sim.sh --seconds 10
```

LeRobot depends on the headless OpenCV distribution. The optional GUI wheel
shares its `cv2` namespace; force-reinstalling the GUI wheel restores HighGUI
when the headless wheel wins install ordering. The headless commands do not
need this workaround. A later `uv sync` may restore the locked headless wheel.

## Run the pretrained policy without hardware

```bash
bash scripts/08_download_model.sh
bash scripts/05_rollout.sh --headless --episodes 2 --seconds 30 --save-dataset
bash scripts/09_inspect_dataset.sh --full-video
```

The model is `ngocthuong2212/icra_2026_SIM_ONLY`, pinned to revision
`5d9827d9b75ab3aa602bd5fc2aa57acc13f20586` (~200 MB). Download it before an
offline demo. Rollout uses the saved pre/postprocessors and normalization, not
statistics from a newly recorded episode. The code disables the initial
ImageNet download because the full trained weights are available locally.

`--save-dataset` records **top/front videos, measured state, model action,
timestamp and task metadata** as LeRobotDataset in `data/so101_tutorial`.
Without this flag only JSONL traces are saved. The output root must be new;
existing datasets are never silently overwritten. Change `dataset.root` for
each new rollout. A reset discards the in-progress episode; stopping early
keeps completed episodes and discards the partial episode.

Each episode has 900 frames at 30 FPS when `--seconds 30` is used. CPU execution
may take several minutes per episode. Timestamp is the dataset timeline, not
USB latency or wall-clock speed. Original physics advances 0.033 seconds per
action (33 steps × 0.001 seconds). `--no-sleep` skips wall-clock pacing without
changing the physics. Reward is currently always zero; there is no automatic
task-success detector, and completing a rollout does not certify pick success.

Remove `--headless` to see top/front camera preview. Q/Esc stops; R restarts the
current episode and clears ACT queues. Every command creates its **own**
simulator; `02_start_sim.sh` is a preview, not a server needed by other commands.

## Configure your own run

```bash
cp configs/tutorial.json configs/local.json
```

Edit `configs/local.json`, then append `--config configs/local.json` to commands.
JSON paths are relative to the repository root. Script launchers work from any
working directory if invoked with the correct script path.

| Setting | Meaning |
|---|---|
| `simulation_config` | `configs/simulation/original.json`; keep the bundled scene/placement conventions |
| `leader.port` | Actual USB port, preferably `/dev/serial/by-id/...` |
| `leader.id` | ID associated with your leader's calibration |
| `leader.calibration_dir` | Your local calibration directory, ignored by Git |
| `dataset.repo_id` | `local/...` offline; `YOUR_HF_USERNAME/name` for upload |
| `dataset.root` | A new local output directory for a new run |
| `dataset.private` | Visibility when creating a Hub repo; does not change existing visibility |
| `num_episodes`, `episode_seconds` | Recording defaults; rollout also accepts CLI overrides |
| `policy.device` | `auto` or `cpu` by default; `cuda` requires replacing the CPU wheels and a matching driver |
| `fps`, `seed` | Keep 30 FPS; seed controls reset/placement |

The rollout example `configs/rollout.example.json` sets two episodes and a
separate output root. Copy it to `configs/rollout.json`, set your Hub username,
and use `--config configs/rollout.json`. Do not commit tokens or calibration.

## Connect and calibrate a real leader

Hardware is only used by teleop/record. No real follower is connected.
Find the port using `.venv/bin/lerobot-find-port`, then calibrate **your** robot:

```bash
.venv/bin/lerobot-calibrate --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 --teleop.id=my_so101_leader \
  --teleop.calibration_dir=calibration
```

Set the same port/ID/calibration directory in local config. Confirm USB access
and that every joint turns in the correct direction. Calibration is specific
to the physical device; this repo deliberately ships no calibration file.

```bash
bash scripts/01_check_setup.sh --config configs/local.json --with-leader
bash scripts/06_teleoperate.sh --config configs/local.json --seconds 60
bash scripts/03_record_episode.sh --config configs/local.json
```

Record defaults to one 25-second episode, with prepare/rest periods from the
collector config. Enter/Space/Right saves; R/Left re-records; Q/Esc saves and
stops. The record command uses the config's episode count/duration rather than
rollout CLI overrides. For an existing dataset use a new root or `--resume`:
`num_episodes` is the desired **total** episode count, not the number to append.

After a lost USB connection, stop with Ctrl+C, reconnect/check the port, and
restart teleop. Do not run two processes against one leader port.

## Understand and visualize the dataset

```bash
bash scripts/09_inspect_dataset.sh --config configs/local.json --full-video
bash scripts/04_visualize.sh --config configs/local.json --episode-index 0
```

At time t, `observation.state` and camera images are captured **before** applying
`action_t`. State is measured qpos; action is a target, so their values can
legitimately differ. The dataset stores:

| Feature | Contents |
|---|---|
| `observation.images.top`, `.front` | Virtual camera RGB video, 640×480 |
| `observation.state` | Five joint positions in degrees + gripper 0–100 |
| `action` | Six absolute targets in the same order and units |
| `timestamp`, `frame_index`, `episode_index` | Frame/episode timeline |
| `task_index` + task metadata | Task description |

Order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
The simulator takes radians internally; one shared adapter handles recording
and the inverse action conversion for rollout. Gripper 0–100 maps to −10°…100°.
Rollout saves the unnormalized model target before MuJoCo actuator clipping.
Camera names are genuinely **top/front**, not a renamed wrist camera.

Rerun shows synchronized video/state/action. Pause and seek to approach, grasp
and release, and compare a joint's measured state with its target. For a
headless machine export a recording instead of opening the viewer:

```bash
.venv/bin/python -m lerobot.scripts.lerobot_dataset_viz \
  --repo-id local/so101_tutorial --root data/so101_tutorial \
  --episode-index 0 --num-workers 0 --save 1 --output-dir outputs/rerun
```

## Upload to Hugging Face

Set `dataset.repo_id` to your own account/repo; keep `dataset.root` pointing to
the recorded dataset. Authenticate locally, then upload:

```bash
.venv/bin/hf auth login
bash scripts/07_upload_dataset.sh --config configs/local.json
```

This validates local data then calls `LeRobotDataset.push_to_hub` with videos,
parquet, metadata and a dataset card. It refuses `local/...` destinations.
No upload occurs during teleop/record/rollout. Uploading `trace.jsonl` alone
with `hf upload` is **not** a video-backed LeRobotDataset.

Example recorded policy dataset: https://huggingface.co/datasets/vasco281204/so101-rollout
(two episodes, 1800 frames; this is an optional example, not a dependency).

## Tests, scope and provenance

```bash
.venv/bin/python -m unittest discover -s tests -v
bash scripts/02_start_sim.sh --headless --seconds 1
```

CI checks a fresh dependency install, units/camera contracts, simulator render,
and a small record/read/video-decode integration test without physical hardware.
Model download/inference is checked separately locally; CI does not claim
hardware calibration or task success.

The training dataset declared by the checkpoint is
`ngocthuong2212/so101_mujoco_follower_7_31` (200 episodes / 60245 frames).
These totals match the parent project's Original manifest, but training metadata
does not contain the XML. Matching schema does not prove identical visual/task
distribution. Human demonstrations differ from policy rollouts; an episode
recorded now is not used for on-the-spot training. Sim success does not guarantee
success on physical hardware.

See [SOURCES.md](SOURCES.md), [VALIDATION.md](VALIDATION.md), [NOTICE.md](NOTICE.md)
and [LICENSE](LICENSE). This distribution includes only Original, not the parent
research project's Delay-Shift V1/V2 or its analysis scripts.
