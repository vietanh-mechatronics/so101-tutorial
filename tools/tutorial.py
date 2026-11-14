#!/usr/bin/env python3
"""Beginner commands over the project's existing MuJoCo implementation.

No environment or hardware implementation is copied from LeRobot. All simulator
commands use MujocoXmlEnv + SimDatasetAdapter, including the same gripper map.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HOME = Path(__file__).resolve().parents[1]
ROOT = HOME if (HOME / 'src/sim_mujoco/baseline').is_dir() else HOME.parent
for source in (ROOT / 'src/sim_mujoco/baseline', ROOT / 'src', ROOT / 'lerobot/src'):
    sys.path.insert(0, str(source))
os.environ['PYTHONPATH'] = os.pathsep.join(sys.path[:3] + [os.environ.get('PYTHONPATH', '')])
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', os.environ['MUJOCO_GL'])
os.environ.setdefault('HF_DATASETS_CACHE', str(HOME / 'outputs/hf_datasets_cache'))


def path(value: str) -> Path:
    result = Path(value).expanduser()
    if result.is_absolute():
        return result
    # Accept commands copied from the original parent project in the standalone repo.
    if ROOT == HOME and result.parts and result.parts[0] == 'so101-tutorial':
        result = Path(*result.parts[1:])
    return ROOT / result


def read_json(filename: Path) -> dict:
    return json.loads(filename.read_text())


def load_settings(args: argparse.Namespace) -> tuple[dict, dict]:
    cfg = read_json(path(args.config))
    sim = read_json(path(cfg['simulation_config']))
    if cfg['fps'] != 30:
        raise ValueError('This tutorial/checkpoint uses 30 FPS; keep fps=30.')
    if not math.isfinite(cfg['episode_seconds']) or cfg['episode_seconds'] <= 0 or cfg['num_episodes'] < 1:
        raise ValueError('episode_seconds must be finite and >0; num_episodes must be >=1.')
    # Existing scene/config assets resolve relative to the repository root.
    os.chdir(ROOT)
    return cfg, sim


def make_environment(cfg: dict, sim: dict):
    import numpy as np
    from sim_mujoco.baseline.mujoco_xml_env import MujocoXmlEnv, MujocoXmlEnvConfig
    from sim_mujoco.baseline.sim_adapter import SimAdapterConfig, SimDatasetAdapter

    env = MujocoXmlEnv(MujocoXmlEnvConfig.from_dict(sim['env']))
    try:
        obs, info = env.reset(seed=cfg['seed'])
        adapter = SimDatasetAdapter(SimAdapterConfig.from_dict(sim['adapter']))
        adapter.prepare(obs, np.zeros(env.action_space.shape, dtype=np.float32))
        if tuple(adapter.action_names) != tuple(env.action_names):
            raise ValueError('Adapter action order does not match XML actuator order.')
        if adapter.state_names != adapter.action_names or len(adapter.action_names) != 6:
            raise ValueError('Tutorial requires the same six state/action joints.')
        return env, adapter, obs, info
    except BaseException:
        env.close()
        raise


def policy_to_sim(values, adapter):
    """Invert the recording adapter: degrees -> rad; gripper 0..100 -> rad."""
    import numpy as np
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(adapter.action_names),) or not np.isfinite(values).all():
        raise ValueError(f'Invalid model action: {values}')
    result = values.copy()
    for i, name in enumerate(adapter.action_names):
        transform = adapter.config.feature_value_transforms.get(name)
        if transform:
            value = np.clip(values[i], transform.output_min, transform.output_max)
            ratio = (value - transform.output_min) / (transform.output_max - transform.output_min)
            result[i] = transform.input_min + ratio * (transform.input_max - transform.input_min)
        else:
            result[i] /= adapter.angle_scale
    return result


def policy_observation(obs, adapter):
    import numpy as np
    import torch
    values = adapter.observation_to_values(obs)
    # Unbatched: the saved to_batch_processor adds the batch dimension once.
    batch = {'observation.state': torch.tensor([values[n] for n in adapter.state_names], dtype=torch.float32)}
    for name in adapter.image_names:
        image = np.ascontiguousarray(values[name])
        batch[f'observation.images.{name}'] = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return batch


def validate_policy(model_config: dict, obs, adapter) -> None:
    inputs = policy_observation(obs, adapter)
    expected = model_config['input_features']
    if set(inputs) != set(expected):
        raise ValueError(f'Camera/state mismatch: sim={list(inputs)}, model={list(expected)}')
    for name, tensor in inputs.items():
        if list(tensor.shape) != expected[name]['shape']:
            raise ValueError(f'{name}: sim shape {list(tensor.shape)} != model {expected[name]["shape"]}')
    if model_config['type'] != 'act' or model_config['output_features']['action']['shape'] != [6]:
        raise ValueError('This rollout supports ACT with six absolute joint targets.')


def leader_config(cfg: dict) -> None:
    leader = cfg['leader']
    if not Path(leader['port']).exists():
        raise ValueError(f'Leader port not found: {leader["port"]}. Set leader.port in your config.')
    os.environ.update(REAL_TELEOP_PORT=leader['port'], REAL_TELEOP_ID=leader['id'],
                      REAL_TELEOP_USE_DEGREES='true',
                      REAL_TELEOP_CALIBRATION_DIR=str(path(leader['calibration_dir'])))


def command_check(cfg: dict, sim: dict, args) -> None:
    print('Python:', sys.executable, sys.version.split()[0])
    failures = []
    for name in ('lerobot', 'torch', 'mujoco', 'gymnasium', 'huggingface_hub', 'av', 'cv2'):
        try:
            module = importlib.import_module(name)
            print(name, getattr(module, '__version__', ''), getattr(module, '__file__', ''))
            if name == 'lerobot' and not getattr(module, '__file__', None):
                raise ImportError('LeRobot namespace found, but package is not installed.')
        except Exception as exc:
            failures.append(name)
            print('MISSING/BROKEN:', name, str(exc))
    if failures:
        raise RuntimeError(f'Install dependencies first: {failures}')
    import numpy as np
    env, adapter, obs, _ = make_environment(cfg, sim)
    try:
        hold = np.array([adapter.observation_to_values(obs)[n] for n in adapter.action_names])
        for _ in range(3):
            obs, *_ = env.step(policy_to_sim(hold, adapter))
        print('Scene:', sim['env']['xml_path'])
        print('Cameras:', {n: list(v.shape) for n, v in obs['pixels'].items()})
        print('Joint order:', adapter.action_names)
        print('Physics seconds/control step:', env.model.opt.timestep * env.cfg.frame_skip)
        print('Render and physics: OK. This does not certify leader or policy task success.')
    finally:
        env.close()
    print('Leader port:', cfg['leader']['port'], 'exists=', Path(cfg['leader']['port']).exists())
    if args.with_leader:
        leader_config(cfg)
        from sim_mujoco.baseline.real_leader_script import real_leader_action, close_real_leader
        try:
            print('Leader configured; checking raw connection on teleop is recommended.')
            # The closed simulator still provides its action schema; no physics is used here.
            print('Leader target (radians):', real_leader_action(None, None, 0, 0, env))
        finally:
            close_real_leader()


def command_record(cfg: dict, sim: dict, args) -> None:
    leader_config(cfg)
    sim['dataset'].update(cfg['dataset'])
    sim['dataset'].update(root=str(path(cfg['dataset']['root'])), fps=cfg['fps'],
                          num_episodes=cfg['num_episodes'], episode_time_s=cfg['episode_seconds'],
                          push_to_hub=False, auto_resume_if_root_exists=False)
    sim['recording'].update(control_fps=cfg['fps'], top_grid={'enabled': False})
    sim['env']['seed'] = cfg['seed']
    sim['action_source'] = {'type': 'scripted',
                            'scripted_callable': 'sim_mujoco.baseline.real_leader_script:real_leader_action'}
    out = HOME / 'outputs/configs'
    out.mkdir(parents=True, exist_ok=True)
    filename = out / f'record_{time.time_ns()}.json'
    filename.write_text(json.dumps(sim, indent=2))
    command = [sys.executable, '-m', 'sim_mujoco.baseline.collect_sim_dataset', '--config', str(filename), '--no-push']
    if args.resume:
        command.append('--resume')
    subprocess.run(command, check=True)


def local_dataset(cfg: dict):
    from lerobot.datasets import LeRobotDataset
    root = path(cfg['dataset']['root'])
    if not (root / 'meta/info.json').is_file():
        raise FileNotFoundError(f'Local dataset not found: {root}. Record an episode first.')
    return LeRobotDataset(cfg['dataset']['repo_id'], root=root, video_backend='pyav')


def command_inspect(cfg: dict, full_video: bool = False) -> None:
    import numpy as np
    dataset = local_dataset(cfg)
    if len(dataset) == 0:
        raise ValueError('Dataset has no frames.')
    # Check tabular data without decoding all video frames.
    df = dataset.hf_dataset
    for name in ('observation.state', 'action', 'timestamp', 'episode_index', 'frame_index', 'task_index'):
        if name not in dataset.features:
            raise ValueError(f'Missing feature: {name}')
    for camera in ('top', 'front'):
        if f'observation.images.{camera}' not in dataset.features:
            raise ValueError(f'Missing camera: {camera}')
    stamps = np.asarray(df['timestamp']).reshape(-1)
    episodes = np.asarray(df['episode_index']).reshape(-1)
    for episode in np.unique(episodes):
        times = stamps[episodes == episode]
        if len(times) > 1 and not np.allclose(np.diff(times), 1 / dataset.fps, atol=1e-4):
            raise ValueError(f'Non-uniform timestamps in episode {episode}')
        print(f'Episode {episode}: {len(times)} frames, duration ~= {len(times)/dataset.fps:.2f}s')
    for name in ('observation.state', 'action'):
        values = np.asarray(df[name])
        if values.shape != (len(dataset), 6) or not np.isfinite(values).all():
            raise ValueError(f'Invalid {name}: shape={values.shape}')
    frame = dataset[0]  # Also prove both video streams can be decoded.
    print('First frame:', {k: list(v.shape) for k, v in frame.items() if hasattr(v, 'shape')})
    print('Local metadata, timestamps, six-channel vectors and first camera frames: OK')
    if full_video:
        import av
        for camera in ('top', 'front'):
            count = 0
            for filename in sorted((dataset.root / 'videos' / f'observation.images.{camera}').rglob('*.mp4')):
                with av.open(str(filename)) as container:
                    for video_frame in container.decode(video=0):
                        if (video_frame.height, video_frame.width) != (480, 640):
                            raise ValueError(f'Unexpected video size in {filename}')
                        count += 1
            if count != len(dataset):
                raise ValueError(f'{camera}: decoded {count} frames, expected {len(dataset)}')
            print(f'Full video decode {camera}: {count} frames OK')


def model_directory(cfg: dict) -> Path:
    directory = path(cfg['policy']['local_path'])
    for filename in ('config.json', 'model.safetensors', 'policy_preprocessor.json', 'policy_postprocessor.json'):
        if not (directory / filename).is_file():
            raise FileNotFoundError(f'Missing {directory / filename}. Run 08_download_model.sh first.')
    for name in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        for step in read_json(directory / name)['steps']:
            if step.get('state_file') and not (directory / step['state_file']).is_file():
                raise FileNotFoundError(f'Missing saved normalization: {step["state_file"]}')
    return directory


def load_policy(cfg: dict, obs, adapter):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    directory = model_directory(cfg)
    raw = read_json(directory / 'config.json')
    validate_policy(raw, obs, adapter)
    device = cfg['policy']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy_cfg = PreTrainedConfig.from_pretrained(str(directory), cli_overrides=[f'--device={device}'])
    # We load full trained weights, so no ImageNet download is needed at construction.
    policy_cfg.pretrained_backbone_weights = None
    policy = get_policy_class(policy_cfg.type).from_pretrained(str(directory), config=policy_cfg)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(directory),
        preprocessor_overrides={'device_processor': {'device': device}})
    print('Policy:', cfg['policy']['repo_id'], 'device:', device)
    return policy, pre, post


def command_sim(cfg: dict, sim: dict, args) -> None:
    import numpy as np
    env, adapter, obs, info = make_environment(cfg, sim)
    close_leader = None
    log_file = None
    dataset = None
    cv2 = None
    try:
        if args.command == 'teleop':
            leader_config(cfg)
            from sim_mujoco.baseline.real_leader_script import real_leader_action, close_real_leader
            close_leader = close_real_leader
        if args.command == 'rollout':
            policy, pre, post = load_policy(cfg, obs, adapter)
            import torch
            run_dir = HOME / 'outputs' / f'rollout_{time.time_ns()}'
            run_dir.mkdir(parents=True)
            (run_dir / 'config.json').write_text(json.dumps({'tutorial': cfg, 'simulation': sim}, indent=2))
            log_file = (run_dir / 'trace.jsonl').open('w')
            print('Trace:', run_dir)
            if args.save_dataset:
                from lerobot.datasets import LeRobotDataset
                from lerobot.utils.feature_utils import build_dataset_frame
                dataset = LeRobotDataset.create(
                    repo_id=cfg['dataset']['repo_id'], root=path(cfg['dataset']['root']),
                    fps=cfg['fps'], robot_type=sim['dataset']['robot_type'],
                    features=adapter.build_dataset_features(use_videos=True),
                    use_videos=True, video_backend='pyav', streaming_encoding=True,
                    encoder_threads=2, encoder_queue_maxsize=30)
                (dataset.root / 'rollout_provenance.json').write_text(
                    json.dumps({'tutorial': cfg, 'simulation': sim,
                                'source': 'learned_policy', 'trace_directory': str(run_dir)}, indent=2))
                print('LeRobotDataset:', dataset.root)
        if not args.headless:
            import cv2
        episode = 0
        while episode < (args.episodes or cfg['num_episodes']):
            restart = False
            obs, info = env.reset(seed=cfg['seed'] + episode, options={'episode_index': episode})
            if args.command == 'rollout':
                policy.reset()
                pre.reset()
                post.reset()
            hold = obs['agent_pos'].copy()
            seconds = args.seconds if args.seconds is not None else cfg['episode_seconds']
            steps = max(1, round(seconds * cfg['fps']))
            for step in range(steps):
                started = time.perf_counter()
                if args.command == 'teleop':
                    action = real_leader_action(obs, info, step, episode, env)
                elif args.command == 'rollout':
                    with torch.inference_mode():
                        predicted = post(policy.select_action(pre(policy_observation(obs, adapter))))
                    targets = predicted.detach().cpu().numpy().reshape(-1)
                    action = policy_to_sim(targets, adapter)
                    state_values = adapter.observation_to_values(obs)
                    log_file.write(json.dumps({'episode': episode, 'frame': step,
                        'timestamp': step / cfg['fps'],
                        'state': [state_values[n] for n in adapter.state_names],
                        'action': targets.tolist(), 'sim_action_rad': action.tolist()}) + '\n')
                else:
                    action = hold
                if cv2 is not None:
                    images = np.concatenate([obs['pixels'][n] for n in adapter.image_names], axis=1)
                    cv2.imshow('SO101 | top / front | Q: stop, R: reset', cv2.cvtColor(images, cv2.COLOR_RGB2BGR))
                    key = cv2.waitKey(1) & 0xff
                    if key in (27, ord('q')):
                        return
                    if key == ord('r'):
                        restart = True
                        if log_file:
                            log_file.write(json.dumps({'event': 'reset', 'episode': episode}) + '\n')
                        break
                if dataset is not None:
                    frame = build_dataset_frame(dataset.features, state_values, prefix='observation')
                    # Store the model's unnormalized target, before simulator clipping.
                    frame['action'] = targets.astype(np.float32)
                    frame['task'] = sim['dataset']['single_task']
                    dataset.add_frame(frame)
                obs, reward, terminated, truncated, info = env.step(action)
                elapsed = time.perf_counter() - started
                if step == 0 or elapsed > 2 / cfg['fps'] and step % 30 == 0:
                    print(f'episode={episode} frame={step} compute={elapsed*1000:.1f}ms sim_time={env.data.time:.3f}')
                if not args.no_sleep:
                    time.sleep(max(0, 1 / cfg['fps'] - elapsed))
                if terminated or truncated:
                    break
            if not restart:
                if dataset is not None:
                    dataset.save_episode()
                    print(f'Saved rollout episode {episode}, frames={step + 1}', flush=True)
                episode += 1
            elif dataset is not None:
                dataset.clear_episode_buffer()
        print('Completed. Environment reward is currently always zero; assess pick/place visually.')
    finally:
        if dataset is not None:
            try:
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer()
            finally:
                dataset.finalize()
        if log_file:
            log_file.close()
        if close_leader:
            close_leader()
        if cv2 is not None:
            cv2.destroyAllWindows()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'sim', 'teleop', 'record', 'visualize', 'inspect', 'download', 'rollout', 'upload'])
    parser.add_argument('--config', default=str(HOME / 'configs/tutorial.json'))
    parser.add_argument('--headless', action='store_true', help='Disable camera preview windows')
    parser.add_argument('--no-sleep', action='store_true', help='Run physics/inference without wall-clock pacing')
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--episode-index', type=int, default=0)
    parser.add_argument('--with-leader', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save-dataset', action='store_true', help='Record rollout cameras/state/action as LeRobotDataset')
    parser.add_argument('--full-video', action='store_true', help='Decode every video frame during inspect')
    args = parser.parse_args()
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error('--seconds must be finite and >0')
    if args.episodes is not None and args.episodes < 1:
        parser.error('--episodes must be >=1')
    cfg, sim = load_settings(args)
    if args.command == 'check':
        command_check(cfg, sim, args)
    elif args.command in ('sim', 'teleop', 'rollout'):
        command_sim(cfg, sim, args)
    elif args.command == 'record':
        command_record(cfg, sim, args)
    elif args.command == 'inspect':
        command_inspect(cfg, full_video=args.full_video)
    elif args.command == 'download':
        from huggingface_hub import snapshot_download
        directory = path(cfg['policy']['local_path'])
        snapshot_download(cfg['policy']['repo_id'], revision=cfg['policy']['revision'], local_dir=directory)
        model_directory(cfg)
        print('Downloaded checkpoint and processors:', directory)
    elif args.command == 'visualize':
        if not (path(cfg['dataset']['root']) / 'meta/info.json').is_file():
            raise FileNotFoundError('Record local data first.')
        subprocess.run([sys.executable, '-m', 'lerobot.scripts.lerobot_dataset_viz',
            '--repo-id', cfg['dataset']['repo_id'], '--root', str(path(cfg['dataset']['root'])),
            '--episode-index', str(args.episode_index), '--mode', 'local', '--num-workers', '0'], check=True)
    elif args.command == 'upload':
        repo_id = cfg['dataset']['repo_id']
        if repo_id.startswith('local/') or '/' not in repo_id:
            raise ValueError('Set dataset.repo_id to your Hugging Face account/name before upload.')
        command_inspect(cfg)
        dataset = local_dataset(cfg)
        provenance = dataset.root / 'rollout_provenance.json'
        description = 'SO-101 demonstrations recorded in the act_policy_sim2real MuJoCo environment.'
        if provenance.is_file():
            recorded = read_json(provenance)
            model = recorded['tutorial']['policy']
            description = (
                f"Learned-policy rollout of `{model['repo_id']}` at revision `{model['revision']}` "
                f"in the act_policy_sim2real MuJoCo environment. "
                f"{dataset.num_episodes} episodes, {len(dataset)} frames, {dataset.fps} FPS. "
                "Includes top/front RGB videos, measured observation.state and predicted action "
                "before simulator clipping. The five arm joints are in degrees; gripper is 0–100. "
                "Observation is recorded before applying the paired action. "
                "Episode duration is on the dataset timeline; computation may run slower than real time. "
                "These are policy predictions, not human demonstrations or action replay. "
                "The simulator does not provide a task-success detector; no success rate is claimed."
            )
        dataset.push_to_hub(private=cfg['dataset']['private'], tags=['so101', 'mujoco', 'tutorial'],
                            dataset_description=description)
        print(f'https://huggingface.co/datasets/{repo_id}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped; simulator/leader cleanup runs on exit.', file=sys.stderr)
        sys.exit(130)
