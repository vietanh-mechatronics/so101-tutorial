"""Hardware-free integration: real scene + LeRobot writer/readback + videos."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('integration_tutorial', ROOT / 'tools/tutorial.py')
tutorial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tutorial)


class IntegrationTest(unittest.TestCase):
    def test_record_and_decode_real_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sim = tutorial.read_json(ROOT / 'configs/simulation/original.json')
            sim['dataset'].update(root=str(tmp / 'dataset'), repo_id='local/integration',
                                  num_episodes=1, episode_time_s=.1, push_to_hub=False,
                                  streaming_encoding=True)
            sim['action_source'] = {'type': 'constant', 'constant': [0.] * 6}
            sim['recording'].update(prepare_time_s=0, rest_time_s=0,
                                    display_camera_preview=False, teleop_during_prepare=False,
                                    teleop_during_rest=False, require_enter_before_episode=False)
            filename = tmp / 'collect.json'
            filename.write_text(json.dumps(sim))
            subprocess.run([sys.executable, '-m', 'sim_mujoco.baseline.collect_sim_dataset',
                            '--config', str(filename), '--no-push'],
                           check=True, cwd=ROOT, env=os.environ.copy(), timeout=120)
            cfg = tutorial.read_json(ROOT / 'configs/tutorial.json')
            cfg['dataset'].update(root=str(tmp / 'dataset'), repo_id='local/integration')
            tutorial.command_inspect(cfg, full_video=True)
            dataset = tutorial.local_dataset(cfg)
            self.assertEqual(dataset.num_episodes, 1)
            self.assertEqual(len(dataset), 3)
            self.assertEqual(tuple(dataset[2]['observation.state'].shape), (6,))


if __name__ == '__main__':
    unittest.main()
