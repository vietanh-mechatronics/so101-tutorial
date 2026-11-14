"""Checks semantic boundaries: units, camera interface and timestep pairing."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('tutorial', Path(__file__).resolve().parents[1] / 'tools/tutorial.py')
tutorial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tutorial)
from sim_mujoco.baseline.sim_adapter import SimAdapterConfig, SimDatasetAdapter


class ContractTests(unittest.TestCase):
    def setUp(self):
        sim = tutorial.read_json(tutorial.ROOT / 'configs/simulation/original.json')
        self.adapter = SimDatasetAdapter(SimAdapterConfig.from_dict(sim['adapter']))
        self.obs = {'agent_pos': np.array([0.1, -.2, .3, -.4, .5, .6], dtype=np.float32),
                    'pixels': {n: np.zeros((480, 640, 3), dtype=np.uint8) for n in ('top', 'front')}}
        self.adapter.prepare(self.obs, np.zeros(6, dtype=np.float32))

    def test_recording_to_rollout_unit_roundtrip(self):
        for gripper in (np.deg2rad(-10), 0, np.deg2rad(45), np.deg2rad(100)):
            radians = np.array([.2, -.5, .7, -1.1, 1.2, gripper], dtype=np.float32)
            recorded = self.adapter.action_to_values(radians)
            restored = tutorial.policy_to_sim([recorded[n] for n in self.adapter.action_names], self.adapter)
            np.testing.assert_allclose(restored, radians, atol=1e-6)

    def test_gripper_saturates_but_joint_targets_keep_degrees(self):
        result = tutorial.policy_to_sim([90, 0, 0, 0, 0, 120], self.adapter)
        self.assertAlmostEqual(result[0], np.pi / 2, places=6)
        self.assertAlmostEqual(result[5], np.deg2rad(100), places=6)
        with self.assertRaises(ValueError):
            tutorial.policy_to_sim([float('nan')] * 6, self.adapter)

    def test_model_input_is_unbatched_rgb_in_zero_one_range(self):
        self.obs['pixels']['top'][:] = 255
        frame = tutorial.policy_observation(self.obs, self.adapter)
        self.assertEqual(tuple(frame['observation.images.top'].shape), (3, 480, 640))
        self.assertEqual(float(frame['observation.images.top'].max()), 1)
        state = self.adapter.observation_to_values(self.obs)
        np.testing.assert_allclose(frame['observation.state'].numpy(), [state[n] for n in self.adapter.state_names])

    def test_rejects_wrong_camera_even_with_same_shape(self):
        config = {'type': 'act', 'input_features': {
            'observation.state': {'shape': [6]},
            'observation.images.front': {'shape': [3, 480, 640]},
            'observation.images.wrist': {'shape': [3, 480, 640]}},
            'output_features': {'action': {'shape': [6]}}}
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            tutorial.validate_policy(config, self.obs, self.adapter)


if __name__ == '__main__':
    unittest.main()
