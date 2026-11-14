# Sources

The bundled simulator, Original XML/meshes, placement table and dataset contract
are adapted from `vietanh-mechatronics/act_policy_sim2real`.
See `SOURCES.json` for the parent commit, original paths and both upstream/release SHA-256 hashes. Text line endings/trailing whitespace
are normalized; obsolete machine-specific XML comments are removed. Physics
parameters, geometry and camera settings are unchanged.

`tools/tutorial.py`, launchers, examples, tests and documentation package the
original workflow for beginners. LeRobot itself is installed from the exact Git
revision recorded in pyproject.toml and uv.lock. The policy is downloaded only
by the explicit download command at its pinned Hugging Face revision.

No Delay-Shift simulator, personal hardware calibration, trained weights,
datasets, videos or credentials are included. Original licenses and attribution
are kept under LICENSE, NOTICE.md and assets/mujoco/so101/.
