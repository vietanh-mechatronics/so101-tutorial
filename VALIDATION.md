# Validation — 2026-11-14

Tested on Linux x86-64 using a **fresh virtual environment created by
`uv sync --frozen`**, not the developer's Conda environment. The lockfile pins
LeRobot 0.7.1 to Git revision fe1c54594f6ba275fccf210b9f7b11f0e2324b89 and
PyTorch 2.11.0 CPU wheels.

Passed locally:

- Four unit tests: joint/gripper unit roundtrip, saturation/nonfinite action
  rejection, RGB tensor layout/range, and wrong-camera interface rejection.
- Integration test: load the bundled Original XML/meshes, render top/front,
  collect three frames via the real LeRobot writer, finalize, read back state/
  action/timestamps and fully decode both videos.
- Real ACT inference in the standalone environment using the pinned SIM_ONLY
  checkpoint: two short episodes, with LeRobotDataset video/state/action writing.
- Check setup, full video decode of the saved rollout, Python compile and shell
  syntax checks.

The earlier parent-project run recorded two 30-second episodes (1800 frames)
and uploaded them to https://huggingface.co/datasets/vasco281204/so101-rollout.
All recorded state/action pairs matched the policy trace; both camera videos
were fully decoded and remote video/parquet hashes verified. The standalone
packaging test is shorter and does not claim to repeat that full-duration run.

Not verified:

- Hardware leader connection/calibration/USB latency: no leader was connected.
- Pick/place task success or sim-to-real generalization: no automatic success
  detector is supplied by this simulator.
- Interactive OpenCV/Rerun desktop UI in a clean environment. Headless tests
  verify data/video only; the README documents the GUI OpenCV wheel conflict.
- Non-Linux platforms and NVIDIA CUDA setup. Default installation is CPU-only.
- GitHub-hosted CI: a workflow is included, but its remote result is only known
  after the repository is published and GitHub runs it.
