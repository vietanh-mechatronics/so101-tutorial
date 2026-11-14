# Attribution and scope

This tutorial adapts the MuJoCo integration from
https://github.com/vietanh-mechatronics/act_policy_sim2real at commit
`af8226f` (Apache-2.0). The original scene geometry, joint dynamics, cameras,
placement rules and unit conversions are preserved. Text formatting is normalized and obsolete XML comments removed. This release includes only
the Original simulator; the research project's Delay-Shift V1/V2 are not included.

LeRobot is an external dependency pinned to commit
`fe1c54594f6ba275fccf210b9f7b11f0e2324b89` of
https://github.com/huggingface/lerobot (Apache-2.0). It is not vendored.
The upstream license headers in copied source files are preserved.

MuJoCo assets retain their upstream attribution and license:
`assets/mujoco/so101/LICENSE.upstream`, `README.upstream.md`, `CITATION.cff`.

No robot calibration, dataset, model weights, authentication tokens or local
Python environment are distributed in this repository. Model and dataset
references are documented in README.md; downloads are separate explicit steps.
