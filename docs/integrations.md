# Integration path

The default benchmark intentionally avoids heavyweight robotics dependencies so that all checked-in results can be reproduced on CPU.

Two extension boundaries are included:

- `adapters/maniskill.py` for mapping a richer manipulation simulator into the benchmark state/action interface
- `adapters/lerobot.py` for exposing recovery-aware policies through a broader robot policy stack

A higher-fidelity follow-up should preserve the same experimental protocol while replacing the toy state vector with image/proprioceptive observations and the discrete skills with continuous robot actions.

Relevant upstream documentation:

- ManiSkill: https://maniskill.readthedocs.io/en/latest/
- LeRobot: https://huggingface.co/docs/lerobot/index
