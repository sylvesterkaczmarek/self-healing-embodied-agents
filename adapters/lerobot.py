"""Interface notes for exposing a recovery-aware policy through LeRobot.

No LeRobot dependency is required for the default benchmark. This file defines the
minimal conceptual boundary used by downstream integrations.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class PolicyAdapter(Protocol):
    def select_action(self, observation: dict[str, np.ndarray]) -> np.ndarray: ...
