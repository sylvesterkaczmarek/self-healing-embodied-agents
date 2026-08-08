"""Integration boundary for a future ManiSkill-backed environment.

The core benchmark deliberately has no ManiSkill dependency. A concrete adapter should
map ManiSkill observations into WorldState and policy outputs into Action or a richer
action representation while preserving the detector/recovery interfaces.
"""

from __future__ import annotations

from typing import Protocol

from self_healing_embodied_agents.types import Action, WorldState


class EmbodiedEnvironmentAdapter(Protocol):
    def reset(self) -> WorldState: ...
    def step(self, action: Action): ...
