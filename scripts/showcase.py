"""Offline recovery and evidence-contract demonstration using a saved local model."""

from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
from self_healing_embodied_agents.agents import (
    SelfHealingAgent,
    AgentConfig,
    AlwaysReplanAgent,
)
from self_healing_embodied_agents.env import TabletopManipulationEnv
from self_healing_embodied_agents.evidence import validate_episode
from self_healing_embodied_agents.recovery import RecoveryMemory
from self_healing_embodied_agents.training import load_bundle
from self_healing_embodied_agents.types import Action, ActionKind


def demonstrate(model_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    bundle = load_bundle(model_path)
    cases = []
    records = []

    def record(name, passed):
        if not passed:
            raise AssertionError(name)
        cases.append({"case": name, "status": "PASS"})

    learned = SelfHealingAgent(bundle).run_episode(
        TabletopManipulationEnv(seed=17, perturbation="compound_slip_block")
    )
    basic = SelfHealingAgent(detector="postcondition", recovery="nominal").run_episode(
        TabletopManipulationEnv(seed=17, perturbation="compound_slip_block")
    )
    records.extend({**r.as_dict(), "event_log": r.event_log} for r in (learned, basic))
    record("learned_agent_recovers_in_symbolic_fault_case", learned.success)
    record("simple_control_also_recovers", basic.success)
    memory = RecoveryMemory()
    SelfHealingAgent(bundle, memory=memory).run_episode(
        TabletopManipulationEnv(seed=17, perturbation="grasp_slip")
    )
    before = deepcopy(memory)
    frozen = SelfHealingAgent(bundle, memory=memory, update_memory=False).run_episode(
        TabletopManipulationEnv(seed=23, perturbation="grasp_slip")
    )
    record("frozen_evaluation_preserves_memory", memory == before)
    short = AlwaysReplanAgent(config=AgentConfig(max_steps=2)).run_episode(
        TabletopManipulationEnv(seed=17)
    )
    record(
        "insufficient_budget_does_not_claim_completion",
        not short.success and short.steps == 2,
    )
    false = deepcopy(records[0])
    false["success"] = 0
    try:
        validate_episode(false, complete=True)
    except ValueError:
        rejected = True
    else:
        rejected = False
    record("contradictory_success_is_rejected", rejected)
    env = TabletopManipulationEnv(perturbation="transient_occlusion")
    env.reset()
    result = env.step(Action(ActionKind.MOVE_TO_OBJECT))
    result.events[0]["failure"] = False
    record(
        "returned_events_cannot_rewrite_ground_truth", env.events[0]["failure"] is True
    )
    records.extend({**r.as_dict(), "event_log": r.event_log} for r in (frozen, short))
    (output / "episodes.json").write_text(
        json.dumps(records, indent=2, allow_nan=False) + "\n"
    )
    report = {
        "status": "PASS",
        "passed": len(cases),
        "cases": cases,
        "scope": "Synthetic simulator demonstration, not physical-robot testing or evidence of learned superiority.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(demonstrate(args.model, args.out), indent=2))
