"""The one polyloop Environment for voice: a verifier produces judged conversations against a target that
serves the policy; the ledger turns them into rewards for the controller and hints for OPSD.

    task          = whatever the verifier calls a task (Coval test case, Pipecat scenario)
    K episodes    = verifier.run(k=K)
    reward        = Judged.reward (mean of the verifier's metrics), None = unscored
    trace session = Judged.session, the id the proxy keyed the conversation by
    hindsight     = Judged.explanation, joined to traces by session id

Roles: with a second target (`candidate`), the candidate is scored on its own endpoint while production
traffic stays on the live adapter; with one target, incumbent and candidate are scored one after the other.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from polyloop.environment import BaseEnvironment, Policy
from polyloop.harness.rollout import TaskResult

from polyvoice.core.ledger import Ledger
from polyvoice.core.types import Judged, Target, Task, Verifier


class VoiceEnvironment(BaseEnvironment):
    name = "voice"
    needs_engine_warm = True   # the proxy samples from the trainer's engines

    def __init__(self, cfg, store, log=print, *, verifier: Verifier, targets: dict[str, Target],
                 pass_value: float = 1.0, hint_failures_only: bool = True, ledger: str = "ledger.jsonl",
                 temperature: float | None = None, require_public: bool = False, **options):
        super().__init__(cfg, store, log=log, **options)
        self.verifier, self.targets = verifier, targets
        self.pass_value, self.hint_failures_only = pass_value, hint_failures_only
        self.ledger = Ledger(store.root / ledger)
        self.temperature = cfg.gate.temperature if temperature is None else temperature
        self.require_public = require_public

    @property
    def ledger_path(self) -> Path:
        return self.ledger.path

    # ---- roles --------------------------------------------------------------
    def _role(self, policy: Policy) -> str:
        inc = (self.store.incumbent() or {}).get("id", "base")
        pid = (policy or {}).get("id", "base")
        return "candidate" if "candidate" in self.targets and pid not in (inc, "base", "adhoc") else "default"

    # ---- Environment --------------------------------------------------------
    def load_tasks(self, dataset: str, *, limit: int | None = None, seed: int = 0, names: list[str] | None = None):
        return self.verifier.load_tasks(dataset, limit=limit, seed=seed, names=names)

    def preflight(self) -> list[str]:
        problems = self.verifier.preflight([self.cfg.tasks, self.cfg.gate.holdout])
        if abs(self.temperature - self.cfg.gate.temperature) > 1e-9:
            problems.append(f"{self.verifier.name} temperature {self.temperature} != gate.temperature {self.cfg.gate.temperature}")
        for role, t in self.targets.items():
            problems += [f"{role}: {p}" for p in t.check(require_public=self.require_public)]
        return problems

    def run_rollouts(self, *, label: str, tasks: list[Task], policy: Policy, k: int, temperature: float,
                     out: Path | None, on_result: Callable[[TaskResult], None]) -> list[TaskResult]:
        if not tasks:
            return []
        role = self._role(policy)
        target = self.targets[role]
        pid = (policy or {}).get("id", "base")
        results = {t.task_name: TaskResult(task=t.task_name) for t in tasks}
        grouped: dict[str, dict] = {}
        target.serve(policy)
        self.log(f"{self.verifier.name}: {role} target serves {pid} ({(policy or {}).get('sampler_path') or 'base model'})")
        judged: list[Judged] = []
        try:
            judged = self.verifier.run(label=label, tasks=tasks, k=k, policy_id=pid, target=target, out=out, role=role)
            for j in judged:
                target.finish_session(j.session)   # flush the last turn while the override is still in place
        finally:
            target.clear()
        mirror = (out / "rewards.jsonl") if out else None
        for j in judged:
            self.ledger.append(j, label=label, policy_id=pid, role=role, source=self.verifier.name, mirror=mirror)
            g = grouped.setdefault(j.task, {"rewards": [], "failed": 0, "unscored": 0, "seconds": 0.0})
            g["seconds"] += j.seconds
            if j.status != "COMPLETED":
                g["failed"] += 1
            elif j.reward is None:
                g["unscored"] += 1
            else:
                g["rewards"].append(j.reward)
        for t in tasks:
            res, g = results[t.task_name], grouped.get(t.task_name)
            if not g or not g["rewards"]:
                res.error = f"no scored simulations (failed={g['failed'] if g else 0}, unscored={g['unscored'] if g else 0})"
            else:
                n = len(g["rewards"])
                res.rewards, res.turns, res.tokens, res.stop_reasons = g["rewards"], [0] * n, [0] * n, [None] * n
                res.seconds = g["seconds"] / n
            on_result(res)
        return [results[t.task_name] for t in tasks]

    def session_hints(self) -> dict[str, str]:
        return self.ledger.session_hints(pass_value=self.pass_value, failures_only=self.hint_failures_only)

    def excluded_sessions(self) -> set[str]:
        return self.ledger.excluded_sessions()
