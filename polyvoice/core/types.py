"""Shapes shared by every verifier and target."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Judged:
    """One judged conversation, whatever produced it."""
    task: str                       # task name (test case id, scenario name)
    session: str                    # id the proxy/observer keyed the traces by (join key)
    reward: float | None            # None = unscored (never a training signal)
    explanation: str | None = None  # judge text → hindsight hint
    status: str = "COMPLETED"       # vendor status; anything but COMPLETED counts as failed
    seconds: float = 0.0
    aliases: list[str] = field(default_factory=list)   # other ids the same session is known by
    extra: dict = field(default_factory=dict)          # vendor payload worth keeping (run id, messages, ...)


@dataclass
class Task:
    """Minimal task object the controller can see (`.task_name`); the verifier owns the rest."""
    task_name: str
    payload: Any = None


@runtime_checkable
class Target(Protocol):
    """Where the bot under test gets its LLM answers, and how a policy is bound to it."""
    proxy_url: str | None
    public_url: str | None

    def serve(self, policy: dict | None) -> None: ...
    def clear(self) -> None: ...
    def finish_session(self, session: str) -> None: ...
    def check(self) -> list[str]: ...


@runtime_checkable
class Verifier(Protocol):
    """Where judged conversations come from."""
    name: str

    def load_tasks(self, dataset: str, *, limit: int | None = None, seed: int = 0,
                   names: list[str] | None = None) -> list[Task]: ...
    def preflight(self, datasets: list[str]) -> list[str]: ...
    def run(self, *, label: str, tasks: list[Task], k: int, policy_id: str, target: Target,
            out: Any | None) -> list[Judged]: ...
