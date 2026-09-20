"""The per-session ledger: every judged conversation, keyed by the session id the traces carry.

Hints (for OPSD rows) come from scored sessions that were NOT held-out evaluations; excluded sessions
are everything a held-out evaluation produced. Labels starting with `evaluate` mark the holdout."""
from __future__ import annotations

import json
from pathlib import Path

from polyloop.events import now_iso

from polyvoice.core.types import Judged


def is_holdout_label(label: str | None) -> bool:
    return bool(label) and (label.startswith("evaluate") or label == "eval")


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, j: Judged, *, label: str, policy_id: str, role: str, source: str, mirror: Path | None = None) -> dict:
        rec = {"ts": now_iso(), "session": j.session, "aliases": j.aliases, "task": j.task, "policy_id": policy_id,
               "role": role, "label": label, "source": source, "status": j.status, "reward": j.reward,
               "explanation": j.explanation, **{k: v for k, v in j.extra.items() if k in ("run_id", "simulation_output_id")}}
        line = json.dumps(rec) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(line)
        if mirror is not None:
            with mirror.open("a") as f:
                f.write(line)
        return rec

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def session_hints(self, *, pass_value: float, failures_only: bool) -> dict[str, str]:
        hints: dict[str, str] = {}
        for rec in self.read():
            if is_holdout_label(rec.get("label")):
                continue
            r = rec.get("reward")
            if r is None:
                continue
            passed = r >= pass_value
            if failures_only and passed:
                continue
            text = f"Hindsight from the evaluator: this conversation {'passed' if passed else 'failed'} (score {r:.2f})."
            if rec.get("explanation"):
                text += "\n" + rec["explanation"]
            for sid in _ids(rec):
                hints[sid] = text
        return hints

    def excluded_sessions(self) -> set[str]:
        out: set[str] = set()
        for rec in self.read():
            if is_holdout_label(rec.get("label")):
                out |= _ids(rec)
        return out


def _ids(rec: dict) -> set[str]:
    ids = {rec.get("session"), rec.get("simulation_output_id"), *(rec.get("aliases") or [])}
    return {s for s in ids if s}
