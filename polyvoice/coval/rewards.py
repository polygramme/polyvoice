"""Rewards and hindsight from Coval metric outputs, and the per-session ledger."""
from __future__ import annotations

import json
from pathlib import Path


def _num(v) -> float | None:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        low = v.strip().lower()
        if low in ("true", "pass", "passed", "yes"):
            return 1.0
        if low in ("false", "fail", "failed", "no"):
            return 0.0
        try:
            return float(low)
        except ValueError:
            return None
    return None


def reward_from_metrics(metrics: list[dict], metric_ids: list[str]) -> float | None:
    """Mean of the reward metrics' values on one simulation; None unless every one scored."""
    by_id = {m.get("metric_id"): m for m in metrics if m.get("status", "COMPLETED") == "COMPLETED"}
    got = []
    for mid in metric_ids:
        m = by_id.get(mid)
        v = _num(m.get("value")) if m else None
        if v is None:
            return None
        got.append(v)
    return sum(got) / len(got) if got else None


def judge_explanations(metrics: list[dict], metric_ids: list[str], max_chars: int) -> str | None:
    """Fold the reward metrics' explanations into one hindsight block."""
    parts = []
    for m in metrics:
        if m.get("metric_id") not in metric_ids or m.get("status", "COMPLETED") != "COMPLETED":
            continue
        text = (m.get("explanation") or (m.get("result") or {}).get("llm", {}).get("answer_explanation") or "").strip()
        if not text:
            continue
        parts.append(f"{m.get('metric_name') or m['metric_id']} = {m.get('value')}: {text}")
    if not parts:
        return None
    out = "\n".join(parts)
    if len(out) > max_chars:
        out = out[: max_chars // 2] + "\n...\n" + out[-max_chars // 2:]
    return out


def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def is_holdout_label(label: str | None) -> bool:
    return bool(label) and (label.startswith("evaluate") or label == "eval")


def session_hints(ledger: list[dict], *, pass_value: float, failures_only: bool) -> dict[str, str]:
    """Session id -> hindsight text, from scored sessions that are NOT held-out evaluations."""
    hints: dict[str, str] = {}
    for rec in ledger:
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
        for sid in filter(None, {rec.get("session"), rec.get("simulation_output_id")}):
            hints[sid] = text
    return hints


def excluded_sessions(ledger: list[dict]) -> set[str]:
    """Every session produced by a held-out evaluation: never a training row."""
    out: set[str] = set()
    for rec in ledger:
        if is_holdout_label(rec.get("label")):
            out |= {s for s in (rec.get("session"), rec.get("simulation_output_id")) if s}
    return out
