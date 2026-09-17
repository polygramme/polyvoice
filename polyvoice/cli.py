"""polyvoice: account, check, seed, ledger. The cycle itself is `polyloop run`."""
from __future__ import annotations

import json
from pathlib import Path

import click
from polyloop.config import load_loop
from polyloop.environment import load_environment
from polyloop.events import LoopStore

from polyvoice.coval import client as cc
from polyvoice.coval import rewards as rw


def _env(loop_path):
    cfg = load_loop(loop_path)
    store = LoopStore(cfg.runs_path, cfg.name)
    return cfg, store, load_environment(cfg, store, log=click.echo)


@click.group()
def main():
    """A voice agent's LLM that improves from its own calls (polyloop-rl + Coval)."""


@main.command("account")
def account():
    """What the Coval account has: agents, personas, test sets, metrics (read-only)."""
    c = cc.CovalClient.from_env()
    for res, key in (("agents", "agents"), ("personas", "personas"), ("test-sets", "test_sets"), ("metrics", "metrics")):
        rows = c.list(res, key)
        click.echo(f"{res}: {len(rows)}")
        for r in rows:
            click.echo(f"  {r.get('id') or r.get('agent_id'):24} {r.get('display_name') or r.get('name') or r.get('metric_name') or ''}"
                       f"  {r.get('model_type') or r.get('metric_type') or r.get('test_set_type') or ''}")


@main.command("check")
@click.option("--loop", "loop_path", required=True)
def check(loop_path):
    """Preflight the Coval environment of a loop without launching anything."""
    cfg, store, env = _env(loop_path)
    problems = env.preflight()
    pool = env.load_tasks(cfg.tasks)
    hold = env.load_tasks(cfg.gate.holdout, limit=cfg.gate.holdout_limit, seed=cfg.gate.holdout_seed)
    click.echo(f"pool {cfg.tasks}: {len(pool)} test cases; holdout {cfg.gate.holdout}: {len(hold)}")
    need = max(4, cfg.gate.holdout_limit // 2)
    if len(hold) < need:
        problems.append(f"holdout has {len(hold)} cases but the gate needs >= {need} (holdout_limit {cfg.gate.holdout_limit})")
    for p in problems:
        click.echo(f"PROBLEM: {p}")
    click.echo("ok" if not problems else f"{len(problems)} problem(s)")
    raise SystemExit(1 if problems else 0)


@main.command("seed")
@click.option("--test-set", "ts_id", required=True, help="Coval test set id (8 chars), or NEW:<display name> to create one.")
@click.option("--file", "path", required=True, type=click.Path(exists=True), help="JSON list of {input, expected, description}.")
def seed(ts_id, path):
    """Create SCENARIO test cases in a Coval test set (free; no simulation is launched)."""
    c = cc.CovalClient.from_env()
    if ts_id.startswith("NEW:"):
        ts = c.create_test_set(ts_id[4:])
        ts_id = ts.get("id") or ts.get("test_set_id")
        click.echo(f"created test set {ts_id}")
    cases = json.loads(Path(path).read_text())
    made = cc.seed_test_cases(c, ts_id, cases)
    click.echo(f"{len(made)} new test cases in {ts_id} ({len(cases) - len(made)} already present); dataset = coval://{ts_id}")


@main.command("ledger")
@click.option("--loop", "loop_path", required=True)
@click.option("--failed", is_flag=True, help="Only sessions below pass_value.")
def ledger(loop_path, failed):
    """Every scored simulation: session, task, policy, label, reward, judge explanation."""
    cfg, store, env = _env(loop_path)
    for rec in rw.read_ledger(env.ledger_path):
        if failed and (rec.get("reward") is None or rec["reward"] >= env.pass_value):
            continue
        click.echo(f"{rec['label']:20} {rec['policy_id']:28} {str(rec.get('task'))[:12]:12} {rec.get('reward')}  {rec['session']}")
        if failed and rec.get("explanation"):
            click.echo("    " + rec["explanation"].replace("\n", "\n    "))
