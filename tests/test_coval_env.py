import json
from pathlib import Path

import httpx
import pytest
import yaml
from polyloop.config import load_loop
from polyloop.environment import load_environment
from polyloop.events import LoopStore
from polyloop.stages import Runner

from polyvoice.coval import client as cc
from polyvoice.coval import rewards as rw
from polyvoice.envs import coval as envmod
from tests.fake_coval import FakeCoval


@pytest.fixture
def fake():
    return FakeCoval()


@pytest.fixture
def client(fake):
    return cc.CovalClient(api_base="https://api.test/v1", api_key="k", transport=httpx.MockTransport(fake.handler))


@pytest.fixture
def proxy_calls(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None, **kw):
        calls.append((url, json))
        return httpx.Response(200, json={"ok": True})

    def fake_get(url, timeout=None, **kw):
        return httpx.Response(200, json={"live": "base"})

    monkeypatch.setattr(envmod.httpx, "post", fake_post)
    monkeypatch.setattr(envmod.httpx, "get", fake_get)
    return calls


def _loop(tmp_path, **opts):
    cfg = {
        "name": "dental", "model": "m", "tasks": "coval://tsPOOL00", "runs_dir": str(tmp_path / "runs"),
        "proxy_url": "http://127.0.0.1:8787",
        "environment": {"kind": "polyvoice.envs.coval:CovalEnvironment",
                        "options": {"agent_id": "agBASE", "persona_id": "pSTD", "metric_ids": ["mA", "mB"],
                                    "public_url": "https://t.example", "poll_seconds": 0, **opts}},
        "stages": [{"kind": "opsd", "min_rows": 1}],
        "filter": {"pool_sample": 5, "rollouts_per_task": 2, "min_tasks": 1, "target_tasks": 1, "max_rounds": 1},
        "gate": {"holdout": "coval://tsHOLD00", "holdout_limit": 4, "repeats": 2},
    }
    p = tmp_path / "loop.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _runner(tmp_path, client, **opts):
    cfg = load_loop(_loop(tmp_path, **opts))
    store = LoopStore(cfg.runs_path, cfg.name)
    cycle = store.cycle(store.new_cycle_id())
    runner = Runner(cfg, cycle, log=lambda m: None)
    runner.env._client = client
    return runner, cycle


def test_registered_by_entry_point_name_or_path(tmp_path):
    from polyloop.environment import resolve
    assert resolve("polyvoice.envs.coval:CovalEnvironment").name == "coval"


def test_client_shapes(client, fake):
    tasks = client.test_cases("tsPOOL00")
    assert [t.task_name for t in tasks] == ["tc0", "tc1", "tc2", "tc3", "tc4"]   # paginated
    meta = cc.agent_metadata("https://t.example/v1/chat/completions", auth_header=None, temperature=1.0, max_tokens=256)
    assert meta["custom_headers"]["X-Session-Id"] == "{{simulation_output_id}}"
    assert json.loads(meta["input_template"].replace("{{messages}}", "[]"))["max_tokens"] == 256
    with pytest.raises(cc.CovalError):
        client.launch_run(agent_id="a", persona_id="p", ts_id="t", metric_ids=[], test_case_ids=["x"], iterations=51,
                          concurrency=5, display_name="d", tags=[])


def test_rewards_and_explanations():
    ms = [{"metric_id": "mA", "value": 0, "status": "COMPLETED", "explanation": "no phone number", "metric_name": "task"},
          {"metric_id": "mB", "value": "true", "status": "COMPLETED"}]
    assert rw.reward_from_metrics(ms, ["mA", "mB"]) == 0.5
    assert rw.reward_from_metrics(ms, ["mA", "mC"]) is None
    assert rw.judge_explanations(ms, ["mA", "mB"], 500) == "task = 0: no phone number"


def test_filter_through_coval(tmp_path, client, fake, proxy_calls):
    fake.scores[("tc1", 0)] = 0.0
    runner, cycle = _runner(tmp_path, client)
    runner.snapshot()
    runner.filter()
    st = cycle.state
    assert st["train_task_names"] == ["tc1"]                        # 0 < mean < 1 only for tc1
    launch = fake.launches[0]
    assert launch["options"]["iteration_count"] == 2 and launch["options"]["concurrency"] == 5
    assert sorted(launch["options"]["test_case_ids"]) == ["tc0", "tc1", "tc2", "tc3", "tc4"]
    assert launch["metric_ids"] == ["mA", "mB"]
    # one agent duplicated and pointed at the tunnel
    assert fake.patches[0][1]["metadata"]["chat_endpoint"] == "https://t.example/v1/chat/completions"
    # the proxy was told to serve the base model, then cleared; every session flushed
    assert proxy_calls[0] == ("http://127.0.0.1:8787/admin/serve", {"sampler_path": None})
    assert proxy_calls[-1] == ("http://127.0.0.1:8787/admin/serve", {"clear": True})
    assert sum(1 for u, _ in proxy_calls if "/admin/session/" in u) == 10
    # tc3 simulations FAILED -> task errored, not scored
    results = [json.loads(l) for l in (cycle.subdir("filter") / "results.jsonl").read_text().splitlines()]
    assert {r["task"]: r["error"] for r in results if r["error"]} == {"tc3": "no scored simulations (failed=2, unscored=0)"}
    # ledger has the judge's explanation for the failed session
    led = rw.read_ledger(runner.env.ledger_path)
    bad = [r for r in led if r["reward"] == 0.5]
    assert bad and "phone number" in bad[0]["explanation"]
    hints = runner.env.session_hints()
    assert set(hints) == {"run0-tc1-0"} and "failed (score 0.50)" in hints["run0-tc1-0"]


def test_evaluate_uses_policy_paths_and_excludes_holdout(tmp_path, client, fake, proxy_calls):
    runner, cycle = _runner(tmp_path, client)
    runner.snapshot()
    cycle.update(candidate={"id": "dental-c1", "sampler_path": "tinker://cand"})
    runner.evaluate()
    served = [b for u, b in proxy_calls if u.endswith("/admin/serve") and "sampler_path" in b]
    assert served == [{"sampler_path": None}, {"sampler_path": "tinker://cand"}]
    st = cycle.state
    assert set(st["eval_candidate"]) == {"h0", "h1", "h2", "h3"} and all(v == 1.0 for v in st["eval_candidate"].values())
    excluded = runner.env.excluded_sessions()
    assert len(excluded) == 4 * 2 * 2 and all(s.startswith("run") for s in excluded)
    assert runner.env.session_hints() == {}                          # holdout sessions never hint training


def test_candidate_proxy_role(tmp_path, client, fake, proxy_calls):
    runner, cycle = _runner(tmp_path, client, candidate={"public_url": "https://c.example", "proxy_url": "http://127.0.0.1:8788"})
    runner.snapshot()
    cycle.update(candidate={"id": "dental-c1", "sampler_path": "tinker://cand"})
    runner.evaluate()
    urls = [u for u, b in proxy_calls if u.endswith("/admin/serve")]
    assert urls[0].startswith("http://127.0.0.1:8787") and urls[2].startswith("http://127.0.0.1:8788")
    endpoints = {p[1]["metadata"]["chat_endpoint"] for p in fake.patches}
    assert endpoints == {"https://t.example/v1/chat/completions", "https://c.example/v1/chat/completions"}


def test_recipe_loads():
    cfg = load_loop(Path(__file__).resolve().parents[1] / "recipes" / "dental" / "loop.yaml")
    assert cfg.environment.kind == "coval" and cfg.stages[0].kind == "opsd"
    assert cfg.gate.holdout_limit == 8
    hold = json.loads((Path(__file__).resolve().parents[1] / "recipes" / "dental" / "scenarios-holdout.json").read_text())
    assert len(hold) >= max(4, cfg.gate.holdout_limit // 2)
