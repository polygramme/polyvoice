"""Coval as the environment and the verifier for a polyloop cycle.

    task          = one Coval test case (a caller scenario + expected behaviours)
    K episodes    = one Coval run, iteration_count=K, over the chosen test case ids
    reward        = mean of the configured metrics on each simulated conversation
    trace session = the Coval simulation id (Coval sends it as X-Session-Id to the proxy)
    hindsight     = the judge's verdict and explanation, joined to the traces by session id

Which adapter answers a run: before each rollout the environment tells the polyloop proxy to
serve the policy's sampler path (`POST /admin/serve`, a transient override that never touches
live.json) and clears it afterwards. With a second proxy instance (`polyloop proxy --slot
candidate`) the candidate is scored on its own endpoint while production traffic stays on the
live adapter; with one instance, incumbent and candidate are scored one after the other.

Coval reaches a proxy through a public https URL (it refuses private addresses). One Coval agent
is created per endpoint by duplicating the recipe's base agent; the ids are cached in the store.

loop.yaml:

    environment:
      kind: coval
      options:
        agent_id: ...            # a MODEL_TYPE_CHAT agent; its config is copied per endpoint
        persona_id: ...
        metric_ids: [...]        # reward = their mean; binary judges score 0/1
        public_url: https://...  # tunnel to the proxy at proxy_url
        candidate:               # optional second proxy
          public_url: https://...
          proxy_url: http://127.0.0.1:8788
        concurrency: 5
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import httpx
from polyloop.environment import BaseEnvironment, Policy
from polyloop.events import now_iso
from polyloop.harness.rollout import TaskResult

from polyvoice.coval import client as cc
from polyvoice.coval import rewards as rw


class CovalEnvironment(BaseEnvironment):
    name = "coval"
    needs_engine_warm = True   # the proxy samples from the trainer's engines

    def __init__(self, cfg, store, log=print, *, agent_id: str, persona_id: str, metric_ids: list[str], public_url: str,
                 proxy_url: str | None = None, candidate: dict | None = None, concurrency: int = 5, poll_seconds: int = 15,
                 run_timeout: int = 3600, temperature: float | None = None, max_tokens: int = 256, pass_value: float = 1.0,
                 hint_failures_only: bool = True, max_judge_chars: int = 1200, api_base: str = "https://api.coval.dev/v1",
                 api_key_env: str = "COVAL_API_KEY", ledger: str = "coval_ledger.jsonl", check_public: bool = True,
                 client: Any = None):
        super().__init__(cfg, store, log=log)
        self.agent_id, self.persona_id, self.metric_ids = agent_id, persona_id, list(metric_ids)
        self.proxies = {"default": {"public_url": public_url, "proxy_url": proxy_url or cfg.proxy_url}}
        if candidate:
            self.proxies["candidate"] = {"public_url": candidate["public_url"],
                                         "proxy_url": candidate.get("proxy_url") or cfg.proxy_url}
        self.concurrency, self.poll_seconds, self.run_timeout = concurrency, poll_seconds, run_timeout
        self.temperature = cfg.gate.temperature if temperature is None else temperature
        self.max_tokens, self.pass_value = max_tokens, pass_value
        self.hint_failures_only, self.max_judge_chars = hint_failures_only, max_judge_chars
        self.api_base, self.api_key_env = api_base, api_key_env
        self.ledger_path = store.root / ledger
        self.check_public = check_public  # False where the host cannot reach its own public URL (e.g. inside a Modal container)
        self._client = client

    # ---- pieces ---------------------------------------------------------
    @property
    def client(self) -> cc.CovalClient:
        if self._client is None:
            self._client = cc.CovalClient.from_env(api_base=self.api_base, api_key_env=self.api_key_env)
        return self._client

    def _role(self, policy: Policy) -> str:
        inc = (self.store.incumbent() or {}).get("id", "base")
        pid = (policy or {}).get("id", "base")
        return "candidate" if "candidate" in self.proxies and pid not in (inc, "base", "adhoc") else "default"

    def _proxy_post(self, proxy_url: str, path: str, body: dict, timeout: float = 600) -> dict:
        r = httpx.post(proxy_url.rstrip("/") + path, json=body, timeout=timeout)
        if r.status_code >= 400:
            raise cc.CovalError(f"proxy {proxy_url}{path} -> {r.status_code} {r.text[:200]}")
        return r.json() if r.content else {}

    def _agent_for(self, role: str) -> str:
        cache = self.store.cache("polyvoice_agents")
        endpoint = self.proxies[role]["public_url"].rstrip("/") + "/v1/chat/completions"
        key = f"{self.agent_id}:{endpoint}"
        if key in cache:
            return cache[key]["agent_id"]
        agent_id = self.client.agent_for_endpoint(
            self.agent_id, display_name=f"{self.cfg.name} {role}", chat_endpoint=endpoint, auth_header=None,
            temperature=self.temperature, max_tokens=self.max_tokens, customer_agent_id=f"polyvoice/{self.cfg.name}/{role}")
        cache[key] = {"agent_id": agent_id, "chat_endpoint": endpoint, "role": role, "created": now_iso()}
        self.store.save_cache("polyvoice_agents", cache)
        self.log(f"coval: agent {agent_id} -> {endpoint}")
        return agent_id

    # ---- Environment ------------------------------------------------------
    def load_tasks(self, dataset: str, *, limit: int | None = None, seed: int = 0, names: list[str] | None = None):
        return cc.load_tasks(dataset, client=self.client, limit=limit, seed=seed, names=names)

    def preflight(self) -> list[str]:
        problems = []
        try:
            agent = self.client.agent(self.agent_id)
            if agent.get("model_type") not in (None, "MODEL_TYPE_CHAT"):
                problems.append(f"coval agent {self.agent_id} is {agent.get('model_type')}, expected MODEL_TYPE_CHAT")
            self.client.persona(self.persona_id)
            for ds in (self.cfg.tasks, self.cfg.gate.holdout):
                self.client.test_set(cc.test_set_id(ds))
            for mid in self.metric_ids:
                if not self.client.metric(mid):
                    problems.append(f"metric {mid} does not resolve")
        except Exception as exc:
            problems.append(f"coval api: {type(exc).__name__}: {str(exc)[:200]}")
        if abs(self.temperature - self.cfg.gate.temperature) > 1e-9:
            problems.append(f"coval temperature {self.temperature} != gate.temperature {self.cfg.gate.temperature}")
        for role, p in self.proxies.items():
            if not p["proxy_url"]:
                problems.append(f"{role} proxy_url unset")
            else:
                try:
                    r = httpx.get(p["proxy_url"].rstrip("/") + "/admin/status", timeout=10)
                    if r.status_code >= 400:
                        problems.append(f"{role} proxy {r.status_code} at {p['proxy_url']}")
                except Exception as exc:
                    problems.append(f"{role} proxy unreachable at {p['proxy_url']}: {exc}")
            if not str(p["public_url"]).startswith("https://"):
                problems.append(f"{role} public_url must be https (Coval refuses http and private IPs): {p['public_url']}")
            elif self.check_public:
                try:
                    r = httpx.get(p["public_url"].rstrip("/") + "/healthz", timeout=15)
                    if r.status_code >= 400:
                        problems.append(f"{role} public proxy {r.status_code} at {p['public_url']}/healthz")
                except Exception as exc:
                    problems.append(f"{role} public proxy unreachable at {p['public_url']}: {exc}")
        return problems

    def run_rollouts(self, *, label: str, tasks: list[Any], policy: Policy, k: int, temperature: float,
                     out: Path | None, on_result: Callable[[TaskResult], None]) -> list[TaskResult]:
        if not tasks:
            return []
        role = self._role(policy)
        proxy_url = self.proxies[role]["proxy_url"]
        agent_id = self._agent_for(role)
        ts_id = tasks[0].test_set_id
        if not ts_id:
            raise cc.CovalError("tasks carry no test_set_id")
        pid = (policy or {}).get("id", "base")
        results = {t.task_name: TaskResult(task=t.task_name) for t in tasks}
        ids = [t.task_name for t in tasks]
        self._proxy_post(proxy_url, "/admin/serve", {"sampler_path": (policy or {}).get("sampler_path")})
        self.log(f"coval: {role} proxy serves {pid} ({(policy or {}).get('sampler_path') or 'base model'})")
        try:
            for start in range(0, len(ids), cc.MAX_CASES_PER_RUN):
                chunk = ids[start:start + cc.MAX_CASES_PER_RUN]
                t0 = time.monotonic()
                run = self.client.launch_run(agent_id=agent_id, persona_id=self.persona_id, ts_id=ts_id,
                                             metric_ids=self.metric_ids, test_case_ids=chunk, iterations=k,
                                             concurrency=self.concurrency, display_name=f"{self.cfg.name} {label} {pid}",
                                             tags=["polyvoice", self.cfg.name, label, pid])
                run_id = cc._id(run, "run_id", "id")
                self.log(f"coval: run {run_id}: {len(chunk)} test cases x {k} against {pid}")
                run = self.client.wait_run(run_id, poll=self.poll_seconds, timeout=self.run_timeout, log=self.log)
                if run.get("status") != "COMPLETED":
                    raise cc.CovalError(f"run {run_id} ended {run.get('status')}: {run.get('error') or run.get('error_status')}")
                sims = self.client.simulations(run_id)
                seconds = time.monotonic() - t0
                grouped: dict[str, dict] = {}
                with self.ledger_path.open("a") as lf:
                    for sim in sims:
                        sid = cc._id(sim, "simulation_id", "id")
                        self._finish_session(proxy_url, sid)
                        task = sim.get("test_case_id") or "?"
                        g = grouped.setdefault(task, {"rewards": [], "sessions": [], "failed": 0, "unscored": 0})
                        reward = explanation = None
                        if sim.get("status", "COMPLETED") != "COMPLETED":
                            g["failed"] += 1
                        else:
                            try:
                                metrics = self.client.simulation_metrics(sid)
                            except cc.CovalError as exc:
                                self.log(f"coval: metrics fetch failed for {sid}: {exc}")
                                metrics = []
                            reward = rw.reward_from_metrics(metrics, self.metric_ids)
                            explanation = rw.judge_explanations(metrics, self.metric_ids, self.max_judge_chars)
                            if reward is None:
                                g["unscored"] += 1
                            else:
                                g["rewards"].append(reward)
                                g["sessions"].append(sid)
                        rec = {"ts": now_iso(), "session": sid, "simulation_output_id": sim.get("simulation_output_id"),
                               "task": task, "run_id": run_id, "policy_id": pid, "role": role, "label": label,
                               "status": sim.get("status"), "reward": reward, "explanation": explanation}
                        lf.write(json.dumps(rec) + "\n")
                        if out:
                            with (out / "rewards.jsonl").open("a") as f:
                                f.write(json.dumps(rec) + "\n")
                for task in chunk:
                    res, g = results[task], grouped.get(task)
                    res.seconds = seconds / max(1, len(chunk))
                    if not g or not g["rewards"]:
                        res.error = f"no scored simulations (failed={g['failed'] if g else 0}, unscored={g['unscored'] if g else 0})"
                    else:
                        res.rewards = g["rewards"]
                        res.turns = [0] * len(g["rewards"])
                        res.tokens = [0] * len(g["rewards"])
                        res.stop_reasons = [None] * len(g["rewards"])
                    on_result(res)
        finally:
            self._proxy_post(proxy_url, "/admin/serve", {"clear": True})
        return [results[t] for t in ids]

    def _finish_session(self, proxy_url: str, session: str) -> None:
        """Coval never sends X-Session-Done; flush the last turn so it is recorded."""
        try:
            httpx.post(proxy_url.rstrip("/") + f"/admin/session/{session}/done", timeout=10)
        except Exception:
            pass

    def session_hints(self) -> dict[str, str]:
        return rw.session_hints(rw.read_ledger(self.ledger_path), pass_value=self.pass_value,
                                failures_only=self.hint_failures_only)

    def excluded_sessions(self) -> set[str]:
        return rw.excluded_sessions(rw.read_ledger(self.ledger_path))
