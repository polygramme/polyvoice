"""Coval as a verifier: one Coval run of K iterations over the chosen test cases, judged by the configured
metrics; reward = mean of the metrics, hint = the judges' explanations. Coval sends its simulation id to the
bot's chat endpoint as X-Session-Id, so `Judged.session` joins straight to the proxy's traces."""
from __future__ import annotations

import time
from typing import Any

from polyloop.events import now_iso

from polyvoice.coval import client as cc
from polyvoice.coval import rewards as rw
from polyvoice.core.types import Judged, Target, Task


class CovalVerifier:
    name = "coval"

    def __init__(self, *, agent_id: str, persona_id: str, metric_ids: list[str], loop_name: str, store, log=print,
                 concurrency: int = 5, poll_seconds: int = 15, run_timeout: int = 3600, temperature: float = 1.0,
                 max_tokens: int = 256, max_judge_chars: int = 1200, api_base: str = "https://api.coval.dev/v1",
                 api_key_env: str = "COVAL_API_KEY", client: Any = None):
        self.agent_id, self.persona_id, self.metric_ids = agent_id, persona_id, list(metric_ids)
        self.loop_name, self.store, self.log = loop_name, store, log
        self.concurrency, self.poll_seconds, self.run_timeout = concurrency, poll_seconds, run_timeout
        self.temperature, self.max_tokens, self.max_judge_chars = temperature, max_tokens, max_judge_chars
        self.api_base, self.api_key_env = api_base, api_key_env
        self._client = client

    @property
    def client(self) -> cc.CovalClient:
        if self._client is None:
            self._client = cc.CovalClient.from_env(api_base=self.api_base, api_key_env=self.api_key_env)
        return self._client

    # ---- tasks ------------------------------------------------------------
    def load_tasks(self, dataset: str, *, limit=None, seed=0, names=None) -> list[Task]:
        return [Task(task_name=t.task_name, payload=t)
                for t in cc.load_tasks(dataset, client=self.client, limit=limit, seed=seed, names=names)]

    def preflight(self, datasets: list[str]) -> list[str]:
        problems = []
        try:
            agent = self.client.agent(self.agent_id)
            if agent.get("model_type") not in (None, "MODEL_TYPE_CHAT"):
                problems.append(f"coval agent {self.agent_id} is {agent.get('model_type')}, expected MODEL_TYPE_CHAT")
            self.client.persona(self.persona_id)
            for ds in datasets:
                self.client.test_set(cc.test_set_id(ds))
            for mid in self.metric_ids:
                if not self.client.metric(mid):
                    problems.append(f"metric {mid} does not resolve")
        except Exception as exc:
            problems.append(f"coval api: {type(exc).__name__}: {str(exc)[:200]}")
        return problems

    # ---- agents -----------------------------------------------------------
    def agent_for(self, role: str, chat_endpoint: str) -> str:
        """One Coval agent per endpoint (duplicated from the recipe's base agent); ids cached in the store.
        Agents are found by display name: Coval keeps customer_agent_id unique forever, even across deletes."""
        cache = self.store.cache("polyvoice_agents")
        key = f"{self.agent_id}:{chat_endpoint}"
        if key in cache:
            return cache[key]["agent_id"]
        agent_id = self.client.agent_for_endpoint(
            self.agent_id, display_name=f"{self.loop_name} {role}", chat_endpoint=chat_endpoint, auth_header=None,
            temperature=self.temperature, max_tokens=self.max_tokens, customer_agent_id=f"polyvoice/{self.loop_name}/{role}")
        cache[key] = {"agent_id": agent_id, "chat_endpoint": chat_endpoint, "role": role, "created": now_iso()}
        self.store.save_cache("polyvoice_agents", cache)
        self.log(f"coval: agent {agent_id} -> {chat_endpoint}")
        return agent_id

    # ---- run --------------------------------------------------------------
    def run(self, *, label: str, tasks: list[Task], k: int, policy_id: str, target: Target, out=None,
            role: str = "default") -> list[Judged]:
        ts_id = tasks[0].payload.test_set_id
        if not ts_id:
            raise cc.CovalError("tasks carry no test_set_id")
        agent_id = self.agent_for(role, target.chat_endpoint)
        ids = [t.task_name for t in tasks]
        judged: list[Judged] = []
        for start in range(0, len(ids), cc.MAX_CASES_PER_RUN):
            chunk = ids[start:start + cc.MAX_CASES_PER_RUN]
            t0 = time.monotonic()
            run = self.client.launch_run(agent_id=agent_id, persona_id=self.persona_id, ts_id=ts_id,
                                         metric_ids=self.metric_ids, test_case_ids=chunk, iterations=k,
                                         concurrency=self.concurrency, display_name=f"{self.loop_name} {label} {policy_id}",
                                         tags=["polyvoice", self.loop_name, label, policy_id])
            run_id = cc._id(run, "run_id", "id")
            self.log(f"coval: run {run_id}: {len(chunk)} test cases x {k} against {policy_id}")
            run = self.client.wait_run(run_id, poll=self.poll_seconds, timeout=self.run_timeout, log=self.log)
            if run.get("status") != "COMPLETED":
                raise cc.CovalError(f"run {run_id} ended {run.get('status')}: {run.get('error') or run.get('error_status')}")
            sims = self.client.simulations(run_id)
            per = (time.monotonic() - t0) / max(1, len(chunk))
            for sim in sims:
                sid = cc._id(sim, "simulation_id", "id")
                status = sim.get("status", "COMPLETED")
                reward = explanation = None
                if status == "COMPLETED":
                    try:
                        metrics = self.client.simulation_metrics(sid)
                    except cc.CovalError as exc:
                        self.log(f"coval: metrics fetch failed for {sid}: {exc}")
                        metrics = []
                    reward = rw.reward_from_metrics(metrics, self.metric_ids)
                    explanation = rw.judge_explanations(metrics, self.metric_ids, self.max_judge_chars)
                soid = sim.get("simulation_output_id")
                judged.append(Judged(task=sim.get("test_case_id") or "?", session=sid, reward=reward,
                                     explanation=explanation, status=status, seconds=per,
                                     aliases=[soid] if soid else [],
                                     extra={"run_id": run_id, "simulation_output_id": soid}))
        return judged
