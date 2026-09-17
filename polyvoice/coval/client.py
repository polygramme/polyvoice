"""A thin client for Coval's public v1 API (https://docs.coval.ai/api-reference/v1/introduction).

Only what the loop needs: test cases of a test set, agents (duplicate + point at an endpoint),
runs (launch, poll), simulated conversations of a run and their metric outputs. httpx only,
X-API-Key auth. Response envelopes verified against the live API on 2026-09-17:
lists return {"<resource>": [...], "next_page_token": ""}; filter values must be double-quoted
(`test_set_id="abc12345"`; the unquoted form in the SDK docs is rejected); `/conversations/simulated` lists
simulations with `test_case_id`; `/conversations/simulated/{id}/metrics` returns the per-metric
outputs with `value` and `explanation`.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable

import httpx

COVAL_PREFIX = "coval://"
MAX_ITERATIONS = 50        # options.iteration_count cap
MAX_CONCURRENCY = 100      # options.concurrency cap (plans allow less: Starter 5, Growth 25)
MAX_CASES_PER_RUN = 100    # options.test_case_ids cap


def is_coval_dataset(dataset: str) -> bool:
    return str(dataset).startswith(COVAL_PREFIX)


def test_set_id(dataset: str) -> str:
    if not is_coval_dataset(dataset):
        raise ValueError(f"not a coval dataset (expected coval://<test_set_id>): {dataset!r}")
    return dataset[len(COVAL_PREFIX):].strip("/")


@dataclass
class CovalTask:
    """The slice of a Coval test case the loop keys on."""
    task_name: str                       # the test case id
    test_set_id: str
    input_str: str = ""
    expected_behaviors: list[str] = field(default_factory=list)
    description: str | None = None
    input_type: str | None = None

    @classmethod
    def from_api(cls, rec: dict) -> "CovalTask":
        return cls(task_name=rec["id"], test_set_id=rec.get("test_set_id", ""), input_str=rec.get("input_str") or "",
                   expected_behaviors=list(rec.get("expected_behaviors") or []), description=rec.get("description"),
                   input_type=rec.get("input_type"))


class CovalError(RuntimeError):
    pass


def _filter(field: str, value: str) -> str:
    return f'{field}="{value}"'


def _id(rec: dict, *names: str) -> str:
    for n in names:
        if rec.get(n):
            return rec[n]
    raise CovalError(f"record without {names}: {json.dumps(rec)[:200]}")


class CovalClient:
    def __init__(self, api_base: str = "https://api.coval.dev/v1", api_key: str | None = None, timeout: float = 60.0,
                 transport: httpx.BaseTransport | None = None):
        if not api_key:
            raise CovalError("Coval API key missing (set COVAL_API_KEY)")
        self.http = httpx.Client(base_url=api_base.rstrip("/"), headers={"X-API-Key": api_key}, timeout=timeout,
                                 transport=transport)

    @classmethod
    def from_env(cls, api_base: str = "https://api.coval.dev/v1", api_key_env: str = "COVAL_API_KEY", **kw) -> "CovalClient":
        return cls(api_base=os.environ.get("COVAL_API_BASE", api_base), api_key=os.environ.get(api_key_env), **kw)

    # ---- transport ----------------------------------------------------
    def _req(self, method: str, path: str, **kw) -> dict:
        r = self.http.request(method, path, **kw)
        if r.status_code >= 400:
            raise CovalError(f"{method} {path} -> {r.status_code}: {r.text[:400]}")
        return r.json() if r.content else {}

    def _paged(self, path: str, key: str, params: dict | None = None, page_size: int = 100) -> list[dict]:
        out: list[dict] = []
        token = None
        while True:
            q = dict(params or {}, page_size=page_size)
            if token:
                q["page_token"] = token
            data = self._req("GET", path, params=q)
            out.extend(data.get(key) or [])
            token = data.get("next_page_token")
            if not token:
                return out

    # ---- resources ----------------------------------------------------
    def list(self, resource: str, key: str | None = None) -> list[dict]:
        return self._paged(f"/{resource}", key or resource.replace("-", "_"))

    def test_cases(self, ts_id: str) -> list[CovalTask]:
        recs = self._paged("/test-cases", "test_cases", {"filter": _filter("test_set_id", ts_id), "order_by": "create_time"})
        return [CovalTask.from_api(r) for r in recs]

    def test_set(self, ts_id: str) -> dict:
        d = self._req("GET", f"/test-sets/{ts_id}")
        return d.get("test_set") or d

    def create_test_set(self, display_name: str, description: str = "", test_set_type: str = "SCENARIO") -> dict:
        """Create a test set, or return the existing one with this display name."""
        for ts in self.list("test-sets", "test_sets"):
            if ts.get("display_name") == display_name:
                return ts
        d = self._req("POST", "/test-sets", json={"display_name": display_name, "description": description,
                                                  "test_set_type": test_set_type})
        return d.get("test_set") or d

    def agent(self, agent_id: str) -> dict:
        return self._req("GET", f"/agents/{agent_id}").get("agent", {})

    def persona(self, persona_id: str) -> dict:
        return self._req("GET", f"/personas/{persona_id}").get("persona", {})

    def metric(self, metric_id: str) -> dict:
        return self._req("GET", f"/metrics/{metric_id}").get("metric", {})

    def agent_for_endpoint(self, base_agent_id: str, *, display_name: str, chat_endpoint: str, auth_header: str | None,
                           temperature: float, max_tokens: int, customer_agent_id: str) -> str:
        """Duplicate the base agent and point the copy at `chat_endpoint` (an OpenAI-compatible
        chat completions URL). PATCH replaces `metadata` wholesale, so the base's is read first."""
        dup = self._req("POST", f"/agents/{base_agent_id}/duplicate", json={})
        agent = dup.get("agent") or dup
        agent_id = _id(agent, "id", "agent_id")
        meta = dict(self.agent(base_agent_id).get("metadata") or {})
        meta.update(agent_metadata(chat_endpoint, auth_header=auth_header, temperature=temperature, max_tokens=max_tokens))
        self._req("PATCH", f"/agents/{agent_id}", json={"display_name": display_name[:200], "metadata": meta,
                                                       "customer_agent_id": customer_agent_id[:100]})
        return agent_id

    def launch_run(self, *, agent_id: str, persona_id: str, ts_id: str, metric_ids: list[str], test_case_ids: list[str],
                   iterations: int, concurrency: int, display_name: str, tags: list[str]) -> dict:
        if iterations > MAX_ITERATIONS:
            raise CovalError(f"iteration_count {iterations} > {MAX_ITERATIONS} (Coval cap)")
        if len(test_case_ids) > MAX_CASES_PER_RUN:
            raise CovalError(f"{len(test_case_ids)} test cases in one run > {MAX_CASES_PER_RUN} (Coval cap)")
        body = {
            "agent_id": agent_id, "persona_id": persona_id, "test_set_id": ts_id, "metric_ids": metric_ids,
            "options": {"iteration_count": iterations, "concurrency": min(concurrency, MAX_CONCURRENCY),
                        "test_case_ids": test_case_ids},
            "metadata": {"display_name": display_name[:200], "tags": tags[:20], "created_by": "polyvoice"},
        }
        d = self._req("POST", "/runs", json=body)
        return d.get("run") or d

    def run(self, run_id: str) -> dict:
        d = self._req("GET", f"/runs/{run_id}")
        return d.get("run") or d

    def wait_run(self, run_id: str, *, poll: int, timeout: int, log: Callable[[str], None] = print) -> dict:
        t0 = time.monotonic()
        last = ""
        while True:
            run = self.run(run_id)
            status = run.get("status")
            prog = run.get("progress") or {}
            line = (f"coval run {run_id}: {status} {prog.get('completed_test_cases', '?')}/{prog.get('total_test_cases', '?')}"
                    f" done, {prog.get('failed_test_cases', 0)} failed")
            if line != last:
                log(line)
                last = line
            if status in ("COMPLETED", "FAILED", "CANCELLED", "DELETED"):
                return run
            if time.monotonic() - t0 > timeout:
                raise CovalError(f"run {run_id} still {status} after {timeout}s")
            time.sleep(poll)

    def simulations(self, run_id: str) -> list[dict]:
        """Simulations of a run with their test_case_id (no `include=metric_values`: on that
        path the API nulls test_case_id). Metric values come from simulation_metrics()."""
        return self._paged("/conversations/simulated", "simulated_conversations", {"filter": _filter("run_id", run_id)})

    def simulation_metrics(self, simulation_id: str) -> list[dict]:
        return self._paged(f"/conversations/simulated/{simulation_id}/metrics", "metrics", {"view": "BASIC"})

    def simulation(self, simulation_id: str) -> dict:
        d = self._req("GET", f"/conversations/simulated/{simulation_id}")
        return d.get("simulated_conversation") or d.get("simulation") or d

    def submit_conversation(self, *, agent_id: str, transcript: list[dict], metric_ids: list[str] | None = None,
                            metadata: dict | None = None, external_id: str | None = None) -> dict:
        """Production monitoring: score a real conversation with the org's metric policy."""
        body = {"agent_id": agent_id, "transcript": transcript}
        if metric_ids:
            body["metrics"] = metric_ids
        if metadata:
            body["metadata"] = metadata
        if external_id:
            body["external_conversation_id"] = external_id
        return self._req("POST", "/conversations:submit", json=body)


def agent_metadata(chat_endpoint: str, *, auth_header: str | None, temperature: float, max_tokens: int) -> dict:
    """CHAT-agent metadata (docs.coval.ai/concepts/agents/connections/chat): Coval POSTs the
    OpenAI chat body to `chat_endpoint`; the simulation id rides X-Session-Id so the proxy keys
    the conversation's trace by the id Coval reports scores under."""
    meta = {
        "chat_endpoint": chat_endpoint,
        "custom_headers": {"X-Session-Id": "{{simulation_output_id}}", "X-Turn-Type": "main"},
        "input_template": json.dumps({"model": "polyvoice", "messages": "{{messages}}", "temperature": temperature,
                                      "max_tokens": max_tokens}).replace('"{{messages}}"', "{{messages}}"),
        "response_message_path": "choices.0.message.content",
        "response_format": "chat_completions",
        "strip_message_timestamps": True,
    }
    if auth_header:
        meta["authorization_header"] = auth_header
    return meta


def load_tasks(dataset: str, *, client: CovalClient, limit: int | None = None, seed: int = 0,
               names: list[str] | None = None) -> list[CovalTask]:
    tasks = client.test_cases(test_set_id(dataset))
    if names is not None:
        wanted = set(names)
        tasks = [t for t in tasks if t.task_name in wanted]
    if limit and len(tasks) > limit:
        rng = random.Random(seed)
        tasks = sorted(rng.sample(tasks, limit), key=lambda t: t.task_name)
    return tasks


def seed_test_cases(client: CovalClient, ts_id: str, cases: list[dict]) -> list[str]:
    """Create SCENARIO test cases from {"input", "expected": [...], "description"} records;
    returns the new ids. Cases whose input already exists in the set are skipped."""
    have = {t.input_str.strip() for t in client.test_cases(ts_id)}
    made = []
    for c in cases:
        if c["input"].strip() in have:
            continue
        body = {"test_set_id": ts_id, "input_str": c["input"], "input_type": c.get("input_type", "SCENARIO"),
                "expected_behaviors": list(c.get("expected", [])), "description": c.get("description", "")}
        r = client._req("POST", "/test-cases", json=body)
        tc = r.get("test_case") or r
        made.append(tc.get("id", "?"))
    return made
