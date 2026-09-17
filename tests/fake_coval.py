"""Enough of api.coval.dev/v1 for the environment, in the shapes the live API returned on
2026-09-17: list envelopes {"<resource>": [...], "next_page_token": ""}; simulations listed
under /conversations/simulated carry test_case_id; metric outputs come from
/conversations/simulated/{id}/metrics."""
from __future__ import annotations

import json

import httpx


class FakeCoval:
    def __init__(self):
        self.cases = {"tsPOOL00": [{"id": f"tc{i}", "test_set_id": "tsPOOL00", "input_str": f"scenario {i}", "expected_behaviors": ["x"]} for i in range(5)],
                      "tsHOLD00": [{"id": f"h{i}", "test_set_id": "tsHOLD00", "input_str": f"hold {i}"} for i in range(4)]}
        self.agents = {"agBASE": {"id": "agBASE", "display_name": "base", "model_type": "MODEL_TYPE_CHAT",
                                  "metadata": {"chat_endpoint": "https://old/v1/chat/completions", "custom_data": "{}"}}}
        self.personas = {"pSTD": {"id": "pSTD", "name": "Standard Customer"}}
        self.metrics = {"mA": {"id": "mA", "metric_name": "task completed", "metric_type": "METRIC_LLM_BINARY"},
                        "mB": {"id": "mB", "metric_name": "no price", "metric_type": "METRIC_LLM_BINARY"}}
        self.runs, self.sims, self.launches, self.patches = {}, {}, [], []
        self.polls = 0
        self.scores = {}   # (test_case_id, iteration) -> mA score; default 1.0

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["X-API-Key"] == "k"
        path, q, m = request.url.path.replace("/v1", "", 1), dict(request.url.params), request.method
        if m == "GET" and path == "/test-cases":
            ts = q["filter"].split("=", 1)[1]
            page = int(q.get("page_token") or 0)
            items = self.cases[ts][page * 2:(page + 1) * 2]
            nxt = str(page + 1) if (page + 1) * 2 < len(self.cases[ts]) else ""
            return httpx.Response(200, json={"test_cases": items, "next_page_token": nxt})
        if m == "POST" and path == "/test-cases":
            body = json.loads(request.content)
            rec = {"id": f"new{len(self.cases[body['test_set_id']])}", **body}
            self.cases[body["test_set_id"]].append(rec)
            return httpx.Response(201, json={"test_case": rec})
        if m == "GET" and path.startswith("/test-sets/"):
            return httpx.Response(200, json={"test_set": {"id": path.split("/")[2], "display_name": "x"}})
        if m == "GET" and path.startswith("/agents/"):
            return httpx.Response(200, json={"agent": self.agents[path.split("/")[2]]})
        if m == "GET" and path.startswith("/personas/"):
            return httpx.Response(200, json={"persona": self.personas[path.split("/")[2]]})
        if m == "GET" and path.startswith("/metrics/"):
            return httpx.Response(200, json={"metric": self.metrics[path.split("/")[2]]})
        if m == "POST" and path.endswith("/duplicate"):
            src = self.agents[path.split("/")[2]]
            new = {**json.loads(json.dumps(src)), "id": f"ag{len(self.agents)}"}
            self.agents[new["id"]] = new
            return httpx.Response(201, json={"agent": new})
        if m == "PATCH" and path.startswith("/agents/"):
            body = json.loads(request.content)
            self.agents[path.split("/")[2]].update(body)
            self.patches.append((path.split("/")[2], body))
            return httpx.Response(200, json={"agent": self.agents[path.split("/")[2]]})
        if m == "POST" and path == "/runs":
            body = json.loads(request.content)
            rid = f"run{len(self.runs)}"
            self.launches.append(body)
            self.runs[rid] = {"run_id": rid, "status": "PENDING", "progress": {"total_test_cases": 0, "completed_test_cases": 0}}
            sims = []
            for tc in body["options"]["test_case_ids"]:
                for it in range(body["options"]["iteration_count"]):
                    status = "FAILED" if tc == "tc3" else "COMPLETED"
                    sims.append({"simulation_id": f"{rid}-{tc}-{it}", "simulation_output_id": f"{rid}-{tc}-{it}", "run_id": rid,
                                 "status": status, "test_case_id": tc, "agent_id": body["agent_id"]})
            self.sims[rid] = sims
            return httpx.Response(200, json={"run": self.runs[rid]})
        if m == "GET" and path.startswith("/runs/"):
            rid = path.split("/")[2]
            self.polls += 1
            run = self.runs[rid]
            run["status"] = "COMPLETED" if self.polls >= 2 else "IN PROGRESS"
            return httpx.Response(200, json={"run": run})
        if m == "GET" and path == "/conversations/simulated":
            rid = q["filter"].split("=", 1)[1]
            assert "include" not in q
            return httpx.Response(200, json={"simulated_conversations": self.sims[rid], "next_page_token": ""})
        if m == "GET" and path.endswith("/metrics") and path.startswith("/conversations/simulated/"):
            sid = path.split("/")[3]
            rid, tc, it = sid.rsplit("-", 2)
            score = self.scores.get((tc, int(it)), 1.0)
            return httpx.Response(200, json={"metrics": [
                {"metric_output_id": "o1", "metric_id": "mA", "metric_name": "task completed", "value": score, "status": "COMPLETED",
                 "explanation": "The agent never asked for a phone number." if score == 0 else "All behaviours present."},
                {"metric_output_id": "o2", "metric_id": "mZ", "metric_name": "unrelated", "value": 3, "status": "COMPLETED", "explanation": "ignored"},
                {"metric_output_id": "o3", "metric_id": "mB", "metric_name": "no price", "value": True, "status": "COMPLETED", "explanation": None},
            ], "next_page_token": ""})
        return httpx.Response(404, text=f"no route {m} {path}")
