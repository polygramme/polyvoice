"""Coval as the environment and the verifier for a polyloop cycle: `VoiceEnvironment` with a `CovalVerifier`
and one `ProxyTarget` per endpoint. Registered as `coval` (entry point `polyloop.environments`).

loop.yaml:

    environment:
      kind: coval
      options:
        agent_id: ...            # a MODEL_TYPE_CHAT agent; its config is copied per endpoint
        persona_id: ...
        metric_ids: [...]        # reward = their mean; binary judges score 0/1
        public_url: https://...  # tunnel to the proxy at proxy_url (Coval refuses http and private IPs)
        candidate:               # optional second proxy
          public_url: https://...
          proxy_url: http://127.0.0.1:8788
        concurrency: 5
        check_public: false      # where the host cannot reach its own public URL (e.g. inside a Modal container)
"""
from __future__ import annotations

from typing import Any

from polyvoice.envs.voice import VoiceEnvironment
from polyvoice.targets.proxy import ProxyTarget
from polyvoice.verifiers.coval import CovalVerifier


class CovalEnvironment(VoiceEnvironment):
    name = "coval"

    def __init__(self, cfg, store, log=print, *, agent_id: str, persona_id: str, metric_ids: list[str], public_url: str,
                 proxy_url: str | None = None, candidate: dict | None = None, concurrency: int = 5, poll_seconds: int = 15,
                 run_timeout: int = 3600, temperature: float | None = None, max_tokens: int = 256, pass_value: float = 1.0,
                 hint_failures_only: bool = True, max_judge_chars: int = 1200, api_base: str = "https://api.coval.dev/v1",
                 api_key_env: str = "COVAL_API_KEY", ledger: str = "coval_ledger.jsonl", check_public: bool = True,
                 client: Any = None):
        temp = cfg.gate.temperature if temperature is None else temperature
        verifier = CovalVerifier(agent_id=agent_id, persona_id=persona_id, metric_ids=metric_ids, loop_name=cfg.name,
                                 store=store, log=log, concurrency=concurrency, poll_seconds=poll_seconds,
                                 run_timeout=run_timeout, temperature=temp, max_tokens=max_tokens,
                                 max_judge_chars=max_judge_chars, api_base=api_base, api_key_env=api_key_env, client=client)
        targets = {"default": ProxyTarget(proxy_url or cfg.proxy_url, public_url, check_public=check_public, log=log)}
        if candidate:
            targets["candidate"] = ProxyTarget(candidate.get("proxy_url") or cfg.proxy_url, candidate["public_url"],
                                               check_public=check_public, log=log)
        super().__init__(cfg, store, log=log, verifier=verifier, targets=targets, pass_value=pass_value,
                         hint_failures_only=hint_failures_only, ledger=ledger, temperature=temp, require_public=True)

    # test/ops conveniences kept from the first version
    @property
    def _client(self):
        return self.verifier._client

    @_client.setter
    def _client(self, c):
        self.verifier._client = c

    @property
    def client(self):
        return self.verifier.client

    @property
    def proxies(self) -> dict[str, dict]:
        return {role: {"public_url": t.public_url, "proxy_url": t.proxy_url} for role, t in self.targets.items()}
