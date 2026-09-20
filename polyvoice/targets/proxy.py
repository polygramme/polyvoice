"""The polyloop proxy as a target: a transient `/admin/serve` override binds the policy's sampler path,
`/admin/session/<id>/done` flushes a session's last turn, and `public_url` is the https tunnel an external
simulator dials (Coval refuses http and private addresses)."""
from __future__ import annotations

import httpx


class ProxyError(RuntimeError):
    pass


class ProxyTarget:
    def __init__(self, proxy_url: str | None, public_url: str | None = None, *, check_public: bool = True,
                 log=print):
        self.proxy_url, self.public_url, self.check_public, self.log = proxy_url, public_url, check_public, log

    def _post(self, path: str, body: dict, timeout: float = 600) -> dict:
        r = httpx.post(self.proxy_url.rstrip("/") + path, json=body, timeout=timeout)
        if r.status_code >= 400:
            raise ProxyError(f"proxy {self.proxy_url}{path} -> {r.status_code} {r.text[:200]}")
        return r.json() if r.content else {}

    @property
    def chat_endpoint(self) -> str:
        base = (self.public_url or self.proxy_url).rstrip("/")
        return base + "/v1/chat/completions"

    @property
    def openai_base(self) -> str:
        return self.proxy_url.rstrip("/") + "/v1"

    def serve(self, policy: dict | None) -> None:
        self._post("/admin/serve", {"sampler_path": (policy or {}).get("sampler_path")})

    def clear(self) -> None:
        self._post("/admin/serve", {"clear": True})

    def finish_session(self, session: str) -> None:
        """External simulators never send X-Session-Done; flush the last turn so it is recorded."""
        try:
            httpx.post(self.proxy_url.rstrip("/") + f"/admin/session/{session}/done", timeout=10)
        except Exception:
            pass

    def check(self, *, require_public: bool = False) -> list[str]:
        problems = []
        if not self.proxy_url:
            problems.append("proxy_url unset")
        else:
            try:
                r = httpx.get(self.proxy_url.rstrip("/") + "/admin/status", timeout=10)
                if r.status_code >= 400:
                    problems.append(f"proxy {r.status_code} at {self.proxy_url}")
            except Exception as exc:
                problems.append(f"proxy unreachable at {self.proxy_url}: {exc}")
        if require_public:
            if not str(self.public_url).startswith("https://"):
                problems.append(f"public_url must be https (external simulators refuse http and private IPs): {self.public_url}")
            elif self.check_public:
                try:
                    r = httpx.get(self.public_url.rstrip("/") + "/healthz", timeout=15)
                    if r.status_code >= 400:
                        problems.append(f"public proxy {r.status_code} at {self.public_url}/healthz")
                except Exception as exc:
                    problems.append(f"public proxy unreachable at {self.public_url}: {exc}")
        return problems
