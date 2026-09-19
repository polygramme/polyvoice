# Drafts from the 2026-09-19 Pipecat implementation research

Written against pipecat `dbdf21a` (post-1.11.0) by reading the source; **not yet run**. They are the
starting point for `pipecat-polyvoice` (observer, LLM helper) and `polyvoice/verifiers/pipecat_evals.py`.
See `docs/PIPECAT-IMPL.md` for the facts each line relies on.

- `observer.py`        PolyvoiceObserver: one JSONL record per LLM request
- `llm.py`             PolyvoiceLLMService + polyvoice_llm(): X-Session-Id / X-Turn-Type headers
- `pipecat_evals.py`   PipecatEvalsVerifier: K text-mode simulations per scenario, one bot per rollout
- `canary_bot.py`      LLMSwitcher canary: incumbent vs candidate chosen per session by stable hash
- `otel_receiver.py`   OTLP/HTTP receiver mapping Pipecat `llm` spans to trace records
- `test_observer.py`   pytest using pipecat.tests.utils.run_test
