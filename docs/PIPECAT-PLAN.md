# polyvoice × Pipecat: how to generalise the trainer so it works with everyone

Research date 2026-09-19, against pipecat `dbdf21a` (1.8-era), pipecat-cli, pipecat-flows, pipecat-examples,
smart-turn, skills, the Roark observer SDK, the Pipecat docs tree, and the public docs of Coval, Cekura,
Bluejay, Roark and Arize. Line numbers refer to `src/pipecat/` unless a repo is named.

## 0. The one-paragraph answer

Pipecat separates the LLM from everything else by construction: every text LLM vendor except Anthropic,
Gemini and Bedrock is a subclass of one `OpenAILLMService` that takes a `base_url` and headers
(`services/openai/base_llm.py:150`). PhoneLLM's own example already runs that way, with the plain OpenAI
service pointed at a Modal endpoint (`pipecat-examples/phonellm/server/bot.py:139-151`). So polyvoice does
not integrate with 30 LLM vendors; it *is* the LLM vendor for a bot, and STT, TTS, transports, serializers,
memory and audio filters are untouched because they sit before or after the LLM in the same pipeline.
What has to be generalised is the other side: where judged conversations come from (Pipecat Evals locally,
Coval/Cekura/Bluejay/Roark remotely, production via an observer) and how a candidate adapter reaches the
bot under test. That is two small protocols, `Verifier` and `AgentTarget`, plus one observer class.

## 1. What Pipecat gives us (verified on code)

### 1.1 The LLM slot

| Fact | Where |
|---|---|
| `OpenAILLMService(api_key, base_url, default_headers, settings=Settings(model, temperature, extra={...}))` | `openai/base_llm.py:150-165, 254-287` |
| All of Fireworks, Together, Groq, NIM, Qwen, Baseten, OpenRouter, Cerebras, Mistral, DeepSeek, Perplexity, SambaNova, Sarvam, Novita, Nebius, Inception, Crusoe, xAI, Ollama, Azure subclass it and pass `**kwargs` through | `services/<vendor>/llm.py` |
| Per-request headers without subclassing: `Settings.extra` is merged into the `create()` kwargs, and the OpenAI SDK accepts `extra_headers`/`extra_body` there | `base_llm.py:383`, `services/settings.py:104` |
| Mid-call settings change (model name, `extra`) via `LLMUpdateSettingsFrame`; `base_url` is fixed at construction | `llm_service.py:709-729`, `base_llm.py:239-248` |
| Candidate vs incumbent in one bot: `LLMSwitcher` + `ManuallySwitchServiceFrame`; there is no A/B or shadow mechanism | `pipeline/llm_switcher.py:24`, `frames/frames.py:2554` |
| Wire format is OpenAI messages (`system`, `user`, `assistant` with `tool_calls`, `tool`, `developer`); tool results are `IN_PROGRESS` placeholders on the re-prompt, then `json.dumps(result)` | `llm_response_universal.py:1852-1878, 2018-2028` |
| Many vendors set `supports_developer_role = False`, so the adapter rewrites `developer` to `user`; the canonical bot sends a `developer` message to trigger the greeting | `adapters/services/open_ai_adapter.py:244-261` |
| Speech-to-speech services (OpenAI Realtime, Gemini Live, Nova Sonic, Ultravox, Grok, Inworld) consume audio frames directly; no text LLM hook | `openai/realtime/llm.py:443`, `google/gemini_live/llm.py:365`, ... |
| TTFB, processing time and token usage arrive as `MetricsFrame` when `PipelineParams(enable_metrics=True)` | `processors/metrics/frame_processor_metrics.py:107-341` |

### 1.2 Session identity

`runner_args.session_id` exists on every transport path (`runner/types.py:166`); Pipecat Cloud sets it to its
own session id, and the Roark observer resolves its call id the same way. `PipelineWorker(conversation_id=)`
only feeds tracing. The LLM service is constructed per session inside `bot()`, so the id can go into
`default_headers` with no framework change.

### 1.3 Observers and tracing (production capture without the proxy)

- `BaseObserver.on_push_frame(FramePushed)` sees every frame with a pipeline-clock timestamp
  (`observers/base_observer.py`). The frames that carry what a trainer needs:
  `LLMContextFrame.context.get_messages()` arriving at an `LLMService` (skip `speculation=True`),
  `LLMFullResponseStart/End` + `LLMTextFrame`, `FunctionCallInProgressFrame`/`FunctionCallResultFrame`
  (untruncated result), `InterruptionFrame`, `MetricsFrame`, `EndFrame.reason`/`CancelFrame.reason`.
  `worker.turn_tracking_observer` emits `on_turn_ended(n, duration, was_interrupted)`.
- Roark's community observer is the precedent: one class, subscribes to transcription/TTS/function-call/
  end frames, ships transcript + tool calls + audio to its API (`pipecat_roark/observer.py:377-500`).
  It does not capture the LLM context; we can.
- OpenTelemetry spans carry `input` (messages JSON), `output`, `tools`, `metrics.ttfb`, `gen_ai.usage.*`
  per LLM call under `conversation` → `turn` (`utils/tracing/service_attributes.py:186-275`). Good as a
  secondary source when a customer already exports; the observer is richer (end reason, untruncated tool
  results) and one line to wire.
- Pipecat Flows puts each node's prompt and tools into the same `LLMContext`; only the node name is missing
  and is readable from `flow_manager.current_node` at request time.

### 1.4 Pipecat Evals (the free sandbox)

- Library API: `EvalScenarioFile.load()`, `EvalSession.from_scenario(scenario, bot_url, params, persona_llm,
  judge)`, `await session.run()` → `EvalSimulationResult(succeeded, reason, metrics[score, passed, reason,
  verdicts[turn, passed, reason]], messages, ended_by, duration_ms, ...)` (`evals/results.py:212-283`).
- One WebSocket client per bot process (`transports/websocket/server.py:247-255`), no context reset between
  runs: **one bot process per rollout**. The suite spawns `python bot.py -t eval --port N`, N = base+index,
  and runs the harness in a subprocess. Env is inherited, so we can pass a session id per bot.
- Judge and persona are any OpenAI-compatible endpoint (`factory:` or `OpenAILLMService(base_url=)`);
  default is local Ollama `gemma4:12b`. Text mode: T persona calls + 1 judge call per rollout. Audio mode
  adds Kokoro TTS for the persona and Moonshine STT on the bot's speech, both local CPU.
- Judge reasons exist only for turns judged **no** (`evals/judge.py:135-137`): that is exactly the
  hindsight-hint shape we already use from Coval.
- Scripted scenarios give deterministic rewards: function-call name + args subset match, `within_ms`
  latency budgets, `text_contains/excludes`, `absent` (`evals/script.py:336-387`).
- `pipecat eval suite` writes `results.jsonl`, one record per run, with messages, metrics, verdicts, reason.

### 1.5 Partners (Pipecat docs "Evals → Third-party platforms")

| | Attach to bot | Run batch on demand | Per-conversation API output | Trains weights |
|---|---|---|---|---|
| Coval | external caller (Pipecat Cloud or WebSocket); upload for monitoring | yes, `iteration_count` ≤ 50, `concurrency` ≤ 100 | messages, tool calls, metric score + explanation, audio, latency, end reason | no |
| Cekura | external caller (Pipecat Cloud automated, or a Daily room per run); upload `/observe/` | yes, k via `frequency`; concurrency not exposed | transcript, typed metrics with score + explanation, audio; tool calls undocumented | no |
| Bluejay | Pipecat Direct, phone, or CHIRP WebSocket (own protocol); OTLP; `/v1/evaluate` upload | yes, `runs_per_digital_human`, `max_concurrent` ≤ 500; has `agent_version_id` | transcript, typed metrics + reasoning, latency percentiles, tool calls expected vs actual | no |
| Roark | **in-pipeline observer** + OTel + simulated caller | yes, `iterationCount` ≤ 10000, `maxConcurrentJobs`; 10–25 account lines | transcript with offsets, tool invocations, metrics with reasoning + confidence, `endedStatus`, audio | no (Autoimprove iterates prompts) |
| Arize / Phoenix | OTel only | no simulator | spans with `eval.*` labels; Phoenix exports fine-tuning JSONL | no |

No vendor emits token ids or the exact LLM context; every one re-renders from transcript text. The proxy
stays the only source of on-policy tokens; vendors supply `conversation_id → judged metrics`.

### 1.6 Ecosystem conventions for a trained component

- Smart Turn is the one trained primitive: open data on HF under `pipecat-ai/*-data-vX-train/test`, a Modal
  training script, a per-language/per-dataset Markdown benchmark, `Local*/Http*` class pair, weights bundled
  in the wheel, and an env flag (`PIPECAT_SMART_TURN_LOG_DATA=1`) that captures production inputs for the
  next training round. An explicit data-licence clause in the contribution guide.
- PhoneLLM is the one BYO-LLM precedent: HF weights → Modal OpenAI-compatible endpoint →
  `OpenAILLMService(base_url=)`, a shipped `evals/manifest.yaml`, `pcc-deploy.toml`, and a
  `.claude/skills/setup/SKILL.md`.
- Community packages are `pipecat-<vendor>` with a single-file example, README "Tested with Pipecat vX",
  BSD-2, a docs PR with the community badge, and a `#community-integrations` post. There is no training
  category; the evals "platforms" page and the examples page are the two homes.
- Pipecat Cloud has no GPU profiles; the model server lives elsewhere (Modal, Cerebrium, own box). A secret
  change needs `pipecat cloud deploy --force`; there is no canary or weighted rollout.

## 2. Target structure for polyvoice

```
polyvoice/
  core/
    types.py        Conversation, Message, ToolCall, Metric, Scenario, AgentTarget   (§1.5 shapes)
    rewards.py      reward_from_metrics(), hints_from_metrics()  — vendor-agnostic (today: coval/rewards.py)
    ledger.py       append/read judged conversations keyed by session id; holdout exclusion
  verifiers/        one module per source of judged conversations
    base.py         Judge (submit/fetch) and Simulator (list/create scenarios, run, status, results) protocols
    pipecat_evals.py  local: spawns `bot.py -t eval` per rollout, EvalSession, results → Conversation
    coval.py        today's client + rewards, reshaped to the protocol
    cekura.py  bluejay.py  roark.py     simulators (REST)         — after the first two
    arize.py  mlflow.py                 judges over uploaded traces — later
  targets/          how a verifier reaches the bot, and how a policy is bound to it
    proxy.py        polyloop proxy: public URL + /admin/serve slot override (today's behaviour)
    pipecat_cloud.py  agent name + API key; bind = secret update + `deploy --force`; version = image tag
    local_bot.py    subprocess `bot.py -t eval --port N` with POLYVOICE_* env (for pipecat_evals)
  traces/           where daytime conversations come from
    proxy.py        token-exact (rlcli/polyloop proxy JSONL) — unchanged
    observer.py     text-level from PolyvoiceObserver (§3.2), same JSONL schema, `token_exact: false`
    otel.py         OTLP receiver → same schema (for teams that already export)
  pipecat/          the light, pip-installable half: `pipecat-polyvoice`
    llm.py          polyvoice_llm(session_id, ...) -> OpenAILLMService, PolyvoiceLLMService (speculation → X-Turn-Type)
    observer.py     PolyvoiceObserver(BaseObserver)
    switcher.py     candidate/incumbent LLMSwitcher helper for in-bot A/B
  envs/
    voice.py        VoiceEnvironment(verifier, target, policy_binder) — the ONE polyloop Environment
  cli.py            account / check / seed / ledger / scenarios convert
recipes/<name>/     loop.yaml, bot.py, system.md, evals/{pool,holdout}/*.yaml, pcc-deploy.toml
.claude/skills/polyvoice/SKILL.md
```

Rules that fall out of the research:

1. **One environment, many verifiers.** `CovalEnvironment` today mixes three concerns: driving Coval,
   turning metrics into rewards and hints, and flipping the proxy slot. Split them. `VoiceEnvironment`
   implements polyloop's `load_tasks / run_rollouts / preflight / session_hints / excluded_sessions`; the
   verifier supplies scenarios and judged conversations; the target binds the policy.
2. **Scenario portability.** Pipecat's simulated-scenario YAML (persona, goal, success, metrics with
   criteria) is the richest open format and every simulator can be seeded from it (Coval test cases, Bluejay
   `/v1/scenario`, Roark flows + personas; Cekura only through MCP today). `polyvoice scenarios convert`
   makes the recipe's `evals/` directory the single source of truth for pool and holdout.
3. **Two packages.** `pipecat-polyvoice` (observer, LLM helper, switcher; depends only on pipecat) so any
   bot can capture traces and point at a proxy without installing a trainer. `polyvoice` (verifiers,
   targets, environment; depends on polyloop) runs the loop.
4. **Session id is the join key everywhere.** `runner_args.session_id` → `X-Session-Id` header →
   `PipelineWorker(conversation_id=)` → observer records → verifier `Conversation.id` where the vendor
   exposes it (Roark `pipecatCallId`; Coval sends its simulation id as `X-Session-Id` already). Where a
   vendor cannot carry it (Cekura, Bluejay self-hosted), join by run + scenario + time and mark the record.
5. **Speculative inferences are not turns.** The universal aggregator issues `LLMContextFrame(speculation=
   True)`; those requests must carry `X-Turn-Type: speculative` and be dropped from training rows.

## 3. The Pipecat surface

### 3.1 `bot.py` in a recipe

```python
import os, uuid
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.runner import WorkerRunner
from polyvoice.pipecat import PolyvoiceObserver, polyvoice_llm

transport_params = {
    "eval":   lambda: EvalTransportParams(audio_in_enabled=True, audio_out_enabled=True),
    "daily":  lambda: DailyParams(audio_in_enabled=True, audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}

async def bot(runner_args: RunnerArguments):
    session_id = runner_args.session_id or str(uuid.uuid4())
    transport = await create_transport(runner_args, transport_params)
    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    tts = CartesiaTTSService(api_key=os.environ["CARTESIA_API_KEY"])
    llm = polyvoice_llm(session_id, system_instruction=open("system.md").read())
    #   = OpenAILLMService(base_url=POLYVOICE_PROXY_URL, default_headers={"X-Session-Id": session_id},
    #                      settings=Settings(model=POLYVOICE_MODEL, temperature=..., extra=...))
    context = LLMContext()
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()))
    pipeline = Pipeline([transport.input(), stt, user_agg, llm, tts, transport.output(), assistant_agg])
    worker = PipelineWorker(pipeline, params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
                            conversation_id=session_id,
                            observers=[PolyvoiceObserver(session_id, sink=os.getenv("POLYVOICE_TRACE_SINK"))],
                            idle_timeout_secs=runner_args.pipeline_idle_timeout_secs)
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        context.add_message({"role": "developer", "content": "Greet the caller briefly."})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await runner.cancel(reason="client-disconnected")

    await runner.run()

if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
```

The same file serves three jobs: `python bot.py -t eval` for local rollouts and the gate, `-t daily` or
`-t webrtc` in production, and `pipecat cloud deploy` unchanged. Nothing in it is polyvoice-specific except
the two imports; a team with an existing bot changes two lines (the LLM constructor and the observer).

Other ways in, for bots we do not own:

| Way in | What it gives | Cost to the team |
|---|---|---|
| `polyvoice_llm()` / `base_url` | trained adapter answers the calls; token-exact traces via the proxy | 1 line |
| `PolyvoiceObserver` | text-level traces (messages, tool calls, interruptions, TTFB, end reason) from any bot, any LLM vendor | 1 line |
| OTLP receiver | traces from bots that already export OpenTelemetry (Arize, MLflow, Langfuse users) | config |
| `-t eval` | the bot is already a Pipecat Evals target; our sandbox needs nothing more | 0 |
| `LLMSwitcher` helper | candidate and incumbent in one bot; switch per session by hash for a canary | a few lines |
| Pipecat Cloud target | verifier reaches the deployed bot by agent name; promote = secret + `deploy --force` | config |

### 3.2 `PolyvoiceObserver`

Subscribes to `LLMContextFrame` (when the destination is an `LLMService` and `speculation` is false: the
exact messages sent), `LLMFullResponseStart/End` + `LLMTextFrame` (the reply), `FunctionCallInProgressFrame` /
`FunctionCallResultFrame` (tool calls with untruncated results), `InterruptionFrame`, `MetricsFrame`
(`TTFBMetricsData` for the LLM processor), `EndFrame` / `CancelFrame` (`reason`), and hooks
`worker.turn_tracking_observer.on_turn_ended` for `was_interrupted`. Writes one JSONL record per turn in the
polyloop trace schema with `token_exact: false`, to a file or to the trainer's `/traces` endpoint. Reads
`flow_manager.current_node` when Flows is present.

### 3.3 Which Pipecat services polyvoice covers

| Row in the Pipecat services table | Status | Why |
|---|---|---|
| STT (AssemblyAI, Deepgram, ElevenLabs, Gradium, Whisper, ...) | unchanged | upstream of the LLM; the trained model never sees audio |
| TTS (Cartesia, ElevenLabs, Gradium, Inworld, Kokoro, ...) | unchanged | downstream; only the reply text changes |
| Transport, Serializers, Audio processing, Memory, Vision, Video | unchanged | orthogonal to the LLM slot |
| LLM: OpenAI-compatible vendors (Fireworks, Together, Groq, NIM, Baseten, Ollama, ...) | **serving hosts** | any of them that serves the trained adapter (vLLM `--lora-modules`, Fireworks multi-LoRA, NIM, Ollama for merged GGUF) is a valid `base_url`; the loop's sampler is one of them |
| LLM: Anthropic, Gemini, Bedrock (native SDK) | teacher or incumbent only | closed weights; can be the OPSD teacher or the "large model + same prompt" arm of the gate, never the trainee |
| Speech-to-speech (OpenAI Realtime, Gemini Live, Nova Sonic, Ultravox, Grok) | out of scope | audio-native, no text LLM hook |
| Turn detection (Smart Turn), VAD | unchanged, and the precedent | the ecosystem's trained-primitive conventions come from here |
| Analytics (Roark, Arize, MLflow, Noveum, Future AGI, Sentry, Finchvox) | trace sources / judges | observer or OTel; none simulate |

ElevenLabs Agents (the platform, not the Pipecat TTS row) has a custom-LLM slot that is the same
`base_url` contract, so the trained endpoint plugs in there too; its traces would need its own capture path.

## 4. How the tiers fit

```
tier 0  Pipecat Evals, text mode, local judge        thousands of rollouts/night, $0     → RL/OPSD sampling, filter
tier 1  Pipecat Evals, audio mode, local             hundreds                            → interruption/turn-taking + latency regressions
tier 2  Coval / Cekura / Bluejay / Roark simulator   K × holdout, per-minute billing      → the gate: stronger judges, audio + latency metrics, persona variety
tier 3  production, observer or proxy                 whatever the day brings             → hints, failures → new scenarios, canary
```

Tier 0 answers the "training needs thousands of rollouts" problem in docs/PLAN.md. Tier 2 stays the
promotion decision because a 12B local judge is not the judge we want to promote on. Tier 3 is where the
observer matters: teams that already run Pipecat get the daytime half without changing their LLM vendor.

One bot process per rollout means concurrency is bounded by CPU and RAM per bot (Silero VAD, aggregators,
no audio in text mode) and by the persona/judge server's parallelism (`OLLAMA_NUM_PARALLEL`, or a vLLM
judge). Budget: 8 turns × 1 persona call + 1 judge call per rollout.

## 5. Build order

| Step | What | Size |
|---|---|---|
| 1 | `core/types.py`, `core/rewards.py`, `core/ledger.py`; `verifiers/coval.py` reshaped; `envs/voice.py` replaces `envs/coval.py` (tests keep passing against `fake_coval`) | 1 day |
| 2 | `verifiers/pipecat_evals.py` + `targets/local_bot.py`; dental recipe gets `bot.py` and `evals/{pool,holdout}/*.yaml`; `polyvoice scenarios convert` (Pipecat YAML → Coval test cases) | 2 days |
| 3 | `pipecat-polyvoice`: `polyvoice_llm()`, `PolyvoiceObserver`, `traces/observer.py`; example bot; README "Tested with Pipecat vX" | 1–2 days |
| 4 | Full cycle on Modal: tier 0 rollouts → OPSD → tier 2 Coval gate; RESULTS.md | 1 night |
| 5 | Docs PR to `pipecat-ai/docs` evals platforms page + examples page; `.claude/skills/polyvoice/SKILL.md` in the PhoneLLM-example style; Discord post | 1 day |
| 6 | `verifiers/roark.py` (observer-native, clean join key), then Bluejay (`agent_version_id`), then Cekura | 1 day each |

## 6. Open questions (what to answer before calling this "works with everyone")

Rewards and judges
1. Does the local judge agree with Coval's? Run gemma4:12b (and a hosted judge via `factory:`) over the 162
   judged dental transcripts we already have and report agreement, as we did for Jev.
2. Does a text-mode win survive audio? Pipecat Evals audio mode is free; gate the PhoneLLM adapter in both
   and report the delta. If text-mode rewards are not predictive, tier 0 is only good for sampling.
3. How much of the reward can be deterministic? Scripted scenarios give tool-call argument matches and
   `within_ms` budgets; measure what fraction of dental-style scenarios can be scored without a judge.
4. Tools in the sandbox: bots call real functions. Rollouts need a mock tool layer with seeded state
   (τ-bench style DB diff) so tool discipline is verifiable, not judged.

Traces and identity
5. Does Coval's Pipecat Cloud connection expose the Pipecat session id (Roark exposes `pipecatCallId`)? If
   not, the Coval join stays run + scenario + time.
6. How do speculative inferences show up at the proxy today, and are any in our OPSD rows?
7. `IN_PROGRESS` tool placeholders and `developer` messages: does our renderer tokenise the Pipecat context
   the way the sampler saw it? Add a logprob-replay check on a Pipecat-captured session.
8. Flows: is the node name worth a column in the trace, and should hints be scoped per node?

Serving and promotion
9. Where does the adapter get served for teams that do not run our proxy? Candidates: vLLM `--lora-modules`,
   Fireworks multi-LoRA, NIM, a merged checkpoint on Ollama/Baseten. Decide the "serving backend" contract
   and which ones `promote` can flip without a redeploy.
10. Pipecat Cloud promotion needs `deploy --force` after a secret change. Recommend "stable proxy URL, flip
    the adapter server-side" and document the fallback.
11. Canary: no weighted rollout in Pipecat Cloud. Per-session hash in the proxy or `LLMSwitcher` in the bot?
12. Throughput: how many `bot.py -t eval` processes fit on one CPU box, and what judge server keeps up with
    64 concurrent rollouts?

Ecosystem
13. Which docs home does the Pipecat team want: evals platforms page, examples page, or both? Ask in
    `#community-integrations` before the PR.
14. Package split: is `pipecat-polyvoice` (observer + LLM helper) worth publishing before the trainer is
    generic, as the low-friction door?
15. Licence for distributed adapters on top of PhoneLLM (BSD-2 + NVIDIA open model licence) and a data
    clause for contributed scenarios and traces, following smart-turn's contribution guide.
16. Should the recipe's scenarios live in Pipecat's YAML format only, with vendors seeded from it, or do we
    also import vendor-native test sets? (Recommendation: Pipecat YAML is canonical.)
17. Cost of the gate per vendor at K=3 over 32 holdout scenarios: Coval 100-way concurrency vs Roark 10–25
    lines vs Bluejay 500. Which one is the default gate in the docs?
