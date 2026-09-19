# polyvoice × Pipecat: implementation details

Companion to `PIPECAT-PLAN.md`. Verified on pipecat `dbdf21a` (post-1.11.0, 2026-09-19), pipecat-flows,
pipecat-examples, the Roark observer SDK, PyPI, Modal docs and the vendor docs of Coval, Cekura, Bluejay and
Roark. Paths are `src/pipecat/…` unless a repo is named. Draft code here is written against that source and
has not been run; the drafts live in `docs/drafts/pipecat/` and the first coding step is to run their tests.

## 0. Is this a me-too of the PhoneLLM example?

No, and the research makes the line sharp. The PhoneLLM example is an inference-plus-eval repo. Daily's
Modal endpoint is a *dedicated inference endpoint* for fixed weights (`modal endpoint create --model
pipecat-ai/phonellm-alpha-1`); nothing in that repo or in Modal's endpoint product trains anything.

| | PhoneLLM example | polyvoice |
|---|---|---|
| Weights | ships (BF16 + NVFP4, BSD-2 over the Nemotron licence) | starts from theirs |
| Serving | Modal dedicated endpoint, `OpenAILLMService(base_url=)` | any OpenAI-compatible server that can hot-load a LoRA; ships the polyloop proxy + sampler |
| Evals | 3 scenario YAMLs + manifest, local gemma judge, pass/fail exit code | same YAML format as pool and holdout; judge reasons become training hints |
| Benchmark harness | "will be published as PhoneBench matures" | n/a; adopts PhoneBench as a gate source if released |
| Training code / data | not published; "full-parameter SFT with NeMo" | open: hinted on-policy self-distillation on captured sessions |
| Per-customer adaptation | services: "The Pipecat team works directly with enterprise customers to train models for specific use cases" | self-serve, per-deployment adapter |
| Production capture | `observers=[]` in their bot | `PolyvoiceObserver` writes training rows |
| Gate / promotion | none ("no baseline comparison mechanism") | paired held-out gate, bootstrap CI, promote / rollback |
| Adapter lineage, receipts | no such object | `lineage.json`, `receipts/` |

Daily's own model card says the loop is the future: "most production agents will continually improve, using
feedback loops built around targeted evals and production instrumentation" and "updating model weights
every month (or even more often) is now a viable strategy", followed by "come talk to us". Every
improvement loop that exists in the Pipecat partner list (Roark Autoimprove, Cekura's self-improving-agent
skill, Coval's skills) edits prompts and configs, not weights. Nobody in the ecosystem has the adapter, the
gate, or the promote verb. The endpoint is the only shared surface and it is a commodity.

Two facts sharpen the serving story. Modal dedicated endpoints accept custom weights only as a full HF
repo or volume; LoRA is not mentioned anywhere in their docs. Fireworks allows LoRA only on dedicated
deployments and lists base models, and Nemotron 3 Nano is not among them. So "promote a LoRA without a
redeploy" is itself a capability nobody hosted provides today.

The risk is real but bounded: Daily has the model, an unpublished harness with calibrated judges, a
training stack and Modal. Their likely shape is a full-parameter fine-tune SKU delivered by their team,
served as merged weights. polyvoice stays complementary by starting from their weights, using their scenario
format, and shipping an adapter back into their bot; if they publish PhoneBench, it becomes the default gate.

What must be visibly different in the recipe, so nobody reads it as a copy:

- README line one names the object: "trains a LoRA on your Pipecat bot's LLM from judged calls and
  promotes it only when it beats the incumbent on a held-out set." Not a model description.
- First screen is a receipt (incumbent vs candidate, delta, CI, verdict, adapter and parent hashes, row
  count, hint source), not `modal endpoint create`. Commit the PhoneLLM cycle-1 receipt as the example.
- Directory shape: `bot/` unchanged from their layout, plus `capture/`, `train/`, `gate/` (with
  `evals/holdout/` the loop is forbidden to read), `adapters/` with lineage and receipts.
- Skills: `/setup` does their steps then installs the observer, runs the incumbent baseline over the
  holdout and registers it in lineage; `/cycle`, `/promote`, `/rollback` are verbs their repo does not have.
- Credit them explicitly: "starts from pipecat-ai/phonellm-alpha-1; bot layout follows
  pipecat-examples/phonellm".

## 1. Versions, extras, venvs

- Latest `pipecat-ai` is 1.11.0 (2026-09-18); releases every one to three weeks. The CLI now ships inside
  `pipecat-ai[cli]`; the standalone `pipecat-ai-cli` package is legacy (last 1.3.0).
- The extra is **`[evals]`**, plural: `pipecat-ai[cli]` + `[kokoro]` + `[moonshine]`. A text-mode
  simulation needs only the base package (the harness imports Silero VAD, which uses the base
  `onnxruntime`; Kokoro, Moonshine and Whisper are imported lazily).
- Pins: `pipecat-polyvoice` needs `pipecat-ai[openai]>=1.9.0,<2` (`LLMContextFrame.speculation` arrived in
  1.9.0; the PhoneLLM example itself floors at 1.8.1). The trainer's verifier needs
  `pipecat-ai>=1.11.0,<2` (`EvalScenarioFile.load`, `scenarios:` files, `runner_body: {path:}`).
- No torch in the base package. Friction with a trainer venv is `numba>=0.61.2` + llvmlite (vLLM pins
  numba exactly in some releases) and the CPU audio stack. **Run the harness in its own uv venv.** The
  `_session_subprocess` worker is `python -m pipecat.evals._session_subprocess cfg.json`, so the trainer can
  drive it with the harness interpreter; `pipecat eval suite --python` only substitutes the *bot*
  interpreter, the worker uses `sys.executable`.
- Deprecated and to avoid: `PipelineTask`/`PipelineRunner` (1.3.0, removed in 2.0), `OpenAILLMService(model=,
  params=)` (0.0.105), a leading `system` message inside `LLMContext` (1.9.0; use
  `Settings.system_instruction`), `service: openai` in judge/simulator blocks (1.8.0; use `factory:`).

## 2. The LLM slot, exactly

- `OpenAILLMService(*, api_key, base_url, default_headers: Mapping[str,str] | None, settings=Settings(...),
  retry_timeout_secs=5.0, retry_on_timeout=False)`; `default_headers` goes straight into
  `AsyncOpenAI(default_headers=)` (`services/openai/base_llm.py:150-287`).
- `Settings` fields: `model, temperature, max_tokens, max_completion_tokens, top_p, seed, system_instruction,
  extra: dict, …`. `build_chat_completion_params()` ends with `params.update(self._settings.extra)` and the
  dict is passed verbatim to `chat.completions.create(**params)`, with no key filtering (`:351-386`). So
  `extra={"extra_headers": {...}, "extra_body": {...}}` reaches the SDK, and the SDK honours
  `extra_headers` on streaming calls.
- Mid-call: `LLMUpdateSettingsFrame(delta=Settings(extra={"extra_headers": {...}}))` merges `extra`
  key-by-key and persists until overwritten. Nothing is turn-scoped, hence the subclass below.
- `speculation` is only visible on the `LLMContextFrame` in `process_frame`; `_process_context(context)`
  and `build_chat_completion_params(params_from_context)` never see it. Capture it in a `process_frame`
  override before calling `super()`. Processing is sequential per processor, so an instance attribute is
  safe.
- Vendors set `supports_developer_role = False` and the adapter rewrites `developer` → `user`. The proxy
  should accept `developer` as-is, and the renderer must tokenise it the way the sampler will see it.

```python
# polyvoice/pipecat/llm.py
from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.llm import OpenAILLMService


class PolyvoiceLLMService(OpenAILLMService):
    """OpenAILLMService that tags every request with X-Session-Id and X-Turn-Type: main|speculative."""

    def __init__(self, *, session_id: str, **kwargs):
        super().__init__(**kwargs)
        self._session_id, self._turn_type = session_id, "main"

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, LLMContextFrame):            # before base_llm consumes the flag
            self._turn_type = "speculative" if frame.speculation else "main"
        await super().process_frame(frame, direction)

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        params = super().build_chat_completion_params(params_from_context)
        headers = dict(params.get("extra_headers") or {})
        headers["X-Turn-Type"] = self._turn_type
        headers.setdefault("X-Session-Id", self._session_id)
        params["extra_headers"] = headers
        return params


def polyvoice_llm(session_id: str, *, base_url: str, model: str, system_instruction: str | None = None,
                  temperature: float | None = None, api_key: str = "polyvoice") -> OpenAILLMService:
    settings = PolyvoiceLLMService.Settings(model=model)
    if system_instruction is not None:
        settings.system_instruction = system_instruction
    if temperature is not None:
        settings.temperature = temperature
    return PolyvoiceLLMService(session_id=session_id, base_url=base_url, api_key=api_key, settings=settings,
                               default_headers={"X-Session-Id": session_id})
```

## 3. The observer, exactly

Facts that shape it:

- `BaseObserver` hooks are all async: `on_push_frame(FramePushed(source, destination, frame, direction,
  timestamp_ns))`, `on_process_frame`, `on_processor_setup`, `on_startup_warmup`, `on_pipeline_started()`
  (no args). `BaseObject` gives `setup(task_manager)`, `cleanup()`, `create_task`, `cancel_task`. The worker
  calls `setup` and `cleanup` for each observer.
- Callbacks run on a per-observer queue and task, lagging the pipeline and never blocking it. **An
  exception inside a callback is logged and silently kills that observer for the rest of the session.**
  Wrap every handler.
- The observer cannot reach the worker; bind it after construction (`observer.bind(worker)`) to hook
  `worker.turn_tracking_observer.add_event_handler("on_turn_ended", fn(observer, n, duration_s,
  was_interrupted))`.
- Request capture: `LLMContextFrame(context, speculation=False)` arriving at an `LLMService` destination.
  The assistant aggregator re-runs the LLM after tool results by pushing the frame **upstream**, so accept
  both directions, dedupe on `frame.id`. `context.get_messages(truncate_large_values=True)` returns deep
  copies with base64 media replaced; text is intact. There is no `system` on the context: the system prompt
  is `Settings.system_instruction`, prepended by the adapter. Tools: `context.tools.standard_tools →
  FunctionSchema.to_default_dict()`.
- Reply capture: `LLMFullResponseStartFrame`, `LLMTextFrame.text`, `LLMFullResponseEndFrame`, all pushed
  DOWNSTREAM by the LLM; filter `source is llm` to avoid double counting as downstream processors forward
  them. They carry no model name; take it from `MetricsFrame`.
- Tool calls: `FunctionCallsStartedFrame`, `FunctionCallInProgressFrame(function_name, tool_call_id,
  arguments, cancel_on_interruption, group_id)`, `FunctionCallResultFrame(…, result, error, properties)`
  are **broadcast in both directions as two instances with different ids**; count `source is llm and
  direction == DOWNSTREAM`.
- Speculation: during a speculative turn the LLM's frames are held by a gate and released on
  `UserStoppedSpeakingFrame`, or dropped on `EagerEndOfTurnCancelFrame` / `InterruptionFrame`. A
  speculative turn may therefore produce no reply frames; mark the record `discarded`.
- Metrics need `PipelineParams(enable_metrics=True, enable_usage_metrics=True)`. `MetricsFrame.data` items:
  `TTFBMetricsData(processor="OpenAILLMService#0", model, value_s)`, `ProcessingMetricsData`,
  `LLMUsageMetricsData(value=LLMTokenUsage(...))`.
- End: `EndFrame(reason)`, `CancelFrame(reason)`, `StopFrame`. `runner.cancel(reason=)` sets the reason;
  idle timeout gives the exact string `"idle timeout"`; fatal errors `"fatal error: …"`.
- Flows: no frame carries the node name. `flow_manager.current_node` (a property, the name string) is
  updated *after* the context frames are queued, so read it at `LLMContextFrame` time through a
  late-bound getter (`lambda: holder["fm"].current_node`), since the manager needs the worker and the
  observer must exist before the worker.

Record shape (one JSONL line per LLM request):

```
session_id, turn, speculation, discarded, messages, tools, tool_choice, reply, tool_calls[{id, name,
arguments, result, error}], was_interrupted, llm_ttfb_s, usage, model, node, end_reason, started_at, ended_at
```

Draft class: `PolyvoiceObserver(BaseObserver)` with `__init__(*, session_id, path, llm=None, node_getter=None)`,
a writer task started in `on_pipeline_started` via `self.create_task`, `cleanup()` that finishes the open
turn, drains the queue, cancels the writer and calls `super().cleanup()`, and `on_push_frame` that
delegates to `_handle` inside try/except. The full draft is `docs/drafts/pipecat/observer.py` and
becomes the first file of `pipecat-polyvoice`.

Test: `pipecat.tests.utils.run_test(processor, frames_to_send=[LLMContextFrame(...), SleepFrame(0.1)],
expected_down_frames=[...], observers=[obs])` with a `FakeLLM(FrameProcessor)` that consumes the context
frame and pushes start / text / broadcast tool call and result / end. Assert one record with the expected
messages, reply, tool call and `end_reason == "end"`. For the LLM subclass, `patch.object(PolyvoiceLLMService,
"create_client")`, set `_client = AsyncMock()`, stub `_process_context`, push a normal and a speculative
`LLMContextFrame`, assert headers `["main", "speculative"]`. Pipecat's tests use `unittest.
IsolatedAsyncioTestCase` or `@pytest.mark.asyncio`; their `conftest.py` stubs `dotenv.load_dotenv`.

## 4. Pipecat Evals as the verifier, exactly

Signatures (from `evals/`):

- `EvalScenarioFile.load(path) -> EvalScenarioFile(name, path, scenarios)`; iterable; `file["<file>/<name>"]`.
  `!include` resolves relative to the including file; file-level keys are defaults merged per entry by
  whole-value replacement. `is_scenario_file(path)` skips fragments. Kind: `persona:` → simulation,
  `turns:` → script.
- `EvalSimulationScenario(name, persona, goal, success, simulator={}, metrics=[], judge=_DEFAULT_JUDGE,
  bot_audio=False, transcriber=None, user_audio=False, user_speech=None, max_turns=20, max_duration_s=300,
  max_silence_s=30, runs=1, trigger_disconnect=False)`. Metrics: `EvalSimulationMetric(name, criterion,
  min_score, measure in {turns, duration, words, latency, function_calls}, min_value, max_value, calls)`.
- `EvalSession.from_scenario(scenario, bot_url, *, params=None, persona_llm=None, judge=None, user_tts=None,
  bot_stt=None)`; `await session.run() -> EvalSimulationResult | EvalScriptResult`. Connect failure and a
  missing `bot-ready` (10 s) come back as results with `error`, never exceptions.
- `EvalSessionParams(connect_timeout_s=5.0, default_timeout_ms=60000, record_path=None, cache_dir=None,
  use_cache=True, stop_bot=False, trigger_disconnect=False)`.
- `EvalJudge(service: LLMService, *, max_tokens=200)`; `EvalJudge.from_config(dict)`. Judge and simulator
  config dicts accept `factory` (dotted path to a callable taking the dict and returning an
  OpenAI-compatible service), `service: ollama`, `model`, `endpoint`, `extra`. **There is no `api_key`
  key**, so any keyed provider goes through `factory:` or by passing `judge=` / `persona_llm=` objects.
  Persona LLM must support tool calling (`end_call`).
- `EvalSimulationResult(simulation_name, succeeded, reason, error, metrics[EvalSimulationMetricScore(name,
  score, passed, reason, min_score, verdicts[EvalSimulationTurnVerdict(turn, passed, reason, verdict)],
  value, failure_kind)], messages[{role, content}], turns, ended_by, end_call, duration_ms, events_seen,
  debug_log)`; `.passed`, `.failure`. Tool calls are not in `messages`; they are `events_seen` entries of
  type `function_call`. Per-turn reasons exist only for turns judged no.
- One WebSocket client per bot process; no context reset between runs. **One bot process per rollout.**
  Spawn `python bot.py -t eval --host H --port N [--runner-body file]`; the "Bot ready!" print is not a
  readiness signal, the harness polls TCP then waits for RTVI `bot-ready`. `stop_bot=True` sends
  `eval-cancel`, which the bot turns into `CancelWorkerFrame` and exits; then terminate/kill after 10 s.
  Port in use raises inside the transport task and the bot does not exit; pre-check with a socket bind.
- Concurrency: no shared state that breaks N in-process text sessions. Judge cache is in-memory per
  `EvalJudge` instance, keyed on criterion + rendered transcript, so build one judge per rollout and
  repeats never collide. Set `Settings(temperature=0.0, seed=…)` for the judge; defaults are NOT_GIVEN.
  Suite defaults: base port 7900, connect timeout 60 s, worker cap 600 s, `OMP_NUM_THREADS = cores //
  concurrency`.
- Cost per text rollout: T persona calls + 1 run-level judge call.

Draft verifier: `PipecatEvalsVerifier(scenarios_dir, bot_path, policy_url, judge: LLMEndpoint, persona:
LLMEndpoint, k=4, concurrency=4, base_port=7900, bot_python=sys.executable, env={})` with `scenarios()`,
`run(results_path)` (semaphore, one port per job) and `_rollout(scenario, i, port)` that spawns the bot with
`POLYVOICE_SESSION` and `POLYVOICE_POLICY_URL` in env, runs `EvalSession.from_scenario(..., persona_llm=,
judge=)` under a 600 s cap, stops the bot, and returns a `Conversation(scenario, rollout, session_id,
messages, reward, hint, passed, succeeded, ended_by, metrics, tool_calls, duration_ms, error)`. Default
reward: `None` on error, else `succeeded × mean(metric scores)`. Hint: `result.failure or result.reason`
plus each failed verdict's `(metric, turn, reason)`. Factories `judge_llm(cfg)` / `persona_llm(cfg)` read
`POLYVOICE_JUDGE_*` / `POLYVOICE_PERSONA_*` env for the subprocess mode.

Example scenario in the exact schema (`recipes/dental/evals/pool/book_cleaning.yaml`):

```yaml
name: book_cleaning
max_turns: 8
max_duration_s: 120
scenarios:
  - name: new_patient
    persona: |
      Priya Shah, a new patient calling a dental office to book a routine cleaning. Polite, answers one
      question at a time; gives details only when asked: DOB 03/14/1990, phone 555-0177, no insurance,
      prefers weekday mornings next week, no dental pain.
    goal: "Book a routine cleaning next week in the morning as a new patient, then end the call."
    success: "the bot collected the caller's name and phone number, booked a weekday-morning cleaning next week by calling the booking tool, and confirmed the date and time back"
    metrics:
      - name: one_question
        criterion: "the reply asks for at most one piece of information"
        min_score: 0.8
      - name: no_hallucinated_booking
        criterion: "the reply does not claim an appointment is booked unless a book_appointment call happened before it"
        min_score: 1
      - measure: function_calls
        calls: [check_availability, {name: book_appointment, args: {service: cleaning, new_patient: true}}]
      - measure: words
        max_value: 60
```

Deterministic rewards from scripted scenarios: `EvalExpectation(event, within_ms, text_contains,
text_excludes, calls[EvalFunctionCall(name, args)], eval, absent, …)`; function-call args are a subset match;
all expectations of a turn share one deadline anchored at the send; `EvalScriptTurnResult.expectations[].
matched`, `.status`, `.duration_ms`; failure `kind` is machine-readable (`missing_function_call`,
`function_args_mismatch`, `timeout`, `judge_no`, …).

## 5. Transports: one bot for eval, simulators and production

- `-t` choices: `daily, eval, livekit, moq, vonage, webrtc, websocket, twilio, telnyx, plivo, exotel`.
  `create_transport` looks up one key per route; a missing key raises, extra keys are ignored, so a superset
  dict is safe. `eval` needs `EvalTransportParams` and no FastAPI. `daily` → `DailyParams`; `webrtc` →
  `TransportParams` (SmallWebRTC, offer at `POST /api/offer`); `websocket` → `FastAPIWebsocketParams`.
- **The generic websocket transport has no default serializer and silently drops everything without one.**
  Set `serializer=ProtobufFrameSerializer()` or a custom `FrameSerializer(setup, serialize, deserialize)`.
- Simulators against a self-hosted bot:

| Simulator | Path | Stock transport | Custom serializer | Session join |
|---|---|---|---|---|
| Pipecat Evals | harness dials the bot's eval WebSocket | `EvalTransport` | no | yes: env / `--runner-body` |
| Coval WebSocket | `wss://` PCM16 or mu-law, binary or base64 JSON, custom headers | `FastAPIWebsocketTransport` | **yes** (raw PCM) | yes: `{{run_id}}` / `{{simulation_output_id}}` in headers or init JSON |
| Coval → Pipecat Cloud | start API → Daily room | Daily | no | no documented run id in `body` |
| Bluejay CHIRP | Basic-auth WS, raw PCM16 16 kHz, JSON events | `FastAPIWebsocketTransport` (own route for a real 401) | **yes** | yes: `X-Simulation-Result-Id` upgrade header |
| Roark self-hosted | SmallWebRTC offer endpoint | `SmallWebRTCTransport` | no | via Roark observer |
| Cekura manual | we mint a Daily room + token per run | Daily from `body` | no | yes, we choose the body |

  A single `RawPcm16Serializer(sample_rate)` covers Coval and Bluejay. Minting a Daily room per rollout is
  `DailyRESTHelper.create_room(DailyRoomParams(privacy="private", properties=DailyRoomProperties(exp, …)))` +
  `get_token(room_url, expiry, owner)`; the bot then builds `DailyTransport` from `runner_args.body`.
- Recipe `transport_params`: `eval`, `daily`, `webrtc`, `websocket` (protobuf by default, raw PCM by env).
  Leave telephony out. Extras: `runner`, `daily`, `webrtc`, `websocket`, `silero` (empty), `openai`.

## 6. Canary in one bot

`LLMSwitcher(llms=[incumbent, candidate], strategy_type=ServiceSwitcherManual)` is a `ParallelPipeline`
with per-branch filters; index 0 is active at start; `register_function` and `LLMContext(tools=)` handlers
propagate to all members; both branches share the one `LLMContext`. Switch with
`worker.queue_frames([ManuallySwitchServiceFrame(service=candidate)])` **before** `LLMRunFrame` in
`on_client_connected`. Pick the branch with `sha1(session_id) % 100 < canary_pct`, never `hash()`. Give the
two services distinct `name=` so `MetricsData.processor` splits TTFB per branch. `LLMUpdateSettingsFrame`
reaches the active member only unless `reach_inactive_services=True` or `service=`. Failover strategy
switches on a non-fatal `ErrorFrame` from an unusable service. Flows accepts an `LLMSwitcher` directly.

## 7. OpenTelemetry as a secondary trace source

Span tree `conversation` → `turn` → `llm | stt | tts`. `llm` attributes: `gen_ai.request.model`, `input`
(JSON string of OpenAI messages, base64 media replaced), `tools` (JSON), `gen_ai.system_instructions`,
`output` (concatenated text, partial on interrupt), `metrics.ttfb`, `gen_ai.usage.input_tokens` /
`output_tokens` / cache / reasoning. `conversation.id` is on `conversation` and `turn`, not on `llm`; resolve
via parent ids, and keep a pending list across batches because `llm` spans arrive before their `turn`
closes. Missing versus the observer: tool results, finish reason, end reason, node name, and speculative
requests are indistinguishable. Enable with `setup_tracing(service_name, exporter=OTLPSpanExporter())` and
`PipelineWorker(enable_tracing=True, conversation_id=session_id, additional_span_attributes={...})`.
Receiver: FastAPI `POST /v1/traces` parsing `ExportTraceServiceRequest` from `opentelemetry-proto`; draft in
`docs/drafts/pipecat/otel_receiver.py`.

## 8. Packaging, skills, Cloud promotion

- `pipecat-polyvoice`: hatchling, src layout `src/pipecat_polyvoice`, BSD-2, `requires-python>=3.11`,
  deps `pipecat-ai[openai]>=1.9.0,<2` + httpx; extras `evals = [pipecat-ai[evals]>=1.11.0,<2]`,
  `trainer = [polyvoice]`, `dev`. README in the Roark shape: intro, "Tested with pipecat-ai 1.11.0
  (compatible with >=1.9.0,<2)", maintainer attribution, quick start, how it works, running modes,
  examples, configuration reference, development, changelog, licence. Docs PR: a row under **Analytics &
  Monitoring** in `supported-services.mdx`, a page using the `<CommunityMaintained>` snippet, and a card on
  the evals "platforms" page.
- `pipecat init` has a hard-coded service registry with no plugin hook and no `--base-url` flag; no
  community LLM is listed today. Path now: scaffold with `--llm openai_llm --eval` and let `polyvoice init`
  post-edit `bot.py`. A registry PR (`ServiceDefinition(value="polyvoice_llm", package="pipecat-polyvoice",
  include_params=["api_key","base_url"])`) is possible but expect pushback.
- Skills: marketplace `.claude-plugin/marketplace.json` with `plugins[{name, description, version, source,
  skills}]`; `SKILL.md` frontmatter is `name` + `description`. `polyvoice:setup` in the PhoneLLM style
  (CLIs → proxy credentials → health check → point the bot → sync and lint → headless eval → suite smoke →
  first cycle, billable, confirm first → optional Cloud). Rules copied verbatim: never invent credentials,
  never create billable infrastructure without confirming, never print secrets, browser auth via `! cmd`,
  `--skip` on secrets and `--force` on deploy because prompts hang.
- Pipecat Cloud from code: the `pipecatcloud` SDK is sessions only, but the documented REST API has
  `PUT /v1/secrets/{setName}` and `POST /v1/agents/{agentName}` with `forceRedeploy: true` under the private
  API key. So `polyloop promote` can flip `POLYVOICE_MODEL` and force a redeploy in ~30 lines of httpx,
  with `pipecat cloud secrets set --skip` + `deploy --force` as the CLI fallback. Warm instances keep old
  code until their session ends; there is no canary at the platform level, hence §6.

## 9. Order of work

1. `pipecat-polyvoice`: `llm.py`, `observer.py`, tests with `run_test`. Half a day.
2. `polyvoice/verifiers/pipecat_evals.py` + `targets/local_bot.py`; dental recipe `bot.py` and
   `evals/{pool,holdout}`; harness venv; `polyvoice scenarios convert` to Coval. One to two days.
3. `envs/voice.py` over the verifier protocol; Coval reshaped. One day.
4. Modal cycle: tier 0 rollouts → OPSD → Coval gate; commit the receipt. One night.
5. `RawPcm16Serializer` + Coval WebSocket path (session join without Pipecat Cloud). Half a day.
6. Skill, README first screen, docs PR, Discord post. One day.
