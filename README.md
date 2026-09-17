# polyvoice

A voice agent's language model that improves from its own calls. polyvoice is a recipe package for
[polyloop-rl](https://github.com/runnerelectrode/polyloop-rl): it plugs [Coval](https://coval.ai)
into the loop as the environment and the verifier, so the cycle polyloop already runs for coding
agents (snapshot, preflight, filter, train, evaluate, gate, promote) runs for the LLM inside a
phone receptionist.

```
Coval persona ──calls──▶ polyloop proxy (public https) ──▶ your Tinker server (LoRA adapter)
      │                        │ traces/<date>.jsonl (token-exact, session = simulation id)
      ▼                        ▼
Coval metrics: reward + judge explanation ──ledger──▶ hinted OPSD rows ──▶ candidate adapter
      ▼
held-out Coval test set, base vs candidate, K repeats ──▶ paired receipt ──▶ promote (proxy serves it)
```

What lives here and not in polyloop: the Coval client, the ledger that joins every simulated
conversation to its score and the judge's explanation, the environment that maps a Coval run
onto polyloop's rollout contract, a CLI to check the account and seed test sets, and the
`dental` recipe (a receptionist system prompt, 16 pool scenarios, 8 held-out scenarios).

Status: **v0, not yet run against Coval end to end.** The client's paths and envelopes were
verified against the live API on 2026-09-17 with read-only calls; the fake in `tests/` follows
those shapes. Two shapes are still unconfirmed until the first run: that a simulation's
`simulation_id` equals the `{{simulation_output_id}}` Coval sends in the header (the ledger
records both), and that binary judge values arrive as numbers or booleans (`rewards._num` also
accepts "true"/"false"/"pass"/"fail").

## What Coval does in this loop, and what it must not

- **Held-out gate and filter**: a run = agent × persona × test set, `iteration_count` = K. Each
  simulation reports its `test_case_id`, so polyloop's paired receipt works unchanged.
- **Hindsight for OPSD**: the judges' explanations of failed conversations become the hint the
  teacher sees; the student re-samples the same turns on policy.
- **Never the RL rollout engine.** Every simulation is a real-time conversation billed in
  simulation minutes (Starter: 100 min/month, $0.40/min over, 5 concurrent; Growth: 1,000 min,
  25 concurrent). This recipe is OPSD-only for that reason. RL rollouts belong in a text sandbox
  with a persona LLM and ASR-noise injection, see `docs/PLAN.md`.

Per cycle as shipped: filter 8 scenarios × 2 = 16 conversations, gate 8 × 4 × 2 policies = 64.

## Setup

On the GPU node (polyloop's `scripts/node_up.sh` brings the server up), plus:

```bash
uv pip install -e .                         # pulls polyloop-rl at the pinned ref
export COVAL_API_KEY=...                    # never in the recipe

# 1. the proxy Coval will call, with the receptionist prompt prepended to every conversation
polyloop proxy --loop recipes/dental/loop.yaml --port 8787 --system-prompt recipes/dental/system.md
cloudflared tunnel --url http://127.0.0.1:8787      # note the https URL -> environment.options.public_url
# optional second instance so production traffic stays on the live adapter while a candidate is scored:
polyloop proxy --loop recipes/dental/loop.yaml --port 8788 --slot candidate --system-prompt recipes/dental/system.md

# 2. test sets (free), agent and persona ids
polyvoice account                                                        # what the account has
polyvoice seed --test-set NEW:dental-pool    --file recipes/dental/scenarios-pool.json
polyvoice seed --test-set NEW:dental-holdout --file recipes/dental/scenarios-holdout.json
# in the Coval UI: a MODEL_TYPE_CHAT agent (any endpoint; polyvoice re-points a copy), and the
# metrics: a composite from expected behaviours + one binary judge ("never states a price").
# Fill agent_id, persona_id, metric_ids, public_url and the two coval:// datasets in loop.yaml.

# 3. check without spending, then run
polyvoice check --loop recipes/dental/loop.yaml
polyloop run    --loop recipes/dental/loop.yaml
polyvoice ledger --loop recipes/dental/loop.yaml --failed
```

The agent registered on Coval is a duplicate of `agent_id` pointed at `<public_url>/v1/chat/completions`
with `X-Session-Id: {{simulation_output_id}}`, so each conversation's trace is keyed by the id Coval
reports scores under. Before each rollout the environment tells the proxy which adapter to serve
(`POST /admin/serve`, never `live.json`) and clears it afterwards.

## Layout

```
polyvoice/
  coval/client.py    v1 API: test cases, agents (duplicate + endpoint), runs, simulations, metric outputs, conversations:submit
  coval/rewards.py   reward = mean of metric values, judge explanations, ledger, session hints, holdout exclusions
  envs/coval.py      CovalEnvironment: polyloop's Environment protocol on Coval runs
  cli.py             polyvoice account | check | seed | ledger
recipes/dental/      loop.yaml, system.md, scenarios-pool.json, scenarios-holdout.json, program.md
tests/               fake Coval in the live API's shapes; the polyloop runner driven end to end
docs/PLAN.md         the full voice loop plan (sandbox rollouts, production scoring, model choice)
```

## License

Apache-2.0.
