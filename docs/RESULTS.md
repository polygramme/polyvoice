# Results

## dental, Qwen3.5-4B, cycle 1 (2026-09-18, Modal H100:2, Coval)

The first complete cycle of the voice loop on Coval simulations: filter, hinted OPSD on the captured
calls, paired evaluation of base vs candidate on held-out scenarios, gate. The gate **rejected** the
candidate, correctly for the numbers below. Raw artifacts in `docs/results/dental-cycle1/`.

**Setup**
- Environment: Coval persona "Standard Customer" calling the polyloop proxy over a Modal port forward;
  every conversation is a proxy session keyed by Coval's simulation id.
- Reward: mean of two Coval metrics per conversation, a composite over each scenario's expected
  behaviours and a binary judge "never states a price or diagnosis". The judges' explanations are the
  OPSD hindsight hints.
- Model: Qwen/Qwen3.5-4B, LoRA rank 32, SkyRL Tinker server (Megatron), proxy prepends the receptionist
  system prompt. Train: OPSD only, 4 steps, 16 groups of 2, on 108 rows built from the filter's calls.
- Pool 16 scenarios, held-out 8 scenarios; gate = 8 × 4 repeats per policy, paired, 95% bootstrap CI,
  min_delta 0.05, max_regressions 2.

**Numbers**

| | |
|---|---|
| filter | 8 scenarios × 2 = 16 conversations under base, mean reward 0.938 |
| OPSD teacher KL per step | 0.309 → 0.170 → 0.232 → 0.211 |
| incumbent (base) held-out mean | 0.888 |
| candidate held-out mean | 0.938 |
| paired delta | +0.049, CI95 [−0.021, +0.117] |
| wins / losses / ties (scenarios) | 4 / 1 / 3 |
| checks | enough_tasks ✓, logprob_agreement ✓, regressions_within_cap ✓, paired_delta_above_min ✗ (0.049 < 0.05) |
| Coval conversations | 82 (2 smoke, 16 filter, 32 + 32 evaluate) |
| GPU-seconds | preflight 57, filter 352, train 705, evaluate 663 + 568 |
| wall clock | 23 min for the resumed train→gate half; the filter half ran in an earlier container |

Per scenario, incumbent → candidate: 1.00→0.88, 0.81→0.81, 0.88→0.94, 1.00→1.00, 0.88→1.00,
0.79→1.00, 0.81→0.94, 0.94→0.94.

**Reading**
- The loop runs end to end on real simulated calls: Coval scored every conversation, the proxy captured
  every turn token-exact (20 turns per 2 calls in the smoke; 108 rows from 16 calls), the judge's
  explanations attached to the right sessions, and the held-out sessions were excluded from training.
- The candidate is directionally better (4 wins, 1 loss) but a hair under the 0.05 minimum, and 8
  scenarios give a ±7-point interval. Nightly cycles with a larger held-out set are what turn this into a
  curve; the pool is also easy for the base model (0.89 to 0.94), so harder scenarios are the next lever.
- Cost: about $8 of Modal H100:2 for the run itself; the night's ten launches (tunnel, agent-id and import
  bugs) roughly doubled that. Coval: about 80 simulation minutes.

## PhoneLLM Alpha 1 boot test (2026-09-18, Modal H100:2)

Two attempts, same recipe with `model: pipecat-ai/phonellm-alpha-1` and `renderer: nemotron3_disable_thinking`.

- **Trainer: works.** SkyRL's Megatron backend loaded the 30B hybrid Mamba-MoE (`NemotronHForCausalLM`) and
  created a LoRA model in ~2.5 min ("Created LoRA model" in the server log).
- **Sampler: fixed (2026-09-18 evening).** vLLM's engine core had died at init (`EngineCore: -11`). A sampler-only
  boot on one H100 (vLLM 0.26.0, transformers 5.8, torch 2.11+cu128, mamba_ssm + causal_conv1d present) served
  the model both without and with LoRA at 8k context / 16 sequences / eager. With LoRA the model takes 65.7 GiB,
  so the 4B settings (32k context, 64 sequences, prefix caching, 4 LoRA slots) left no room and the engine crashed.
  Fix: `max_model_len 8192`, `max_num_seqs 16`, `enable_prefix_caching false`, `max_loras 2`, eager, memory
  fraction 0.93. With those the loop's proxy came up on PhoneLLM in 13.7 min and the smoke scored 2/2 held-out
  conversations (mean reward 0.625; the 4B scored 0.875 on the same two).
- Cost: ~$6 for the two failed attempts, ~$1 for the debug, ~$3 for the smoke.
