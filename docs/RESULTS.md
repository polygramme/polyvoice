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
- **Sampler: fails.** vLLM's engine core died during initialization (`Failed core proc(s): {'EngineCore': -11}`,
  a segmentation fault) with CUDA graphs on and again with `enforce_eager`. The root-cause line is in Ray's
  engine-core log, not captured. Next step is a sampler-only boot of vLLM on the model with the engine log
  visible, without LoRA first, then with; the likely fixes are a newer vLLM in the SkyRL environment (LoRA on the
  hybrid architecture) or a kernel build change.
- Cost: ~$6 of Modal H100:2 for the two attempts.
