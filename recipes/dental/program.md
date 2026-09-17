# dental: what the proposer may do

The controller (`polyloop run`) owns the two Coval test sets, the metric ids, the gate, the budget
and the lineage. A proposer may open PRs that:

- add scenarios to `scenarios-pool.json` (then `polyvoice seed`); a scenario is a caller intent plus
  the behaviours the judge checks, written the way the metric prompts read them;
- change the `opsd` stage (group size, steps, learning rate, hint length, `min_rows`) with the
  receipt of the cycle that motivated it;
- change `system.md`; a prompt change is a policy change and goes through the same gate;
- diagnose rejected cycles from `polyvoice ledger --failed` and `cycles/<id>/eval/*/rewards.jsonl`.

Not allowed: `gate.*`, `promote.*`, `environment.options.metric_ids`, the test set ids, `lineage.json`.
