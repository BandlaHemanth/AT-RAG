# Step 5D: TAE V2 topic-selection policy

`scripts/tae_v2_policy.py` is a standard-library-only, deterministic eligibility
policy. It accepts `QueryInference150k.infer()` results and returns a JSON-safe
decision. No model fitting, document encoding, artifact writes, threshold search,
LLM generation, answer grading, question rewriting or evaluation is performed.
The existing query component, retrieval engine and original AT-RAG loop are unchanged.

The supplied development observations (strict Recall@5 12/236, three-topic 13/236,
unfiltered 16/236; corpus coverage 61/236; 40/100 outlier predictions) are
**exploratory context**, not final TAE V2 results or evidence that these thresholds
improve accuracy. None of those counts enters the routing implementation.

## Signals and provisional configuration

Let `p` be the predicted topic, `s` its scalar HDBSCAN strength, `w(t)` the saved
per-topic membership signal, and `M = sum_t w(t)` the absolute membership mass.
For positive M, `c(t)=w(t)/M` is a relative mass share. It is a routing statistic,
not a calibrated probability of evidence adequacy. The original weights and mass
are preserved in the record; no scalar is substituted for a distribution.

| Threshold | Default | Meaning |
| --- | ---: | --- |
| `min_strength` | 0.10 | Minimum reliable scalar strength for any adaptive filtered route |
| `single_min_strength` | 0.50 | Additional scalar requirement for single-topic retrieval |
| `single_min_concentration` | 0.70 | Required predicted-topic share `c(p)` for single-topic retrieval |
| `min_alternative_share` | 0.10 | Minimum share for a positive-weight alternative; also defines whether the predicted anchor counts as meaningful |
| `min_membership_mass` | 0.10 | Absolute mass floor, preventing tiny signals from appearing strong merely after division by M |
| `multi_min_selected_share` | 0.80 | Required share captured by the candidate set for multi-topic retrieval only |

**All defaults are provisional round-number starting values: unoptimized and
empirically unvalidated.** Synthetic tests verify the code, not the scientific
quality of these values. Every threshold is configurable in [0,1]; the single
strength floor must be >= the general strength floor. Invalid configurations
raise `ValueError`. Zero-strength and zero-mass signals remain unusable even if
their respective configurable floors are set to zero. The topic cap is fixed at
three and cannot be enlarged by configuration.

## Exact adaptive priority

1. An invalid/missing prediction produces unfiltered retrieval. A prediction of
   `-1` produces unfiltered retrieval regardless of other signals.
2. Missing, malformed, nonfinite or zero scalar strength produces unfiltered
   retrieval; so does `s < min_strength`.
3. Require status `available` and a nonempty, valid per-topic distribution.
   Topic IDs must be unique non-outlier integers and weights must be finite and
   nonnegative. Missing/unavailable signals, contradictory availability metadata,
   duplicate topic IDs, invalid mass or a reported sum inconsistent with the
   actual weights produce unfiltered retrieval. Values/sums may exceed one only
   by the inference component's 1e-6 numerical validation tolerance. This tolerance
   never relaxes a routing threshold.
4. Zero mass or `M < min_membership_mass` produces unfiltered retrieval. A missing
   predicted topic or `w(p)=0` contradicts using that prediction as a supported
   filtered anchor and also produces unfiltered retrieval.
5. **Single-topic wins first** when p is the unique largest-weight topic,
   `s >= single_min_strength`, and `c(p) >= single_min_concentration`. Exact ties
   cannot trigger single-topic routing. The multi-topic coverage threshold does
   not apply to this separate single-topic gate.
6. Otherwise form a candidate list: p first, then up to two non-outlier alternatives
   with positive weight and `c(t) >= min_alternative_share`, ranked by descending
   weight and ascending topic ID for exact ties. The predicted topic is always
   retained. At least two candidates must meet the same significance share floor;
   a very weak but positive anchor may therefore accompany two meaningful alternatives.
7. **Multi-topic** is chosen only if at least two meaningful candidates exist and
   their candidate-set mass share is `>= multi_min_selected_share`. Otherwise choose
   unfiltered retrieval (`insufficient_meaningful_topics` or `diffuse_membership`).

All numeric routing comparisons are inclusive at equality, except the explicitly
strict positive-strength/positive-mass and unique-leader requirements. A non-leading
prediction is not automatically invalid: BERTopic's approximate label and largest
soft membership can legitimately disagree. Such a question can use a supported
multi-topic set. Conversely, a weak scalar plus one dominant topic with no meaningful
alternatives is sent unfiltered, never silently downgraded to single-topic.

## Comparators

Pass `policy=` to the common `decide()` or `retrieve_with_policy()` functions:

| Name | Deterministic definition |
| --- | --- |
| `strict` | Predicted topic only, **including -1**; strength/distribution do not gate it |
| `tae_v1` | Unfiltered iff prediction is -1; otherwise predicted topic only. This is the requested outlier-only V1-style comparator |
| `fixed_three` | Predicted topic, including -1, plus up to two highest positive-weight non-outlier alternatives; no adaptive thresholds. If a distribution is missing, malformed or has no positive alternatives, retain the prediction alone |
| `unfiltered` | Always search all corpus occurrences, including outliers |

All except always-unfiltered require a valid prediction; when it is malformed,
they return a logged unfiltered error decision. `fixed_three` deliberately defines
an anchor-plus-two comparator. Do not claim it exactly reproduces the earlier
13/236 exploratory run unless that run's selection and outlier handling match.
It never fills unused topic slots with zero-weight or invented topics. Comparator
records contain no adaptive thresholds, and supplying thresholds to one is an error.

## API, decision records and use

```python
from tae_v2_policy import TAEV2Thresholds, decide, retrieve_with_policy

# inference_result is an existing QueryInference150k.infer(question) result.
# engine is the SAME existing Shared150kRetrieval for every comparator.
thresholds = TAEV2Thresholds()  # provisional, not a tuned configuration
decision = decide(inference_result, thresholds=thresholds)
hits = engine.retrieve(inference_result["embedding"], k=5, **decision["retrieval_kwargs"])

# Equivalent thin delegation; it does not rerank, deduplicate, refill or change k.
bundle = retrieve_with_policy(engine, inference_result, policy="tae_v2", thresholds=thresholds, k=5)
strict_bundle = retrieve_with_policy(engine, inference_result, policy="strict", k=5)
```

The returned record contains `policy`, `policy_version`, `configuration_sha256`,
`threshold_status`, all six `thresholds`, `route`, `selected_topic_ids`, exact engine
`retrieval_kwargs`, and a stable machine-readable `reason`. Its `signals` include
prediction, scalar, availability, reported/actual mass, concentration, leader
status and candidate weights/coverage. `signal_errors` identifies malformed signal
families without serializing invalid NaN/Infinity objects. Attach a question ID and
persist each record in the caller's new experiment output when evaluation is later
authorized; the policy itself writes no log files or source artifacts.

The decision reads only six inference fields: `topic_id`, `membership_strength`,
`topic_memberships`, `membership_sum`, `membership_status`, and
`membership_unavailable_reason`. It never inspects the embedding, question text,
gold support, answers, corpus coverage or retrieval outcomes. The retrieval adapter
additionally passes the existing embedding directly to the engine; a missing or
invalid embedding cannot be repaired by a routing fallback. Similarity, top-k,
duplicate occurrence preservation and tie ordering belong solely to the unchanged
retrieval engine. Empty retrieval results never trigger another routing attempt.

Run tests:

```bash
python -B -m unittest discover -s tests -p 'test_tae_v2_policy.py' -v
```

To inspect one **existing inference JSON** (no encoding or retrieval), including
the wrapper printed by the query-inference CLI:

```bash
python -B scripts/tae_v2_policy.py --inference-json /path/to/query_inference.json --policy tae_v2
```

The CLI writes the decision only to stdout. Optional `--thresholds-json` accepts
a JSON object with partial threshold overrides; duplicate keys and invalid values
are rejected. For later comparisons, freeze and record the chosen configuration,
reuse the same validated engine, inference outputs, question lists and k, and
report overall corpus coverage separately from retrieval conditional on evidence
availability. Keep uncovered questions in main results. This step performs no
full-scale evaluation and does not open `test_subsampled.jsonl`.
