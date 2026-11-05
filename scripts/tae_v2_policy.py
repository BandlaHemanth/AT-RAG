"""Step 5D: deterministic TAE V2 eligibility routing, with provisional thresholds.

Standard library only. Accept QueryInference150k.infer() output; never encode,
fit, evaluate, inspect labels, or change Shared150kRetrieval ranking/top-k.
See docs/tae_v2_policy.md for the exact priority and comparator definitions.
"""

import argparse
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path


VERSION = 1
MAX_TOPICS = 3
MAX_TOPIC_ID = (1 << 63) - 1
NUMERIC_TOLERANCE = 1e-6  # validation of HDBSCAN output, never a routing tolerance
POLICIES = ("tae_v2", "strict", "tae_v1", "fixed_three", "unfiltered")
INTERPRETATION = "Uncalibrated HDBSCAN signals; no guarantee of evidence adequacy."


@dataclass(frozen=True)
class TAEV2Thresholds:
    """Round-number starting values, NOT optimized or empirically validated."""

    min_strength: float = 0.10
    single_min_strength: float = 0.50
    single_min_concentration: float = 0.70
    min_alternative_share: float = 0.10
    min_membership_mass: float = 0.10
    multi_min_selected_share: float = 0.80

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= 1 or not math.isfinite(value):
                raise ValueError(f"{field.name} must be a finite number in [0,1]")
            object.__setattr__(self, field.name, float(value))
        if self.single_min_strength < self.min_strength:
            raise ValueError("single_min_strength must be >= min_strength")


def _config(thresholds):
    if thresholds is None:
        return TAEV2Thresholds()
    if isinstance(thresholds, TAEV2Thresholds):
        return thresholds
    if isinstance(thresholds, Mapping):
        try:
            return TAEV2Thresholds(**thresholds)
        except TypeError as error:
            raise ValueError(f"Unknown/invalid threshold configuration: {error}") from error
    raise ValueError("thresholds must be TAEV2Thresholds or a threshold mapping")


def _topic(value, *, outlier_allowed):
    if (isinstance(value, bool) or not isinstance(value, Integral) or
            not (-1 if outlier_allowed else 0) <= value <= MAX_TOPIC_ID):
        return None
    return int(value)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= 1 + NUMERIC_TOLERANCE:
        return None
    value = float(value)
    return value if math.isfinite(value) and 0 <= value <= 1 + NUMERIC_TOLERANCE else None


def _read_signals(inference):
    """Explicit field allowlist. Embeddings, answers, support and other fields are ignored."""
    snapshot = {
        "predicted_topic": None, "membership_strength": None, "membership_status": None,
        "membership_unavailable_reason": None, "reported_membership_sum": None,
        "total_membership_mass": None, "membership_topic_count": 0,
        "predicted_membership": None, "predicted_concentration": None,
        "top_membership_topic": None, "top_concentration": None, "predicted_is_unique_leader": None,
        "candidate_topic_ids": [], "candidate_membership_mass": None, "candidate_mass_share": None,
        "candidate_meaningful_topics": None, "candidate_memberships": [],
    }
    errors = {"prediction": None, "strength": None, "distribution": None}
    if not isinstance(inference, Mapping):
        errors.update(prediction="invalid_inference_result", strength="missing_membership_strength",
                      distribution="missing_membership_signals")
        return snapshot, errors, []
    predicted = _topic(inference.get("topic_id"), outlier_allowed=True)
    snapshot["predicted_topic"] = predicted
    if predicted is None:
        errors["prediction"] = "invalid_predicted_topic"
    strength_value = inference.get("membership_strength")
    snapshot["membership_strength"] = _number(strength_value)
    if snapshot["membership_strength"] is None:
        errors["strength"] = "missing_membership_strength" if strength_value is None else "invalid_membership_strength"

    status = inference.get("membership_status")
    unavailable_reason = inference.get("membership_unavailable_reason")
    snapshot["membership_status"] = status if isinstance(status, str) else None
    snapshot["membership_unavailable_reason"] = unavailable_reason if isinstance(unavailable_reason, str) else None
    reported = inference.get("membership_sum")
    snapshot["reported_membership_sum"] = _number(reported)
    entries = inference.get("topic_memberships")

    def failed(reason):
        errors["distribution"] = reason
        return snapshot, errors, []

    if not isinstance(status, str) or status not in ("available", "unavailable"):
        return failed("invalid_membership_status")
    if status == "unavailable":
        if entries is not None or reported is not None:
            return failed("contradictory_membership_status")
        return failed("membership_unavailable")
    if unavailable_reason is not None or entries is None:
        return failed("contradictory_membership_status")
    if not isinstance(entries, (list, tuple)):
        return failed("invalid_membership_distribution")
    if not entries:
        return failed("empty_membership_distribution")
    if snapshot["reported_membership_sum"] is None:
        return failed("invalid_reported_membership_sum")
    weights = {}
    for item in entries:
        if not isinstance(item, Mapping):
            return failed("invalid_membership_entry")
        topic, weight = _topic(item.get("topic_id"), outlier_allowed=False), _number(item.get("membership"))
        if topic is None or weight is None:
            return failed("invalid_membership_entry")
        if topic in weights:
            return failed("duplicate_membership_topic")
        weights[topic] = weight
    total = math.fsum(weights.values())
    if total > 1 + NUMERIC_TOLERANCE:
        return failed("invalid_membership_mass")
    if not math.isclose(total, snapshot["reported_membership_sum"], abs_tol=NUMERIC_TOLERANCE, rel_tol=0):
        return failed("contradictory_membership_sum")
    ranked = sorted(weights.items(), key=lambda pair: (-pair[1], pair[0]))
    snapshot.update(total_membership_mass=total, membership_topic_count=len(ranked),
                    predicted_membership=weights.get(predicted), top_membership_topic=ranked[0][0])
    if total > 0:
        snapshot["top_concentration"] = ranked[0][1] / total
        if predicted in weights:
            snapshot["predicted_concentration"] = weights[predicted] / total
            snapshot["predicted_is_unique_leader"] = all(weights[predicted] > value for topic, value in ranked if topic != predicted)
    return snapshot, errors, ranked


def _record(policy, mode, topics, reason, snapshot, errors, thresholds):
    if mode == "single":
        kwargs = {"mode": "single", "topic_id": topics[0]}
    elif mode == "multi":
        kwargs = {"mode": "multi", "topic_ids": list(topics)}
    else:
        kwargs = {"mode": "unfiltered"}
    configuration = {"policy": policy, "version": VERSION, "thresholds": thresholds}
    digest = hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {
        "policy": policy, "policy_version": VERSION, "route": mode, "selected_topic_ids": list(topics),
        "retrieval_kwargs": kwargs, "reason": reason, "signals": snapshot, "signal_errors": errors,
        "thresholds": thresholds, "configuration_sha256": digest,
        "threshold_status": "provisional_unoptimized_unvalidated" if policy == "tae_v2" else "not_applicable",
        "signal_interpretation": INTERPRETATION,
    }


def decide(inference_result, *, policy="tae_v2", thresholds=None):
    """Return a JSON-safe routing record. Bad signals fail open to unfiltered.

    Unknown policies/invalid configuration raise ValueError (setup mistakes).
    Comparators use only the signals specified in their documented definitions.
    No gold fields or retrieval outcomes enter this function.
    """
    if not isinstance(policy, str) or policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    if policy != "tae_v2" and thresholds is not None:
        raise ValueError("Comparator policies do not accept adaptive thresholds")
    config = _config(thresholds) if policy == "tae_v2" else None
    snapshot, errors, ranked = _read_signals(inference_result)
    predicted, strength = snapshot["predicted_topic"], snapshot["membership_strength"]

    def finish(mode, topics, reason):
        return _record(policy, mode, topics, reason, snapshot, errors, {} if config is None else asdict(config))

    if policy == "unfiltered":
        return finish("unfiltered", [], "always_unfiltered")
    if errors["prediction"] is not None:
        return finish("unfiltered", [], errors["prediction"])
    if policy == "strict":
        return finish("single", [predicted], "strict_predicted_topic")
    if policy == "tae_v1":
        if predicted == -1:
            return finish("unfiltered", [], "v1_outlier_fallback")
        return finish("single", [predicted], "v1_non_outlier_strict")
    if policy == "fixed_three":
        # An explicit, threshold-free anchor-plus-two comparator, including -1
        # when it is the prediction. Never pad with zero-weight or invented IDs.
        alternatives = [topic for topic, value in ranked if topic != predicted and value > 0][:MAX_TOPICS - 1]
        selected = [predicted] + alternatives
        weights, total = dict(ranked), snapshot["total_membership_mass"]
        snapshot["candidate_topic_ids"] = selected
        snapshot["candidate_memberships"] = [{"topic_id": topic, "membership": weights.get(topic),
            "share": weights[topic] / total if total and topic in weights else None} for topic in selected]
        if ranked:
            snapshot["candidate_membership_mass"] = math.fsum(weights.get(topic, 0.0) for topic in selected)
            snapshot["candidate_mass_share"] = snapshot["candidate_membership_mass"] / total if total else None
        return finish("single" if len(selected) == 1 else "multi", selected,
                      "fixed_three_predicted_only" if len(selected) == 1 else "fixed_three_expansion")

    # Adaptive priority: outlier -> scalar reliability -> distribution validity
    # -> absolute mass -> predicted-topic consistency -> single -> multi -> full.
    if predicted == -1:
        return finish("unfiltered", [], "predicted_outlier")
    if errors["strength"] is not None:
        return finish("unfiltered", [], errors["strength"])
    if strength <= 0:
        return finish("unfiltered", [], "zero_membership_strength")
    if strength < config.min_strength:
        return finish("unfiltered", [], "low_membership_strength")
    if errors["distribution"] is not None:
        return finish("unfiltered", [], errors["distribution"])
    total = snapshot["total_membership_mass"]
    if total <= 0:
        return finish("unfiltered", [], "zero_membership_mass")
    if total < config.min_membership_mass:
        return finish("unfiltered", [], "low_membership_mass")
    predicted_weight = snapshot["predicted_membership"]
    if predicted_weight is None:
        return finish("unfiltered", [], "predicted_topic_missing_from_memberships")
    if predicted_weight <= 0:
        return finish("unfiltered", [], "predicted_topic_has_zero_membership")

    alternatives = [(topic, value) for topic, value in ranked
                    if topic != predicted and value > 0 and value / total >= config.min_alternative_share][:MAX_TOPICS - 1]
    selected = [predicted] + [topic for topic, _ in alternatives]
    selected_mass = math.fsum([predicted_weight] + [value for _, value in alternatives])
    meaningful = int(predicted_weight / total >= config.min_alternative_share) + len(alternatives)
    snapshot.update(candidate_topic_ids=selected, candidate_membership_mass=selected_mass,
                    candidate_mass_share=selected_mass / total, candidate_meaningful_topics=meaningful,
                    candidate_memberships=[{"topic_id": predicted, "membership": predicted_weight, "share": predicted_weight / total}] +
                        [{"topic_id": topic, "membership": value, "share": value / total} for topic, value in alternatives])

    if (snapshot["predicted_is_unique_leader"] and strength >= config.single_min_strength and
            snapshot["predicted_concentration"] >= config.single_min_concentration):
        return finish("single", [predicted], "strong_single_topic")
    if len(selected) < 2 or meaningful < 2:
        return finish("unfiltered", [], "insufficient_meaningful_topics")
    if selected_mass / total < config.multi_min_selected_share:
        return finish("unfiltered", [], "diffuse_membership")
    return finish("multi", selected, "bounded_multi_topic")


def retrieve_with_policy(engine, inference_result, *, policy="tae_v2", thresholds=None, k=5):
    """Thin delegation to the SAME engine for every policy; k is passed unchanged."""
    decision = decide(inference_result, policy=policy, thresholds=thresholds)
    if not isinstance(inference_result, Mapping) or "embedding" not in inference_result:
        raise ValueError("An inference result with an embedding is required for retrieval")
    results = engine.retrieve(inference_result["embedding"], k=k, **decision["retrieval_kwargs"])
    return {"decision": decision, "results": results}


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-json", type=Path, required=True, help="One infer() result or query-inference CLI JSON object")
    parser.add_argument("--policy", choices=POLICIES, default="tae_v2")
    parser.add_argument("--thresholds-json", type=Path, help="Optional adaptive threshold overrides; defaults remain provisional")
    args = parser.parse_args(argv)
    try:
        with args.inference_json.open(encoding="utf-8") as stream:
            inference = json.load(stream, object_pairs_hook=_unique_json_object)
        if isinstance(inference, Mapping) and "result" in inference:
            inference = inference["result"]
        thresholds = None
        if args.thresholds_json is not None:
            with args.thresholds_json.open(encoding="utf-8") as stream:
                thresholds = json.load(stream, object_pairs_hook=_unique_json_object)
            if not isinstance(thresholds, Mapping):
                raise ValueError("Threshold configuration must be a JSON object")
        print(json.dumps(decide(inference, policy=args.policy, thresholds=thresholds), sort_keys=True,
                         ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Policy routing failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
