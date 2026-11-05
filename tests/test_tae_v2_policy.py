"""Synthetic routing and real shared-engine compatibility; no model/data runs."""

from collections import Counter
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import tae_v2_policy as policy
from shared_150k_retrieval import Shared150kRetrieval, ENCODER_ID, ENCODER_REVISION


def result(weights=((7, .8), (20, .2)), *, topic=7, strength=.8):
    return {"topic_id": topic, "membership_strength": strength,
            "topic_memberships": [{"topic_id": key, "membership": value} for key, value in weights],
            "membership_sum": math.fsum(value for _, value in weights), "membership_status": "available",
            "membership_unavailable_reason": None, "signal_interpretation": "uncalibrated",
            "embedding": np.eye(1, 384, dtype=np.float32)[0]}


class PolicyTests(unittest.TestCase):
    def assert_route(self, inference, route, selected, reason=None, **settings):
        decision = policy.decide(inference, **settings)
        self.assertEqual(decision["route"], route)
        self.assertEqual(decision["selected_topic_ids"], selected)
        if reason is not None:
            self.assertEqual(decision["reason"], reason)
        json.dumps(decision, allow_nan=False)
        return decision

    def test_strong_single_topic_and_single_priority(self):
        decision = self.assert_route(result(((7, .8), (20, .2))), "single", [7], "strong_single_topic")
        self.assertEqual(decision["retrieval_kwargs"], {"mode": "single", "topic_id": 7})
        self.assertEqual(decision["threshold_status"], "provisional_unoptimized_unvalidated")
        self.assertIn("Uncalibrated", decision["signal_interpretation"])
        # Alternative 20 is meaningful, but the single gate deliberately comes first.
        self.assertEqual(decision["signals"]["candidate_topic_ids"], [7, 20])

    def test_bounded_multi_topic_caps_at_three_and_includes_prediction(self):
        inference = result(((7, .4), (20, .25), (42, .2), (90, .15)), strength=.3)
        decision = self.assert_route(inference, "multi", [7, 20, 42], "bounded_multi_topic")
        self.assertAlmostEqual(decision["signals"]["candidate_mass_share"], .85)
        self.assertEqual(decision["retrieval_kwargs"], {"mode": "multi", "topic_ids": [7, 20, 42]})

    def test_non_leading_prediction_is_not_automatically_a_contradiction(self):
        inference = result(((7, .2), (20, .5), (42, .3)), strength=.7)
        decision = self.assert_route(inference, "multi", [7, 20, 42])
        self.assertFalse(decision["signals"]["predicted_is_unique_leader"])

    def test_weak_positive_anchor_can_only_use_multi_with_two_meaningful_alternatives(self):
        self.assert_route(result(((7, .01), (20, .50), (42, .49)), strength=.8), "multi", [7, 20, 42])
        self.assert_route(result(((7, .01), (20, .99)), strength=.8), "unfiltered", [], "insufficient_meaningful_topics")

    def test_diffuse_memberships_and_absolute_low_mass_go_unfiltered(self):
        inference = result(tuple((topic, .1) for topic in range(10)), topic=0, strength=.8)
        self.assert_route(inference, "unfiltered", [], "diffuse_membership")
        self.assert_route(result(((7, .008), (20, .002))), "unfiltered", [], "low_membership_mass")

    def test_outlier_priority_and_no_adaptive_outlier_filter(self):
        inference = result(topic=-1, strength=0)
        inference.update(membership_status="unavailable", topic_memberships=None, membership_sum=None)
        self.assert_route(inference, "unfiltered", [], "predicted_outlier")

    def test_strength_reliability_boundaries(self):
        inference = result(((7, .5), (20, .5)), strength=.1)
        self.assert_route(inference, "multi", [7, 20])
        inference["membership_strength"] = math.nextafter(.1, 0)
        self.assert_route(inference, "unfiltered", [], "low_membership_strength")
        for value in (0, None, float("nan"), float("inf"), -.1, 1.1, True, "0.8", [0.8], 10**5000):
            with self.subTest(value_type=type(value)):
                inference["membership_strength"] = value
                self.assert_route(inference, "unfiltered", [])

    def test_single_strength_boundary(self):
        inference = result(((7, .75), (20, .25)), strength=.5)
        self.assert_route(inference, "single", [7])
        inference["membership_strength"] = math.nextafter(.5, 0)
        self.assert_route(inference, "multi", [7, 20])

    def test_concentration_boundary_and_unique_leader_requirement(self):
        inference = result(((7, .7), (20, .3)), strength=.8)
        self.assert_route(inference, "single", [7])
        value = math.nextafter(.7, 0)
        inference = result(((7, value), (20, 1-value)), strength=.8)
        self.assert_route(inference, "multi", [7, 20])
        self.assert_route(result(((7, .5), (20, .5))), "multi", [7, 20],
                          thresholds={"single_min_concentration": .5})

    def test_alternative_significance_boundary(self):
        inference = result(((7, .65), (20, .25), (42, .1)), strength=.3)
        self.assert_route(inference, "multi", [7, 20, 42])
        self.assert_route(inference, "multi", [7, 20], thresholds={"min_alternative_share": math.nextafter(.1, 1)})

    def test_absolute_mass_boundary(self):
        inference = result(((7, .06), (20, .04)), strength=.3)
        self.assert_route(inference, "multi", [7, 20])
        self.assert_route(inference, "unfiltered", [], "low_membership_mass",
                          thresholds={"min_membership_mass": math.nextafter(.1, 1)})

    def test_multi_mass_coverage_boundary(self):
        inference = result(((7, .4), (20, .25), (42, .15), (90, .1), (91, .1)), strength=.3)
        self.assert_route(inference, "multi", [7, 20, 42])
        self.assert_route(inference, "unfiltered", [], "diffuse_membership",
                          thresholds={"multi_min_selected_share": math.nextafter(.8, 1)})

    def test_low_strength_does_not_silently_degrade_to_single_when_no_alternatives(self):
        self.assert_route(result(((7, .95), (20, .05)), strength=.3), "unfiltered", [], "insufficient_meaningful_topics")

    def test_zero_mass_and_empty_distributions(self):
        self.assert_route(result(((7, 0), (20, 0))), "unfiltered", [], "zero_membership_mass")
        self.assert_route(result(()), "unfiltered", [], "empty_membership_distribution")
        self.assert_route(result(((7, 0), (20, 0))), "unfiltered", [], "zero_membership_mass",
                          thresholds={"min_membership_mass": 0})

    def test_missing_or_unavailable_signals(self):
        inference = result()
        inference.update(membership_status="unavailable", topic_memberships=None, membership_sum=None,
                         membership_unavailable_reason="membership_exemplars_unavailable")
        decision = self.assert_route(inference, "unfiltered", [], "membership_unavailable")
        self.assertEqual(decision["signals"]["membership_unavailable_reason"], "membership_exemplars_unavailable")
        for key in ("membership_status", "membership_sum", "topic_memberships", "membership_strength"):
            inference = result()
            inference.pop(key)
            self.assert_route(inference, "unfiltered", [])

    def test_contradictory_signal_metadata_and_predicted_topic_support(self):
        cases = []
        inference = result()
        inference["membership_status"] = "unavailable"
        cases.append(inference)
        inference = result()
        inference["membership_sum"] = .2
        cases.append(inference)
        inference = result()
        inference["membership_unavailable_reason"] = "not_available"
        cases.append(inference)
        for inference in cases:
            self.assert_route(inference, "unfiltered", [])
        self.assert_route(result(((20, .8), (42, .2))), "unfiltered", [], "predicted_topic_missing_from_memberships")
        self.assert_route(result(((7, 0), (20, 1))), "unfiltered", [], "predicted_topic_has_zero_membership")

    def test_malformed_results_distributions_and_ids_fail_safely(self):
        for value in (None, [], "bad", 3, {}):
            self.assert_route(value, "unfiltered", [])
        for topic in (-2, True, np.bool_(True), 7.0, "7", None, 10**5000):
            inference = result(topic=topic)
            self.assert_route(inference, "unfiltered", [])
        bad_entries = ["bad", {}, [None], [{"topic_id": -1, "membership": 1}],
                       [{"topic_id": 7, "membership": True}], [{"topic_id": 7, "membership": float("nan")}],
                       [{"topic_id": 7, "membership": -.1}], [{"topic_id": 7, "membership": 2}],
                       [{"topic_id": 7, "membership": .5}, {"topic_id": 7, "membership": .5}],
                       [{"topic_id": 7, "membership": .8}, {"topic_id": 20, "membership": .8}]]
        for entries in bad_entries:
            inference = result()
            inference["topic_memberships"] = entries
            self.assert_route(inference, "unfiltered", [])
        inference = result()
        inference["membership_status"] = np.array(["available", "unavailable"])
        self.assert_route(inference, "unfiltered", [])

    def test_deterministic_ranking_and_records_independent_of_input_order(self):
        inference = result(((7, .3), (42, .25), (20, .25), (90, .2)), strength=.3)
        expected = policy.decide(inference)
        self.assertEqual(expected["selected_topic_ids"], [7, 20, 42])
        rng = random.Random(42)
        for _ in range(10):
            rng.shuffle(inference["topic_memberships"])
            before = copy.deepcopy(inference["topic_memberships"])
            self.assertEqual(policy.decide(inference), expected)
            self.assertEqual(inference["topic_memberships"], before)

    def test_ignores_embedding_answers_support_labels_and_retrieval_outcomes(self):
        allowed = {"topic_id", "membership_strength", "topic_memberships", "membership_sum",
                   "membership_status", "membership_unavailable_reason"}
        class Guarded(dict):
            def get(self, key, default=None):
                if key not in allowed:
                    raise AssertionError(f"Policy inspected forbidden field: {key}")
                return super().get(key, default)
        inference = Guarded(result())
        before = policy.decide(inference)
        inference.update(embedding=object(), answers=object(), supporting_facts=object(),
                         is_supporting=object(), gold_coverage=object(), retrieval_success=object())
        self.assertEqual(policy.decide(inference), before)
        json.dumps(before, allow_nan=False)

    def test_configuration_validation_immutability_and_logged_thresholds(self):
        defaults = policy.TAEV2Thresholds()
        with self.assertRaises(AttributeError):
            defaults.min_strength = .3
        for settings in ({"unknown": .1}, {"min_strength": .8, "single_min_strength": .2},
                         {"min_strength": -1}, {"min_membership_mass": float("inf")},
                         {"single_min_concentration": True}, {"min_alternative_share": "0.1"},
                         {"multi_min_selected_share": float("nan")}, {"min_strength": 10**5000}):
            with self.assertRaises(ValueError):
                policy.decide(result(), thresholds=settings)
        decision = policy.decide(result(), thresholds={"single_min_strength": .9})
        self.assertEqual(decision["thresholds"]["single_min_strength"], .9)
        self.assertNotEqual(decision["configuration_sha256"], policy.decide(result())["configuration_sha256"])
        for name in ("bad", [], np.array(["tae_v2", "strict"])):
            with self.assertRaises(ValueError):
                policy.decide(result(), policy=name)

    def test_strict_and_outlier_only_v1_comparators(self):
        inference = result(topic=-1, strength=0)
        self.assert_route(inference, "single", [-1], "strict_predicted_topic", policy="strict")
        self.assert_route(inference, "unfiltered", [], "v1_outlier_fallback", policy="tae_v1")
        inference = result(strength=None)
        inference.update(topic_memberships=None, membership_status="unavailable", membership_sum=None)
        self.assert_route(inference, "single", [7], policy="strict")
        self.assert_route(inference, "single", [7], "v1_non_outlier_strict", policy="tae_v1")
        self.assert_route(inference, "unfiltered", [], policy="tae_v2")

    def test_fixed_three_is_threshold_free_anchor_plus_two_and_can_include_outlier(self):
        inference = result(((42, .1), (20, .1), (90, .1)), topic=-1, strength=0)
        self.assert_route(inference, "multi", [-1, 20, 42], "fixed_three_expansion", policy="fixed_three")
        inference = result(((7, .99), (20, .000001), (42, 0)), strength=None)
        self.assert_route(inference, "multi", [7, 20], policy="fixed_three")
        inference.update(membership_status="unavailable", membership_sum=None, topic_memberships=None)
        self.assert_route(inference, "single", [7], "fixed_three_predicted_only", policy="fixed_three")
        for name in ("strict", "tae_v1", "fixed_three", "unfiltered"):
            with self.assertRaises(ValueError):
                policy.decide(result(), policy=name, thresholds={})

    def test_always_unfiltered_and_invalid_prediction_comparator_fallbacks(self):
        self.assert_route(None, "unfiltered", [], "always_unfiltered", policy="unfiltered")
        for name in policy.POLICIES:
            self.assert_route({"topic_id": "bad"}, "unfiltered", [], policy=name)

    def test_retrieval_adapter_passes_k_and_engine_kwargs_unchanged(self):
        engine = mock.Mock()
        engine.retrieve.return_value = [{"row": 1}, {"row": 2}]
        inference = result(((7, .4), (20, .3), (42, .3)), strength=.3)
        output = policy.retrieve_with_policy(engine, inference, k=10)
        args, kwargs = engine.retrieve.call_args
        self.assertIs(args[0], inference["embedding"])
        self.assertEqual(kwargs, {"mode": "multi", "topic_ids": [7, 20, 42], "k": 10})
        self.assertIs(output["results"], engine.retrieve.return_value)
        with self.assertRaises(ValueError):
            policy.retrieve_with_policy(engine, {"topic_id": -1})

    def test_cli_reads_existing_inference_json_and_writes_only_stdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inference.json"
            inference = result()
            inference["embedding"] = inference["embedding"].tolist()
            path.write_text(json.dumps({"validation": {}, "result": inference}))
            before = path.read_bytes()
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(policy.main(["--inference-json", str(path)]), 0)
            self.assertEqual(json.loads(stdout.getvalue())["route"], "single")
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual([p.name for p in Path(temporary).iterdir()], ["inference.json"])

    def test_cli_custom_config_and_duplicate_key_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, config = Path(temporary) / "inference.json", Path(temporary) / "config.json"
            inference = result()
            inference.pop("embedding")  # deciding a route never needs the vector
            path.write_text(json.dumps(inference))
            config.write_text(json.dumps({"single_min_strength": .9}))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                policy.main(["--inference-json", str(path), "--thresholds-json", str(config)])
            decision = json.loads(stdout.getvalue())
            self.assertEqual(decision["route"], "multi")
            self.assertEqual(decision["thresholds"]["single_min_strength"], .9)
            config.write_text('{"min_strength":0.2,"min_strength":0.3}')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                policy.main(["--inference-json", str(path), "--thresholds-json", str(config)])
            self.assertEqual(error.exception.code, 1)


class EngineCompatibilityTests(unittest.TestCase):
    def test_all_comparators_and_adaptive_routes_use_the_identical_shared_engine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = [7, 20, -1, 42, 7, 20, -1, 42]
            matrix = np.zeros((8, 384), dtype=np.float32)
            matrix[:, 0] = 1
            np.save(root / "embeddings.npy", matrix, allow_pickle=False)
            rows, assignments = [], []
            for i, topic in enumerate(labels):
                text = "Repeated original text" if i in (0, 4) else f"Text {i}"
                row = {"occurrence_id": f"2wikimultihopqa:train:{'a'*64}:{i+1}:0", "paragraph_text": text,
                       "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "question_id": f"q-{i}",
                       "source_split": "train", "source_line": i+1, "context_position": 0, "idx": 0, "title": "Fixture"}
                rows.append(row)
                assignments.append({**{k: v for k, v in row.items() if k != "paragraph_text"}, "row": i,
                                    "topic": topic, "bertopic": f"Topic {topic}", "membership_strength": .8})
            corpus, topics, manifest = root / "corpus.jsonl", root / "occurrence_topics.jsonl", root / "manifest.json"
            corpus.write_text("".join(json.dumps(row)+"\n" for row in rows))
            topics.write_text("".join(json.dumps(row)+"\n" for row in assignments))
            counts = Counter(labels)
            saved = {"schema_version": 1, "status": "complete", "occurrences": 8,
                     "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(), "encoder_id": ENCODER_ID,
                     "encoder_revision": ENCODER_REVISION, "topic_summary": {"occurrences": 8,
                     "topic_sizes": {str(k): v for k, v in counts.items()}, "non_outlier_topics": 3,
                     "outlier_occurrences": 2, "outlier_fraction": .25}, "artifacts": {}}
            for path in (root / "embeddings.npy", topics):
                saved["artifacts"][path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
            manifest.write_text(json.dumps(saved))
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            with Shared150kRetrieval(corpus, root / "embeddings.npy", topics, manifest, block_size=2,
                    expected_count=8, expected_corpus_sha256=saved["corpus_sha256"]) as engine:
                for name in policy.POLICIES:
                    for inference in (result(), result(((7, .4), (20, .3), (42, .3)), strength=.3), result(topic=-1)):
                        for k in (1, 5, 10):
                            output = policy.retrieve_with_policy(engine, inference, policy=name, k=k)
                            direct = engine.retrieve(inference["embedding"], k=k, **output["decision"]["retrieval_kwargs"])
                            self.assertEqual(output["results"], direct)
                            self.assertLessEqual(len(output["decision"]["selected_topic_ids"]), 3)
                duplicates = policy.retrieve_with_policy(engine, result(), policy="strict", k=2)["results"]
                self.assertEqual([r["row"] for r in duplicates], [0, 4])
                self.assertEqual(duplicates[0]["paragraph_text"], duplicates[1]["paragraph_text"])
                self.assertNotEqual(duplicates[0]["occurrence_id"], duplicates[1]["occurrence_id"])
            self.assertEqual(before, {path.name: path.read_bytes() for path in root.iterdir()})


if __name__ == "__main__":
    unittest.main()
