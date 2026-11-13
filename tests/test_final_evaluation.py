import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import joblib

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fit = load("fit_final", ROOT / "scripts" / "fit_tae_r_topic_models.py")
router = load("router_final", ROOT / "scripts" / "tae_r_router.py")
fixture = load("fixture_final", ROOT / "tests" / "test_fit_tae_r_topic_models.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FinalEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.FitTests.setUpClass()
        f = fixture.FitTests
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.corpora = root / "corpora"
        shutil.copytree(f.corpora, cls.corpora)
        # Stand-in final set: the validation corpus and validation questions (embeddings exist for them).
        shutil.copy(cls.corpora / "retrieval_validation.jsonl", cls.corpora / "retrieval_final.jsonl")
        shutil.copy(cls.corpora / "questions_validation.jsonl", cls.corpora / "questions_final_eval.jsonl")
        files = {p.stem: {"file": p.name, "sha256": sha(p)} for p in cls.corpora.glob("*.jsonl")}
        (cls.corpora / "manifest.json").write_text(json.dumps({"files": files}))
        cls.frozen = dict(f.frozen, corpus_manifest=sha(cls.corpora / "manifest.json"))
        cls.f, cls.final = f, root / "final"

    @classmethod
    def tearDownClass(cls):
        fixture.FitTests.tearDownClass()
        cls.tmp.cleanup()

    def test_assign_final_guard_and_outputs_then_final_evaluation(self):
        with self.assertRaises(ValueError):
            fit.run(self.corpora, self.f.emb, self.final, ["assign_final"], ["H1"], frozen=self.frozen,
                    config=fixture.SMALL, encode=fixture.fake_encode, log=lambda m: None, source_dir=self.f.out)
        fit.run(self.corpora, self.f.emb, self.final, ["assign_final"], ["H1", "K1"], frozen=self.frozen,
                config=fixture.SMALL, encode=fixture.fake_encode, log=lambda m: None,
                allow_final=True, source_dir=self.f.out)
        for name in ("questions_final", "H1/assign_final", "K1/assign_final"):
            self.assertTrue(fit.Stage(self.final, name).complete(), name)
        for name in ("H1", "K1"):
            val_report, val_rows, _ = router.run(self.corpora, self.f.emb, self.f.out, name, frozen=self.frozen, log=lambda m: None)
            model_path = Path(self.tmp.name) / f"{name}.joblib"
            joblib.dump(val_report["router_model"], model_path)
            report, rows, choices = router.run_final(self.corpora, self.f.emb, self.f.out, self.final, name, model_path,
                                                     frozen=self.frozen, log=lambda m: None)
            val = {r["question_id"]: r for r in val_rows if r["split"] == "validation"}
            for r in rows:
                self.assertEqual(r["outcomes"]["unfiltered"], val[r["question_id"]]["outcomes"]["unfiltered"])
                for route, out in r["outcomes"].items():
                    self.assertGreaterEqual(r["outcomes"]["oracle"]["all@5"], out["all@5"])
            self.assertEqual(set(report["methods"]) >= {"tae_r", "tae_r_p1", "unfiltered", "strict", "oracle"}, True)
            self.assertIn("tae_r_vs_unfiltered", report["comparisons"])
            self.assertIn("tae_r_vs_strict", report["comparisons"])
            self.assertEqual(report["methods"]["unfiltered"]["mean_similarity_computations"], report["final_documents"])


if __name__ == "__main__":
    unittest.main()
