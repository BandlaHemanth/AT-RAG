import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = load("prepare_tae_r_embedding_input")


def doc(text):
    return {"doc_id": hashlib.sha256(text.encode()).hexdigest(), "titles": ["T"], "text": text}


class PrepareTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.corpora = self.root / "corpora"
        self.corpora.mkdir()
        parts = {"topic_fit_population": ["f1", "f2"], "retrieval_final": ["a", "shared", "é"],
                 "retrieval_validation": ["b", "shared"]}
        files = {}
        for name, texts in parts.items():
            data = "".join(json.dumps(doc(t), ensure_ascii=False) + "\n" for t in texts).encode()
            (self.corpora / (name + ".jsonl")).write_bytes(data)
            files[name] = {"file": name + ".jsonl", "count": len(texts),
                           "sha256": hashlib.sha256(data).hexdigest()}
        raw = json.dumps({"files": files}).encode()
        (self.corpora / "manifest.json").write_bytes(raw)
        self.frozen = hashlib.sha256(raw).hexdigest()

    def test_union_is_unique_sorted_and_summarised(self):
        summary = prep.prepare(self.corpora, self.root / "out", self.frozen)
        self.assertEqual(summary["unique_paragraphs"], 6)
        self.assertEqual(summary["paragraphs_in_both_retrieval_corpora"], 1)
        raw = (self.root / "out" / "documents.jsonl").read_bytes()
        self.assertEqual(summary["documents_sha256"], hashlib.sha256(raw).hexdigest())
        rows = [json.loads(x) for x in raw.decode().splitlines()]
        self.assertEqual([r["document_id"] for r in rows], sorted(r["document_id"] for r in rows))
        for r in rows:
            self.assertEqual(r["document_id"], hashlib.sha256(r["paragraph_text"].encode()).hexdigest())

    def test_refusals(self):
        with self.assertRaises(ValueError):
            prep.prepare(self.corpora, self.root / "a", "0" * 64)
        prep.prepare(self.corpora, self.root / "b", self.frozen)
        with self.assertRaises(ValueError):
            prep.prepare(self.corpora, self.root / "b", self.frozen)
        path = self.corpora / "retrieval_final.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            prep.prepare(self.corpora, self.root / "c", self.frozen)


if __name__ == "__main__":
    unittest.main()
