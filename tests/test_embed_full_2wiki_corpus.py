"""No model downloads/GPU needed: mock encode() and exercise actual mmap I/O."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/embed_full_2wiki_corpus.py"
SPEC = importlib.util.spec_from_file_location("embed_full_2wiki_corpus", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class MockEncoder:
    def __init__(self, fail_on_call=None, invalid=None):
        self.calls = []
        self.fail_on_call = fail_on_call
        self.invalid = invalid

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("simulated interruption")
        rows = []
        for text in texts:
            vector = np.zeros(384, dtype=np.float32)
            vector[:32] = np.frombuffer(hashlib.sha256(text.encode("utf-8")).digest(), dtype=np.uint8)
            vector /= np.linalg.norm(vector)
            rows.append(vector)
        result = np.stack(rows)
        if self.invalid == "shape":
            return result[:, :-1]
        if self.invalid == "dtype":
            return result.astype(np.float64)
        if self.invalid == "nan":
            result[0, 0] = np.nan
        if self.invalid == "normalization":
            result *= 2
        return result


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "documents.jsonl"
        self.summary = self.root / "summary.json"
        self.output = self.root / "embeddings"
        self.work = self.root / "embeddings.inprogress"
        self.texts = ["first", " Café 東京 ", "third\nparagraph", "fourth", "fifth", "sixth", "last"]
        self.ids = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in self.texts]
        self.rows = [{"document_id": doc_id, "paragraph_text": text, "title": "ignored",
                      "is_supporting": False, "reasoning_steps": ["ignored"]}
                     for doc_id, text in zip(self.ids, self.texts)]
        self.write_corpus()
        self.expected_sha = builder.file_sha256(self.source)
        self.encoder = MockEncoder()
        self.factory = mock.Mock(return_value=self.encoder)
        self.environment = {"python": "mock", "libraries": {"numpy": np.__version__},
                            "device_identity": "mock CPU"}
        patch = mock.patch.object(builder, "runtime_environment", return_value=self.environment)
        patch.start()
        self.addCleanup(patch.stop)

    def write_corpus(self):
        self.source.write_bytes(b"\n".join(builder.json_bytes(row).rstrip(b"\n") for row in self.rows))
        self.summary.write_bytes(builder.json_bytes({"documents_sha256": builder.file_sha256(self.source),
                                                     "unique_paragraphs": len(self.rows)}))

    def run_job(self, **options):
        defaults = {"expected_sha256": self.expected_sha, "expected_count": len(self.rows),
                    "batch_size": 2, "checkpoint_batches": 1, "encoder_factory": self.factory}
        defaults.update(options)
        output = defaults.pop("output_dir", self.output)
        return builder.run_job(self.source, output, **defaults)

    def interrupt(self, checkpoint_batches=1):
        self.encoder.fail_on_call = 4
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            self.run_job(checkpoint_batches=checkpoint_batches)
        self.assertFalse(self.output.exists())
        self.assertFalse((self.work / "artifacts/manifest.json").exists())

    def test_batches_original_text_order_ids_and_manifest_integrity(self):
        manifest = self.run_job()
        self.assertEqual([text for texts, _ in self.encoder.calls for text in texts], self.texts)
        self.assertEqual([len(texts) for texts, _ in self.encoder.calls], [2, 2, 2, 1])
        for _, options in self.encoder.calls:
            self.assertEqual(options, {"batch_size": 2, "device": "cpu", "normalize_embeddings": True,
                                       "precision": "float32", "convert_to_numpy": True,
                                       "show_progress_bar": False})
        vectors = np.load(self.output / "embeddings.npy", mmap_mode="r", allow_pickle=False)
        self.assertIsInstance(vectors, np.memmap)
        self.assertEqual(vectors.shape, (7, 384))
        self.assertEqual(vectors.dtype, np.dtype("float32"))
        np.testing.assert_array_equal(vectors, MockEncoder().encode(self.texts))
        mapping = [json.loads(line) for line in (self.output / "document_ids.jsonl").read_bytes().splitlines()]
        self.assertEqual(mapping, [{"row": row, "document_id": doc_id} for row, doc_id in enumerate(self.ids)])
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["processed_count"], 7)
        config = manifest["configuration"]
        self.assertEqual(config["model_id"], "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(config["model_revision"], "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
        self.assertEqual(config["corpus_sha256"], self.expected_sha)
        self.assertEqual(config["max_seq_length"], 256)
        self.assertEqual(config["dimensions"], 384)
        self.assertEqual(config["dtype"], "float32")
        self.assertTrue(config["normalize_embeddings"])
        self.assertEqual(config["environment"], self.environment)
        for name, evidence in manifest["artifacts"].items():
            self.assertEqual(evidence["sha256"], builder.file_sha256(self.output / name))
            self.assertEqual(evidence["bytes"], (self.output / name).stat().st_size)
        self.assertEqual(json.loads((self.output / "manifest.json").read_bytes()), manifest)
        self.assertFalse(self.work.exists())
        self.assertEqual(sorted(path.name for path in self.output.iterdir()),
                         ["document_ids.jsonl", "embeddings.npy", "manifest.json"])

    def test_resume_reencodes_only_uncommitted_tail_and_matches_uninterrupted_bytes(self):
        self.interrupt(checkpoint_batches=2)
        state = json.loads((self.work / "checkpoint.json").read_bytes())
        self.assertEqual(state["processed_count"], 4)  # Six rows written, only four committed.
        # Model a torn final mapping line and bad, uncommitted matrix pages.
        with (self.work / "artifacts/document_ids.jsonl").open("ab") as stream:
            stream.write(b'{"row":')
        matrix = np.load(self.work / "artifacts/embeddings.npy", mmap_mode="r+")
        matrix[4:] = np.nan
        matrix.flush()
        del matrix
        resumed = MockEncoder()
        self.run_job(resume=True, checkpoint_batches=2, encoder_factory=lambda device: resumed)
        self.assertEqual([text for texts, _ in resumed.calls for text in texts], self.texts[4:])
        other = self.root / "uninterrupted"
        self.run_job(output_dir=other, checkpoint_batches=2, encoder_factory=lambda device: MockEncoder())
        for name in ("embeddings.npy", "document_ids.jsonl", "manifest.json"):
            self.assertEqual((self.output / name).read_bytes(), (other / name).read_bytes())

    def test_checkpoint_publication_failure_keeps_previous_progress(self):
        original = builder.atomic_json
        calls = 0

        def fail_checkpoint(path, value):
            nonlocal calls
            if path.name == "checkpoint.json":
                calls += 1
                if calls == 3:
                    raise OSError("checkpoint disk failure")
            return original(path, value)

        with mock.patch.object(builder, "atomic_json", side_effect=fail_checkpoint):
            with self.assertRaisesRegex(OSError, "checkpoint disk failure"):
                self.run_job()
        self.assertEqual(json.loads((self.work / "checkpoint.json").read_bytes())["processed_count"], 2)
        fresh = MockEncoder()
        self.run_job(resume=True, encoder_factory=lambda device: fresh)
        self.assertEqual([text for texts, _ in fresh.calls for text in texts], self.texts[2:])

    def test_completed_rerun_is_refused_without_modification(self):
        self.run_job()
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        for resume in (False, True):
            with self.assertRaises(FileExistsError):
                self.run_job(resume=resume)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})
        self.assertEqual(self.factory.call_count, 1)

    def test_existing_empty_output_is_refused(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_job()
        self.factory.assert_not_called()

    def test_resume_must_be_explicit(self):
        self.interrupt()
        with self.assertRaises(FileExistsError):
            self.run_job()

    def test_incompatible_batch_device_model_and_environment_are_refused(self):
        self.interrupt()
        for changed in ({"batch_size": 3}, {"checkpoint_batches": 2}, {"device": "cuda:0"}):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(ValueError, "differs from the saved job"):
                    self.run_job(resume=True, **changed)
        with mock.patch.object(builder, "MODEL_REVISION", "different-revision"):
            with self.assertRaisesRegex(ValueError, "differs from the saved job"):
                self.run_job(resume=True)
        with mock.patch.object(builder, "runtime_environment", return_value={"different": "runtime"}):
            with self.assertRaisesRegex(ValueError, "differs from the saved job"):
                self.run_job(resume=True)

    def test_corpus_checksum_and_count_fail_before_encoder_or_workspace(self):
        for options in ({"expected_sha256": "0" * 64}, {"expected_count": 8}):
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, "summary disagrees"):
                    self.run_job(**options)
        # Keep the original summary while changing actual bytes, to test hashing.
        with self.source.open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "Corpus verification failed"):
            self.run_job()
        self.assertFalse(self.work.exists())
        self.factory.assert_not_called()

    def test_invalid_document_id_is_rejected(self):
        self.rows[0]["document_id"] = "f" * 64
        self.write_corpus()
        with self.assertRaisesRegex(ValueError, "document_id does not match"):
            self.run_job(expected_sha256=builder.file_sha256(self.source))
        self.factory.assert_not_called()

    def test_new_valid_corpus_cannot_be_mixed_into_existing_job(self):
        self.interrupt()
        self.rows[-1]["paragraph_text"] = "different last paragraph"
        self.rows[-1]["document_id"] = hashlib.sha256(b"different last paragraph").hexdigest()
        self.write_corpus()
        with self.assertRaisesRegex(ValueError, "differs from the saved job"):
            self.run_job(resume=True, expected_sha256=builder.file_sha256(self.source))

    def test_truncated_matrix_is_not_resumed(self):
        self.interrupt()
        with (self.work / "artifacts/embeddings.npy").open("r+b") as stream:
            stream.truncate(128)
        with self.assertRaises(ValueError):
            self.run_job(resume=True)

    def test_initialization_failure_can_resume_from_zero(self):
        original = builder.atomic_json

        def fail_initial_checkpoint(path, value):
            if path.name == "checkpoint.json":
                raise OSError("initial checkpoint failed")
            return original(path, value)

        with mock.patch.object(builder, "atomic_json", side_effect=fail_initial_checkpoint):
            with self.assertRaisesRegex(OSError, "initial checkpoint failed"):
                self.run_job()
        self.factory.assert_not_called()
        self.run_job(resume=True)
        self.assertEqual([text for texts, _ in self.encoder.calls for text in texts], self.texts)

    def test_corrupt_committed_embeddings_are_rejected_before_encoder(self):
        self.interrupt()
        path = self.work / "artifacts/embeddings.npy"
        matrix = np.load(path, mmap_mode="r+")
        matrix[0] = np.roll(matrix[0], 1)  # Still normalized, but not the committed bytes.
        matrix.flush()
        del matrix
        factory = mock.Mock()
        with self.assertRaisesRegex(ValueError, "prefix checksum mismatch"):
            self.run_job(resume=True, encoder_factory=factory)
        factory.assert_not_called()

    def test_corrupt_committed_mapping_is_rejected_before_encoder(self):
        self.interrupt()
        path = self.work / "artifacts/document_ids.jsonl"
        path.write_bytes(path.read_bytes().replace(self.ids[0].encode(), b"f" * 64, 1))
        with self.assertRaisesRegex(ValueError, "prefix checksum mismatch"):
            self.run_job(resume=True)

    def test_malformed_checkpoint_is_rejected(self):
        self.interrupt()
        path = self.work / "checkpoint.json"
        original = json.loads(path.read_bytes())
        for changed in ({"processed_count": -1}, {"processed_count": 100},
                        {"embedding_prefix_sha256": None}, {"mapping_bytes": -1}):
            with self.subTest(changed=changed):
                path.write_bytes(builder.json_bytes({**original, **changed}))
                with self.assertRaisesRegex(ValueError, "Invalid or incompatible"):
                    self.run_job(resume=True)

    def test_invalid_encoder_outputs_never_publish_success(self):
        for invalid in ("shape", "dtype", "nan", "normalization"):
            with self.subTest(invalid=invalid):
                output = self.root / invalid
                with self.assertRaises(ValueError):
                    self.run_job(output_dir=output, encoder_factory=lambda device: MockEncoder(invalid=invalid))
                self.assertFalse(output.exists())
                self.assertEqual(json.loads((self.root / (invalid + ".inprogress/checkpoint.json")).read_bytes())[
                    "processed_count"], 0)

    def test_input_changed_during_encoding_cannot_publish(self):
        original = self.encoder.encode

        def change_source(*args, **kwargs):
            with self.source.open("ab") as stream:
                stream.write(b"\n")
            return original(*args, **kwargs)

        self.encoder.encode = change_source
        with self.assertRaisesRegex(ValueError, "checksum/count changed"):
            self.run_job()
        self.assertFalse(self.output.exists())

    def test_final_rename_failure_can_resume_without_reencoding(self):
        with mock.patch.object(builder.os, "rename", side_effect=OSError("publish failed")):
            with self.assertRaisesRegex(OSError, "publish failed"):
                self.run_job()
        self.assertFalse(self.output.exists())
        factory = mock.Mock(side_effect=AssertionError("Must not reencode committed rows"))
        self.run_job(resume=True, encoder_factory=factory)
        factory.assert_not_called()

    def test_concurrent_completed_output_is_preserved(self):
        rename = builder.os.rename

        def concurrent_publish(source, destination):
            Path(destination).mkdir()
            (Path(destination) / "manifest.json").write_bytes(b"another run")
            return rename(source, destination)

        with mock.patch.object(builder.os, "rename", side_effect=concurrent_publish):
            with self.assertRaises(OSError):
                self.run_job()
        self.assertEqual((self.output / "manifest.json").read_bytes(), b"another run")

    def test_workspace_lock_blocks_concurrent_writer(self):
        self.work.mkdir()
        with builder.job_lock(self.work):
            with self.assertRaisesRegex(ValueError, "Another process"):
                self.run_job(resume=True)
        self.factory.assert_not_called()

    def test_streaming_uses_iteration_instead_of_whole_source_reads(self):
        original_open = Path.open
        source = self.source

        class IterationOnly:
            def __enter__(self):
                self.stream = original_open(source, "rb")
                return self

            def __exit__(self, *args):
                self.stream.close()

            def __iter__(self):
                return iter(self.stream)

            def read(self, *args):
                raise AssertionError("Whole-corpus read is forbidden")

            readlines = read

        def guarded_open(path, *args, **kwargs):
            return IterationOnly() if path == source else original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", guarded_open):
            self.run_job()
        self.assertLessEqual(max(len(texts) for texts, _ in self.encoder.calls), 2)

    def test_production_loader_pins_model_revision_and_pilot_settings(self):
        torch = mock.MagicMock()
        encoder = mock.MagicMock()
        encoder.get_sentence_embedding_dimension.return_value = 384
        constructor = mock.Mock(return_value=encoder)
        module = types.SimpleNamespace(SentenceTransformer=constructor)
        with mock.patch.dict(sys.modules, {"torch": torch, "sentence_transformers": module}):
            loaded = builder.load_encoder("cpu")
        self.assertIs(loaded, encoder)
        constructor.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2",
                                            revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                                            device="cpu", trust_remote_code=False)
        self.assertEqual(encoder.max_seq_length, 256)
        encoder.eval.assert_called_once()
        torch.manual_seed.assert_called_once_with(42)
        torch.use_deterministic_algorithms.assert_called_once_with(True)

    def test_cpu_cuda_selection_is_explicit(self):
        self.assertEqual(builder.canonical_device("cuda"), "cuda:0")
        self.assertEqual(builder.canonical_device("cuda:1"), "cuda:1")
        for device in ("auto", "gpu", "cuda:-1"):
            with self.assertRaises(ValueError):
                self.run_job(device=device)

    def test_cuda_device_is_passed_to_loader_and_every_batch(self):
        manifest = self.run_job(device="cuda:1", batch_size=3)
        self.factory.assert_called_once_with("cuda:1")
        self.assertEqual([len(texts) for texts, _ in self.encoder.calls], [3, 3, 1])
        self.assertTrue(all(options["device"] == "cuda:1" for _, options in self.encoder.calls))
        self.assertEqual(manifest["configuration"]["device"], "cuda:1")

    def test_cli_help_and_bad_checksum_need_no_model(self):
        help_result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--help"],
                                     capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        failed = subprocess.run([sys.executable, "-B", str(SCRIPT), "--input", str(self.source),
                                 "--output-dir", str(self.output)], capture_output=True, text=True)
        self.assertEqual(failed.returncode, 1)
        self.assertIn("summary disagrees", failed.stderr)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
