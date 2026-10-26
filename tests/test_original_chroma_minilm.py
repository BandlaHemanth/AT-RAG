"""Compatibility checks with stand-ins only: no models, downloads, or real index."""

from contextlib import redirect_stdout, redirect_stderr
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("minilm_ingestion_runner", ROOT / "scripts/ingest_original_2wiki_minilm.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
adapter = runner.ADAPTER


class FakeEncoder:
    instances = []

    def __init__(self, *args, **kwargs):
        self.constructor = (args, kwargs)
        self.calls = []
        self.instances.append(self)

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        rows = [[float(len(text))] + [0.0] * 383 for text in texts]
        return types.SimpleNamespace(tolist=lambda: rows)


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        FakeEncoder.instances.clear()
        adapter.get_embedding_model.cache_clear()
        self.patch = mock.patch.dict(sys.modules, {
            "sentence_transformers": types.SimpleNamespace(SentenceTransformer=FakeEncoder),
        })
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(adapter.get_embedding_model.cache_clear)

    def test_lazy_import_and_identical_document_query_configuration(self):
        embeddings = adapter.get_embedding_model()
        self.assertIs(embeddings, adapter.get_embedding_model())
        self.assertEqual(embeddings.embed_documents([]), [])
        self.assertEqual(FakeEncoder.instances, [])
        texts = [" Café\n東京 ", "duplicate", "duplicate"]
        vectors = embeddings.embed_documents(texts)
        query = embeddings.embed_query(texts[0])
        self.assertEqual(query, vectors[0])
        self.assertEqual(vectors[1], vectors[2])
        self.assertEqual(len(vectors), 3)
        self.assertEqual(len(query), 384)
        encoder = FakeEncoder.instances[0]
        self.assertEqual(encoder.constructor, ((adapter.MODEL_ID,), {
            "revision": adapter.MODEL_REVISION, "device": "cpu", "trust_remote_code": False,
        }))
        self.assertEqual(encoder.max_seq_length, 256)
        self.assertEqual(encoder.calls[0][0], texts)
        self.assertEqual(encoder.calls[1][0], [texts[0]])
        self.assertEqual(encoder.calls[0][1], encoder.calls[1][1])
        self.assertTrue(encoder.calls[0][1]["normalize_embeddings"])
        self.assertEqual(encoder.calls[0][1]["precision"], "float32")
        self.assertEqual(encoder.calls[0][1]["batch_size"], 32)

    def test_invalid_vectors_fail_instead_of_entering_chroma(self):
        embeddings = adapter.MiniLMEmbeddings()
        for rows in ([[0.0]], [[float("nan")] * 384], []):
            with self.subTest(rows=len(rows)):
                embeddings._encoder = types.SimpleNamespace(encode=lambda *a, rows=rows, **k: types.SimpleNamespace(tolist=lambda: rows))
                with self.assertRaises(ValueError):
                    embeddings.embed_documents(["one"])
        with self.assertRaises(TypeError):
            adapter.MiniLMEmbeddings().embed_query(None)

    def test_importing_runner_loads_no_model_libraries(self):
        code = (
            "import runpy,sys; runpy.run_path(sys.argv[1],run_name='import_only'); "
            "assert not {'bertopic','sentence_transformers','torch','chromadb','transformers'} & set(sys.modules)"
        )
        completed = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT / "scripts/ingest_original_2wiki_minilm.py")], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_langchain_interface_preserves_similarity_filter_and_total_top5(self):
        from langchain_community.vectorstores import Chroma

        # Bypass the constructor: no Chroma client/collection/index is created.
        store = Chroma.__new__(Chroma)
        store._embedding_function = adapter.MiniLMEmbeddings()
        store._collection = mock.Mock()
        store._collection.query.return_value = {
            "documents": [["duplicate"]], "metadatas": [[{"bertopic": "Topic 11"}]],
            "distances": [[0.1]],
        }
        store.add_texts(["duplicate", "duplicate"], metadatas=[{"bertopic": "Topic 11"}] * 2)
        inserted = store._collection.upsert.call_args.kwargs
        self.assertEqual(inserted["documents"], ["duplicate", "duplicate"])
        self.assertEqual(len(set(inserted["ids"])), 2)
        self.assertEqual(len(inserted["embeddings"]), 2)
        retriever = store.as_retriever(search_type="similarity", search_kwargs={
            "filter": {"bertopic": "Topic 11"}, "k": 5,
        })
        documents = retriever.invoke("query")
        query = store._collection.query.call_args.kwargs
        self.assertEqual(query["where"], {"bertopic": "Topic 11"})
        self.assertEqual(query["n_results"], 5)
        self.assertEqual(len(query["query_embeddings"][0]), 384)
        self.assertEqual(documents[0].page_content, "duplicate")


class IngestionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.dataset = self.directory / "test_subsampled.jsonl"
        self.topic_model = self.directory / "saved-topic-model"
        self.topic_model.write_bytes(b"stand-in model; never unpickled")
        self.training_requirements = self.directory / "training-freeze.txt"
        self.destination = self.directory / "chroma"
        self.rows = [{"question_id": str(q), "contexts": [
            {"paragraph_text": f"duplicate-{i % 2}", "idx": i,
             "title": f"title-{q}", "is_supporting": i % 3 == 0}
            for i in range(10)]} for q in range(500)]
        # Exercise the original metadata defaults without changing document order.
        del self.rows[0]["contexts"][0]["title"]
        del self.rows[0]["contexts"][0]["is_supporting"]
        self.dataset.write_text("".join(json.dumps(row) + "\n" for row in self.rows))

    def original_dependencies(self):
        loaded = []
        classified = []
        stores = []
        topics = [-1, 0, 11, 0, 1] * 1000

        class TopicModel:
            def transform(inner, texts):
                classified.append(list(texts))
                return topics, None

            def fit_transform(inner, *args, **kwargs):
                raise AssertionError("Training is forbidden")

        class BERTopic:
            @staticmethod
            def load(path):
                loaded.append(Path(path).resolve())
                return TopicModel()

        class Chroma:
            def __init__(inner, **kwargs):
                inner.kwargs = kwargs
                inner.batches = []
                inner.persisted = False
                stores.append(inner)

            def add_texts(inner, **kwargs):
                # Capture calls only: do not compute embeddings or build an index.
                inner.batches.append(kwargs)

            def persist(inner):
                inner.persisted = True

        modules = {
            "decouple": types.SimpleNamespace(config=lambda name: os.environ[name]),
            "bertopic": types.SimpleNamespace(BERTopic=BERTopic),
            "transformers": types.SimpleNamespace(AutoTokenizer=object, AutoModel=object),
            "langchain": types.ModuleType("langchain"),
            "langchain.embeddings": types.SimpleNamespace(OpenAIEmbeddings=object),
            "langchain.vectorstores": types.SimpleNamespace(Chroma=Chroma),
        }
        return modules, loaded, classified, stores, topics

    def test_original_ingestion_keeps_5000_occurrences_metadata_topics_and_batches(self):
        modules, loaded, classified, stores, topics = self.original_dependencies()
        before_path, before_cwd = sys.path[:], Path.cwd()
        with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ, {}, clear=True):
            with redirect_stdout(io.StringIO()):
                database = runner.create_original_store(self.dataset, self.topic_model, self.destination)
            self.assertNotIn("OPENAI_API_KEY", os.environ)
        self.assertEqual(Path.cwd(), before_cwd)
        self.assertEqual(sys.path, before_path)
        self.assertNotIn("utils", sys.modules)
        self.assertNotIn("train_topic_model", sys.modules)
        self.assertEqual(loaded, [self.topic_model])
        texts = [c["paragraph_text"] for row in self.rows for c in row["contexts"]]
        self.assertEqual(classified, [texts])
        self.assertIs(database, stores[0])
        self.assertEqual(database.kwargs, {
            "persist_directory": str(self.destination),
            "embedding_function": adapter.get_embedding_model(),
        })
        self.assertEqual([len(b["texts"]) for b in database.batches], [1000] * 5)
        self.assertTrue(all(set(b) == {"texts", "metadatas"} for b in database.batches))
        self.assertEqual([t for b in database.batches for t in b["texts"]], texts)
        metadatas = [m for b in database.batches for m in b["metadatas"]]
        expected = [{"idx": c["idx"], "title": c.get("title", "Unknown"),
                     "is_supporting": c.get("is_supporting", "N/A"), "bertopic": f"Topic {topic}"}
                    for c, topic in zip([c for row in self.rows for c in row["contexts"]], topics)]
        self.assertEqual(metadatas, expected)
        self.assertEqual(len(texts), 5000)
        self.assertEqual(len(set(texts)), 2)
        self.assertTrue(database.persisted)

    def test_existing_destination_is_not_appended(self):
        self.destination.mkdir()
        with self.assertRaises(FileExistsError):
            runner.create_original_store(self.dataset, self.topic_model, self.destination)

    def test_broken_destination_symlink_is_rejected(self):
        self.destination.symlink_to(self.directory / "missing")
        with self.assertRaisesRegex(ValueError, "symlink"):
            runner.inspect_inputs(self.dataset, self.topic_model, self.destination)
        with self.assertRaisesRegex(ValueError, "symlink"):
            runner.create_original_store(self.dataset, self.topic_model, self.destination)

    def test_import_failure_restores_key_path_modules_and_cwd(self):
        modules, *_ = self.original_dependencies()
        before_path, before_cwd = sys.path[:], Path.cwd()
        original_loader = runner.load_module

        def fail(name, path):
            if name == "_at_rag_original_ingestion":
                raise ImportError("deliberate stand-in failure")
            return original_loader(name, path)

        with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(runner, "load_module", side_effect=fail):
                with self.assertRaises(ImportError):
                    runner.create_original_store(self.dataset, self.topic_model, self.destination)
            self.assertNotIn("OPENAI_API_KEY", os.environ)
        self.assertEqual(sys.path, before_path)
        self.assertEqual(Path.cwd(), before_cwd)
        self.assertNotIn("utils", sys.modules)

    def test_existing_api_key_is_preserved(self):
        modules, *_ = self.original_dependencies()
        with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-not-a-credential"}):
            with redirect_stdout(io.StringIO()):
                runner.create_original_store(self.dataset, self.topic_model, self.destination)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "test-not-a-credential")

    def test_preflight_is_read_only_and_rejects_wrong_artifacts(self):
        with mock.patch.object(runner, "DATASET_SHA256", runner.file_sha256(self.dataset)), mock.patch.object(runner, "TOPIC_MODEL_SHA256", runner.file_sha256(self.topic_model)):
            info = runner.inspect_inputs(self.dataset, self.topic_model, self.destination)
            self.assertEqual((info["questions"], info["context_occurrences"]), (500, 5000))
            self.assertFalse(self.destination.exists())
            self.topic_model.write_bytes(b"wrong model")
            with self.assertRaisesRegex(ValueError, "BERTopic SHA-256"):
                runner.inspect_inputs(self.dataset, self.topic_model, self.destination)
        with self.assertRaisesRegex(ValueError, "Dataset SHA-256"):
            runner.inspect_inputs(self.dataset, self.topic_model, self.destination)

    def test_cli_defaults_to_preflight_without_entering_ingestion(self):
        with mock.patch.object(runner, "inspect_inputs", return_value={}), mock.patch.object(runner, "inspect_environment", return_value={}):
            with mock.patch.object(runner, "create_original_store", side_effect=AssertionError("Must not index")) as create:
                with redirect_stdout(io.StringIO()) as output:
                    result = runner.main(["--topic-model", str(self.topic_model), "--training-requirements", str(self.training_requirements)])
                self.assertEqual(result, 0)
                self.assertEqual(json.loads(output.getvalue())["status"], "preflight_passed")
                create.assert_not_called()

    def test_execute_verifies_record_count_and_only_then_writes_manifest(self):
        def captured_store(*args):
            self.destination.mkdir()
            return types.SimpleNamespace(_collection=types.SimpleNamespace(count=lambda: 5000))

        arguments = ["--topic-model", str(self.topic_model), "--training-requirements", str(self.training_requirements), "--persist-directory", str(self.destination), "--execute"]
        with mock.patch.object(runner, "inspect_inputs", return_value={}), mock.patch.object(runner, "inspect_environment", return_value={}):
            with mock.patch.object(runner, "create_original_store", side_effect=captured_store) as create, redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(arguments), 0)
                create.assert_called_once()
        report = json.loads((self.destination / "minilm_ingestion_manifest.json").read_text())
        self.assertEqual(report["stored_records"], 5000)
        self.assertEqual(report["retrieval_embedding_substitution"], adapter.embedding_settings())
        self.assertEqual(len(report["compatibility_file_sha256"]), 3)

    def test_failed_record_count_does_not_publish_success_manifest(self):
        store = types.SimpleNamespace(_collection=types.SimpleNamespace(count=lambda: 4999))
        arguments = ["--topic-model", str(self.topic_model), "--training-requirements", str(self.training_requirements), "--persist-directory", str(self.destination), "--execute"]
        with mock.patch.object(runner, "inspect_inputs", return_value={}), mock.patch.object(runner, "inspect_environment", return_value={}):
            with mock.patch.object(runner, "create_original_store", return_value=store), redirect_stderr(io.StringIO()):
                self.assertEqual(runner.main(arguments), 1)
        self.assertFalse((self.destination / "minilm_ingestion_manifest.json").exists())

    def test_environment_mismatch_fails_before_ingestion(self):
        arguments = ["--topic-model", str(self.topic_model), "--training-requirements", str(self.training_requirements), "--execute"]
        with mock.patch.object(runner, "inspect_inputs", return_value={}), mock.patch.object(runner, "inspect_environment", side_effect=ValueError("Dependency mismatch: torch")):
            with mock.patch.object(runner, "create_original_store") as create, redirect_stderr(io.StringIO()):
                self.assertEqual(runner.main(arguments), 1)
                create.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_original_sources_are_unchanged(self):
        for relative, expected in runner.SOURCE_SHA256.items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected, relative)


class TrainingEnvironmentTests(unittest.TestCase):
    """Synthetic metadata only; do not install the newer training stack in Cloud."""

    SCIENTIFIC = {
        "bertopic": "0.16.3", "transformers": "4.45.1",
        "sentence-transformers": "5.7.0", "torch": "2.14.1", "numpy": "1.26.4",
        "scikit-learn": "1.9.1", "scipy": "1.17.1", "umap-learn": "0.5.12",
        "hdbscan": "0.8.44", "numba": "0.68.0", "llvmlite": "0.50.0",
    }

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.freeze = Path(temporary.name) / "synthetic-training-freeze.txt"
        # The other 51 names are intentionally fictional test fixtures, not a
        # fabricated reconstruction of the unavailable real Kaggle freeze.
        self.training = {**self.SCIENTIFIC, **{f"fixture-package-{i}": "1.0" for i in range(51)}}
        self.profile = runner.read_exact_pins(runner.PROFILE_PATH)
        self.installed = {**self.profile, **self.training}
        self.write_freeze()
        patches = [
            mock.patch.object(runner.platform, "python_version", return_value="3.12.3"),
            mock.patch.object(runner.metadata, "version", side_effect=self.installed_version),
            mock.patch.object(runner.subprocess, "run", return_value=types.SimpleNamespace(returncode=0, stdout="No broken requirements found.", stderr="")),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def installed_version(self, name):
        if name not in self.installed:
            raise runner.metadata.PackageNotFoundError(name)
        return self.installed[name]

    def write_freeze(self):
        self.freeze.write_text("".join(f"{name}=={version}\n" for name, version in self.training.items()))

    def test_profile_preserves_all_eleven_supplied_scientific_versions(self):
        self.assertEqual(set(runner.SCIENTIFIC_PACKAGES), set(self.SCIENTIFIC))
        self.assertEqual({name: self.profile[name] for name in self.SCIENTIFIC}, self.SCIENTIFIC)
        self.assertNotIn("huggingface-hub", self.profile)
        self.assertNotIn("pandas", self.profile)

    def test_matching_complete_freeze_records_provenance(self):
        result = runner.inspect_environment(self.freeze)
        self.assertEqual(result["python"], "3.12.3")
        self.assertEqual(result["verified_training_packages"], 62)
        self.assertEqual(result["training_requirements_sha256"], runner.file_sha256(self.freeze))
        self.assertEqual(result["packages"]["sentence-transformers"], "5.7.0")
        self.assertEqual(result["packages"]["torch"], "2.14.1")

    def test_each_scientific_version_mismatch_is_rejected(self):
        for name in self.SCIENTIFIC:
            with self.subTest(package=name):
                self.installed[name] = "0.0"
                with self.assertRaisesRegex(ValueError, f"Dependency mismatch: {name}"):
                    runner.inspect_environment(self.freeze)
                self.installed[name] = self.SCIENTIFIC[name]

    def test_python_patch_and_minor_mismatches_are_rejected(self):
        for version in ("3.12.14", "3.13.0"):
            with self.subTest(python=version), mock.patch.object(runner.platform, "python_version", return_value=version):
                with self.assertRaisesRegex(ValueError, "Python mismatch"):
                    runner.inspect_environment(self.freeze)

    def test_preflight_also_checks_the_other_backed_up_packages(self):
        self.installed["fixture-package-50"] = "2.0"
        with self.assertRaisesRegex(ValueError, "Dependency mismatch: fixture-package-50"):
            runner.inspect_environment(self.freeze)

    def test_missing_package_is_rejected(self):
        del self.installed["torch"]
        with self.assertRaisesRegex(ValueError, "Missing dependency: torch"):
            runner.inspect_environment(self.freeze)

    def test_installed_transitive_conflicts_are_rejected(self):
        failed = types.SimpleNamespace(returncode=1, stdout="fixture dependency needs another version", stderr="")
        with mock.patch.object(runner.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(ValueError, "Installed dependency conflict"):
                runner.inspect_environment(self.freeze)

    def test_partial_training_freeze_is_rejected(self):
        self.training = self.SCIENTIFIC.copy()
        self.write_freeze()
        with self.assertRaisesRegex(ValueError, "complete 62-package training freeze"):
            runner.inspect_environment(self.freeze)

    def test_profile_conflict_is_reported_without_changing_training_versions(self):
        self.training["sentence-transformers"] = "3.1.1"
        self.write_freeze()
        before = self.freeze.read_bytes()
        with self.assertRaisesRegex(ValueError, "Training/profile conflict: sentence-transformers"):
            runner.inspect_environment(self.freeze)
        self.assertEqual(self.freeze.read_bytes(), before)

    def test_cuda_local_build_is_preserved_when_recorded(self):
        self.training["torch"] = "2.14.1+cu130"
        self.installed["torch"] = "2.14.1+cu130"
        self.write_freeze()
        self.assertEqual(runner.inspect_environment(self.freeze)["packages"]["torch"], "2.14.1+cu130")
        self.installed["torch"] = "2.14.1+cpu"
        with self.assertRaisesRegex(ValueError, "Dependency mismatch: torch"):
            runner.inspect_environment(self.freeze)

    def test_ambiguous_requirement_syntax_is_not_silently_ignored(self):
        for content in ("torch>=2.14\n", "torch @ https://example.invalid/torch.whl\n", "numpy==1.*\n", "foo_bar==1.0\nfoo-bar==1.0\n"):
            with self.subTest(content=content):
                self.freeze.write_text(content)
                with self.assertRaises(ValueError):
                    runner.read_exact_pins(self.freeze)


if __name__ == "__main__":
    unittest.main()
