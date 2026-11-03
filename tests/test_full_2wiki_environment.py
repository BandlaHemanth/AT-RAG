"""Metadata/installer stand-ins only; no installs, model loads, or training."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("full_profiles", ROOT / "scripts/full_2wiki_environment.py")
e = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e)


class ProfileTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.freeze = self.root / "original-62.txt"
        self.profile = e.original_helper().read_exact_pins(e.PROFILE_ROOT / "training-py312.txt")
        # Fictional remaining names: not a reconstruction of the private backup.
        self.recorded = {**self.profile, **{f"fixture-{i}": "1.0" for i in range(50)}}
        self.write_freeze()
        self.run = mock.Mock(return_value=types.SimpleNamespace(returncode=0, stdout="ok", stderr=""))
        for patch in (mock.patch.object(e.platform, "python_version", return_value="3.12.3"),
                      mock.patch.object(e.subprocess, "run", self.run)):
            patch.start()
            self.addCleanup(patch.stop)

    def write_freeze(self):
        self.freeze.write_text("".join(f"{name}=={version}\n" for name, version in self.recorded.items()))

    def inspect(self, profile="training", installed=None):
        packages = dict(e.plan(self.freeze, profile)["expected_packages"] if installed is None else installed)
        with mock.patch.object(e, "package_versions", return_value=packages):
            return e.inspect_environment(self.freeze, profile)

    def test_training_preserves_all_62_without_retrieval_packages_or_packaging_downgrade(self):
        before = self.freeze.read_bytes()
        result = self.inspect()
        self.assertEqual(result["packages"], self.recorded)
        self.assertEqual(result["packages"]["packaging"], "26.3")
        self.assertFalse(any(e.integration(name) for name in result["expected_packages"]))
        self.assertEqual(result["deliberate_pin_changes"], {})
        self.assertEqual(result["pip_check"], "passed")
        self.assertEqual(self.freeze.read_bytes(), before)

    def test_scientific_pins_match_the_original_training_record_and_merged_profile(self):
        old = e.original_helper()
        old_pins = old.read_exact_pins(old.PROFILE_PATH)
        for name in old.SCIENTIFIC_PACKAGES:
            self.assertEqual(self.profile[name], old_pins[name])
            original = self.recorded[name]
            self.recorded[name] = "0.0"
            self.write_freeze()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Recorded training pin mismatch"):
                e.plan(self.freeze, "training")
            self.recorded[name] = original
        self.write_freeze()

    def test_installed_scientific_and_other_recorded_mismatches_are_not_ignored(self):
        for name in ("sentence-transformers", "torch", "scikit-learn", "fixture-49", "packaging"):
            packages = {**self.recorded, name: "0.0"}
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, f"Dependency mismatch: {name}"):
                self.inspect(installed=packages)

    def test_training_rejects_inherited_langchain_or_chroma(self):
        for name in ("langchain", "langchain-core", "chromadb"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "contains retrieval packages"):
                self.inspect(installed={**self.recorded, name: "1.0"})

    def test_freeze_integration_exclusion_is_explicit_and_recorded(self):
        del self.recorded["fixture-49"]
        self.recorded["langchain"] = "0.3.2"
        self.write_freeze()
        result = e.plan(self.freeze, "training")
        self.assertEqual(result["excluded_integration_pins"], {"langchain": "0.3.2"})
        self.assertNotIn("langchain", result["expected_packages"])

    def test_retrieval_changes_only_packaging_and_preserves_model_contract(self):
        train, retrieval = self.inspect(), self.inspect("retrieval")
        self.assertEqual(retrieval["packages"]["packaging"], "24.2")
        self.assertEqual(set(retrieval["deliberate_pin_changes"]), {"packaging"})
        self.assertEqual(retrieval["packages"]["langchain"], "0.3.2")
        self.assertEqual(retrieval["packages"]["langgraph"], "0.2.28")
        self.assertEqual(e.model_contract(train), e.model_contract(retrieval))
        retrieval["packages"]["fixture-49"] = "other"
        self.assertNotEqual(e.model_contract(train), e.model_contract(retrieval))

    def test_additional_profile_conflict_fails_instead_of_upgrading_a_recorded_dependency(self):
        del self.recorded["fixture-49"]
        self.recorded["httpx"] = "0.28.1"
        self.write_freeze()
        with self.assertRaisesRegex(ValueError, "Unapproved profile/freeze conflict: httpx"):
            e.plan(self.freeze, "retrieval")

    def test_local_torch_build_and_complete_record_remain_required(self):
        self.recorded["torch"] = "2.14.1+cu130"
        self.write_freeze()
        self.assertEqual(self.inspect()["packages"]["torch"], "2.14.1+cu130")
        with self.assertRaisesRegex(ValueError, "Dependency mismatch: torch"):
            self.inspect(installed={**self.recorded, "torch": "2.14.1+cpu"})
        del self.recorded["fixture-49"]
        self.write_freeze()
        with self.assertRaisesRegex(ValueError, "complete 62-package"):
            e.plan(self.freeze, "training")

    def test_python_and_transitive_dependency_conflicts_fail(self):
        with mock.patch.object(e.platform, "python_version", return_value="3.13.0"):
            with self.assertRaisesRegex(ValueError, "Python mismatch"):
                self.inspect()
        self.run.return_value = types.SimpleNamespace(returncode=1, stdout="conflict", stderr="")
        with self.assertRaisesRegex(ValueError, "Installed dependency conflict"):
            self.inspect()

    def test_install_resolves_first_then_checks_versions_and_imports_without_training(self):
        destination = self.root / "training-venv"
        before = self.freeze.read_bytes()
        with mock.patch("builtins.print"):
            result = e.install(self.freeze, "training", destination)
        commands = [call.args[0] for call in self.run.call_args_list]
        self.assertIn("venv", commands[0])
        self.assertIn("--dry-run", commands[1])
        self.assertIn("--report", commands[1])
        self.assertNotIn("--dry-run", commands[2])
        self.assertIn("--verify-imports", commands[3])
        self.assertIn("-c", commands[1])
        self.assertTrue(all("--no-deps" not in command for command in commands))
        expected = e.original_helper().read_exact_pins(destination / "at-rag-effective-requirements.txt")
        self.assertEqual(expected, self.recorded)
        self.assertEqual(result["status"], "installed_and_verified")
        self.assertEqual(self.freeze.read_bytes(), before)

    def test_failed_resolution_stops_before_install_and_can_retry_only_matching_venv(self):
        destination = self.root / "training-venv"
        self.run.side_effect = [types.SimpleNamespace(returncode=0), subprocess.CalledProcessError(1, "resolver")]
        with mock.patch("builtins.print"), self.assertRaises(subprocess.CalledProcessError):
            e.install(self.freeze, "training", destination)
        self.assertEqual(self.run.call_count, 2)
        self.run.reset_mock(side_effect=True)
        self.run.return_value = types.SimpleNamespace(returncode=0)
        with mock.patch("builtins.print"):
            e.install(self.freeze, "training", destination)
        with self.assertRaisesRegex(ValueError, "different profile/freeze"):
            e.install(self.freeze, "retrieval", destination)

    def test_unknown_existing_environment_is_never_modified(self):
        destination = self.root / "unknown-venv"
        destination.mkdir()
        with self.assertRaisesRegex(ValueError, "unrecognized"):
            e.install(self.freeze, "training", destination)
        self.run.assert_not_called()

    def test_import_check_constructs_no_model_and_imports_no_retrieval_in_training(self):
        e.verify_imports("training")
        code = self.run.call_args.args[0][-1]
        self.assertIn("from bertopic import BERTopic", code)
        self.assertNotIn("BERTopic(", code)
        self.assertNotIn("import chromadb", code)
        self.assertNotIn("import langchain", code)


if __name__ == "__main__":
    unittest.main()
