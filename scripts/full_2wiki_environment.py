"""Exact isolated profiles for full 2Wiki; no model loading or training.

install must be launched with the existing Python 3.12.3 interpreter. It creates
its own venv, resolves before installing, checks every retained training pin,
runs pip check and import-only checks, then records the complete installed set.
"""

import argparse
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = ROOT / "compat/full_2wiki"
PYTHON = "3.12.3"


def original_helper():
    spec = util.spec_from_file_location("_full_original_pins", ROOT / "scripts/ingest_original_2wiki_minilm.py")
    result = util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def integration(name):
    return name.startswith(("langchain", "langgraph")) or name in {"chromadb", "chroma-hnswlib"}


def plan(freeze, profile):
    """Keep the full record immutable; report every exclusion/override explicitly."""
    if profile not in {"training", "retrieval"}:
        raise ValueError("Unknown environment profile")
    helper = original_helper()
    recorded = helper.read_exact_pins(freeze)
    training = helper.read_exact_pins(PROFILE_ROOT / "training-py312.txt")
    overlay = helper.read_exact_pins(PROFILE_ROOT / f"{profile}-py312.txt")
    if len(recorded) != 62:
        raise ValueError(f"Expected the original complete 62-package record; got {len(recorded)}")
    for name, expected in training.items():
        if name not in recorded or not helper.version_matches(recorded[name], expected):
            raise ValueError(f"Recorded training pin mismatch: {name}; expected {expected}")
    excluded = {name: version for name, version in recorded.items() if integration(name)}
    effective = {name: version for name, version in recorded.items() if name not in excluded}
    changes = {}
    for name, version in overlay.items():
        if name in effective and not helper.version_matches(effective[name], version):
            if profile != "retrieval" or name != "packaging":
                raise ValueError(f"Unapproved profile/freeze conflict: {name}=={effective[name]} versus {version}")
            changes[name] = {"recorded": effective[name], "target": version,
                             "reason": "langchain-core 0.3.63 requires packaging>=23.2,<25"}
            effective[name] = version
        elif name not in effective:
            effective[name] = version
        # A matching local Torch/CUDA build recorded in the freeze takes precedence.
    return {"profile": profile, "python": PYTHON, "recorded_training_packages": recorded,
            "training_requirements_sha256": sha(freeze),
            "profile_sha256": sha(PROFILE_ROOT / f"{profile}-py312.txt"),
            "training_profile_sha256": sha(PROFILE_ROOT / "training-py312.txt"),
            "excluded_integration_pins": excluded, "deliberate_pin_changes": changes,
            "expected_packages": effective}


def package_versions():
    import re
    result = {}
    for distribution in metadata.distributions():
        name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
        if name in result:
            raise ValueError(f"Duplicate installed distribution: {name}")
        result[name] = distribution.version
    return dict(sorted(result.items()))


def inspect_environment(freeze, profile):
    result = plan(freeze, profile)
    errors = []
    if platform.python_version() != PYTHON:
        errors.append(f"Python mismatch: {platform.python_version()}; expected {PYTHON}")
    packages = package_versions()
    helper = original_helper()
    for name, expected in result["expected_packages"].items():
        if name not in packages or not helper.version_matches(packages[name], expected):
            errors.append(f"Dependency mismatch: {name}=={packages.get(name, 'missing')}; expected {expected}")
    if profile == "training":
        forbidden = sorted(name for name in packages if integration(name))
        if forbidden:
            errors.append(f"Training environment contains retrieval packages: {forbidden}; use an isolated venv")
    if errors:
        raise ValueError("Environment mismatch:\n- " + "\n- ".join(errors))
    checked = subprocess.run([sys.executable, "-B", "-m", "pip", "check"],
                             capture_output=True, text=True, check=False)
    if checked.returncode:
        raise ValueError("Installed dependency conflict:\n" + (checked.stdout + checked.stderr).strip())
    result.update(packages=packages, pip_check="passed", environment_checker_sha256=sha(__file__))
    return result


def model_contract(environment):
    """All retained recorded versions except the declared packaging replacement.

    This includes pickle dependencies (joblib/pandas), tokenizer/backend versions,
    and local Torch builds, not just the eleven top-level scientific packages.
    Actual versions must match across environments even for local build tags.
    """
    recorded = environment["recorded_training_packages"]
    names = sorted(name for name in recorded if name != "packaging" and not integration(name))
    return {"python": environment["python"],
            "training_requirements_sha256": environment["training_requirements_sha256"],
            "packages": {name: environment["packages"][name] for name in names}}


def verify_imports(profile):
    # Import classes only: no constructor, downloaded weights, client, or model.
    code = ("import numpy, scipy, sklearn, torch, transformers, sentence_transformers; "
            "import umap, hdbscan, numba, llvmlite; from bertopic import BERTopic; ")
    if profile == "training":
        code += "import sys; assert not any(n == 'chromadb' or n.startswith('langchain') for n in sys.modules)"
    else:
        code += ("from langchain_community.vectorstores import Chroma; "
                 "from langchain_community.chat_models import ChatOpenAI; "
                 "from langgraph.graph import StateGraph, START, END")
    subprocess.run([sys.executable, "-B", "-c", code], check=True)


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def install(freeze, profile, destination):
    """Create/retry only a venv owned by this installer; never mutate the caller."""
    if platform.python_version() != PYTHON:
        raise ValueError(f"Launch install with your existing Python {PYTHON} interpreter; got {platform.python_version()}")
    from contextlib import ExitStack
    import fcntl
    destination = Path(destination).absolute()
    configuration = plan(freeze, profile)
    configuration["installer_sha256"] = sha(__file__)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        handle = stack.enter_context(destination.with_name(destination.name + ".install.lock").open("a+b"))
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another installer owns this environment") from error
        marker = destination / ".at-rag-install.json"
        if destination.exists():
            if not marker.is_file() or json.loads(marker.read_text()) != configuration:
                raise ValueError("Existing venv is unrecognized or has a different profile/freeze; choose a fresh destination")
        else:
            # Mark before venv creation so interruption during ensurepip can be retried.
            destination.mkdir()
            write_json(marker, configuration)
        subprocess.run([sys.executable, "-B", "-m", "venv", str(destination)], check=True)
        interpreter = destination / "bin/python"
        requirements = destination / "at-rag-effective-requirements.txt"
        requirements.write_text("".join(f"{name}=={version}\n" for name, version in
                                         sorted(configuration["expected_packages"].items())))
        # Print the explicit changes before pip is invoked; do not alter the input freeze.
        print(json.dumps({"profile": profile, "excluded_integration_pins": configuration["excluded_integration_pins"],
                          "deliberate_pin_changes": configuration["deliberate_pin_changes"]}, indent=2), flush=True)
        command = [str(interpreter), "-B", "-m", "pip", "install", "-r", str(requirements), "-c", str(requirements)]
        subprocess.run(command + ["--dry-run", "--report", str(destination / "pip-resolution.json")], check=True)
        subprocess.run(command, check=True)
        subprocess.run([str(interpreter), "-B", str(Path(__file__).resolve()), "check", "--profile", profile,
                        "--training-requirements", str(Path(freeze).resolve()), "--verify-imports",
                        "--report", str(destination / "verified-environment.json")], check=True)
    return {"status": "installed_and_verified", "python": str(interpreter),
            "report": str(destination / "verified-environment.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["install", "check", "plan"])
    parser.add_argument("--profile", choices=["training", "retrieval"], required=True)
    parser.add_argument("--training-requirements", type=Path, required=True)
    parser.add_argument("--venv", type=Path)
    parser.add_argument("--verify-imports", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "install":
            if not args.venv:
                raise ValueError("install requires --venv")
            result = install(args.training_requirements, args.profile, args.venv)
        elif args.action == "plan":
            result = plan(args.training_requirements, args.profile)
        else:
            result = inspect_environment(args.training_requirements, args.profile)
            if args.verify_imports:
                verify_imports(args.profile)
                result["import_check"] = "passed; no model constructed"
        if args.report:
            write_json(args.report, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError, ImportError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Full 2Wiki environment stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
