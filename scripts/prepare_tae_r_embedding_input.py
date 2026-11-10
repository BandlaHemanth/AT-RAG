"""Make the single embedding input for the frozen TAE-R corpora.

Reads the corpora written by scripts/build_tae_r_corpora.py, verifies them against
the recorded corpus manifest, and writes the union of unique paragraphs as
documents.jsonl (document_id, paragraph_text; sorted by document_id) plus the
summary.json that scripts/embed_full_2wiki_corpus.py expects. Computes no
embeddings. Standard library only.

  python -B scripts/prepare_tae_r_embedding_input.py \
      --corpora-dir /kaggle/working/tae_r_corpora_v1_run01 \
      --output-dir /kaggle/working/tae_r_embedding_input_v1
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

FROZEN_CORPUS_MANIFEST_SHA256 = "d6905fc1c4c64f71c394b15c1fbdcc3c8d6e7a15c7e3b55c710673181057d96e"
CORPUS_FILES = ("topic_fit_population", "retrieval_final", "retrieval_validation")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def prepare(corpora_dir, output_dir, frozen_manifest_sha256=FROZEN_CORPUS_MANIFEST_SHA256):
    require(not os.path.lexists(output_dir), "Output directory exists; use a fresh one")
    require(not output_dir.resolve().is_relative_to(corpora_dir.resolve()),
            "Output directory must be outside the corpora directory")
    raw_manifest = (corpora_dir / "manifest.json").read_bytes()
    require(sha256_hex(raw_manifest) == frozen_manifest_sha256, "Corpus manifest differs from the frozen record")
    manifest = json.loads(raw_manifest.decode("utf-8"))
    texts, membership = {}, {}
    for name in CORPUS_FILES:
        info = manifest["files"][name]
        raw = (corpora_dir / info["file"]).read_bytes()
        require(sha256_hex(raw) == info["sha256"], f"{info['file']}: hash differs from manifest")
        lines = [line for line in raw.decode("utf-8").splitlines() if line.strip()]
        require(len(lines) == info["count"], f"{info['file']}: count differs from manifest")
        for line in lines:
            record = json.loads(line)
            text, doc = record["text"], record["doc_id"]
            require(doc == hashlib.sha256(text.encode("utf-8")).hexdigest(), "doc_id is not the text hash")
            require(texts.setdefault(doc, text) == text, "Same doc_id with different text")
            membership.setdefault(doc, set()).add(name)
    fit = {d for d, m in membership.items() if "topic_fit_population" in m}
    require(not any(len(membership[d]) > 1 for d in fit), "Topic-fitting paragraph also in a retrieval corpus")
    data = b"".join((json.dumps({"document_id": d, "paragraph_text": texts[d]}, ensure_ascii=False,
                                sort_keys=True) + "\n").encode("utf-8") for d in sorted(texts))
    summary = {"documents_sha256": sha256_hex(data), "unique_paragraphs": len(texts),
               "source": "tae-r-corpora-v1", "corpus_manifest_sha256": frozen_manifest_sha256,
               "per_corpus_counts": {n: manifest["files"][n]["count"] for n in CORPUS_FILES},
               "paragraphs_in_both_retrieval_corpora": sum(
                   1 for m in membership.values() if {"retrieval_final", "retrieval_validation"} <= m)}
    output_dir.mkdir(parents=True)
    with (output_dir / "documents.jsonl").open("xb") as stream:
        stream.write(data)
    with (output_dir / "summary.json").open("xb") as stream:
        stream.write((json.dumps(summary, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpora-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = prepare(args.corpora_dir, args.output_dir)
    except (OSError, ValueError, KeyError) as error:
        print(f"Embedding input failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
