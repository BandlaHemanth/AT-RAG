"""Fit the TAE-R candidate topic models (protocol v1.2, section 11).

Stages (each resumable: a stage whose manifest exists and whose files still match
their recorded SHA-256 is skipped):

  umap       fit one shared 5-D UMAP on the topic-fitting population; transform
             the validation retrieval corpus
  questions  embed router_train + validation questions (pinned MiniLM) and
             transform them with the shared UMAP
  cluster    fit each candidate: H1-H3 HDBSCAN on the shared 5-D UMAP output;
             K1-K2 spherical k-means on the normalised 384-D embeddings
  assign     label the validation retrieval corpus and the router_train +
             validation questions with each candidate (inference only)

The final retrieval corpus and final_eval questions are never read here.
Inputs are verified against the frozen corpus manifest and embedding hashes.

  python -B scripts/fit_tae_r_topic_models.py \
      --corpora-dir .../tae_r_corpora_v1_run01 --embeddings-dir .../tae-r-embeddings-v1 \
      --output-dir /kaggle/working/tae_r_topics_v1 --stages umap
"""
import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

PROTOCOL = "tae-r-protocol-v1.2"
FROZEN = {
    "corpus_manifest": "d6905fc1c4c64f71c394b15c1fbdcc3c8d6e7a15c7e3b55c710673181057d96e",
    "embeddings.npy": "19c5b4d6cb3bf0773f848a0a490f8787d8af8025ef616f5c6a074df0c0abd809",
    "document_ids.jsonl": "af08575835f1db2dd3ac445e1ff47650faab416e796d50558fee093fec9d80ee",
}
CONFIG = {
    "umap": {"n_neighbors": 15, "n_components": 5, "min_dist": 0.0, "metric": "cosine",
             "random_state": 42, "low_memory": True},
    "hdbscan": {"metric": "euclidean", "cluster_selection_method": "eom", "prediction_data": True},
    "kmeans": {"seed": 42, "max_iter": 100, "tol": 1e-4},
    "candidates": {
        "H1": {"family": "hdbscan", "min_cluster_size": 50, "min_samples": 50},
        "H2": {"family": "hdbscan", "min_cluster_size": 100, "min_samples": 10},
        "H3": {"family": "hdbscan", "min_cluster_size": 100, "min_samples": 2},
        "K1": {"family": "kmeans", "k": 100},
        "K2": {"family": "kmeans", "k": 300},
    },
    "encoder": {"model_id": "sentence-transformers/all-MiniLM-L6-v2",
                "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                "max_seq_length": 256, "batch_size": 128},
}
STAGES = ("umap", "questions", "cluster", "assign")
QUESTION_SPLITS = ("router_train", "validation")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def jsonl_ids(path, key):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line)[key] for line in stream if line.strip()]


def load_inputs(corpora_dir, embeddings_dir, frozen):
    require(sha256_file(corpora_dir / "manifest.json") == frozen["corpus_manifest"],
            "Corpus manifest differs from the frozen record")
    manifest = json.loads((corpora_dir / "manifest.json").read_text(encoding="utf-8"))
    needed = ["topic_fit_population", "retrieval_validation"] + ["questions_" + s for s in QUESTION_SPLITS]
    for name in needed:
        info = manifest["files"][name]
        require(sha256_file(corpora_dir / info["file"]) == info["sha256"], f"{info['file']}: hash differs")
    for name in ("embeddings.npy", "document_ids.jsonl"):
        require(sha256_file(embeddings_dir / name) == frozen[name], f"{name}: hash differs from frozen record")
    rows = jsonl_ids(embeddings_dir / "document_ids.jsonl", "document_id")
    row_of = {doc: i for i, doc in enumerate(rows)}
    embeddings = np.load(embeddings_dir / "embeddings.npy", mmap_mode="r")
    require(embeddings.shape == (len(rows), embeddings.shape[1]), "embedding rows differ from document ids")
    fit_ids = jsonl_ids(corpora_dir / "topic_fit_population.jsonl", "doc_id")
    val_ids = jsonl_ids(corpora_dir / "retrieval_validation.jsonl", "doc_id")
    questions = {}
    for split in QUESTION_SPLITS:
        with open(corpora_dir / f"questions_{split}.jsonl", encoding="utf-8") as stream:
            questions[split] = [(r["question_id"], r["question_text"]) for r in map(json.loads, stream)]
    return {"embeddings": embeddings, "row_of": row_of, "fit_ids": fit_ids, "val_ids": val_ids,
            "questions": questions}


def rows_for(inputs, ids):
    index = np.array([inputs["row_of"][doc] for doc in ids], dtype=np.int64)
    return np.ascontiguousarray(inputs["embeddings"][index], dtype=np.float32)


class Stage:
    """A directory of outputs plus a manifest of their hashes; resumable."""

    def __init__(self, root, name):
        self.dir = root / name
        self.manifest = self.dir / "stage_manifest.json"

    def complete(self):
        if not self.manifest.is_file():
            return False
        recorded = json.loads(self.manifest.read_text(encoding="utf-8"))["files"]
        return all((self.dir / f).is_file() and sha256_file(self.dir / f) == h for f, h in recorded.items())

    def save_array(self, name, array):
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / (name + ".tmp.npy")
        np.save(tmp, array)
        os.replace(tmp, self.dir / name)

    def save_object(self, name, obj):
        import joblib
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / (name + ".tmp")
        joblib.dump(obj, tmp)
        os.replace(tmp, self.dir / name)

    def save_json(self, name, data):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / name).write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    def finish(self, info):
        files = sorted(p.name for p in self.dir.iterdir() if p.is_file() and p.name != "stage_manifest.json")
        record = {"protocol": PROTOCOL, "files": {f: sha256_file(self.dir / f) for f in files},
                  "info": info, "peak_rss_mib": round(peak_rss_mib(), 1)}
        tmp = self.dir / "stage_manifest.json.tmp"
        tmp.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.manifest)
        return record

    def load(self, name):
        if name.endswith(".npy"):
            return np.load(self.dir / name)
        import joblib
        return joblib.load(self.dir / name)


def normalise(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def best_centroid(points, centroids, chunk=65536):
    labels = np.empty(len(points), dtype=np.int32)
    scores = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), chunk):
        sims = points[start:start + chunk] @ centroids.T
        labels[start:start + chunk] = sims.argmax(axis=1)
        scores[start:start + chunk] = sims.max(axis=1)
    return labels, scores


def spherical_kmeans(points, k, seed, max_iter, tol):
    from scipy.sparse import csr_matrix
    from sklearn.cluster import kmeans_plusplus
    centroids, _ = kmeans_plusplus(points, k, random_state=seed)
    centroids = normalise(centroids)
    labels = None
    for iteration in range(1, max_iter + 1):
        new, best = best_centroid(points, centroids)
        changed = len(points) if labels is None else int((new != labels).sum())
        labels = new
        onehot = csr_matrix((np.ones(len(points), dtype=np.float32), (labels, np.arange(len(points)))),
                            shape=(k, len(points)))
        sums = np.asarray(onehot @ points, dtype=np.float64)
        empty = np.flatnonzero(np.bincount(labels, minlength=k) == 0)
        if len(empty):  # re-seed empty clusters with the worst-fitting points, deterministically
            worst = np.argsort(best, kind="stable")[:len(empty)]
            sums[empty] = points[worst]
        centroids = normalise(sums)
        if changed / len(points) < tol:
            break
    labels, best = best_centroid(points, centroids)
    return centroids, labels, {"iterations": iteration, "mean_cosine_to_centroid": float(best.mean())}


def topic_centroids(points, labels):
    topics = np.array(sorted(t for t in set(labels.tolist()) if t >= 0), dtype=np.int32)
    centroids = np.zeros((len(topics), points.shape[1]), dtype=np.float64)
    for i, topic in enumerate(topics):
        centroids[i] = points[labels == topic].sum(axis=0)
    return topics, normalise(centroids)


def size_summary(labels):
    sizes = np.bincount(labels[labels >= 0]) if (labels >= 0).any() else np.array([], dtype=np.int64)
    sizes = sizes[sizes > 0]
    return {"topics": int(len(sizes)), "outliers": int((labels < 0).sum()),
            "outlier_rate": float((labels < 0).mean()), "documents": int(len(labels)),
            "largest_topic": int(sizes.max()) if len(sizes) else 0,
            "median_topic": float(np.median(sizes)) if len(sizes) else 0.0,
            "smallest_topic": int(sizes.min()) if len(sizes) else 0}


def stage_umap(inputs, root, config, log):
    stage = Stage(root, "umap")
    if stage.complete():
        return log("umap: complete, skipped")
    import umap
    started = time.time()
    fit_points = rows_for(inputs, inputs["fit_ids"])
    model = umap.UMAP(**config["umap"]).fit(fit_points)
    fit_seconds = time.time() - started
    stage.save_object("umap_model.joblib", model)
    stage.save_array("fit_5d.npy", model.embedding_.astype(np.float32))
    stage.save_array("validation_corpus_5d.npy",
                     model.transform(rows_for(inputs, inputs["val_ids"])).astype(np.float32))
    record = stage.finish({"fit_documents": len(inputs["fit_ids"]), "validation_documents": len(inputs["val_ids"]),
                           "fit_seconds": round(fit_seconds, 1), "total_seconds": round(time.time() - started, 1),
                           "umap": config["umap"]})
    log(f"umap: done in {record['info']['total_seconds']} s, peak RSS {record['peak_rss_mib']} MiB")


def stage_questions(inputs, root, config, log, encode):
    stage = Stage(root, "questions")
    if stage.complete():
        return log("questions: complete, skipped")
    umap_stage = Stage(root, "umap")
    require(umap_stage.complete(), "run the umap stage first")
    model = umap_stage.load("umap_model.joblib")
    started = time.time()
    ids, texts = [], []
    for split in QUESTION_SPLITS:
        for qid, text in inputs["questions"][split]:
            ids.append([split, qid])
            texts.append(text)
    vectors = normalise(np.asarray(encode(texts), dtype=np.float32))
    stage.save_json("question_ids.json", ids)
    stage.save_array("question_embeddings.npy", vectors)
    stage.save_array("question_5d.npy", model.transform(vectors).astype(np.float32))
    stage.finish({"questions": {s: len(inputs["questions"][s]) for s in QUESTION_SPLITS},
                  "encoder": config["encoder"], "seconds": round(time.time() - started, 1)})
    log(f"questions: {len(ids)} embedded and transformed")


def stage_cluster(inputs, root, config, names, log):
    umap_stage = Stage(root, "umap")
    fit_points = None
    for name in names:
        spec = config["candidates"][name]
        stage = Stage(root, name + "/fit")
        if stage.complete():
            log(f"cluster {name}: complete, skipped")
            continue
        if fit_points is None:
            fit_points = rows_for(inputs, inputs["fit_ids"])
        started = time.time()
        if spec["family"] == "hdbscan":
            import hdbscan
            require(umap_stage.complete(), "run the umap stage first")
            model = hdbscan.HDBSCAN(min_cluster_size=spec["min_cluster_size"], min_samples=spec["min_samples"],
                                    **config["hdbscan"]).fit(umap_stage.load("fit_5d.npy"))
            labels = model.labels_.astype(np.int32)
            stage.save_object("hdbscan_model.joblib", model)
            stage.save_array("fit_strengths.npy", model.probabilities_.astype(np.float32))
            topics, centroids = topic_centroids(fit_points, labels)
            extra = {}
        else:
            centroids, labels, extra = spherical_kmeans(fit_points, spec["k"], config["kmeans"]["seed"],
                                                        config["kmeans"]["max_iter"], config["kmeans"]["tol"])
            topics = np.arange(spec["k"], dtype=np.int32)
        stage.save_array("fit_labels.npy", labels)
        stage.save_array("centroids_384.npy", centroids)
        stage.save_array("centroid_topic_ids.npy", topics)
        record = stage.finish(dict(spec=spec, seconds=round(time.time() - started, 1), **size_summary(labels), **extra))
        info = record["info"]
        log(f"cluster {name}: {info['topics']} topics, outliers {info['outlier_rate']:.2%}, "
            f"{info['seconds']} s, peak RSS {record['peak_rss_mib']} MiB")


def stage_assign(inputs, root, config, names, log):
    umap_stage, q_stage = Stage(root, "umap"), Stage(root, "questions")
    require(umap_stage.complete() and q_stage.complete(), "run the umap and questions stages first")
    for name in names:
        spec = config["candidates"][name]
        fit_stage, stage = Stage(root, name + "/fit"), Stage(root, name + "/assign_validation")
        require(fit_stage.complete(), f"run the cluster stage for {name} first")
        if stage.complete():
            log(f"assign {name}: complete, skipped")
            continue
        started = time.time()
        if spec["family"] == "hdbscan":
            import hdbscan
            model = fit_stage.load("hdbscan_model.joblib")
            doc_labels, doc_strengths = hdbscan.approximate_predict(model, umap_stage.load("validation_corpus_5d.npy"))
            q_points = q_stage.load("question_5d.npy")
            q_labels, q_strengths = hdbscan.approximate_predict(model, q_points)
            stage.save_array("question_membership.npy",
                             hdbscan.membership_vector(model, q_points).astype(np.float32))
        else:
            centroids = fit_stage.load("centroids_384.npy")
            doc_labels, doc_strengths = best_centroid(rows_for(inputs, inputs["val_ids"]), centroids)
            q_vectors = q_stage.load("question_embeddings.npy")
            q_labels, q_strengths = best_centroid(q_vectors, centroids)
            stage.save_array("question_centroid_scores.npy", (q_vectors @ centroids.T).astype(np.float32))
        stage.save_array("validation_corpus_labels.npy", np.asarray(doc_labels, dtype=np.int32))
        stage.save_array("validation_corpus_strengths.npy", np.asarray(doc_strengths, dtype=np.float32))
        stage.save_array("question_labels.npy", np.asarray(q_labels, dtype=np.int32))
        stage.save_array("question_strengths.npy", np.asarray(q_strengths, dtype=np.float32))
        record = stage.finish({"corpus": size_summary(np.asarray(doc_labels)),
                               "questions": size_summary(np.asarray(q_labels)),
                               "seconds": round(time.time() - started, 1)})
        log(f"assign {name}: corpus outliers {record['info']['corpus']['outlier_rate']:.2%}, "
            f"question outliers {record['info']['questions']['outlier_rate']:.2%}")


def sentence_encoder(config, device):
    from sentence_transformers import SentenceTransformer
    enc = config["encoder"]
    model = SentenceTransformer(enc["model_id"], revision=enc["revision"], device=device)
    model.max_seq_length = enc["max_seq_length"]

    def encode(texts):
        return model.encode(texts, batch_size=enc["batch_size"], normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)
    return encode


def run(corpora_dir, embeddings_dir, output_dir, stages, names, *, frozen=FROZEN, config=CONFIG,
        encode=None, device="cuda", log=print):
    require(set(stages) <= set(STAGES), f"stages must be among {STAGES}")
    require(set(names) <= set(config["candidates"]), "unknown candidate")
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(corpora_dir, embeddings_dir, frozen)
    log(f"inputs verified: {len(inputs['fit_ids'])} fit, {len(inputs['val_ids'])} validation-corpus documents")
    if "umap" in stages:
        stage_umap(inputs, output_dir, config, log)
    if "questions" in stages:
        stage_questions(inputs, output_dir, config, log, encode or sentence_encoder(config, device))
    if "cluster" in stages:
        stage_cluster(inputs, output_dir, config, names, log)
    if "assign" in stages:
        stage_assign(inputs, output_dir, config, names, log)
    (output_dir / "run_config.json").write_text(json.dumps({"protocol": PROTOCOL, "config": config, "frozen": frozen},
                                                           indent=1, sort_keys=True) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpora-dir", type=Path, required=True)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    parser.add_argument("--candidates", nargs="+", default=list(CONFIG["candidates"]))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    try:
        run(args.corpora_dir, args.embeddings_dir, args.output_dir, args.stages, args.candidates,
            device=args.device, log=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))
    except (OSError, ValueError, KeyError) as error:
        print(f"Topic fitting failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
