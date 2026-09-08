#!/usr/bin/env python3
"""
ICCSI-2026 stage 2b -- retrieval evaluation harness.

Runs the full grid:

    embedding model  x  script representation  x  retrieval mode

over the 1,288-verse corpus, scored against the 200 hand-verified queries.

    representations : devanagari | iast | hk | ascii_fold
    modes           : dense | hybrid (BM25 + dense, fused by RRF)
    metrics         : nDCG@10, MRR, Recall@5, Recall@10
                      + 95% bootstrap CI on nDCG@10

Each query has exactly one relevant document, so nDCG@10 reduces to
1/log2(rank+1) when the gold verse is retrieved in the top 10, and 0
otherwise. Per-query scores are written out too, so you can run paired
significance tests between any two configurations afterwards.

Usage
-----
    pip install sentence-transformers numpy

    # 1. smoke-test the whole pipeline in ~10 seconds, no downloads:
    python run_retrieval.py --stub

    # 2. one model, all four representations:
    python run_retrieval.py --models e5

    # 3. the full grid (slow -- see note below):
    python run_retrieval.py

Runtime
-------
Encoding 1,288 documents x 4 representations x 6 models is 24 passes.
On CPU that is roughly 1.5-2.5 hours; on a single GPU, a few minutes.
Embeddings are cached under ./emb_cache/, so re-runs and added metrics
cost nothing. Run --stub first: it validates the metrics, the fusion and
the file formats without downloading a single model.

If you are short on time, run --models e5 labse bge first. Those three
carry the argument; the rest strengthen it.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CACHE = Path("emb_cache")
CACHE.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Models. `qp`/`dp` are the query and document prefixes the model was trained
# with -- E5 in particular degrades badly without them, which is a real and
# frequently-missed confound in comparison papers.
# ---------------------------------------------------------------------------
MODELS = {
    "e5":     dict(hf="intfloat/multilingual-e5-large",
                   qp="query: ", dp="passage: "),
    "labse":  dict(hf="sentence-transformers/LaBSE", qp="", dp=""),
    "bge":    dict(hf="BAAI/bge-m3", qp="", dp=""),
    "mpnet":  dict(hf="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                   qp="", dp=""),
    "minilm": dict(hf="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                   qp="", dp=""),
    "muril":  dict(hf="google/muril-base-cased", qp="", dp="",
                   note="not a sentence-transformer: mean-pooled encoder "
                        "baseline, expected to underperform -- report as such"),
}

REPRESENTATIONS = ["devanagari", "iast", "hk", "ascii_fold"]
MODES = ["dense", "hybrid"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_corpus(path):
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not recs:
        sys.exit(f"{path} is empty")
    return recs


def load_queries(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    kept, dropped = [], 0
    for r in rows:
        if r.get("verified", "").strip().lower() != "y":
            dropped += 1
            continue
        if not r.get("query_1", "").strip():
            dropped += 1
            continue
        kept.append({"item_id": r["item_id"],
                     "query": r["query_1"].strip(),
                     "gold": r["verse_id"].strip(),
                     "text_id": r["verse_id"].split(":")[0],
                     "length_band": r.get("length_band", "")})
    if not kept:
        sys.exit("no verified queries found -- is the `verified` column filled?")
    print(f"queries: {len(kept)} verified, {dropped} skipped")
    return kept


# ---------------------------------------------------------------------------
# Encoding (cached)
# ---------------------------------------------------------------------------
_loaded = {}


def get_model(key):
    if key in _loaded:
        return _loaded[key]
    from sentence_transformers import SentenceTransformer
    hf = MODELS[key]["hf"]
    print(f"  loading {hf} ...")
    _loaded[key] = SentenceTransformer(hf)
    return _loaded[key]


def stub_encode(texts):
    """Deterministic pseudo-embeddings, for validating the pipeline offline.

    Character-trigram hashing: shares structure with real embeddings enough
    that ranking behaves sensibly, while needing no model download.
    """
    dim = 256
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        t = unicodedata.normalize("NFC", t.lower())
        for j in range(max(1, len(t) - 2)):
            h = int(hashlib.md5(t[j:j + 3].encode("utf-8")).hexdigest()[:8], 16)
            out[i, h % dim] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(n, 1e-9, None)


def encode(texts, model_key, prefix, tag, stub=False):
    if stub:
        return stub_encode([prefix + t for t in texts])
    sig = hashlib.md5(("|".join(texts) + model_key + prefix)
                      .encode("utf-8")).hexdigest()[:16]
    f = CACHE / f"{model_key}_{tag}_{sig}.npy"
    if f.exists():
        return np.load(f)
    m = get_model(model_key)
    v = m.encode([prefix + t for t in texts], batch_size=32,
                 show_progress_bar=True, normalize_embeddings=True)
    v = np.asarray(v, dtype=np.float32)
    np.save(f, v)
    return v


# ---------------------------------------------------------------------------
# BM25 (inline -- avoids a dependency, and lets us control tokenisation)
# ---------------------------------------------------------------------------
TOKEN = re.compile(r"[^\s\u0964\u0965|.,;:!?'\"()\[\]-]+")


def tokenize(s):
    return [t for t in TOKEN.findall(unicodedata.normalize("NFC", s.lower())) if t]


class BM25:
    """Standard Okapi BM25.

    Note for the write-up: Sanskrit compounds and sandhi mean whitespace
    tokenisation under-segments badly, so BM25 is expected to be weak on
    devanagari/iast and somewhat better on ascii_fold. That contrast is a
    result worth reporting, not a bug to hide.
    """

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.n = len(self.docs)
        self.len = np.array([len(d) for d in self.docs], dtype=np.float32)
        self.avg = float(self.len.mean()) if self.n else 0.0
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5))
                    for t, c in df.items()}
        self.postings = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t in c:
                self.postings[t].append(i)

    def scores(self, query):
        s = np.zeros(self.n, dtype=np.float32)
        for t in tokenize(query):
            if t not in self.idf:
                continue
            idf = self.idf[t]
            for i in self.postings[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                s[i] += idf * f * (self.k1 + 1) / denom
        return s


def rrf(rank_lists, k=60):
    """Reciprocal rank fusion. Hyperparameter-light, so the hybrid result
    is not an artefact of a tuned mixing weight."""
    score = defaultdict(float)
    for ranks in rank_lists:
        for pos, idx in enumerate(ranks):
            score[idx] += 1.0 / (k + pos + 1)
    return sorted(score, key=score.get, reverse=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def per_query_metrics(rank):
    """rank is 1-based position of the single gold doc, or None."""
    if rank is None:
        return dict(ndcg10=0.0, mrr=0.0, r5=0.0, r10=0.0)
    return dict(
        ndcg10=(1.0 / math.log2(rank + 1)) if rank <= 10 else 0.0,
        mrr=1.0 / rank,
        r5=1.0 if rank <= 5 else 0.0,
        r10=1.0 if rank <= 10 else 0.0,
    )


def bootstrap_ci(values, n=1000, seed=0):
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    a = np.asarray(values, dtype=np.float64)
    means = a[rng.integers(0, len(a), size=(n, len(a)))].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus_corrected.jsonl")
    ap.add_argument("--queries", default="benchmark_worksheet_verified.csv")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--reps", nargs="*", default=REPRESENTATIONS)
    ap.add_argument("--modes", nargs="*", default=MODES)
    ap.add_argument("--stub", action="store_true",
                    help="offline smoke test: no model downloads")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--per-query-out", default="results_per_query.csv")
    args = ap.parse_args()

    if args.stub:
        args.models = ["stub"]
        MODELS["stub"] = dict(hf="(offline stub)", qp="", dp="")

    corpus = load_corpus(args.corpus)
    queries = load_queries(args.queries)

    doc_ids = [r["id"] for r in corpus]
    pos = {d: i for i, d in enumerate(doc_ids)}
    missing = [q["item_id"] for q in queries if q["gold"] not in pos]
    if missing:
        sys.exit(f"gold verse ids not present in corpus: {missing[:5]} "
                 f"({len(missing)} total) -- corpus/benchmark mismatch")

    qtexts = [q["query"] for q in queries]
    gold = np.array([pos[q["gold"]] for q in queries])

    rows, per_query = [], []

    for mk in args.models:
        cfg = MODELS[mk]
        if cfg.get("note"):
            print(f"\n[{mk}] {cfg['note']}")
        for rep in args.reps:
            docs = [r[rep] for r in corpus]
            print(f"\n=== {mk} / {rep} ===")
            D = encode(docs, mk, cfg["dp"], f"doc_{rep}", args.stub)
            Q = encode(qtexts, mk, cfg["qp"], "qry", args.stub)
            sim = Q @ D.T                                   # both normalised
            dense_order = np.argsort(-sim, axis=1)

            bm = BM25(docs) if "hybrid" in args.modes else None

            for mode in args.modes:
                scores = defaultdict(list)
                for qi, q in enumerate(queries):
                    if mode == "dense":
                        order = dense_order[qi][:100]
                    else:
                        bs = bm.scores(q["query"])
                        border = np.argsort(-bs)[:100]
                        order = rrf([list(dense_order[qi][:100]),
                                     list(border)])[:100]
                    order = list(order)
                    rank = order.index(gold[qi]) + 1 if gold[qi] in order else None
                    m = per_query_metrics(rank)
                    for k, v in m.items():
                        scores[k].append(v)
                    per_query.append(dict(model=mk, representation=rep,
                                          mode=mode, item_id=q["item_id"],
                                          text_id=q["text_id"],
                                          length_band=q["length_band"],
                                          rank=rank if rank else "",
                                          **m))
                lo, hi = bootstrap_ci(scores["ndcg10"])
                row = dict(model=mk, representation=rep, mode=mode,
                           n=len(queries),
                           ndcg10=round(float(np.mean(scores["ndcg10"])), 4),
                           ndcg10_lo=round(lo, 4), ndcg10_hi=round(hi, 4),
                           mrr=round(float(np.mean(scores["mrr"])), 4),
                           recall5=round(float(np.mean(scores["r5"])), 4),
                           recall10=round(float(np.mean(scores["r10"])), 4))
                rows.append(row)
                print(f"  {mode:7} nDCG@10 {row['ndcg10']:.4f} "
                      f"[{lo:.4f},{hi:.4f}]  MRR {row['mrr']:.4f}  "
                      f"R@5 {row['recall5']:.4f}  R@10 {row['recall10']:.4f}")

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(args.per_query_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_query[0]))
        w.writeheader()
        w.writerows(per_query)

    print(f"\n{'=' * 70}")
    print(f"{len(rows)} configurations -> {args.out}")
    print(f"per-query scores          -> {args.per_query_out}")
    best = max(rows, key=lambda r: r["ndcg10"])
    print(f"\nbest: {best['model']} / {best['representation']} / {best['mode']} "
          f"  nDCG@10 = {best['ndcg10']}")
    print("\nUse results_per_query.csv for paired significance tests between "
          "configurations, and to break results down by text and length band.")


if __name__ == "__main__":
    main()
