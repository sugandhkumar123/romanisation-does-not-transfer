# Romanisation Does Not Transfer

Benchmark, corpus pipeline and results for:

> Sugandh Kumar. *Romanisation Does Not Transfer: Script Representation in Cross-Lingual Retrieval over Classical Sanskrit.* ICCSI 2026, Springer IFIP AICT. (to appear)

**Finding.** Across six multilingual encoders spanning three training paradigms, Devanagari
outperforms every romanised representation of the same Sanskrit text by 4.4x to 43.6x, and the
best-scoring romanised configuration is statistically indistinguishable from a semantics-free
character-hash baseline. Script representation accounts for roughly an order of magnitude more
variance than model selection.

---

## Why there is no corpus file here

The three source e-texts come from [sanskritdocuments.org](https://sanskritdocuments.org),
which asks that its prepared files not be redistributed. The verses themselves are public
domain; the prepared encodings are not ours to repost.

So this repository ships **the pipeline that builds the corpus** and **query-to-verse-reference
pairs**, not a dump of the text. Run `code/corpus_build.py` and you get the identical
1,288-verse corpus the paper used.

---

## Reproducing the paper

```bash
pip install -r code/requirements.txt

python code/corpus_build.py --out corpus.jsonl --report-gaps
python code/fix_corpus.py                      # two documented scribal corrections
python code/run_retrieval.py --stub            # offline smoke test, ~10 s
python code/run_retrieval.py --models e5 labse bge mpnet minilm muril \
       --queries data/benchmark_200.csv
```

Expected corpus: **195 / 701 / 392 = 1,288 verses**, zero transliteration round-trip failures.
The Gita is 701 rather than 700 and the Hatha Yoga Pradipika 392 rather than 389 because of
edition differences documented in Section 3.1 of the paper.

Encoding all six models over four representations takes roughly 2 hours on CPU and a few
minutes on GPU. Embeddings cache to `emb_cache/`, so re-runs are free.

---

## Contents

### `code/`

| File | Purpose |
|---|---|
| `corpus_build.py` | Fetches and normalises the three source texts into 1,288 verses in four parallel script representations |
| `fix_corpus.py` | Applies two documented scribal corrections (HYP 2.1, 4.45) with justifications |
| `check_gaps.py` | Diagnoses whether missing verse numbers are source recension differences or parser errors |
| `sample_benchmark.py` | Stratified sample by text, chapter and verse-length tercile; fixed seed |
| `merge_drafts.py` | Merges the LLM drafts into the verification worksheet |
| `drafts_gita.py`, `drafts_ys_hyp.py` | The **unverified** LLM drafts, released for provenance |
| `run_retrieval.py` | The 6 x 4 x 2 evaluation grid, metrics and bootstrap CIs |
| `robustness.py` | Paired permutation tests, per-slice consistency and query-subsampling analysis (paper Section 6.6): `python code/robustness.py results/results_per_query.csv` |
| `mkfig.py`, `mkfig_pipeline.py` | Regenerate Figures 1 and 2 |

### `data/`

| File | Purpose |
|---|---|
| `benchmark_200.csv` | **The benchmark.** 200 English queries, each with one gold verse reference |
| `benchmark_200_unverified_drafts.csv` | The LLM drafts *before* author verification |

Verse text is deliberately absent; `verse_id` (e.g. `bhagavad_gita:2.47`) resolves against the
corpus that `corpus_build.py` produces.

### `results/`

| File | Purpose |
|---|---|
| `results_all_models.csv` | All 48 configurations: nDCG@10 with 95% CI, MRR, Recall@5, Recall@10 |
| `results_per_query.csv` | Per-query rank and metrics, for paired significance tests and slicing |
| `results_character_hash_baseline.csv` | The semantics-free baseline, measured per representation |

---

## How the benchmark was built, precisely

Glosses and queries were **drafted by the large language model Claude (Anthropic)** from the Sanskrit,
then **verified by the first author against the Sanskrit source text**. Seven of 200 required correction
(3.5%), all in the gloss: four substantive errors of Sanskrit reading (YS 2.36, 3.10, 3.14; HYP 4.76)
and three glosses that omitted a detail their query asked about (BG 17.23, 17.27; HYP 2.10). No query
needed rewording. The same pass found two scribal errors in the source e-texts (HYP 2.1, 4.45).

This is model-assisted annotation with human verification, not independent expert annotation.
The unverified drafts are included above so that the provenance of every item can be audited.

Queries were written so as not to reuse distinctive vocabulary from the target verse: a query
containing a rare term that appears verbatim in the gold verse would test string matching rather
than semantic retrieval. An automated check confirms no query shares a token longer than six
characters with its gold verse.

---

## Two transliteration pitfalls worth knowing about

Both would corrupt a Sanskrit retrieval evaluation silently rather than failing loudly. The first is
a defect in the library; the second is standard ITRANS behaviour that bites if steps run in the wrong order.

1. **`indic-transliteration` does not implement the ITRANS explicit virama `.h`.** It renders the
   period as a danda and strands the `h`, so `vid.hdhyanAdI` becomes `vid|hdhyanādī` — a
   sentence-boundary mark inside a word. At least 33 verses were affected.
2. **Verse references must be parsed before transliteration.** ITRANS maps `.` to the danda, so
   transliterating first turns reference `1.1` into `1|1` and destroys the numbering.

`corpus_build.py` handles both, and gates every verse through a
Devanagari to Harvard-Kyoto to Devanagari round trip.

---

## Licence

Code: MIT. Benchmark (`data/`) and results: CC BY 4.0.

Source verses are public domain. The prepared e-texts are from sanskritdocuments.org and are not
redistributed here.

## Citation

```bibtex
@inproceedings{kumar2026romanisation,
  title     = {Romanisation Does Not Transfer: Script Representation in
               Cross-Lingual Retrieval over Classical Sanskrit},
  author    = {Kumar, Sugandh},
  booktitle = {International Conference on Computational Intelligence and
               Sustainable Innovation (ICCSI 2026)},
  series    = {IFIP Advances in Information and Communication Technology},
  publisher = {Springer},
  year      = {2026}
}
```
