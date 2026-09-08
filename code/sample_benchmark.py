#!/usr/bin/env python3
"""
ICCSI-2026 stage 2a -- draw the benchmark sample.

Takes corpus.jsonl (1,288 verses) and draws a stratified sample you will
hand-verify. Writes two files:

    benchmark_worksheet.csv  <- you fill this in
    benchmark_sample.jsonl   <- machine-readable copy of the same rows

Why a sample and not all 1,288: published translations of these texts are
misaligned against the Sanskrit in places (the sacred-texts HYP edition
drifts by one verse from ~1.63 onward, and its own production note admits
chapter III is unreliable). Aligning 1,288 pairs automatically would inherit
those errors silently, and wrong gold labels do not merely add noise to
nDCG -- they invert rankings. 200 pairs you have personally checked is a
stronger benchmark than 1,288 you have not.

Stratification -- proportional across:
    * text          (yoga_sutras / bhagavad_gita / hatha_yoga_pradipika)
    * chapter       (so no chapter is unrepresented)
    * verse length  (short / medium / long terciles, computed per text)

Length matters because retrieval quality varies sharply with document
length, and an unstratified draw would let that confound the script
comparison that is the point of the paper.

Usage
-----
    python sample_benchmark.py                     # 200 items, seed 20261206
    python sample_benchmark.py --n 200 --seed 42
    python sample_benchmark.py --texts yoga_sutras bhagavad_gita

Filling in the worksheet
------------------------
For each row, consult a published public-domain translation:
    Yoga Sutras  -- Woods 1914 (archive.org: yogasystemofpata00wooduoft)
                    or Vivekananda, Raja Yoga, 1896
    Bhagavad Gita -- Telang 1882, Sacred Books of the East vol. 8
                    (sacred-texts.com/hin/sbe08/, or Wikisource)
    HYP          -- Pancham Sinh 1914 (VERIFY THE VERSE NUMBER -- this
                    edition drifts against the Sanskrit)

Then fill three columns:

    gloss_en   one-sentence English meaning of THIS verse
    query_1    a natural question a practitioner would ask, whose answer
               is this verse. Do NOT reuse the verse's distinctive words --
               a query that copies rare terms tests string matching, not
               semantics, and will flatter every model equally.
    verified   y = the translation genuinely corresponds to this Sanskrit
               n = it does not (numbering drift) -> row is dropped

query_2 is optional: a second, differently-phrased query for the same
verse. Worth doing for maybe a third of rows -- it lets you report
query-paraphrase robustness, which reviewers ask about.

Leave `verified` blank and the row is treated as unchecked, not as passed.
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def load(path):
    if not Path(path).exists():
        sys.exit(f"{path} not found -- run corpus_build.py first")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def length_band(record, cuts):
    lo, hi = cuts[record["text_id"]]
    n = record["n_chars"]
    return "short" if n <= lo else ("long" if n > hi else "medium")


def tercile_cuts(records):
    """Per-text length terciles -- the texts differ hugely in verse length."""
    cuts = {}
    by_text = defaultdict(list)
    for r in records:
        by_text[r["text_id"]].append(r["n_chars"])
    for tid, lens in by_text.items():
        lens.sort()
        cuts[tid] = (lens[len(lens) // 3], lens[2 * len(lens) // 3])
    return cuts


def allocate(strata, n_total):
    """Largest-remainder apportionment, with >=1 per non-empty stratum."""
    sizes = {k: len(v) for k, v in strata.items()}
    pool = sum(sizes.values())
    if n_total > pool:
        n_total = pool
    raw = {k: sizes[k] * n_total / pool for k in sizes}
    alloc = {k: min(sizes[k], max(1, int(raw[k]))) for k in sizes}

    # correct the total, largest fractional remainder first
    def fix(delta):
        order = sorted(raw, key=lambda k: raw[k] - int(raw[k]),
                       reverse=delta > 0)
        i = 0
        while delta != 0 and i < len(order) * 40:
            k = order[i % len(order)]
            if delta > 0 and alloc[k] < sizes[k]:
                alloc[k] += 1
                delta -= 1
            elif delta < 0 and alloc[k] > 1:
                alloc[k] -= 1
                delta += 1
            i += 1
        return delta

    fix(n_total - sum(alloc.values()))
    return alloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261206)
    ap.add_argument("--texts", nargs="*", default=None,
                    help="restrict to these text_ids")
    ap.add_argument("--out-csv", default="benchmark_worksheet.csv")
    ap.add_argument("--out-jsonl", default="benchmark_sample.jsonl")
    args = ap.parse_args()

    records = load(args.corpus)
    if args.texts:
        records = [r for r in records if r["text_id"] in args.texts]
        if not records:
            sys.exit(f"no records for texts {args.texts}")

    cuts = tercile_cuts(records)
    strata = defaultdict(list)
    for r in records:
        ch = r["ref"].partition(".")[0]
        strata[(r["text_id"], ch, length_band(r, cuts))].append(r)

    alloc = allocate(strata, args.n)

    rng = random.Random(args.seed)
    picked = []
    for key in sorted(strata):
        bucket = sorted(strata[key], key=lambda r: r["id"])
        picked.extend(rng.sample(bucket, alloc[key]))
    picked.sort(key=lambda r: (r["text_id"],
                               int(r["ref"].split(".")[0]),
                               int(r["ref"].split(".")[1])))

    cols = ["item_id", "verse_id", "title", "ref", "n_chars", "length_band",
            "iast", "devanagari", "gloss_en", "query_1", "query_2", "verified",
            "notes"]
    with open(args.out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(picked, 1):
            w.writerow({
                "item_id": f"q{i:04d}",
                "verse_id": r["id"],
                "title": r["title"],
                "ref": r["ref"],
                "n_chars": r["n_chars"],
                "length_band": length_band(r, cuts),
                "iast": r["iast"],
                "devanagari": r["devanagari"],
                "gloss_en": "", "query_1": "", "query_2": "",
                "verified": "", "notes": "",
            })

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for i, r in enumerate(picked, 1):
            f.write(json.dumps({"item_id": f"q{i:04d}",
                                "verse_id": r["id"],
                                "length_band": length_band(r, cuts),
                                **r}, ensure_ascii=False) + "\n")

    per_text = defaultdict(int)
    per_band = defaultdict(int)
    for r in picked:
        per_text[r["text_id"]] += 1
        per_band[length_band(r, cuts)] += 1

    print(f"sampled {len(picked)} verses  (seed {args.seed})")
    print(f"  by text : {dict(sorted(per_text.items()))}")
    print(f"  by length: {dict(sorted(per_band.items()))}")
    print(f"  strata  : {len(strata)}")
    print(f"\n-> {args.out_csv}   (fill in gloss_en, query_1, verified)")
    print(f"-> {args.out_jsonl}")
    print("\nOpen the CSV in Excel. It is UTF-8 with BOM, so Devanagari and")
    print("IAST diacritics will display correctly without an import step.")


if __name__ == "__main__":
    main()
