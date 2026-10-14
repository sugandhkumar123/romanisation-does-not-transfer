#!/usr/bin/env python3
"""Merge drafted gloss/query text into benchmark_worksheet.csv.

Adds a draft_confidence column. Leaves `verified` blank on purpose: that
column records the first author's check against the Sanskrit source text.
"""
import csv, sys
from drafts_gita import GITA
from drafts_ys_hyp import YOGA_SUTRAS, HYP

drafts = {i: (g, q, c) for i, g, q, c in GITA + YOGA_SUTRAS + HYP}

src = sys.argv[1] if len(sys.argv) > 1 else "benchmark_worksheet.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "benchmark_worksheet_drafted.csv"

rows = list(csv.DictReader(open(src, encoding="utf-8-sig")))
cols = list(rows[0].keys())
cols.insert(cols.index("verified"), "draft_confidence")

missing, filled = [], 0
for r in rows:
    d = drafts.get(r["item_id"])
    if not d:
        missing.append(r["item_id"]); r["draft_confidence"] = ""; continue
    r["gloss_en"], r["query_1"], r["draft_confidence"] = d
    r["verified"] = ""
    filled += 1

with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader(); w.writerows(rows)

print(f"drafted {filled}/{len(rows)} rows -> {out}")
if missing: print("MISSING:", missing)
import collections
print("confidence:", dict(collections.Counter(r["draft_confidence"] for r in rows)))
