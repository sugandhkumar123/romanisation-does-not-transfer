#!/usr/bin/env python3
"""
Answer one question: are the verses missing from corpus.jsonl absent from
the SOURCE FILE (an editorial recension difference -- fine, just document
it), or is corpus_build.py eating them (a parser bug -- must fix)?

    python check_gaps.py

Reads raw/*.itx and corpus.jsonl from the current directory. No arguments,
no shell quoting, works the same on Windows and Linux.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

RAW = Path("raw")
CORPUS = Path("corpus.jsonl")

# how each source writes a verse marker, given chapter and verse number
MARKERS = {
    "bhagavad_gita": lambda c, v: [rf"\|\|\s*{c}\\?-{v}\s*\|\|",
                                   rf"\|\|\s*{c}\\?\.{v}\s*\|\|"],
    # HYP numbers verses per chapter, so a bare "|| 103||" occurs in more
    # than one chapter. Reported context may be from the wrong chapter --
    # treat an HYP hit as "exists somewhere", not as chapter-accurate.
    "hatha_yoga_pradipika": lambda c, v: [rf"\|\|\s*{v}\s*\|\|"],
    "yoga_sutras": lambda c, v: [rf"\|\|\s*{c}\\?\.{v}\s*\|\|"],
}
RAW_NAME = {
    "bhagavad_gita": "bhagavad_gita.itx",
    "hatha_yoga_pradipika": "hatha_yoga_pradipika.itx",
    "yoga_sutras": "yoga_sutras.itx",
}


def gaps(records):
    by = defaultdict(lambda: defaultdict(set))
    for r in records:
        ch, _, v = r["ref"].partition(".")
        if v.isdigit():
            by[r["text_id"]][int(ch)].add(int(v))
    out = defaultdict(list)
    for tid, chapters in by.items():
        for ch in sorted(chapters):
            seen = chapters[ch]
            for v in range(1, max(seen) + 1):
                if v not in seen:
                    out[tid].append((ch, v))
    return out


def main():
    if not CORPUS.exists():
        raise SystemExit(f"{CORPUS} not found -- run corpus_build.py first")

    records = [json.loads(l) for l in CORPUS.open(encoding="utf-8")]
    found = gaps(records)

    if not found:
        print("no gaps in corpus.jsonl -- nothing to check")
        return

    for tid, missing in found.items():
        path = RAW / RAW_NAME[tid]
        print(f"\n{'=' * 66}\n{tid}  ({len(missing)} gap(s))")
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")

        for ch, v in missing:
            hits = []
            for pat in MARKERS[tid](ch, v):
                hits = list(re.finditer(pat, text))
                if hits:
                    break
            print(f"\n  chapter {ch}, verse {v}:")
            if not hits:
                print("    NOT IN SOURCE -> editorial recension difference.")
                print("    Document it in Section 3; no code change needed.")
                continue
            # Bare "|| N||" markers carry no chapter, so the same number
            # appears once per chapter. Show every occurrence rather than
            # assuming the first one is the relevant chapter.
            if len(hits) > 1:
                print(f"    AMBIGUOUS -- {len(hits)} markers numbered {v} in "
                      f"this file (one per chapter). Contexts:")
            else:
                print("    PRESENT IN SOURCE -> the parser dropped it. Context:")
            for h in hits[:4]:
                a, b = max(0, h.start() - 200), h.end() + 50
                ctx = re.sub(r"\s+", " ", text[a:b]).strip()
                print(f"      ...{ctx}...")

    print(f"\n{'=' * 66}")
    print("Send me any block marked PRESENT IN SOURCE, with its context line.")


if __name__ == "__main__":
    main()
