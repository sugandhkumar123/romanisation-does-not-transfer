#!/usr/bin/env python3
"""Apply documented editorial corrections to corpus.jsonl.

Two scribal errors were found in the source e-texts during benchmark
verification. Both are in the source itself, not introduced by the parser
(a scan for systematic corruption found none). They are corrected here
rather than silently, so the change is reproducible and citable.

Each correction is listed with its justification. Section 3 of the paper
should state that two verses carried source-level scribal errors, which
were corrected against the standard reading, and cite this file.

    python fix_corpus.py            # writes corpus_corrected.jsonl
    python fix_corpus.py --inplace  # overwrites corpus.jsonl
"""

import argparse
import json
import re
import unicodedata

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
except ImportError:
    raise SystemExit("pip install indic-transliteration")

# verse_id -> (wrong Devanagari substring, correct substring, justification)
CORRECTIONS = [
    ("hatha_yoga_pradipika:2.1",
     "\u0926\u0943\u0927\u0947",              # dṛdhe
     "\u0926\u0943\u0922\u0947",              # dṛḍhe
     "source reads dṛdhe (dental dh); the word is dṛḍhe 'firm', from √dṛh. "
     "Retroflex ḍh is the standard reading in all editions."),
    ("hatha_yoga_pradipika:4.45",
     "\u0938\u0942\u0930\u094d\u091a\u094d\u092f\u093e",   # sūrcyā
     "\u0938\u0942\u0930\u094d\u092f\u093e",               # sūryā
     "source reads sūrcyā, not a Sanskrit form. The phrase is "
     "sūryā-candramasor madhye, 'between sun and moon'."),
]


def representations(dev):
    dev = unicodedata.normalize("NFC", dev)
    iast = transliterate(dev, sanscript.DEVANAGARI, sanscript.IAST)
    hk = transliterate(dev, sanscript.DEVANAGARI, sanscript.HK)
    fold = re.sub(r"[\u0300-\u036f]", "", unicodedata.normalize("NFD", iast))
    return {"devanagari": dev, "iast": iast, "hk": hk,
            "ascii_fold": unicodedata.normalize("NFC", fold).lower()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus.jsonl")
    ap.add_argument("--out", default="corpus_corrected.jsonl")
    ap.add_argument("--inplace", action="store_true")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.corpus, encoding="utf-8")]
    index = {r["id"]: r for r in records}

    applied, failed = 0, []
    for vid, wrong, right, why in CORRECTIONS:
        r = index.get(vid)
        if r is None:
            failed.append((vid, "verse not in corpus"))
            continue
        if wrong not in r["devanagari"]:
            if right in r["devanagari"]:
                print(f"  [skip] {vid} already corrected")
            else:
                failed.append((vid, "neither form found -- check manually"))
            continue
        r.update(representations(r["devanagari"].replace(wrong, right)))
        r["n_chars"] = len(r["devanagari"])
        r["editorial_note"] = why
        applied += 1
        print(f"  [fix ] {vid}\n         {why}")

    out = args.corpus if args.inplace else args.out
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\napplied {applied}/{len(CORRECTIONS)} corrections -> {out}")
    for vid, msg in failed:
        print(f"  [FAIL] {vid}: {msg}")


if __name__ == "__main__":
    main()
