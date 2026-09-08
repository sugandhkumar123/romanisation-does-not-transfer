#!/usr/bin/env python3
"""
ICCSI-2026 : Cross-Lingual Semantic Retrieval over Classical Yoga Texts
Stage 1 -- corpus construction.  (v2 -- verified source URLs)

Emits every verse in four parallel representations:

    devanagari  : NFC-normalised, Vedic accents stripped
    iast        : yogascittavrttinirodhah (with diacritics)
    hk          : yogazcittavRttinirodhaH
    ascii_fold  : yogascittavrttinirodhah (diacritics removed)

Those four columns are the independent variable of the paper's ablation.

Usage
-----
    pip install indic-transliteration requests
    python corpus_build.py --out corpus.jsonl
    python corpus_build.py --out corpus.jsonl --offline   # use ./raw/ cache

Source-file quirks handled here
-------------------------------
1. yogasuutra.itx contains the WHOLE TEXT TWICE -- once with Vedic accents
   (sasvara) and once without. Accents are ITRANS \\' and \\` markers. We
   strip them, which makes the two copies identical, then de-duplicate on
   the sutra reference. Skip this and you get ~390 sutras with accent
   diacritics polluting the script-representation ablation.
2. ITRANS escapes: \\- (word-join), \\. (in verse numbers), \\' \\` (accents).
   These must be removed BEFORE transliteration or ITRANS mis-parses them.
3. hyp.itx and bhagvadnew.itx number verses 1..N *per chapter*, restarting
   each chapter. We detect the reset (verse number goes down) and prefix
   the chapter, so refs are globally unique: "2.17", not a second "17".
4. Parenthesised variant readings trail the verse number and would
   otherwise be glued onto the NEXT verse. Stripped.

Licensing
---------
The verses themselves are public domain. sanskritdocuments.org asks that
its prepared files not be reposted. So: cite the source and access date,
and release your BENCHMARK (query -> verse-reference pairs) rather than a
dump of the source text. That is the safe and standard thing to do, and it
is what a reviewer expects for a retrieval benchmark anyway.
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
except ImportError:
    sys.exit("pip install indic-transliteration")

RAW = Path("raw")
RAW.mkdir(exist_ok=True)

# --- URLs verified 2026-09-06 ------------------------------------------------
SOURCES = {
    "yoga_sutras": {
        "title": "Patanjali Yoga Sutras",
        "url": "https://sanskritdocuments.org/doc_yoga/yogasuutra.itx",
        "unit": "sutra",
        "expected": 195,
        "numbered_refs": True,     # file already gives 1.1, 1.2, ...
    },
    "bhagavad_gita": {
        "title": "Bhagavad Gita",
        "url": "https://sanskritdocuments.org/doc_giitaa/bhagvadnew.itx",
        "unit": "verse",
        "expected": 700,
        "numbered_refs": False,    # restarts each chapter -> infer chapter
    },
    "hatha_yoga_pradipika": {
        "title": "Hatha Yoga Pradipika",
        "url": "https://sanskritdocuments.org/doc_yoga/hyp.itx",
        "unit": "verse",
        "expected": 389,
        "numbered_refs": False,
    },
}

# Vedic accent / svara marks that survive into Devanagari
ACCENTS = dict.fromkeys(
    [0x0951, 0x0952, 0x0953, 0x0954, 0x1CDA, 0x1CDB, 0x1CDC, 0x1CDD,
     0x1CE0, 0x1CE1, 0xA8E0, 0xA8F1], None)

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

LATEX = re.compile(r"\\[a-zA-Z]+\*?(\{[^{}]*\})*")
COMMENT = re.compile(r"^%.*$", re.M)
PAREN = re.compile(r"\([^()]{0,200}\)")
# verse terminator carrying a number:  || 1.2||   || 17||   dandas too
VERSE_END = re.compile(r"[\u0965|]{1,2}\s*([\u0966-\u096F\d]+)"
                       r"(?:\\?\.([\u0966-\u096F\d]+))?\s*[\u0965|]{1,2}")


def fetch(text_id, meta, offline):
    cache = RAW / f"{text_id}.itx"
    if cache.exists():
        print(f"  [cache] {cache}")
        return cache.read_text(encoding="utf-8", errors="replace")
    if offline:
        print(f"  [skip ] no cached {cache}")
        return None
    import requests
    print(f"  [fetch] {meta['url']}")
    try:
        r = requests.get(meta["url"], timeout=30,
                         headers={"User-Agent": "academic-corpus-build/1.0"})
        r.raise_for_status()
    except Exception as e:                                   # noqa: BLE001
        print(f"  [FAIL ] {e}\n          save manually to {cache}")
        return None
    r.encoding = "utf-8"
    cache.write_text(r.text, encoding="utf-8")
    return r.text


def drop_preamble(t):
    """Cut the ITRANS/LaTeX header block.

    These files open with #include=ijag.inc, \\parindent=100pt and similar,
    none of it Sanskrit. Left in, it transliterates into a garbage record
    that also steals the first verse number and shifts chapter inference.
    """
    for marker in ("\\endtitles", "\\begin{document}"):
        i = t.find(marker)
        if i != -1:
            return t[i + len(marker):]
    return t


def unescape_itrans(t):
    """Remove ITRANS escapes that would otherwise corrupt transliteration."""
    t = drop_preamble(t)
    t = COMMENT.sub(" ", t)
    t = re.sub(r"\\[`'\"]", "", t)       # Vedic accent / svarita markers
    # ITRANS ".h" is an explicit virama. indic-transliteration does NOT
    # implement it: it renders "." as a danda and leaves the "h", producing
    # "vid|hdhyanAdI" inside a word. Deleting ".h" gives the correct
    # conjunct ("vid.hdhyanAdI" -> "viddhyanAdI" -> viddhyanAdI).
    t = t.replace(".h", "")
    # bhagvadnew.itx writes refs as "1\\-1" (chapter\\-verse). Convert those to
    # "1.1" BEFORE stripping word-join hyphens, or the ref collapses to "11".
    t = re.sub(r"(\d)\\-(\d)", r"\1.\2", t)
    t = t.replace("\\-", "")            # word-join hyphen
    t = t.replace("\\_", "")
    t = re.sub(r"\\\.", ".", t)         # escaped period inside verse numbers
    t = LATEX.sub(" ", t)               # \documentstyle, \section{...}, ...
    t = re.sub(r"\[[^\[\]]{0,120}\]", " ", t)   # leftover [11pt,multicol,...]
    t = t.replace("##", " ")            # ITRANS non-Sanskrit delimiter
    t = PAREN.sub(" ", t)               # variant readings
    t = t.replace("\u200b", "").replace("\u00a0", " ")
    return re.sub(r"[ \t]+", " ", t)


def strip_accents_dev(s):
    return unicodedata.normalize("NFC", s.translate(ACCENTS))


def to_devanagari(t):
    if re.search(r"[\u0900-\u097F]", t):
        return t
    return transliterate(t, sanscript.ITRANS, sanscript.DEVANAGARI)


# A colophon opens with "iti" AND names the work or chapter unit. Matching
# bare "iti" is WRONG: real verses begin that way -- BG 13.19 (iti kShetraM
# tathA), BG 18.63 (iti te j~nAnam), HYP 1.31, HYP 3.128 were all silently
# deleted by that mistake.
COLOPHON_ROMAN = re.compile(
    r"^iti\b.{0,240}?(adhyAyaH|upadeshaH|pAdaH|upaniShat|pradIpikAyAM|"
    r"yogashAstre|saMhitAyAM|samApt)", re.I | re.S)
COLOPHON_DEV = re.compile(
    r"^\u0907\u0924\u093f.{0,240}?(\u0905\u0927\u094d\u092f\u093e\u092f\u0903|"
    r"\u0909\u092a\u0926\u0947\u0936\u0903|\u092a\u093e\u0926\u0903|"
    r"\u0909\u092a\u0928\u093f\u0937\u0924\u094d|\u0938\u092e\u093e\u092a\u094d\u0924)",
    re.S)


def is_colophon(chunk):
    return bool(COLOPHON_ROMAN.match(chunk) or COLOPHON_DEV.match(chunk))


CHAPTER_WORD = re.compile(
    r"^\s*\S*(?:padeshaH|adhyAyaH|pAdaH|\u092a\u0926\u0947\u0936\u0903|"
    r"\u0905\u0927\u094d\u092f\u093e\u092f\u0903)\s+", re.I)


def strip_leading_matter(chunk):
    """Remove invocation lines and chapter names glued to a verse.

    A double danda inside a chunk is always a heading or invocation
    delimiter (half-verses use a SINGLE danda), so anything before the last
    one is not verse text. Chapter names -- prathamopadeshaH, dvitIyo-
    .adhyAyaH -- trail a bare chapter marker and are stripped by name.
    """
    if "||" in chunk:
        chunk = chunk.rsplit("||", 1)[-1]
    if "\u0965" in chunk:
        chunk = chunk.rsplit("\u0965", 1)[-1]
    chunk = CHAPTER_WORD.sub("", chunk)
    return chunk.strip(" \u0964|\u0965")


def split_verses(text, numbered_refs):
    """Yield (ref, chunk) from the SOURCE script.

    Splitting must happen before transliteration: ITRANS maps '.' to the
    Devanagari danda, so transliterating first turns the reference "1.1"
    into "1|1" and the verse numbering is lost.
    """
    out, cursor = [], 0
    chapter, prev_n = 1, 0

    for m in VERSE_END.finditer(text):
        chunk = text[cursor:m.start()]
        cursor = m.end()

        chunk = re.sub(r"\s+", " ", chunk).strip(" \n\u0964|\u0965")
        chunk = strip_leading_matter(chunk)
        # drop colophons / bare chapter-header markers: these carry no verse
        # text, and must NOT advance the chapter counter (hyp.itx opens each
        # chapter with a bare "|| 1||" that would otherwise fake a reset)
        if len(chunk) < 8 or is_colophon(chunk):
            continue

        a = m.group(1).translate(DEV_DIGITS)
        b = m.group(2).translate(DEV_DIGITS) if m.group(2) else None

        explicit = b is not None
        if explicit:
            ref = f"{a}.{b}"
        elif numbered_refs:
            ref = a
        else:
            n = int(a)
            if n <= prev_n:                 # counter reset -> new chapter
                chapter += 1
            prev_n = n
            ref = f"{chapter}.{n}"

        out.append((ref, chunk, explicit))

    # If the text numbers verses as chapter.verse, any marker WITHOUT a
    # sub-number is a chapter colophon ("iti ... adhyAyaH || 18||"), not a
    # verse. Drop those rather than letting them invent chapters.
    if out and sum(e for _, _, e in out) >= 0.5 * len(out):
        out = [(r, c) for r, c, e in out if e]
    else:
        out = [(r, c) for r, c, _ in out]
    return out


def representations(dev):
    dev = strip_accents_dev(dev)
    iast = transliterate(dev, sanscript.DEVANAGARI, sanscript.IAST)
    hk = transliterate(dev, sanscript.DEVANAGARI, sanscript.HK)
    fold = re.sub(r"[\u0300-\u036f]", "", unicodedata.normalize("NFD", iast))
    return {"devanagari": dev,
            "iast": iast,
            "hk": hk,
            "ascii_fold": unicodedata.normalize("NFC", fold).lower()}


def roundtrip_ok(dev):
    back = transliterate(transliterate(dev, sanscript.DEVANAGARI, sanscript.HK),
                         sanscript.HK, sanscript.DEVANAGARI)
    return unicodedata.normalize("NFC", back) == unicodedata.normalize("NFC", dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpus.jsonl")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--report-gaps", action="store_true",
                    help="list missing verse numbers per chapter")
    args = ap.parse_args()

    records, rt_fail = [], 0

    for text_id, meta in SOURCES.items():
        print(f"\n{meta['title']}")
        raw = fetch(text_id, meta, args.offline)
        if not raw:
            continue

        src = unescape_itrans(raw)
        verses = [(ref, to_devanagari(chunk))
                  for ref, chunk in split_verses(src, meta["numbered_refs"])]
        n_raw = len(verses)

        # de-duplicate: yogasuutra ships the text twice (sasvara + nihsvara)
        seen, uniq = set(), []
        for ref, dev in verses:
            if ref in seen:
                continue
            seen.add(ref)
            uniq.append((ref, dev))
        if n_raw != len(uniq):
            print(f"  [dedup] {n_raw} -> {len(uniq)} "
                  f"(dropped {n_raw - len(uniq)} duplicate refs)")

        print(f"  [split] {len(uniq)} {meta['unit']}s "
              f"(expected ~{meta['expected']})")
        if abs(len(uniq) - meta["expected"]) > 0.1 * meta["expected"]:
            print("  [WARN ] count off by >10% -- open the raw file and check "
                  "VERSE_END against its numbering convention")

        for i, (ref, dev) in enumerate(uniq, 1):
            if not roundtrip_ok(dev):
                rt_fail += 1
            records.append({
                "id": f"{text_id}:{ref}",
                "text_id": text_id,
                "title": meta["title"],
                "unit": meta["unit"],
                "ref": ref,
                "seq": i,
                "n_chars": len(dev),
                "translation_en": None,      # stage 2
                **representations(dev),
            })

    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 62}")
    print(f"wrote {len(records)} verses -> {args.out}")
    print(f"round-trip failures: {rt_fail}"
          + ("  <-- inspect: a lossy round-trip biases the script ablation"
             if rt_fail else ""))
    print("\nspot-check the first few records before trusting the file.")


if __name__ == "__main__":
    main()
