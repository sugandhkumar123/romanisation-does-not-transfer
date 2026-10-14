"""Robustness analyses requested by reviewers, computed from results_per_query_all6.csv (dense mode).
1. paired sign-flip permutation test, Devanagari vs each romanisation, per model
2. does Devanagari beat every romanisation within every text and every length tercile?
3. query-subsampling: how often does Devanagari beat the best romanisation with only n queries?
"""
import csv, random, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
D = defaultdict(dict); meta = {}
for r in rows:
    if r["mode"] != "dense": continue
    D[(r["model"], r["representation"])][r["item_id"]] = float(r["ndcg10"])
    meta[r["item_id"]] = (r["text_id"], r["length_band"])
items = sorted(meta)
models = ["bge", "e5", "labse", "mpnet", "minilm", "muril"]
roms = ["iast", "hk", "ascii_fold"]
mean = lambda xs: sum(xs) / len(xs)
rng = random.Random(20261012)

def perm_p(diff, n_perm=20000):
    obs = abs(mean(diff)); hits = 0
    for _ in range(n_perm):
        s = sum(d if rng.random() < 0.5 else -d for d in diff) / len(diff)
        if abs(s) >= obs: hits += 1
    return (hits + 1) / (n_perm + 1)

print("== 1. paired permutation test, nDCG@10, Devanagari minus romanisation ==")
worst = {}
for m in models:
    dev = [D[(m, "devanagari")][i] for i in items]
    out = []
    for rp in roms:
        rom = [D[(m, rp)][i] for i in items]
        diff = [a - b for a, b in zip(dev, rom)]
        p = perm_p(diff)
        wins = sum(d > 0 for d in diff); losses = sum(d < 0 for d in diff)
        out.append((rp, round(mean(diff), 4), wins, losses, p))
    worst[m] = max(o[4] for o in out)
    print(m, out)
print("max p per model:", {m: round(p, 5) for m, p in worst.items()})

print("\n== 2. Devanagari vs best romanisation within each slice ==")
slices = {}
for i, (t, l) in meta.items():
    slices.setdefault(("text", t), []).append(i); slices.setdefault(("len", l), []).append(i)
total = held = 0
for m in models:
    res = []
    for k, ids in sorted(slices.items()):
        dv = mean([D[(m, "devanagari")][i] for i in ids])
        br = max(mean([D[(m, rp)][i] for i in ids]) for rp in roms)
        total += 1; held += dv > br
        res.append((k[1], len(ids), round(dv, 3), round(br, 3), dv > br))
    print(m, res)
print(f"Devanagari > best romanisation in {held}/{total} model-slice cells")

print("\n== 3. query subsampling (10,000 draws without replacement) ==")
for n in (20, 30, 50, 100):
    line = []
    for m in models:
        win = 0
        for _ in range(10000):
            ids = rng.sample(items, n)
            dv = mean([D[(m, "devanagari")][i] for i in ids])
            br = max(mean([D[(m, rp)][i] for i in ids]) for rp in roms)
            win += dv > br
        line.append(f"{m}:{win/100:.1f}%")
    print(f"n={n}:", " ".join(line))

print("\n== 4. queries ever retrieved in top 10 ==")
for m in models:
    r10 = lambda rp: {i for i in items if D[(m, rp)][i] > 0}
    dev = r10("devanagari"); rom = set().union(*(r10(rp) for rp in roms))
    print(m, "dev:", len(dev), "any romanisation:", len(rom), "rom-only:", len(rom - dev))
