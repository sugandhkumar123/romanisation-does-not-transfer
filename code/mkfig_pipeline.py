import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Drawn at final printed width (LNCS text block 12.2 cm = 4.8 in) so source pt == printed pt.
# Springer 4.5: figure lettering >= 6 pt, line art >= 800 dpi.
FIG_W, FIG_H = 4.7, 3.2
XMAX, YMAX = 105.0, 53.0
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=1200)
ax.set_xlim(-3.5, XMAX + 3.5); ax.set_ylim(-0.3, YMAX + 0.5); ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)   # axes span the full printed width
GREY = "#ececec"; EC = "#333333"
BODY, LABEL = 6.0, 7.0                      # both >= 6 pt

_checks = []
def box(x, y, w, h, text, fc="white", fs=BODY, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                                facecolor=fc, edgecolor=EC, linewidth=0.85))
    t = ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, fontweight=weight, linespacing=1.4)
    _checks.append((t, x, y, w, h, text))

def arr(x1, y1, x2, y2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=7, lw=0.85, linestyle=ls,
                                 color=EC, shrinkA=0, shrinkB=0))

W, G = 24.0, 3.0
xs = [i * (W + G) for i in range(4)]          # 0, 26, 52, 78 -> right edge 100

ax.text(-3.5, 50.5, "A.  Corpus construction", fontsize=LABEL, fontweight="bold")
yA, hA = 37.0, 10.5
box(xs[0], yA, W, hA, "3 source e-texts\n(ITRANS)\nSutras · Gita · HYP")
box(xs[1], yA, W, hA, "normalise\npreamble, escapes\nvirama, colophons")
box(xs[2], yA, W, hA, "1,288 verses\nround-trip\nverified", fc=GREY, weight="bold")
box(xs[3], yA, W, hA, "4 representations\nDevanagari, IAST\nHarvard-Kyoto\nASCII-folded", fc=GREY, weight="bold")
for i in range(3):
    arr(xs[i] + W, yA + hA/2, xs[i+1], yA + hA/2)

ax.text(-3.5, 32.0, "B.  Benchmark construction", fontsize=LABEL, fontweight="bold",
        zorder=5, bbox=dict(facecolor="white", edgecolor="none", pad=1.2))  # dotted connector passes behind the label
yB, hB = 19.5, 10.5
box(xs[0], yB, W, hB, "stratified sample\ntext × chapter\n× length tercile")
box(xs[1], yB, W, hB, "LLM drafts gloss\n+ query, author\nverifies vs source")
box(xs[2], yB, W, hB, "200 query–verse\npairs, single gold\n(7/200 corrected)", fc=GREY, weight="bold")
for i in range(2):
    arr(xs[i] + W, yB + hB/2, xs[i+1], yB + hB/2)
lx, ly = xs[0] + W/2, 35.6
ax.plot([xs[2] + W/2, xs[2] + W/2], [yA, ly], color=EC, lw=0.85, ls=":")
ax.plot([xs[2] + W/2, lx], [ly, ly], color=EC, lw=0.85, ls=":")
arr(lx, ly, lx, yB + hB, ls=":")

ax.text(-3.5, 14.5, "C.  Evaluation", fontsize=LABEL, fontweight="bold")
yC, hC = 1.0, 10.0
box(14, yC, 91, hC, "", fc=GREY)                # frame only; two text blocks below
t1 = ax.text(59.5, yC + hC - 2.6, "6 embedding models  ×  4 representations  ×  2 retrieval modes  =  48 configurations",
             ha="center", va="center", fontsize=BODY, fontweight="bold")
t2 = ax.text(59.5, yC + 3.4, "dense (cosine)  ·  hybrid (BM25 + RRF)  ·  nDCG@10 · MRR · R@5 · R@10\n"
             "1,000-sample bootstrap CI  ·  baselines: random, character-hash",
             ha="center", va="center", fontsize=BODY, linespacing=1.4)
_checks.append((t1, 14, yC + hC/2, 91, hC/2, "C bold line"))
_checks.append((t2, 14, yC, 91, hC*0.6 + 1.0, "C detail lines"))
arr(xs[3] + W/2, yA, xs[3] + W/2, yC + hC)
arr(xs[2] + W/2, yB, xs[2] + W/2, yC + hC)

# --- verify every label actually fits inside its box, and nothing leaves the canvas
fig.canvas.draw()
inv = ax.transData.inverted()
bad = []
for t, x, y, w, h, label in _checks:
    bb = inv.transform(t.get_window_extent(fig.canvas.get_renderer()))
    tw, th = bb[1][0] - bb[0][0], bb[1][1] - bb[0][1]
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    clear_in = (w - tw) / span * FIG_W
    if clear_in < 0.06 or th > h - 1.0:
        bad.append((label.split('\n')[0][:30], f'clearance {clear_in:.3f}in'))
    if bb[0][0] < -3.0 or bb[1][0] > XMAX + 3.0:
        bad.append(("OFF-CANVAS: " + label.split("\n")[0][:24], round(bb[0][0], 1), round(bb[1][0], 1), 0, 0))
if bad:
    print("OVERFLOW:")
    for b in bad: print("  ", b)
else:
    print("all labels fit inside their boxes; nothing off-canvas")

plt.savefig("figure1_pipeline.png", bbox_inches="tight", pad_inches=0.03)
plt.savefig("figure1_pipeline.pdf", bbox_inches="tight", pad_inches=0.03)
