import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# drawn at final printed width so lettering is >= 6 pt in print; greyscale +
# dash pattern carry the meaning, since Springer prints figures in black and white
FIG_W = 5.9
reps = ["Devanagari", "IAST", "Harvard-Kyoto", "ASCII-folded"]
data = {
    "BGE-M3":    [0.248, 0.009, 0.013, 0.005],
    "mE5-large": [0.190, 0.014, 0.012, 0.006],
    "LaBSE":     [0.110, 0.007, 0.008, 0.009],
    "mpnet":     [0.096, 0.002, 0.002, 0.000],
    "MiniLM":    [0.091, 0.002, 0.005, 0.000],
    "MuRIL":     [0.030, 0.007, 0.003, 0.005],
}
stub = [0.0138, 0.0032, 0.0022, 0.0000]

fig, ax = plt.subplots(figsize=(FIG_W, 3.2), dpi=1200)
x = np.arange(len(reps)); w = 0.13
greys = ["#111111", "#3d3d3d", "#5f5f5f", "#828282", "#a5a5a5", "#cccccc"]
for i, (name, vals) in enumerate(data.items()):
    ax.bar(x + (i - 2.5) * w, vals, w, label=name,
           color=greys[i], edgecolor="black", linewidth=0.45)
for j, s in enumerate(stub):
    ax.plot([x[j] - 3.1 * w, x[j] + 3.1 * w], [s, s],
            color="black", ls=(0, (4, 2)), lw=1.4, zorder=6,
            label="character-hash baseline" if j == 0 else None)
ax.annotate("character-hash baseline\n(measured per representation)",
            xy=(x[1] + 3.1 * w, stub[1]), xytext=(1.98, 0.082),
            color="black", fontsize=7,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

ax.set_ylabel("nDCG@10", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(reps, fontsize=8)
ax.tick_params(axis="y", labelsize=7.5)
ax.legend(fontsize=7, ncol=4, frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(0, 0.30)
plt.tight_layout()
plt.savefig("figure2_script_representation.png", bbox_inches="tight")
plt.savefig("figure2_script_representation.pdf", bbox_inches="tight")
print("ok")
