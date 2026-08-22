import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "/mnt/user-data/uploads/2. Pix2Geomdel.V2/Revision/results/9_Directional_variograms.csv"
df = pd.read_csv(CSV)

C_REF, C_PRD, INK, INK2 = "#2a78d6", "#eb6834", "#1a1a1a", "#5a5a5a"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.7,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK, "figure.dpi": 900,
})

DOMS  = [("porosity", "Porosity", "fraction²"),
         ("permeability", "Permeability", "(log₁₀ mD)²"),
         ("vcl", "VCL", "fraction²")]
DIRS  = sorted(df["Direction"].unique(), key=lambda s: int(s.split()[0]))
SHORT = {d: d.split()[0] + "°" for d in DIRS}

fig, axes = plt.subplots(len(DIRS), 3, figsize=(7.2, 7.6), sharex=True)
for i, dirn in enumerate(DIRS):
    for j, (dom, title, unit) in enumerate(DOMS):
        ax = axes[i, j]
        s = df[(df["Domain"] == dom) & (df["Direction"] == dirn)].sort_values("Lag (cells)")
        x = s["Lag (cells)"]
        ax.fill_between(x, s["Ref 95% low"], s["Ref 95% high"], color=C_REF, alpha=0.18, lw=0)
        ax.fill_between(x, s["Pred 95% low"], s["Pred 95% high"], color=C_PRD, alpha=0.18, lw=0)
        ax.plot(x, s["Reference gamma"], color=C_REF, lw=1.4, label="Reference")
        ax.plot(x, s["Predicted gamma"], color=C_PRD, lw=1.4, ls=(0, (4, 2)), label="Predicted")
        ax.grid(True, lw=0.4, color="#e6e6e6")
        ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.tick_params(length=2.5, pad=1.5)
        if i == 0:
            ax.set_title(f"{title}\nγ [{unit}]", fontsize=8, pad=5)
        if j == 0:
            ax.set_ylabel(SHORT[dirn], fontsize=8.5, labelpad=6)
        if i == len(DIRS) - 1:
            ax.set_xlabel("Lag (grid cells)", fontsize=7.5)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(6)

axes[0, 0].legend(frameon=False, fontsize=7, loc="lower right", handlelength=1.8)
fig.text(0.012, 0.5, "Direction (azimuth on the map slice)", rotation=90,
         va="center", fontsize=8.5, color=INK)
fig.tight_layout(rect=[0.035, 0.0, 1, 1])
fig.subplots_adjust(hspace=0.22, wspace=0.30)
fig.savefig("figures_new/Fig_B_variograms.png", dpi=900, bbox_inches="tight", facecolor="white")
fig.savefig("figures_new/Fig_B_variograms.pdf", bbox_inches="tight", facecolor="white")
print("wrote Fig_B_variograms.png / .pdf")
