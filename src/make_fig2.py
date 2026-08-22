"""Figure 2 — dataset construction under the leakage-free protocol.
Renders a 900 dpi PNG + vector PDF, and appends a fully editable
native-shape slide to figures_new/Figures_editable.pptx."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

C_TR, C_VA, C_TE, C_BU = "#2a78d6", "#eda100", "#1baf7a", "#c2c6cc"
INK, INK2, RULE = "#0b0b0b", "#52514e", "#c9ccd1"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "figure.dpi": 900})

# geometry (figure units 0-100 x 0-52)
COL_X, COL_W, COL_TOP, COL_BOT = 5.0, 4.6, 45.0, 9.0
BLOCKS = [(1, 36, C_TR, "Training", "layers 1–36", "36 layers"),
          (37, 37, C_BU, "", "buffer", ""),
          (38, 44, C_VA, "Validation", "layers 38–44", "7 layers"),
          (45, 45, C_BU, "", "buffer", ""),
          (46, 54, C_TE, "Test", "layers 46–54", "9 layers")]
H = (COL_TOP - COL_BOT) / 54.0

fig = plt.figure(figsize=(7.4, 3.85))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")

def ytop(n): return COL_TOP - (n - 1) * H

# ---- depth column -------------------------------------------------------
for a, b, col, name, rng, cnt in BLOCKS:
    y1, y0 = ytop(a), ytop(b + 1)
    ax.add_patch(Rectangle((COL_X, y0), COL_W, y1 - y0, facecolor=col,
                           edgecolor="white", lw=0.7, zorder=2))
    yc = (y0 + y1) / 2
    xl = COL_X + COL_W + 1.4
    if name:
        ax.plot([COL_X + COL_W + 0.3, xl - 0.4], [yc, yc], lw=0.7, color=RULE, zorder=1)
        ax.text(xl, yc + 1.05, name, ha="left", va="center", fontsize=8, weight="bold")
        ax.text(xl, yc - 1.35, rng + "  ·  " + cnt, ha="left", va="center",
                fontsize=6.8, color=INK2)
    else:
        ax.text(xl, yc, "buffer, discarded", ha="left", va="center",
                fontsize=6.3, color=INK2, style="italic")
ax.add_patch(Rectangle((COL_X, COL_BOT), COL_W, COL_TOP - COL_BOT,
                       fill=False, edgecolor=INK2, lw=0.8, zorder=3))

ax.text(COL_X + COL_W / 2, COL_TOP + 2.6, "54 retained layers", ha="center",
        fontsize=9, weight="bold")
ax.annotate("", xy=(COL_X - 1.6, COL_BOT), xytext=(COL_X - 1.6, COL_TOP),
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK2, mutation_scale=9))
ax.text(COL_X - 2.9, (COL_TOP + COL_BOT) / 2, "increasing depth", rotation=90,
        ha="center", va="center", fontsize=7, color=INK2)

# ---- flow boxes ---------------------------------------------------------
def card(cx, cy, w, h, title, lines, col):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=1.1",
                                facecolor="white", edgecolor=col, lw=1.4, zorder=3))
    ax.text(cx, cy + h / 2 - 3.0, title, ha="center", va="center",
            fontsize=8.4, weight="bold", color=col, zorder=4)
    ax.text(cx, cy - 1.6, "\n".join(lines), ha="center", va="center",
            fontsize=7, color=INK2, zorder=4, linespacing=1.6)

card(46, 34.5, 25, 15, "Augmentation",
     ["applied to the training", "layers only", "30 realisations per layer"], C_TR)
card(46, 15.0, 25, 15, "No augmentation",
     ["validation and test layers", "kept as original slices"], C_TE)
card(80, 25.0, 24, 20, "Paired images",
     ["1,116 training pairs", "7 validation pairs", "9 test pairs", "", "per translation task"], INK2)

def arrow(p, q, col, lw=1.1):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10,
                                 lw=lw, color=col, shrinkA=0, shrinkB=0, zorder=2))
arrow((26.0, 34.5), (33.5, 34.5), C_TR)
arrow((26.0, 15.0), (33.5, 15.0), C_TE)
arrow((58.5, 34.5), (68, 29.0), INK2, 1.0)
arrow((58.5, 15.0), (68, 21.0), INK2, 1.0)
ax.text(32.5, 36.4, "× 31", ha="right", fontsize=7, color=C_TR, weight="bold")
ax.text(32.5, 16.9, "× 1", ha="right", fontsize=7, color=C_TE, weight="bold")

ax.plot([5, 95], [5.4, 5.4], lw=0.6, color=RULE)
ax.text(50, 3.0, "The split is performed at layer level before augmentation, so no transformed copy "
        "of a validation or test layer can appear in training.",
        ha="center", va="center", fontsize=7.2, color=INK2, style="italic")

fig.savefig("figures_final/Figure_3.png", dpi=900, bbox_inches="tight", facecolor="white")
fig.savefig("figures_new/Fig_2_split.pdf", bbox_inches="tight", facecolor="white")
print("wrote Figure_2 png/pdf")
