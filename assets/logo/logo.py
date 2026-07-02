import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_facecolor("#0d1117")
fig.patch.set_facecolor("#0d1117")
ax.set_xlim(0, 295*2)
ax.set_ylim(30, 370)
ax.set_aspect("equal")
ax.axis("off")
ax.invert_yaxis()

# ── Hintergrundgitter ──────────────────────────────────────────────────────────
grid_color = "#4a90d9"
grid_alpha = 0.18
grid_spacing = 20
for x in np.arange(15, 590, grid_spacing):
    ax.plot([x, x], [0, 370], color=grid_color, lw=0.4, alpha=grid_alpha)
for y in np.arange(10, 370, grid_spacing):
    ax.plot([0, 590], [y, y], color=grid_color, lw=0.4, alpha=grid_alpha)

# ── Gravitationsfeld-Ringe ─────────────────────────────────────────────────────
cx, cy = 295, 190
for r, alpha in [(110, 0.28), (138, 0.18), (164, 0.10)]:
    ring = Circle((cx, cy), r, fill=False, edgecolor=grid_color,
                  linewidth=0.75, linestyle=(0, (5, 8)), alpha=alpha)
    ax.add_patch(ring)

# ── Großer Körper (Target) ─────────────────────────────────────────────────────
target = Circle((cx, cy), 85, facecolor="#1a5fa3", edgecolor="#4a90d9",
                linewidth=1.5, alpha=0.87, zorder=3)
ax.add_patch(target)

# Partikel im Target (geclippt durch Maske)
target_clip = Circle((cx, cy), 83, transform=ax.transData)
xs = np.arange(215, 380, 1/2*grid_spacing) + grid_spacing//4
ys = np.arange(110, 275, 1/2*grid_spacing) + grid_spacing//4
for x in xs:
    for y in ys:
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        if dist < 82:
            ax.plot(x, y, "o", color="#6ab4f8", markersize=2.5,
                    alpha=0.65, zorder=4, clip_on=True)

# ── Anflugbahn des Impaktors ───────────────────────────────────────────────────
ax.plot([458, 404], [80, 112], color="#e07820", lw=1.5,
        linestyle=(0, (4, 5)), alpha=0.55, zorder=2)

# ── Impaktor ──────────────────────────────────────────────────────────────────
ix, iy = 371, 133
impactor = Circle((ix, iy), 20, facecolor="#e07820", edgecolor="#f0a040",
                  linewidth=1.5, alpha=0.87, zorder=5)
ax.add_patch(impactor)


# Partikel im Impaktor
# for dx in [-8, 0, 8]:
#     for dy in [-8, 0, 8]:
#         if np.sqrt(dx**2 + dy**2) < 18:
#             ax.plot(ix + dx, iy + dy, "o", color="#fad090",
#                     markersize=2.2, alpha=0.80, zorder=6)
target_clip = Circle((ix, iy), 20, transform=ax.transData)
xs = np.arange(ix-20, ix+20, 1/2*grid_spacing) + grid_spacing//4
ys = np.arange(iy-20, iy+20, 1/2*grid_spacing) + grid_spacing//4
for x in xs:
    for y in ys:
        dist = np.sqrt((x - ix)**2 + (y - iy)**2)
        if dist < 19:
            ax.plot(x, y, "o", color="#fad090", markersize=2.5,
                    alpha=0.65, zorder=4, clip_on=True)

# ── Einschlags-Strahlenkranz ───────────────────────────────────────────────────
burst_origin = (363, 139)
burst_dirs = [
    (0,   -24, 0.90),
    (11,  -25, 0.85),
    (19,  -19, 0.80),
    (24,  -11, 0.75),
    (24,    0, 0.70),
    (23,   11, 0.65),
    (-9,  -24, 0.82),
]
for dx, dy, alpha in burst_dirs:
    ax.plot([burst_origin[0], burst_origin[0] + dx],
            [burst_origin[1], burst_origin[1] + dy],
            color="#ffd060", lw=1.75, solid_capstyle="round",
            alpha=alpha, zorder=7)

# ── Ejecta-Partikel ────────────────────────────────────────────────────────────
ejecta = [
    (363, 108, 2.5, "#ffd060", 0.90),
    (380, 104, 3.0, "#ffd060", 0.85),
    (395,  97, 2.5, "#ffd060", 0.75),
    (408, 106, 2.0, "#ffd060", 0.62),
    (390, 116, 3.0, "#ffd060", 0.80),
    (406, 120, 2.0, "#ffd060", 0.58),
    (395, 133, 3.0, "#e07820", 0.85),
    (413, 132, 2.0, "#e07820", 0.63),
    (405, 146, 2.5, "#e07820", 0.75),
    (392, 153, 3.0, "#e07820", 0.70),
    (421, 118, 1.5, "#ffd060", 0.48),
    (426, 138, 1.5, "#e07820", 0.44),
    (378, 158, 2.0, "#e07820", 0.55),
]
for ex, ey, es, ec, ea in ejecta:
    ax.plot(ex, ey, "o", color=ec, markersize=es, alpha=ea, zorder=8)

# ── Schriftzug SMASH ──────────────────────────────────────────────────────────
ax.text(cx, 308, "SMASH",
        ha="center", va="center",
        fontsize=62, fontweight="bold", color="#1a5fa3",
        fontfamily="DejaVu Sans",
        zorder=9)

ax.text(cx, 338, "Self-gravitating Meshfree Analysis of Shock & Hypervelocity",
        ha="center", va="center",
        fontsize=8.5, color="#8ab4d8",
        fontfamily="DejaVu Sans",
        zorder=9)

plt.tight_layout(pad=0)
# plt.show()
plt.savefig("smash_logo.png",
            dpi=200, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.savefig("smash_logo.svg",
            bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Gespeichert: smash_logo.png und smash_logo.svg")