import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# # ----------------------------
# # Styling
# # ----------------------------
# sns.set_theme(style="white")
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "cm"

# # ----------------------------
# # Helpers
# # ----------------------------
# def gaussian_smooth(y, sigma=8):
#     """Simple Gaussian smoothing without scipy."""
#     radius = int(4 * sigma + 1)
#     xk = np.arange(-radius, radius + 1)
#     kernel = np.exp(-(xk ** 2) / (2 * sigma ** 2))
#     kernel /= kernel.sum()
#     ypad = np.pad(y, (radius, radius), mode="edge")
#     return np.convolve(ypad, kernel, mode="same")[radius:-radius]

# def smooth_curve(x_anchor, y_anchor, x_dense, wiggle_amp=0.0, wiggle_freq=1.0, wiggle_phase=0.0):
#     """
#     Interpolate anchor points, smooth them, and add a small structured wiggle
#     to mimic a hand-drawn shape while keeping the curve smooth.
#     """
#     y = np.interp(x_dense, x_anchor, y_anchor)
#     y = gaussian_smooth(y, sigma=10)

#     if wiggle_amp > 0:
#         # small smooth irregularity
#         wiggle = (
#             np.sin(wiggle_freq * x_dense + wiggle_phase)
#             + 0.5 * np.sin(0.55 * wiggle_freq * x_dense + 1.7 + wiggle_phase)
#         )
#         wiggle /= np.max(np.abs(wiggle))
#         y = y + wiggle_amp * wiggle

#     return y

# # ----------------------------
# # Data construction
# # ----------------------------
# x = np.linspace(0, 10, 900)

# # Anchor points chosen to resemble the sketch:
# # three upper-region methods with fluctuations and dips
# # one lower-region red curve with a deeper drop
# xk = np.array([0.0, 1.2, 2.4, 3.6, 4.8, 6.2, 7.4, 8.6, 10.0])

# yellow_anchor = np.array([0.76, 0.73, 0.68, 0.62, 0.57, 0.58, 0.61, 0.70, 0.80])
# green_anchor  = np.array([0.28, 0.35, 0.55, 0.73, 0.75, 0.72, 0.60, 0.50, 0.45])
# purple_anchor = np.array([0.12, 0.20, 0.30, 0.47, 0.60, 0.72, 0.79, 0.77, 0.68])
# red_anchor    = np.array([0.50, 0.50, 0.49, 0.46, 0.41, 0.20, 0.02, 0.16, 0.28])

# yellow = smooth_curve(xk, yellow_anchor, x, wiggle_amp=0.010, wiggle_freq=1.8, wiggle_phase=0.1)
# green  = smooth_curve(xk, green_anchor,  x, wiggle_amp=0.012, wiggle_freq=1.6, wiggle_phase=0.9)
# purple = smooth_curve(xk, purple_anchor, x, wiggle_amp=0.010, wiggle_freq=1.7, wiggle_phase=1.8)
# red    = smooth_curve(xk, red_anchor,    x, wiggle_amp=0.008, wiggle_freq=1.5, wiggle_phase=2.4)

# # Keep all curves within plotting bounds
# for arr in (yellow, green, purple, red):
#     np.clip(arr, 0.0, 1.0, out=arr)

# # Pointwise maximum envelope of the three upper curves
# envelope = np.maximum.reduce([yellow, green, purple])
# envelope = gaussian_smooth(envelope, sigma=6)

# # Lift very slightly so the envelope visually sits on top of the curves
# envelope = np.minimum(envelope + 0.03, 0.93)

# # ----------------------------
# # Plot
# # ----------------------------
# fig, ax = plt.subplots(figsize=(11, 6.8))

# # # Shaded upper-performance region
# # ax.fill_between(x, 0, envelope, color="#5DA5F5", alpha=0.20, zorder=1)

# # Envelope curve
# ax.plot(
#     x, envelope,
#     color="#5DA5F5",
#     linewidth=13,
#     alpha=0.8,
#     solid_capstyle="round",
#     zorder=5
# )

# # Performance trajectories
# ax.plot(x, yellow, color="darkgoldenrod", linewidth=5, alpha=0.98, solid_capstyle="round", zorder=4, ls="--")
# ax.plot(x, green,  color="#7FBF2E", linewidth=6, alpha=0.98, solid_capstyle="round", zorder=4, ls="--")
# ax.plot(x, purple, color="#8E24AA", linewidth=6, alpha=0.98, solid_capstyle="round", zorder=4, ls="--")
# ax.plot(x, red,    color="#D7191C", linewidth=7, alpha=0.99, solid_capstyle="round", zorder=4,)

# # ----------------------------
# # Gap annotation
# # ----------------------------
# x_gap = 7.5
# y_top = np.interp(x_gap, x, envelope)
# y_bot = np.interp(x_gap, x, red)

# ax.annotate(
#     "",
#     xy=(x_gap, y_top),
#     xytext=(x_gap, y_bot),
#     arrowprops=dict(arrowstyle="<->", lw=3.5, color="black", mutation_scale=25),
#     zorder=10,
# )

# ax.text(
#     x_gap + 0.20, 0.5 * (y_top + y_bot),
#     "Gap",
#     fontsize=50,
#     va="center",
#     ha="left",
#     color="black"
# )

# # Labels for pi* and pi
# ax.text(10.35, envelope[-1] + 0.02, r"$\pi^\star$", color="#1976D2", fontsize=55, va="center")
# ax.text(10.10, red[-1] - 0.02,      r"$\pi$",       color="#D7191C", fontsize=55, va="center")

# # ----------------------------
# # Axes formatting
# # ----------------------------
# ax.set_xlabel("Time", fontsize=50, labelpad=14)
# ax.set_ylabel("Performance", fontsize=50, labelpad=16)

# ax.set_xlim(-0.1, 10.8)
# ax.set_ylim(0.0, 1.02)

# # Hide ticks to match the stylized sketch more closely
# ax.set_xticks([])
# ax.set_yticks([])

# # Clean frame
# for spine in ("top", "right", "left", "bottom"):
#     ax.spines[spine].set_visible(False)

# # Axis arrows
# xmin, xmax = ax.get_xlim()
# ymin, ymax = ax.get_ylim()

# arrow_kw = dict(
#     arrowstyle="-|>",
#     lw=2.8,
#     color="black",
#     mutation_scale=25,
#     shrinkA=0,
#     shrinkB=0,
# )

# ax.annotate(
#     "",
#     xy=(xmax, ymin),
#     xytext=(xmin, ymin),
#     arrowprops=arrow_kw,
#     annotation_clip=False,
#     zorder=20,
# )

# ax.annotate(
#     "",
#     xy=(xmin, ymax),
#     xytext=(xmin, ymin),
#     arrowprops=arrow_kw,
#     annotation_clip=False,
#     zorder=20,
# )

# plt.tight_layout()
# out_path = './scripts/plots/gap.svg'
# fig.savefig(str(out_path), dpi=300, bbox_inches="tight", pad_inches=0.08)

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Output paths
png_path = "./scripts/plots/ipv4_header_diagram.png"
svg_path = "./scripts/plots/ipv4_header_diagram.svg"

# Figure setup
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_xlim(0, 32)
ax.set_ylim(0, 7)
ax.set_aspect("equal")
ax.axis("off")

# Colors similar to the reference
blue = "#a9d8f3"
green = "#a8efc8"
pink = "#f3b2c5"
edge = "black"

def cell(x, y, w, h, label, color, fontsize=13, weight="bold"):
    ax.add_patch(
        Rectangle(
            (x, y), w, h,
            facecolor=color,
            edgecolor=edge,
            linewidth=1.6
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        family="DejaVu Sans"
    )

# Draw from top to bottom
# Row 1
cell(0, 6, 4, 1, "Version", blue)
cell(4, 6, 3, 1, "IHL", green)
cell(7, 6, 9, 1, "Type of Service", pink)
cell(16, 6, 16, 1, "Total Length", pink)

# Row 2
cell(0, 5, 16, 1, "Identification", green)
cell(16, 5, 7, 1, "Flags", green)
cell(23, 5, 9, 1, "Fragment Offset", green)

# Row 3
cell(0, 4, 7, 1, "Time to Live", pink)
cell(7, 4, 9, 1, "Protocol", pink)
cell(16, 4, 16, 1, "Header Checksum", green)

# Row 4
cell(0, 3, 32, 1, "Source Address", blue)

# Row 5
cell(0, 2, 32, 1, "Destination Address", blue)

# Row 6
cell(0, 1, 26, 1, "Options", green)
cell(26, 1, 6, 1, "Padding", green)

# Outer border
ax.add_patch(Rectangle((0, 1), 32, 6, fill=False, edgecolor="black", linewidth=2.0))

plt.tight_layout(pad=0.25)
fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

