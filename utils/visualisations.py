"""
Clean charts — single accent colour, white-on-dark, minimal styling.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO

BG     = "#0f172a"
CARD   = "#1e293b"
BORDER = "#334155"
TEXT   = "#e2e8f0"
SUB    = "#94a3b8"
ACCENT = "#3b82f6"
DIM    = "#263248"


def make_confidence_bar(classes, probs, pred_idx):
    n = len(classes)
    fig, ax = plt.subplots(figsize=(5.5, 1.6 + n * 0.42), facecolor=CARD)
    ax.set_facecolor(CARD)

    colors = [ACCENT if i == pred_idx else DIM for i in range(n)]
    bars = ax.barh(classes, probs, color=colors, height=0.5,
                   edgecolor="none")

    for bar, p, i in zip(bars, probs, range(n)):
        ax.text(min(p + 0.012, 0.97), bar.get_y() + bar.get_height() / 2,
                f"{p:.0%}", va="center", ha="left",
                color=TEXT if i == pred_idx else SUB,
                fontsize=9, fontweight="600" if i == pred_idx else "normal")

    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Probability", color=SUB, fontsize=8)
    ax.tick_params(colors=SUB, labelsize=8.5)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.yaxis.set_tick_params(labelcolor=TEXT)
    ax.xaxis.set_tick_params(labelcolor=SUB)
    ax.set_facecolor(CARD)
    fig.patch.set_facecolor(CARD)
    plt.tight_layout(pad=0.8)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=CARD, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


def make_confidence_gauge(confidence, label):
    fig, ax = plt.subplots(figsize=(3.6, 2.2),
                           subplot_kw=dict(aspect="equal"),
                           facecolor=CARD)
    ax.set_facecolor(CARD)

    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), color=BORDER, linewidth=13,
            solid_capstyle="round")

    filled = np.linspace(np.pi, np.pi - confidence * np.pi, 200)
    color = ACCENT if confidence >= 0.65 else ("#f59e0b" if confidence >= 0.45 else "#ef4444")
    ax.plot(np.cos(filled), np.sin(filled), color=color, linewidth=13,
            solid_capstyle="round")

    ax.text(0, -0.1, f"{confidence:.0%}", ha="center", va="center",
            color=TEXT, fontsize=20, fontweight="600")
    ax.text(0, -0.52, "confidence", ha="center", color=SUB, fontsize=8)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.7, 1.15)
    ax.axis("off")
    fig.patch.set_facecolor(CARD)
    plt.tight_layout(pad=0.2)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=CARD, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


def make_scale_attention_bars(scale_names, activations):
    fig, ax = plt.subplots(figsize=(4.5, 2.2), facecolor=CARD)
    ax.set_facecolor(CARD)
    x = np.arange(len(scale_names))
    ax.bar(x, activations, color=ACCENT, alpha=0.85,
           edgecolor="none", width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scale_names, color=SUB, fontsize=8)
    ax.set_ylabel("Activation", color=SUB, fontsize=8)
    ax.tick_params(colors=SUB)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_ylim(0, max(activations) * 1.3)
    for xi, v in zip(x, activations):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", color=SUB, fontsize=7.5)
    fig.patch.set_facecolor(CARD)
    plt.tight_layout(pad=0.8)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=CARD, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf
