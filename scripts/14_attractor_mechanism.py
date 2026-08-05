"""
Figure 14 -- The bubble mechanism, read off the attractor.

Panel (a): the attractor in the plane of consecutive deviations, with each
point coloured by which forecasting rule holds the largest share at that
moment. Panel (b): a single bubble episode drawn as a time series, with the
same colouring shown as a band underneath. Panel (c): the average composition
of the market conditional on the size of the deviation.

Together they describe one mechanism. Near the fundamental value no rule
dominates. As the deviation starts to grow, trend followers gain share, which
pushes the price further, which increases their share again -- the feedback
that produces the long climbs in panel (b). The climb ends when the deviation
becomes large enough that pessimists, forecasting a constant near zero, are the
ones positioned correctly for the reversal; their share jumps and the price
collapses in a single period. The cycle then restarts.

Nothing external triggers the crash. It arrives because the population that
drove the bubble is, at the top of it, the population positioned worst for what
happens next.
"""

from _common import FIGDIR, announce

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

from bh1998 import calibration, simulate
from bh1998.plotting import MUTED, label_panel, save


def main() -> None:
    p, rules = calibration("chaotic", n_steps=40_000, burn_in=5_000)
    res = simulate(p, rules).trimmed()
    x, n = res.x, res.n
    dominant = np.argmax(n, axis=1)

    cmap = ListedColormap(res.colors)
    norm = BoundaryNorm(np.arange(-0.5, len(res.colors) + 0.5), cmap.N)

    fig = plt.figure(figsize=(15.0, 5.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.25, 1.0],
                          height_ratios=[4, 1], wspace=0.28, hspace=0.08)

    # --- (a) attractor -----------------------------------------------------
    ax = fig.add_subplot(gs[:, 0])
    ax.scatter(x[:-1], x[1:], c=dominant[:-1], cmap=cmap, norm=norm,
               s=1.4, alpha=0.55, linewidths=0, rasterized=True)
    ax.plot([x.min(), x.max()], [x.min(), x.max()], color=MUTED, lw=0.7,
            ls=":")
    ax.set_xlabel(r"$x_{t-1}$")
    ax.set_ylabel(r"$x_t$")
    ax.set_title("Attractor, coloured by dominant rule", fontsize=10.5)
    label_panel(ax, "(a)")

    # --- (b) one episode ---------------------------------------------------
    peak = int(np.argmax(x[:4000]))
    lo, hi = max(peak - 20, 0), peak + 14
    seg_x = x[lo:hi]
    seg_d = dominant[lo:hi]
    t = np.arange(seg_x.size)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, seg_x, color="#333333", lw=1.0, zorder=1)
    ax.scatter(t, seg_x, c=seg_d, cmap=cmap, norm=norm, s=34, zorder=2,
               edgecolor="white", linewidths=0.4)
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_xlim(0, t[-1])
    ax.set_xticklabels([])
    ax.set_ylabel(r"Deviation $x_t$")
    ax.set_title("A single bubble and its collapse", fontsize=10.5)
    label_panel(ax, "(b)")

    axb = fig.add_subplot(gs[1, 1])
    axb.imshow(seg_d[None, :], cmap=cmap, norm=norm, aspect="auto",
               extent=[0, t[-1], 0, 1], interpolation="nearest")
    axb.set_yticks([])
    axb.set_xlabel("Period")
    axb.grid(False)
    axb.text(0.012, 0.5, "dominant rule", transform=axb.transAxes,
             fontsize=8.2, va="center", ha="left", color="white",
             fontweight="bold")

    # --- (c) composition against deviation ---------------------------------
    ax = fig.add_subplot(gs[:, 2])
    edges = np.quantile(x, np.linspace(0, 1, 26))
    edges = np.unique(edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(centres) - 1)
    comp = np.vstack(
        [n[idx == k].mean(axis=0) if np.any(idx == k) else np.full(n.shape[1], np.nan)
         for k in range(len(centres))]
    )
    ax.stackplot(centres, comp.T, colors=res.colors, alpha=0.92, linewidth=0)
    ax.set_xlim(centres[0], centres[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"Deviation $x_t$")
    ax.set_ylabel("Average share")
    ax.set_title("Composition against deviation", fontsize=10.5)
    ax.grid(False)
    label_panel(ax, "(c)")

    handles = [
        Line2D([], [], marker="o", ls="", color=c, label=nm, markersize=7)
        for nm, c in zip(res.names, res.colors)
    ]
    fig.legend(handles, res.names, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.suptitle(
        "Bubbles are built by trend followers and ended by the traders "
        "betting against them",
        fontsize=13, fontweight="bold", y=1.01,
    )
    announce(save(fig, FIGDIR / "fig14_attractor_mechanism.png").name)


if __name__ == "__main__":
    main()
