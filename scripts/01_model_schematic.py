"""
Figure 1 -- Structure of the adaptive belief system.

Draws the timing loop of the model: beliefs are formed from past deviations,
fractions are updated from past fitness, the market clears, profits are
realised, and fitness feeds back into the next period's fractions. The feedback
edge from profit back to fractions is what makes the system nonlinear.
"""

from _common import FIGDIR, announce

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from bh1998.plotting import ACCENT, INK, MUTED, HILITE, save

BOXES = {
    "beliefs": (0.5, 3.05, "Belief formation",
                r"$f_{h,t}=\alpha_h+\varphi_h x_{t-1}+\gamma_h\Delta x_{t-1}$"
                "\n" r"$\qquad+\theta_h f_{h,t-1}+\xi_h\bar{a}_{t-1}$"),
    "fractions": (0.5, 1.15, "Evolutionary selection",
                  r"$n_{h,t}=\delta n_{h,t-1}+(1-\delta)\dfrac{e^{\beta U_{h,t-1}}}"
                  r"{\sum_k e^{\beta U_{k,t-1}}}$"),
    "market": (5.35, 3.05, "Market clearing",
               r"$R\,x_t=\sum_{h}n_{h,t}f_{h,t}$"),
    "fitness": (5.35, 1.15, "Realised fitness",
                r"$\pi_{h,t}=\dfrac{(x_t-Rx_{t-1})(f_{h,t-1}-Rx_{t-1})}{a\sigma^2}-C_h$"
                "\n" r"$U_{h,t}=\pi_{h,t}+w\,U_{h,t-1}$"),
}

W, H = 4.05, 1.35


def add_box(ax, x, y, title, body, edge):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), W, H,
            boxstyle="round,pad=0.06,rounding_size=0.14",
            linewidth=1.4, edgecolor=edge, facecolor="white", zorder=3,
        )
    )
    ax.text(x + W / 2, y + H - 0.28, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=edge, zorder=4)
    ax.text(x + W / 2, y + H / 2 - 0.30, body, ha="center", va="center",
            fontsize=9.6, color=INK, zorder=4)


def arrow(ax, p0, p1, color, rad=0.0, style="-|>", lw=1.5, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle=style, mutation_scale=15,
            connectionstyle=f"arc3,rad={rad}",
            linewidth=lw, linestyle=ls, color=color, zorder=2,
            shrinkA=2, shrinkB=2,
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0.40, 5.45)
    ax.axis("off")

    add_box(ax, *BOXES["beliefs"], ACCENT)
    add_box(ax, *BOXES["market"], ACCENT)
    add_box(ax, *BOXES["fractions"], HILITE)
    add_box(ax, *BOXES["fitness"], HILITE)

    # beliefs -> market
    arrow(ax, (4.65, 3.72), (5.25, 3.72), ACCENT)
    ax.text(4.95, 3.90, r"$f_{h,t}$", ha="center", fontsize=9.5, color=ACCENT)

    # fractions -> market  (up the left, across)
    arrow(ax, (4.62, 2.05), (5.28, 3.16), ACCENT, rad=-0.30)
    ax.text(4.62, 2.74, r"$n_{h,t}$", ha="right", fontsize=9.5, color=ACCENT)

    # market -> fitness
    arrow(ax, (7.38, 2.99), (7.38, 2.60), HILITE)
    ax.text(7.55, 2.79, r"$x_t$", ha="left", fontsize=9.5, color=HILITE)

    # fitness -> fractions  (the feedback loop)
    arrow(ax, (5.25, 1.82), (4.65, 1.82), HILITE)
    ax.text(4.95, 2.00, r"$U_{h,t}$", ha="center", fontsize=9.5, color=HILITE)

    # market -> beliefs (one-period lag, dashed)
    ax.plot([7.38, 7.38, 2.55], [4.42, 4.92, 4.92],
            ls=(0, (5, 3)), color=MUTED, lw=1.5, zorder=2, solid_capstyle="butt")
    arrow(ax, (2.55, 4.92), (2.55, 4.46), MUTED, ls=(0, (5, 3)))
    ax.text(4.95, 5.14, r"one-period information lag:  $x_{t}\rightarrow x_{t-1}$",
            ha="center", fontsize=9.5, color="#666666")

    ax.text(0.5, 0.55,
            "Blue: price formation within the period.    "
            "Red: the evolutionary feedback that makes the system nonlinear.",
            fontsize=9, color="#666666")

    fig.suptitle("The adaptive belief system, one period",
                 fontsize=13, fontweight="bold", y=0.97)
    announce(save(fig, FIGDIR / "fig01_model_schematic.png").name)


if __name__ == "__main__":
    main()
