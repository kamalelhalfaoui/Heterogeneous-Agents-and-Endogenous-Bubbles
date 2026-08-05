"""
Figure 4 -- Evolutionary dynamics of the population shares.

Top row: the share of each forecasting rule over time. Bottom row: the same
information as a stacked area, which makes the total composition of the market
legible at a glance.

The shares are extremely volatile because the baseline sets delta = 0 and
w = 0: nobody is locked into a rule and fitness has no memory, so the entire
population re-sorts itself every single period on the basis of one period of
profit. Figure 11 relaxes both assumptions.
"""

from _common import FIGDIR, announce

import matplotlib.pyplot as plt
import numpy as np

from bh1998 import ABSParams, market, simulate
from bh1998.plotting import MUTED, label_panel, save

SCENARIOS = [
    (6, 2.0, r"Six types, $\beta=2$"),
    (5, 2.0, r"Five types, $\beta=2$"),
    (6, 20.0, r"Six types, $\beta=20$"),
    (5, 20.0, r"Five types, $\beta=20$"),
]

WINDOW = 60


def main() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 6.6),
                             gridspec_kw={"height_ratios": [1.0, 1.0]})

    legend_handles: dict[str, object] = {}

    for k, (n_types, beta, title) in enumerate(SCENARIOS):
        p = ABSParams(R=1.05, beta=beta, n_steps=8_000, burn_in=2_000)
        res = simulate(p, market(n_types)).trimmed()
        n = res.n[:WINDOW]
        t = np.arange(WINDOW)

        # --- line view ---------------------------------------------------
        ax = axes[0, k]
        for j, (name, colour) in enumerate(zip(res.names, res.colors)):
            (line,) = ax.plot(t, n[:, j], color=colour, lw=1.4)
            legend_handles.setdefault(name, line)
        ax.axhline(1.0 / n_types, color=MUTED, lw=0.8, ls=":")
        ax.set_ylim(0, 1.0)
        ax.set_xlim(0, WINDOW - 1)
        ax.set_title(title, fontsize=10.5)
        label_panel(ax, f"({chr(97 + k)})")
        if k == 0:
            ax.set_ylabel(r"Share $n_{h,t}$")

        # --- stacked view --------------------------------------------------
        ax = axes[1, k]
        ax.stackplot(t, n.T, colors=res.colors, alpha=0.92, linewidth=0)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, WINDOW - 1)
        ax.set_xlabel("Period $t$")
        ax.grid(False)
        label_panel(ax, f"({chr(101 + k)})")
        if k == 0:
            ax.set_ylabel("Cumulative share")

    fig.legend(
        list(legend_handles.values()), list(legend_handles.keys()),
        loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.035), frameon=False,
    )
    fig.suptitle(
        "Who is in the market: population shares under evolutionary selection",
        fontsize=13, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    announce(save(fig, FIGDIR / "fig04_fraction_dynamics.png").name)


if __name__ == "__main__":
    main()
