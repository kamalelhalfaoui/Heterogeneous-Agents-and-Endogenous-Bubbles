"""
Figure 5 -- Phase plots: population share against price deviation.

Each panel plots the share of one forecasting rule against the contemporaneous
deviation from fundamental value. Because the model is deterministic here,
these are not scatter clouds in the statistical sense: they are projections of
the attractor onto the plane (x_t, n_{h,t}), and their shape is a signature of
the rule.

The chaotic calibration is used here deliberately. Under the benchmark cycle of
Figure 2 the attractor consists of six points, so every panel below would
collapse to six dots: there is nothing to see because there is nothing going on.
Structure in a phase plot requires an attractor with structure.

Reading them: a rule whose share peaks where the deviation is large and
positive is one that profits from bubbles; a rule whose share peaks near zero
deviation profits from mean reversion. The vertical spread at a given x is the
memory of the system -- the same price can be reached from different histories,
and the shares differ accordingly.
"""

from _common import FIGDIR, announce

import matplotlib.pyplot as plt
import numpy as np

from bh1998 import calibration, simulate
from bh1998.plotting import MUTED, label_panel, save


def panel(ax, x, n, name, colour, letter):
    ax.plot(x, n, ".", ms=1.6, alpha=0.30, color=colour, rasterized=True)
    ax.axvline(0, color=MUTED, lw=0.7, ls=":")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(name, fontsize=10.5, color=colour)
    label_panel(ax, letter)


def main() -> None:
    for n_types, tag in ((6, "six"), (5, "five")):
        p, rules = calibration(
            "chaotic", n_steps=20_000, burn_in=2_000, n_types=n_types
        )
        res = simulate(p, rules).trimmed()

        ncols = 3
        nrows = int(np.ceil(n_types / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(11.4, 3.4 * nrows),
                                 sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()

        for j, (name, colour) in enumerate(zip(res.names, res.colors)):
            panel(axes[j], res.x, res.n[:, j], name, colour, f"({chr(97 + j)})")

        for j in range(n_types, len(axes)):
            axes[j].set_visible(False)

        for j, ax in enumerate(axes[:n_types]):
            if j // ncols == nrows - 1 or j + ncols >= n_types:
                ax.set_xlabel(r"Deviation $x_t$")
            if j % ncols == 0:
                ax.set_ylabel(r"Share $n_{h,t}$")

        fig.suptitle(
            f"Phase plots of the attractor, {tag} trader types "
            r"($\beta=10$, $\varphi=1.0$, $R=1.05$)",
            fontsize=13, fontweight="bold", y=1.0,
        )
        fig.tight_layout()
        announce(save(fig, FIGDIR / f"fig05_phase_plots_{tag}.png").name)


if __name__ == "__main__":
    main()
