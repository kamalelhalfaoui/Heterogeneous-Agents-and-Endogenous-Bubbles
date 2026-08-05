"""
Figure 3 -- What the fundamentalist anchor does to the price.

The six-type ecology is compared with the same ecology minus the
fundamentalists, at two intensities of choice. Each panel shows the deviation
from fundamental value together with 50- and 200-period trailing means, so that
both the short-run oscillation and the medium-run drift are visible.

The trailing means are computed causally (backward-looking only). A centred
moving average would let future information leak into the current value, which
is precisely the mistake that makes a mean-reverting series look better behaved
than it is.
"""

from _common import FIGDIR, TABDIR, announce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bh1998 import ABSParams, market, simulate
from bh1998.analysis import descriptive_stats
from bh1998.plotting import ACCENT, HILITE, MUTED, label_panel, save


def trailing_mean(z: np.ndarray, w: int) -> np.ndarray:
    """Backward-looking moving average; the first ``w-1`` entries are nan."""
    out = np.full_like(z, np.nan, dtype=float)
    if z.size >= w:
        c = np.cumsum(np.insert(z, 0, 0.0))
        out[w - 1 :] = (c[w:] - c[:-w]) / w
    return out


SCENARIOS = [
    (6, 2.0, "Six types, $\\beta=2$"),
    (5, 2.0, "Five types (no fundamentalists), $\\beta=2$"),
    (6, 20.0, "Six types, $\\beta=20$"),
    (5, 20.0, "Five types (no fundamentalists), $\\beta=20$"),
]


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 6.8), sharex=True)
    rows = []

    for k, (n_types, beta, title) in enumerate(SCENARIOS):
        ax = axes.flat[k]
        p = ABSParams(R=1.05, beta=beta, n_steps=12_000, burn_in=2_000)
        res = simulate(p, market(n_types)).trimmed()
        x = res.x

        ax.plot(x, color=ACCENT, lw=0.9, alpha=0.75, label=r"$x_t$",
                rasterized=True)
        ax.plot(trailing_mean(x, 25), color=HILITE, lw=1.6,
                label="25-period trailing mean")
        ax.plot(trailing_mean(x, 100), color="#0b6e4f", lw=1.8,
                label="100-period trailing mean")
        ax.axhline(0, color=MUTED, lw=0.9, ls=":")

        ax.set_xlim(0, 160)
        ax.set_ylim(-2.0, 2.0)
        ax.set_title(title, fontsize=10.5)
        label_panel(ax, f"({chr(97 + k)})")
        if k >= 2:
            ax.set_xlabel("Period $t$")
        if k % 2 == 0:
            ax.set_ylabel(r"Deviation $x_t$")

        st = descriptive_stats(x)
        ax.text(
            0.985, 0.045,
            f"$\\sigma_x$ = {st['Std. deviation']:.3f}\n"
            f"$|x|$ mean = {np.mean(np.abs(x)):.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.6),
        )

        rows.append(
            {
                "Types": n_types,
                "beta": beta,
                "Std. dev": st["Std. deviation"],
                "Mean |x|": float(np.mean(np.abs(x))),
                "Skewness": st["Skewness"],
                "Excess kurtosis": st["Excess kurtosis"],
            }
        )

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.035), frameon=False)
    fig.suptitle(
        "Removing the fundamentalist anchor: deviation from fundamental value",
        fontsize=13, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    announce(save(fig, FIGDIR / "fig03_fundamentalist_anchor.png").name)

    tab = pd.DataFrame(rows).round(4)
    tab.to_csv(TABDIR / "table_anchor_effect.csv", index=False)
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
