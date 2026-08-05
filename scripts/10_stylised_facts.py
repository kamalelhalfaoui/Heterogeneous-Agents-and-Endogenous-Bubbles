"""
Figure 10 -- Stylised facts, and what the model needs in order to produce them.

Real return series are close to unpredictable in their sign, heavy-tailed in
their distribution, and strongly persistent in their magnitude. The third
property, volatility clustering, is the hardest for a low-dimensional model to
generate, and it is the one that separates the two rows of this figure.

Top row: the baseline calibration, in which traders have no memory at all
(delta = 0, w = 0). The population re-sorts completely every period on the
basis of a single period of profit.

Bottom row: the same market with memory in both the fitness measure (w = 0.99)
and the population shares (delta = 0.9), so that rules are evaluated on
accumulated performance and traders are slow to abandon them.

The difference is not cosmetic. Without memory the model produces bubbles at
near-regular intervals, so the autocorrelation of absolute returns oscillates
around zero with sharp spikes at the dominant period -- that is periodicity, not
clustering, and it is the opposite of what the data show. With memory the
switching becomes intermittent: the market spends long stretches quiet and then
erupts, and the magnitude of returns becomes genuinely persistent.

The economic reading is that volatility clustering here is a consequence of
traders being slow to abandon a strategy. Instant reallocation destroys it.
"""

from _common import FIGDIR, TABDIR, announce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from bh1998 import calibration, simulate
from bh1998.analysis import stylised_facts
from bh1998.plotting import ACCENT, HILITE, MUTED, label_panel, save

ROWS = [
    ("stochastic", "No memory  ($\\delta = 0$, $w = 0$)"),
    ("clustering", "With memory  ($\\delta = 0.9$, $w = 0.99$)"),
]
MAX_LAG = 60


def main() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15.4, 7.6))
    records = {}

    for i, (key, row_label) in enumerate(ROWS):
        p, rules = calibration(key, n_steps=60_000, burn_in=6_000, seed=7)
        res = simulate(p, rules)
        sf = stylised_facts(res, max_lag=MAX_LAG)
        r = sf["returns"]
        records[row_label.split("  ")[0]] = sf

        colour = MUTED if i == 0 else ACCENT
        emph = "#9a9a9a" if i == 0 else HILITE

        # --- returns ------------------------------------------------------
        ax = axes[i, 0]
        ax.plot(r[:1200], color=colour, lw=0.55, rasterized=True)
        ax.axhline(0, color=MUTED, lw=0.7, ls=":")
        ax.set_ylabel(f"{row_label}\n\n" r"Log return $r_t$", fontsize=9.5)
        if i == 0:
            ax.set_title("Return series", fontsize=10.5)
        if i == 1:
            ax.set_xlabel("Period")
        label_panel(ax, f"({chr(97 + 4 * i)})", dx=-0.30)

        # --- distribution -------------------------------------------------
        ax = axes[i, 1]
        ax.hist(r, bins=140, density=True, color=colour, alpha=0.55,
                edgecolor="none")
        xs = np.linspace(np.quantile(r, 0.0005), np.quantile(r, 0.9995), 400)
        pdf = stats.norm.pdf(xs, r.mean(), r.std())
        # The fitted normal underflows many orders of magnitude below anything
        # the histogram can show; drawing it unclipped would stretch the log
        # axis down to 1e-25 and compress the actual data into a thin strip.
        floor = pdf.max() * 1e-6
        keep = pdf > floor
        ax.plot(xs[keep], pdf[keep], color=emph, lw=1.6)
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor)
        if i == 0:
            ax.set_title("Density vs normal (log scale)", fontsize=10.5)
        if i == 1:
            ax.set_xlabel(r"$r_t$")
        ax.text(
            0.03, 0.06,
            f"excess kurtosis {sf['excess_kurtosis']:.1f}\n"
            f"Hill index {sf['hill_index']:.2f}",
            transform=ax.transAxes, fontsize=8.6, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc",
                      lw=0.6, alpha=0.9),
        )
        label_panel(ax, f"({chr(98 + 4 * i)})")

        # --- ACF of returns -----------------------------------------------
        ci = sf["acf_ci"]
        lags = np.arange(1, MAX_LAG + 1)
        ax = axes[i, 2]
        ax.bar(lags, sf["acf_returns"], color=colour, width=0.85)
        ax.axhspan(-ci, ci, color="#b0b0b0", alpha=0.30, lw=0)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylim(-0.42, 0.42)
        if i == 0:
            ax.set_title("ACF of returns", fontsize=10.5)
        if i == 1:
            ax.set_xlabel("Lag")
        ax.text(
            0.97, 0.94,
            f"mean $|\\rho|$ = {np.mean(np.abs(sf['acf_returns'][:20])):.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.6,
        )
        label_panel(ax, f"({chr(99 + 4 * i)})")

        # --- ACF of absolute returns ---------------------------------------
        ax = axes[i, 3]
        ax.bar(lags, sf["acf_abs_returns"], color=emph, width=0.85)
        ax.axhspan(-ci, ci, color="#b0b0b0", alpha=0.30, lw=0)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylim(-0.42, 0.42)
        if i == 0:
            ax.set_title("ACF of absolute returns", fontsize=10.5)
        if i == 1:
            ax.set_xlabel("Lag")
        mean_abs = np.mean(sf["acf_abs_returns"][:20])
        ax.text(
            0.97, 0.94, f"mean $\\rho$ = {mean_abs:+.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.6,
            color=emph if mean_abs > 0.05 else "#555555",
            fontweight="bold" if mean_abs > 0.05 else "normal",
        )
        label_panel(ax, f"({chr(100 + 4 * i)})")

    fig.suptitle(
        "Volatility clustering requires memory: without it the model produces "
        "periodicity instead",
        fontsize=13.5, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    announce(save(fig, FIGDIR / "fig10_stylised_facts.png").name)

    tab = pd.DataFrame(
        {
            label: {
                "Excess kurtosis": sf["excess_kurtosis"],
                "Skewness": sf["skew"],
                "Jarque-Bera p": sf["jarque_bera_p"],
                "Hill index (5% tail)": sf["hill_index"],
                "Mean |ACF| of returns (lags 1-20)":
                    float(np.mean(np.abs(sf["acf_returns"][:20]))),
                "Mean ACF of |returns| (lags 1-20)":
                    float(np.mean(sf["acf_abs_returns"][:20])),
            }
            for label, sf in records.items()
        }
    ).round(4)
    tab.to_csv(TABDIR / "table_stylised_facts.csv")
    print(tab.to_string())


if __name__ == "__main__":
    main()
