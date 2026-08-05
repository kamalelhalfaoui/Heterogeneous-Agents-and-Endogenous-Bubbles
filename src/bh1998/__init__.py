"""
bh1998
======

A research-grade implementation of the Brock & Hommes (1998) adaptive belief
system, with an evolutionary ecology of heterogeneous forecasting rules.

Quick start
-----------
>>> from bh1998 import ABSParams, simulate, market
>>> res = simulate(ABSParams(beta=2.0, R=1.05), market(6))
>>> res.mean_fraction().round(3)

Author: Kamal El Halfaoui
"""

from .agents import (
    BeliefRule,
    FIVE_TYPE_MARKET,
    SIX_TYPE_MARKET,
    adapter,
    fundamentalist,
    learner,
    market,
    optimist,
    pessimist,
    trend_follower,
)
from .model import ABSParams, ABSResult, fundamental_price, simulate
from .analysis import (
    autocorrelation,
    bifurcation,
    descriptive_stats,
    detect_period,
    hill_estimator,
    lyapunov_rosenstein,
    lyapunov_scan,
    parameter_sweep_2d,
    stylised_facts,
    survival_table,
)
from .plotting import use_style, save

__version__ = "1.0.0"
__author__ = "Kamal El Halfaoui"

# --------------------------------------------------------------------------
# Named calibrations used throughout the figures and the documentation.
# Each one sits in a qualitatively different region of parameter space.
# --------------------------------------------------------------------------

CALIBRATIONS: dict[str, dict] = {
    "benchmark": dict(
        params=dict(R=1.05, beta=2.0, delta=0.0, w=0.0),
        rules=dict(alpha_opt=1.5, phi_trend=0.2),
        note="Low intensity of choice: the market settles on a stable cycle.",
    ),
    "quasiperiodic": dict(
        params=dict(R=1.05, beta=20.0, delta=0.0, w=0.0),
        rules=dict(alpha_opt=1.5, phi_trend=0.2),
        note="High intensity of choice: motion on a torus, no repeating period.",
    ),
    "chaotic": dict(
        params=dict(R=1.05, beta=10.0, delta=0.0, w=0.0),
        rules=dict(alpha_opt=0.3, phi_trend=1.0),
        note="Weak biases, strong extrapolation: positive Lyapunov exponent.",
    ),
    "stochastic": dict(
        params=dict(R=1.05, beta=10.0, delta=0.0, w=0.0, noise=0.02),
        rules=dict(alpha_opt=0.3, phi_trend=1.0),
        note="The chaotic skeleton plus small IID shocks; used for stylised facts.",
    ),
    "clustering": dict(
        params=dict(R=1.05, beta=10.0, delta=0.9, w=0.99, noise=0.1),
        rules=dict(alpha_opt=0.3, phi_trend=1.05),
        note=(
            "Memory in both fitness (w) and fractions (delta). This is the only "
            "region found in which the model reproduces volatility clustering: "
            "without memory the magnitude of returns is periodic rather than "
            "clustered."
        ),
    ),
}


def calibration(name: str, **overrides):
    """Build ``(ABSParams, rules)`` for one of the named calibrations.

    Parameters
    ----------
    name : str
        One of the keys of :data:`CALIBRATIONS`.
    **overrides
        Applied to :class:`ABSParams` (e.g. ``n_steps=20_000``).
    """
    if name not in CALIBRATIONS:
        raise KeyError(
            f"Unknown calibration {name!r}. Choose from {list(CALIBRATIONS)}."
        )
    spec = CALIBRATIONS[name]
    n_types = overrides.pop("n_types", 6)
    p = ABSParams(**{**spec["params"], **overrides})
    r = market(n_types, **spec["rules"])
    return p, r


__all__ = [
    "ABSParams", "ABSResult", "simulate", "fundamental_price",
    "BeliefRule", "market", "fundamentalist", "trend_follower", "adapter",
    "learner", "optimist", "pessimist", "SIX_TYPE_MARKET", "FIVE_TYPE_MARKET",
    "descriptive_stats", "bifurcation", "lyapunov_rosenstein", "lyapunov_scan",
    "autocorrelation", "hill_estimator", "stylised_facts", "detect_period",
    "parameter_sweep_2d", "survival_table",
    "use_style", "save", "CALIBRATIONS", "calibration",
    "__version__", "__author__",
]
