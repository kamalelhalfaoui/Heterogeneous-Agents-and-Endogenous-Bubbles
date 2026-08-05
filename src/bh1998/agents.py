"""
agents.py
=========

Belief rules for the Brock & Hommes (1998) adaptive-belief system.

Every trader type in this project is a *parameter restriction* of one single
linear predictor rule:

    f_{h,t} = a_h
            + p_h * x_{t-1}
            + g_h * (x_{t-1} - x_{t-2})
            + t_h * f_{h,t-1}
            + k_h * (xbar_{t-1} + x_{t-1}) / 2

with

    a_h   (alpha)   constant bias         -- optimism / pessimism
    p_h   (phi)     trend extrapolation   -- chartism
    g_h   (gamma)   momentum on changes   -- learning / adjusting
    t_h   (theta)   own-forecast inertia  -- adaptive expectations
    k_h   (xi)      anchoring on history  -- anchor-and-adjust
    xbar  is the running sample mean of the deviation series.

Writing all types as restrictions of one rule means the simulation kernel never
branches on agent identity: it is a single matrix operation per period, which is
what makes the sweeps and bifurcation diagrams in this repository cheap.

Author: Kamal El Halfaoui
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np

__all__ = [
    "BeliefRule",
    "fundamentalist",
    "trend_follower",
    "adapter",
    "learner",
    "optimist",
    "pessimist",
    "SIX_TYPE_MARKET",
    "FIVE_TYPE_MARKET",
    "market",
    "stack_rules",
]


@dataclass(frozen=True)
class BeliefRule:
    """A single heterogeneous belief type.

    Parameters
    ----------
    name : str
        Human readable label used in figures and tables.
    alpha : float
        Constant bias in the forecast of the deviation from fundamental.
    phi : float
        Loading on the last observed deviation (pure trend extrapolation).
    gamma : float
        Loading on the last observed *change* in the deviation (momentum).
    theta : float
        Loading on the type's own previous forecast (adaptive inertia).
    xi : float
        Loading on the anchor term ``(xbar_{t-1} + x_{t-1}) / 2``.
    cost : float
        Per-period information cost ``C_h`` subtracted from realised profit.
    color : str
        Matplotlib colour used consistently across every figure.
    """

    name: str
    alpha: float = 0.0
    phi: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    xi: float = 0.0
    cost: float = 0.0
    color: str = "#333333"

    def as_vector(self) -> np.ndarray:
        """Return the five belief loadings as a flat array."""
        return np.array(
            [self.alpha, self.phi, self.gamma, self.theta, self.xi], dtype=float
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def replace(self, **kwargs) -> "BeliefRule":
        """Return a copy with selected fields overridden (for sweeps)."""
        d = asdict(self)
        d.update(kwargs)
        return BeliefRule(**d)


# --------------------------------------------------------------------------
# The six canonical archetypes
# --------------------------------------------------------------------------
# Colours follow a single perceptually ordered palette so that the same trader
# type is always the same colour in every figure of the repository.

_C_FUND = "#1b3a6b"   # deep navy   - fundamentalists
_C_TREND = "#c0392b"  # brick red   - trend followers
_C_ADAPT = "#1e8449"  # green       - adapters
_C_LEARN = "#b7791f"  # ochre       - learners
_C_OPT = "#6c3483"    # purple      - optimists
_C_PESS = "#7f8c8d"   # grey        - pessimists


def fundamentalist(cost: float = 0.0) -> BeliefRule:
    """f = 0. Believes the price returns to fundamental value.

    This is the only type that pays an information cost in the baseline
    calibration: knowing the fundamental value is not free.
    """
    return BeliefRule("Fundamentalists", cost=cost, color=_C_FUND)


def trend_follower(alpha: float = 0.8, phi: float = 0.2) -> BeliefRule:
    """f = alpha + phi * x_{t-1}. Pure chartist with a constant bias."""
    return BeliefRule("Trend followers", alpha=alpha, phi=phi, color=_C_TREND)


def adapter(phi: float = 0.8) -> BeliefRule:
    """f = phi * x_{t-1} + (1 - phi) * f_{t-1}: adaptive expectations.

    The restriction ``phi + theta = 1`` makes the rule an exponentially weighted
    moving average of past deviations, so the forecast is unbiased in the long
    run whenever the deviation series is stationary.
    """
    return BeliefRule("Adapters", phi=phi, theta=1.0 - phi, color=_C_ADAPT)


def learner(gamma: float = 1.0, xi: float = 0.1) -> BeliefRule:
    """f = gamma * (x_{t-1} - x_{t-2}) + xi * (xbar_{t-1} + x_{t-1}) / 2.

    Reacts to the most recent change and anchors on the historical average --
    the "anchor and adjust" rule of Hommes (2013).
    """
    return BeliefRule("Learners", gamma=gamma, xi=xi, color=_C_LEARN)


def optimist(alpha: float = 1.5) -> BeliefRule:
    """f = alpha > 0. Permanent upward bias."""
    return BeliefRule("Optimists", alpha=alpha, color=_C_OPT)


def pessimist(alpha: float = -1.5) -> BeliefRule:
    """f = alpha < 0. Permanent downward bias."""
    return BeliefRule("Pessimists", alpha=alpha, color=_C_PESS)


# --------------------------------------------------------------------------
# Standard market compositions
# --------------------------------------------------------------------------

SIX_TYPE_MARKET: tuple[BeliefRule, ...] = (
    fundamentalist(cost=0.05),
    trend_follower(),
    adapter(),
    learner(),
    optimist(),
    pessimist(),
)
"""Full ecology: the fundamentalist anchor plus five boundedly rational types."""

FIVE_TYPE_MARKET: tuple[BeliefRule, ...] = (
    trend_follower(),
    adapter(),
    learner(),
    optimist(),
    pessimist(),
)
"""Counterfactual ecology with the fundamentalist anchor removed."""


def market(n_types: int = 6, **overrides) -> tuple[BeliefRule, ...]:
    """Convenience constructor for the two standard ecologies.

    Parameters
    ----------
    n_types : {5, 6}
        6 includes fundamentalists, 5 removes them.
    **overrides
        Passed through to the individual constructors, e.g.
        ``market(6, alpha_opt=3.0, phi_trend=1.1)``.
    """
    a_opt = overrides.get("alpha_opt", 1.5)
    a_pess = overrides.get("alpha_pess", -a_opt)
    a_tr = overrides.get("alpha_trend", 0.8)
    p_tr = overrides.get("phi_trend", 0.2)
    p_ad = overrides.get("phi_adapt", 0.8)
    g_le = overrides.get("gamma_learn", 1.0)
    x_le = overrides.get("xi_learn", 0.1)
    c_fund = overrides.get("cost_fund", 0.05)

    core = (
        trend_follower(a_tr, p_tr),
        adapter(p_ad),
        learner(g_le, x_le),
        optimist(a_opt),
        pessimist(a_pess),
    )
    if n_types == 6:
        return (fundamentalist(c_fund),) + core
    if n_types == 5:
        return core
    raise ValueError("n_types must be 5 or 6; build a custom tuple otherwise.")


def stack_rules(rules: Sequence[BeliefRule]) -> tuple[np.ndarray, np.ndarray]:
    """Pack a sequence of rules into the arrays consumed by the kernel.

    Returns
    -------
    loadings : ndarray, shape (H, 5)
        Columns are ``[alpha, phi, gamma, theta, xi]``.
    costs : ndarray, shape (H,)
    """
    loadings = np.vstack([r.as_vector() for r in rules])
    costs = np.array([r.cost for r in rules], dtype=float)
    return loadings, costs
