"""
model.py
========

Simulation kernel for the Brock & Hommes (1998) adaptive belief system (ABS).

The economy contains one risk-free asset paying gross return ``R = 1 + r`` and
one risky asset in zero net supply.  Traders are myopic mean-variance
maximisers, so type ``h`` demands

    z_{h,t} = E_{h,t}[p_{t+1} + y_{t+1} - R p_t] / (a * sigma^2)

Market clearing at zero outside supply, together with the fundamental price
``p* = ybar / (R - 1)`` and the deviation ``x_t = p_t - p*``, collapses the whole
system into a scalar law of motion

    R x_t = sum_h n_{h,t} f_{h,t}                                     (pricing)

    pi_{h,t} = (x_t - R x_{t-1}) (f_{h,t-1} - R x_{t-1}) / (a sigma^2) - C_h

    U_{h,t} = pi_{h,t} + w U_{h,t-1}                                  (memory)

    n_{h,t} = delta n_{h,t-1} + (1 - delta) softmax(beta U_{h,t-1})_h  (selection)

Two implementation points matter and are handled here but are absent from naive
transcriptions of the model:

1.  The discrete-choice weights are computed with a **log-sum-exp shift**.  For
    ``beta`` above roughly 5 the raw ``exp(beta * U)`` overflows to ``inf`` and
    the fractions become ``nan``; the shifted form is exact and never overflows,
    which is what makes the bifurcation diagram in ``beta`` computable at all.

2.  The timing is strictly **causal**.  ``f_{h,t}`` is built only from
    information dated ``t-1`` and earlier, and the profit that drives selection
    at ``t`` is the profit earned by the position taken at ``t-1``.  This is the
    one-period information lag that generates the model's endogenous dynamics.

Author: Kamal El Halfaoui
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .agents import BeliefRule, stack_rules, market

__all__ = ["ABSParams", "ABSResult", "simulate", "fundamental_price"]

# Any |x| beyond this is treated as a divergent bubble rather than a number.
_EXPLOSION_CAP = 1e8


# ==========================================================================
# Parameters
# ==========================================================================


@dataclass
class ABSParams:
    """Structural and numerical parameters of the adaptive belief system.

    Attributes
    ----------
    R : float
        Gross risk-free return ``1 + r``. Must exceed 1.
    beta : float
        Intensity of choice. ``beta = 0`` freezes the fractions at uniform;
        ``beta -> inf`` puts all mass on last period's best performer.
    delta : float
        Fraction of agents who do not revise ("memory in the fractions"),
        in ``[0, 1)``.
    w : float
        Geometric memory in the fitness measure, in ``[0, 1)``.
    a : float
        Coefficient of absolute risk aversion.
    sigma : float
        Conditional standard deviation of excess returns used in demands.
    ybar : float
        Mean dividend. Only used to place the fundamental price level.
    noise : float
        Standard deviation of an IID shock added to the deviation each period.
        ``0.0`` gives the deterministic skeleton; a small positive value gives
        the stochastic model used for the stylised-facts analysis.
    seed : int or None
        Seed for the shock process.
    n_steps : int
        Number of periods simulated.
    burn_in : int or None
        Periods discarded from the *analysis* helpers (the raw arrays are
        always returned in full). When left as ``None`` it is set to whichever
        is smaller of 500 periods and a fifth of the run, so that short
        exploratory runs do not have to specify it explicitly.
    x0 : float
        Initial deviation used to seed the two required lags.
    """

    R: float = 1.05
    beta: float = 2.0
    delta: float = 0.0
    w: float = 0.0
    a: float = 1.0
    sigma: float = 1.0
    ybar: float = 1.0
    noise: float = 0.0
    seed: int | None = 0
    n_steps: int = 10_000
    burn_in: int | None = None
    x0: float = 0.22

    def __post_init__(self) -> None:
        if self.R <= 1.0:
            raise ValueError("R must be strictly greater than 1.")
        if not 0.0 <= self.delta < 1.0:
            raise ValueError("delta must lie in [0, 1).")
        if not 0.0 <= self.w < 1.0:
            raise ValueError("w must lie in [0, 1).")
        if self.n_steps < 10:
            raise ValueError("n_steps must be at least 10.")
        if self.burn_in is None:
            self.burn_in = min(500, self.n_steps // 5)
        if not 0 <= self.burn_in < self.n_steps:
            raise ValueError("burn_in must lie in [0, n_steps).")


def fundamental_price(params: ABSParams) -> float:
    """Fundamental price ``p* = ybar / (R - 1)`` implied by the parameters."""
    return params.ybar / (params.R - 1.0)


# ==========================================================================
# Result container
# ==========================================================================


@dataclass
class ABSResult:
    """Everything a run produces, plus derived quantities computed on demand."""

    x: np.ndarray            # (T,)      deviation from fundamental
    n: np.ndarray            # (T, H)    population fractions
    f: np.ndarray            # (T, H)    forecasts
    profit: np.ndarray       # (T, H)    realised per-period profits
    fitness: np.ndarray      # (T, H)    accumulated fitness U
    xbar: np.ndarray         # (T,)      running sample mean of x
    params: ABSParams
    rules: tuple[BeliefRule, ...]
    diverged_at: int | None = None   # first period where |x| exceeded the cap

    # -- convenience -------------------------------------------------------

    @property
    def names(self) -> list[str]:
        return [r.name for r in self.rules]

    @property
    def colors(self) -> list[str]:
        return [r.color for r in self.rules]

    @property
    def n_types(self) -> int:
        return len(self.rules)

    @property
    def price(self) -> np.ndarray:
        """Level of the price, ``p_t = p* + x_t``."""
        return fundamental_price(self.params) + self.x

    def returns(self, log: bool = True) -> np.ndarray:
        """Period returns on the *price level*.

        Computing returns on ``x`` directly (as ``(x_t - x_{t-1}) / x_{t-1}``)
        is numerically meaningless because ``x`` crosses zero repeatedly and the
        ratio explodes. Returns are therefore always taken on ``p = p* + x``,
        which stays bounded away from zero for the calibrations used here.
        """
        p = self.price
        if np.any(p <= 0):
            raise ValueError(
                "Price hit a non-positive level; raise ybar or lower the bias "
                "parameters before computing returns."
            )
        if log:
            return np.diff(np.log(p))
        return np.diff(p) / p[:-1]

    def trimmed(self) -> "ABSResult":
        """Drop the burn-in sample from every time-indexed array."""
        b = self.params.burn_in
        return ABSResult(
            x=self.x[b:],
            n=self.n[b:],
            f=self.f[b:],
            profit=self.profit[b:],
            fitness=self.fitness[b:],
            xbar=self.xbar[b:],
            params=self.params,
            rules=self.rules,
            diverged_at=self.diverged_at,
        )

    @property
    def diverged(self) -> bool:
        """True if the trajectory left the admissible region before the end."""
        return self.diverged_at is not None

    def mean_profit(self) -> np.ndarray:
        """Post-burn-in average profit per type, shape (H,)."""
        return self.trimmed().profit.mean(axis=0)

    def mean_fraction(self) -> np.ndarray:
        """Post-burn-in average population share per type, shape (H,)."""
        return self.trimmed().n.mean(axis=0)

    def running_mean_profit(self) -> np.ndarray:
        """Expanding-window mean profit, shape (T, H). Used for Figure 5."""
        cs = np.cumsum(self.profit, axis=0)
        denom = np.arange(1, self.profit.shape[0] + 1)[:, None]
        return cs / denom


# ==========================================================================
# Numerics
# ==========================================================================


def _softmax(u: np.ndarray, beta: float) -> np.ndarray:
    """Overflow-safe discrete-choice weights ``exp(beta u_h) / sum exp(beta u)``.

    Subtracting the maximum before exponentiating leaves the ratio unchanged
    but bounds every exponent above by zero. Without this the model cannot be
    simulated for the large ``beta`` values where its most interesting dynamics
    live.
    """
    z = beta * u
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


# ==========================================================================
# Kernel
# ==========================================================================


def simulate(
    params: ABSParams | None = None,
    rules: Sequence[BeliefRule] | None = None,
    n_init: np.ndarray | None = None,
) -> ABSResult:
    """Run the adaptive belief system forward.

    Parameters
    ----------
    params : ABSParams, optional
        Structural parameters. Defaults to the baseline calibration.
    rules : sequence of BeliefRule, optional
        The trader ecology. Defaults to the six-type market.
    n_init : ndarray, optional
        Initial population shares. Defaults to uniform. Must sum to one.

    Returns
    -------
    ABSResult
    """
    params = params or ABSParams()
    rules = tuple(rules) if rules is not None else market(6)
    H = len(rules)
    T = params.n_steps

    loadings, costs = stack_rules(rules)          # (H, 5), (H,)
    alpha, phi, gamma, theta, xi = (loadings[:, j] for j in range(5))

    R, beta, delta, w = params.R, params.beta, params.delta, params.w
    scale = params.a * params.sigma ** 2

    if n_init is None:
        n_init = np.full(H, 1.0 / H)
    n_init = np.asarray(n_init, dtype=float)
    if n_init.shape != (H,):
        raise ValueError(f"n_init must have shape ({H},).")
    if not np.isclose(n_init.sum(), 1.0):
        raise ValueError("n_init must sum to one.")

    rng = np.random.default_rng(params.seed)
    shocks = (
        rng.normal(0.0, params.noise, size=T) if params.noise > 0 else np.zeros(T)
    )

    # ---- allocate -------------------------------------------------------
    x = np.zeros(T)
    xbar = np.zeros(T)
    n = np.zeros((T, H))
    f = np.zeros((T, H))
    profit = np.zeros((T, H))
    fitness = np.zeros((T, H))

    # ---- seed the two required lags -------------------------------------
    x[0] = params.x0
    x[1] = params.x0 * 1.05
    xbar[0] = x[0]
    xbar[1] = 0.5 * (x[0] + x[1])
    n[0] = n_init
    n[1] = n_init
    # A forecast rule with inertia needs f_{-1}; anchoring it on the initial
    # deviation avoids an artificial transient in the adapter series.
    f[0] = alpha + (phi + xi) * x[0]
    f[1] = alpha + phi * x[0] + theta * f[0] + xi * x[0]

    run_sum = x[0] + x[1]
    diverged_at: int | None = None

    # ---- main recursion --------------------------------------------------
    for t in range(2, T):
        # (1) forecasts, using information dated t-1 and earlier only
        f[t] = (
            alpha
            + phi * x[t - 1]
            + gamma * (x[t - 1] - x[t - 2])
            + theta * f[t - 1]
            + xi * 0.5 * (xbar[t - 1] + x[t - 1])
        )

        # (2) evolutionary selection on last period's fitness
        n[t] = delta * n[t - 1] + (1.0 - delta) * _softmax(fitness[t - 1], beta)

        # (3) market clearing
        x[t] = float(n[t] @ f[t]) / R + shocks[t]

        # (4) realised profit on the position taken at t-1
        excess = x[t] - R * x[t - 1]
        profit[t] = excess * (f[t - 1] - R * x[t - 1]) / scale - costs

        # (5) fitness with geometric memory
        fitness[t] = profit[t] + w * fitness[t - 1]

        # (6) update the running anchor
        run_sum += x[t]
        xbar[t] = run_sum / (t + 1)

        # (7) explosion guard.
        # When trend extrapolation is strong (phi > R) and the stabilising
        # fundamentalist share collapses, the deviation can grow without bound.
        # That is a genuine property of the model, not a coding error, but it
        # must be caught: left alone it overflows to inf and silently poisons
        # every downstream statistic with nan. The run is truncated and flagged
        # instead, and the arrays are held at their last finite value.
        if not np.isfinite(x[t]) or abs(x[t]) > _EXPLOSION_CAP:
            diverged_at = t
            x[t:] = x[t - 1]
            n[t:] = n[t - 1]
            f[t:] = f[t - 1]
            profit[t:] = 0.0
            fitness[t:] = fitness[t - 1]
            xbar[t:] = xbar[t - 1]
            break

    return ABSResult(
        x=x, n=n, f=f, profit=profit, fitness=fitness, xbar=xbar,
        params=params, rules=rules, diverged_at=diverged_at,
    )
