"""
analysis.py
===========

Diagnostics applied to simulated adaptive-belief-system paths.

Contents
--------
descriptive_stats      moments of the deviation series
bifurcation            attractor of x as a control parameter is varied
lyapunov_rosenstein    largest Lyapunov exponent from a delay embedding
lyapunov_scan          largest Lyapunov exponent along a parameter grid
autocorrelation        sample ACF with Bartlett bands
hill_estimator         tail index of the return distribution
stylised_facts         the standard battery of financial stylised facts
parameter_sweep_2d     volatility / kurtosis surfaces over a parameter grid
survival_table         long-run ecology of strategies

Author: Kamal El Halfaoui
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

from .agents import BeliefRule, market
from .model import ABSParams, ABSResult, simulate

__all__ = [
    "descriptive_stats",
    "bifurcation",
    "lyapunov_rosenstein",
    "lyapunov_scan",
    "autocorrelation",
    "hill_estimator",
    "stylised_facts",
    "parameter_sweep_2d",
    "survival_table",
]


# ==========================================================================
# Moments
# ==========================================================================


def descriptive_stats(x: np.ndarray, label: str = "x") -> pd.Series:
    """Mean, median, dispersion and shape of a series.

    Kurtosis is reported in *excess* form (normal = 0), which is the convention
    used throughout this repository.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return pd.Series(
        {
            "Mean": np.mean(x),
            "Median": np.median(x),
            "Std. deviation": np.std(x, ddof=1),
            "Skewness": stats.skew(x),
            "Excess kurtosis": stats.kurtosis(x, fisher=True),
            "Min": np.min(x),
            "Max": np.max(x),
        },
        name=label,
    )


# ==========================================================================
# Bifurcation
# ==========================================================================


def bifurcation(
    values: Sequence[float],
    param: str = "beta",
    rules: Sequence[BeliefRule] | None = None,
    base: ABSParams | None = None,
    n_steps: int = 3_000,
    burn_in: int = 2_000,
    keep: int = 200,
    rule_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace the attractor of ``x`` as one parameter is varied.

    For each value the model is run, the transient is discarded and the last
    ``keep`` points of the trajectory are recorded. A fixed point contributes a
    single dot, a period-``k`` cycle contributes ``k`` dots, and a chaotic or
    quasi-periodic attractor contributes a dense vertical band.

    Parameters
    ----------
    values : sequence of float
        Grid for the control parameter.
    param : str
        Either the name of a field of :class:`ABSParams`, or a key accepted by
        :func:`bh1998.agents.market` (e.g. ``"phi_trend"``), in which case the
        ecology is rebuilt at each grid point.
    rules : sequence of BeliefRule, optional
        Fixed ecology. Ignored when ``param`` refers to a belief loading.
    rule_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`market` when rebuilding.

    Returns
    -------
    xs : ndarray, shape (len(values) * keep,)
        Parameter value repeated for every retained point.
    ys : ndarray, shape (len(values) * keep,)
        Retained values of the deviation.
    """
    base = base or ABSParams()
    rule_kwargs = dict(rule_kwargs or {})
    is_structural = hasattr(base, param)
    n_types = len(rules) if rules is not None else 6
    if rules is None:
        rules = market(n_types, **rule_kwargs)

    xs, ys = [], []
    if not len(list(values)):
        return np.array([]), np.array([])
    for v in values:
        if is_structural:
            p = replace(base, n_steps=n_steps, burn_in=burn_in, **{param: float(v)})
            these_rules = rules
        else:
            p = replace(base, n_steps=n_steps, burn_in=burn_in)
            these_rules = market(n_types, **{**rule_kwargs, param: float(v)})
        res = simulate(p, these_rules)
        # A divergent run contributes values at the explosion cap, which would
        # dominate the vertical scale and hide the attractor everywhere else.
        # Such runs are dropped: the parameter value simply has no attractor.
        if res.diverged:
            continue
        tail = res.x[-keep:]
        tail = tail[np.isfinite(tail)]
        xs.append(np.full(tail.size, v))
        ys.append(tail)

    return np.concatenate(xs), np.concatenate(ys)


# ==========================================================================
# Lyapunov exponents
# ==========================================================================


def lyapunov_rosenstein(
    x: np.ndarray,
    emb_dim: int = 5,
    lag: int = 1,
    min_sep: int = 20,
    horizon: int = 30,
    fit_range: tuple[int, int] = (1, 12),
    return_curve: bool = False,
):
    """Largest Lyapunov exponent via the Rosenstein et al. (1993) algorithm.

    The series is embedded in ``emb_dim`` dimensions using delay ``lag``. For
    every embedded point the nearest neighbour that is at least ``min_sep``
    periods away in time is located (the Theiler window, which prevents
    temporally adjacent points from masquerading as dynamical neighbours). The
    mean log separation of these pairs is then tracked forward, and the largest
    exponent is the slope of the resulting curve over its initial linear stretch.

    A positive value is the standard numerical signature of deterministic chaos:
    nearby states separate exponentially, so long-horizon prediction fails even
    though the system contains no random shocks at all.

    Returns
    -------
    float
        Estimated exponent in nats per period, or ``nan`` if the embedding is
        degenerate (e.g. the series has converged to a fixed point).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 200 or np.std(x) < 1e-12:
        return float("nan")

    m, tau = emb_dim, lag
    n_pts = x.size - (m - 1) * tau
    if n_pts <= horizon + min_sep + 2:
        return float("nan")

    # Delay embedding: row i is [x_i, x_{i+tau}, ..., x_{i+(m-1)tau}]
    emb = np.column_stack([x[i * tau : i * tau + n_pts] for i in range(m)])
    usable = n_pts - horizon

    # Nearest neighbour outside the Theiler window. A KD-tree is queried for the
    # closest few candidates and the first one far enough away in time is kept;
    # this is orders of magnitude cheaper than the full pairwise distance matrix
    # and is what makes a Lyapunov scan over a fine parameter grid feasible.
    tree = cKDTree(emb[:usable])
    n_cand = min(usable, max(2 * min_sep, 40))
    _, cand = tree.query(emb[:usable], k=n_cand, workers=-1)
    cand = np.atleast_2d(cand)

    rows = np.arange(usable)[:, None]
    valid = np.abs(cand - rows) >= min_sep
    first = np.argmax(valid, axis=1)
    has_any = valid.any(axis=1)
    if has_any.sum() < 50:
        return float("nan")
    i_idx = np.arange(usable)[has_any]
    j_idx = cand[has_any, first[has_any]]

    # Track mean log divergence forward. Separations are floored at a tiny
    # fraction of the attractor size rather than discarded: on an exactly
    # periodic orbit neighbouring points coincide, and dropping those pairs
    # would return nan for precisely the regimes we most want to label as
    # non-chaotic. With the floor a stable cycle yields a flat curve, hence an
    # exponent of approximately zero.
    floor = 1e-12 * max(np.std(x), 1e-12)
    curve = np.full(horizon + 1, np.nan)
    for k in range(horizon + 1):
        d = np.linalg.norm(emb[i_idx + k] - emb[j_idx + k], axis=1)
        d = np.maximum(d, floor)
        curve[k] = np.mean(np.log(d))

    lo, hi = fit_range
    hi = min(hi, horizon)
    seg = curve[lo : hi + 1]
    if np.any(~np.isfinite(seg)) or seg.size < 3:
        return float("nan")
    slope = np.polyfit(np.arange(lo, hi + 1), seg, 1)[0]
    if return_curve:
        return float(slope), curve
    return float(slope)


def lyapunov_scan(
    values: Sequence[float],
    param: str = "beta",
    base: ABSParams | None = None,
    rules: Sequence[BeliefRule] | None = None,
    n_steps: int = 6_000,
    burn_in: int = 2_000,
    **kwargs,
) -> np.ndarray:
    """Largest Lyapunov exponent along a one-dimensional parameter grid."""
    base = base or ABSParams()
    rules = rules or market(6)
    out = np.empty(len(values))
    for i, v in enumerate(values):
        p = replace(base, n_steps=n_steps, burn_in=burn_in, **{param: float(v)})
        res = simulate(p, rules).trimmed()
        out[i] = lyapunov_rosenstein(res.x, **kwargs)
    return out


# ==========================================================================
# Return diagnostics
# ==========================================================================


def autocorrelation(z: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """Sample autocorrelation of ``z`` for lags ``1 .. max_lag``."""
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)] - np.mean(z[np.isfinite(z)])
    denom = np.dot(z, z)
    if denom <= 0:
        return np.zeros(max_lag)
    return np.array([np.dot(z[:-k], z[k:]) / denom for k in range(1, max_lag + 1)])


def hill_estimator(z: np.ndarray, tail_fraction: float = 0.05) -> float:
    """Hill tail index of ``|z|``.

    Values near 3 are the canonical "cubic law" of financial returns; a value
    of ``inf`` (or a very large number) indicates thin, roughly exponential
    tails. Requires at least 30 order statistics in the tail.
    """
    z = np.abs(np.asarray(z, dtype=float))
    z = np.sort(z[np.isfinite(z) & (z > 0)])[::-1]
    k = int(tail_fraction * z.size)
    if k < 30:
        return float("nan")
    top = z[:k]
    return float(1.0 / np.mean(np.log(top / z[k])))


def stylised_facts(result: ABSResult, max_lag: int = 40) -> dict:
    """The standard battery of stylised facts for a simulated return series.

    Returns a dictionary holding the returns themselves, their moments, the
    autocorrelation of raw and absolute returns, a Jarque-Bera test and a Hill
    tail index. Volatility clustering is present when the ACF of raw returns is
    essentially flat while the ACF of absolute returns decays slowly and stays
    positive.
    """
    r = result.trimmed().returns(log=True)
    r = r[np.isfinite(r)]
    acf_r = autocorrelation(r, max_lag)
    acf_abs = autocorrelation(np.abs(r), max_lag)
    jb_stat, jb_p = stats.jarque_bera(r)
    return {
        "returns": r,
        "mean": float(np.mean(r)),
        "std": float(np.std(r, ddof=1)),
        "skew": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r, fisher=True)),
        "jarque_bera": float(jb_stat),
        "jarque_bera_p": float(jb_p),
        "hill_index": hill_estimator(r),
        "acf_returns": acf_r,
        "acf_abs_returns": acf_abs,
        "acf_ci": 1.96 / np.sqrt(r.size),
    }


# ==========================================================================
# Sweeps
# ==========================================================================


def parameter_sweep_2d(
    grid_a: Sequence[float],
    grid_b: Sequence[float],
    param_a: str,
    param_b: str,
    statistic: Callable[[ABSResult], float],
    base: ABSParams | None = None,
    n_types: int = 6,
    n_steps: int = 3_000,
    burn_in: int = 1_500,
    rule_kwargs: dict | None = None,
) -> np.ndarray:
    """Evaluate a scalar statistic over a two-dimensional parameter grid.

    Either parameter may be structural (a field of :class:`ABSParams`) or a
    belief loading understood by :func:`bh1998.agents.market`; the two cases can
    be mixed freely.

    Returns
    -------
    ndarray, shape (len(grid_b), len(grid_a))
        Laid out so that ``imshow`` puts ``param_a`` on the horizontal axis.
    """
    base = base or ABSParams()
    rule_kwargs = dict(rule_kwargs or {})
    out = np.empty((len(grid_b), len(grid_a)))

    for j, vb in enumerate(grid_b):
        for i, va in enumerate(grid_a):
            struct, belief = {}, dict(rule_kwargs)
            for name, val in ((param_a, va), (param_b, vb)):
                (struct if hasattr(base, name) else belief)[name] = float(val)
            p = replace(base, n_steps=n_steps, burn_in=burn_in, **struct)
            res = simulate(p, market(n_types, **belief))
            try:
                out[j, i] = statistic(res)
            except Exception:
                out[j, i] = np.nan
    return out


def survival_table(results: dict[str, ABSResult]) -> pd.DataFrame:
    """Long-run ecology: average share and average profit per type per scenario.

    Parameters
    ----------
    results : dict
        Mapping from scenario label to :class:`ABSResult`.
    """
    rows = []
    for label, res in results.items():
        shares = res.mean_fraction()
        profits = res.mean_profit()
        for name, s, pr in zip(res.names, shares, profits):
            rows.append(
                {
                    "Scenario": label,
                    "Type": name,
                    "Mean share": s,
                    "Mean profit": pr,
                }
            )
    return pd.DataFrame(rows)


def detect_period(x: np.ndarray, max_period: int = 64, tol: float = 1e-6) -> int:
    """Smallest ``k`` such that the tail of ``x`` repeats with period ``k``.

    Returns ``0`` if no period up to ``max_period`` fits within ``tol``, which
    is the signature of quasi-periodic or chaotic motion. Used to colour the
    bifurcation diagram by dynamical regime.
    """
    x = np.asarray(x, dtype=float)
    tail = x[-4 * max_period :]
    if tail.size < 2 * max_period or np.ptp(tail) < tol:
        return 1 if np.ptp(tail) < tol else 0
    for k in range(1, max_period + 1):
        if np.max(np.abs(tail[k:] - tail[:-k])) < tol:
            return k
    return 0
