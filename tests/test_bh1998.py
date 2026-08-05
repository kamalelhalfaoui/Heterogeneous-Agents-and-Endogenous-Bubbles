"""
Tests for the adaptive belief system.

Run with ``pytest -q`` from the repository root.

The suite covers three things: that the accounting identities of the model hold
exactly (shares sum to one, market clearing is satisfied, restrictions on belief
rules do what they claim), that the numerics survive the parameter ranges the
figures actually use (large beta, explosive trend extrapolation), and that the
chaos diagnostics recover known answers on systems whose exponents are published.

The last group matters most. A Lyapunov estimator that is never checked against
a known system is an elaborate way of generating a number with no content.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bh1998 import (  # noqa: E402
    ABSParams,
    adapter,
    calibration,
    descriptive_stats,
    detect_period,
    fundamental_price,
    fundamentalist,
    hill_estimator,
    lyapunov_rosenstein,
    market,
    optimist,
    simulate,
)
from bh1998.model import _softmax  # noqa: E402


# ==========================================================================
# Accounting identities
# ==========================================================================


def test_shares_sum_to_one():
    res = simulate(ABSParams(n_steps=2_000), market(6))
    assert np.allclose(res.n.sum(axis=1), 1.0, atol=1e-12)


def test_shares_are_non_negative():
    res = simulate(ABSParams(beta=30.0, n_steps=2_000), market(6))
    assert (res.n >= 0).all()


def test_market_clearing_identity():
    """R x_t must equal the share-weighted average forecast, to machine precision."""
    p = ABSParams(n_steps=1_500, noise=0.0)
    res = simulate(p, market(6))
    lhs = p.R * res.x[2:]
    rhs = np.einsum("th,th->t", res.n[2:], res.f[2:])
    assert np.allclose(lhs, rhs, atol=1e-10)


def test_fundamental_price():
    p = ABSParams(R=1.05, ybar=1.0)
    assert fundamental_price(p) == pytest.approx(20.0)


# ==========================================================================
# Belief rules
# ==========================================================================


def test_fundamentalist_forecast_is_zero():
    res = simulate(ABSParams(n_steps=500), [fundamentalist(), optimist()])
    assert np.allclose(res.f[:, 0], 0.0)


def test_adapter_weights_sum_to_one():
    """The restriction phi + theta = 1 is what makes the rule an EWMA."""
    a = adapter(phi=0.35)
    assert a.phi + a.theta == pytest.approx(1.0)


def test_market_rejects_bad_size():
    with pytest.raises(ValueError):
        market(4)


def test_rule_replace_is_a_copy():
    a = optimist(1.5)
    b = a.replace(alpha=2.5)
    assert a.alpha == 1.5 and b.alpha == 2.5


# ==========================================================================
# Numerical robustness
# ==========================================================================


def test_softmax_matches_naive_form_when_naive_is_safe():
    u = np.array([0.3, -0.2, 1.1, 0.0])
    beta = 2.0
    naive = np.exp(beta * u) / np.exp(beta * u).sum()
    assert np.allclose(_softmax(u, beta), naive)


def test_softmax_survives_extreme_fitness():
    """The naive form overflows here; the shifted form must not."""
    u = np.array([900.0, -900.0, 0.0])
    w = _softmax(u, beta=50.0)
    assert np.isfinite(w).all()
    assert w.sum() == pytest.approx(1.0)
    assert w[0] == pytest.approx(1.0)


def test_large_beta_produces_finite_path():
    res = simulate(ABSParams(beta=200.0, n_steps=3_000), market(6))
    assert np.isfinite(res.x).all()


def test_explosive_calibration_is_flagged_not_nan():
    """Strong extrapolation makes the price diverge; that must be caught."""
    res = simulate(
        ABSParams(R=1.05, beta=10.0, n_steps=5_000),
        market(6, alpha_opt=0.3, phi_trend=1.30),
    )
    assert res.diverged
    assert res.diverged_at is not None
    assert np.isfinite(res.x).all()


def test_zero_beta_gives_uniform_shares():
    res = simulate(ABSParams(beta=0.0, n_steps=800), market(6))
    assert np.allclose(res.n[5:], 1.0 / 6, atol=1e-12)


def test_noise_is_reproducible_under_seed():
    a = simulate(ABSParams(noise=0.05, seed=42, n_steps=1_000), market(6))
    b = simulate(ABSParams(noise=0.05, seed=42, n_steps=1_000), market(6))
    c = simulate(ABSParams(noise=0.05, seed=43, n_steps=1_000), market(6))
    assert np.array_equal(a.x, b.x)
    assert not np.array_equal(a.x, c.x)


# ==========================================================================
# Parameter validation
# ==========================================================================


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(R=1.0),
        dict(R=0.9),
        dict(delta=1.0),
        dict(delta=-0.1),
        dict(w=1.5),
        dict(n_steps=3),
        dict(burn_in=10_000, n_steps=500),
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ABSParams(**kwargs)


def test_bad_initial_shares_are_rejected():
    with pytest.raises(ValueError):
        simulate(ABSParams(n_steps=200), market(6), n_init=np.full(6, 0.5))


# ==========================================================================
# Chaos diagnostics, validated against published values
# ==========================================================================


def _logistic(r: float, n: int = 6_000) -> np.ndarray:
    x = np.empty(n)
    x[0] = 0.4
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x[1_000:]


def _henon(n: int = 6_000, a: float = 1.4, b: float = 0.3) -> np.ndarray:
    x = np.empty(n)
    y = np.empty(n)
    x[0] = y[0] = 0.1
    for i in range(1, n):
        x[i] = 1 - a * x[i - 1] ** 2 + y[i - 1]
        y[i] = b * x[i - 1]
    return x[1_000:]


def test_lyapunov_recovers_henon_exponent():
    """The Henon map has a largest Lyapunov exponent of about 0.419."""
    assert lyapunov_rosenstein(_henon()) == pytest.approx(0.419, abs=0.06)


def test_lyapunov_positive_for_fully_chaotic_logistic():
    assert lyapunov_rosenstein(_logistic(4.0)) > 0.3


def test_lyapunov_near_zero_on_a_stable_cycle():
    assert abs(lyapunov_rosenstein(_logistic(3.5))) < 0.01


def test_period_doubling_cascade_is_detected():
    assert detect_period(_logistic(3.2)) == 2
    assert detect_period(_logistic(3.5)) == 4
    assert detect_period(_logistic(3.55)) == 8
    assert detect_period(_logistic(3.9)) == 0  # aperiodic


def test_hill_index_recovers_a_known_pareto_tail():
    rng = np.random.default_rng(0)
    sample = rng.pareto(3.0, size=200_000) + 1.0
    assert hill_estimator(sample, 0.02) == pytest.approx(3.0, rel=0.12)


# ==========================================================================
# Named calibrations behave as advertised
# ==========================================================================


def test_benchmark_calibration_is_a_stable_cycle():
    p, rules = calibration("benchmark", n_steps=6_000, burn_in=3_000)
    x = simulate(p, rules).trimmed().x
    assert detect_period(x) > 0
    assert abs(lyapunov_rosenstein(x)) < 0.01


def test_chaotic_calibration_has_a_positive_exponent():
    p, rules = calibration("chaotic", n_steps=8_000, burn_in=3_000)
    res = simulate(p, rules)
    assert not res.diverged
    assert lyapunov_rosenstein(res.trimmed().x) > 0.02


def test_unknown_calibration_raises():
    with pytest.raises(KeyError):
        calibration("nonexistent")


# ==========================================================================
# Result helpers
# ==========================================================================


def test_returns_are_computed_on_a_positive_price_level():
    res = simulate(ABSParams(n_steps=3_000), market(6))
    r = res.returns()
    assert np.isfinite(r).all()
    assert r.size == res.x.size - 1


def test_trimmed_drops_exactly_the_burn_in():
    p = ABSParams(n_steps=2_000, burn_in=600)
    res = simulate(p, market(6))
    assert res.trimmed().x.size == 2_000 - 600


def test_descriptive_stats_reports_excess_kurtosis():
    rng = np.random.default_rng(1)
    st = descriptive_stats(rng.normal(size=200_000))
    assert st["Excess kurtosis"] == pytest.approx(0.0, abs=0.06)
    assert st["Mean"] == pytest.approx(0.0, abs=0.02)


def test_running_mean_profit_matches_a_direct_average():
    res = simulate(ABSParams(n_steps=500), market(6))
    rmp = res.running_mean_profit()
    assert rmp[99] == pytest.approx(res.profit[:100].mean(axis=0))
