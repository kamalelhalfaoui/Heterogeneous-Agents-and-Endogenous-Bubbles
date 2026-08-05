# The model

*Kamal El Halfaoui*

This note derives the adaptive belief system implemented in `src/bh1998/`, states
the timing convention precisely, and records the numerical decisions that the
code depends on.

---

## 1. The asset market

There is one risk-free asset in perfectly elastic supply paying gross return
$R = 1 + r > 1$, and one risky asset in zero net supply paying a stochastic
dividend $y_t$ with mean $\bar{y}$.

Wealth of a trader of type $h$ evolves as

$$
W_{h,t+1} = R W_{h,t} + \left( p_{t+1} + y_{t+1} - R p_t \right) z_{h,t}
$$

where $z_{h,t}$ is the number of units of the risky asset held. Traders are
myopic mean-variance maximisers with risk aversion $a$, so type $h$ solves

$$
\max_{z}\ \ \mathbb{E}_{h,t}\!\left[W_{h,t+1}\right] - \frac{a}{2}\,
\mathbb{V}_{h,t}\!\left[W_{h,t+1}\right]
$$

Assuming a common and constant conditional variance $\sigma^2$ of excess
returns, the first-order condition gives the demand

$$
z_{h,t} = \frac{\mathbb{E}_{h,t}\!\left[p_{t+1} + y_{t+1} - R p_t\right]}
{a \sigma^{2}}
$$

Market clearing with zero outside supply requires $\sum_h n_{h,t} z_{h,t} = 0$,
where $n_{h,t}$ is the fraction of traders using rule $h$. Hence

$$
R\,p_t = \sum_h n_{h,t}\, \mathbb{E}_{h,t}\!\left[p_{t+1} + y_{t+1}\right]
$$

### The fundamental price

Under homogeneous rational expectations the price solves
$R p_t^{*} = \mathbb{E}_t[p^{*}_{t+1} + y_{t+1}]$. With an IID dividend the
bounded solution is the constant

$$
p^{*} = \frac{\bar{y}}{R - 1}
$$

Define the **deviation from fundamental value**

$$
x_t \equiv p_t - p^{*}
$$

Every type is assumed to know the fundamental and to disagree only about the
deviation, so beliefs take the form
$\mathbb{E}_{h,t}[p_{t+1} + y_{t+1}] = p^{*} + f_{h,t}$. Substituting collapses
the market to a single scalar equation:

$$
\boxed{\;R\,x_t = \sum_{h=1}^{H} n_{h,t}\, f_{h,t}\;}
\tag{1}
$$

The entire model is this equation plus a rule for $f_{h,t}$ and a rule for
$n_{h,t}$.

---

## 2. Belief rules

Every type is a parameter restriction of one linear predictor:

$$
f_{h,t} = \alpha_h
+ \varphi_h\, x_{t-1}
+ \gamma_h\,(x_{t-1} - x_{t-2})
+ \theta_h\, f_{h,t-1}
+ \xi_h\, \frac{\bar{x}_{t-1} + x_{t-1}}{2}
\tag{2}
$$

where $\bar{x}_{t-1} = \frac{1}{t}\sum_{s=0}^{t-1} x_s$ is the running sample
mean. All terms are dated $t-1$ or earlier, so $f_{h,t}$ is measurable with
respect to information available when the forecast is made.

| Type | Restriction | Rule | Interpretation |
|---|---|---|---|
| Fundamentalists | all zero | $f = 0$ | price returns to fundamental value |
| Trend followers | $\alpha,\varphi \neq 0$ | $f = \alpha + \varphi x_{t-1}$ | biased chartist |
| Adapters | $\varphi + \theta = 1$ | $f = \varphi x_{t-1} + (1-\varphi) f_{t-1}$ | adaptive expectations (EWMA) |
| Learners | $\gamma, \xi \neq 0$ | $f = \gamma \Delta x_{t-1} + \xi \bar{a}_{t-1}$ | anchor and adjust |
| Optimists | $\alpha > 0$ | $f = \alpha$ | permanent upward bias |
| Pessimists | $\alpha < 0$ | $f = \alpha$ | permanent downward bias |

The adapter restriction $\varphi + \theta = 1$ is what makes the rule an
exponentially weighted moving average of past deviations, and therefore
unbiased in the long run whenever $x_t$ is stationary. This matters for
interpreting Figure 12: the adapters are the best-calibrated forecasters in the
market and they are also the least profitable.

Fundamentalists pay an information cost $C_1 = 0.05$; all other types pay
nothing. Knowing the fundamental value is not free, while extrapolating the
last price move is.

---

## 3. Fitness and evolutionary selection

Realised profit is the excess return on the position taken one period earlier:

$$
\pi_{h,t} = \left(x_t - R x_{t-1}\right)
\frac{f_{h,t-1} - R x_{t-1}}{a \sigma^{2}} - C_h
\tag{3}
$$

Fitness accumulates profit geometrically:

$$
U_{h,t} = \pi_{h,t} + w\, U_{h,t-1}, \qquad w \in [0,1)
\tag{4}
$$

so $1/(1-w)$ is roughly the effective number of periods a rule is judged on.

Shares are updated by a discrete-choice rule with inertia $\delta$:

$$
n_{h,t} = \delta\, n_{h,t-1} + (1-\delta)\,
\frac{\exp\!\left(\beta U_{h,t-1}\right)}
{\sum_{k=1}^{H} \exp\!\left(\beta U_{k,t-1}\right)}
\tag{5}
$$

The **intensity of choice** $\beta \geq 0$ governs how sharply traders move
toward whichever rule has been performing best. At $\beta = 0$ shares stay
uniform regardless of performance; as $\beta \to \infty$ all mass concentrates
on last period's winner.

### Parameter summary

| Symbol | Meaning | Baseline |
|---|---|---|
| $R$ | gross risk-free return | $1.05$ |
| $\beta$ | intensity of choice | $2$ |
| $\delta$ | inertia in shares | $0$ |
| $w$ | memory in fitness | $0$ |
| $a$ | risk aversion | $1$ |
| $\sigma$ | conditional volatility | $1$ |
| $\bar y$ | mean dividend | $1$ |
| $C_1$ | fundamentalist information cost | $0.05$ |

---

## 4. Timing

Within period $t$, in order:

1. **Forecasts.** Each type evaluates equation (2) using $x_{t-1}, x_{t-2}$,
   $f_{h,t-1}$ and $\bar{x}_{t-1}$.
2. **Selection.** Shares are set by (5) from fitness $U_{h,t-1}$.
3. **Clearing.** The price is set by (1): $x_t = \frac{1}{R}\sum_h n_{h,t} f_{h,t}$.
4. **Settlement.** Profit (3) is realised on the position taken at $t-1$, which
   was based on $f_{h,t-1}$.
5. **Fitness update.** Equation (4).

The one-period information lag between steps 3 and 4 is not a modelling
convenience — it is the source of the dynamics. Traders act on a forecast, the
price then moves, and only afterwards do they learn whether the forecast paid.
Remove the lag and the feedback loop that generates cycles and chaos disappears.

---

## 5. Numerical decisions

Three choices in the implementation are load-bearing.

### 5.1 Overflow-safe discrete choice

Equation (5) is never evaluated in its literal form. For $\beta U \gtrsim 700$
the exponential overflows to `inf` and the shares become `nan`; in double
precision this happens for $\beta$ around 5 with typical fitness values. The
code instead uses the algebraically identical shifted form

$$
\frac{\exp(\beta U_h)}{\sum_k \exp(\beta U_k)}
= \frac{\exp\!\left(\beta U_h - m\right)}{\sum_k \exp\!\left(\beta U_k - m\right)},
\qquad m = \max_k \beta U_k
$$

Every exponent is now at most zero. Without this the bifurcation diagram in
Figure 8, which runs to $\beta = 40$, could not be computed at all.

### 5.2 Divergence is caught, not ignored

When trend extrapolation is strong ($\varphi > R$) and the fundamentalist share
collapses, $x_t$ grows without bound. This is a genuine property of the model,
not a bug, but it must be detected: left alone it overflows and silently fills
every downstream statistic with `nan`. The kernel truncates the run, records
`diverged_at`, and holds the arrays at their last finite value. Divergent runs
are excluded from bifurcation diagrams and hatched on the regime map.

### 5.3 Returns are taken on the price, not the deviation

The deviation $x_t$ crosses zero constantly, so $(x_t - x_{t-1})/x_{t-1}$ is
numerically meaningless. All return statistics use the price level
$p_t = p^{*} + x_t$, which with $\bar y = 1$ and $R = 1.05$ sits at $p^{*} = 20$
and stays comfortably positive:

$$
r_t = \ln p_t - \ln p_{t-1}
$$

---

## 6. Steady state and local stability

Set $\delta = w = 0$ and suppose all types share a common forecast at a fixed
point. A steady state $\bar{x}$ of the deterministic skeleton satisfies

$$
R \bar{x} = \sum_h \bar{n}_h \left[
\alpha_h + \varphi_h \bar{x} + \theta_h \bar{f}_h + \xi_h \bar{x}
\right]
$$

With no biases ($\alpha_h = 0$ for all $h$) the fundamental steady state
$\bar{x} = 0$ always exists, and the shares there are uniform because all types
earn identical zero profit. With biases present the steady state is nonzero and
the shares are not uniform.

Local stability turns on the weighted extrapolation coefficient
$\sum_h \bar{n}_h \varphi_h$ relative to $R$. Intuitively, a unit shock to
$x_{t-1}$ moves the aggregate forecast by that weighted coefficient and the
price by $1/R$ times it, so the fundamental steady state is locally stable when

$$
\frac{1}{R}\sum_h \bar{n}_h\, \varphi_h < 1
$$

This is why $\varphi = R$ is marked on Figure 9, and why the complex region
begins somewhat *before* it: the destabilising rule does not need to satisfy
$\varphi > R$ on its own, only to hold enough share that the weighted average
crosses the threshold. Because $\bar n_h$ itself depends on $\beta$, the
stability boundary is a curve in the $(\beta, \varphi)$ plane rather than a
vertical line — which is exactly what Figure 13 maps.

---

## 7. Diagnostics

### Largest Lyapunov exponent

Estimated from the simulated series by the Rosenstein algorithm. The series is
embedded in $m$ dimensions with delay $\tau$; for each embedded point the
nearest neighbour at least `min_sep` periods away in time is found (the Theiler
window, which prevents temporally adjacent points from being mistaken for
dynamical neighbours); the mean log separation of these pairs is tracked
forward; and $\lambda_{\max}$ is the slope of the initial linear stretch:

$$
\frac{1}{N}\sum_{i} \ln d_i(k) \;\approx\; \lambda_{\max}\, k + \text{const}
$$

**Validation.** The implementation is tested against systems with published
exponents (`tests/test_bh1998.py`):

| System | True $\lambda_{\max}$ | Estimated |
|---|---|---|
| Hénon map ($a=1.4$, $b=0.3$) | $0.419$ | $0.411$ |
| Logistic map, $r = 4$ | $\ln 2 \approx 0.693$ | $0.585$ |
| Logistic map, $r = 3.5$ (period 4) | $< 0$ | $\approx 0$ |

The estimator resolves exponents of order $0.05$ comfortably. Anything within
$\pm 0.01$ of zero is reported as indistinguishable from zero, which is why the
Lyapunov panels carry an explicit tolerance band.

### Period detection

The smallest $k \leq 64$ such that the tail of the series repeats to within
$10^{-6}$; $0$ if none exists. Recovers the logistic period-doubling cascade
$1 \to 2 \to 4 \to 8 \to$ aperiodic exactly.

### Hill tail index

$$
\hat{\zeta}^{-1} = \frac{1}{k}\sum_{i=1}^{k} \ln \frac{|r|_{(i)}}{|r|_{(k)}}
$$

over the top $k$ order statistics of $|r_t|$. Empirical equity returns
typically give $\hat\zeta \approx 3$, the "cubic law". Validated against a
Pareto sample with known index.

---

## 8. Known limitations

- **The conditional variance is constant.** Demands use a fixed $\sigma^2$ rather
  than a conditional variance updated from realised returns. Endogenous
  time-varying risk would be a natural extension and would plausibly strengthen
  volatility clustering.
- **Belief parameters are fixed.** Types cannot adjust $\alpha, \varphi, \gamma,
  \theta, \xi$ over time; only the *shares* evolve. The rule set is exogenous
  and closed.
- **Dividends are IID.** The fundamental price is therefore constant, and all
  price movement is deviation. A stochastic fundamental would separate news
  from endogenous dynamics.
- **Zero net supply.** There is no outside asset supply or wealth accumulation,
  so richer traders do not gain influence. Selection operates on the number of
  followers, not on capital.
- **Costs are static.** The fundamentalist cost is a fixed per-period charge and
  does not scale with the informational advantage it buys.

---

## 9. References

- Brock, W. A. and Hommes, C. H. (1998). "Heterogeneous beliefs and routes to
  chaos in a simple asset pricing model." *Journal of Economic Dynamics and
  Control* 22(8-9), 1235-1274.
- Brock, W. A. and Hommes, C. H. (1997). "A rational route to randomness."
  *Econometrica* 65(5), 1059-1095.
- Hommes, C. H. (2013). *Behavioral Rationality and Heterogeneous Expectations
  in Complex Economic Systems.* Cambridge University Press.
- Rosenstein, M. T., Collins, J. J. and De Luca, C. J. (1993). "A practical
  method for calculating largest Lyapunov exponents from small data sets."
  *Physica D* 65(1-2), 117-134.
- Hill, B. M. (1975). "A simple general approach to inference about the tail of
  a distribution." *Annals of Statistics* 3(5), 1163-1174.
- LeBaron, B. (2006). "Agent-based computational finance." In *Handbook of
  Computational Economics*, vol. 2, 1187-1233.
- Cont, R. (2001). "Empirical properties of asset returns: stylized facts and
  statistical issues." *Quantitative Finance* 1(2), 223-236.
