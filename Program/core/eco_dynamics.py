"""
Ecological dynamics for the life-history extension of the S-FEEF framework:
DENSITY DEPENDENCE (population regulation) and STOCHASTIC ENVIRONMENT
(bet-hedging). Built on the same baseline survival P0 = 1 - p_dead(pi*)
fixed by the S-FEEF perceptual policy.

Constant-effort strategy (for clarity of the two effects):
    survival    s(u) = P0 (1-u)^k         (somatic maintenance)
    fecundity   b(u) = c u^gamma           (expected recruitment per adult)
    R0(u) = b(u) / (1 - s(u))                  (expected lifetime offspring)

A perennial (s>0) is iteroparous; s->0 (u->1) is semelparity (annual).

DENSITY DEPENDENCE: recruitment (production of newborns) is regulated
Beverton-Holt style, R_eff = R / (1 + R/K_R). With regulation at the
recruitment stage, the ESS maximises R0 (Mylius & Diekmann 1995): density fixes
the equilibrium ABUNDANCE but does not shift the optimal strategy.

STOCHASTIC ENVIRONMENT: recruitment is multiplied each year by an
environmental shock eps_t, lognormal (mean 1, variance sigma^2), correlated across all
offspring of that year. The relevant fitness is the STOCHASTIC GROWTH RATE
(Lyapunov exponent) a = E[log lambda_t], not the arithmetic mean. Spreading
reproduction over more years (iteroparity) averages independent shocks and reduces the
variance of log-fitness: BET-HEDGING. Hence high sigma^2 favours lower u
(more iteroparous), even at equal arithmetic mean.
"""
from __future__ import annotations
import numpy as np


def survival(u, P0, k=1.0):
    return P0 * np.power(1.0 - u, k)


def fecundity(u, gamma, c=1.0):
    return c * np.power(u, gamma)


def R0(u, P0, gamma, k=1.0, c=1.0):
    s = survival(u, P0, k)
    return fecundity(u, gamma, c) / (1.0 - s) if s < 1 else np.inf


# ---------------------------------------------------------------------------
# DENSITY DEPENDENCE (deterministic): equilibrium and regulation
# ---------------------------------------------------------------------------
def equilibrium_population(u, P0, gamma, K=100.0, k=1.0, c=1.0,
                           years=2000) -> dict:
    """Iterates the perennial model with Beverton-Holt recruitment until
    equilibrium. n = adults. Recruitment R = b(u)*n regulated as
    R/(1+R/K). n(t+1) = s*n + R_eff."""
    s = survival(u, P0, k); b = fecundity(u, gamma, c)
    n = 1.0
    for _ in range(years):
        R = b * n
        R_eff = R / (1.0 + R / K)
        n = s * n + R_eff
    return {'N_star': float(n), 's': float(s), 'b': float(b), 'R0': R0(u, P0, gamma, k, c)}


# ---------------------------------------------------------------------------
# STOCHASTIC ENVIRONMENT: stochastic growth rate (Lyapunov)
# ---------------------------------------------------------------------------
def stochastic_growth_rate(u, P0, gamma, sigma, k=1.0, c=1.0,
                           years=20000, seed=0, density_K=None) -> float:
    """Long-term growth rate a = <log lambda_t> of a constant-effort
    strategy, with a lognormal environmental shock on recruitment.

    lambda_t = s + b*eps_t   (one adult: survives with s, produces b*eps_t newborns).
    If density_K is given, recruitment is regulated (Beverton-Holt) and we measure
    the invasion fitness around the resident's equilibrium.
    """
    rng = np.random.default_rng(seed)
    s = survival(u, P0, k); b = fecundity(u, gamma, c)
    # eps lognormal with mean 1 and variance sigma^2
    mu_log = -0.5 * np.log(1.0 + sigma ** 2)
    sd_log = np.sqrt(np.log(1.0 + sigma ** 2))
    eps = rng.lognormal(mu_log, sd_log, size=years)
    if density_K is None:
        lam = s + b * eps
        return float(np.mean(np.log(lam)))
    # with density: simulate the resident at stochastic equilibrium, then measure
    # the marginal growth (here the same strategy: growth ~ 0 at steady state)
    n = 1.0
    logs = []
    for t in range(years):
        R = b * eps[t] * n
        R_eff = R / (1.0 + R / density_K)
        n_next = s * n + R_eff
        logs.append(np.log(max(n_next / n, 1e-12)))
        n = max(n_next, 1e-9)
    return float(np.mean(logs[len(logs) // 2:]))


def optimal_effort_stochastic(P0, gamma, sigma, k=1.0, c=1.0,
                              n_grid=121, years=40000, seed=0) -> dict:
    """u* that maximises the stochastic growth rate (no density)."""
    us = np.linspace(0.001, 0.999, n_grid)
    a = np.array([stochastic_growth_rate(u, P0, gamma, sigma, k, c, years, seed) for u in us])
    i = int(np.argmax(a))
    return {'u_star': float(us[i]), 'a_star': float(a[i]), 'sigma': float(sigma)}
