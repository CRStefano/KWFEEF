"""
AGE-STRUCTURED extension of the S-FEEF framework (explicit time).

Unlike the geometric-renewal model (lifehistory.py), here
survival and fecundity depend on AGE. The organism chooses an ENTIRE
allocation SCHEDULE u(a) (fraction of the energy budget diverted from
somatic maintenance to reproduction, at each age a). For age a:

    survival   s(a) = P0 * (1 - u(a))^k         (maintenance -> survival)
    fecundity  m(a) = c * u(a)^gamma            (reproduction)

with P0 = 1 - p_dead(pi*) fixed by the S-FEEF perceptual policy of the foraging model.

Fitness is Fisher's reproductive value V(0), solved by BACKWARD
INDUCTION (dynamic programming), with a physical maximum age A:

    V(A) = 0
    V(a) = max_{u in [0,1]}  [ c*u^gamma + P0*(1-u)^k * V(a+1) ]

V(a) is the reproductive value: the expected number of offspring still to be produced
from age a onward. Since V(a+1) DECREASES approaching A, the optimal
share u(a) INCREASES with age and hence realised survival s(a)
DECREASES: SENESCENCE emerges as the optimum, with no imposed deterioration.
At the final age V=0 -> u=1 -> s=0: terminal reproduction and death (the salmon).
"""
from __future__ import annotations
import numpy as np


def optimise_schedule(P0: float, gamma: float, A: int = 15, k: float = 1.0,
                      c: float = 1.0, n_grid: int = 2001) -> dict:
    """Backward induction. Returns schedule u(a), survival s(a),
    reproductive value V(a), cumulative survival l(a), R0=V(0)."""
    ug = np.linspace(0.0, 1.0, n_grid)
    fec = c * np.power(ug, gamma)              # m(u)
    surv = P0 * np.power(1.0 - ug, k)          # s(u)

    V = np.zeros(A + 1)
    u_star = np.zeros(A)
    s_star = np.zeros(A)
    m_star = np.zeros(A)
    for a in range(A - 1, -1, -1):
        g = fec + surv * V[a + 1]
        i = int(np.argmax(g))
        u_star[a] = ug[i]
        s_star[a] = surv[i]
        m_star[a] = fec[i]
        V[a] = g[i]

    # cumulative survival l(a) = prod_{j<a} s(j)
    l = np.ones(A)
    for a in range(1, A):
        l[a] = l[a - 1] * s_star[a - 1]

    return {
        'ages': np.arange(A),
        'u': u_star,            # reproductive share by age
        's': s_star,            # survival by age
        'm': m_star,            # fecundity by age
        'V': V[:A],             # reproductive value (Fisher) by age
        'l': l,                 # cumulative survival (survivorship)
        'R0': float(V[0]),      # fitness (expected lifetime offspring)
        'P0': float(P0),
        'gamma': float(gamma),
    }


def life_expectancy(res: dict) -> float:
    """Life expectancy = sum of the survivorship l(a)."""
    return float(np.sum(res['l']))


def age_at_peak_reproduction(res: dict) -> int:
    """Age at the peak of realised reproduction l(a)*m(a)."""
    realised = res['l'] * res['m']
    return int(np.argmax(realised))


def is_semelparous(res: dict, thresh: float = 0.97) -> bool:
    """Semelparity: almost all reproduction concentrated at a single age."""
    realised = res['l'] * res['m']
    tot = realised.sum()
    return bool(tot > 0 and realised.max() / tot >= thresh)
