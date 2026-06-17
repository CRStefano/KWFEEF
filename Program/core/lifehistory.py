"""
Life-history extension of the S-FEEF framework (Eq. 11 of the paper).

It moves the bookkeeping of negentropy from the individual to the lineage: the organism
allocates an energy budget between SOMATIC MAINTENANCE and REPRODUCTION via a
fraction u in [0,1] (u = share diverted to reproduction). Per-season survival
is P(u) = P0 * (1-u)^k, where P0 = 1 - p_dead is the baseline survival
fixed by the S-FEEF perceptual policy of the foraging model; per-season
fecundity is B(u) = u^gamma. The lineage fitness (expected reproductive
value, perennial organism) is the geometric sum

    Phi(u) = B(u) / (1 - P(u)) = u^gamma / (1 - P0 (1-u)^k).

u->0   : no reproduction (Phi=0).
0<u*<1 : ITEROPARITY (repeated reproduction, soma maintained).
u*=1   : SEMELPARITY (suicidal reproduction: P(1)=0, the organism dies).

For k=1 there is a sharp phase transition: the optimum is interior if gamma < P0,
and at the boundary (u*=1, semelparity) if gamma >= P0. Hence

    gamma_c = P0 = 1 - p_dead.

High baseline mortality (low P0) lowers gamma_c and favours semelparity,
in agreement with classical theory (Cole; Charnov & Schaffer). It is the same
pattern as the S-FEEF dark-room transition: "spend on a resource only up to
its marginal return".
"""
from __future__ import annotations
import numpy as np


def lifetime_fitness(u, P0: float, gamma: float, k: float = 1.0):
    """Phi(u) = u^gamma / (1 - P0 (1-u)^k). Vectorised in u."""
    u = np.asarray(u, dtype=float)
    surv = P0 * np.power(np.clip(1.0 - u, 0.0, 1.0), k)
    denom = 1.0 - surv
    with np.errstate(divide='ignore', invalid='ignore'):
        phi = np.where(denom > 1e-12, np.power(u, gamma) / denom, np.power(u, gamma))
    return phi


def critical_gamma(P0: float, k: float = 1.0) -> float:
    """Critical gamma of the iteroparity/semelparity transition.
    For k=1 it is exactly P0; for general k from the sign of Phi'(1^-)."""
    if abs(k - 1.0) < 1e-9:
        return float(P0)
    # general criterion: u*=1 if Phi'(1^-) > 0. Numerical estimate of the gamma threshold.
    lo, hi = 1e-4, 50.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _is_semelparous(P0, mid, k):
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))


def _is_semelparous(P0, gamma, k, n=20001) -> bool:
    u = np.linspace(1e-6, 1.0, n)
    phi = lifetime_fitness(u, P0, gamma, k)
    return int(np.argmax(phi)) == n - 1


def optimal_allocation(P0: float, gamma: float, k: float = 1.0, n: int = 20001) -> dict:
    """Optimises Phi(u) on [0,1]. Returns u*, Phi*, regime and gamma_c."""
    u = np.linspace(1e-6, 1.0, n)
    phi = lifetime_fitness(u, P0, gamma, k)
    i = int(np.argmax(phi))
    u_star = float(u[i])
    semel = (i >= n - 2) or (u_star > 0.999)
    return {
        'u_star': u_star,
        'phi_star': float(phi[i]),
        'regime': 'SEMELPAROUS' if semel else 'ITEROPAROUS',
        'gamma_c': critical_gamma(P0, k),
        'P0': float(P0),
    }
