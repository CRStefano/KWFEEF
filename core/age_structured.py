"""
Estensione AGE-STRUCTURED del framework S-FEEF (tempo esplicito).

A differenza del modello a rinnovo geometrico (lifehistory.py), qui la
sopravvivenza e la fecondita' dipendono dall'ETA'. L'organismo sceglie un INTERO
SCHEDULE di allocazione u(a) (frazione del budget energetico dirottata dalla
manutenzione somatica alla riproduzione, a ogni eta' a). Per eta' a:

    sopravvivenza  s(a) = P0 * (1 - u(a))^k         (manutenzione -> sopravvivenza)
    fecondita'     m(a) = c * u(a)^gamma            (riproduzione)

con P0 = 1 - p_dead(pi*) fissato dalla policy percettiva S-FEEF del foraging.

La fitness e' il valore riproduttivo di Fisher V(0), risolto per INDUZIONE A
RITROSO (programmazione dinamica), con una eta' massima fisica A:

    V(A) = 0
    V(a) = max_{u in [0,1]}  [ c*u^gamma + P0*(1-u)^k * V(a+1) ]

V(a) e' il reproductive value: il numero atteso di figli ancora da produrre a
partire dall'eta' a. Poiche' V(a+1) DECRESCE avvicinandosi ad A, la quota
ottimale u(a) CRESCE con l'eta' e quindi la sopravvivenza realizzata s(a)
DECRESCE: la SENESCENZA emerge come ottimo, senza alcun deterioramento imposto.
All'ultima eta' V=0 -> u=1 -> s=0: riproduzione terminale e morte (il salmone).
"""
from __future__ import annotations
import numpy as np


def optimise_schedule(P0: float, gamma: float, A: int = 15, k: float = 1.0,
                      c: float = 1.0, n_grid: int = 2001) -> dict:
    """Induzione a ritroso. Restituisce schedule u(a), sopravvivenza s(a),
    reproductive value V(a), sopravvivenza cumulata l(a), R0=V(0)."""
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

    # sopravvivenza cumulata l(a) = prod_{j<a} s(j)
    l = np.ones(A)
    for a in range(1, A):
        l[a] = l[a - 1] * s_star[a - 1]

    return {
        'ages': np.arange(A),
        'u': u_star,            # quota riproduttiva per eta'
        's': s_star,            # sopravvivenza per eta'
        'm': m_star,            # fecondita' per eta'
        'V': V[:A],             # reproductive value (Fisher) per eta'
        'l': l,                 # sopravvivenza cumulata (survivorship)
        'R0': float(V[0]),      # fitness (figli attesi nella vita)
        'P0': float(P0),
        'gamma': float(gamma),
    }


def life_expectancy(res: dict) -> float:
    """Aspettativa di vita = somma della survivorship l(a)."""
    return float(np.sum(res['l']))


def age_at_peak_reproduction(res: dict) -> int:
    """Eta' al picco della riproduzione realizzata l(a)*m(a)."""
    realised = res['l'] * res['m']
    return int(np.argmax(realised))


def is_semelparous(res: dict, thresh: float = 0.97) -> bool:
    """Semelparita': quasi tutta la riproduzione concentrata in una sola eta'."""
    realised = res['l'] * res['m']
    tot = realised.sum()
    return bool(tot > 0 and realised.max() / tot >= thresh)
