"""
Estensione life-history del framework S-FEEF (Eq. 11 del paper).

Sposta la contabilita' della negentropia dall'individuo al lignaggio: l'organismo
alloca un budget energetico fra MANTENIMENTO SOMATICO e RIPRODUZIONE tramite una
frazione u in [0,1] (u = quota dirottata alla riproduzione). La sopravvivenza
per stagione e P(u) = P0 * (1-u)^k, dove P0 = 1 - p_dead e' la sopravvivenza
basale fissata dalla policy percettiva S-FEEF del modello di foraggiamento; la
fecondita' per stagione e B(u) = u^gamma. La fitness di lignaggio (valore
riproduttivo atteso, organismo perenne) e' la somma geometrica

    Phi(u) = B(u) / (1 - P(u)) = u^gamma / (1 - P0 (1-u)^k).

u->0   : nessuna riproduzione (Phi=0).
0<u*<1 : ITEROPARITA' (riproduzione ripetuta, soma mantenuto).
u*=1   : SEMELPARITA' (riproduzione suicida: P(1)=0, l'organismo muore).

Per k=1 esiste una transizione di fase netta: l'ottimo e' interno se gamma < P0,
ed e' al bordo (u*=1, semelparita') se gamma >= P0. Quindi

    gamma_c = P0 = 1 - p_dead.

Alta mortalita' basale (P0 basso) abbassa gamma_c e favorisce la semelparita',
in accordo con la teoria classica (Cole; Charnov & Schaffer). E' lo stesso
schema della transizione dark-room di S-FEEF: "spendi su una risorsa solo fino
al suo ritorno marginale".
"""
from __future__ import annotations
import numpy as np


def lifetime_fitness(u, P0: float, gamma: float, k: float = 1.0):
    """Phi(u) = u^gamma / (1 - P0 (1-u)^k). Vettoriale in u."""
    u = np.asarray(u, dtype=float)
    surv = P0 * np.power(np.clip(1.0 - u, 0.0, 1.0), k)
    denom = 1.0 - surv
    with np.errstate(divide='ignore', invalid='ignore'):
        phi = np.where(denom > 1e-12, np.power(u, gamma) / denom, np.power(u, gamma))
    return phi


def critical_gamma(P0: float, k: float = 1.0) -> float:
    """gamma critico della transizione iteroparita'/semelparita'.
    Per k=1 e' esattamente P0; per k generico si calcola dal segno di Phi'(1^-)."""
    if abs(k - 1.0) < 1e-9:
        return float(P0)
    # criterio generale: u*=1 se Phi'(1^-) > 0. Stima numerica della soglia in gamma.
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
    """Ottimizza Phi(u) su [0,1]. Restituisce u*, Phi*, regime e gamma_c."""
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
