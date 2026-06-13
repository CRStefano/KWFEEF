"""
Dinamiche ecologiche per l'estensione life-history del framework S-FEEF:
DENSITA'-DIPENDENZA (regolazione di popolazione) e AMBIENTE STOCASTICO
(bet-hedging). Costruite sulla stessa sopravvivenza basale P0 = 1 - p_dead(pi*)
fissata dalla policy percettiva S-FEEF.

Strategia a effort costante (per chiarezza dei due effetti):
    sopravvivenza   s(u) = P0 (1-u)^k         (mantenimento somatico)
    fecondita'      b(u) = c u^gamma           (reclutamento atteso per adulto)
    R0(u) = b(u) / (1 - s(u))                  (figli attesi nella vita)

Un perenne (s>0) e' iteroparo; s->0 (u->1) e' semelparia (annuale).

DENSITA'-DIPENDENZA: il reclutamento (produzione di nuovi nati) e' regolato alla
Beverton-Holt, R_eff = R / (1 + R/K_R). Con regolazione sullo stadio di
reclutamento, l'ESS massimizza R0 (Mylius & Diekmann 1995): la densita' fissa
l'ABBONDANZA di equilibrio ma non sposta la strategia ottimale.

AMBIENTE STOCASTICO: il reclutamento e' moltiplicato ogni anno da uno shock
ambientale eps_t lognormale (media 1, varianza sigma^2), correlato fra tutta la
prole di quell'anno. La fitness rilevante e' il TASSO DI CRESCITA STOCASTICO
(esponente di Lyapunov) a = E[log lambda_t], non la media aritmetica. Spalmare
la riproduzione su piu' anni (iteroparia) media shock indipendenti e riduce la
varianza di log-fitness: BET-HEDGING. Quindi sigma^2 alto favorisce u piu' basso
(piu' iteroparo), anche a parita' di media aritmetica.
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
# DENSITA'-DIPENDENZA (deterministica): equilibrio e regolazione
# ---------------------------------------------------------------------------
def equilibrium_population(u, P0, gamma, K=100.0, k=1.0, c=1.0,
                           years=2000) -> dict:
    """Itera il modello perenne con reclutamento Beverton-Holt fino
    all'equilibrio. n = adulti. Reclutamento R = b(u)*n regolato a
    R/(1+R/K). n(t+1) = s*n + R_eff."""
    s = survival(u, P0, k); b = fecundity(u, gamma, c)
    n = 1.0
    for _ in range(years):
        R = b * n
        R_eff = R / (1.0 + R / K)
        n = s * n + R_eff
    return {'N_star': float(n), 's': float(s), 'b': float(b), 'R0': R0(u, P0, gamma, k, c)}


# ---------------------------------------------------------------------------
# AMBIENTE STOCASTICO: tasso di crescita stocastico (Lyapunov)
# ---------------------------------------------------------------------------
def stochastic_growth_rate(u, P0, gamma, sigma, k=1.0, c=1.0,
                           years=20000, seed=0, density_K=None) -> float:
    """Tasso di crescita a lungo termine a = <log lambda_t> di una strategia a
    effort costante, con shock ambientale lognormale sul reclutamento.

    lambda_t = s + b*eps_t   (un adulto: sopravvive con s, produce b*eps_t nati).
    Se density_K e' dato, il reclutamento e' regolato (Beverton-Holt) e si misura
    la fitness di invasione attorno all'equilibrio del residente.
    """
    rng = np.random.default_rng(seed)
    s = survival(u, P0, k); b = fecundity(u, gamma, c)
    # eps lognormale con media 1 e varianza sigma^2
    mu_log = -0.5 * np.log(1.0 + sigma ** 2)
    sd_log = np.sqrt(np.log(1.0 + sigma ** 2))
    eps = rng.lognormal(mu_log, sd_log, size=years)
    if density_K is None:
        lam = s + b * eps
        return float(np.mean(np.log(lam)))
    # con densita': simula il residente all'equilibrio stocastico, poi misura
    # la crescita marginale (qui usiamo la stessa strategia: crescita ~ 0 a regime)
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
    """u* che massimizza il tasso di crescita stocastico (no densita')."""
    us = np.linspace(0.001, 0.999, n_grid)
    a = np.array([stochastic_growth_rate(u, P0, gamma, sigma, k, c, years, seed) for u in us])
    i = int(np.argmax(a))
    return {'u_star': float(us[i]), 'a_star': float(a[i]), 'sigma': float(sigma)}
