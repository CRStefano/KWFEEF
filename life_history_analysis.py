"""
Analisi life-history (Eq. 11 del paper): transizione iteroparita'/semelparita'
costruita SOPRA il modello di foraggiamento S-FEEF.

P0 = 1 - p_dead(pi*) e' la sopravvivenza basale per ciclo, fissata dalla policy
percettiva S-FEEF ottimale e dall'orizzonte tau. La fitness di lignaggio e'
Phi(u) = u^gamma / (1 - P0 (1-u)^k). La strategia ottimale u* salta da interna
(iteroparita') a 1 (semelparita') quando gamma supera gamma_c = P0 (k=1).
"""
import numpy as np
from core.model import SemanticForagingModel, set_partitions
from core.lifehistory import optimal_allocation, critical_gamma


def optimal_policy_pdead(tau, alpha=9.5):
    m = SemanticForagingModel(mode='pos', agentlevel_fe=10, timescale=tau)
    ip = m.get_initial_distribution(2, 'uniform', 'food')
    best = None
    for part in set_partitions(6):
        ptau = ip @ np.linalg.matrix_power(m._build_transition(part), tau)
        I = m._mi_partition(part); V = m._viability(ptau); F = alpha * I - V
        if best is None or F < best[0]:
            best = (F, I, V, m._p_dead(ptau))
    return best[3]


def main():
    with open('lifehistory_results.txt', 'w') as f:
        f.write("=== LIFE-HISTORY ALLOCATION (Eq. 11) ===\n")
        f.write("Phi(u) = u^gamma / (1 - P0 (1-u)^k),  P0 = 1 - p_dead(pi*),  k=1\n")
        f.write("gamma_c = P0 (transizione iteroparita' -> semelparita')\n\n")
        f.write("Foraging mortality (via tau) -> strategia ottima (gamma=0.85):\n")
        f.write(f"{'tau':>4} {'p_dead':>8} {'P0=gamma_c':>11} {'u*(0.85)':>9} {'regime':>13}\n")
        for tau in [2, 3, 4, 5, 6, 7, 10]:
            pd = optimal_policy_pdead(tau); P0 = 1 - pd
            r = optimal_allocation(P0, 0.85)
            f.write(f"{tau:>4} {pd:>8.3f} {P0:>11.3f} {r['u_star']:>9.3f} {r['regime']:>13}\n")
        f.write("\nu* vs gamma for three mortality regimes:\n")
        for P0 in [0.92, 0.61, 0.36]:
            f.write(f"  P0={P0:.2f} (gamma_c={P0:.2f}): ")
            f.write(' '.join('%.2f' % optimal_allocation(P0, g)['u_star']
                             for g in [0.2, 0.4, 0.6, 0.8, 1.0]) + "  (gamma=0.2..1.0)\n")
    print("scritto lifehistory_results.txt")


if __name__ == "__main__":
    main()
