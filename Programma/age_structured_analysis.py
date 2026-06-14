"""
Analisi del modello AGE-STRUCTURED accoppiato al foraggiamento S-FEEF.
Mostra: (1) senescenza emergente; (2) come la mortalita' da foraggiamento P0
(via orizzonte tau) regola durata della vita e velocita' di senescenza.
"""
import numpy as np
from core.model import SemanticForagingModel, set_partitions
from core.age_structured import optimise_schedule, life_expectancy, age_at_peak_reproduction, is_semelparous


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
    with open('age_structured_results.txt', 'w') as f:
        f.write("=== AGE-STRUCTURED LIFE HISTORY (backward induction) ===\n")
        f.write("s(a)=P0(1-u(a))^k,  m(a)=c u(a)^gamma,  V(a)=max_u[m+s V(a+1)], A=15,k=1,c=1\n\n")

        f.write("(1) Senescenza emergente (P0=0.92, gamma=0.6):\n")
        r = optimise_schedule(P0=0.92, gamma=0.6, A=15)
        f.write("  age   : " + ' '.join('%5d' % a for a in r['ages']) + "\n")
        f.write("  u(a)  : " + ' '.join('%.2f' % x for x in r['u']) + "\n")
        f.write("  s(a)  : " + ' '.join('%.2f' % x for x in r['s']) + "\n")
        f.write("  -> u(a) sale, s(a) scende: senescenza senza deterioramento imposto.\n\n")

        f.write("(2) Mortalita' da foraggiamento (tau) -> durata vita e regime (gamma=0.6):\n")
        f.write("  tau  p_dead    P0   vita_attesa  picco_eta  regime\n")
        for tau in [2, 3, 4, 5, 7, 10]:
            pd = optimal_policy_pdead(tau); P0 = 1 - pd
            r = optimise_schedule(P0=P0, gamma=0.6, A=15)
            reg = 'semelparo' if is_semelparous(r) else 'iteroparo'
            f.write("  %3d  %.3f   %.3f    %.2f        %2d       %s\n" % (
                tau, pd, P0, life_expectancy(r), age_at_peak_reproduction(r), reg))

        f.write("\n(3) Rendimenti riproduttivi (gamma) -> iteroparita'/semelparita' (P0=0.90):\n")
        for g in [0.5, 0.7, 0.9, 1.0, 1.4]:
            r = optimise_schedule(P0=0.90, gamma=g, A=15)
            real = r['l'] * r['m']; conc = real.max() / real.sum()
            reg = 'SEMELPARO' if is_semelparous(r) else 'iteroparo'
            f.write("  gamma=%.1f: concentrazione riprod=%3.0f%%  vita=%.2f  %s\n" % (
                g, 100 * conc, life_expectancy(r), reg))
    print("scritto age_structured_results.txt")


if __name__ == "__main__":
    main()
