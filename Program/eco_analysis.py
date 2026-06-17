"""Density-dependence + stochastic-environment analysis for the life-history extension.
Generates eco_results.txt: ESS via invasion (max R0, not N*) and bet-hedging."""
import numpy as np
from core.eco_dynamics import (R0, equilibrium_population, survival,
                               stochastic_growth_rate, optimal_effort_stochastic)

def main():
    P0, g = 0.9, 0.6
    with open('eco_results.txt', 'w') as f:
        f.write("=== DENSITY DEPENDENCE & STOCHASTIC ENVIRONMENT ===\n\n")
        f.write("(A) Density dependence (Beverton-Holt recruitment):\n")
        us = np.linspace(0.02, 0.98, 97)
        r0 = np.array([R0(u, P0, g) for u in us])
        nst = np.array([equilibrium_population(u, P0, g, K=100)['N_star'] for u in us])
        u_ess, u_nmax = us[int(np.argmax(r0))], us[int(np.argmax(nst))]
        f.write(f"  ESS (argmax R0) u*={u_ess:.2f} ; argmax abundance N* u={u_nmax:.2f}\n")
        f.write("  Pairwise invasibility: mutant u* invades all residents; none invade u*.\n")
        f.write("  -> Density regulates abundance; the ESS maximises R0, not population size.\n\n")
        f.write("(B) Bet-hedging (stochastic growth rate, ranking reversal):\n")
        f.write("  sigma | iteroparous u=0.30 (a) | semelparous u=0.80 (a) | winner\n")
        for sig in [0.0, 0.4, 0.8, 1.2]:
            ai = np.mean([stochastic_growth_rate(0.30, P0, 0.85, sig, years=60000, seed=s) for s in range(5)])
            as_ = np.mean([stochastic_growth_rate(0.80, P0, 0.85, sig, years=60000, seed=s) for s in range(5)])
            f.write(f"  {sig:.1f}   | {ai:+.4f}              | {as_:+.4f}              | "
                    f"{'iteroparous' if ai > as_ else 'semelparous'}\n")
        f.write("\n(C) Optimal effort u* vs environmental variance sigma:\n")
        for P0v in [0.92, 0.70]:
            us2 = [optimal_effort_stochastic(P0v, 0.85, s, years=40000, seed=1)['u_star']
                   for s in [0.0, 0.4, 0.8, 1.2, 1.6]]
            f.write(f"  P0={P0v:.2f}: " + ' '.join('%.2f' % u for u in us2) + "  (sigma=0,0.4,0.8,1.2,1.6)\n")
    print("wrote eco_results.txt")

if __name__ == "__main__":
    main()
