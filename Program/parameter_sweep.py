"""Parameter sweep over the S-FEEF + life-history models: what is robust, what
emerges. Generates sweep_results.txt."""
import numpy as np
from core.model import SemanticForagingModel, set_partitions
from core.lifehistory import optimal_allocation, critical_gamma
from core.eco_dynamics import optimal_effort_stochastic, R0, equilibrium_population

PARTS = list(set_partitions(6))

def analyse(mode='pos', tau=5, fdr=0.1, er=1, de=100.0, fe=10.0, alpha=9.5):
    m = SemanticForagingModel(mode=mode, timescale=tau, food_disappear_rate=fdr,
                              eat_radius=er, dead_entropy=de, agentlevel_fe=fe)
    ip = m.get_initial_distribution(2, 'uniform', 'food')
    r = m.compute_pareto_front(ip, alpha=alpha)
    ks, vs = np.array(r['info_curve_ks']), np.array(r['info_curve_vs'])
    # non-monotonicity: number of drops in the envelope + above-threshold penalty
    dips = int(np.sum(np.diff(vs) < -1e-6))
    vmax = vs.max(); vfull = vs[-1]
    over_pen = float(vmax - vfull)            # >0 => info beyond the peak is harmful
    # misleading gap: at equal I, spread V_best - V_worst (over the cloud)
    sk, sv = m.scatter_ks, m.scatter_vs
    gap = 0.0
    for lvl in np.unique(np.round(sk, 4)):
        mask = np.abs(sk - lvl) < 1e-4
        if mask.sum() > 1:
            gap = max(gap, float(sv[mask].max() - sv[mask].min()))
    return dict(I0=r['actual_mi'], SST=r['semantic_saturation_threshold'],
                Istar=r['best_feef_mi'], ac=r['alpha_crit'], kwsi=r['kw_semantic_information'],
                dips=dips, over_pen=over_pen, gap=gap,
                sub=(r['best_feef_mi'] < r['semantic_saturation_threshold'] - 1e-6))

out = []
out.append("="*78)
out.append("A) FORAGING S-FEEF SWEEP (2-D model). baseline: tau5 fdr0.1 er1 de100 fe10")
out.append("="*78)
base = analyse()
out.append(f"  BASELINE: I0={base['I0']:.3f} SST={base['SST']:.3f} I*={base['Istar']:.3f} "
           f"a_c={base['ac']:.2f} dips={base['dips']} over_pen={base['over_pen']:.2f} "
           f"misleading_gap={base['gap']:.1f} subthreshold={base['sub']}")
out.append("")
out.append("  -- eat_radius (bodily affordance) --")
for er in [0,1,2]:
    r=analyse(er=er); out.append(f"   er={er}: SST={r['SST']:.3f} I*={r['Istar']:.3f} a_c={r['ac']:.2f} "
        f"dips={r['dips']} over_pen={r['over_pen']:.2f} misleading_gap={r['gap']:.1f}")
out.append("  -- food_disappear_rate (volatility) --")
for fdr in [0.0,0.1,0.3,0.5]:
    r=analyse(fdr=fdr); out.append(f"   fdr={fdr}: SST={r['SST']:.3f} I*={r['Istar']:.3f} a_c={r['ac']:.2f} "
        f"dips={r['dips']} over_pen={r['over_pen']:.2f} gap={r['gap']:.1f}")
out.append("  -- dead_entropy lambda (death penalty) --")
for de in [50,100,500]:
    r=analyse(de=de); out.append(f"   lambda={de}: SST={r['SST']:.3f} I*={r['Istar']:.3f} a_c={r['ac']:.2f} kwsi={r['kwsi']:.2f}")
out.append("  -- agentlevel_fe (physiological reserve) --")
for fe in [5,10,20]:
    r=analyse(fe=fe); out.append(f"   fe={fe}: SST={r['SST']:.3f} I*={r['Istar']:.3f} a_c={r['ac']:.2f}")
out.append("  -- mode --")
for mode in ['pos','neg','neutral']:
    r=analyse(mode=mode); out.append(f"   {mode}: I0={r['I0']:.3f} SST={r['SST']:.3f} I*={r['Istar']:.3f} a_c={r['ac']:.2f} kwsi={r['kwsi']:.2f}")
out.append("  -- tau (horizon) --")
for tau in [3,5,7,10]:
    r=analyse(tau=tau); out.append(f"   tau={tau}: SST={r['SST']:.3f} I*={r['Istar']:.3f} a_c={r['ac']:.2f} dips={r['dips']}")

out.append("")
out.append("="*78)
out.append("B) LIFE-HISTORY: universality of gamma_c = P0")
out.append("="*78)
for P0 in [0.3,0.5,0.7,0.9]:
    gc=critical_gamma(P0)
    out.append(f"   P0={P0:.2f} -> gamma_c={gc:.3f} (expected {P0:.3f}); match={abs(gc-P0)<0.01}")

out.append("")
out.append("="*78)
out.append("C) BET-HEDGING: sigma* threshold (switch to iteroparity) vs P0")
out.append("="*78)
for P0 in [0.92,0.80,0.70]:
    prev=None; sstar=None
    for sig in np.linspace(0,1.6,33):
        u=optimal_effort_stochastic(P0,0.85,sig,years=20000,seed=2)['u_star']
        if prev is not None and prev>=0.5 and u<0.5: sstar=sig
        prev=u
    out.append(f"   P0={P0:.2f}: sigma at which u* drops below 0.5 ~ {sstar}")

txt="\n".join(out)
open("sweep_results.txt","w").write(txt)
print(txt)
