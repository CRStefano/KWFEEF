import json
import numpy as np
from core.model import SemanticForagingModel
import core.information as info

def run_experiment(f, name, mode='pos', food_loc='uniform', target='food', alpha=4.0, agent_fe=10):
    model = SemanticForagingModel(mode=mode, agentlevel_fe=agent_fe)
    initp = model.get_initial_distribution(agentinitloc=2, initial_foodloc=food_loc, logical_target=target)
    metrics = model.compute_pareto_front(initp, alpha=alpha)

    # Critical alpha (dark-room transition threshold)
    alpha_crit = metrics.get('alpha_crit', float('inf'))

    f.write(f"\n--- EXPERIMENT: {name} ---\n")
    f.write(f"Total MI I(X;Y):                         {metrics['actual_mi']:.4f} bits\n")
    f.write(f"Semantic Saturation Threshold (SST):       {metrics['semantic_saturation_threshold']:.4f} bits\n")
    f.write(f"SST/MI (semantic coverage):              {metrics['sst_efficiency']*100:.1f}%\n")
    f.write(f"K&W Semantic Information (V_reg - V_int):  {metrics['kw_semantic_information']:.4f} V-units\n")
    f.write(f"S-FEEF optimal MI (I*):                    {metrics['best_feef_mi']:.4f} bits\n")
    if metrics['semantic_saturation_threshold'] > 0:
        f.write(f"I* / SST (sub-threshold compression):        {metrics['best_feef_mi']/metrics['semantic_saturation_threshold']*100:.1f}%\n")
    else:
        f.write("I* / SST: N/A\n")
    f.write(f"S-FEEF optimal score:                     {metrics['best_feef_score']:.4f}\n")
    f.write(f"Critical alpha (dark-room transition):     {alpha_crit:.4f}\n")
    f.write(f"Alpha used vs critical alpha:              {alpha:.1f} vs {alpha_crit:.2f} -> {'INFO-SEEKING' if alpha < alpha_crit else 'DARK ROOM'}\n")


def run_sensitivity(f, alpha_values=[1.0, 2.0, 4.0, 6.0, 6.9, 7.0, 8.0, 10.0]):
    f.write("\n\n--- SENSITIVITY ANALYSIS (Phase transition in Alpha) ---\n")
    f.write(f"{'Alpha':>6} | {'I* (bits)':>10} | {'SST (bits)':>11} | {'I*/SST':>7} | {'FEEF score':>11} | {'Regime':>12}\n")
    f.write("-" * 72 + "\n")
    model = SemanticForagingModel(mode='pos', agentlevel_fe=10)
    initp = model.get_initial_distribution(agentinitloc=2, initial_foodloc='uniform', logical_target='food')
    # Compute alpha_crit once (consistent with the model, Eq. 8)
    m_base = model.compute_pareto_front(initp, alpha=4.0)
    alpha_crit = m_base['alpha_crit']
    for a in alpha_values:
        m = model.compute_pareto_front(initp, alpha=a)
        sst = m['semantic_saturation_threshold']
        feef_mi = m['best_feef_mi']
        regime = "INFO-SEEKING" if a < alpha_crit else "DARK ROOM"
        ratio = f"{feef_mi/sst*100:.1f}%" if sst > 0 else "N/A"
        f.write(f"  {a:6.1f} | {feef_mi:10.4f} | {sst:11.4f} | {ratio:>7} | {m['best_feef_score']:11.4f} | {regime:>12}\n")


def run_tau_sensitivity(f, tau_values=[1, 2, 3, 4, 5, 6, 7, 8, 10]):
    f.write("\n\n--- SENSITIVITY ANALYSIS (Timescale τ) ---\n")
    f.write(f"{'τ':>3} | {'p_dead':>8} | {'SST':>8} | {'I*':>8} | {'alpha_c':>9}\n")
    f.write("-" * 50 + "\n")
    for tau in tau_values:
        model = SemanticForagingModel(mode='pos', agentlevel_fe=10, timescale=tau)
        initp = model.get_initial_distribution(agentinitloc=2, initial_foodloc='uniform', logical_target='food')
        evolved = initp.dot(model.timeevolvedmx)
        deadprob = sum(evolved[ndx] for ndx in range(model.num_states) if model.id2state_dict[ndx][2]==0)
        metrics = model.compute_pareto_front(initp, alpha=4.0)
        sst = metrics['semantic_saturation_threshold']
        feef_mi = metrics['best_feef_mi']
        alpha_c = metrics.get('alpha_crit', float('inf'))
        f.write(f"  {tau:3d} | {deadprob:8.4f} | {sst:8.4f} | {feef_mi:8.4f} | {alpha_c:9.4f}\n")


if __name__ == "__main__":
    with open('results_v2.txt', 'w', encoding='utf-8') as f:
        # Sim 1: Baseline
        run_experiment(f, "Baseline (Aligned Targets)", target='food', alpha=4.0)

        # Sim 2: Costly (Semantic Sacrifice)
        run_experiment(f, "High Evolutionary Cost for Info Processing", target='food', alpha=6.0)

        # Sim 3: Decoupled Generative Model (Decoupled/Delusional)
        run_experiment(f, "Decoupled Generative Model (Uniform Logical Targets)", target='uniform', alpha=4.0)

        # Sim 4: Dark Room (System ignores Viability)
        run_experiment(f, "Dark Room Approximation (Alpha->inf)", target='food', alpha=9.5)

        # Sensitivity Analysis
        run_sensitivity(f)

        # Tau Sensitivity Analysis
        run_tau_sensitivity(f)
