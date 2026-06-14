"""
Funzioni di teoria dell'informazione per il framework S-FEEF.
Implementa misure di informazione semantica, entropia e MI.
"""

import numpy as np


def entropy(p: np.ndarray, base: float = 2.0) -> float:
    """Entropia di Shannon H(X) in bit (o nats se base=e)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p) / np.log(base)))


def mutual_information(p_joint: np.ndarray, base: float = 2.0) -> float:
    """
    Mutua informazione I(X;Y) da distribuzione congiunta p(x,y).
    I(X;Y) = H(X) + H(Y) - H(X,Y)
    """
    p_joint = np.asarray(p_joint, dtype=float)
    p_joint = p_joint / (p_joint.sum() + 1e-12)

    p_x = p_joint.sum(axis=1)
    p_y = p_joint.sum(axis=0)

    h_x = entropy(p_x, base)
    h_y = entropy(p_y, base)
    h_xy = entropy(p_joint.ravel(), base)

    return max(0.0, h_x + h_y - h_xy)


def kl_divergence(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    """Divergenza KL D_KL(P || Q)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = (p > 0) & (q > 0)
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask]) / np.log(base)))


def conditional_entropy(p_joint: np.ndarray, base: float = 2.0) -> float:
    """Entropia condizionale H(Y|X) da distribuzione congiunta."""
    p_joint = np.asarray(p_joint, dtype=float)
    p_joint = p_joint / (p_joint.sum() + 1e-12)
    h_xy = entropy(p_joint.ravel(), base)
    h_x = entropy(p_joint.sum(axis=1), base)
    return max(0.0, h_xy - h_x)


def semantic_saturation_threshold(ks: np.ndarray, vs: np.ndarray,
                                   frac: float = 0.05) -> float:
    """
    Semantic Saturation Threshold (SST) -- definizione a RITORNO MARGINALE.

    Sulla frontiera cumulative-best (monotona) V*(I) = max_{I' <= I} V(I'), la
    SST e' il piu' grande I il cui ritorno marginale di viability per bit supera
    una soglia floor = frac * alpha_c, dove alpha_c e' il ritorno marginale
    massimo dalla policy nulla. Oltre la SST il ritorno e' ormai svanito.

    Questa definizione e' molto piu' robusta della vecchia "I minimo che
    raggiunge V_max entro una tolleranza" su una frontiera NON monotona: il
    valore non si sposta su un picco secondario, ed e' stabile per frac fra
    ~1%% e ~15%% (vedi robustness_analysis.py).
    """
    ks = np.asarray(ks, dtype=float)
    vs = np.asarray(vs, dtype=float)
    order = np.argsort(ks)
    ks, vs = ks[order], vs[order]
    if len(ks) < 2 or (ks[-1] - ks[0]) < 1e-12:
        return float(ks[-1])
    vbest = np.maximum.accumulate(vs)                 # frontiera cumulative-best
    if vbest[-1] - vbest[0] < 1e-10:                   # viability piatta -> nessuna info utile
        return float(ks[0])
    ac = max(((vbest[i] - vbest[0]) / (ks[i] - ks[0]))
             for i in range(1, len(ks)) if ks[i] - ks[0] > 1e-9)
    floor = frac * ac
    sst = ks[0]
    for i in range(1, len(ks)):
        if ks[i] - ks[i - 1] > 1e-9 and (vbest[i] - vbest[i - 1]) / (ks[i] - ks[i - 1]) > floor:
            sst = ks[i]
    return float(sst)


def kw_semantic_information(v_intrinsic: float, v_regulated: float) -> float:
    """
    Informazione Semantica di Kolchinsky & Wolpert:
    SI_KW = V_regulated - V_intrinsic
    Misura quanta informazione è causalmente rilevante per la viabilità.
    """
    return float(v_regulated - v_intrinsic)


def pareto_front_feef(ks: np.ndarray, vs: np.ndarray,
                       alpha: float) -> tuple[np.ndarray, int, float, float]:
    """
    Calcola la curva S-FEEF e il punto ottimale.

    S-FEEF(I) = alpha * I - V(I)

    Returns:
        fs: curva S-FEEF
        best_idx: indice del minimo
        best_mi: MI ottimale I*
        best_score: score S-FEEF minimo
    """
    ks = np.asarray(ks, dtype=float)
    vs = np.asarray(vs, dtype=float)
    fs = alpha * ks - vs
    best_idx = int(np.argmin(fs))
    return fs, best_idx, float(ks[best_idx]), float(fs[best_idx])


def critical_alpha(ks: np.ndarray, vs: np.ndarray) -> float:
    """
    Calcola l'alpha critico = pendenza iniziale della curva V(I).
    Se alpha < alpha_crit → INFO-SEEKING
    Se alpha > alpha_crit → DARK ROOM
    """
    ks = np.asarray(ks)
    vs = np.asarray(vs)
    if len(ks) < 2 or (ks[1] - ks[0]) < 1e-10:
        return float('inf')
    return float((vs[1] - vs[0]) / (ks[1] - ks[0]))
