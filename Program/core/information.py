"""
Information-theory functions for the S-FEEF framework.
Implements semantic-information measures, entropy and MI.
"""

import numpy as np


def entropy(p: np.ndarray, base: float = 2.0) -> float:
    """Shannon entropy H(X) in bits (or nats if base=e)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p) / np.log(base)))


def mutual_information(p_joint: np.ndarray, base: float = 2.0) -> float:
    """
    Mutual information I(X;Y) from the joint distribution p(x,y).
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
    """KL divergence D_KL(P || Q)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = (p > 0) & (q > 0)
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask]) / np.log(base)))


def conditional_entropy(p_joint: np.ndarray, base: float = 2.0) -> float:
    """Conditional entropy H(Y|X) from the joint distribution."""
    p_joint = np.asarray(p_joint, dtype=float)
    p_joint = p_joint / (p_joint.sum() + 1e-12)
    h_xy = entropy(p_joint.ravel(), base)
    h_x = entropy(p_joint.sum(axis=1), base)
    return max(0.0, h_xy - h_x)


def semantic_saturation_threshold(ks: np.ndarray, vs: np.ndarray,
                                   frac: float = 0.05) -> float:
    """
    Semantic Saturation Threshold (SST) -- MARGINAL-RETURN definition.

    On the cumulative-best (monotone) frontier V*(I) = max_{I' <= I} V(I'), the
    SST is the largest I whose marginal viability return per bit exceeds
    a floor = frac * alpha_c, where alpha_c is the maximum marginal
    return from the null policy. Beyond the SST the return has essentially vanished.

    This definition is far more robust than the old "smallest I that
    reaches V_max within a tolerance" on a NON-monotone frontier: the
    value does not jump to a secondary peak, and is stable for frac between
    ~1% and ~15% (see robustness_analysis.py).
    """
    ks = np.asarray(ks, dtype=float)
    vs = np.asarray(vs, dtype=float)
    order = np.argsort(ks)
    ks, vs = ks[order], vs[order]
    if len(ks) < 2 or (ks[-1] - ks[0]) < 1e-12:
        return float(ks[-1])
    vbest = np.maximum.accumulate(vs)                 # cumulative-best frontier
    if vbest[-1] - vbest[0] < 1e-10:                   # flat viability -> no useful info
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
    Kolchinsky & Wolpert Semantic Information:
    SI_KW = V_regulated - V_intrinsic
    Measures how much information is causally relevant to viability.
    """
    return float(v_regulated - v_intrinsic)


def pareto_front_feef(ks: np.ndarray, vs: np.ndarray,
                       alpha: float) -> tuple[np.ndarray, int, float, float]:
    """
    Computes the S-FEEF curve and the optimal point.

    S-FEEF(I) = alpha * I - V(I)

    Returns:
        fs: S-FEEF curve
        best_idx: index of the minimum
        best_mi: optimal MI I*
        best_score: minimum S-FEEF score
    """
    ks = np.asarray(ks, dtype=float)
    vs = np.asarray(vs, dtype=float)
    fs = alpha * ks - vs
    best_idx = int(np.argmin(fs))
    return fs, best_idx, float(ks[best_idx]), float(fs[best_idx])


def critical_alpha(ks: np.ndarray, vs: np.ndarray) -> float:
    """
    Computes the critical alpha = initial slope of the V(I) curve.
    If alpha < alpha_crit → INFO-SEEKING
    If alpha > alpha_crit → DARK ROOM
    """
    ks = np.asarray(ks)
    vs = np.asarray(vs)
    if len(ks) < 2 or (ks[1] - ks[0]) < 1e-10:
        return float('inf')
    return float((vs[1] - vs[0]) / (ks[1] - ks[0]))
