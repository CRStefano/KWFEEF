"""SemanticForagingModel - S-FEEF framework (revision C, 2-D environment).

Two-dimensional environment: 5 food cells at the 4 corners and the centre of a
3x3 grid (9 agent positions). The perceptual policies are the B_6 = 203
partitions of Y = {5 cells} U {absent}. Under a partition that merges
spatially opposite cells, the perceived centroid lands on a WRONG cell: the
coarse perception becomes actively MISLEADING, and viability V(I) is non-
monotone (excess information can reduce viability).

Everything else from revision B remains: viability Eq. (2), causal coupling
via logical_target, a Metropolis thermodynamic substrate satisfying Local
Detailed Balance to machine precision.
"""
from __future__ import annotations
import numpy as np
from . import information as info_mod

# 5 food cells: 4 corners + centre of a 3x3 grid
CELLS = {0: (0, 0), 1: (0, 2), 2: (1, 1), 3: (2, 0), 4: (2, 2)}
GRID = [(x, y) for x in range(3) for y in range(3)]   # 9 agent positions
N_FOOD = 5
NO_FOOD = N_FOOD          # index 5 = no food
N_Y = N_FOOD + 1          # |Y| = 6


def set_partitions(n: int):
    """All partitions of {0,...,n-1} (Bell number B_n; B_6 = 203)."""
    if n <= 0:
        yield []
        return

    def helper(elements):
        if len(elements) == 1:
            yield [[elements[0]]]
            return
        first_el = elements[0]
        for rest in helper(elements[1:]):
            yield [[first_el]] + rest
            for i in range(len(rest)):
                yield rest[:i] + [[first_el] + rest[i]] + rest[i + 1:]

    yield from helper(list(range(n)))


def _manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class SemanticForagingModel:
    def __init__(self, mode='pos', timescale=5, food_disappear_rate=0.1,
                 eat_radius=1, dead_entropy=100.0, agentlevel_fe=10.0):
        self.mode = mode
        self.timescale = int(timescale)
        self.food_disappear_rate = float(food_disappear_rate)
        self.eat_radius = int(eat_radius)
        self.lam = float(dead_entropy)
        self.agentlevel_fe = float(agentlevel_fe)
        self.base_hazard = float(np.clip(1.0 / self.agentlevel_fe, 0.0, 0.95))
        self._build_state_space()
        self._assign_free_energies()
        self._T_base = self._build_transition(self._trivial_partition())
        self.timeevolvedmx = np.linalg.matrix_power(self._T_base, self.timescale)

    # ------------------------------------------------------------------
    def _build_state_space(self):
        self.states = []
        self.state2id = {}
        self.id2state_dict = {}
        idx = 0
        for pos in GRID:
            for f in range(N_Y):
                for v in (0, 1):
                    s = (pos, f, v)
                    self.states.append(s)
                    self.state2id[s] = idx
                    self.id2state_dict[idx] = s
                    idx += 1
        self.num_states = len(self.states)

    def _assign_free_energies(self):
        gamma_alive, gamma_food = 10.0, 2.0
        gamma_dead = -np.log(2.0) * self.lam / 10.0
        self.F = np.zeros(self.num_states)
        for idx, (pos, f, v) in enumerate(self.states):
            F = gamma_alive if v == 1 else gamma_dead
            if f != NO_FOOD:
                F += gamma_food
            self.F[idx] = F
        # Boltzmann distribution (thermal equilibrium) pi_k ~ exp(-F_k):
        # the equilibrium is dominated by dead states (low F). Viability is
        # the distance from this equilibrium (see _viability).
        w = np.exp(-self.F - np.max(-self.F))
        self._pi_eq = w / w.sum()

    def _grid_neighbours(self, pos):
        x, y = pos
        nb = [pos]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                nb.append((nx, ny))
        return nb

    def equilibrium_reference_matrix(self):
        """Metropolis chain on F: thermodynamic substrate (exact LDB)."""
        n = self.num_states
        M = np.zeros((n, n))
        neigh = [[] for _ in range(n)]
        for i, (pos, f, v) in enumerate(self.states):
            cand = []
            for p2 in self._grid_neighbours(pos):
                if p2 != pos:
                    cand.append((p2, f, v))
            for f2 in range(N_Y):
                if f2 != f:
                    cand.append((pos, f2, v))
            cand.append((pos, f, 1 - v))
            for s2 in cand:
                neigh[i].append(self.state2id[s2])
        n_prop = max(len(x) for x in neigh)
        for i in range(n):
            for j in neigh[i]:
                M[i, j] += min(1.0, np.exp(self.F[i] - self.F[j])) / n_prop
            M[i, i] += 1.0 - M[i].sum()
        return M

    def ldb_max_error(self):
        M = self.equilibrium_reference_matrix()
        rel = 0.0
        n = self.num_states
        for i in range(n):
            for j in range(n):
                if M[i, j] > 1e-12 and M[j, i] > 1e-12:
                    ratio = (M[i, j] / M[j, i]) / np.exp(self.F[i] - self.F[j])
                    rel = max(rel, abs(ratio - 1.0))
        return float(rel)

    # ------------------------------------------------------------------
    @staticmethod
    def _trivial_partition():
        return [list(range(N_Y))]

    @staticmethod
    def _identity_partition():
        return [[y] for y in range(N_Y)]

    def _class_representative(self, cls):
        """Grid cell nearest to the spatial centroid of the class.
        NO protection: a class merging opposite cells produces a
        centroid on a wrong cell (misleading perception)."""
        fm = [c for c in cls if c != NO_FOOD]
        if not fm:
            return None
        cx = np.mean([CELLS[c][0] for c in fm])
        cy = np.mean([CELLS[c][1] for c in fm])
        return min(GRID, key=lambda g: (g[0] - cx) ** 2 + (g[1] - cy) ** 2)

    def _perception_map(self, partition):
        if not getattr(self, '_coupled', True):
            return {y: None for y in range(N_Y)}
        rep = {}
        for cls in partition:
            r = self._class_representative(cls)
            for y in cls:
                rep[y] = r
        return rep

    def _step(self, pos, tgt):
        """One Manhattan step toward (mode pos) or away from (neg) the target.
        mode neutral or absent target -> stay (random walk handled upstream)."""
        if tgt is None:
            return pos
        if self.mode == 'neg':
            best = pos
            bestd = _manh(pos, tgt)
            for p2 in self._grid_neighbours(pos):
                if _manh(p2, tgt) > bestd:
                    bestd = _manh(p2, tgt)
                    best = p2
            return best
        # mode pos: avvicinati di un passo
        if tgt == pos:
            return pos
        x, y = pos
        if abs(tgt[0] - x) >= abs(tgt[1] - y) and tgt[0] != x:
            return (x + (1 if tgt[0] > x else -1), y)
        if tgt[1] != y:
            return (x, y + (1 if tgt[1] > y else -1))
        return (x + (1 if tgt[0] > x else -1), y)

    def _build_transition(self, partition, normalise=True):
        rep = self._perception_map(partition)
        T = np.zeros((self.num_states, self.num_states))
        neutral = (self.mode == 'neutral')
        for si, (pos, f, v) in enumerate(self.states):
            if v == 0:
                T[si, si] = 1.0
                continue
            tgt = rep[f] if f != NO_FOOD else None
            if neutral or tgt is None:
                moves = self._grid_neighbours(pos)
                move_probs = {m: 1.0 / len(moves) for m in moves}
            else:
                move_probs = {self._step(pos, tgt): 1.0}
            for a2, pmove in move_probs.items():
                ate = (f != NO_FOOD) and (_manh(a2, CELLS[f]) <= self.eat_radius)
                if ate:
                    for fn in range(N_FOOD):
                        T[si, self.state2id[(a2, fn, 1)]] += pmove / N_FOOD
                else:
                    hz = self.base_hazard
                    if f == NO_FOOD:
                        T[si, self.state2id[(a2, NO_FOOD, 1)]] += pmove * (1 - hz)
                        T[si, self.state2id[(a2, NO_FOOD, 0)]] += pmove * hz
                    else:
                        pd = self.food_disappear_rate
                        T[si, self.state2id[(a2, f, 1)]] += pmove * (1 - pd) * (1 - hz)
                        T[si, self.state2id[(a2, f, 0)]] += pmove * (1 - pd) * hz
                        T[si, self.state2id[(a2, NO_FOOD, 1)]] += pmove * pd * (1 - hz)
                        T[si, self.state2id[(a2, NO_FOOD, 0)]] += pmove * pd * hz
        if normalise:
            rs = T.sum(axis=1, keepdims=True)
            rs = np.where(rs < 1e-12, 1.0, rs)
            T = T / rs
        return T

    # ------------------------------------------------------------------
    def get_initial_distribution(self, agentinitloc=2, initial_foodloc='uniform',
                                 logical_target='food'):
        p = np.zeros(self.num_states)
        start = CELLS.get(agentinitloc, (1, 1))
        if initial_foodloc == 'uniform':
            food_dist = {f: 1.0 / N_FOOD for f in range(N_FOOD)}
        elif isinstance(initial_foodloc, int) and 0 <= initial_foodloc < N_FOOD:
            food_dist = {initial_foodloc: 1.0}
        else:
            food_dist = {f: 1.0 / N_FOOD for f in range(N_FOOD)}
        for f, fp in food_dist.items():
            p[self.state2id[(start, f, 1)]] += fp
        self._p_R = np.zeros(N_Y)
        for f, fp in food_dist.items():
            self._p_R[f] = fp
        self._logical_target = str(logical_target)
        self._coupled = (self._logical_target == 'food') and (self.mode != 'neutral')
        return p

    def _mi_partition(self, partition):
        if not getattr(self, '_coupled', True):
            return 0.0
        p_R = self._p_R
        cp = np.array([s for s in (sum(p_R[y] for y in cls) for cls in partition) if s > 0])
        return info_mod.entropy(cp)

    def _viability(self, p_tau):
        # Viability = non-equilibrium free energy = D_KL(p_tau || pi_eq) in
        # bits: distance from thermal equilibrium (Boltzmann), i.e. the
        # Kolchinsky-Wolpert notion of viability, DERIVED from the free energies F
        # rather than postulated. No separate lambda penalty parameter:
        # the depth of the dead state is already inside pi_eq.
        p = np.asarray(p_tau, dtype=float)
        mask = p > 1e-15
        return float(np.sum(p[mask] * np.log2(p[mask] / self._pi_eq[mask])))

    def _p_dead(self, p_tau):
        return float(sum(p_tau[i] for i, (pos, f, v) in enumerate(self.states) if v == 0))

    # ------------------------------------------------------------------
    def compute_pareto_front(self, initp, alpha=9.5, n_points=None):
        partitions = list(set_partitions(N_Y))
        ks_raw, vs_raw = [], []
        for part in partitions:
            T_tau = np.linalg.matrix_power(self._build_transition(part), self.timescale)
            p_tau = initp @ T_tau
            ks_raw.append(self._mi_partition(part))
            vs_raw.append(self._viability(p_tau))
        ks_raw = np.array(ks_raw)
        vs_raw = np.array(vs_raw)
        # full scatter kept for the non-monotonicity figure
        self.scatter_ks = ks_raw.copy()
        self.scatter_vs = vs_raw.copy()

        order = np.argsort(ks_raw)
        ks_sorted, vs_sorted = ks_raw[order], vs_raw[order]
        ks_u, vs_u = [], []
        i, nptot = 0, len(ks_sorted)
        while i < nptot:
            k0 = ks_sorted[i]
            j, vmax = i, vs_sorted[i]
            while j < nptot and (ks_sorted[j] - k0) <= 1e-9:
                vmax = max(vmax, vs_sorted[j])
                j += 1
            ks_u.append(k0)
            vs_u.append(vmax)
            i = j
        ks_u, vs_u = np.array(ks_u), np.array(vs_u)
        actual_mi = float(ks_u[-1])
        sst = info_mod.semantic_saturation_threshold(ks_u, vs_u, frac=0.05)
        v_reg = float(self._viability(initp @ np.linalg.matrix_power(
            self._build_transition(self._identity_partition()), self.timescale)))
        v_int = float(self._viability(initp @ np.linalg.matrix_power(
            self._build_transition(self._trivial_partition()), self.timescale)))
        kw_si = info_mod.kw_semantic_information(v_int, v_reg)
        sst_efficiency = float(sst / actual_mi) if actual_mi > 1e-10 else 0.0
        fs, best_idx, best_feef_mi, best_feef_score = info_mod.pareto_front_feef(ks_u, vs_u, alpha)
        alpha_crit = self._critical_alpha(ks_u, vs_u)
        return {
            'actual_mi': actual_mi,
            'semantic_saturation_threshold': sst,
            'sst_efficiency': sst_efficiency,
            'kw_semantic_information': kw_si,
            'best_feef_mi': best_feef_mi,
            'best_feef_score': best_feef_score,
            'alpha_crit': alpha_crit,
            'info_curve_ks': ks_u.tolist(),
            'info_curve_vs': vs_u.tolist(),
            'feef_curve_fs': fs.tolist(),
        }

    def _critical_alpha(self, ks, vs):
        if len(ks) < 2:
            return float('inf')
        v0, k0 = vs[0], ks[0]
        best = -np.inf
        for k, v in zip(ks[1:], vs[1:]):
            if k - k0 > 1e-9:
                best = max(best, (v - v0) / (k - k0))
        return float(best) if best > 0 else float('inf')
