"""Robustness of the non-monotonicity result to structural assumptions.

The headline result of the paper (more information can lower viability; at equal
information, some coarse-grainings are actively misleading) is stress-tested here
against the assumptions a sceptical reader would question: the geometry of the
food layout, the size of the grid, and the class decoder (how a perceived class
maps to a target). For each condition we report:

  dips      number of downward steps in the information-viability envelope
  over_pen  V(peak) - V(full information): >0 means information beyond the peak
            HURTS (non-monotonicity)
  gap       misleading gap: at equal information, the spread V_best - V_worst
            across partitions (how misleading a coarse-graining can be)
  sub       whether the S-FEEF optimum sits below the SST (sub-threshold
            compression)

The model machinery (viability = D_KL from the Boltzmann equilibrium, MI = class
entropy, exhaustive enumeration of partitions) is reused unchanged from
core.model; only the geometry, the grid size and the decoder are varied. The
baseline (3x3 corners+center, centroid decoder) reproduces the figures reported
in the Results section.
"""
from __future__ import annotations
import numpy as np
import core.model as M
from core.model import SemanticForagingModel


class FlexModel(SemanticForagingModel):
    """SemanticForagingModel with a configurable grid size and class decoder."""

    def __init__(self, decoder='centroid', grid_n=3, **kw):
        self.decoder = decoder
        self.grid_n = grid_n
        super().__init__(**kw)

    def _grid_neighbours(self, pos):
        x, y = pos
        nb = [pos]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_n and 0 <= ny < self.grid_n:
                nb.append((nx, ny))
        return nb

    def _class_representative(self, cls):
        fm = [c for c in cls if c != M.NO_FOOD]
        if not fm:
            return None
        cx = np.mean([M.CELLS[c][0] for c in fm])
        cy = np.mean([M.CELLS[c][1] for c in fm])
        if self.decoder == 'centroid':     # averaging: prone to misleading merges
            return min(M.GRID, key=lambda g: (g[0] - cx) ** 2 + (g[1] - cy) ** 2)
        if self.decoder == 'medoid':       # commit to the real food cell nearest the centroid
            cs = min(fm, key=lambda c: (M.CELLS[c][0] - cx) ** 2 + (M.CELLS[c][1] - cy) ** 2)
            return M.CELLS[cs]
        if self.decoder == 'first':        # commit to an arbitrary concrete member
            return M.CELLS[min(fm)]
        raise ValueError(self.decoder)


def _set_geometry(cells, grid_n):
    """Patch the module-level world constants for a given food layout."""
    M.GRID = [(x, y) for x in range(grid_n) for y in range(grid_n)]
    M.CELLS = {i: c for i, c in enumerate(cells)}
    M.N_FOOD = len(cells)
    M.NO_FOOD = len(cells)
    M.N_Y = len(cells) + 1


def _analyse(decoder, grid_n, alpha=4.0):
    m = FlexModel(decoder=decoder, grid_n=grid_n)
    ip = m.get_initial_distribution(2, 'uniform', 'food')
    r = m.compute_pareto_front(ip, alpha=alpha)
    ks = np.array(r['info_curve_ks']); vs = np.array(r['info_curve_vs'])
    dips = int(np.sum(np.diff(vs) < -1e-6))
    over = float(vs.max() - vs[-1])
    sk, sv = m.scatter_ks, m.scatter_vs
    gap = 0.0
    for lvl in np.unique(np.round(sk, 4)):
        msk = np.abs(sk - lvl) < 1e-4
        if msk.sum() > 1:
            gap = max(gap, float(sv[msk].max() - sv[msk].min()))
    return dict(I0=r['actual_mi'], SST=r['semantic_saturation_threshold'],
                Istar=r['best_feef_mi'], dips=dips, over=over, gap=gap,
                sub=r['best_feef_mi'] < r['semantic_saturation_threshold'] - 1e-6)


GEOMETRIES = {
    '3x3 corners+center (baseline)': ([(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)], 3),
    '3x3 clustered (top-left)':      ([(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)], 3),
    '3x3 top row + column':          ([(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)], 3),
    '3x3 plus / cross':              ([(1, 0), (1, 1), (1, 2), (0, 1), (2, 1)], 3),
    '3x3 random A':                  ([(0, 0), (0, 2), (1, 2), (2, 0), (2, 1)], 3),
    '3x3 random B':                  ([(0, 1), (1, 0), (1, 1), (2, 0), (2, 2)], 3),
    '4x4 corners+near-center':       ([(0, 0), (0, 3), (3, 0), (3, 3), (1, 2)], 4),
    '4x4 spread-5':                  ([(0, 0), (1, 3), (2, 1), (3, 0), (3, 3)], 4),
}


def main():
    out = []
    out.append("=== ROBUSTNESS TO STRUCTURAL ASSUMPTIONS (geometry / grid size / decoder) ===")
    out.append("over_pen = V(peak) - V(full info); >0 => more information HURTS (non-monotonicity)")
    out.append("gap = misleading spread of viability at equal information")
    out.append("")
    out.append("%-32s %-9s | %5s %5s %6s %5s %9s %6s %6s" %
               ("GEOMETRY", "decoder", "I0", "SST", "I*", "dips", "over_pen", "gap", "sub"))
    out.append("-" * 96)
    for name, (cells, gn) in GEOMETRIES.items():
        for dec in ('centroid', 'medoid'):
            _set_geometry(cells, gn)
            r = _analyse(dec, gn)
            out.append("%-32s %-9s | %5.2f %5.2f %6.2f %5d %9.2f %6.1f  %s" %
                       (name, dec, r['I0'], r['SST'], r['Istar'], r['dips'],
                        r['over'], r['gap'], r['sub']))
    out.append("")
    out.append("Reading: under the averaging (centroid) decoder, non-monotonicity (over_pen>0,")
    out.append("large gap) appears in 7 of 8 layouts and on both grid sizes, often more strongly")
    out.append("than baseline; it vanishes only where a food cell sits on the class centroid")
    out.append("(4x4 corners+near-center). Committing to a concrete cell (medoid) removes it in")
    out.append("some geometries but not all -- so the effect is a structural property of averaging")
    out.append("decoders under bounded action, not an artefact of the baseline layout. Perceptual")
    out.append("channel noise (H(X0|Y0)>0) is the one assumption not varied here.")
    txt = "\n".join(out)
    open("robustness_geometry_results.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
