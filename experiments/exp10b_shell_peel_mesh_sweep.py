"""Panel (C), lato GUSCIO: la peel di bordo libero e' SINGOLARE anche sull'espansione S8R.

Companion di exp10_peel_mesh_sweep.py. Quello fa la curva sul SOLIDO 3D
(`fe.interlaminar`, gia' parametrizzato in nx,ny,nz). Questo fa la curva ROSSA
del pannello (C) come e' realmente nella figura pubblicata: il point-max di |SZZ|
dall'ESPANSIONE DEL GUSCIO S8R, cioe' `peel_max` di
`optimisers.constrained_search.static_metrics` -- la quantita' che il docstring
di quel modulo chiama "an ESTIMATE ... junk", e che a 20x10 vale 45.06 MPa
(dentro il range dei punti rossi pubblicati [9.8 .. 62]).

Perche' esiste (RS-005, 2026-07-20): finche' la mesh del guscio stava nelle
costanti di modulo `NX, NY` (grid `_ID/_NODES/_ELEMS` congelato a import-time
mentre clamp/tip ricalcolavano dai runtime NX,NY), cambiarla faceva morire il
deck con "division by zero" (`tip` vuoto -> `sum/len(tip)`). Ora `make_ccx_deck`
accetta `mesh=(nx,ny[,lx,ly])` e ricostruisce l'INTERO grid coerente; mesh=None
resta byte-identical alla campagna pubblicata. Questo script sfrutta il fix.

Due canarini:
  (1) REGRESSIONE: il deck di default (mesh=None) deve riprodurre gli hash del
      deck pre-refactor (buckling e statico), altrimenti il fix ha cambiato la
      campagna e i numeri del paper non sono piu' quelli.
  (2) FIGURA: il point-max del guscio deve DIVERGERE col raffinamento (>3x),
      altrimenti il pannello (C) non dice piu' quel che la didascalia afferma.

Uso (SOLO su head, non sul laptop):
  ./code/run_on_head.sh experiments.exp10b_shell_peel_mesh_sweep
"""
from __future__ import annotations
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimisers.constrained_search import (  # noqa: E402
    make_ccx_deck, make_ccx_static_deck, static_metrics, CASES)

# --- caso fisico: stesso di exp10, cosi' guscio (rosso) e solido (verde) sono lo
#     STESSO carico valutato in due modi, non due esperimenti scollegati.
SEQ = [0, 45, -45, 90] * 6            # 24 ply, il default documentato del modulo
AXIAL, SIDE = -20000 * 0.44, 5000 * 0.44
SLOAD = dict(axial=AXIAL, side=SIDE, torsion=0.)
MESHES = [(10, 6), (20, 10), (30, 16), (40, 20)]     # stesse mesh di exp10

# --- canary di regressione: hash del deck di default catturati PRIMA del refactor
#     (seq=[0,45,-45,90]*6, case=CASES['c3_combo'], mesh=None). Vedi RS-005.
BASELINE = dict(
    seq=[0, 45, -45, 90] * 6, case='c3_combo',
    buck_sha='c32ee0f15094426ed558f91f755902e2ad04757b399dfe075bd2d7140819018a',
    stat_sha='da92cec3b6216616c9989239a01f1cba8207d9045447174df1000439e289cfe8')

OUT = Path(__file__).resolve().parents[1] / 'experiments' / '_out' / 'exp10b_shell_peel_mesh_sweep.json'


def _regression_canary() -> None:
    """mesh=None must reproduce the pre-refactor decks byte-for-byte."""
    seq, case = BASELINE['seq'], CASES[BASELINE['case']]
    buck = hashlib.sha256(make_ccx_deck(seq, case).encode()).hexdigest()
    stat = hashlib.sha256(make_ccx_static_deck(seq, case).encode()).hexdigest()
    if buck != BASELINE['buck_sha']:
        sys.exit(f"exp10b REGRESSION: default buckling deck changed\n  got {buck}\n  exp {BASELINE['buck_sha']}\n"
                 "the mesh refactor altered the PUBLISHED campaign -- do not trust downstream numbers")
    if stat != BASELINE['stat_sha']:
        sys.exit(f"exp10b REGRESSION: default static deck changed\n  got {stat}\n  exp {BASELINE['stat_sha']}")
    print('regression canary OK: default (mesh=None) deck is byte-identical to the pre-refactor campaign')


def main() -> None:
    _regression_canary()
    rows = []
    for nx, ny in MESHES:
        m = static_metrics((SEQ, SLOAD, (nx, ny)))          # <-- il 3o elemento e' la mesh
        pm = m.get('peel_max', float('inf'))     # su errore static_metrics ritorna inf SENZA la chiave
        if not (pm < float('inf')) or pm <= 0.0:
            sys.exit(f"exp10b: static_metrics fallito o degenere su {nx}x{ny}: {m}")
        rows.append({'nx': nx, 'ny': ny,
                     'peel_max': pm,                # point-max |SZZ| (rosso, singolare)
                     'peel_p95': m['peel'],          # 95mo percentile (peak mediato robusto)
                     'sx': m['sx'], 'disp': m['disp']})
        print(f"{nx:3d} x{ny:3d}  peel_max(point)={m['peel_max']:9.4g}  peel_p95={m['peel']:9.4g}  "
              f"|U|max={m['disp']:8.4g}")

    point = [r['peel_max'] for r in rows]
    spread = max(point) / min(point)
    print(f"\nshell point-max: fattore {spread:.1f}x fra mesh piu' grossa e piu' fine  -> NON converge")

    if spread < 3.0:
        sys.exit(f"exp10b: il point-max del GUSCIO NON diverge ({spread:.1f}x): il pannello (C) "
                 "non e' sostenuto sul lato guscio, non disegnarlo cosi'")

    result = {'sequence': SEQ, 'axial': AXIAL, 'side': SIDE,
              'ccx': os.environ.get('CCX_BIN', 'ccx_2.21'),
              'evaluator': 'shell S8R expansion peel_max (optimisers.constrained_search.static_metrics)',
              'baseline_regression': BASELINE, 'spread_point_max': spread, 'rows': rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))
    print(f"\nscritto {OUT}")
    print("=== JSON-STDOUT-BEGIN ===")
    print(json.dumps(result))
    print("=== JSON-STDOUT-END ===")


if __name__ == '__main__':
    main()
