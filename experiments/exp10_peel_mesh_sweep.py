"""Panel (C) della figura dei pitfall: la peel di bordo libero e' SINGOLARE.

Ricostruisce i dati del pannello (C), che nel repo non erano mai stati salvati
(RS-005). Le due curve del pannello sono DUE VALUTATORI DIVERSI, non due
post-processing dello stesso campo:

  rosso  "naive point-max"  -> massimo puntuale di |SZZ| dall'ESPANSIONE DEL
                               GUSCIO S8R, cioe' `peel_max` di
                               `optimisers.constrained_search.static_metrics`.
                               E' la quantita' che il docstring di quel modulo
                               chiama esplicitamente "an ESTIMATE ... junk".
  verde  "averaged"         -> criterio di Whitney-Nuismer mediato sul SOLIDO 3D,
                               cioe' `Q`/`peel` di `fe.interlaminar.interlaminar`.

Il punto della figura: il rosso NON converge sotto raffinamento (singolarita' di
Pipes-Pagano), il verde si'.

STATO (2026-07-20): questo script fa il lato SOLIDO. Il lato GUSCIO (curva rossa,
`peel_max` di static_metrics) e' ora in exp10b_shell_peel_mesh_sweep.py: il freeze
della mesh in `constrained_search` -- NX,NY costanti di modulo, deck statico che
moriva con "division by zero" oltre 20x10 -- e' stato sciolto (make_ccx_deck accetta
mesh=(nx,ny), default byte-identical). Il point-max del guscio diverge 4.5x
(10x6 .. 40x20), come il solido qui.

Uso:  CCX_BIN=ccx_2.21 PYTHONPATH=$PWD python3 experiments/exp10_peel_mesh_sweep.py
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fe.interlaminar import interlaminar  # noqa: E402

SEQ = [0, 45, -45, 90] * 6          # 24 ply, il default documentato del modulo
MESHES = [(10, 6), (20, 10), (30, 16), (40, 20)]
OUT = Path(__file__).resolve().parents[1] / 'data' / 'exp10_peel_mesh_sweep.json'

# DUE casi di carico, e la ragione per cui non basta il primo (2026-08-26).
#
# `demo_compression` e' il caso storico di questo modulo (il vecchio fattore 0,44 applicato ai
# carichi di riferimento): e' in COMPRESSIONE. Col parser .frd corretto -- prima scartava in
# silenzio ogni riga con sigma_xx negativo, cioe' quasi tutte, sotto compressione -- si scopre
# che li' il peel mediato e' **identicamente zero a ogni mesh**: la media di banda della sigma_zz
# e' presa in parte positiva (<sigma_zz>+, per definizione del criterio) e sotto quel carico e'
# ovunque compressiva. Il termine di peel del criterio si spegne e Q resta retto dai due termini
# di taglio interlaminare. Il contrasto "massimo puntuale che diverge contro media che no" li'
# non e' dimostrabile sul peel: la media non e' limitata, e' assente.
#
# `campaign_S1` e' invece il carico che la campagna usa davvero (STATIC_LOAD del modulo di
# ricerca, in trazione), e li' il contrasto esiste e si misura sul peel, che e' cio' che il paper
# afferma. Si sweepano entrambi: il primo perche' e' la storia di questo esperimento e va tenuta,
# il secondo perche' e' quello che sostiene il paragrafo del manoscritto.
from optimisers.constrained_search import STATIC_LOAD  # noqa: E402

LOAD_CASES = {
    'demo_compression': {'axial': -20000 * 0.44, 'side': 5000 * 0.44},
    'campaign_S1': {'axial': STATIC_LOAD['axial'], 'side': STATIC_LOAD['side']},
}
#: Il caso su cui poggia il pannello (C) della figura e la frase del manoscritto.
FIGURE_CASE = 'campaign_S1'


def sweep(case_name: str, axial: float, side: float) -> list:
    """Uno sweep di mesh su un caso di carico. Torna le righe, non le stampa e basta."""
    rows = []
    print(f"\n=== {case_name}: axial={axial:g}, side={side:g} ===")
    for nx, ny in MESHES:
        r = interlaminar(SEQ, axial=axial, side=side, nx=nx, ny=ny)
        if 'error' in r:
            sys.exit(f'exp10: ccx fallito su {case_name} {nx}x{ny}: {r}')
        rows.append({'nx': nx, 'ny': ny, 'nz': r['nz'],
                     'peel_point': r['peel_point'], 'peel_avg': r['peel'],
                     'Q': r['Q'], 'bc_resid': r['bc_resid']})
        print(f"{nx:3d} x{ny:3d}  point={r['peel_point']:9.4g}  avg={r['peel']:9.4g}  "
              f"Q={r['Q']:9.3g}  bc_resid={r['bc_resid']:7.3g}")
    return rows


def _factor(vals) -> float:
    """max/min, che e' la convenzione con cui il manoscritto riporta i fattori dello sweep."""
    lo = min(vals)
    return float('inf') if lo == 0 else max(vals) / lo


def main() -> None:
    cases = {}
    for name, ld in LOAD_CASES.items():
        rows = sweep(name, ld['axial'], ld['side'])
        point_f = _factor([r['peel_point'] for r in rows])
        avg_f = _factor([r['peel_avg'] for r in rows])
        q_f = _factor([r['Q'] for r in rows])
        print(f"  fattori (max/min): point {point_f:.1f}x | "
              f"peel medio {'assente (identicamente 0)' if avg_f == float('inf') else f'{avg_f:.1f}x'}"
              f" | Q {q_f:.2f}x {'crescente' if rows[-1]['Q'] > rows[0]['Q'] else 'decrescente'}")
        bc = [r['bc_resid'] for r in rows]
        if bc[-1] > bc[0]:
            # Il residuo della condizione di superficie libera sulla tau recuperata dovrebbe
            # essere ~0 e CALARE col raffinamento. Sale: il recupero per equilibrio si degrada
            # sulle mesh fini. Non e' un effetto della correzione del parser -- e' un problema
            # del recupero che la correzione ha smesso di mascherare, perche' un campo in gran
            # parte azzerato rispettava benissimo una condizione di superficie libera.
            print(f"  ATTENZIONE: bc_resid cresce col raffinamento ({bc[0]:.3g} -> {bc[-1]:.3g} "
                  f"MPa): il recupero per equilibrio si degrada sulle mesh fini.")
        cases[name] = {'axial': ld['axial'], 'side': ld['side'], 'rows': rows,
                       'point_factor': point_f, 'peel_avg_factor': None if avg_f == float('inf') else avg_f,
                       'Q_factor': q_f, 'peel_avg_absent': avg_f == float('inf')}

    # CANARINI, sul caso che sostiene la figura e la frase del manoscritto.
    fig = cases[FIGURE_CASE]
    if fig['point_factor'] < 3.0:
        sys.exit(f"exp10: su {FIGURE_CASE} il point-max NON diverge "
                 f"({fig['point_factor']:.1f}x): il pannello (C) non e' piu' sostenuto dai dati")
    if fig['Q_factor'] > 3.0:
        sys.exit(f"exp10: su {FIGURE_CASE} il criterio mediato varia troppo "
                 f"({fig['Q_factor']:.1f}x): la meta' verde del pannello (C) non regge")
    if fig['peel_avg_absent']:
        sys.exit(f"exp10: su {FIGURE_CASE} il peel mediato e' identicamente zero: il contrasto "
                 f"peel-contro-peel non e' dimostrabile su questo caso di carico")

    OUT.write_text(json.dumps(
        {'sequence': SEQ, 'meshes': MESHES, 'figure_case': FIGURE_CASE,
         'ccx': __import__('fe.ccx_bin', fromlist=['resolved']).resolved(),
         'note': 'Due casi di carico dal 26/08/2026: demo_compression e\' quello storico di '
                 'questo modulo, campaign_S1 e\' il carico della campagna ed e\' quello su cui '
                 'poggiano il pannello (C) e la frase del manoscritto. Sotto compressione il '
                 'peel mediato e\' identicamente zero (la media di banda e\' presa in parte '
                 'positiva e li\' la sigma_zz media e\' compressiva), quindi il contrasto '
                 'peel-contro-peel non e\' dimostrabile su quel caso.',
         # compatibilita': i consumatori storici leggevano 'rows' e 'axial'/'side'
         'axial': cases[FIGURE_CASE]['axial'], 'side': cases[FIGURE_CASE]['side'],
         'rows': cases[FIGURE_CASE]['rows'],
         'cases': cases}, indent=1) + '\n')
    print(f'\nscritto {OUT} (figura e manoscritto: caso {FIGURE_CASE})')


if __name__ == '__main__':
    main()
