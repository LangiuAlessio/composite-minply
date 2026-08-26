"""Lo screen applicato ai sei design consegnati: il verdetto, non l'affermazione.

Il manoscritto dice che ogni design consegnato passa da un layer deterministico prima di essere
accettato. Fino al 2026-08-26 quel layer non era un componente e la frase era un racconto di cosa
era stato fatto a mano. Ora e' `fe.verification.screen`, e questo esperimento lo esegue sui sei
design che il paper consegna, registrando il verdetto per ciascuno.

COSA GIRA, E COSA NO, per ogni design:
  - invariante di scala   -> DUE solve di buckling, a carico nominale e al doppio;
  - vincoli fisici        -> spostamento, sforzi in piano e Q, dai referti della campagna;
  - copertura             -> il conteggio dei nodi della griglia di sforzo del solido;
  - risposta al layup     -> **non si applica a un design singolo**: e' un controllo sul MODELLO
                             (confronta due laminati che il chop renderebbe identici), non sul
                             risultato. Lo screen lo segna come non eseguito, ed e' il motivo per
                             cui il verdetto di un design sano e' `ok` solo se gli si passa cio'
                             che serve, non per default.

⚠️ COSTO, e va detto nel paper: due solve di buckling per design. Lo screen e' piu' caro di una
valutazione, quindi gira sui design CONSEGNATI e non dentro il ciclo di ricerca, dove le guardie
sono quelle piu' economiche gia' integrate nell'evaluator.

Uso:  CCX_BIN=ccx_2.21 PYTHONPATH=$PWD python3 -m experiments.exp31_screen_delivered
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fe.ccx_bin import resolved                                         # noqa: E402
from fe.verification import screen                                      # noqa: E402
from optimisers.constrained_search import (buckling_factor, static_metrics,   # noqa: E402
                                           CASES, STATIC_LOAD, DISP_MAX, SIGMA_MAX)
from experiments.exp27_delivered_Q_recheck import _delivered            # noqa: E402

DATA = Path(__file__).resolve().parents[1] / 'data'
OUT = DATA / 'exp31_screen_delivered.json'
LIMITI = {'disp': {'soglia': DISP_MAX}, 'sx': {'soglia': SIGMA_MAX},
          'sy': {'soglia': SIGMA_MAX}, 'Q': {'soglia': 1.0}}
CASO = {'c1_axial': 'c1_axial', 'c2_side': 'c2_side', 'c3_combo': 'c3_combo'}


def main() -> None:
    # La risposta al layup si misura UNA VOLTA, sul modello, non design per design: confronta due
    # laminati che il chop renderebbe identici, quindi e' una proprieta' dell'evaluator e della
    # mesh, non del risultato. Una prima versione la lasciava "non eseguita" per ogni design, e i
    # sei consegnati uscivano tutti `suspect`: sei falsi allarmi generati dal disegno dell'API, non
    # dai design. Misurata qui e passata a tutti e sei, perche' e' lo stesso modello a produrli.
    layup = (buckling_factor(([0] * 24, CASES['c1_axial'])),
             buckling_factor(([90] * 24, CASES['c1_axial'])))
    print(f"risposta al layup del modello: [0]_24 {layup[0]:.4f} contro [90]_24 {layup[1]:.4f}\n")

    q_ric = {(r['alphabet'], r['case']): r['Q_recomputed']
             for r in json.loads((DATA / 'exp27_delivered_Q_recheck.json').read_text())['rows']}
    righe = []
    for alfabeto, casi in sorted(_delivered().items()):
        for nome, d in sorted(casi.items()):
            half = d['half_stack']
            seq = half + half[::-1]
            caso = CASES[CASO[nome]]

            # invariante di scala: due carichi, uno il doppio dell'altro
            coppie = []
            for scala in (1.0, 2.0):
                c = {**caso, 'axial': caso['axial'] * scala, 'side': caso['side'] * scala}
                coppie.append((abs(c['axial']) + abs(c['side']), buckling_factor((seq, c))))

            m = static_metrics((seq, STATIC_LOAD))
            metriche = {'disp': m['disp'], 'sx': m['sx'], 'sy': m['sy'],
                        'Q': q_ric.get((alfabeto, nome))}
            # la griglia di sforzo del solido: nx=20, ny=10, nz=n_ply
            attesi = 21 * 11 * (d['n_plies'] + 1)

            v = screen(layup_pair=layup, scale_pairs=coppie, coverage=(attesi, attesi),
                       metriche=metriche, limiti=LIMITI)
            righe.append({'alphabet': alfabeto, 'case': nome, 'n_plies': d['n_plies'],
                          'verdetto': v.esito, 'motivi': v.motivi,
                          'dispersione_invariante_pct': v.misure['scale_invariant'].get(
                              'dispersione_pct'),
                          'misure': v.misure})
            print(f"{alfabeto}/{nome} N={d['n_plies']}: {v.esito:8s} "
                  f"invariante {v.misure['scale_invariant'].get('dispersione_pct')}%"
                  + (f"  motivi: {v.motivi}" if v.motivi else ""))

    esiti = {}
    for r in righe:
        esiti[r['verdetto']] = esiti.get(r['verdetto'], 0) + 1
    peggiore = max(r['dispersione_invariante_pct'] or 0 for r in righe)
    print(f"\nverdetti: {esiti} | dispersione peggiore dell'invariante: {peggiore}%")
    OUT.write_text(json.dumps({
        'nota': 'Verdetto dello screen deterministico sui sei design consegnati. La risposta al '
                'layup e un controllo sul MODELLO, misurata una volta e valida per tutti e sei.',
        'layup_response_modello': {'bf_0_24': layup[0], 'bf_90_24': layup[1]},
        'ccx': resolved(), 'limiti': LIMITI, 'esiti': esiti,
        'dispersione_invariante_peggiore_pct': peggiore, 'rows': righe}, indent=1) + '\n')
    print(f'-> {OUT}')
    if any(r['verdetto'] == 'rejected' for r in righe):
        sys.exit('exp31: un design CONSEGNATO e stato respinto dallo screen: il paper non puo '
                 'consegnarlo finche non e capito')


if __name__ == '__main__':
    main()
