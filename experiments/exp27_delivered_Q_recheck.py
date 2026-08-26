"""Rimisura il Q di delaminazione dei sei design consegnati, col parser .frd corretto.

Perche' esiste (2026-08-26). `fe/interlaminar._parse_stress_grid` estraeva il numero del nodo
con `split()` su un formato a colonne fisse: ogni riga il cui PRIMO valore era negativo (cioe'
sigma_xx in compressione) veniva scartata in silenzio e la cella restava 0.0. Misurato: nella
configurazione di exp10 il campo era riempito al 15,9%, e il parser vecchio riproduce il numero
pubblicato a dieci cifre significative -- quindi i Q consegnati sono stati calcolati su un campo
in gran parte azzerato.

Questo modulo NON rifa' la ricerca: prende le sequenze gia' consegnate da
data/exp3_minply_sequences.json e ricalcola SOLO il loro Q, con gli stessi carichi di exp3
(STATIC_LOAD) e la stessa mesh (nx=20, ny=10). E' il modo economico di sapere se il vincolo di
delaminazione regge ancora: la ricerca non va rifatta se il verdetto non cambia.

Uso:  CCX_BIN=/usr/bin/ccx PYTHONPATH=$PWD python3 -m experiments.exp27_delivered_Q_recheck
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimisers.constrained_search import STATIC_LOAD            # noqa: E402
from fe.interlaminar import interlaminar                         # noqa: E402

DATA = Path(__file__).resolve().parents[1] / 'data'
SRC = DATA / 'exp3_minply_sequences.json'
SRC_C1 = DATA / 'exp3b_c1_freq_constrained.json'
OUT = DATA / 'exp27_delivered_Q_recheck.json'


def _delivered() -> dict:
    """I sei design DAVVERO consegnati nel paper, che non stanno tutti nello stesso file.

    ⚠️ Trappola pagata il 26/08/2026, alla prima stesura di questo modulo: il blocco
    `delivered` di exp3 da' C1 a **46** ply, ma il C1 del paper e' quello a **48** ply, che
    viene da exp3b (la ripetizione col vincolo di frequenza imposto: a 46 ply la frequenza non
    passa). Leggendo solo exp3 si ricontrollavano quindi due design che il paper NON consegna,
    cioe' 2 righe su 6 sbagliate, e nel modo peggiore: numeri plausibili, dello stesso ordine di
    grandezza, sotto l'etichetta giusta.
    """
    out = {}
    e3 = json.loads(SRC.read_text())['delivered']
    for alpha_name, cases in e3.items():
        out[alpha_name] = {k: v for k, v in cases.items() if k != 'c1_axial'}
    e3b = json.loads(SRC_C1.read_text())
    assert e3b['case'] == 'c1_axial', e3b['case']
    for alpha_name, v in e3b['designs'].items():
        ch = dict(v['chosen'])
        ch['n_plies'] = e3b['n_plies']
        out.setdefault(alpha_name, {})['c1_axial'] = ch
    return out


def main() -> None:
    delivered = _delivered()
    rows = []
    for alpha_name, cases in delivered.items():
        for case_name, d in sorted(cases.items()):
            half = d['half_stack']
            seq = half + half[::-1]
            assert len(seq) == d['n_plies'], (len(seq), d['n_plies'])
            r = interlaminar(seq, axial=STATIC_LOAD['axial'], side=STATIC_LOAD['side'],
                             nx=20, ny=10)
            if 'error' in r:
                print(f"{alpha_name}/{case_name}: SOLVE FALLITA {r}", flush=True)
                rows.append({'alphabet': alpha_name, 'case': case_name, 'error': r.get('error')})
                continue
            q_old, q_new = d.get('Q'), r['Q']
            ratio = (q_new / q_old) if q_old else None
            rows.append({'alphabet': alpha_name, 'case': case_name, 'n_plies': d['n_plies'],
                         'Q_published': q_old, 'Q_recomputed': q_new,
                         'ratio': ratio, 'peel_avg': r['peel'], 'peel_point': r['peel_point'],
                         'still_feasible': bool(q_new < 1.0)})
            print(f"{alpha_name}/{case_name} N={d['n_plies']}: Q {q_old} -> {q_new:.6g}"
                  f"  (x{ratio:.2f})  feasible={q_new < 1.0}" if ratio else
                  f"{alpha_name}/{case_name}: Q -> {q_new:.6g}", flush=True)
    verdict = all(r.get('still_feasible') for r in rows if 'Q_recomputed' in r)
    OUT.write_text(json.dumps({
        'note': 'Q ricalcolato col parser .frd corretto (audit F1, 2026-08-26); la ricerca '
                'NON e stata rifatta, solo il vincolo dei sei design consegnati. C1 viene da '
                'exp3b (48 ply, vincolo di frequenza imposto), non dai 46 ply di exp3.',
        'loads': STATIC_LOAD, 'mesh': {'nx': 20, 'ny': 10},
        'ccx': __import__('fe.ccx_bin', fromlist=['resolved']).resolved(),
        'all_still_feasible': verdict, 'rows': rows}, indent=1) + '\n')
    print(f"\nTutti ancora ammissibili (Q<1): {verdict}\n-> {OUT}")


if __name__ == '__main__':
    main()
