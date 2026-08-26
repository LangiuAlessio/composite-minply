"""Confronto ccx-vs-Abaqus sulle frequenze proprie, su un deck che QUESTO bundle genera.

Perche' esiste (2026-08-26). La riga della frequenza nella tabella di validazione del paper
(ccx 41,96 / Abaqus 41,91 Hz) NON e' su nessun deck di questo bundle: viene da un crop a 928 nodi
del modello Abaqus originale del coautore, che non e' qui. `exp1_abaqus_validation.py` la stampa
soltanto: e' un numero trascritto, non calcolato. L'audit del 2026-07-20 (rilievo A3) l'aveva gia'
segnalato, e una ricostruzione della geometria era stata ritirata perche' fittava UN parametro
libero su UN numero, senza potere residuo di verifica.

La via d'uscita non e' ricostruire il deck del coautore: e' **rifare il confronto su un deck che il
bundle possiede**. Il pannello S8R a 661 nodi e 60 ply e' quello delle righe di buckling della
stessa tabella, lo genera `fe.reference_cases.make_freq_deck`, e sta sotto il tetto di 1000 nodi
della Learning Edition di Abaqus -- che e' con ogni probabilita' il motivo per cui anche il crop del
coautore ne aveva 928.

Misurato il 2026-08-26 (ccx 2.21 sul Mac, Abaqus 2026 Learning Edition su Windows):

    modo    ccx [Hz]   Abaqus [Hz]   scarto
      1       532.91        531.69     0.23%
      2      2067.20       2038.90     1.39%
      3      3159.32       3124.80     1.10%
      4      3768.54       3767.50     0.03%
      5      6485.75       6363.00     1.93%

Cinque modi invece di uno: un confronto su piu' modi ha potere di falsificazione che un singolo
numero non ha, ed e' il motivo per cui questo sostituisce bene la riga trascritta. Il numero NON e'
41,96 Hz e non deve esserlo: quello era un altro pannello, piu' grande. Questo e' il pannello del
paper.

Il lato Abaqus non gira da qui (licenza commerciale, macchina Windows): l'artefatto e' versionato in
`data/exp29_abaqus_freq661.dat`, cioe' il .dat di Abaqus per intero, cosi' che chiunque possa
rileggere i suoi autovalori senza avere la licenza. Il lato ccx si rigenera con questo script.

Uso:  CCX_BIN=ccx_2.21 PYTHONPATH=$PWD python3 -m experiments.exp29_modal_cross_solver
"""
from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fe.ccx_bin import CCX, resolved                      # noqa: E402
from fe.reference_cases import make_freq_deck             # noqa: E402

SEQ = [0, 45, -45, 90] * 15          # 60 ply, il pannello delle righe di buckling
NFREQ = 6
ABQ_DAT = Path(__file__).resolve().parents[1] / 'data' / 'exp29_abaqus_freq661.dat'
OUT = Path(__file__).resolve().parents[1] / 'data' / 'exp29_modal_cross_solver.json'


def ccx_frequencies() -> list[float]:
    d = tempfile.mkdtemp()
    try:
        deck = make_freq_deck(SEQ, nfreq=NFREQ)
        # Contare le righe che "iniziano con una cifra" conta anche gli elementi e gli NSET:
        # la prima versione di questo script stampava 934 nodi per un deck che ne ha 661, cioe'
        # un numero falso in un esperimento che serve a verificare numeri. Si conta dentro la
        # sezione *NODE e basta.
        nodes, in_node = 0, False
        for ln in deck.splitlines():
            if ln.startswith('*'):
                in_node = ln.upper().startswith('*NODE')
                continue
            if in_node and ',' in ln:
                nodes += 1
        open(d + '/job.inp', 'w').write(deck)
        r = subprocess.run([CCX, '-i', 'job'], cwd=d, capture_output=True, text=True,
                           timeout=600, env={**os.environ, 'OMP_NUM_THREADS': '1'})
        if r.returncode != 0 or not os.path.exists(d + '/job.dat'):
            sys.exit(f'exp29: ccx non ha prodotto risultati (exit {r.returncode})')
        dat = open(d + '/job.dat').read()
        f = [float(x) for x in re.findall(
            r'^\s+\d+\s+[\d.E+\-]+\s+[\d.E+\-]+\s+([\d.E+\-]+)', dat, re.M)]
        print(f'deck: {nodes} nodi, {len(SEQ)} ply')
        return f[:5]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def abaqus_frequencies() -> list[float]:
    """Riletti dal .dat versionato: il lato Abaqus non e' rieseguibile senza licenza."""
    if not ABQ_DAT.exists():
        sys.exit(f'exp29: manca {ABQ_DAT.name}, il .dat di Abaqus versionato')
    t = ABQ_DAT.read_text(errors='ignore')
    i = t.index('E I G E N V A L U E    O U T P U T')
    rows = [ln.split() for ln in t[i:i + 900].splitlines()
            if ln.strip() and ln.split()[0].isdigit()]
    return [float(r[3]) for r in rows[:5]]


def main() -> None:
    fc, fa = ccx_frequencies(), abaqus_frequencies()
    rows = [{'mode': k, 'ccx_Hz': round(a, 2), 'abaqus_Hz': round(b, 2),
             'diff_pct': round(100 * (a - b) / b, 2)}
            for k, (a, b) in enumerate(zip(fc, fa), 1)]
    print(f"{'modo':>5} {'ccx [Hz]':>12} {'Abaqus [Hz]':>12} {'scarto':>8}")
    for r in rows:
        print(f"{r['mode']:5d} {r['ccx_Hz']:12.2f} {r['abaqus_Hz']:12.2f} {r['diff_pct']:7.2f}%")
    worst = max(abs(r['diff_pct']) for r in rows)
    if worst > 3.0:
        sys.exit(f'exp29: scarto massimo {worst:.2f}% oltre il 3% -- i due solutori non stanno '
                 f'piu\' rispondendo la stessa cosa, non riportare il confronto')
    OUT.write_text(json.dumps({
        'deck': '661-node S8R panel, 60 plies, generated by fe.reference_cases.make_freq_deck',
        'ccx': resolved(), 'abaqus': 'Abaqus 2026 Learning Edition (node cap 1000)',
        'abaqus_artefact': ABQ_DAT.name, 'rows': rows, 'worst_diff_pct': worst}, indent=1) + '\n')
    print(f'\nscarto massimo {worst:.2f}% -> {OUT}')


if __name__ == '__main__':
    main()
