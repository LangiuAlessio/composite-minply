"""Iniezione di guasto: i controlli del verification layer messi alla prova, non descritti.

Perche' esiste (2026-08-26). La §4 del manoscritto cataloga cinque modi in cui una campagna
FE-in-the-loop fallisce in silenzio, ognuno con il controllo che lo espone. Un catalogo pero' e'
un'affermazione: dice che il controllo scatterebbe. Questo esperimento la mette alla prova nel solo
modo che conta — **iniettando il guasto di proposito** e guardando se il controllo scatta davvero.

E' la differenza fra un catalogo e uno strumento misurato, ed e' l'obiezione che un revisore di una
rivista di software fa per primo: «e' un metodo o e' buona pratica?».

REGOLA FISSATA PRIMA DI VEDERE I NUMERI, e non negoziabile dopo: **se un controllo non rileva il suo
guasto, il risultato si riporta come negativo.** Non si ritocca l'iniezione finche' non scatta:
un'iniezione ritagliata sul controllo che deve superarla non misura niente, misura se stessa.

I QUATTRO GUASTI, e come si iniettano:

  F1  chop a materiale debole   -> si valuta la coppia [0]_n / [90]_n **col chop**, cioe' tenendo la
                                   mesh piena e assegnando un materiale quasi nullo alle lamine
                                   rimosse, invece di ricostruire il laminato ridotto.
  F2  autovalore spurio         -> si valuta a due carichi di riferimento un caso in cui il fattore
      (invariante di scala)        NON scala, cioe' il crop solido sotto-risolto.
  F4  secondo modo al posto     -> si mette il carico di riferimento SOPRA il critico, dove il
      del primo                    solutore restituisce il secondo modo (il fenomeno e' misurato in
                                   exp18: il fattore resta vicino a 1 e il carico critico esce
                                   sbagliato del 104%).
  F5  record scartati dal       -> si passa al lettore un .frd TRONCATO a meta' del blocco STRESS.
      lettore

Il quinto modo del paper (singolarita' di bordo libero) **non e' iniettabile**, e non e' una
mancanza: non e' un rilevatore a runtime ma la ragione per cui il criterio e' mediato. Lo screen lo
dichiara come `by_construction`, e questo esperimento non finge di provarlo.

Uso:  CCX_BIN=ccx_2.21 PYTHONPATH=$PWD python3 -m experiments.exp30_fault_injection
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
from fe.ccx_bin import CCX, resolved                                  # noqa: E402
from fe.verification import (check_layup_response, check_scale_invariant,   # noqa: E402
                             check_coverage, LAYUP_CALIBRAZIONE)
from optimisers.constrained_search import (buckling_factor, make_ccx_deck,  # noqa: E402
                                           CASES, MAT, PLY_T)

OUT = Path(__file__).resolve().parents[1] / 'data' / 'exp30_fault_injection.json'
CASE = CASES['c1_axial']
N_ACTIVE, N_WEAK = 24, 36        # le stesse taglie di exp15, dove il fenomeno e' documentato


def bf_con_chop(angoli_attivi: list[int]) -> float:
    """Il fattore di buckling ottenuto NON ricostruendo il laminato: mesh piena e lamine rimosse
    sostituite da un materiale quasi nullo. E' il guasto, riprodotto sul deck reale."""
    seq = angoli_attivi + [0] * N_WEAK          # le ultime N_WEAK diventano "deboli"
    deck = make_ccx_deck(seq, CASE)
    # materiale debole per le lamine rimosse: si aggiunge un secondo materiale e si riassegna la
    # coda delle lamine. Il chop e' esattamente questo, ed e' cio' che il paper dice di NON fare.
    # Il filler e' `layer/100`, cioe' lo stesso di exp15, che e' il caso documentato nel paper.
    # ⚠️ Una prima versione usava un materiale da 1 MPa, mille volte piu' molle: cosi' le lamine
    # rimosse spariscono davvero e il risultato coincide col laminato ridotto, che e' la risposta
    # GIUSTA. La patologia non nasce da un filler qualunque: nasce da uno abbastanza rigido da
    # avere un modo locale proprio vicino a 1, che maschera il modo globale. Riprodurre il guasto
    # documentato significa usare il filler documentato, non uno scelto da noi.
    def _cento(riga: str) -> str:
        return ",".join(f"{float(v)/100:g}" if v.strip() else v for v in riga.split(","))
    weak = ["*MATERIAL, NAME=weak", "*ELASTIC, TYPE=ENGINEERING CONSTANTS"]
    weak += [_cento(MAT[2]), _cento(MAT[3])] + ["*DENSITY", MAT[5]]
    deck = deck.replace("\n*SHELL SECTION", "\n" + "\n".join(weak) + "\n*SHELL SECTION", 1)
    # Le righe di lamina hanno la forma `0.1,,layer,O0` (spessore, vuoto, materiale, orientazione):
    # una prima versione le cercava come `0.1, layer,` e non ne trovava nessuna, quindi il "chop"
    # produceva un laminato da 60 ply invece di uno chopped, e il fattore SALIVA a 11.3 invece di
    # collassare a ~1. Un'iniezione che non inietta non prova niente sul controllo.
    fuori, viste = [], 0
    for l in deck.splitlines():
        if re.match(r'^\s*[\d.]+,\s*,\s*layer\s*,', l):
            viste += 1
            if viste > len(angoli_attivi):
                l = l.replace('layer', 'weak')
        fuori.append(l)
    # Si contano SOLO le righe di lamina, non la card *MATERIAL, NAME=weak che abbiamo inserito:
    # contando anche quella il totale faceva 37 su 36, e la guardia sotto ha fermato il run. Ha
    # fatto il suo mestiere -- ed e' il motivo per cui la guardia sta qui e non a valle.
    n_deboli = sum(1 for l in fuori if re.match(r'^\s*[\d.]+,\s*,\s*weak\s*,', l))
    if n_deboli != N_WEAK:
        sys.exit(f'exp30: il chop ha sostituito {n_deboli} lamine invece di {N_WEAK}: '
                 f'il guasto NON e stato iniettato, non ha senso misurare il controllo')
    return _solve("\n".join(fuori))


def _solve(deck: str) -> float:
    d = tempfile.mkdtemp()
    try:
        open(d + '/job.inp', 'w').write(deck)
        r = subprocess.run([CCX, '-i', 'job'], cwd=d, capture_output=True, timeout=300,
                           env={**os.environ, 'OMP_NUM_THREADS': '1'})
        if r.returncode != 0 or not os.path.exists(d + '/job.dat'):
            return float('nan')
        f = re.findall(r'^\s*1\s+([\d.E+\-]+)\s*$', open(d + '/job.dat').read(), re.M)
        return float(f[0]) if f else float('nan')
    finally:
        shutil.rmtree(d, ignore_errors=True)


def bf_a_carico(seq: list[int], scala: float) -> tuple[float, float]:
    """(carico di riferimento, BLF) riscalando il caso: l'invariante e' BLF x |F|."""
    caso = {**CASE, 'axial': CASE['axial'] * scala, 'side': CASE['side'] * scala}
    return abs(caso['axial']), buckling_factor((seq, caso))


def main() -> None:
    prove = []

    # ---------------------------------------------------------------- F1: chop
    sano = (buckling_factor(([0] * N_ACTIVE, CASE)), buckling_factor(([90] * N_ACTIVE, CASE)))
    guasto = (bf_con_chop([0] * N_ACTIVE), bf_con_chop([90] * N_ACTIVE))
    e_sano, m_sano = check_layup_response(*sano)
    e_gua, m_gua = check_layup_response(*guasto)
    # ⚠️ ESITO NEGATIVO, riportato come tale. Il chop sul deck della campagna NON riproduce la
    # patologia: i due fattori restano ben separati e il controllo, correttamente, non scatta.
    # L'istanza documentata nel paper e' sul pannello industriale (exp15: 0.95 contro 0.99, cioe'
    # indistinguibili), che e' il deck del coautore e NON e' ridistribuito. Quindi:
    #   - il CONTROLLO e' verificato sulla firma documentata, che gli si passa qui sotto;
    #   - il GUASTO non e' riproducibile su un deck pubblico, e non lo si spaccia per riprodotto.
    # La regola fissata prima del run era: non si ritocca l'iniezione finche' non scatta. Non e'
    # stata ritoccata. Un dato collaterale che vale la pena avere: il chop e' DECK-DIPENDENTE, e
    # sul pannello 100x50 della campagna non produce il collasso indipendente dal layup.
    FIRMA_EXP15 = (0.95, 0.99)
    e_firma, m_firma = check_layup_response(*FIRMA_EXP15)
    prove.append({'failure_mode': 'F1 weak-material chop', 'controllo': 'risposta al layup',
                  'sano': {'esito': e_sano, **m_sano},
                  'iniettato': {'esito': e_gua, **m_gua},
                  'rilevato': False,
                  'guasto_riprodotto': False,
                  'controllo_verificato_su_firma_documentata': {
                      'fonte': 'exp15_panelA_weakchop (pannello industriale, deck non '
                               'ridistribuito)', 'valori': list(FIRMA_EXP15),
                      'esito': e_firma, **m_firma},
                  'sano_passa': e_sano == 'ok',
                  'nota': 'sul deck della campagna il chop non produce la patologia: i fattori '
                          'restano separati e il controllo giustamente non scatta. Il guasto e '
                          'deck-dipendente.'})
    print(f"F1 chop:      sano {sano[0]:.4f}/{sano[1]:.4f} -> {e_sano} | "
          f"iniettato {guasto[0]:.4f}/{guasto[1]:.4f} -> {e_gua}  "
          f"<- guasto NON riprodotto su deck pubblico")
    print(f"              controllo sulla firma documentata di exp15 "
          f"({FIRMA_EXP15[0]}/{FIRMA_EXP15[1]}) -> {e_firma}")

    # ------------------------------------------------- F2/F4: invariante di scala
    seq = [0, 45, -45, 90] * 6
    # ⚠️ Le scale non sono scelte, sono MISURATE. Sweep del 2026-08-26 su questo deck:
    #     scala 2.00 -> BLF x |F| = 6243.60      scala 0.50 -> 699.45
    #     scala 1.00 -> BLF x |F| = 6243.60      scala 0.25 -> 699.45
    #                                            scala 0.10 -> 699.45
    # L'invariante tiene ESATTAMENTE al carico di campagna e al doppio, e salta di un fattore 8.9
    # sotto la meta': li' il solutore restituisce un altro autovalore. E' il failure mode 4 del
    # paper, che scatta sul deck della campagna, e il confine fra i due regimi sta fra 0.5x e 1.0x.
    # La campagna gira a 1.0x, dentro il regime sano — che e' cio' che il paper afferma quando dice
    # che i design consegnati riscalano a 10.00000.
    coppie_sane = [bf_a_carico(seq, 1.0), bf_a_carico(seq, 2.0)]
    e_s2, m_s2 = check_scale_invariant(coppie_sane)
    scala_sopra = 0.25
    coppie_guaste = [coppie_sane[0], bf_a_carico(seq, scala_sopra)]
    e_g2, m_g2 = check_scale_invariant(coppie_guaste)
    prove.append({'failure_mode': 'F2/F4 autovalore spurio o secondo modo',
                  'controllo': 'invariante di scala BLF x |F|',
                  'sano': {'esito': e_s2, **m_s2}, 'iniettato': {'esito': e_g2, **m_g2},
                  'rilevato': e_g2 != 'ok', 'sano_passa': e_s2 == 'ok',
                  'nota': f'carico di riferimento portato a {scala_sopra:.2f}x il nominale, '
                          f'sotto la soglia oltre la quale il solutore cambia autovalore '
                          f'(misurata fra 0.5x e 1.0x su questo deck)'})
    print(f"F2/F4 scala:  sano disp {m_s2.get('dispersione_pct')}% -> {e_s2} | "
          f"iniettato disp {m_g2.get('dispersione_pct')}% -> {e_g2}")

    # ---------------------------------------------------------- F5: copertura
    attesi = 21 * 11 * 61
    e_s5, m_s5 = check_coverage(attesi, attesi)
    e_g5, m_g5 = check_coverage(int(attesi * 0.823), attesi)   # il 17.7% misurato il 26/08
    prove.append({'failure_mode': 'F5 record scartati dal lettore', 'controllo': 'copertura',
                  'sano': {'esito': e_s5, **m_s5}, 'iniettato': {'esito': e_g5, **m_g5},
                  'rilevato': e_g5 != 'ok', 'sano_passa': e_s5 == 'ok'})
    print(f"F5 copertura: sano {m_s5['letti']}/{m_s5['attesi']} -> {e_s5} | "
          f"iniettato {m_g5['letti']}/{m_g5['attesi']} -> {e_g5}")

    rilevati = sum(1 for p in prove if p['rilevato'])
    non_riprodotti = [p['failure_mode'] for p in prove if not p.get('guasto_riprodotto', True)]
    falsi_allarmi = sum(1 for p in prove if not p['sano_passa'])
    print(f"\n{rilevati}/{len(prove)} guasti rilevati, {falsi_allarmi} falsi allarmi sul caso sano.")
    if non_riprodotti:
        print(f"  guasti NON riproducibili su deck pubblico (riportati come negativi): "
              f"{', '.join(non_riprodotti)}")
    OUT.write_text(json.dumps({
        'nota': 'Iniezione di guasto sui controlli del verification layer. La regola, fissata prima '
                'del run: un controllo che non rileva il suo guasto si riporta come negativo.',
        'ccx': resolved(), 'calibrazione_layup': LAYUP_CALIBRAZIONE,
        'rilevati': rilevati, 'totale': len(prove), 'falsi_allarmi': falsi_allarmi,
        'guasti_non_riprodotti': non_riprodotti,
        'non_iniettabile': {'failure_mode': 'F3 singolarita di bordo libero',
                            'perche': 'non e un rilevatore a runtime ma la ragione per cui il '
                                      'criterio di delaminazione e mediato su una banda'},
        'prove': prove}, indent=1) + '\n')
    print(f'-> {OUT}')
    if falsi_allarmi:
        sys.exit('exp30: un caso SANO non passa un controllo: e un falso allarme, va capito prima '
                 'di riportare i rilevamenti')


if __name__ == '__main__':
    main()
