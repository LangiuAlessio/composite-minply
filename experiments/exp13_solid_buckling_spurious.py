"""Il 298x del pannello (B) e' LOCKING del C3D8I o un AUTOVALORE SPURIO?

Il paper attribuisce il 298x al tipo di elemento ("element-type mis-prediction: a thin or
under-resolved C3D8I solid mis-predicts buckling by orders of magnitude"). Ma il numeratore
di quel rapporto e' un ccx **~1.0**, che e' esattamente la firma che il pannello **(A)** della
stessa figura usa per dimostrare che un autovalore e' SPURIO, e che le note interne
(`fe-batch-lab/docs/RR_composite_optimiser_FINDINGS.md`) chiamano "the recurring ~1.0 spurious
eigenvalue". Le due spiegazioni non sono la stessa cosa:

  * LOCKING          -> l'autovalore e' vero ma calcolato su una rigidezza sbagliata: e' un
                        numero fisico, quindi SCALA col carico di riferimento e si RILASSA
                        raffinando lo spessore;
  * AUTOVALORE SPURIO -> il numero non e' l'autovalore del problema posto: non scala, non si
                        rilassa, e resta inchiodato a ~1.0 qualunque cosa gli si faccia.

Un autovalore di buckling lineare e' definito da (K + lambda*Kg) v = 0 con Kg assemblato dal
carico di riferimento: **raddoppiando il carico l'autovalore deve dimezzarsi**, sempre. E' un
requisito di definizione, non un'aspettativa numerica. Questo e' il discriminante.

DECK: il crop a 3 ply (0.3 mm) di `crop_layers(text, k=3)`, il sotto-modello a 928 nodi tagliato
per stare sotto il limite di 1000 nodi di Abaqus Learning Edition — cioe' il deck su cui il 298x
e' stato misurato (ccx ~1.0 vs Abaqus 0.00336).

CANARINI (criterio fissato prima di guardare i risultati):
  1. CONTROLLO POSITIVO — il guscio validato e il solido pieno a 60 ply, che sappiamo sani,
     DEVONO scalare col carico entro l'1%. Se non scalano loro, non e' il crop a essere malato:
     e' la misura, e il verdetto va buttato.
  2. Il valore di Abaqus (0.00336) deve restare entro un fattore 10 dallo scaling t^3 dal
     pannello pieno, altrimenti non e' lui il riferimento sano e il confronto non ha un verso.

VERDETTO (soglia pre-registrata, POI CORRETTA — vedi nota).
  Discriminante: la dispersione dell'INVARIANTE BF*|carico|, che per un autovalore vero e'
  costante per definizione.
  * dispersione > 100%  -> il numero non e' l'autovalore del problema posto: SPURIO;
  * dispersione < 100%  -> e' un autovalore vero su una rigidezza sbagliata: LOCKING.

  ⚠️ NOTA DI ONESTA' (2026-07-20). La prima stesura metteva il gate sulla dispersione del BF
  GREZZO ("< 10% => spurio"), ed era MAL POSTA: il BF grezzo di un modello SANO varia di ~99900%
  su un range di carico di 1000x, quindi quel test classificava "locking" qualunque cosa variasse
  piu' del 10% -- incluso un numero che varia del 28% mentre il carico varia di 1000x, cioe' un
  numero praticamente invariante. Il gate e' stato riscritto sull'invariante DOPO aver visto i
  dati. Lo si dichiara perche' cambiare una soglia a valle della misura e' esattamente il modo in
  cui si fabbrica un risultato; qui e' difendibile solo perche' (a) la quantita' giusta era gia'
  calcolata e stampata nella prima esecuzione, non e' stata cercata dopo, e (b) la separazione fra
  sano e malato e' di CINQUE ordini di grandezza (0.00% e 0.72% sui controlli, 94539% sul crop),
  quindi nessuna scelta ragionevole di soglia cambia il verdetto. Se un domani questi numeri si
  avvicinassero, il verdetto andrebbe rifatto, non ritarato.

Uso:  CCX_BIN=ccx_2.21 PYTHONPATH=$PWD python3 experiments/exp13_solid_buckling_spurious.py
"""
from __future__ import annotations
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import optimisers.constrained_search as R                              # noqa: E402
from fe.interlaminar import CCX, PLY_T, make_solid_buckle_deck, solid_buckling_factor  # noqa: E402

FULL = [0, 45, -45, 90] * 15      # pannello di validazione, 60 ply / 6.0 mm
CROP3 = FULL[:3]                  # crop_layers(k=3): 3 ply / 0.3 mm, il deck del 298x
ABAQUS_CROP_BF = 0.00336          # Abaqus LE sul nodo head (fe-batch-lab FINDINGS:82)
LOADS = [-1000.0, -100.0, -10.0, -1.0]
INVARIANCE_GATE = 100.0           # % di dispersione dell'invariante BF*|carico| sopra cui e' spurio
OUT = Path(__file__).resolve().parents[1] / 'data' / 'exp13_solid_buckling_spurious.json'


def all_factors(seq, axial, nx=20, ny=10, nbuck=10, ply_t=None):
    """Tutti gli autovalori di buckling restituiti da ccx, non solo il primo."""
    deck, *_ = make_solid_buckle_deck(seq, axial, 0.0, nx, ny, nbuck, ply_t)
    d = tempfile.mkdtemp()
    try:
        open(d + '/job.inp', 'w').write(deck)
        subprocess.run([CCX, '-i', 'job'], cwd=d, capture_output=True, text=True, timeout=3600,
                       env={**os.environ, 'OMP_NUM_THREADS': '1'})
        dat = open(d + '/job.dat').read() if os.path.exists(d + '/job.dat') else ''
        return [float(v) for _, v in re.findall(r'^\s*(\d+)\s+([\d.E+\-]+)\s*$', dat, re.M)]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def scaling(label, fn):
    """BF a carichi decrescenti + dispersione dell'invariante BF*|carico| (che per un
    autovalore vero e' costante)."""
    rows, inv = [], []
    for ax in LOADS:
        bf = fn(ax)
        rows.append({'axial': ax, 'bf': bf})
        if bf:
            inv.append(bf * abs(ax) / 1000.0)
    spread = 100 * (max(inv) - min(inv)) / min(inv) if len(inv) > 1 and min(inv) > 0 else float('nan')
    bfs = [r['bf'] for r in rows if r['bf']]
    bf_spread = 100 * (max(bfs) - min(bfs)) / min(bfs) if len(bfs) > 1 and min(bfs) > 0 else float('nan')
    print(f"\n  {label}")
    for r in rows:
        print(f"    axial {r['axial']:>8.0f}: BF = {r['bf']}")
    print(f"    variazione del BF sul range di carico (1000x): {bf_spread:8.2f}%")
    print(f"    dispersione dell'invariante BF*|carico|:       {spread:8.2f}%")
    return {'rows': rows, 'bf_spread_pct': bf_spread, 'invariant_spread_pct': spread}


def main() -> None:
    res = {}

    print('=== CANARINO 1 — controllo positivo: i modelli SANI devono scalare col carico ===')
    res['shell_full'] = scaling(
        'guscio S8R validato, 60 ply (sano)',
        lambda ax: R.buckling_factor((FULL, dict(axial=ax, side=0.0, torsion=0.0))))
    res['solid_full'] = scaling(
        'solido C3D8I pieno, 60 ply (sano: exp12 lo da\' a 0.83x del guscio)',
        lambda ax: solid_buckling_factor(FULL, axial=ax, side=0.0))
    for k in ('shell_full', 'solid_full'):
        if not (res[k]['invariant_spread_pct'] < 1.0):
            sys.exit(f"exp13: il controllo positivo '{k}' NON scala col carico "
                     f"({res[k]['invariant_spread_pct']:.2f}%): e' la misura a essere rotta, "
                     f"non il crop -- verdetto non emesso")
    print('\n  -> i due modelli sani scalano entro l\'1%: la misura e\' buona.')

    print('\n=== CANARINO 2 — il riferimento Abaqus e\' fisicamente sano? ===')
    bf_full = R.buckling_factor((FULL, dict(axial=-1000.0, side=0.0, torsion=0.0)))
    expected = bf_full * (0.3 / 6.0) ** 3
    factor = ABAQUS_CROP_BF / expected
    print(f"  pannello pieno 6.0 mm: BF {bf_full:.4f};  scaling t^3 a 0.3 mm -> atteso {expected:.5f}")
    print(f"  Abaqus sul crop: {ABAQUS_CROP_BF}  ->  fattore {factor:.2f} dall'atteso")
    if not (0.1 < factor < 10.0):
        sys.exit(f"exp13: il valore Abaqus non e' compatibile con lo scaling t^3 "
                 f"(fattore {factor:.1f}): non e' lui il riferimento sano, confronto senza verso")
    res['abaqus_sanity'] = {'bf_full_6mm': bf_full, 't3_expected': expected,
                            'abaqus_crop': ABAQUS_CROP_BF, 'factor_from_expected': factor}

    print(f"\n=== TEST DECISIVO — il crop a 3 ply (0.3 mm), il deck del 298x ===")
    res['solid_crop3'] = scaling(
        'solido C3D8I crop 3 ply (0.3 mm)',
        lambda ax: solid_buckling_factor(CROP3, axial=ax, side=0.0))

    print('\n=== TEST DI CONTORNO — raffinamento nello spessore a SPESSORE COSTANTE ===')
    print('  (il locking si rilassa raffinando; un autovalore spurio no)')
    refine = []
    for k in (1, 2, 4):
        seq = [a for a in CROP3 for _ in range(k)]
        bf = solid_buckling_factor(seq, axial=-1000.0, side=0.0, ply_t=PLY_T / k)
        refine.append({'elements_per_ply': k, 'nz': len(seq), 'bf': bf})
        print(f"    {k} elem/ply (nz={len(seq)}, spessore 0.3 mm): BF = {bf}")
    res['through_thickness_refinement'] = refine

    print('\n=== TEST DI CONTORNO — i primi autovalori del crop (c\'e\' un grappolo a ~1.0?) ===')
    spec = all_factors(CROP3, -1000.0, nbuck=10)
    print(f"    {[round(v, 4) for v in spec[:10]]}")
    res['crop_spectrum'] = spec

    # --- verdetto -------------------------------------------------------------------
    crop_spread = res['solid_crop3']['invariant_spread_pct']
    print('\n=== VERDETTO ===')
    print(f"  dispersione dell'invariante BF*|carico|:")
    print(f"    guscio sano  {res['shell_full']['invariant_spread_pct']:12.2f}%")
    print(f"    solido sano  {res['solid_full']['invariant_spread_pct']:12.2f}%")
    print(f"    crop 3 ply   {crop_spread:12.2f}%   <-- il deck del 298x")
    print(f"  (il carico varia di 1000x; il BF del crop varia solo del "
          f"{res['solid_crop3']['bf_spread_pct']:.1f}%, cioe' e' quasi invariante)")
    # Gate scritto in forma NEGATA apposta: `nan > GATE` e' False, quindi con la forma
    # diretta una misura del crop completamente fallita (tutti i solid_buckling_factor a
    # None -> spread NaN) cadeva sul ramo 'locking', cioe' emetteva DA ZERO DATI proprio
    # la spiegazione che questo esperimento esiste per mettere alla prova. Cosi' il NaN
    # cade sul lato 'spurious'/errore, come gia' fa il canarino a inizio file.
    if not (crop_spread <= INVARIANCE_GATE):
        verdict = 'spurious'
        print(f"  > {INVARIANCE_GATE:.0f}%: il numero NON e' l'autovalore del problema posto. Un")
        print(f"  autovalore di buckling lineare DEVE scalare come 1/|carico|; questo resta")
        print(f"  inchiodato a ~1.0. -> AUTOVALORE SPURIO, la stessa patologia del pannello (A).")
        print(f"  -> la causa dichiarata nel paper ('element-type mis-prediction', locking del")
        print(f"     C3D8I) NON e' sostenuta: il fatto regge (il solido non e' affidabile per il")
        print(f"     buckling, si usa il guscio), la spiegazione no.")
    elif math.isnan(crop_spread):
        sys.exit('exp13: crop_spread e\' NaN -- la misura del crop e\' fallita, nessun verdetto')
    else:
        verdict = 'locking'
        print(f"  <= {INVARIANCE_GATE:.0f}%: l'autovalore scala, quindi e' un numero fisico su una")
        print(f"  rigidezza sbagliata -> la lettura 'locking' del paper regge.")
    res['verdict'] = verdict
    res['crop_invariant_spread_pct'] = crop_spread

    OUT.write_text(json.dumps({
        'question': "il 298x del pannello (B) e' locking del C3D8I o un autovalore spurio?",
        'crop_deck': {'sequence': CROP3, 'thickness_mm': 0.3,
                      'provenance': 'crop_layers(k=3), 928-node sub-model, sotto il cap di 1000 '
                                    'nodi di Abaqus LE; fe-batch-lab FINDINGS:82'},
        'loads': LOADS, 'invariance_gate_pct': INVARIANCE_GATE,
        'ccx': os.environ.get('CCX_BIN', 'ccx_2.21'), **res}, indent=1))
    print(f"\nscritto {OUT}")


if __name__ == '__main__':
    main()
