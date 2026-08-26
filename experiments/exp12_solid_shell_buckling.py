"""Pannello (B) della figura dei pitfall: il 298x e' ccx/Abaqus o ccx/guscio-validato?

Il corpo del paper e la didascalia della Fig. 9 dicono che il solido C3D8I sbaglia il
buckling "relative to the validated shell"; l'asse del pannello (B) dice invece
"| ccx / Abaqus |". Sono due denominatori diversi e solo uno puo' essere quello giusto.
La differenza e' operativa, non stilistica:

  * denominatore = GUSCIO VALIDATO -> il 298x e' roba nostra, si rigenera in casa;
  * denominatore = ABAQUS-SOLIDO   -> serve il numero del coautore (G. Canale).

Questo script decide la questione misurando il rapporto solido/guscio con ccx da solo.
Se il solido sbagliasse davvero di ~300x contro il guscio validato, il rapporto si
vedrebbe qui.

DECK DI VALIDAZIONE, identificato il 2026-07-20. La Tabella 7 del paper non dichiara
la sequenza usata e il deck ("rr_shell_composite.py" in VALIDATION.md) non e' mai
esistito in questo bundle. E' stato ritrovato per ricerca diretta sulla mesh a 661 nodi
di `constrained_search`: e' il quasi-isotropo a 60 ply [0/45/-45/90]x15 (6.0 mm, lo
stesso spessore che VALIDATION.md attribuisce al laminato di riferimento), che riproduce
ENTRAMBE le righe di buckling della tabella (canarino 1 sotto).

CANARINI (criterio fissato prima di guardare i risultati):
  1. il deck di validazione riproduce la colonna CalculiX della Tabella 7 entro lo 0.5%
     (assiale 10.84, combinato 3.985), altrimenti non stiamo misurando quel pannello;
  2. l'autovalore scala linearmente col carico di riferimento entro lo 0.1%
     (BF x |axial| invariante), altrimenti siamo nella QUARTA failure mode del paper --
     carico di riferimento oltre il critico, autovalore restituito semplicemente
     sbagliato -- e ogni rapporto calcolato qui e' rumore;
  3. ogni riga riportata ha BF > 1 su ENTRAMBI i modelli, per lo stesso motivo: il
     carico di riferimento e' scelto piccolo apposta. Le righe che non lo rispettano
     sono scartate, non mediate.

VERDETTO (soglia pre-registrata): il denominatore e' il guscio validato SOLO se qualche
configurazione supera un rapporto di 100x. Il 298 e' un fattore 300: se il massimo
misurato resta di ordine 1, l'ipotesi "guscio" e' falsificata e il numero di Abaqus
serve davvero.

Uso:  CCX_BIN=ccx_2.21 PYTHONPATH=$PWD python3 experiments/exp12_solid_shell_buckling.py
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import optimisers.constrained_search as R          # noqa: E402  guscio S8R validato
from fe.interlaminar import solid_buckling_factor  # noqa: E402  solido C3D8I

# Il deck di validazione della Tabella 7 (vedi docstring).
SEQ_REF = [0, 45, -45, 90] * 15
PAPER_TABLE7 = {'axial': 10.84, 'combined': 3.985}

# Carico di riferimento dello sweep: 10x sotto quello di validazione, per tenere BF > 1
# ovunque e restare fuori dal regime patologico (canarino 3).
AXIAL = -100.0

# asse 1: spessore. asse 2: risoluzione nel piano. Il paper accusa "thin OR under-resolved".
PLY_COUNTS = [4, 8, 12, 16, 20, 24, 32, 40, 60]
MESHES = [(2, 1), (4, 2), (5, 3), (10, 5), (20, 10)]

RATIO_GATE = 100.0   # sotto questo, "denominatore = guscio validato" e' falsificato
OUT = Path(__file__).resolve().parents[1] / 'data' / 'exp12_solid_shell_buckling.json'


def shell_bf(seq, axial, side=0.0):
    return R.buckling_factor((seq, dict(axial=axial, side=side, torsion=0.0)))


def quasi_iso(n):
    """Le prime n ply della sequenza quasi-isotropa di riferimento."""
    return ([0, 45, -45, 90] * (n // 4 + 1))[:n]


def main() -> None:
    # --- canarino 1: il deck di validazione e' davvero quello della Tabella 7 ----------
    got = {'axial': shell_bf(SEQ_REF, -1000.0),
           'combined': shell_bf(SEQ_REF, -1000.0, side=5000.0)}
    print('=== CANARINO 1 — il deck di validazione riproduce la Tabella 7 ===')
    for k, paper in PAPER_TABLE7.items():
        dev = 100 * abs(got[k] - paper) / paper
        print(f"  {k:9s}: ccx {got[k]:8.4f}   paper {paper:7.3f}   scarto {dev:.3f}%")
        if dev > 0.5:
            sys.exit(f"exp12: il deck NON riproduce la riga '{k}' della Tabella 7 "
                     f"({got[k]:.4f} vs {paper}): non stiamo misurando il pannello (B)")

    # --- canarino 2: l'autovalore scala col carico (quarta failure mode) ---------------
    print('\n=== CANARINO 2 — linearita\' dell\'autovalore nel carico di riferimento ===')
    scaled = []
    for ax in (-1000.0, -100.0, -10.0):
        bf = shell_bf(SEQ_REF, ax)
        inv = bf * abs(ax) / 1000.0
        scaled.append(inv)
        print(f"  axial {ax:>8.0f}: BF {bf:10.4f}   BF*|axial|/1000 = {inv:8.4f}")
    spread = 100 * (max(scaled) - min(scaled)) / min(scaled)
    print(f"  dispersione: {spread:.4f}%")
    if spread > 0.1:
        sys.exit(f"exp12: l'autovalore NON scala col carico ({spread:.3f}%): siamo nella "
                 f"quarta failure mode, i rapporti sotto sarebbero rumore")

    # --- misura: solido C3D8I vs guscio validato --------------------------------------
    rows, skipped = [], []
    print(f"\n=== SWEEP SPESSORE (un elemento per ply, mesh 20x10, axial {AXIAL:.0f}) ===")
    print(f"{'ply':>4} {'t[mm]':>6} {'guscio':>10} {'solido':>10} {'solido/guscio':>14}")
    for n in PLY_COUNTS:
        seq = quasi_iso(n)
        sh = shell_bf(seq, AXIAL)
        so = solid_buckling_factor(seq, axial=AXIAL, side=0.0, nx=20, ny=10)
        row = {'axis': 'thickness', 'plies': n, 'thickness_mm': round(n * 0.1, 3),
               'nx': 20, 'ny': 10, 'shell_bf': sh, 'solid_bf': so}
        # canarino 3: sotto BF=1 il carico di riferimento supera il critico -> riga nulla
        if so is None or sh is None or sh <= 1.0 or so <= 1.0:
            row['skipped'] = 'BF <= 1 (carico oltre il critico) o ccx senza autovalore'
            skipped.append(row)
            print(f"{n:>4} {n*0.1:>6.1f} {sh:>10.4f} "
                  f"{(so if so else float('nan')):>10.4f} {'SCARTATA (BF<=1)':>14}")
            continue
        row['ratio'] = so / sh
        rows.append(row)
        print(f"{n:>4} {n*0.1:>6.1f} {sh:>10.4f} {so:>10.4f} {row['ratio']:>14.3f}")

    print(f"\n=== SWEEP RISOLUZIONE NEL PIANO (60 ply / 6.0 mm, axial {AXIAL:.0f}) ===")
    sh_ref = shell_bf(SEQ_REF, AXIAL)
    print(f"  guscio di riferimento: BF = {sh_ref:.4f}")
    for nx, ny in MESHES:
        so = solid_buckling_factor(SEQ_REF, axial=AXIAL, side=0.0, nx=nx, ny=ny)
        row = {'axis': 'in_plane_mesh', 'plies': 60, 'thickness_mm': 6.0,
               'nx': nx, 'ny': ny, 'shell_bf': sh_ref, 'solid_bf': so}
        if so is None or so <= 1.0:
            row['skipped'] = 'ccx senza autovalore o BF <= 1'
            skipped.append(row)
            continue
        row['ratio'] = so / sh_ref
        rows.append(row)
        print(f"  solido {nx:>2}x{ny:<2}: BF {so:>10.4f}   solido/guscio = {row['ratio']:>7.3f}")

    # --- verdetto ---------------------------------------------------------------------
    worst = max(rows, key=lambda r: max(r['ratio'], 1.0 / r['ratio']))
    worst_ratio = max(worst['ratio'], 1.0 / worst['ratio'])
    print(f"\n=== VERDETTO ===")
    print(f"  configurazioni valide: {len(rows)}  (scartate {len(skipped)})")
    print(f"  scostamento massimo solido-vs-guscio: {worst_ratio:.3f}x "
          f"({worst['plies']} ply, mesh {worst['nx']}x{worst['ny']})")
    if worst_ratio >= RATIO_GATE:
        verdict = 'shell'
        print(f"  >= {RATIO_GATE:.0f}x: il denominatore del 298x PUO' essere il guscio "
              f"validato -> il pannello (B) si rigenera in casa.")
    else:
        verdict = 'abaqus'
        print(f"  << {RATIO_GATE:.0f}x: con ccx da solo il solido C3D8I NON sbaglia di "
              f"ordini di grandezza contro il guscio validato.")
        print(f"  -> l'ipotesi 'denominatore = guscio validato' e' FALSIFICATA: il 298x "
              f"e' un rapporto ccx/Abaqus sul SOLIDO e il numero di Abaqus manca.")
        print(f"  -> di conseguenza l'asse del pannello (B) e' corretto e sono il corpo "
              f"del paper e la didascalia della Fig. 9 a dover essere corretti.")

    OUT.write_text(json.dumps({
        'question': "pannello (B): il 298x e' ccx/Abaqus o ccx/guscio-validato?",
        'validation_deck': {'sequence': SEQ_REF, 'n_plies': len(SEQ_REF),
                            'thickness_mm': round(len(SEQ_REF) * 0.1, 3),
                            'paper_table7': PAPER_TABLE7, 'reproduced': got},
        'axial_ref': AXIAL, 'ratio_gate': RATIO_GATE,
        'ccx': __import__('fe.ccx_bin', fromlist=['resolved']).resolved(),
        'rows': rows, 'skipped': skipped,
        'worst_ratio': worst_ratio, 'verdict': verdict,
    }, indent=1))
    print(f"\nscritto {OUT}")


if __name__ == '__main__':
    main()
