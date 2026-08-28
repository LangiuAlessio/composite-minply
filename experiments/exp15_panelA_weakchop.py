"""Panel (A) della figura dei pitfall: il weak-material "chop" ROMPE il buckling.

Generatore in-bundle del pannello (A), portato qui il 2026-07-20. Era
l'ultimo pannello della Fig. 9 senza sorgente dentro `code/`: il modello weak-chop
(Giacomo: "assegna un materiale dummy/debole ai ply rimossi") vive nel solido
Abaqus a 14k nodi, non nel guscio dell'ottimizzatore. La macchineria di traduzione
Abaqus->ccx ora e' in `fe/abq2ccx_rr.py` (portata dal lab di sviluppo FE); questo script
rigenera i quattro punti dallo stesso deck.

La tesi del pannello: con filler debole (`layer_weak` = `layer`/100) i ply molli
bucklano per primi -> modo locale spurio a fattore ~1.0 che MASCHERA il buckling
globale reale. Il primo autovalore diventa privo di senso: inchiodato a ~1.0
qualunque sia il layup O il numero di ply attivi. Contro il laminato reale che
risponde correttamente (BF ~20 a 60 ply tutti 0).

Riproduce (caso 3, assiale) le barre pubblicate:
    [0]_60  no-weak  -> 20.1   (reale, risponde al layup)
    [0]_24  +36 weak -> 0.95   (~1.0)
    [90]_24 +36 weak -> 0.99   (~1.0, UGUALE a [0] -> layup irrilevante)
    [0]_12  +48 weak -> 0.98   (~1.0)

The source deck (`decks/Composite_buckling_3.inp`) is a coauthor's model and is not
redistributed with this bundle (see decks/README.md). Run on a compute node, not a laptop:

    ./code/run_on_head.sh experiments.exp15_panelA_weakchop
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fe.abq2ccx_rr import set_laminate, translate  # noqa: E402

from fe.ccx_bin import CCX          # un solo default per il binario ccx (audit F12)
DECK = Path(os.environ.get("RR_DECK", ROOT / "decks" / "Composite_buckling_3.inp"))
OUT = ROOT / "experiments" / "_out" / "exp15_panelA_weakchop.json"

# (label, active plies, published bf1, absolute tolerance on the reproduced bf1)
VARIANTS = [
    ("[0]_60  no-weak", [0] * 60, 20.1, 0.6),      # ~3% on the real global mode
    ("[0]_24  +36 weak", [0] * 24, 0.95, 0.05),
    ("[90]_24 +36 weak", [90] * 24, 0.99, 0.05),
    ("[0]_12  +48 weak", [0] * 12, 0.98, 0.05),
]


def buckling_factors(deck_text: str, seq) -> list[float]:
    deck = translate(set_laminate(deck_text, seq), "buckle")
    d = tempfile.mkdtemp()
    try:
        Path(d, "job.inp").write_text(deck)
        subprocess.run([CCX, "-i", "job"], cwd=d, capture_output=True, timeout=900,
                       env={**os.environ, "OMP_NUM_THREADS": "1"})  # threaded buckling wrong in ccx<2.21
        dat = Path(d, "job.dat").read_text() if Path(d, "job.dat").exists() else ""
    finally:
        shutil.rmtree(d, ignore_errors=True)
    # rows are `      1   0.2014199E+02`; the ccx header is spaced ("B U C K L I N G"),
    # so match the numbered scientific-notation rows directly (only table in a *BUCKLE deck).
    return [float(m.group(1)) for ln in dat.splitlines()
            if (m := re.match(r"\s*\d+\s+([-+]?\d*\.\d+[Ee][-+]?\d+)\s*$", ln))]


def main() -> None:
    if not DECK.exists():
        sys.exit(f"exp15: deck sorgente assente: {DECK}\n"
                 "See decks/README.md: the deck is not redistributed with this bundle.")
    text = DECK.read_text()
    print(f"deck: {DECK.name}  ccx: {CCX}\n")
    rows, canary_fail = [], []
    for label, seq, pub, tol in VARIANTS:
        facs = buckling_factors(text, seq)
        bf1 = facs[0] if facs else None
        ok = bf1 is not None and abs(bf1 - pub) <= tol
        if not ok:
            canary_fail.append(f"{label}: bf1={bf1} vs pubblicato {pub} (tol {tol})")
        rows.append({"label": label, "n_active": len(seq), "angle": seq[0],
                     "factors": facs, "bf1": bf1, "published_bf1": pub, "within_tol": ok})
        got = f"{bf1:.4g}" if bf1 is not None else "FAIL"
        print(f"{label:18s} bf1={got:>8s}  (pubblicato {pub}, tol {tol})   {'OK' if ok else 'DRIFT'}")

    # CANARINO: se una barra non riproduce il pubblicato, il pannello (A) del bundle
    # non e' piu' quello della figura -> non fidarsi.
    if canary_fail:
        sys.exit("exp15 CANARY: il pannello (A) non riproduce le barre pubblicate:\n  "
                 + "\n  ".join(canary_fail))

    # e il CLAIM qualitativo, indipendente dai decimali
    real = next(r for r in rows if r["n_active"] == 60)["bf1"]
    weak = [r["bf1"] for r in rows if r["n_active"] < 60]
    a0 = next(r["bf1"] for r in rows if r["angle"] == 0 and r["n_active"] == 24)
    a90 = next(r["bf1"] for r in rows if r["angle"] == 90 and r["n_active"] == 24)
    claim = (real > 5 and all(0.8 < w < 1.3 for w in weak) and abs(a0 - a90) / a0 < 0.15)
    print(f"\nCLAIM pannello (A) riprodotto: {claim}  "
          f"(reale {real:.2f} >> weak {[round(w, 3) for w in weak]}; [0]~[90]: {a0:.3f}~{a90:.3f})")

    result = {"deck": DECK.name, "ccx": CCX, "claim_reproduced": claim, "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))
    print(f"\nscritto {OUT}")


if __name__ == "__main__":
    main()
