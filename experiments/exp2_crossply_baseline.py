"""exp2_crossply_baseline.py -- 60-ply cross-ply baseline vs the Abaqus reference.

Evaluates the 60-ply cross-ply [0/90]_15s at the reference load cases and compares the
CalculiX buckling factors and first natural frequency to Giacomo's Abaqus 2020 reference
(15.22 / 3.70 / 2.47, ~627 Hz). This is the anchor that validates the open-source evaluator
against the commercial one before the optimisation campaign.

Run:  PYTHONPATH=$PWD python experiments/exp2_crossply_baseline.py
"""
from __future__ import annotations

from fe.reference_cases import CASES_V1, first_freq
from optimisers.constrained_search import buckling_factor

REF_ABAQUS = {"C1": 15.22, "C2": 3.70, "C3": 2.47}   # Giacomo's Abaqus 2020 cross-ply reference


def main():
    seq = [0, 90] * 15
    seq = seq + seq[::-1]            # [0/90]_15s, 60 plies
    print(f"cross-ply [0/90]_15s, {len(seq)} plies")
    for cn, case in CASES_V1.items():
        bf = buckling_factor((seq, case))
        print(f"  {cn}: ccx BF = {bf:6.2f}   (Abaqus reference {REF_ABAQUS[cn]})")
    print(f"  first natural frequency: {first_freq(seq):.1f} Hz   (Abaqus reference ~627 Hz)")


if __name__ == "__main__":
    main()
