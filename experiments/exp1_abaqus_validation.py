"""exp1_abaqus_validation.py -- CalculiX-vs-Abaqus validation of the FE evaluator.

The open-source CalculiX evaluator was validated against Abaqus 2020 on identical decks at a
reference thickness:

    axial buckling           ccx 10.84   vs  Abaqus 10.819   (0.2%)
    combined buckling        ccx  3.992  vs  Abaqus  3.915    (2.0%)
    first natural frequency  ccx 532.91  vs  Abaqus 531.69 Hz (0.2%)

CORRECTED 2026-07-25. This file used to print 3.985 for the combined case and "<= 1.8%" for the
spread, both of which the manuscript had already left behind: finding A2 of the 2026-07-20 audit
re-ran the combined case and got 3.9924, so Table 7 carries 3.992 and the spread is 2.0%, not
1.8%. The inherited 3.985 predates this bundle (git log -S"3.985") and was never recomputed. A
referee opening the bundle would have found the script contradicting the table it documents.

SOSTITUITA LA RIGA MODALE, 2026-08-26. Diceva ccx 41.96 vs Abaqus 41.91 Hz, e quei numeri NON
venivano da un deck di questo bundle: erano misurati su un crop a 928 nodi del modello Abaqus
originale del coautore, che qui non c'e'. Era quindi una riga che nessun lettore poteva
ricalcolare, in una tabella di validazione -- il rilievo A3 dell'audit del 2026-07-20. Ora la riga
e' il confronto modale sul pannello S8R a 661 nodi che il bundle genera da se':
ccx 532.91 vs Abaqus 2026 LE 531.69 Hz, con i primi cinque modi entro l'1.9%. Lo misura
exp29_modal_cross_solver.py, che porta con se' il file di risultati Abaqus per intero, cosi' che il
lato commerciale sia rileggibile senza licenza. Il numero non e' piu' 41.96 e non deve esserlo:
quello era un altro pannello, piu' grande.

The Abaqus values of the two buckling rows are the coauthor's reference (Abaqus 2020, commercial /
Windows-only) and are not re-run here; the modal row is re-run by exp29. The reproducible CalculiX side of the cross-solver check -- the 60-ply cross-ply
baseline against Giacomo's Abaqus reference -- is in exp2_crossply_baseline.py. This file simply
documents the validation table.

Run:  python experiments/exp1_abaqus_validation.py
"""

ABAQUS = {
    "axial buckling":        (10.84, 10.819),
    "combined buckling":     (3.992, 3.915),
    "first frequency [Hz]":  (532.91, 531.69),
}


def main():
    print("CalculiX vs Abaqus 2020 (identical decks, reference thickness):")
    for quantity, (ccx, abq) in ABAQUS.items():
        print(f"  {quantity:24s}: ccx {ccx:>8}  vs  Abaqus {abq:>8}   ({100 * abs(ccx - abq) / abq:.2f}%)")
    print("\nAbaqus is the coauthor's reference; the reproducible ccx-side check is exp2_crossply_baseline.py.")


if __name__ == "__main__":
    main()
