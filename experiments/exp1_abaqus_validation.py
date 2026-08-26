"""exp1_abaqus_validation.py -- CalculiX-vs-Abaqus validation of the FE evaluator.

The open-source CalculiX evaluator was validated against Abaqus 2020 on identical decks at a
reference thickness:

    axial buckling           ccx 10.8395 vs  Abaqus 10.819   (0.19%)
    combined buckling        ccx  3.9924 vs  Abaqus  3.9208   (1.83%)
    first natural frequency  ccx 532.91  vs  Abaqus 531.69 Hz (0.23%)

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

AGGIORNATO 2026-08-26 anche sulle DUE RIGHE DI BUCKLING. Erano trascritte dal riferimento del
coautore (Abaqus 2020), e il combinato valeva 3.915. Rieseguite entrambe su Abaqus 2026 Learning
Edition, sullo STESSO deck a 661 nodi che questo bundle genera: assiale 10.819, identico alla cifra
del paper; combinato **3.9208**, non 3.9152.
⚠️ CORRETTO il 26/08 stesso: il 3.9152 NON era il riferimento Abaqus 2020 del coautore, come questa
nota diceva in una prima stesura. Era gia' un re-run su Abaqus LE 2026, fatto il 04/07 con un
generatore di deck diverso (`rr_shell_composite`, variante abq, non piu' nel bundle), che riprodusse
il 3.915 ereditato alla cifra stampata. La differenza dello 0.15% e' quindi fra DUE GENERATORI DI
DECK a parita' di solutore e di versione, non fra due versioni -- ed e' il motivo per cui conta
usare il deck che il bundle genera -- ed e' proprio il motivo per cui ora si riporta la coppia NOSTRA: il paper
afferma che i due solutori vedono lo stesso input, e solo cosi' l'affermazione e' verificabile. I due
file di risultati Abaqus sono versionati accanto (`data/exp1_abaqus_axial661.dat`,
`data/exp1_abaqus_combined661.dat`), come gia' fatto per la riga modale: il lato commerciale si
rilegge senza licenza. Con la coppia nostra lo scarto sul combinato e' 1.83%, non 1.98%. The reproducible CalculiX side of the cross-solver check -- the 60-ply cross-ply
baseline against Giacomo's Abaqus reference -- is in exp2_crossply_baseline.py. This file simply
documents the validation table.

Run:  python experiments/exp1_abaqus_validation.py
"""

ABAQUS = {
    "axial buckling":        (10.8395, 10.819),
    "combined buckling":     (3.9924, 3.9208),
    "first frequency [Hz]":  (532.91, 531.69),
}


def main():
    print("CalculiX 2.21 vs Abaqus 2026 LE, same 661-node deck, reference thickness:")
    for quantity, (ccx, abq) in ABAQUS.items():
        print(f"  {quantity:24s}: ccx {ccx:>8}  vs  Abaqus {abq:>8}   ({100 * abs(ccx - abq) / abq:.2f}%)")
    print("\nBoth sides re-run on the deck this bundle generates; the Abaqus result files are in data/.")


if __name__ == "__main__":
    main()
