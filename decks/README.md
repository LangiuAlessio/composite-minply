# Source Abaqus decks (input data)

One experiment reads an Abaqus deck of the industrial reference case that is **not distributed
here**, because it is a coauthor's model rather than an artefact of this work:

    Composite_buckling_3.inp    (60-ply C3D8I solid, ~14k nodes, load case 3 / axial)

It is `experiments/exp15_panelA_weakchop.py`, which produces panel (A) of the pitfalls figure.
`reproduce.sh` reports it as SKIPPED and carries on with everything else; that panel is the only
result in the paper that a clean clone cannot regenerate, and the paper says so. Panel (B) does not
need it: `experiments/exp13_solid_buckling_spurious.py` builds its own deck.

Nothing in this study is covered by copyright or by any proprietary restriction: the deck is simply
not ours to redistribute. To reproduce that panel, put your own deck in this directory, or
point `RR_DECK` at it, and run it on a machine with CalculiX:

    CCX_REMOTE=<host> ./run_on_head.sh experiments.exp15_panelA_weakchop
