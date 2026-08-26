# min-ply: how many plies does your laminate really need?

**The industrial question this tool answers:** given *your* stacking rules and *your* load cases,
what is the minimum number of plies that meets the buckling (and frequency, stress, displacement,
delamination) requirements --- computed with an open-source solver, on your machine, with no
commercial licence anywhere in the loop?

The entry point is the **min-ply sweep** (`experiments/exp16_minply_sweep.py`, which also generates
`data/tab_minply_generated.tex`): it walks the ply count downwards, runs the constrained search at
each thickness, gates the best design against the full requirement set, and stops at the thinnest
laminate that passes. The three bundled load cases and the two bundled orientation sets are
examples, not limits.

Everything runs on the open-source solver **CalculiX (`ccx`)**: a candidate stacking sequence is turned
into a CalculiX deck, solved, and its buckling factor is the fitness the metaheuristic search (GA / ACO /
PSO) maximises. The other constraints --- first frequency, in-plane stresses, tip displacement and the
interlaminar/delamination index --- are a gate applied to the best design of each ply count, not penalty
terms inside the fitness, and the manufacturing rules are enforced by the generation and repair
operators rather than by penalties. No commercial solver is needed.

## Bring your own design manual (the plug-in point)

The only thing this repository cannot ship is *your company's* stacking rules --- and that is a
plug-in point, not a limitation. The rule set lives in exactly two functions in
`optimisers/constrained_search.py`:

- **`manufacturing_ok(seq)`** --- the hard constructability rules checked on the full laminate
  (bundled defaults: symmetry handled by construction, max 3 consecutive plies of one
  orientation, max 45-degree jump between adjacent plies, surface-ply rules);
- **`guidelines_ok(seq, alpha)`** --- the composition guidelines checked on the counts
  (bundled defaults: balance of the +/- pairs, the 10% rule per direction).

Replace their bodies with your design manual and every layer above them --- generation, repair,
search, min-ply sweep, reporting --- follows without modification, because nothing else encodes a
rule. The two functions are pure predicates on a Python list of ply angles; a rule set that fits
that signature is enough. If you do this for a real design manual, the authors would genuinely
like to hear about it (contact below).

This bundle is also the reproducibility code for the paper *An open-source validated workflow for
minimum-thickness composite laminate design under buckling constraints* (Langiu, Rubino, Canale):
every number in the paper regenerates from the scripts listed in the tables below.

Licence: **MIT** for the code, **CC BY 4.0** for the result artefacts under `data/` and
`experiments/_out/` (see `LICENSE`). CalculiX itself is GPL software distributed by its own authors and
is neither included nor modified here.

## Install

```bash
conda env create -f environment.yml      # or: micromamba create -f environment.yml
conda activate composite-opt              # provides calculix (ccx), python, numpy, scipy
export CCX_BIN=ccx                        # the conda-forge calculix binary on PATH
ccx -v                                    # must print 2.21 -- see "Solver version" below
```

## Replication of results

One command regenerates the results, in two tiers:

```bash
./reproduce.sh quick     # minutes: validation, benchmark, experiment comparison, canaries, figures
./reproduce.sh full      # hours:   the optimisation campaigns (30-seed comparison, min-ply sweeps)
./reproduce.sh all       # both
```

Every script writes a JSON artefact under `data/` and prints what it regenerates. **No number in the
paper is hand-copied**: each one comes out of the script listed against it below.

### Solver version

All published numbers were produced with **CalculiX 2.21** (`CCX_BIN`); `environment.yml` does not pin
the version, so check `ccx -v` before comparing digits. `VALIDATION.md` records an independent re-run on
2.23: the core results reproduce, and the two cases where a number moved are documented there, with the
cause, rather than silently averaged away. Buckling and interlaminar solves are run with
`OMP_NUM_THREADS=1` on purpose (threaded buckling is unreliable below ccx 2.21); parallelism comes from
evaluating independent candidates concurrently, which does not change the search.

### Table / figure → script map

| Paper item | Label | Script | Artefact |
|---|---|---|---|
| Figures 1-3 | `fig:buck`, `fig:freq`, `fig:stress` | `figures/fig_fe_views.py` [^onescript] | `figures/_out/fig_ref_*.pdf` [^prov] |
| Table 6, first block | `tab:validation` | `experiments/exp1_abaqus_validation.py` | printed; see `VALIDATION.md` |
| Table 6, cross-ply block | `tab:validation` | `experiments/exp2_crossply_baseline.py` (all four comparisons, including the 6.9% torsion case) | printed; see `VALIDATION.md` |
| Table 7, Figure 5 | `tab:experimental`, `fig:experimental` | `experiments/exp9_experimental_validation.py` → `figures/fig_validation_parity.py` | `experiments/_out/exp9/` |
| Tables 8-9 | `tab:feasible`, `tab:explicitseq` | `experiments/exp3_minply_sequences.py` (+ `exp3b` for the C1 row) | `data/exp3_minply_sequences.json`, `data/exp3b_c1_freq_constrained.json` |
| Table 10 | `tab:axialsweep` | `experiments/exp3_minply_sequences.py` (`sweeps.set2.c1_axial`) | `data/exp3_minply_sequences.json` |
| Table 11 | `tab:minply` | `experiments/exp16_minply_sweep.py` | `data/exp16_minply_sweep.json`, `data/tab_minply_generated.tex` [^generated] |
| Table 12 | `tab:full` | `experiments/exp4_optimiser_comparison.py` | `data/exp4_optimiser_comparison.json` [^exp4] |
| Figure 6 | `fig:bench` | `experiments/exp6_haftka_walsh.py` → `figures/fig_benchmark_haftka_walsh.py` | `data/exp6_haftka_walsh.json` |
| Figure 7, panel A | `fig:neg` | `experiments/exp15_panelA_weakchop.py` → `figures/fig_pitfalls.py` | `data/exp15_panelA_weakchop.json` |
| Figure 7, panel B | `fig:neg` | `experiments/exp13_solid_buckling_spurious.py` → `figures/fig_pitfalls.py` | `data/exp13_solid_buckling_spurious.json` |
| Section 3.5 (certification, robustness) | --- | `experiments/exp7_fe_certification.py`, `exp8_robustness.py` | printed |
| Section 3.8 (isotropic canaries) | --- | `experiments/exp14_isotropic_canaries.py` | printed |
| Section 3.8 (fourth failure mode) + abstract | --- | `experiments/exp18_reference_load_screen.py` | `data/exp18_reference_load_screen.json` |
| Section 3.6 (budget convergence at N=48) | `sec:sweep` | `experiments/exp19_budget_convergence.py` | `data/exp19_budget_convergence.json` |
| Section 3.4 (extended orientation set) | `sec:alphabet` | `experiments/exp4_optimiser_comparison.py` (sweeps set1 and set2 in one run) | `data/exp4_optimiser_comparison.json` [^exp4] |

[^onescript]: One script produces all three: two eigenvalue solves on the S8R shell plus a static
    solve on the C3D8I solid.
[^prov]: Provenance in `data/fig_fe_views_provenance.json`; a 1% canary on the buckling factor and
    on the first frequency, against Table 6, refuses to write the figure if it drifts.
[^generated]: **Both files are written by the run.** The sweep is the generator: the table in the
    paper is regenerated from it, never transcribed.
[^exp4]: Keys `set1`/`set2`, with `per_seed`, median/IQR and the Holm-corrected tests.
    **`data/exp4_optimiser_comparison_set2.json` is SUPERSEDED**: it predates the deterministic-seed
    fix and its C1 omnibus (p=0.0398) contradicts the published one (p=0.83). It is kept only as a
    record of the earlier revision --- do not verify against it.

> **Artefact-name drift in exp4, and which file the paper is actually on.** Two files in `data/` were
> written by an earlier revision of the script that ran one alphabet at a time:
> `exp4_optimiser_comparison_set2.json` and `exp5_alphabet_set1.json`. **Both are SUPERSEDED and neither
> reproduces the published numbers.** The script in this bundle runs both alphabets in one pass and writes
> a single `exp4_optimiser_comparison.json` with `set1`/`set2` keys, and that is the file Section 3.4 is
> on: it gives best $4.6468 \to 4.7222$ on C1, $2.2627 \to 2.5391$ on C2 and $2.1309 \to 2.2016$ on C3,
> against the paper's 4.65/4.72, 2.26/2.54 and 2.13/2.20; the C1 seed-means give $3.3316 \to 4.0984$,
> i.e. the published $3.33 \to 4.10$ and $+23.0\%$, with $+17.1\%$ / $+18.8\%$ / $+34.9\%$ per optimiser
> and $+10.2\%$ / $+6.9\%$ on C2 / C3. The legacy `exp5_alphabet_set1.json` does NOT: it gives 4.6439 on
> C1 and 2.1147 on C3, which round to 4.64 and 2.11 and contradict the table. Verified 2026-07-25 by
> recomputing every figure of Section 3.4 from both files. Do not verify against either legacy file; they
> are kept as a record of the earlier revision. Being a 30-seed stochastic campaign, a re-run reproduces
> the ranking and the significance pattern, not the digits (`VALIDATION.md`).

> **Re-deriving the statistics without re-running the campaign.**
> `python experiments/exp4_optimiser_comparison.py --restat` recomputes every summary statistic and
> every test in `data/exp4_optimiser_comparison.json` from the `per_seed` records the file already
> holds: no FE solve, no search, seconds instead of 540 `ccx` runs. The 30 per-seed buckling factors
> are the raw data of this experiment; mean, standard deviation, median, IQR, Kruskal--Wallis,
> Mann--Whitney and Holm are functions of them alone. Added for audit finding C2 (2026-07-22): the Holm
> correction had been applied to p-values already rounded to four decimals, which published
> $4\times10^{-4}$ where the raw value gives $3.46\times10^{-4}$ on the GA-ACO shear comparison. The
> verdict was unchanged, but the defect was in the generator, so the fix belongs there and the numbers
> had to be regenerated from the raw record rather than corrected in the manuscript alone. All p-values
> are now stored to four *significant figures*: on a log scale, four decimals turns $6.5\times10^{-6}$
> into `0.0`.

Tables 1-6 are inputs, not results: they state the load cases, the constraints, the material data and the
optimiser settings, and are read from the manuscript, not generated.

Not reproducible from this bundle, and said so in the paper: the 41.96/41.91 Hz row of Table 7, measured
on a 928-node crop of the coauthor's original Abaqus model of the industrial case. The other frequency
row, on the cross-ply anchor, is reproducible (`exp2`).

Run any script from the repository root so the `fe` and `optimisers` packages are importable
(`PYTHONPATH=$PWD`).

## Layout

```
fe/                       finite-element evaluators (all CalculiX-based)
  plate_model.py          2D S8R composite-shell deck + buckling/frequency eval   (was canale_plate)
  interlaminar.py         3D solid peel/delamination index Q (free-edge recovery)  (was rr_interlaminar)
  ccx_runner.py           run a ccx deck and parse its .dat/.frd results           (was generic_run)
  materials.py            orthotropic lamina registry
  frd_parse.py            CalculiX .frd result parser
optimisers/
  metaheuristics.py       GA / ACO / PSO over the (symmetric) ply sequence         (was rr_metaheuristics)
  constrained_search.py   constrained min-ply search, load cases, fitness          (was rr_optimiser)
experiments/              one script per paper experiment (see below)
data/                     regenerated result JSON (committed)
```

The laminates are **mid-plane symmetric**: the optimisers search the half-stack and the full
laminate is the half followed by its mirror.

## Experiments (each reproduces a result in the paper)

| Script | Reproduces | Needs ccx |
|--------|------------|-----------|
| `experiments/exp1_abaqus_validation.py` | ccx-vs-Abaqus validation deck (Abaqus side is the coauthor's reference) | yes |
| `experiments/exp2_crossply_baseline.py` | 60-ply cross-ply baseline vs the Abaqus reference | yes |
| `experiments/exp3_minply_sequences.py`  | the six delivered minimum-ply stacking sequences (paper Table 5) | yes |
| `experiments/exp4_optimiser_comparison.py` | 30-seed GA/ACO/PSO comparison + restricted-vs-extended alphabet | yes |
| `experiments/exp6_haftka_walsh.py`      | Haftka-Walsh benchmark: recover 12/12 + 8/8 global optima | no (closed-form) |
| `experiments/exp7_fe_certification.py`  | exhaustive N=8 certification: GA within 0.51% of the FE global optimum | yes |
| `experiments/exp8_robustness.py`        | buckling-evaluator robustness (the governing mode is the lowest eigenvalue) | yes |

Run, e.g.:

```bash
PYTHONPATH=$PWD NPROC=8 python experiments/exp3_minply_sequences.py
```

The heavier experiments (exp3, exp4) are embarrassingly parallel over candidates/seeds and were
run on a multi-core node; set `NPROC` to the number of cores.

## Validation

`VALIDATION.md` records the result-vs-paper comparison from an independent re-run on a different
CalculiX version (2.23 vs the 2.21 of the paper); the core results reproduce exactly.

## Contact

Alessio Langiu --- Institute of Marine Sciences (ISMAR), National Research Council of Italy (CNR),
Via Archirafi 11, 90123 Palermo, Italy --- <alessio.langiu@cnr.it>. This is the address to write to
about the plug-in point above, about running the tool on a different design manual, or about
anything else in this bundle: a question reaches a person, not a repository.

## Citing

Archived release: **DOI [10.5281/zenodo.22109868](https://doi.org/10.5281/zenodo.22109868)** (concept
DOI --- it always resolves to the latest version; cite this one).

If you use this code, please cite the paper. Load cases and the structural-integrity model
follow the reference cases of Canale et al., *The Open Mechanical Engineering Journal* 12 (2018).

### What cannot run from a clean clone

Two steps need `decks/Composite_buckling_3.inp`, the Abaqus deck of the industrial reference case, which
is not versioned in this repository (it is the coauthor's model): `exp15_panelA_weakchop` and
`exp13_solid_buckling_spurious` (panels A and B of the pitfalls figure). `reproduce.sh` reports them as
SKIPPED and carries on with the rest; set `RR_DECK` to point at your own copy to run them. Everything
else in `quick` runs from what is in this repository.

The Abaqus side of any cross-solver comparison is likewise not reproducible here: Abaqus is commercial,
and its numbers are the coauthor's reference runs, quoted in `VALIDATION.md` with their provenance.

## Provenance of artefacts that are NOT regenerated by this bundle

Declared here so that no gap is silent (audit 2026-07-20, section E):

- **Five figures are the coauthor's Abaqus views**, with no generator script in this repository:
  `gc_buckling.png`, `gc_freq.png`, `gc_model2d.png`, `gc_model3d.png`, `gc_stress.png`. They are
  screenshots of the coauthor's models, not data plots; the request to rescale the colour code of
  one of them (U Magnitude 284.71 -> 1) is still with the coauthor. Every DATA figure in the paper
  has its generator listed in the tables above.
- **The 928-node modal crop** behind the frequency row of the industrial case is derived from the
  coauthor's Abaqus deck (see *What cannot run from a clean clone* above) and is not in the bundle.
- **`data/exp3_minply_sequences.json` mixes two passes, and the reason is a bug that is now
  fixed.** Its `sweeps` rows all carry `Q: null` and `feasible: false`, including the exact ply
  counts and buckling factors that the `delivered` block reports as feasible with a real `Q`.
  The cause was a bare `except Exception: Q = None` around the interlaminar solve in
  `experiments/exp3_minply_sequences.py`: when the solid solver was unavailable, a solve that
  never ran was recorded exactly like one that ran and failed, the ascending sweep never met a
  feasible row, and it therefore ran to the manufacturing cap instead of stopping at the minimum
  thickness. `delivered` comes from a pass in which the solver was available. The delivered ply
  counts and buckling factors are the published ones and are unaffected; what the file does not
  support is the reading of its `sweeps` block as a feasibility scan. The handler now records
  `delam_error` in the row, prints it, and re-raises unless `EXP3_ALLOW_DELAM_FAILURE=1` is set,
  so this cannot recur silently; re-running the experiment regenerates a single-pass file.

- **The seven "Source's own" values of NASA TP-3007 are verified against the source** (checked
  2026-08-26 on the public NTRS scan, <https://ntrs.nasa.gov/api/citations/19900016761/downloads/19900016761.pdf>).
  They are printed in **Table 8, "Analytic Buckling Loads, Critical End-Shortenings, and Nominal
  Thicknesses for Plates Without Cutouts"**, page 17, and every one matches the value used here:
  aluminium 1773, [0_10]s 9272, [90_10]s 2473, [(0/90)_5]s 6544, [(+-30)_6]s 9898, [(+-45)_6]s
  10962, [(+-60)_6]s 5944 lb. The nominal thicknesses in the same table (0.0647, 0.1100, 0.1100,
  0.1100, 0.1176, 0.1300, 0.1176 in.) match the ply thicknesses used here as well. The measured
  loads of Tables 2-7 and the three rows of TP-2528 were already verified.
  NOTE: the audit of 2026-07-20 recorded these seven as "not printed in any table, plausibly
  digitised from figures, NOT VERIFIED". That was wrong, and the entry is withdrawn: they are
  tabulated. The mistake is instructive -- the report is a 1990 scan whose OCR renders the values
  with a thousands space ("9 272") and one measured load with a capital O for the final zero
  ("579O"), so a plain text search for the digits finds nothing and reads as absence.
