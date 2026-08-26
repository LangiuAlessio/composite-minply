# Reproducibility bundle — composite-opt-2026 (paper v2, Canale/Langiu/Rubino)

Goal: a clean, public-GitHub-ready code bundle that reproduces EVERY experiment in the
paper `composite_opt_v3_jcs.tex`, with thorough review, comments, meaningful file names,
and a full re-run of all experiments to validate both the results and the code.

## Compute
- **gw1** (ailab k3s control-plane, 32 cores): ccx 2.23 via conda env `fe`, code synced to
  `~/fe-batch-lab`. FE buckling eval ~0.6 s. Run with
  `MAMBA_ROOT_PREFIX=$HOME/micromamba PYTHONPATH=$HOME/fe-batch-lab CCX_BIN=ccx \
   $HOME/bin/micromamba run -n fe python ...`. Wake gw2-gw8 via `ailab-wake` if more cores needed.
- A Windows workstation holding the **Abaqus** licence, for the ccx-vs-Abaqus validation experiment.
- **Mac**: NO heavy compute (thermal). Editing/orchestration only.
- ⚠️ ccx 2.21 (paper) vs 2.23 (gw1): re-runs must confirm the headline numbers still hold.

## Bundle layout (target, with renames)
```
code/
├── README.md            # problem, install (conda calculix), how to run each experiment
├── environment.yml      # conda-forge: calculix, python, numpy, scipy, pandas
├── fe/
│   ├── plate_model.py        <- canale_plate.py      (S8R shell buckling/freq deck + eval)
│   ├── interlaminar.py       <- rr_interlaminar.py   (3D solid peel/delamination Q)
│   ├── ccx_runner.py         <- generic_run.py       (run a ccx deck, parse results)
│   └── materials.py          <- materials.py         (orthotropic lamina registry)
├── optimisers/
│   ├── metaheuristics.py     <- rr_metaheuristics.py (GA / ACO / PSO, symmetric half-stack)
│   └── constrained_search.py <- rr_optimiser.py      (min-ply constrained search, CASES)
├── experiments/
│   ├── exp1_abaqus_validation.py   (ccx vs Abaqus: 10.84/10.819, 3.985/3.915, 41.96/41.91Hz)
│   ├── exp2_crossply_baseline.py   (60-ply cross-ply: BF 14.63/3.59/2.41, f 614.7 Hz)
│   ├── exp3_minply_sequences.py    (#8: 6 delivered sequences, C1/C2/C3 x 2 alphabets)
│   ├── exp4_optimiser_comparison.py(30-seed GA/ACO/PSO, Kruskal-Wallis + Dunn)
│   ├── (exp5 does not exist as a file: the restricted-vs-extended alphabet study is
│   │    exp4_optimiser_comparison.py run over BOTH alphabets in one execution -- see the
│   │    SUPERSEDED note in README.md about the legacy exp5_alphabet_set1.json)
│   ├── exp6_haftka_walsh.py        <- bench_haftka_walsh.py (12/12 max-buckling, 8/8 min-thk)
│   ├── exp7_fe_certification.py    <- rr_certify_fe.py (exhaustive N=8, GA within 0.51%)
│   └── exp8_robustness.py          <- rr_buckling_mode_diag.py (weak-chop / C3D8I / free-edge)
├── data/                # regenerated result JSON per experiment (committed)
└── figures/             # figure generation from data (Giacomo's Abaqus figs stay in paper)
```

## Workflow per experiment (couples cleaning + validation)
1. Move + rename source -> bundle path; review thoroughly; add docstring/comments; remove dead code.
2. Run the CLEANED file on gw1 (off-Mac).
3. Compare result to the paper number; record PASS/diff in `data/<exp>.json` + a row in `VALIDATION.md`.
4. If a number drifts (ccx 2.23 vs 2.21), flag it for the manuscript.

## Status
- [x] Compute env on gw1 (ccx 2.23, code synced, smoke OK)
- [x] Bundle skeleton
- [ ] fe/ cleaned + commented
- [ ] optimisers/ cleaned + commented
- [ ] exp1 Abaqus validation (needs a machine with an Abaqus licence)
- [ ] exp2 cross-ply baseline
- [ ] exp3 min-ply 6 sequences  <-- Giacomo's #8, highest priority
- [ ] exp4 optimiser comparison (30-seed)
- [ ] exp5 alphabet study
- [ ] exp6 Haftka-Walsh benchmark
- [ ] exp7 FE certification
- [ ] exp8 robustness controls
- [ ] README + environment.yml
- [ ] VALIDATION.md (results-vs-paper table)
- [ ] wire GitHub URL into paper (Giacomo #5) once repo is created
