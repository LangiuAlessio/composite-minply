#!/usr/bin/env bash
# reproduce.sh -- one command to regenerate the results of the paper.
#
#   ./reproduce.sh quick    minutes: validation, benchmark, experimental comparison, canaries, figures
#   ./reproduce.sh full     hours:   the optimisation campaigns (30-seed comparison, min-ply sweeps)
#   ./reproduce.sh all      quick, then full
#
# Environment (see README.md "Replication of results"):
#   conda env create -f environment.yml && conda activate composite-opt
#   export CCX_BIN=ccx        # CalculiX 2.21 -- the version every published number was produced with
#   export NPROC=30           # cores used by the embarrassingly parallel FE evaluations
#
# Every script writes a JSON artefact under data/ (or experiments/_out/) and prints the numbers it
# regenerates. The table/figure -> script map is in README.md; no published number is hand-copied.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export PYTHONPATH="$HERE"
export CCX_BIN="${CCX_BIN:-ccx}"
export NPROC="${NPROC:-$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu)}"

# `ccx -v` exits non-zero and prints a blank first line, so with `set -o pipefail` the naive
# version probe reported "NOT FOUND" for a solver that was there. Swallow the status, grep the banner.
ccx_version() { { "$CCX_BIN" -v 2>&1 || true; } | grep -i -m1 -E 'version' || echo "ccx NOT FOUND"; }

FAILED=()
SKIPPED=()

# A step that cannot run must not take the whole reproduction down with it: some experiments need
# an input that is not redistributable (see "What cannot run from a clean clone" in README.md), and
# a referee running this script wants the other eleven results, not an early exit. Failures are
# collected and reported at the end, and the script exits non-zero if there were any.
run() {   # run <module> <what it reproduces>
  echo ""
  echo "=== $1 -- $2"
  local t0 t1 rc
  t0=$(date +%s)
  set +e
  # ⚠️ NON fra virgolette: uno step puo' portarsi dietro i propri argomenti (exp9 vuole `--all`,
  # senza il quale non scrive exp9_all_sources.json e la figura che lo legge fallisce). Quotato,
  # `-m` riceverebbe "modulo --all" come unico nome di modulo.
  # shellcheck disable=SC2086
  python3 -u -m $1
  rc=$?
  set -e
  t1=$(date +%s)
  if [ "$rc" -eq 0 ]; then
    echo "--- $1 done in $((t1 - t0))s"
  elif [[ " $NEEDS_DECK " == *" $1 "* ]] && [ ! -f "$HERE/decks/Composite_buckling_3.inp" ]; then
    echo "--- $1 SKIPPED: needs decks/Composite_buckling_3.inp, which is not redistributable"
    SKIPPED+=("$1")
  else
    echo "--- $1 FAILED (exit $rc) after $((t1 - t0))s"
    FAILED+=("$1")
  fi
}

# Steps whose only blocker on a clean clone is the non-redistributable deck. A plain space-delimited
# string, not an associative array: macOS still ships bash 3.2, where `declare -A` is a syntax error.
NEEDS_DECK="experiments.exp15_panelA_weakchop"

QUICK=(
  "experiments.exp1_abaqus_validation|tab:validation, buckling rows"
  "experiments.exp2_crossply_baseline|60-ply cross-ply anchor (VALIDATION.md)"
  "experiments.exp6_haftka_walsh|fig:bench, Haftka-Walsh global optima"
  "experiments.exp7_fe_certification|exhaustive N=8 certification (Section 3.5)"
  "experiments.exp8_robustness|buckling-evaluator robustness (Section 3.5)"
  "experiments.exp9_experimental_validation --all|tab:experimental + fig:experimental"
  "experiments.exp14_isotropic_canaries|isotropic canaries (the verification-layer section)"
  "experiments.exp18_reference_load_screen|reference-load screen, 4th failure mode (the verification-layer section) + abstract"
  "experiments.exp15_panelA_weakchop|fig:neg, panel A"
  "experiments.exp13_solid_buckling_spurious|fig:neg, panel B"
  "figures.fig_validation_parity|fig:experimental"
  "figures.fig_benchmark_haftka_walsh|fig:bench"
  "figures.fig_pitfalls|fig:neg"
)

FULL=(
  "experiments.exp3_minply_sequences|tab:feasible, tab:explicitseq + tab:axialsweep"
  "experiments.exp3b_c1_freq_constrained|the C1 48-ply design under the frequency constraint"
  "experiments.exp16_minply_sweep|tab:minply + data/tab_minply_generated.tex"
  "experiments.exp19_budget_convergence|the N=48 budget-convergence check of the budget-convergence subsection (~46 min, 3 seeds)"
  "experiments.exp4_optimiser_comparison|tab:full, 30-seed GA/ACO/PSO comparison"
)

main() {
  local mode="${1:-quick}"
  echo "solver : $(ccx_version)   [the paper used CalculiX 2.21]"
  echo "cores  : $NPROC"
  echo "mode   : $mode"
  local -a queue=()
  case "$mode" in
    quick) queue=("${QUICK[@]}") ;;
    full)  queue=("${FULL[@]}") ;;
    all)   queue=("${QUICK[@]}" "${FULL[@]}") ;;
    *) echo "usage: $0 [quick|full|all]" >&2; exit 2 ;;
  esac
  for entry in "${queue[@]}"; do
    run "${entry%%|*}" "${entry##*|}"
  done
  echo ""
  if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo "SKIPPED (missing non-redistributable input): ${SKIPPED[*]}"
  fi
  if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED: ${FAILED[*]}"
    echo "Artefacts of the successful steps are in data/ and experiments/_out/."
    return 1
  fi
  echo "All done. Artefacts in data/ and experiments/_out/."
}

main "$@"
