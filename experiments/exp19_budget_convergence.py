#!/usr/bin/env python3
"""exp19 -- the budget-convergence check at N=48, which the manuscript quoted without an artefact.

THE generator of the sentence in Section 3.6 ("A convergence check at N=48 raises the best
buckling factor ... as the budget grows") and, through it, of "not yet at plateau" in the
Conclusions. The 2026-07-22 audit found that check had NEITHER script NOR data anywhere in the
leaf, and that it was the ONLY support for the plateau claim: either it gets generated or the
claim goes. This generates it.

What it measures. One guided GA per seed on the pure axial case at N=48, buckling only (the
same setting as exp16, so the two are comparable by construction), run to 2600 evaluations
with the incumbent recorded after the initial population and after every generation. Reading
one long run instead of re-running each budget from scratch is not a shortcut: the GA is
deterministic given its seed, so the run at 1000 evaluations IS the prefix of the run at 2600,
and re-running it would reproduce the same numbers at four times the cost.

The canary is the strongest one available in this bundle. At 1000 evaluations (pop 40 x
[1 + 24 generations]) this is exactly the configuration of exp16, seed for seed, so the curve
must pass through the published sweep values at N=48 -- 4.8955 / 4.6679 / 4.7191 for seeds
1/2/3. If it does not, either the optimiser has drifted since the sweep was run or the sweep is
not reproducible on this machine, and in both cases the budget curve is not to be trusted
either. That check is run automatically here.

Cost: 3 seeds x 2600 evaluations at ~0.27 s each on 8 workers, about 35 minutes. This is not
the exp4 or exp16 campaign; it is one point of the grid taken deeper.

Run:  CCX_BIN=ccx_2.21 NPROC=8 SEEDS=3 PYTHONPATH=<repo>/code \\
      python3 -m experiments.exp19_budget_convergence
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import statistics
import time

from optimisers.constrained_search import ga_best, CASES, ALPHABETS

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data',
                    'exp19_budget_convergence.json')
CASE, N_PLIES, ALPHABET = 'c1_axial', 48, 'set2'
POP, GENS = 40, 64                 # 40 x 65 = 2600 evaluations
N_SEEDS = int(os.environ.get('SEEDS', '3'))
NPROC = int(os.environ.get('NPROC', '8'))
REPORT_AT = [40, 80, 240, 1000, 2600]
# exp16, sweeps/c1_axial at n_plies=48, per_seed. The budget curve must pass through these at
# 1000 evaluations, because that is the same configuration with the same seed.
EXP16_AT_1000 = {1: 4.8955, 2: 4.6679, 3: 4.7191, 4: 4.7439, 5: 4.9341}
TOL = 1e-3


def read_at(trace: list, budget: int) -> float | None:
    """Best incumbent once `budget` evaluations have been spent (None if never reached)."""
    hits = [bf for n, bf in trace if n <= budget]
    return max(hits) if hits else None


def main() -> None:
    case, alpha = CASES[CASE], ALPHABETS[ALPHABET]
    pool = mp.Pool(NPROC)
    print(f'=== budget convergence, {CASE} N={N_PLIES}, guided GA pop={POP} gens={GENS} '
          f'({POP * (GENS + 1)} evaluations), {N_SEEDS} seeds, {NPROC} workers')

    runs, t0 = [], time.time()
    for seed in range(1, N_SEEDS + 1):
        trace: list = []
        ts = time.time()
        bf, seq = ga_best(case, N_PLIES, alpha, pool, random.Random(seed),
                          pop=POP, gens=GENS, guided=True, trace=trace)
        curve = {b: read_at(trace, b) for b in REPORT_AT}
        runs.append({'seed': seed, 'final_buckling': round(bf, 4),
                     'half_stack': seq,
                     'curve': {str(b): (round(v, 4) if v is not None else None)
                               for b, v in curve.items()},
                     'trace': [[n, round(v, 4)] for n, v in trace],
                     'secs': round(time.time() - ts, 1)})
        pretty = '  '.join(f'{b}:{curve[b]:.4f}' for b in REPORT_AT if curve[b] is not None)
        print(f'  seed {seed}: {pretty}   ({time.time() - ts:.0f}s)')

    best = {b: max(r['curve'][str(b)] for r in runs) for b in REPORT_AT}
    mean = {b: statistics.fmean(r['curve'][str(b)] for r in runs) for b in REPORT_AT}
    out = {'case': CASE, 'n_plies': N_PLIES, 'alphabet': ALPHABET,
           'config': {'pop': POP, 'gens': GENS, 'n_seeds': N_SEEDS,
                      'evaluations': POP * (GENS + 1), 'constraint': 'buckling only',
                      'ccx': __import__('fe.ccx_bin', fromlist=['resolved']).resolved(), 'nproc': NPROC},
           'report_at': REPORT_AT,
           'best_by_budget': {str(b): round(best[b], 4) for b in REPORT_AT},
           'mean_by_budget': {str(b): round(mean[b], 4) for b in REPORT_AT},
           'gain_best_pct': round((best[REPORT_AT[-1]] / best[REPORT_AT[0]] - 1) * 100, 1),
           'runs': runs, 'total_secs': round(time.time() - t0, 1)}
    with open(DATA, 'w') as f:
        json.dump(out, f, indent=1)

    print('\n  budget   best   mean')
    for b in REPORT_AT:
        print(f'  {b:6d}  {best[b]:.4f}  {mean[b]:.4f}')
    print(f"\nwrote {os.path.relpath(DATA)}  ({out['total_secs'] / 60:.1f} min)")

    # --- canaries -----------------------------------------------------------------------
    fail = []
    for r in runs:
        got, want = r['curve']['1000'], EXP16_AT_1000.get(r['seed'])
        if want is not None and (got is None or abs(got - want) > TOL):
            fail.append(f"seed {r['seed']}: at 1000 evaluations this run gives {got}, the "
                        f"published exp16 sweep gives {want} for the same seed and the same "
                        f"configuration -- one of the two is not reproducible")
    for r in runs:
        vals = [v for _, v in r['trace']]
        if any(b < a - 1e-9 for a, b in zip(vals, vals[1:])):
            fail.append(f"seed {r['seed']}: the incumbent decreases along the run, which an "
                        f"elitist GA cannot do -- the trace is not a budget curve")
    if best[REPORT_AT[-1]] < best[REPORT_AT[0]]:
        fail.append('more budget did not help at all: there is no curve to report')
    if fail:
        raise SystemExit('exp19 canary FAILED:\n  - ' + '\n  - '.join(fail))
    print(f'canaries: curve passes through the published exp16 values at 1000 evaluations '
          f'({", ".join(f"seed {r["seed"]}={r["curve"]["1000"]}" for r in runs)}), '
          f'and every trace is non-decreasing')


if __name__ == '__main__':
    main()
