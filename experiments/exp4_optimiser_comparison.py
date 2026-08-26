"""exp4_optimiser_comparison.py -- symmetric GA/ACO/PSO comparison + alphabet contrast.

Compares the three metaheuristics over 30 random seeds at a fixed thickness (N=40, the design
load) for the restricted {0,+/-45,90} and the extended {0,+/-30,+/-45,+/-60,90} alphabets, with
each optimiser searching the SYMMETRIC half-stack (the full laminate is the half followed by its
mirror, as everywhere in the paper).

Reproduces:
  - the paper's optimiser-comparison table (per case: mean+/-std (best) and a Kruskal-Wallis
    omnibus p, with post-hoc pairwise Mann-Whitney);
  - the alphabet contrast (Section 3.4): the extended alphabet improves the best buckling factor
    on the shear and combined cases, not on pure axial.

Run:  PYTHONPATH=$PWD NPROC=8 python experiments/exp4_optimiser_comparison.py
"""
from __future__ import annotations

import json
import os
import random
import statistics as st
import sys
import multiprocessing as mp

from scipy.stats import kruskal, mannwhitneyu

from optimisers.constrained_search import buckling_factor as _real_bf, CASES, ALPHABETS
import optimisers.metaheuristics as mh

N = 40
HALF = N // 2


def sig(x: float, digits: int = 4) -> float:
    """Round to `digits` SIGNIFICANT figures, not to `digits` decimals.

    p-values live on a log scale: rounding 1.731e-4 to four decimals gives 2e-4 and rounding
    6.5e-6 gives 0.0, which is not a number a reader can check. Every p-value stored by this
    script goes through here.
    """
    return float(f"{x:.{digits}g}")


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni step-down correction over a family of RAW p-values.

    Sort ascending, multiply the k-th smallest (0-based) by (m - k), then enforce
    monotonicity by a running maximum, and clip at 1. Returns {label: p_holm}.

    CALLER'S CONTRACT: pass the raw p-values, never rounded ones. Until the 2026-07-22 audit
    (finding C2) this function was fed the 4-decimal *printed* values, so the GA-ACO comparison
    on the shear case was corrected as 2 x round(1.731e-4, 4) = 4.0e-4 and the manuscript
    published 4e-4 instead of the true 3.462e-4. The verdict was unaffected (both are far below
    0.05), but a published number was wrong at the generator, not at the printer. The result is
    rounded on the way OUT, and to significant figures.
    """
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for k, (label, p) in enumerate(items):
        running = max(running, (m - k) * p)
        out[label] = sig(min(running, 1.0))
    return out
SEEDS = list(range(1, 31))
# Deterministic per-optimiser seed offsets (replaces hash(opt) % 7, which was process-dependent).
# The three optimisers deliberately do NOT share a seed: they consume randomness differently, so a
# common seed would pair nothing anyway; what matters is that the set is fixed across runs.
OPT_OFFSET = {"GA": 1, "ACO": 3, "PSO": 5}
CASES_ORDER = ("c1_axial", "c2_side", "c3_combo")


def _bf_symmetric(args):
    """Buckling factor of the symmetric laminate built from a half-stack (half + mirror)."""
    half, case = args
    return _real_bf((list(half) + list(half)[::-1], case))


# The optimisers search the half-stack; this scores its mirror-symmetric full laminate.
mh.buckling_factor = _bf_symmetric


def _run(opt, case, pool, rng):
    if opt == "GA":
        return mh.ga(case, HALF, pool, rng, pop=16, gens=10)[0]
    if opt == "ACO":
        return mh.aco(case, HALF, pool, rng, ants=16, iters=10)[0]
    return mh.pso(case, HALF, pool, rng, swarm=16, iters=10)[0]


def compare(alpha_name, pool):
    """Run the 30-seed GA/ACO/PSO comparison for one alphabet; return per-case stats + tests."""
    mh.ALPHA = ALPHABETS[alpha_name]
    out = {}
    for cname in CASES_ORDER:
        case = CASES[cname]
        vals = {"GA": [], "ACO": [], "PSO": []}
        for seed in SEEDS:
            for opt in ("GA", "ACO", "PSO"):
                # Seed offset per optimiser: a FIXED table, not hash(opt). Python randomises string
                # hashing per process (PYTHONHASHSEED), so the previous `hash(opt) % 7` drew a different
                # seed set on every invocation and NOTHING in this table was regenerable -- the one thing
                # a "Replication of results" section must guarantee. Found 2026-07-21 in review.
                vals[opt].append(round(_run(opt, case, pool,
                                            random.Random(seed * 10 + OPT_OFFSET[opt])), 4))
        out[cname] = summarise(vals)
        print(f"[{alpha_name}] {cname}: " +
              " ".join(f"{o}={out[cname][o]['mean']:.2f}({out[cname][o]['best']:.2f})"
                       for o in ("GA", "ACO", "PSO")) +
              f"  KW p={out[cname]['KW']['p']:.4g}", flush=True)
    return out


def summarise(vals: dict) -> dict:
    """Summary statistics and tests for one case, from the 30 per-seed buckling factors.

    Kept separate from the search so that `--restat` can regenerate every published statistic
    from the per-seed record already in the JSON, without re-running the 540-solve campaign.
    """
    _, p = kruskal(vals["GA"], vals["ACO"], vals["PSO"])
    # RAW pairwise p-values: Holm is corrected on these, and only the stored copy is rounded.
    pair_raw = {f"{a}-{b}": float(mannwhitneyu(vals[a], vals[b]).pvalue)
                for a, b in (("GA", "ACO"), ("GA", "PSO"), ("ACO", "PSO"))}
    # The 30 per-seed values are KEPT (audit debt E7, closed 2026-07-21): the published
    # records held only mean/std/best, which is why the bimodal C1 distribution could not be
    # summarised by median/IQR without re-running the whole campaign. Never drop them again.
    # Stored at 4 decimals, not 3: the manuscript prints these at 2, and a value saved as
    # 2.495 re-rounds to 2.50 while the mean over the seeds is 2.4947, i.e. 2.49. Same
    # double-rounding class as the Holm defect above (audit 2026-07-22, C3).
    out = {o: {"mean": round(st.mean(vals[o]), 4), "std": round(st.pstdev(vals[o]), 4),
               "best": round(max(vals[o]), 4),
               "median": round(st.median(vals[o]), 4),
               # q1/q3 by statistics.quantiles (exclusive/Weibull), NOT the linear
               # convention of numpy/R type 7: stated because the two differ here.
               "q1": round(st.quantiles(vals[o], n=4)[0], 4),
               "q3": round(st.quantiles(vals[o], n=4)[2], 4),
               "per_seed": vals[o]} for o in ("GA", "ACO", "PSO")}
    out["KW"] = {"p": sig(p, 6)}
    out["MWU"] = {k: sig(v) for k, v in pair_raw.items()}
    # Holm-Bonferroni over the three pairwise comparisons of THIS case. The manuscript
    # quotes Holm-corrected p-values; before 2026-07-20 this script emitted only the raw
    # Mann-Whitney values and the correction was applied by hand, so the published numbers
    # (correct, but hand-computed) were not reproducible from the bundle. Audit finding B8.
    out["Holm"] = holm(pair_raw)
    return out


DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                         "exp4_optimiser_comparison.json")


def restat():
    """Recompute every statistic in the published JSON from its own per-seed record.

    No FE solve, no search: the 30 per-seed buckling factors ARE the raw data of this
    experiment, and mean/std/median/IQR, Kruskal-Wallis, Mann-Whitney and Holm are all
    functions of them. Added for audit finding C2 (2026-07-22), where the Holm correction had
    been applied to already-rounded p-values and one published number was wrong at the
    generator: the fix has to be regenerated from the raw data, not patched into the print.
    """
    path = os.path.abspath(DATA_PATH)
    res = json.load(open(path))
    for alpha_name, cases in res.items():
        for cname, blk in cases.items():
            vals = {o: blk[o]["per_seed"] for o in ("GA", "ACO", "PSO")}
            new = summarise(vals)
            for key in ("KW", "MWU", "Holm"):
                if blk[key] != new[key]:
                    print(f"  [{alpha_name}] {cname} {key}: {blk[key]} -> {new[key]}")
            for o in ("GA", "ACO", "PSO"):
                changed = {k: (blk[o][k], new[o][k]) for k in ("mean", "std", "best", "median",
                                                               "q1", "q3") if blk[o][k] != new[o][k]}
                if changed:
                    print(f"  [{alpha_name}] {cname} {o}: {changed}")
            cases[cname] = new
    json.dump(res, open(path, "w"), indent=1)
    print(f"restat: {path} rewritten from its own per-seed records (no FE solve).")


def main():
    pool = mp.get_context("fork").Pool(int(os.environ.get("NPROC", str(os.cpu_count()))))
    res = {}
    try:
        for alpha_name in ("set2", "set1"):
            res[alpha_name] = compare(alpha_name, pool)
    finally:
        pool.close()
        pool.join()

    json.dump(res, open(os.path.abspath(DATA_PATH), "w"), indent=1)

    print("\n=== alphabet contrast (best buckling factor, restricted set1 -> extended set2) ===")
    for cname in CASES_ORDER:
        b1 = max(res["set1"][cname][o]["best"] for o in ("GA", "ACO", "PSO"))
        b2 = max(res["set2"][cname][o]["best"] for o in ("GA", "ACO", "PSO"))
        print(f"  {cname}: {b1:.2f} -> {b2:.2f}  ({100 * (b2 - b1) / b1:+.0f}%)")


if __name__ == "__main__":
    if "--restat" in sys.argv:
        restat()          # statistics only, from the stored per-seed record; no FE solve
    else:
        main()            # full campaign: 540 ccx runs, cluster/long-run territory
