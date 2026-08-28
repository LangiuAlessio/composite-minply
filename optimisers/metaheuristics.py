"""Compare three metaheuristics (GA, Ant Colony, Particle Swarm) for the RR composite stacking-sequence
optimisation with the EXTENDED alphabet {0, +-30, +-45, +-60, 90}. Same FE evaluator (validated S8R
shell buckling via ccx) and same manufacturing constraints for all three -> a fair head-to-head.

For each load case (c1 axial / c2 +side / c3 +side+torsion), at a fixed ply count, each optimiser
MAXIMISES the buckling factor; we report the best sequence + factor per optimiser and the winner.
"""
from __future__ import annotations
import os, sys, random, math
sys.path.insert(0, os.path.dirname(__file__))
from optimisers.constrained_search import (buckling_factor, manufacturing_ok, gen_valid, repair, adiff,
                          CASES, ALPHABETS, setup_logging, log, Budget, DEFAULT_LOAD_SCALE)
from multiprocessing import Pool

ALPHA = ALPHABETS["set2"]                 # {0, +-30, +-45, +-60, 90}


def _scaled(case, scale):
    return {**case, "axial": case["axial"] * scale, "side": case["side"] * scale,
            "torsion": case["torsion"] * scale}


def _eval_pop(pop, case, pool):
    return pool.map(buckling_factor, [(p, case) for p in pop])


# --------------------------------------------------------------- GA (repair-based, load-aware)
# Population and iteration counts of this module's optimisers, and the FE cost of one
# (case, optimiser) run. MEASURED by instrumenting the evaluator (2026-08-26): ga and pso
# evaluate the whole population once per iteration plus the initial one (176 solves), aco
# evaluates its ants once per iteration (160). The budget used to be charged with
# constrained_search.EVALS_PER_PLY_STEP (62), which belongs to a different optimiser with a
# different population: the local cap was therefore charged under a third of the real cost.
MH_POP, MH_ITERS = 16, 10
EVALS_PER_MH_RUN = MH_POP * (MH_ITERS + 1)     # 176; aco costs MH_POP*MH_ITERS = 160


def ga(case, n, pool, rng, pop=MH_POP, gens=MH_ITERS):
    seeds = [[0], [0, 90], [45, -45], None]
    P, t = [], 0
    while len(P) < pop and t < pop * 8:
        s = gen_valid(ALPHA, n, rng, prefer=seeds[len(P) % len(seeds)], pbias=0.5); t += 1
        if s: P.append(s)
    fits = _eval_pop(P, case, pool)
    best = max(zip(fits, P), key=lambda x: x[0])
    for _ in range(gens):
        rank = [p for _, p in sorted(zip(fits, P), key=lambda x: -x[0])]
        elite = rank[:max(2, pop // 3)]
        kids = []
        while len(kids) < pop - len(elite):
            a, b = rng.choice(elite), rng.choice(elite)
            cut = rng.randint(1, n - 1)
            child = repair(a[:cut] + b[cut:], ALPHA)
            if rng.random() < 0.3:
                child[rng.randrange(n)] = rng.choice(ALPHA); child = repair(child, ALPHA)
            kids.append(child)
        P = elite + kids
        fits = _eval_pop(P, case, pool)
        best = max([best] + list(zip(fits, P)), key=lambda x: x[0])
    return best


# --------------------------------------------------------------- Ant Colony Optimisation
def aco(case, n, pool, rng, ants=MH_POP, iters=MH_ITERS, rho=0.15, q=1.0):
    A = list(ALPHA)
    tau = [[1.0 for _ in A] for _ in range(n)]            # pheromone[pos][angle]
    best = (-1.0, None)
    for _ in range(iters):
        colony = []
        for _ in range(ants):
            seq = []
            for pos in range(n):
                # feasible angles: manufacturing (<=3 consecutive, <=45 deg step)
                feas = [j for j, a in enumerate(A)
                        if (pos < 1 or adiff(a, seq[-1]) <= 45)
                        and not (pos >= 3 and seq[-1] == seq[-2] == seq[-3] == a)]
                if not feas:
                    feas = list(range(len(A)))
                w = [tau[pos][j] for j in feas]
                tot = sum(w) or 1.0
                r, acc, pick = rng.random() * tot, 0.0, feas[-1]
                for j, wj in zip(feas, w):
                    acc += wj
                    if acc >= r: pick = j; break
                seq.append(A[pick])
            colony.append(repair(seq, ALPHA))
        fits = _eval_pop(colony, case, pool)
        best = max([best] + list(zip(fits, colony)), key=lambda x: x[0])
        for r in range(n):                                # evaporate
            for j in range(len(A)): tau[r][j] *= (1 - rho)
        # deposit on the elite ants (reinforce good position->angle choices)
        for f, seq in sorted(zip(fits, colony), key=lambda x: -x[0])[:max(2, ants // 4)]:
            for pos in range(n):
                tau[pos][A.index(seq[pos])] += q * max(f, 0.0)
    return best


# --------------------------------------------------------------- Particle Swarm (discrete)
def pso(case, n, pool, rng, swarm=MH_POP, iters=MH_ITERS, w=0.5, c1=0.6, c2=0.6):
    """Discrete PSO: each particle rebuilds its sequence position by position.

    READ THE PER-POSITION UPDATE BELOW BEFORE CHANGING THE PAPER'S DESCRIPTION OF IT (audit
    2026-07-22, finding B1 -- the manuscript used to describe an algorithm this is not). Two
    things are easy to get wrong:

    1. The pull toward the personal best passes TWO nested gates, `r < c1` and then
       `rng.random() < c1/(c1+c2+w)`, so its net probability is c1^2/(c1+c2+w) = 0.36/1.7 = 0.212
       at the published settings, NOT the 0.353 of the inner gate alone.
    2. The three draws are applied in sequence and OVERWRITE one another, so the marginal source
       of a position is: random re-draw 0.150, global best 0.510, personal best 0.072, unchanged
       0.268 (verified by 4e6-sample Monte Carlo, and they sum to 1).

    The `w * 0.0 + c1` in the first gate is dead arithmetic (w*0.0 == 0). It is left exactly as
    it stands because this code produced the published tab:full; do not "clean" it without
    re-running the 540-solve campaign.
    """
    X = []
    t = 0
    while len(X) < swarm and t < swarm * 8:
        s = gen_valid(ALPHA, n, rng, prefer=rng.choice([[0], [45, -45], None]), pbias=0.4); t += 1
        if s: X.append(s)
    fits = _eval_pop(X, case, pool)
    pbest = [list(x) for x in X]; pbest_f = list(fits)
    g = max(range(len(X)), key=lambda i: fits[i]); gbest = list(X[g]); gbest_f = fits[g]
    for _ in range(iters):
        for i in range(len(X)):
            child = list(X[i])
            for pos in range(n):                          # discrete velocity: pull toward pbest/gbest
                r = rng.random()
                if r < w * 0.0 + c1:                      # toward personal best
                    if rng.random() < c1 / (c1 + c2 + w): child[pos] = pbest[i][pos]
                if rng.random() < c2:                     # toward global best
                    child[pos] = gbest[pos]
                if rng.random() < w * 0.3:                # inertia/exploration mutation
                    child[pos] = rng.choice(ALPHA)
            X[i] = repair(child, ALPHA)
        fits = _eval_pop(X, case, pool)
        for i in range(len(X)):
            if fits[i] > pbest_f[i]: pbest[i], pbest_f[i] = list(X[i]), fits[i]
            if fits[i] > gbest_f: gbest, gbest_f = list(X[i]), fits[i]
    return (gbest_f, gbest)


OPTS = {"GA": ga, "ACO": aco, "PSO": pso}


if __name__ == "__main__":
    import json, time
    import multiprocessing as _mp
    try:
        _mp.set_start_method("fork")   # macOS spawn can deadlock the Pool
    except RuntimeError:
        pass
    setup_logging()   # errors-only console + bounded rotating journal
    n = int(os.environ.get("NPLY", "40"))
    scale = float(os.environ.get("LOAD_SCALE", str(DEFAULT_LOAD_SCALE)))
    seed = int(os.environ.get("SEED", "1"))
    cases = {k: _scaled(v, scale) for k, v in CASES.items()}
    log.info("=== GA vs ACO vs PSO | alphabet set2 %s | N=%d ply | load_scale=%s ===", ALPHA, n, scale)
    budget = Budget()   # local cap over the whole head-to-head; lifted by COMPOSITE_TARGET=cluster
    results = {}
    with Pool(int(os.environ.get("NPROC", str(os.cpu_count())))) as pool:
        for cname, case in cases.items():
            log.info("[%s] target>%s", cname, CASES[cname]["threshold"])
            results[cname] = {}
            for oname, fn in OPTS.items():
                reason = budget.overrun()
                if reason:
                    log.error("metaheuristics: budget exhausted (%s) at %s/%s -- aborting locally. "
                              "Run the full head-to-head on cluster (COMPOSITE_TARGET=cluster), not this machine.",
                              reason, cname, oname)
                    break
                t0 = time.time()
                bf, seq = fn(case, n, pool, random.Random(seed))
                budget.tick(EVALS_PER_MH_RUN)
                results[cname][oname] = {"bf": round(bf, 3), "seq": seq, "secs": round(time.time()-t0, 1),
                                         "valid": manufacturing_ok(seq)}
                log.info("  %4s: BF=%6.3f  (%.0fs, mfg_ok=%s)", oname, bf, time.time()-t0,
                         manufacturing_ok(seq))
            if results[cname]:
                win = max(results[cname], key=lambda o: results[cname][o]["bf"])
                log.info("  -> winner: %s (BF=%s)", win, results[cname][win]["bf"])
    print("=== best sequence per case (across optimisers) ===")
    for cname in cases:
        if not results.get(cname):
            print(f"{cname}: ABORTED — budget exceeded (see ERROR log); dispatch to cluster")
            continue
        win = max(results[cname], key=lambda o: results[cname][o]["bf"])
        r = results[cname][win]
        print(f"{cname}: {win} BF={r['bf']}  seq={r['seq']}")
    json.dump(results, open(os.path.dirname(__file__) + "/_metaheuristics_set2.json", "w"), indent=2)
