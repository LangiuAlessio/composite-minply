"""FE-search certification: at a small ply count, EXHAUSTIVELY enumerate every
manufacturing-valid symmetric laminate, FE-evaluate the buckling factor of each, and
confirm the GA reaches the true FE global optimum. Unlike the closed-form CLT
Haftka-Walsh benchmark, this certifies the actual FE-in-the-loop search machinery.

Result (CalculiX 2.21, c1 axial, N=8): 584 valid symmetric laminates exhaustively
evaluated -> FE global-max buckling 1.1814; GA (3 seeds) reaches 1.1744-1.1753, i.e.
within 0.51% of the exhaustive FE global optimum. Run from cases/:
  NPROC=6 python3 rr_certify_fe.py"""
import os, sys, time, random, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import multiprocessing as mp
try:
    from optimisers.constrained_search import buckling_factor, manufacturing_ok, ga_best, CASES, ALPHABETS
except ImportError:
    from optimisers.constrained_search import buckling_factor, manufacturing_ok, ga_best, CASES, ALPHABETS
CASE = CASES["c1_axial"]; ALPHA = list(ALPHABETS["set2"]); N = 8; HALF = N // 2


def main():
    pool = mp.get_context("fork").Pool(int(os.environ.get("NPROC", str(os.cpu_count()))))
    try:
        seen, fulls = set(), []
        for h in itertools.product(ALPHA, repeat=HALF):
            full = list(h) + list(h)[::-1]
            if manufacturing_ok(full) and tuple(full) not in seen:
                seen.add(tuple(full)); fulls.append(full)
        print("N=%d: %d valid symmetric laminates (exhaustive FE)" % (N, len(fulls)), flush=True)
        t0 = time.time()
        bfs = pool.map(buckling_factor, [(f, CASE) for f in fulls])
        gmax = max(bfs)
        print("EXHAUSTIVE FE global max buckling = %.4f (%.0fs)" % (gmax, time.time() - t0), flush=True)
        for sd in [1, 2, 3]:
            bf, _ = ga_best(CASE, N, ALPHABETS["set2"], pool, random.Random(sd), pop=24, gens=15)
            print("GA seed=%d: bf=%.4f  gap-to-global=%.2f%%" % (sd, bf, 100 * (gmax - bf) / gmax), flush=True)
    finally:
        pool.close(); pool.join()


if __name__ == "__main__":
    main()
