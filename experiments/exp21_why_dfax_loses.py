"""exp21_why_dfax_loses.py -- perche' il crossover valido-per-costruzione perde: il meccanismo.

`exp20` ha FALSIFICATO l'ipotesi della fase uno: DFAX non batte REGEN, ed e' ultimo in tutti e
sei i blocchi. Questo esperimento non rimette in discussione quel verdetto; chiede **perche'**,
perche' un risultato negativo senza meccanismo e' un aneddoto.

L'IPOTESI, formulata dopo exp20 e quindi ESPLORATIVA per costruzione: il taglio scelto
uniformemente fra quelli ammissibili trascina i figli verso il **grosso del linguaggio**, che e'
povero di lamine a 0 gradi, mentre l'ottimo del caso assiale sta **al tetto**, cioe' nella coda
estrema. Se e' vero, l'operatore combatte contro la selezione invece di assecondarla, e la sua
imparzialita' -- che e' una virtu' quando si conta e si campiona -- diventa un difetto quando si
cerca.

IL TEST CHE LA DISTINGUE DA UN BACO. Non la diversita' in se': un baco e un collasso di
diversita' la fanno calare tutti e due. Si confronta, generazione per generazione, la
composizione media dell'**elite** con quella dei **figli appena generati**:

  se i figli regrediscono verso la media del linguaggio mentre l'elite sta sopra
      -> l'operatore disfa la selezione, e l'ipotesi regge;
  se figli ed elite si muovono insieme
      -> il problema non e' la deriva verso la moda, e l'ipotesi cade.

REGEN e' il controllo perfetto: i suoi figli NON ereditano nulla, quindi devono stare sulla media
del linguaggio per costruzione. Il confronto interessante e' quanto DFAX se ne discosta.

Uso:  PYTHONPATH=$PWD CCX_BIN=/usr/bin/ccx NPROC=30 python3 -m experiments.exp21_why_dfax_loses
"""
from __future__ import annotations

import json
import os
import random
import statistics as st
import sys
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimisers.constrained_search import CASES, ALPHABETS, gen_guided, manufacturing_ok, guidelines_ok
from optimisers.laminate_language import Layout, uniform_sampler, count_guided, ten_pct_min_half
import optimisers.metaheuristics as mh                                          # noqa: F401

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "algo_dfa"))
from crossover_states import dfa_parikh_cuts                                    # noqa: E402

from experiments.exp20_dfa_crossover import (_bf_symmetric, N, HALF, POP, GENS,  # noqa: E402
                                             BUDGET)

SEEDS = list(range(1, 9))
CASE = "c1_axial"          # il caso dove l'ottimo e' 0-ricco, quindi dove il meccanismo si vede
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "_out", "exp21_why_dfax_loses.json")


def language_reference(alpha):
    """Media e tetto ESATTI del linguaggio conforme: i riferimenti contro cui si legge la deriva."""
    total, by_k, stt = count_guided(alpha, N)
    mean = sum(k * w for k, w in by_k.items()) / total
    return dict(mean_c0=mean, mode_c0=max(by_k, key=by_k.get), max_c0=max(by_k),
                ceiling=HALF - 3 * stt["tmin"], total=total)


def _c0(h):
    return list(h).count(0)


def traced_run(arm, alpha, pool, rng):
    """Un run instrumentato: per ogni generazione, composizione di elite e figli."""
    case = CASES[CASE]
    sample, _t, _b, _s = uniform_sampler(alpha, N, rng)
    L = Layout(alpha, N)

    def child(elite, r):
        if arm == "REGEN":
            full = gen_guided(alpha, N, r)
            return (full[:HALF], 0) if full else (sample(), 1)
        ha, hb = r.choice(elite), r.choice(elite)
        cuts = dfa_parikh_cuts(list(ha), list(hb), L)
        if cuts:
            i = r.choice(cuts)
            return list(ha)[:i] + list(hb)[i:], 0
        return sample(), 1

    P = [sample() for _ in range(POP)]
    fits = pool.map(_bf_symmetric, [(p, case) for p in P])
    # IL NUMERO CHE MANCAVA. Il budget e' identico in VALUTAZIONI FE, non in DESIGN DISTINTI:
    # se la popolazione collassa, la maggior parte delle valutazioni ricade su cloni gia' visti.
    # Senza questo conteggio l'esperimento misura «quanti punti nuovi vede ciascun braccio»
    # invece della qualita' dell'ereditarieta', ed e' un confound che cambia la conclusione.
    seen = {tuple(h) for h in P}
    used, rows = len(P), []
    for g in range(GENS):
        rank = [p for _, p in sorted(zip(fits, P), key=lambda x: -x[0])]
        elite = rank[:max(2, POP // 3)]
        kids, fb = [], 0
        while len(kids) < POP - len(elite):
            c, f = child(elite, rng)
            fb += f
            full = list(c) + list(c)[::-1]
            assert manufacturing_ok(full) and guidelines_ok(full, alpha), "figlio non conforme"
            kids.append(c)
        rows.append(dict(
            gen=g,
            elite_c0=st.mean(_c0(h) for h in elite),
            kids_c0=st.mean(_c0(h) for h in kids),
            pop_c0=st.mean(_c0(h) for h in P),
            best_fit=round(max(fits), 4),
            elite_fit=round(st.mean(sorted(fits, reverse=True)[:len(elite)]), 4),
            distinct=len({tuple(h) for h in P}),
            cum_distinct=len(seen),
            fallback=fb,
        ))
        P = elite + kids
        seen |= {tuple(h) for h in P}
        fits = pool.map(_bf_symmetric, [(p, case) for p in P])
        used += len(P)
    assert used == BUDGET, f"budget {used} != {BUDGET}"
    rows[-1]["cum_distinct_final"] = len(seen)
    return rows


def main():
    nproc = int(os.environ.get("NPROC", max(1, (os.cpu_count() or 2) - 2)))
    arms = ("REGEN", "DFAX")
    alphas = ("set1", "set2")
    print(f"exp21: {len(alphas)}x{len(arms)}x{len(SEEDS)} run instrumentati su {CASE} = "
          f"{len(alphas) * len(arms) * len(SEEDS) * BUDGET:,} solve FE")

    out = {}
    pool = mp.Pool(nproc)
    try:
        for aname in alphas:
            alpha = ALPHABETS[aname]
            ref = language_reference(alpha)
            out[aname] = dict(reference=ref, arms={})
            print(f"\n--- {aname}: linguaggio conforme a N={N} -> media c0 {ref['mean_c0']:.2f}, "
                  f"moda {ref['mode_c0']}, tetto {ref['ceiling']} (su {HALF} lamine di half-stack)")
            for arm in arms:
                runs = [traced_run(arm, alpha, pool, random.Random(s * 10 + 7)) for s in SEEDS]
                out[aname]["arms"][arm] = runs
                print(f"  {arm}:")
                print(f"    {'gen':>4} {'elite c0':>9} {'figli c0':>9} {'scarto':>8} "
                      f"{'distinti':>9} {'elite BF':>9}")
                for g in range(GENS):
                    e = st.mean(r[g]["elite_c0"] for r in runs)
                    k = st.mean(r[g]["kids_c0"] for r in runs)
                    d = st.mean(r[g]["distinct"] for r in runs)
                    f = st.mean(r[g]["elite_fit"] for r in runs)
                    if g % 3 == 0 or g == GENS - 1:
                        print(f"    {g:>4} {e:>9.2f} {k:>9.2f} {k - e:>+8.2f} {d:>9.1f} {f:>9.3f}")
    finally:
        pool.close()
        pool.join()

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    json.dump(out, open(DATA_PATH, "w"), indent=1)

    print("\n=== DESIGN DISTINTI VALUTATI, a parita' di 176 valutazioni FE ===")
    for aname in alphas:
        for arm in arms:
            runs = out[aname]["arms"][arm]
            cd = st.mean(r[GENS - 1]["cum_distinct_final"] for r in runs)
            print(f"  {aname} {arm:>5}: {cd:6.1f} design distinti su {BUDGET} valutazioni "
                  f"({100 * cd / BUDGET:.0f}% del budget speso su punti nuovi)")

    print("\n=== il test dell'ipotesi ===")
    for aname in alphas:
        ref = out[aname]["reference"]["mean_c0"]
        for arm in arms:
            runs = out[aname]["arms"][arm]
            last = [r[GENS - 1] for r in runs]
            e = st.mean(r["elite_c0"] for r in last)
            k = st.mean(r["kids_c0"] for r in last)
            print(f"  {aname} {arm:>5}: all'ultima generazione elite c0 {e:.2f}, figli c0 {k:.2f}, "
                  f"media del linguaggio {ref:.2f}  -> i figli tornano verso il linguaggio: "
                  f"{abs(k - ref) < abs(e - ref)}")
    print(f"\n  scritto {DATA_PATH}")


if __name__ == "__main__":
    main()
