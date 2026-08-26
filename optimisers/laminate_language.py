"""The language of admissible laminates: exact counting and exact uniform sampling.

The manufacturing rules (<=45 deg disorientation between consecutive plies, <=3 identical
consecutive plies) forbid a FINITE set of factors, so the set of admissible ply sequences is
a REGULAR language, recognised by a DFA whose state is `(last angle, run length)` -- 12
states on {0,45,-45,90}, 24 on the extended set. The remaining design guidelines are not
regular but are constraints on the PARIKH VECTOR (how many plies of each angle), so a DP on
`(DFA state, counters)` carries them exactly:

    symmetry      the code builds `full = h + h[:n//2][::-1]`, so it is automatic, and for
                  even n the mirror duplicates the last ply of the half-stack: an admissible
                  half-stack must end with run 1.
    balance       c[+t] == c[-t]; on the half-stack because c_full = 2*c_h.
    10% rule      c_full[a] >= 0.1*n, i.e. 2*c_h[a] >= 0.1*n.

What this buys the optimiser: the number of compliant laminates for a given ply count is
COUNTABLE without generating any of them, and one can be drawn UNIFORMLY at random in O(m)
with no rejection -- against the ~2000 build-and-reject attempts per compliant laminate that
`gen_guided` needs (measured 2026-08-06: 99.8 ms vs 0.09 ms per laminate at N=44).

This module is deliberately SELF-CONTAINED (it does not import constrained_search) so that
constrained_search can import from it without a cycle; `adiff` lives here and is imported
back, so the disorientation metric has ONE definition in the codebase. The rules encoded in
`build_dfa` are checked against the live `manufacturing_ok`/`guidelines_ok` by brute-force
enumeration in `algo_dfa/parikh_dp.py --check` -- if they ever drift apart, that check fails.
"""
from __future__ import annotations

MAXRUN = 3                        # <=3 identical consecutive plies
PRINCIPAL = (0, 45, -45, 90)      # the directions the 10% rule applies to


def adiff(a, b):
    """Disorientation between two ply angles, in degrees (0..90)."""
    d = abs(a - b) % 180
    return min(d, 180 - d)


def ten_pct_min_half(n):
    """Smallest half-stack counter satisfying the 10% rule on the n-ply full stack.

    Replicates the FLOAT comparison of the live code (`if c[a] < 0.1 * n`) with c_full =
    2*c_h instead of rewriting it as a ceiling: at N=40 the threshold is exactly 4.0 and
    c_full=4 passes; at N=44 it is 4.4 and c_full=4 does not.
    """
    c = 0
    while 2 * c < 0.1 * n:
        c += 1
    return c


def build_dfa(alpha):
    """States `(last angle, run 1..3)` and transitions of the manufacturing automaton."""
    S = [(a, r) for a in alpha for r in range(1, MAXRUN + 1)]
    idx = {s: i for i, s in enumerate(S)}
    trans = [[] for _ in S]                       # trans[i] = [(next state, angle), ...]
    for (a, r) in S:
        for b in alpha:
            if adiff(a, b) > 45:
                continue                          # forbidden factor of length 2
            if b == a:
                if r + 1 > MAXRUN:
                    continue                      # forbidden factor aaaa
                trans[idx[(a, r)]].append((idx[(a, r + 1)], b))
            else:
                trans[idx[(a, r)]].append((idx[(b, 1)], b))
    return S, idx, trans


class Layout:
    """Which counters the DP tracks, and what for.

    EXACT: angles carrying a threshold (the principal directions present in the alphabet).
    DIFF:  sign pairs without a threshold (+-30, +-60) -- only the DIFFERENCE matters, which
           must close to zero, so tracking both separately would multiply the DP cells for
           nothing.
    FREE:  an angle with neither threshold nor sign partner. Neither alphabet has one.
    """

    def __init__(self, alpha, n):
        m = n // 2
        self.alpha, self.n, self.m = alpha, n, m
        self.tmin = ten_pct_min_half(n)
        self.exact = [a for a in alpha if a in PRINCIPAL]
        self.ei = {a: i for i, a in enumerate(self.exact)}
        self.exact_pairs = [(self.ei[t], self.ei[-t]) for t in self.exact
                            if t > 0 and -t in self.ei]
        self.diff = [t for t in alpha if t > 0 and -t in alpha and t not in PRINCIPAL]
        self.di = {}
        for k, t in enumerate(self.diff):
            self.di[t] = (k, +1)
            self.di[-t] = (k, -1)
        self.free = [a for a in alpha if a not in self.ei and a not in self.di]
        self.thr = [self.tmin if a in PRINCIPAL else 0 for a in self.exact]

    def need(self, ex, df):
        """Plies still needed, at a minimum, to close every constraint from here.

        Each unit of deficit costs at least one ply: thresholds not yet reached, differences
        to bring back to zero, and exact pairs to level AFTER raising both to the threshold.
        It is a lower bound, so pruning on `need > remaining` loses no solution (and the
        brute-force check confirms it).
        """
        tot = 0
        for i, c in enumerate(ex):
            if c < self.thr[i]:
                tot += self.thr[i] - c
        for (i, j) in self.exact_pairs:
            a = max(ex[i], self.thr[i])
            b = max(ex[j], self.thr[j])
            tot += abs(a - b)
        for d in df:
            tot += abs(d)
        return tot

    def accept(self, ex, df):
        """Final counters that are admissible: thresholds, balance, differences closed."""
        for i, c in enumerate(ex):
            if c < self.thr[i]:
                return False
        for (i, j) in self.exact_pairs:
            if ex[i] != ex[j]:
                return False
        return all(d == 0 for d in df)


def count_free(alpha, n, with_mirror=True):
    """Half-stacks obeying the MANUFACTURING rules only, in pure integer arithmetic.

    with_mirror=True counts the accepting ones (final run 1, the constraint the mirror
    imposes); with_mirror=False counts every manufacturing-ok half-stack, which is what
    `gen_valid` produces -- the right denominator for an acceptance rate.
    """
    m = n // 2
    S, idx, trans = build_dfa(alpha)
    v = [0] * len(S)
    for a in alpha:
        v[idx[(a, 1)]] = 1
    for _ in range(m - 1):
        w = [0] * len(S)
        for i, x in enumerate(v):
            if x:
                for (j, _b) in trans[i]:
                    w[j] += x
        v = w
    if not with_mirror:
        return sum(v)
    return sum(v[idx[(a, 1)]] for a in alpha)


def count_free_by_k(alpha, n):
    """Composition (plies at 0 deg in the half-stack) over the manufacturing-only language."""
    m = n // 2
    S, idx, trans = build_dfa(alpha)
    dp = [[0] * (m + 1) for _ in S]
    for a in alpha:
        dp[idx[(a, 1)]][1 if a == 0 else 0] = 1
    for _ in range(m - 1):
        nxt = [[0] * (m + 1) for _ in S]
        for i, row in enumerate(dp):
            if not any(row):
                continue
            for (j, b) in trans[i]:
                tgt = nxt[j]
                if b == 0:
                    for k in range(m):
                        if row[k]:
                            tgt[k + 1] += row[k]
                else:
                    for k in range(m + 1):
                        if row[k]:
                            tgt[k] += row[k]
        dp = nxt
    out = {}
    for a in alpha:
        for k, w in enumerate(dp[idx[(a, 1)]]):
            if w:
                out[k] = out.get(k, 0) + w
    return out


def count_guided(alpha, n, report_sizes=False, keep_levels=False):
    """EXACT count of the half-stacks whose mirror is a compliant n-ply laminate.

    Returns (total, distribution over c_h[0], stats)."""
    assert n % 2 == 0, "only even N (the campaign uses 40/44/48/54); odd n does not duplicate the mid ply"
    L = Layout(alpha, n)
    m = L.m
    S, idx, trans = build_dfa(alpha)
    ne, nd = len(L.exact), len(L.diff)

    cur = {}
    for a in alpha:
        ex = [0] * ne
        df = [0] * nd
        if a in L.ei:
            ex[L.ei[a]] = 1
        elif a in L.di:
            k, s = L.di[a]
            df[k] = s
        key = (idx[(a, 1)], tuple(ex), tuple(df))
        cur[key] = cur.get(key, 0) + 1

    sizes = [len(cur)]
    levels = [cur] if keep_levels else None
    for step in range(m - 1):
        remaining_after = m - (step + 2)
        nxt = {}
        for (si, ex, df), w in cur.items():
            for (sj, b) in trans[si]:
                if b in L.ei:
                    i = L.ei[b]
                    ex2 = ex[:i] + (ex[i] + 1,) + ex[i + 1:]
                    df2 = df
                else:
                    ex2 = ex
                    if b in L.di:
                        k, s = L.di[b]
                        df2 = df[:k] + (df[k] + s,) + df[k + 1:]
                    else:
                        df2 = df
                if L.need(ex2, df2) > remaining_after:
                    continue                       # the constraints can no longer close
                key = (sj, ex2, df2)
                nxt[key] = nxt.get(key, 0) + w
        cur = nxt
        sizes.append(len(cur))
        if keep_levels:
            levels.append(cur)

    by_k = {}
    total = 0
    i0 = L.ei.get(0)
    for (si, ex, df), w in cur.items():
        if S[si][1] != 1:                          # the mirror duplicates the final ply
            continue
        if not L.accept(ex, df):
            continue
        total += w
        k = ex[i0] if i0 is not None else 0
        by_k[k] = by_k.get(k, 0) + w
    stats = dict(states=len(S), tmin=L.tmin, exact=L.exact, diff=L.diff,
                 free=L.free, sizes=sizes, max_cells=max(sizes), cells_total=sum(sizes))
    if keep_levels:
        stats['levels'] = levels
        stats['layout'] = L
        stats['dfa'] = (S, idx, trans)
    return total, by_k, stats


# ---- exact uniform sampling ------------------------------------------------------
# The DP carries, for every level j, how many prefixes reach each cell. Walking a path
# BACKWARDS, choosing each predecessor with probability proportional to its prefix count,
# is uniform over the accepting paths: the standard construction, exact, no rejection,
# no restart, O(m) per sample.
_PREPARED = {}


def _prepare(alpha, n):
    """Build (and cache) the levelled DP for one (alphabet, ply count).

    Cached because `ga_best` draws many individuals per ply count and the DP does not
    depend on the RNG. One entry costs the whole levelled DP (188 694 cells at most, on
    the extended set at N=44), so the cache holds one per (alphabet, N) actually used.
    """
    key = (tuple(alpha), n)
    got = _PREPARED.get(key)
    if got is not None:
        return got
    total, by_k, st = count_guided(alpha, n, keep_levels=True)
    if total == 0:
        raise ValueError(f'no compliant laminate at N={n} for alphabet {alpha}')
    levels, L, (S, idx, trans) = st['levels'], st['layout'], st['dfa']
    preds = [[] for _ in S]
    for i, tl in enumerate(trans):
        for (j, _b) in tl:
            preds[j].append(i)
    finals = [(k, w) for k, w in levels[L.m - 1].items()
              if S[k[0]][1] == 1 and L.accept(k[1], k[2])]
    assert sum(w for _, w in finals) == total, 'levelled DP disagrees with its own total'
    got = dict(total=total, by_k=by_k, st=st, levels=levels, L=L, S=S, trans=trans,
               preds=preds, finals=finals)
    _PREPARED[key] = got
    return got


def _unstep(L, ex, df, b):
    """Remove ply b from the counters (None if impossible)."""
    if b in L.ei:
        i = L.ei[b]
        if ex[i] == 0:
            return None
        return ex[:i] + (ex[i] - 1,) + ex[i + 1:], df
    if b in L.di:
        k, s = L.di[b]
        return ex, df[:k] + (df[k] - s,) + df[k + 1:]
    return ex, df


def sample_compliant_half(alpha, n, rng):
    """Draw ONE half-stack uniformly at random from the compliant language. O(m), no rejection."""
    P = _prepare(alpha, n)
    L, S, levels, preds, finals, total = (P['L'], P['S'], P['levels'], P['preds'],
                                          P['finals'], P['total'])
    r = rng.randrange(total)
    for key, w in finals:
        if r < w:
            break
        r -= w
    (si, ex, df) = key
    seq = [S[si][0]]
    for j in range(L.m - 1, 0, -1):
        b = S[si][0]
        ex_p, df_p = _unstep(L, ex, df, b)
        cands, tot_p = [], 0
        for pi in preds[si]:
            w = levels[j - 1].get((pi, ex_p, df_p))
            if w:
                cands.append((pi, w))
                tot_p += w
        assert tot_p, 'cell reached with no predecessor: inconsistent DP'
        r = rng.randrange(tot_p)
        for pi, w in cands:
            if r < w:
                break
            r -= w
        si, ex, df = pi, ex_p, df_p
        seq.append(S[si][0])
    seq.reverse()
    return seq


def uniform_sampler(alpha, n, rng):
    """(sample, total, by_k, stats) -- the closure form, kept for the analysis scripts."""
    P = _prepare(alpha, n)
    return (lambda: sample_compliant_half(alpha, n, rng),
            P['total'], P['by_k'], P['st'])


if __name__ == '__main__':                # python3 -m optimisers.laminate_language
    # Self-check: every draw is validated by the LIVE rule checkers, not by this module's
    # own idea of the rules -- the whole point is that the two must agree. The exhaustive
    # DP-vs-brute-force verification lives in algo_dfa/parikh_dp.py --check.
    import random
    import time
    from optimisers.constrained_search import manufacturing_ok, guidelines_ok, gen_guided

    # The setup cost is reported SEPARATELY from the per-draw cost, and the break-even is
    # spelled out. Folding the DP build into the per-draw average (as a first version of
    # this check did) makes a 0.066 ms draw look like 2.0 ms, and hides the fact that on the
    # extended set the whole advantage depends on how many laminates you draw per ply count.
    rng = random.Random(20260806)
    ok = True
    print(f'  {"set":>10} {"N":>3} {"compliant":>19} {"DP build":>9} {"draw":>9} '
          f'{"gen_guided":>11} {"break-even":>11}')
    for name, alpha in (('restricted', [0, 45, -45, 90]),
                        ('extended', [0, 30, -30, 45, -45, 60, -60, 90])):
        for n in (40, 44, 48):
            t0 = time.time()
            total = _prepare(alpha, n)['total']          # levelled DP: the one-off cost
            t_dp = time.time() - t0
            t0 = time.time()
            bad = 0
            for _ in range(2000):
                seq = sample_compliant_half(alpha, n, rng)
                full = seq + seq[:n // 2][::-1]
                if not (manufacturing_ok(full) and guidelines_ok(full, alpha)):
                    bad += 1
            t_s = (time.time() - t0) / 2000
            t0 = time.time()
            got = sum(1 for _ in range(200) if gen_guided(alpha, n, rng))
            t_g = (time.time() - t0) / 200
            ok &= (bad == 0)
            be = t_dp / (t_g - t_s) if t_g > t_s else float('inf')
            print(f'  [{"OK " if not bad else "FAIL"}] {name:>10} {n:>3} {total:>19,} '
                  f'{t_dp:8.2f}s {1000*t_s:8.3f}ms {1000*t_g:10.3f}ms '
                  f'{be:10.0f} draws   ({bad} non-compliant, gen_guided {got}/200)')
    print('\n  "break-even" = draws from the SAME ply count needed to repay the DP build.')
    print(f'  ga_best draws pop + gens*(pop - elite) = {12 + 4 * (12 - 4)} guided laminates per ply')
    print('  count, and every ply count needs its own DP: on the extended set the exact')
    print('  sampler is therefore near break-even at the campaign budget, and only pays off')
    print('  clearly when many laminates are drawn per N (or on the restricted set, where')
    print('  the DP is essentially free).')
    print(f'\n  {"self-check passed" if ok else "SELF-CHECK FAILED"}')
    raise SystemExit(0 if ok else 1)
