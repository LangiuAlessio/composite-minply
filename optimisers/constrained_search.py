"""Constrained discrete laminate optimiser for the RR composite panel, on the VALIDATED
S8R composite-shell buckling model (ccx == Abaqus <2%). Minimise the number of plies s.t.
the buckling factor of each load case exceeds its threshold, subject to manufacturing
constraints (<=3 consecutive same orientation; <=45 deg change between consecutive plies).
Per-case optimisation -> a different stacking sequence per buckling case (as Giacomo wants).

FE eval = ccx S8R composite shell buckling (free, no node cap, ~661-node full panel).
Parallel via multiprocessing. Alphabets: {0,45,-45,90} and {0,+-30,+-45,+-60,90}.
"""
from __future__ import annotations
import os, re, subprocess, tempfile, shutil, random
import time
import logging
import logging.handlers

from fe.materials import get_material
from multiprocessing import Pool

from fe.ccx_bin import CCX          # un solo default per il binario ccx (audit F12)

# --- logging policy (2026-06-06) -------------------------------------------------
# An abandoned `python3 -u rr_optimiser.py` once dumped ~4 GB of per-candidate
# progress to its captured stdout over 13 h. Policy: progress goes to a SIZE-BOUNDED
# rotating journal (cannot grow without limit); the console (the captured task
# .output) gets ERROR and above only -- the verbose rest is effectively /dev/null.
log = logging.getLogger("rr_optimiser")


def setup_logging(log_dir=None, max_mb=None, backups=None,
                  console_level=None, file_level=None):
    """Configure the module logger: errors-only console + a bounded rotating journal.

    Env overrides (used when args are None):
      RR_LOG_DIR       journal directory                 (default cases/_out)
      RR_LOG_MAX_MB    max size per journal file in MiB   (default 10; <=0 disables file)
      RR_LOG_BACKUPS   rotated copies kept                (default 5)
      RR_CONSOLE_LEVEL console threshold                  (default ERROR)
      RR_FILE_LEVEL    journal threshold                  (default INFO)

    Total on-disk journal is bounded by (RR_LOG_BACKUPS + 1) * RR_LOG_MAX_MB.
    Set RR_LOG_MAX_MB=0 to send the verbose journal to /dev/null entirely.
    """
    _here = os.path.dirname(os.path.abspath(__file__))
    if log_dir is None:
        log_dir = os.environ.get("RR_LOG_DIR", os.path.join(_here, "_out"))
    if max_mb is None:
        max_mb = float(os.environ.get("RR_LOG_MAX_MB", "10"))
    if backups is None:
        backups = int(os.environ.get("RR_LOG_BACKUPS", "5"))
    console_level = (console_level or os.environ.get("RR_CONSOLE_LEVEL", "ERROR")).upper()
    file_level = (file_level or os.environ.get("RR_FILE_LEVEL", "INFO")).upper()

    for h in list(log.handlers):
        h.close()
        log.removeHandler(h)
    log.setLevel(logging.DEBUG)   # handlers do the filtering
    log.propagate = False

    ch = logging.StreamHandler()  # -> stderr; errors only, so it can never bloat
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    log.addHandler(ch)

    if max_mb and max_mb > 0:      # bounded rotating journal; max_mb<=0 -> /dev/null
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, "rr_optimiser.log"),
            maxBytes=int(max_mb * 1024 * 1024), backupCount=backups, encoding="utf-8")
        fh.setLevel(file_level)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)
    return log


# --- optimisation budget (2026-06-06) --------------------------------------------
# Non-convergence must fail FAST and BOUNDED, not run for days. Each (problem, method)
# optimisation gets a hard LOCAL ceiling on wall-clock and on FE evaluations. The S8R
# panel costs ~1-3 s/eval; a GA case is ~9 ply-steps x pop*(gens+1) ~= 540 evals, a few
# minutes. Defaults leave generous head-room, then abort. A search that legitimately
# needs more is, by policy, a cluster job (COMPOSITE_TARGET=cluster lifts the caps), not
# a laptop job -- that is exactly what melted the Mac for 13 h.
# Single definition of the load multiplier applied to the nominal case loads. Both this
# module and optimisers/metaheuristics.py read it: they used to carry divergent defaults
# (1.0 here, 0.44 there), so running the latter as a module silently loaded the panel at
# 44%. The published campaign runs at 1.0 (experiments/exp4 uses CASES unscaled).
DEFAULT_LOAD_SCALE = 1.0

GA_POP, GA_GENS = 12, 4
# FE solves charged to the budget for one ply-count step of optimise_case_ga, MEASURED by
# instrumenting the evaluator (2026-08-26), not estimated: ga_best re-evaluates the whole
# population every generation (elite included), so it costs GA_POP*(GA_GENS+1) buckling
# solves; a step that clears the buckling threshold then adds one static_metrics solve and
# one interlaminar solid solve on the winner. 62 solves per step at the current settings.
EVALS_PER_PLY_STEP = GA_POP * (GA_GENS + 1) + 2


class Budget:
    """Hard local cap on a single (problem, method) optimisation run.

    Env overrides (used when the matching arg is None):
      RR_MAX_SECONDS_PER_CASE  wall-clock ceiling, seconds   (default 1200 = 20 min)
      RR_MAX_EVALS_PER_CASE    FE-evaluation ceiling          (default 2000)
      COMPOSITE_TARGET=cluster          lift all caps (cluster run)
      RR_ALLOW_LONG=1          lift all caps (explicit opt-in)
    """

    def __init__(self, max_seconds=None, max_evals=None, allow_long=None,
                 clock=time.monotonic):
        if allow_long is None:
            allow_long = (os.environ.get("COMPOSITE_TARGET", "").lower() == "cluster"
                          or os.environ.get("RR_ALLOW_LONG", "0") == "1")
        if max_seconds is None:
            max_seconds = float(os.environ.get("RR_MAX_SECONDS_PER_CASE", "1200"))
        if max_evals is None:
            max_evals = int(os.environ.get("RR_MAX_EVALS_PER_CASE", "2000"))
        self.allow_long = allow_long
        self.max_seconds = None if allow_long else float(max_seconds)
        self.max_evals = None if allow_long else int(max_evals)
        self._clock = clock
        self._start = clock()
        self.evals = 0

    def tick(self, n=1):
        """Account for `n` FE evaluations consumed."""
        self.evals += n

    def elapsed(self):
        return self._clock() - self._start

    def overrun(self):
        """Return a human-readable reason if the budget is blown, else None."""
        if self.max_seconds is not None and self.elapsed() > self.max_seconds:
            return f"wall-clock > {self.max_seconds:.0f}s (elapsed {self.elapsed():.0f}s)"
        if self.max_evals is not None and self.evals > self.max_evals:
            return f"FE evals > {self.max_evals} (used {self.evals})"
        return None
NX, NY, LX, LY = 20, 10, 100.0, 50.0
PLY_T = 0.1
# Deck lines for the campaign lamina, generated from the single source of truth in
# fe/materials.py instead of being typed here: the two used to be independent copies, and a
# third, different lamina (E1 135000, ply 0.125 mm) lived in that registry under the name
# the paper uses. The assert pins the generated text to the bytes this module shipped, so a
# change to the registry entry cannot silently move the published deck.
def _mat_deck_lines(m: dict) -> list[str]:
    def f(x: float) -> str:
        return f"{x:.0f}." if float(x).is_integer() else repr(x)
    keys = ("E1", "E2", "E3", "nu12", "nu13", "nu23", "G12", "G13")
    return ["*MATERIAL, NAME=layer", "*ELASTIC, TYPE=ENGINEERING CONSTANTS",
            ",".join(f(m[k]) for k in keys),
            f(m["G23"]) + ",", "*DENSITY", f"{m['rho']:.2e},"]


MAT = _mat_deck_lines(get_material("canale2018"))
assert MAT == ["*MATERIAL, NAME=layer", "*ELASTIC, TYPE=ENGINEERING CONSTANTS",
               "125100.,7840.,7840.,0.3,0.15,0.15,4600.,4000.", "4000.,",
               "*DENSITY", "1.62e-09,"], MAT

# 3 buckling load cases (axial always -1000) mapped from Composite_buckling_{3,2,1}.inp
# -> Giacomo's email targets >15 / >8 / >5.
CASES = {
    # Clean in-plane validation cases (2026-06-05): the campaign is restricted to in-plane
    # loading, so torsion is zero here.
    # CAUTION (2026-07-20 audit, A4): the earlier comment on this line read "inert for membrane
    # buckling (verified: BF unchanged for torsion 0..60000)". The verification was correct but
    # was being EXTRAPOLATED far past its range.
    # RE-MEASURED 2026-07-22 (audit B3) on the designs actually DELIVERED in the paper: the C1 row
    # below used to be the 46-ply design, superseded when the frequency constraint moved C1 to 48
    # plies, so the paper was quoting torsion sensitivities of a laminate it no longer ships. Both
    # rows are now the 48-ply / 54-ply extended-set laminates of tab:explicitseq (ccx_2.21):
    #   T [Nmm]      0       6e4      1e5      1.5e5    2e5      3e5
    #   C1 axial   4.8955  4.8807  4.8546   4.2336   3.1848   2.1298   (-0.30% at 6e4, -56.5% at 3e5)
    #   C3 combo   4.3832  4.3709  4.3463   3.9537   3.1699   2.2621   (-0.28% at 6e4, -48.4% at 3e5)
    # (the C3 row is unchanged: that design was not superseded.)
    # i.e. <1% below ~1e5 Nmm, dominant above. fe/reference_cases.py uses 3e5 Nmm and its
    # docstring is right to say torsion is NOT inert there. Do not restore the "inert" wording.
    # Three distinct in-plane buckling cases with a common safety-factor target BLF>=4.
    "c1_axial":  dict(axial=-2400., side=0.,    torsion=0., threshold=4.),  # pure axial      -> 0-dominated
    "c2_side":   dict(axial=0.,     side=4900., torsion=0., threshold=4.),  # pure shear      -> +-45-richer
    "c3_combo":  dict(axial=-1500., side=4900., torsion=0., threshold=4.),  # shear + axial   -> mixed, heaviest
}
ALPHABETS = {"set1": [0, 45, -45, 90], "set2": [0, 30, -30, 45, -45, 60, -60, 90]}


# `adiff` (the disorientation metric) and the exact sampler of the compliant language live
# in laminate_language, which knows nothing about this module: one definition each, no
# import cycle. See that module's docstring for why the language is regular.
try:
    from optimisers.laminate_language import adiff, sample_compliant_half
except ImportError:                       # run as a script from inside code/optimisers/
    from laminate_language import adiff, sample_compliant_half


def manufacturing_ok(seq):
    for i in range(len(seq) - 1):
        if adiff(seq[i], seq[i + 1]) > 45:
            return False
    for i in range(len(seq) - 3):
        if seq[i] == seq[i + 1] == seq[i + 2] == seq[i + 3]:
            return False
    return True


def gen_valid(alpha, n, rng, prefer=None, pbias=0.0):
    """Build a manufacturing-valid sequence of length n by construction (or None).
    With prefer=[angles] and pbias in [0,1], bias the next ply toward the preferred
    angles (load-aligned search) while always respecting the <=45 step / <=3 consecutive."""
    for _ in range(3000):
        s = [rng.choice(prefer if (prefer and rng.random() < pbias) else alpha)]
        ok = True
        while len(s) < n:
            cand = [a for a in alpha if adiff(s[-1], a) <= 45
                    and not (len(s) >= 3 and s[-1] == s[-2] == s[-3] == a)]
            if not cand:
                ok = False
                break
            pref_cand = [a for a in cand if prefer and a in prefer]
            pool_ = pref_cand if (pref_cand and rng.random() < pbias) else cand
            s.append(rng.choice(pool_))
        if ok and manufacturing_ok(s):
            return s
    return None


def guidelines_ok(seq, alpha):
    """Standard composite design guidelines (on top of disorientation/contiguity):
    SYMMETRY (no extension-bending coupling), BALANCE (equal +θ/-θ → no shear-extension
    coupling), 10% RULE (>=10% in each principal direction 0/±45/90 that the alphabet has)."""
    from collections import Counter
    n = len(seq)
    if seq != seq[::-1]:                       # symmetry
        return False
    c = Counter(seq)
    for a in alpha:                            # balance
        if a > 0 and -a in alpha and c[a] != c[-a]:
            return False
    for a in (0, 45, -45, 90):                 # 10% rule on principal directions
        if a in alpha and c[a] < 0.1 * n:
            return False
    return True


def gen_guided(alpha, n, rng, prefer=None, pbias=0.0, tries=6000):
    """Build a SYMMETRIC, balanced, 10%-compliant, manufacturing-valid sequence (or None)."""
    half = (n + 1) // 2
    for _ in range(tries):
        h = gen_valid(alpha, half, rng, prefer=prefer, pbias=pbias)
        if not h:
            continue
        full = h + h[:n // 2][::-1]            # mirror -> symmetric
        if manufacturing_ok(full) and guidelines_ok(full, alpha):
            return full
    return None


def gen_guided_exact(alpha, n, rng, prefer=None, pbias=0.0, tries=None):
    """Same contract as `gen_guided` -- a symmetric, balanced, 10%-compliant, manufacturing
    valid n-ply laminate -- but drawn UNIFORMLY and EXACTLY from the compliant language
    instead of built-and-rejected. O(n) per draw, no rejection, never returns None when the
    language is non-empty (and raises, instead of silently failing, when it IS empty).

    Measured 2026-08-06 on the extended set at N=44: 0.09 ms per laminate against 99.8 ms
    for `gen_guided`, which needs ~1993 build-and-reject attempts per success and fails
    outright 4.5% of the time.

    TWO THINGS IT DOES NOT DO, and they are why this is opt-in (`EXACT_SAMPLER=1`) rather
    than the default:
      - `prefer`/`pbias` are IGNORED. A uniform draw has no load-aligned bias, and that bias
        is part of how `ga_best` seeds its population; conditioning the draw on a
        composition (which the DP can do exactly, by weighting the final cells) is the
        proper replacement and is not implemented here.
      - the published campaign ran on `gen_guided`. Switching the default would make the
        delivered designs irreproducible for the sake of a speed-up on a step that is not
        the bottleneck (the FE solve is).
    """
    del prefer, pbias, tries              # deliberately unused: see the docstring
    h = sample_compliant_half(alpha, n, rng)
    return h + h[:n // 2][::-1]


def _nodes_elems(nx=NX, ny=NY, lx=LX, ly=LY):
    ID, nodes, nid = {}, [], 1
    for I in range(2 * nx + 1):
        for J in range(2 * ny + 1):
            if I % 2 == 1 and J % 2 == 1:
                continue
            ID[(I, J)] = nid
            nodes.append((nid, I * lx / (2 * nx), J * ly / (2 * ny)))
            nid += 1
    elems = []
    e = 1
    for i in range(nx):
        for j in range(ny):
            I, J = 2 * i, 2 * j
            elems.append((e, [ID[(I, J)], ID[(I + 2, J)], ID[(I + 2, J + 2)], ID[(I, J + 2)],
                              ID[(I + 1, J)], ID[(I + 2, J + 1)], ID[(I + 1, J + 2)], ID[(I, J + 1)]]))
            e += 1
    return ID, nodes, elems


_ID, _NODES, _ELEMS = _nodes_elems()


def make_ccx_deck(seq, case, mesh=None):
    """Build the S8R shell deck. mesh=None uses the module-default 20x10 grid (byte-identical
    to the published campaign). Pass mesh=(nx,ny) or (nx,ny,lx,ly) for a mesh-refinement sweep
    (panel C of the pitfalls figure): nodes, elements, clamp, tip and CLOAD are ALL rebuilt from
    the same (nx,ny,lx,ly), so the deck stays internally consistent at any resolution -- the
    freeze that made a runtime NX,NY change divide by zero (2026-07-20) is gone."""
    if mesh is None:
        ID, NODES, ELEMS, nx, ny, lx, ly = _ID, _NODES, _ELEMS, NX, NY, LX, LY
    else:
        nx, ny = mesh[0], mesh[1]
        lx, ly = (mesh[2], mesh[3]) if len(mesh) >= 4 else (LX, LY)
        ID, NODES, ELEMS = _nodes_elems(nx, ny, lx, ly)
    L = ["*HEADING", "rr shell optimiser candidate", "*NODE"]
    L += [f"{n}, {x:.4f}, {y:.4f}, 0.0" for n, x, y in NODES]
    L.append("*ELEMENT, TYPE=S8R, ELSET=EALL")
    L += [f"{e}, " + ", ".join(map(str, c)) for e, c in ELEMS]
    angles = sorted(set(seq))
    onm = {a: f"O{str(a).replace('-', 'm')}" for a in angles}
    for a in angles:
        L += [f"*ORIENTATION, NAME={onm[a]}", "1.,0.,0.,0.,1.,0.", f"3, {a}."]
    L.append("*SHELL SECTION, COMPOSITE, ELSET=EALL")
    L += [f"{PLY_T},,layer,{onm[a]}" for a in seq]
    L += MAT
    clamp = [ID[(0, J)] for J in range(2 * ny + 1) if (0, J) in ID]
    L.append("*NSET, NSET=CLAMP")
    L += [",".join(map(str, clamp[k:k + 16])) for k in range(0, len(clamp), 16)]
    L += ["*BOUNDARY", "CLAMP, 1, 6", "*STEP", "*BUCKLE", "2"]
    tip = [(ID[(2 * nx, J)], J * ly / (2 * ny)) for J in range(2 * ny + 1) if (2 * nx, J) in ID]
    yc = sum(y for _, y in tip) / len(tip)
    sumr = sum((y - yc) ** 2 for _, y in tip) or 1.0
    c = case["torsion"] / sumr
    L.append("*CLOAD")
    for n, y in tip:
        L.append(f"{n}, 1, {case['axial'] / len(tip):.6f}")
        L.append(f"{n}, 2, {case['side'] / len(tip):.6f}")
        L.append(f"{n}, 3, {c * (y - yc):.6f}")
    L.append("*END STEP")
    return "\n".join(L) + "\n"


class SolverFailure(RuntimeError):
    """The FE solve did not produce a result. NOT a bad candidate: a failure.

    Esiste perche' fino al 26/08/2026 ogni guasto di questa funzione (binario ccx assente,
    exit code != 0 -- mai controllato --, .dat assente o troncato, regex che non matcha,
    timeout) tornava -1.0, che e' anche un fattore di buckling POSSIBILE (i fattori negativi
    esistono: carico invertito). Il GA lo trattava da candidato pessimo. Dimostrato: una
    campagna intera lanciata senza solutore terminava con un "INFEASIBLE up to N ply" pulito e
    la console muta (la policy di log e' errors-only e il warning finiva nel journal ruotato),
    e la certificazione di exp7 sarebbe passata con gap 0,00%.
    """


# Valvola di sicurezza per gli esperimenti che PROVANO deliberatamente i guasti: se settata,
# si torna alla sentinella storica. Non usarla in una campagna.
_ALLOW_SENTINEL = os.environ.get("CCX_ALLOW_FAILURE_SENTINEL", "0") == "1"
SOLVER_FAILURES = 0          # contatore di processo, letto dagli esperimenti che lo riportano


def _solver_failed(msg: str, exc: Exception | None = None):
    """Un guasto del solutore: contalo, dillo a livello ERROR (che la console vede), e alzalo."""
    global SOLVER_FAILURES
    SOLVER_FAILURES += 1
    log.error("buckling eval: %s -- NON e' un candidato pessimo, e' un guasto del solutore", msg)
    if _ALLOW_SENTINEL:
        return -1.0
    raise SolverFailure(msg) from exc


def buckling_factor(args):
    seq, case = args[0], args[1]
    mesh = args[2] if len(args) > 2 else None  # (seq, case, mesh) opts into a refined grid
    d = tempfile.mkdtemp()
    try:
        open(d + "/job.inp", "w").write(make_ccx_deck(seq, case, mesh=mesh))
        r = subprocess.run([CCX, "-i", "job"], cwd=d, capture_output=True, timeout=120,
                           env={**os.environ, "OMP_NUM_THREADS": "1"})  # ccx<2.21 threaded buckling is wrong
        if r.returncode != 0:
            return _solver_failed("ccx exit code %d" % r.returncode)
        if not os.path.exists(d + "/job.dat"):
            return _solver_failed("ccx non ha scritto job.dat")
        dat = open(d + "/job.dat").read()
        f = re.findall(r"^\s*1\s+([\d.E+\-]+)\s*$", dat, re.M)
        if not f:
            return _solver_failed("nessun autovalore nel .dat (%d byte)" % len(dat))
        return float(f[0])
    except subprocess.TimeoutExpired as e:
        return _solver_failed("ccx timed out (>120s) su un candidato", e)
    except SolverFailure:
        raise
    except Exception as e:
        return _solver_failed("eval fallita (%s)" % e, e)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---- static constraints (displacement + in-plane stress) on the shell ----------
# Composite_static_1.inp load case: axial 20000, side 5000 (own load, scaled by LOAD_SCALE).
STATIC_LOAD = dict(axial=10000., side=3000., torsion=0.)  # 2026-06-05: moderated so static is a check with margin
DISP_MAX, SIGMA_MAX, PEEL_MAX = 1.1, 700.0, 10.0   # constraints 6, 5 (in-plane), 5 (peel)


def _delam_Q(seq, sload):
    """Q dal solutore solido. Un GUASTO alza, non torna None.

    `interlaminar` segnala il guasto con un dict {"error": ...} SENZA la chiave Q: i due loop
    di questo modulo facevano `.get("Q")` e ottenevano None, che poi finiva nel ramo
    `status="delam"` -- cioe' un guasto del solutore era indistinguibile da "calcolato, e Q>=1",
    e lo sweep ascendente proseguiva come se il vincolo fosse stato misurato e violato.
    """
    from fe.interlaminar import interlaminar
    il = interlaminar(seq, axial=sload["axial"], side=sload["side"], nx=20, ny=10)
    if "error" in il or il.get("Q") is None:
        raise SolverFailure("delamination solve fallita: %s"
                            % {k: v for k, v in il.items() if k != "log"})
    return il["Q"]


def _violated():
    """Dict di vincoli violati. Porta le STESSE chiavi del dict di successo, `peel_max`
    compresa: i due dict divergevano, e un consumatore che facesse `m["peel_max"]` andava in
    KeyError sul ramo d'errore, o -- con `.get` -- in un altro None silenzioso."""
    inf = float("inf")
    return dict(disp=inf, sx=inf, sy=inf, peel=inf, peel_max=inf, ishear=inf)


def make_ccx_static_deck(seq, sload, mesh=None):
    """Shell *STATIC deck for the static constraints: max |U| and in-plane ply stress."""
    d = make_ccx_deck(seq, sload, mesh=mesh)  # reuse geometry/mesh/section/CLOAD builder
    d = d.replace("*STEP\n*BUCKLE\n2\n", "*STEP\n*STATIC\n")
    d = d.replace("*END STEP", "*NODE FILE\nU\n*EL FILE\nS\n*END STEP")
    return d


_SCI = re.compile(r"[-+]?\d+\.\d+E[-+]\d+")


def static_metrics(args):
    """From a shell static run, return dict: max |U| (mm) + max-magnitude stress components
    from the full .frd STRESS tensor (ccx expands the shell to 3D, so all 6 are present):
    sx=|SXX|, sy=|SYY|, peel=|SZZ| (through-thickness normal), ishear=max(|SYZ|,|SZX|).
    NOTE: SZZ/interlaminar from a shell EXPANSION is an ESTIMATE; a 3D solid is the gold
    standard for interlaminar peel."""
    seq, sload = args[0], args[1]
    mesh = args[2] if len(args) > 2 else None  # (seq, sload, mesh) opts into a refined grid
    try:
        from fe.frd_parse import parse_frd_static
    except ImportError:
        from fe.frd_parse import parse_frd_static
    d = tempfile.mkdtemp()
    try:
        open(d + "/job.inp", "w").write(make_ccx_static_deck(seq, sload, mesh=mesh))
        subprocess.run([CCX, "-i", "job"], cwd=d, capture_output=True, timeout=120,
                       env={**os.environ, "OMP_NUM_THREADS": "1"})  # ccx<2.21 threaded buckling is wrong
        frd = open(d + "/job.frd").read() if os.path.exists(d + "/job.frd") else ""
        umax = max((x["umag"] for x in parse_frd_static(frd)["disp"]), default=float("inf"))
        sx = sy = 0.0
        szz, ishv = [], []
        instress = False
        nstress = 0                  # righe del blocco STRESS effettivamente lette (vedi sotto)
        for ln in frd.splitlines():
            s = ln.lstrip()
            if s.startswith("-4") and "STRESS" in ln:
                instress = True
                continue
            if instress and s.startswith("-4"):
                break
            if instress and s.startswith("-1"):
                v = [float(x) for x in _SCI.findall(ln)]
                if len(v) >= 6:  # SXX,SYY,SZZ,SXY,SYZ,SZX
                    nstress += 1
                    sx = max(sx, abs(v[0])); sy = max(sy, abs(v[1]))
                    szz.append(abs(v[2])); ishv.append(max(abs(v[4]), abs(v[5])))
        # Le due meta' di questa funzione avevano polarita' OPPOSTE: lo spostamento e'
        # fail-closed (`default=inf`), lo sforzo era fail-open (sx=sy=0.0 di partenza, e il 95o
        # percentile di una lista vuota vale 0.0). Un .frd con blocco DISP ma SENZA blocco
        # STRESS -- output troncato, `*EL FILE` perso, scrittura interrotta -- dava quindi
        # disp finito e TUTTE le metriche di sforzo a 0.0 MPa, e il gate
        # `sx <= SIGMA_MAX and sy <= SIGMA_MAX` passava. Ora manca il blocco = vincoli violati.
        if nstress == 0:
            log.error("static eval: nessuna riga nel blocco STRESS (frd di %d byte) -- "
                      "vincoli di resistenza trattati come VIOLATI, non come soddisfatti",
                      len(frd))
            return _violated()
        # AVERAGED peel criterion: interlaminar SZZ is SINGULAR at free edges, so a point
        # max is ill-posed (mesh-dependent). Use the 95th percentile (robust averaged peak).
        def pct(a, p):
            if not a:
                return 0.0
            a = sorted(a)
            return a[min(len(a) - 1, int(p / 100 * len(a)))]
        return dict(disp=umax, sx=sx, sy=sy,
                    peel=pct(szz, 95), peel_max=max(szz, default=0.0), ishear=pct(ishv, 95))
    except subprocess.TimeoutExpired:
        log.error("static eval: ccx timed out (>120s) -- constraints treated as violated")
        return _violated()
    except Exception as e:
        log.error("static eval failed (%s) -- constraints treated as violated", e)
        return _violated()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---- genetic algorithm (true min-ply search) ------------------------------------
def repair(seq, alpha):
    """Make a sequence manufacturing-valid in place (<=45 step, <=3 consecutive)."""
    out = [seq[0] if seq[0] in alpha else alpha[0]]
    for a in seq[1:]:
        ok = adiff(out[-1], a) <= 45 and not (len(out) >= 3 and out[-1] == out[-2] == out[-3] == a)
        if ok:
            out.append(a)
        else:
            cand = [x for x in alpha if adiff(out[-1], x) <= 45
                    and not (len(out) >= 3 and out[-1] == out[-2] == out[-3] == x)]
            out.append(min(cand, key=lambda x: adiff(x, a)) if cand else out[-1])
    return out


def ga_best(case, n, alpha, pool, rng, pop=GA_POP, gens=GA_GENS, guided=False, trace=None):
    """Evolve a population of valid N-ply sequences to MAXIMISE the buckling factor.
    guided=True restricts the search to symmetric/balanced/10%-compliant laminates (the
    standard design guidelines), using guideline-preserving regeneration instead of
    crossover (which would break symmetry/balance).

    trace, if given, is a list that receives (evaluations_so_far, best_so_far) after the
    initial population and after every generation. It is read-only bookkeeping: it draws no
    random numbers and changes no branch, so a run with trace=None is bit-identical to the
    same run with a trace attached. exp19_budget_convergence.py uses it to read a whole
    budget curve off ONE run instead of re-running the same prefix once per budget point."""
    gen = gen_guided if guided else gen_valid
    if guided and os.environ.get("EXACT_SAMPLER", "0") == "1":
        gen = gen_guided_exact        # opt-in: uniform exact draw, no rejection (see there)
    seeds = [[0], [45, -45], [0, 45, -45], None]
    P = []
    tries = 0
    while len(P) < pop and tries < pop * 8:
        pr = seeds[len(P) % len(seeds)]
        s = gen(alpha, n, rng, prefer=pr, pbias=0.6 if pr else 0.0)
        tries += 1
        if s:
            P.append(s)
    if not P:
        return -1.0, None
    fits = pool.map(buckling_factor, [(p, case) for p in P])
    n_eval = len(P)
    if trace is not None:
        trace.append((n_eval, max(fits)))
    for _ in range(gens):
        ranked = [p for _, p in sorted(zip(fits, P), key=lambda t: -t[0])]
        elite = ranked[:max(2, pop // 3)]
        kids = []
        if guided:                              # regenerate compliant individuals (no crossover)
            while len(kids) < pop - len(elite):
                s = gen(alpha, n, rng, prefer=rng.choice(seeds), pbias=0.6)
                if s:
                    kids.append(s)
        else:
            while len(kids) < pop - len(elite):
                a, b = rng.choice(elite), rng.choice(elite)
                cut = rng.randint(1, n - 1)
                child = repair(a[:cut] + b[cut:], alpha)
                if rng.random() < 0.35:
                    i = rng.randrange(n)
                    child[i] = rng.choice(alpha)
                    child = repair(child, alpha)
                kids.append(child)
        P = elite + kids
        fits = pool.map(buckling_factor, [(p, case) for p in P])
        if trace is not None:
            n_eval += len(P)
            trace.append((n_eval, max(max(fits), trace[-1][1])))
    bf, seq = max(zip(fits, P), key=lambda t: t[0])
    return bf, seq


def optimise_case_ga(name, case, alpha, ply_counts, pool, scale, seed=0, budget=None):
    """Ascending ply-count; per N a GA maximises buckling; first N meeting the buckling
    threshold AND the static constraints (disp<1.1, in-plane sigma<700) wins.

    `budget` (a Budget) is checked at each ply-count step: if the local cap is blown the
    case aborts with an error pointing at cluster, instead of grinding on for hours."""
    rng = random.Random(seed)
    guided = os.environ.get("GUIDELINES", "0") == "1"
    sload = {**STATIC_LOAD, "axial": STATIC_LOAD["axial"] * scale,
             "side": STATIC_LOAD["side"] * scale, "torsion": 0., "threshold": 0.}
    for n in ply_counts:
        if budget is not None:
            reason = budget.overrun()
            if reason:
                log.error("%s: optimisation budget exhausted (%s) at N=%d -- aborting this "
                          "case locally. If a longer search is expected, dispatch to cluster "
                          "(COMPOSITE_TARGET=cluster), not this machine.", name, reason, n)
                return dict(name=name, n_plies=None, seq=None, budget_exceeded=True)
            budget.tick(EVALS_PER_PLY_STEP)
        bf, seq = ga_best(case, n, alpha, pool, rng, guided=guided)
        if seq is None:
            log.info("  %s N=%2d: no guideline-compliant candidate", name, n)
            continue
        if bf < case["threshold"]:
            log.info("  %s N=%2d: GA buckling BF=%7.3f (<%s) -> no", name, n, bf, case["threshold"])
            continue
        m = static_metrics((seq, sload))
        ok_shell = (m["disp"] <= DISP_MAX and m["sx"] <= SIGMA_MAX and m["sy"] <= SIGMA_MAX)
        Q = None
        if ok_shell:  # GOLD-STANDARD delamination check (winner-only: a full solid solve)
            Q = _delam_Q(seq, sload)
        ok = ok_shell and Q is not None and Q < 1.0
        qs = f"{Q:.4f}" if Q is not None else "n/a"
        log.info("  %s N=%2d: BF=%6.2f>=%s | disp=%.3f sx=%.0f sy=%.0f | delam Q=%s -> %s",
                 name, n, bf, case["threshold"], m["disp"], m["sx"], m["sy"], qs,
                 "FEASIBLE" if ok else ("static-fail" if not ok_shell else "delam"))
        if ok:
            return dict(name=name, n_plies=n, bf=bf, seq=seq, delam_Q=Q, **m)
    return dict(name=name, n_plies=None, seq=None)


def optimise_case(name, case, alpha, ply_counts, ncand, pool, seed=0, budget=None):
    """Ascending ply-count; per N a parallel orientation search; first N whose best
    candidate meets the threshold wins (min plies). `budget` aborts the case if the
    local wall-clock / FE-eval cap is blown (see optimise_case_ga)."""
    rng = random.Random(seed)
    # load-aware bias strategies: 0-heavy (axial), +-45-heavy (shear/torsion), mixed, random
    biases = [([0], 0.8), ([45, -45], 0.8), ([0, 45, -45], 0.7), (None, 0.0)]
    for n in ply_counts:
        if budget is not None:
            reason = budget.overrun()
            if reason:
                log.error("%s: optimisation budget exhausted (%s) at N=%d -- aborting this "
                          "case locally. If a longer search is expected, dispatch to cluster "
                          "(COMPOSITE_TARGET=cluster), not this machine.", name, reason, n)
                return dict(name=name, n_plies=None, bf=None, seq=None, budget_exceeded=True)
            budget.tick(ncand)
        cands = []
        for prefer, pb in biases:
            for _ in range(max(1, ncand // len(biases) + 1)):
                s = gen_valid(alpha, n, rng, prefer=prefer, pbias=pb)
                if s:
                    cands.append(s)
        if not cands:
            continue
        facs = pool.map(buckling_factor, [(s, case) for s in cands])
        best_f, best_s = max(zip(facs, cands), key=lambda t: t[0])
        feasible = best_f >= case["threshold"]
        log.info("  %s N=%2d: best BF=%7.3f (target %s) -> %s",
                 name, n, best_f, case["threshold"], "FEASIBLE" if feasible else "no")
        if feasible:
            return dict(name=name, n_plies=n, bf=best_f, seq=best_s)
    return dict(name=name, n_plies=None, bf=None, seq=None)


OPTIMISERS = {"GA", "ACO", "PSO"}
MODELS = {"shell", "solid"}


def _run_optimiser(optimiser, case, n, alpha, pool, rng, guided=False):
    """Route to the requested metaheuristic and return (bf, seq).

    "GA" stays on the in-module guideline-aware ga_best (so guided= keeps working and
    backward compatibility is preserved). "ACO"/"PSO" delegate to rr_metaheuristics,
    whose ga/aco/pso read the alphabet from the module global ALPHA -- we set it
    explicitly per call so the chosen alphabet is honoured (never silent).
    """
    if optimiser == "GA":
        return ga_best(case, n, alpha, pool, rng, guided=guided)
    try:
        import optimisers.metaheuristics as mh
    except ImportError:
        import optimisers.metaheuristics as mh
    mh.ALPHA = alpha                                # honour the requested alphabet
    fn = mh.aco if optimiser == "ACO" else mh.pso
    # small budgets keep the embedded/GUI search responsive; ga_best uses pop=12,gens=4.
    bf, seq = fn(case, n, pool, rng, iters=4)
    return bf, seq


def optimise_stacking(case_name, alphabet_name, ply_counts, scale=1.0, seed=1,
                      pool=None, should_cancel=None, guided=False,
                      optimiser="GA", model="shell",
                      axial=None, side=None, target=None):
    """GUI-facing entry: ascending ply-count metaheuristic search on the fast (validated)
    20x10 generator. Returns a structured result (no stdout). `should_cancel()` is polled
    between ply counts.

    `optimiser` in {GA, ACO, PSO} selects the metaheuristic (CeNoSilentOptimiserOrLoad:
    surfaced in the result, never silent). Explicit `axial`/`side`/`target` override the
    CASES entry so the user supplies the loads (CeNoSilentOptimiserOrLoad). `model` in
    {shell, solid} (SelectModel): "shell" takes the 2D-shell stresses and DROPS the
    delamination Q (reported null + an explicit note), "solid" computes Q via the 3D
    interlaminar solid and includes it in feasibility. Feasibility ALWAYS requires every
    active constraint -- buckling AND sigma_x AND sigma_y AND displacement AND (solid) Q --
    never buckling alone (CeNoBucklingOnlyFeasible).

    -> {case, alphabet, optimiser, model, threshold, loads, feasible, n_plies, bf, sequence,
        delam_Q, delam_note, disp, sx, sy, scale, trace:[{n, bf, status, ...}]}.
        feasible=False with n_plies=None if none found.
    """
    if optimiser not in OPTIMISERS:
        raise ValueError(f"unknown optimiser {optimiser!r}; choose from {sorted(OPTIMISERS)}")
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {sorted(MODELS)}")
    if alphabet_name not in ALPHABETS:
        raise ValueError(f"unknown alphabet {alphabet_name!r}; choose from {list(ALPHABETS)}")
    explicit = axial is not None or side is not None or target is not None
    if explicit:
        # User-supplied loads override the curated case (CeNoSilentOptimiserOrLoad). A
        # missing component is an explicit error rather than a silent default of the
        # curated case it would otherwise leak in from.
        if axial is None or side is None or target is None:
            raise ValueError("explicit loads require axial, side and target together")
        base = {"axial": float(axial), "side": float(side), "torsion": 0.,
                "threshold": float(target)}
    else:
        if case_name not in CASES:
            raise ValueError(f"unknown case {case_name!r}; choose from {list(CASES)}")
        base = CASES[case_name]
    case = {**base, "axial": base["axial"] * scale, "side": base["side"] * scale,
            "torsion": base["torsion"] * scale}
    alpha = ALPHABETS[alphabet_name]
    rng = random.Random(seed)
    sload = {**STATIC_LOAD, "axial": STATIC_LOAD["axial"] * scale,
             "side": STATIC_LOAD["side"] * scale, "torsion": 0., "threshold": 0.}
    loads = {"axial": case["axial"], "side": case["side"], "target": case["threshold"],
             "explicit": explicit}
    drop_note = ("2D shell fallback: delamination Q constraint dropped (3D solid not in use); "
                 "Q reported as null, not fabricated") if model == "shell" else None
    own_pool = pool is None
    if own_pool:
        # fork context so this works embedded in a server (uvicorn) without the spawn
        # re-import-__main__ recursion; falls back to default if fork is unavailable.
        import multiprocessing as _mp
        nproc = int(os.environ.get("NPROC", str(os.cpu_count())))
        try:
            pool = _mp.get_context("fork").Pool(nproc)
        except (ValueError, OSError):
            pool = Pool(nproc)

    def _result(extra):
        r = {"case": case_name, "alphabet": alphabet_name, "optimiser": optimiser,
             "model": model, "threshold": case["threshold"], "loads": loads,
             "scale": scale, "delam_note": drop_note}
        r.update(extra)
        return r

    trace = []
    try:
        for n in ply_counts:
            if should_cancel and should_cancel():
                from febatch.errors import Cancelled  # type: ignore
                raise Cancelled("optimise_stacking cancelled")
            bf, seq = _run_optimiser(optimiser, case, n, alpha, pool, rng, guided=guided)
            if seq is None:
                trace.append({"n": n, "status": "no-candidate"})
                continue
            if bf < case["threshold"]:
                trace.append({"n": n, "bf": round(bf, 3), "status": "buckling-fail"})
                continue
            m = static_metrics((seq, sload))
            ok_shell = (m["disp"] <= DISP_MAX and m["sx"] <= SIGMA_MAX and m["sy"] <= SIGMA_MAX)
            Q = None
            if model == "solid" and ok_shell:       # winner-only gold-standard delamination
                Q = _delam_Q(seq, sload)
            # CeNoBucklingOnlyFeasible: feasibility needs ALL active constraints. In solid
            # mode Q is binding (must be < 1); in shell mode Q is dropped (not fabricated),
            # so feasibility rests on buckling + strength + displacement only.
            if model == "solid":
                ok = ok_shell and Q is not None and Q < 1.0
                status = "feasible" if ok else ("static-fail" if not ok_shell else "delam")
            else:
                ok = ok_shell
                status = "feasible" if ok else "static-fail"
            trace.append({"n": n, "bf": round(bf, 3), "disp": round(m["disp"], 3),
                          "sx": round(m["sx"]), "sy": round(m["sy"]),
                          "delam_Q": round(Q, 4) if Q is not None else None, "status": status})
            if ok:
                return _result({"feasible": True, "n_plies": n, "bf": round(bf, 3),
                                "sequence": seq,
                                "delam_Q": round(Q, 4) if Q is not None else None,
                                "disp": round(m["disp"], 3), "sx": round(m["sx"]),
                                "sy": round(m["sy"]), "trace": trace})
        return _result({"feasible": False, "n_plies": None, "sequence": None,
                        "trace": trace})
    finally:
        if own_pool:
            pool.close(); pool.join()


if __name__ == "__main__":
    import sys
    import multiprocessing as _mp
    # macOS 'spawn' can deadlock this Pool (see tests/conftest.py); the deployed runner
    # is Linux/fork. Force fork so a local run cannot hang forever inside pool.map.
    try:
        _mp.set_start_method("fork")
    except RuntimeError:
        pass

    setup_logging()   # errors-only console + bounded rotating journal in cases/_out
    aset = sys.argv[1] if len(sys.argv) > 1 else "set1"
    alpha = ALPHABETS[aset]
    scale = float(os.environ.get("LOAD_SCALE", str(DEFAULT_LOAD_SCALE)))
    cases = {k: {**v, "axial": v["axial"] * scale, "side": v["side"] * scale,
                 "torsion": v["torsion"] * scale} for k, v in CASES.items()}
    plies = [16, 24, 32, 40, 44, 48, 52, 56, 60]
    ncand = int(os.environ.get("NCAND", "12"))
    t0 = time.time()
    with Pool(int(os.environ.get("NPROC", str(os.cpu_count())))) as pool:
        log.info("=== alphabet %s %s | load_scale %s | GA | %d workers ===",
                 aset, alpha, scale, pool._processes)
        results = {}
        for name, case in cases.items():
            log.info("[%s] buckling target > %s + disp<%s + sigma<%s",
                     name, case["threshold"], DISP_MAX, SIGMA_MAX)
            # one fresh budget per (problem, method): a per-case local ceiling
            budget = Budget()
            results[name] = optimise_case_ga(name, case, alpha, plies, pool, scale,
                                             seed=1, budget=budget)
    # RESULT is the bounded, one-shot deliverable -> always to stdout.
    print(f"\n=== RESULT (alphabet {aset}, load_scale {scale}, GA, {time.time()-t0:.0f}s) ===")
    for name, r in results.items():
        if r.get("seq"):
            print(f"{name}: {r['n_plies']} ply, BF={r['bf']:.2f} | disp={r['disp']:.3f} "
                  f"sx={r['sx']:.0f} sy={r['sy']:.0f} delamQ={r.get('delam_Q',0):.4f} | seq={r['seq']}")
        elif r.get("budget_exceeded"):
            print(f"{name}: ABORTED — optimisation budget exceeded (see ERROR log); "
                  f"dispatch to cluster for a longer search")
        else:
            print(f"{name}: INFEASIBLE up to {plies[-1]} ply (buckling or static)")
