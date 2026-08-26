"""Run an arbitrary CalculiX .inp deck and parse its results into one schema.

Phase 1: ccx solver only; modal + buckling parsing (reuses canale_plate parsers).
Static-displacement / von-Mises (.frd) and the Abaqus backend are Phase 2."""
from __future__ import annotations
import math
import os
import re
import subprocess

from fe.plate_model import CCX, parse_buckling_factors, parse_frequencies

_NODE_RE = re.compile(r"^\s*\*NODE\b", re.I)
_ELEM_RE = re.compile(r"^\s*\*ELEMENT\b", re.I)
_KEYWORD_RE = re.compile(r"^\s*\*")
_TYPE_RE = re.compile(r"TYPE\s*=\s*([A-Za-z0-9]+)", re.I)


def mesh_from_deck(deck_text: str) -> dict:
    """Parse *NODE/*ELEMENT cards into {"nodes": [[x,y,z]], "elements":[{type,conn}]}.

    Node ids are remapped to dense 0-based indices by first appearance. Raises
    ValueError if the deck has no parseable nodes (not a recognisable mesh)."""
    nodes_by_id: dict[int, list[float]] = {}
    raw_elems: list[tuple[str, list[int]]] = []
    mode = None  # "node" | "elem" | None
    cur_type = None
    elem_buf: list[int] | None = None  # open element connectivity (continuation)

    def _flush_elem():
        nonlocal elem_buf
        if elem_buf is not None:
            raw_elems.append((cur_type, elem_buf))
            elem_buf = None

    for line in deck_text.splitlines():
        if _NODE_RE.match(line):
            _flush_elem(); mode = "node"; continue
        if _ELEM_RE.match(line):
            _flush_elem()
            mode = "elem"
            m = _TYPE_RE.search(line)
            cur_type = (m.group(1).upper() if m else "UNKNOWN")
            continue
        if not line.strip() or line.lstrip().startswith("**"):
            continue
        if _KEYWORD_RE.match(line):
            _flush_elem(); mode = None; continue
        parts = [p for p in line.replace(",", " ").split() if p]
        if mode == "node" and len(parts) >= 4:
            nid = int(float(parts[0]))
            nodes_by_id[nid] = [float(parts[1]), float(parts[2]), float(parts[3])]
        elif mode == "elem" and len(parts) >= 1:
            if elem_buf is not None:
                # continuation of the previous element: all tokens are node ids
                elem_buf.extend(int(float(p)) for p in parts)
            else:
                # parts[0] is the element id; the rest is connectivity
                elem_buf = [int(float(p)) for p in parts[1:]]
            if line.rstrip().endswith(","):
                continue  # trailing comma => more connectivity on next line
            _flush_elem()
    if not nodes_by_id:
        raise ValueError("not a recognisable mesh: no *NODE cards found")
    index = {nid: i for i, nid in enumerate(nodes_by_id)}
    node_ids = list(nodes_by_id)  # original node ids in vertex order
    nodes = [nodes_by_id[nid] for nid in nodes_by_id]
    elements = [{"type": t, "conn": [index[i] for i in ids if i in index]}
                for t, ids in raw_elems]
    return {"nodes": nodes, "node_ids": node_ids, "elements": elements}


def parse_results(artifacts: dict, solver: str = "ccx", returncode: int | None = None,
                  expect_modes: int | None = None) -> dict:
    """Normalise solver output into the schema. Phase 1: ccx .dat (modal+buckling).

    `converged` era `bool(modal or buckling or static["disp"])`, cioe' un'euristica SUL PARSE e
    non sul solutore: un .dat parziale (processo ucciso a meta' scrittura) con 3 modi su 10
    dava `converged: True`, e il `returncode` veniva restituito ma mai consultato. Ora, se il
    chiamante li passa, `converged` richiede anche `returncode == 0` e il numero atteso di
    autovalori; se non li passa resta l'euristica, e il campo `converged_basis` dice quale
    delle due e' stata usata, invece di lasciarlo intendere.
    """
    dat = artifacts.get("dat")
    modal, buckling = [], []
    if dat and os.path.exists(dat):
        freqs = parse_frequencies(dat)
        bucks = parse_buckling_factors(dat)
        modal = [{"mode": i + 1, "freq_hz": f}
                 for i, f in enumerate(freqs) if math.isfinite(f) and f > 0]
        buckling = [{"mode": i + 1, "factor": x}
                    for i, x in enumerate(bucks) if math.isfinite(x)]
    static = {"disp": [], "von_mises": []}
    frd = artifacts.get("frd")
    if frd and os.path.exists(frd):
        from fe.frd_parse import parse_frd_static
        with open(frd) as f:
            static = parse_frd_static(f.read())
    converged = bool(modal or buckling or static["disp"])
    basis = "parse-heuristic"
    if returncode is not None:
        converged = converged and returncode == 0
        basis = "returncode"
    if expect_modes is not None:
        converged = converged and len(modal or buckling) >= expect_modes
        basis = basis + "+mode-count"
    return {"modal": modal, "buckling": buckling, "static": static,
            "converged": converged, "converged_basis": basis,
            "returncode": returncode, "solver": solver}


def run_deck(deck_text: str, workdir: str, cpus: int = 2, should_cancel=None) -> dict:
    """Write the deck and run it in ccx. Returns {"dat","frd","log","returncode"}.

    Honours should_cancel before launching the (blocking) solve. The caller owns
    workdir cleanup (mirrors LocalBackend's finally:_rmtree pattern)."""
    if should_cancel is not None and should_cancel():
        from febatch.errors import Cancelled
        raise Cancelled()
    os.makedirs(workdir, exist_ok=True)
    job = "job"
    inp = os.path.join(workdir, job + ".inp")
    with open(inp, "w") as f:
        f.write(deck_text)
    # Eigenvalue (*BUCKLE/*FREQUENCY) solves default to SERIAL (OMP=1): correct on ANY ccx
    # version (ccx<2.21's THREADED eigensolve returns wrong factors, e.g. a spurious ~1.0
    # buckling factor) AND threading gives no speedup for these anyway (ccx ARPACK/spooles
    # doesn't parallelise the eigensolve). Set CCX_EIGEN_THREADED=1 to use the requested cpus
    # (only safe on ccx>=2.21). Static/other steps always use the requested cpus.
    low = deck_text.lower()
    eigen = ("*buckle" in low) or ("*frequency" in low)
    threads = 1 if (eigen and os.environ.get("CCX_EIGEN_THREADED") != "1") else cpus
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    proc = subprocess.run([CCX, job], cwd=workdir, capture_output=True, text=True, env=env)
    log = proc.stdout + "\n" + proc.stderr
    with open(os.path.join(workdir, job + ".log"), "w") as f:
        f.write(log)
    return {"dat": os.path.join(workdir, job + ".dat"),
            "frd": os.path.join(workdir, job + ".frd"),
            "log": log, "returncode": proc.returncode}
