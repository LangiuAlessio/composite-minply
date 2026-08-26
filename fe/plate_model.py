#!/usr/bin/env python3
"""
Canale composite plate — frequency + buckling baseline (CalculiX).

Concrete problem contributed by the coauthor Giacomo Canale (composites):
  Flat rectangular CFRP plate 100 mm x 50 mm, 20 plies @ 0.125 mm = 2.5 mm laminate.
  Evaluate two constraints for a baseline layup:
    - 7th natural frequency (mode 7) > 60 Hz
    - Buckling load > 5000 N (uniaxial in-plane compression along X / 100 mm side)

Unit system: mm, N, MPa (=N/mm^2), tonne, s  -> eigenfrequencies come out in Hz.
  density of CFRP 1600 kg/m^3 = 1.6e-9 tonne/mm^3
  E1=135 GPa = 135000 MPa, etc.

Mesh: structured grid of 8-node quadratic shell elements (S8R), written directly
in Python (a flat rectangle is trivial and this avoids fragile gmsh quad8 extraction).
gmsh is used ONLY to export a STEP solid of the plate for CAD/visualization.

Run:  source .venv/bin/activate && python cases/canale_plate.py
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "_out", "canale")
from fe.ccx_bin import CCX          # un solo default per il binario ccx (audit F12)

LX = 100.0          # mm, plate length along X (compression direction for buckling)
LY = 50.0           # mm, plate width along Y
PLY_T = 0.125       # mm per ply
N_PLIES = 20
THICK = PLY_T * N_PLIES      # 2.5 mm total laminate thickness

# Baseline layup: [45/-45/0/90/0]_2s
#   "_2s" => the 10-ply stack [45,-45,0,90,0,45,-45,0,90,0] mirrored about midplane.
HALF_STACK = [45, -45, 0, 90, 0, 45, -45, 0, 90, 0]   # 10 plies (bottom half)
LAYUP = HALF_STACK + HALF_STACK[::-1]                 # 20 plies, symmetric
assert len(LAYUP) == N_PLIES

# Orthotropic lamina default (generic T300/epoxy), MPa / tonne-mm-s units.
# Single source of truth is the cited registry; MAT re-exports the T300 entry so
# existing `mat=MAT` defaults are unchanged (regression-safe).
# NOT the design-campaign lamina: that one is the `canale2018` entry of the same registry
# (E1 125100 MPa, ply 0.1 mm), which optimisers/constrained_search.py writes into its decks.
from fe.materials import MATERIALS, get_material  # noqa: E402,F401
MAT = MATERIALS["T300/epoxy"]

NEL_X = 40          # elements along X
NEL_Y = 20          # elements along Y
N_FREQ = 10         # eigenvalues for frequency
N_BUCK = 5          # eigenvalues for buckling
REF_LOAD = 1000.0   # N total reference compressive edge load


# ----------------------------------------------------------------------------
# Structured S8R mesh
# ----------------------------------------------------------------------------
@dataclass
class Mesh:
    nodes: dict = field(default_factory=dict)        # nid -> (x,y,z)
    elements: dict = field(default_factory=dict)     # eid -> [8 node ids]
    edge_x0: list = field(default_factory=list)
    edge_x1: list = field(default_factory=list)
    edge_y0: list = field(default_factory=list)
    edge_y1: list = field(default_factory=list)


def build_mesh(nelx: int, nely: int, lx: float = LX, ly: float = LY) -> Mesh:
    """Structured grid of 8-node serendipity quads (CalculiX S8R).

    An S8R element has 4 corner + 4 mid-edge nodes (no center node). We build a
    node grid of dimension (2*nelx+1) x (2*nely+1) and drop the element-center
    nodes (odd-i, odd-j) which serendipity elements do not use.

    `lx`/`ly` set the plate size (default = module LX/LY, so existing callers are
    unchanged); pass them to mesh an arbitrary-size plate (Abaqus .inp export).
    """
    m = Mesh()
    nx = 2 * nelx + 1     # grid points along X (including mid-edge)
    ny = 2 * nely + 1
    dx = lx / (2 * nelx)
    dy = ly / (2 * nely)

    # Node grid; skip element-center points (i odd AND j odd).
    nid = 0
    grid = {}  # (i,j) -> nid
    for j in range(ny):
        for i in range(nx):
            if (i % 2 == 1) and (j % 2 == 1):
                continue  # serendipity: no element-center node
            nid += 1
            x = i * dx
            y = j * dy
            m.nodes[nid] = (x, y, 0.0)
            grid[(i, j)] = nid

    # Elements: corner block of size 2x2 in the (i,j) grid.
    # CalculiX S8R node ordering: 4 corners CCW, then 4 mid-side nodes.
    #   corners: (0,0)(2,0)(2,2)(0,2)
    #   midsides: (1,0)(2,1)(1,2)(0,1)
    eid = 0
    for ej in range(nely):
        for ei in range(nelx):
            i0 = 2 * ei
            j0 = 2 * ej
            n1 = grid[(i0,     j0)]
            n2 = grid[(i0 + 2, j0)]
            n3 = grid[(i0 + 2, j0 + 2)]
            n4 = grid[(i0,     j0 + 2)]
            n5 = grid[(i0 + 1, j0)]
            n6 = grid[(i0 + 2, j0 + 1)]
            n7 = grid[(i0 + 1, j0 + 2)]
            n8 = grid[(i0,     j0 + 1)]
            eid += 1
            m.elements[eid] = [n1, n2, n3, n4, n5, n6, n7, n8]

    # Edge node sets (by coordinate)
    tol = 1e-6
    for nid, (x, y, _z) in m.nodes.items():
        if abs(x) < tol:
            m.edge_x0.append(nid)
        if abs(x - lx) < tol:
            m.edge_x1.append(nid)
        if abs(y) < tol:
            m.edge_y0.append(nid)
        if abs(y - ly) < tol:
            m.edge_y1.append(nid)
    return m


# ----------------------------------------------------------------------------
# Deck writers
# ----------------------------------------------------------------------------
def _nset_lines(name: str, ids: list) -> list[str]:
    out = [f"*NSET, NSET={name}"]
    for k in range(0, len(ids), 8):
        out.append(", ".join(str(i) for i in ids[k:k + 8]))
    return out


def _common_model(m: Mesh, seq=LAYUP, mat=MAT, ply_t=PLY_T) -> list[str]:
    L: list[str] = []
    L.append("*NODE, NSET=NALL")
    for nid, (x, y, z) in sorted(m.nodes.items()):
        L.append(f"{nid}, {x:.6f}, {y:.6f}, {z:.6f}")
    L.append("*ELEMENT, TYPE=S8R, ELSET=PLATE")
    for eid, nn in sorted(m.elements.items()):
        L.append(f"{eid}, " + ", ".join(str(n) for n in nn))

    # edge node sets
    L += _nset_lines("EDGE_X0", sorted(set(m.edge_x0)))
    L += _nset_lines("EDGE_X1", sorted(set(m.edge_x1)))
    L += _nset_lines("EDGE_Y0", sorted(set(m.edge_y0)))
    L += _nset_lines("EDGE_Y1", sorted(set(m.edge_y1)))

    # orthotropic material. CalculiX *ELASTIC,TYPE=ENGINEERING CONSTANTS order:
    #   E1,E2,E3,nu12,nu13,nu23,G12,G13 / G23,T
    L.append("*MATERIAL, NAME=LAMINA")
    L.append("*ELASTIC, TYPE=ENGINEERING CONSTANTS")
    L.append(f"{mat['E1']}, {mat['E2']}, {mat['E3']}, {mat['nu12']}, "
             f"{mat['nu13']}, {mat['nu23']}, {mat['G12']}, {mat['G13']}")
    L.append(f"{mat['G23']}")
    L.append("*DENSITY")
    L.append(f"{mat['rho']}")

    # One *ORIENTATION per unique ply angle. CalculiX composite shells require
    # the per-layer 4th field to name an orientation (not an inline angle).
    # We define local-x by rotating the global X axis by the ply angle about Z
    # (the shell normal). Line 1 gives a point on local-x and a point in the
    # local x-y plane; both lie in the global plate plane (z=0).
    def _ori_name(a: int) -> str:
        return f"OR_{'M' if a < 0 else 'P'}{abs(a)}"

    for ang in sorted(set(seq)):
        rad = np.deg2rad(ang)
        # point on local x-axis
        ax, ay = np.cos(rad), np.sin(rad)
        # point in local x-y plane (local y = +90deg from local x)
        bx, by = -np.sin(rad), np.cos(rad)
        L.append(f"*ORIENTATION, NAME={_ori_name(ang)}, SYSTEM=RECTANGULAR")
        L.append(f"{ax:.8f}, {ay:.8f}, 0.0, {bx:.8f}, {by:.8f}, 0.0")

    # composite shell section: one layer line per ply, referencing an orientation
    L.append("*SHELL SECTION, ELSET=PLATE, COMPOSITE, OFFSET=0")
    for ang in seq:
        L.append(f"{ply_t}, , LAMINA, {_ori_name(ang)}")
    return L


def _ss_boundary() -> list[str]:
    """Simply-supported on all 4 edges: UZ=0 (dof 3) on every edge node,
    plus minimal in-plane restraint to kill rigid-body modes."""
    L = ["*BOUNDARY"]
    # out-of-plane on all four edges
    for s in ("EDGE_X0", "EDGE_X1", "EDGE_Y0", "EDGE_Y1"):
        L.append(f"{s}, 3, 3")
    return L


def _freq_inplane_step(m: Mesh, a: float, n_freq: int = N_FREQ) -> list[str]:
    """In-plane rigid-body restraint + frequency step, as step-local *BOUNDARY so it
    can coexist with a buckle step in a combined deck."""
    n_origin = _corner_node(m, 0.0, 0.0)        # fix UX,UY
    n_xend = _corner_node(m, a, 0.0)            # fix UY
    return ["*STEP", "*FREQUENCY", f"{n_freq}",
            "*BOUNDARY", f"{n_origin}, 1, 2", f"{n_xend}, 2, 2",
            "*NODE FILE", "U", "*END STEP"]


def _buckle_inplane_step(m: Mesh, a: float, b: float, n_buck: int = N_BUCK) -> list[str]:
    """In-plane support + linear buckling perturbation step with compressive load,
    all step-local."""
    n_origin = _corner_node(m, 0.0, 0.0)
    n_yend = _corner_node(m, 0.0, b)
    L = ["*STEP, PERTURBATION", "*BUCKLE", f"{n_buck}",
         "*BOUNDARY", "EDGE_X0, 1, 1", f"{n_origin}, 2, 2", f"{n_yend}, 2, 2"]
    L += _cload_compression(m)
    L.append("*END STEP")
    return L


def write_freq_deck(m: Mesh, path: str, seq=LAYUP, mat=MAT, ply_t=PLY_T,
                    n_freq=N_FREQ, a=LX):
    L = _common_model(m, seq=seq, mat=mat, ply_t=ply_t)
    L += _ss_boundary()
    L += _freq_inplane_step(m, a, n_freq)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def write_buckle_deck(m: Mesh, path: str, seq=LAYUP, mat=MAT, ply_t=PLY_T,
                      n_buck=N_BUCK, a=LX, b=LY):
    L = _common_model(m, seq=seq, mat=mat, ply_t=ply_t)
    L += _ss_boundary()
    L += _buckle_inplane_step(m, a, b, n_buck)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def _cload_compression(m: Mesh) -> list[str]:
    """Distribute REF_LOAD (N) in -X over the x=LX edge nodes.

    For an 8-node quadratic edge, a uniform traction lumps unevenly between
    corner and mid-side nodes (1:4 ratio per quadratic shape functions).
    We use consistent-load weights so the resultant is exactly REF_LOAD:
      corner nodes get weight ~ -1/6 of a segment, mid-side ~ 4/6.
    Simpler and adequate for a buckling reference: weight each edge node by the
    tributary integral of its 1D quadratic shape function along the edge.
    """
    edge = sorted(set(m.edge_x1), key=lambda nid: m.nodes[nid][1])
    ys = np.array([m.nodes[nid][1] for nid in edge])
    # Quadratic line elements along the edge: nodes alternate corner/mid.
    # Build consistent nodal load weights from 1D quadratic shape integration.
    weights = {nid: 0.0 for nid in edge}
    # Each quadratic segment spans 3 consecutive edge nodes (corner,mid,corner).
    # Tributary line length per segment:
    n_seg = (len(edge) - 1) // 2
    for s in range(n_seg):
        a, b, c = edge[2 * s], edge[2 * s + 1], edge[2 * s + 2]
        Lseg = ys[2 * s + 2] - ys[2 * s]
        # consistent load fractions for uniform load on 3-node line: 1/6,4/6,1/6
        weights[a] += Lseg * (1.0 / 6.0)
        weights[b] += Lseg * (4.0 / 6.0)
        weights[c] += Lseg * (1.0 / 6.0)
    total_w = sum(weights.values())   # == edge length LY
    L = ["*CLOAD"]
    for nid in edge:
        f = -REF_LOAD * weights[nid] / total_w
        L.append(f"{nid}, 1, {f:.8f}")
    return L


def _corner_node(m: Mesh, x: float, y: float) -> int:
    tol = 1e-6
    for nid, (nx, ny, _nz) in m.nodes.items():
        if abs(nx - x) < tol and abs(ny - y) < tol:
            return nid
    raise RuntimeError(f"no node at corner ({x},{y})")


# ----------------------------------------------------------------------------
# Run + parse
# ----------------------------------------------------------------------------
def run_ccx(jobpath_noext: str) -> str:
    job = os.path.basename(jobpath_noext)
    cwd = os.path.dirname(jobpath_noext)
    # OMP_NUM_THREADS=1: era l'UNICO call-site del bundle senza questa guardia (7 su 8 ce
    # l'hanno). L'eigensolve threaded di ccx<2.21 restituisce fattori sbagliati -- lo documenta
    # fe/ccx_runner.py -- e su un host che esporta OMP_NUM_THREADS (tipico sui cluster) tutti i
    # numeri di exp9/exp14/exp18 e di evaluate_layup sarebbero stati plausibili, mesh-converged
    # e falsi, senza un segnale. Con ccx 2.21 resta il non-determinismo del threading.
    proc = subprocess.run([CCX, job], cwd=cwd, capture_output=True, text=True,
                          env={**os.environ, "OMP_NUM_THREADS": "1"})
    log = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        log += f"\n[run_ccx] ccx exit code {proc.returncode}"
    with open(jobpath_noext + ".log", "w") as f:
        f.write(log)
    return log


def parse_frequencies(dat_path: str) -> list[float]:
    """Parse the CalculiX eigenvalue/frequency table from the .dat file.

    Format:
      E I G E N V A L U E   O U T P U T
      MODE NO   EIGENVALUE   FREQUENCY(RAD/TIME) (CYCLES/TIME)  ...
       1   ...    ...           ...                <Hz>
    The 5th column is cycles/time (Hz).
    """
    with open(dat_path) as f:
        text = f.read()
    freqs = []
    in_block = False
    for line in text.splitlines():
        u = line.upper()
        # *FREQUENCY prints a column header "MODE NO   EIGENVALUE   FREQUENCY"
        # (no spaced-out "OUTPUT" banner like *BUCKLE), so trigger on that.
        if "MODE NO" in u and "EIGENVALUE" in u:
            in_block = True
            continue
        if in_block:
            parts = line.split()
            # data rows: int, then several floats
            if len(parts) >= 5:
                try:
                    int(parts[0])
                    vals = [float(p) for p in parts[1:]]
                except ValueError:
                    if freqs:  # block ended
                        break
                    continue
                # columns: eigenvalue, omega(real), omega(imag?), freq_rad, freq_hz
                # CalculiX prints: MODE EIGENVALUE FREQ(RAD/TIME) (CYCLES/TIME) ...
                # parts: [mode, eigenvalue, rad/time, cycles/time, ...]
                hz = vals[2]   # index2 of vals -> 3rd numeric after mode = cycles/time
                freqs.append(hz)
            elif freqs and line.strip() == "":
                continue
    return freqs


def parse_buckling_factors(dat_path: str) -> list[float]:
    with open(dat_path) as f:
        text = f.read()
    factors = []
    in_block = False
    for line in text.splitlines():
        u = line.upper()
        # *BUCKLE prints a spaced banner then a "MODE NO   BUCKLING" column
        # header; trigger on the column header.
        if "MODE NO" in u and "BUCKLING" in u:
            in_block = True
            continue
        if in_block:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    int(parts[0])
                    factors.append(float(parts[1]))
                except ValueError:
                    if factors:
                        break
                    continue
            elif factors and line.strip() == "":
                continue
    return factors


# ----------------------------------------------------------------------------
# Per-layup FE evaluator
# ----------------------------------------------------------------------------
def _seq_hash(seq) -> str:
    """Short stable hash of a stacking sequence, for per-layup subdir names."""
    s = ",".join(str(int(a)) for a in seq)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def _full_mat(mat: dict) -> dict:
    """Fill in the 3D orthotropic constants the FE deck needs from a reduced
    in-plane `mat` dict (E1,E2,nu12,G12,rho), using transverse-isotropy-style
    defaults consistent with the module MAT when absent."""
    out = dict(mat)
    out.setdefault("E3", out["E2"])
    out.setdefault("nu13", out["nu12"])
    out.setdefault("nu23", 0.4)
    out.setdefault("G13", out["G12"])
    out.setdefault("G23", 3500.0)
    return out


def evaluate_layup(seq, workdir, mat=MAT, a=LX, b=LY, ply_t=PLY_T,
                   n_freq=10, n_buck=5):
    """Build freq+buckle decks for the given full stacking sequence `seq`,
    run ccx, parse. Returns dict: {mode7_hz, pcr_n, freqs, buck, weight_g}.

    Reuses the existing mesh builder and deck/parse machinery, but parametrized
    by `seq` instead of the module-level LAYUP. Each evaluation runs in its own
    subdir under `workdir` (named by a short hash of `seq`) so parallel/repeated
    runs do not clobber files.
    """
    seq = list(seq)
    mat = _full_mat(mat)
    subdir = os.path.join(workdir, _seq_hash(seq))
    os.makedirs(subdir, exist_ok=True)

    # Mesh is geometry-only (independent of layup); plate is LX x LY.
    m = build_mesh(NEL_X, NEL_Y)

    freq_inp = os.path.join(subdir, "plate_freq.inp")
    buck_inp = os.path.join(subdir, "plate.inp")
    write_freq_deck(m, freq_inp, seq=seq, mat=mat, ply_t=ply_t, n_freq=n_freq)
    write_buckle_deck(m, buck_inp, seq=seq, mat=mat, ply_t=ply_t, n_buck=n_buck)

    run_ccx(freq_inp[:-4])
    freqs = parse_frequencies(freq_inp[:-4] + ".dat")
    run_ccx(buck_inp[:-4])
    buck = parse_buckling_factors(buck_inp[:-4] + ".dat")

    if len(freqs) < 7:
        raise RuntimeError(
            f"evaluate_layup: parsed only {len(freqs)} frequencies (need >=7) "
            f"for seq={seq}; see {freq_inp[:-4]}.log")
    if len(buck) < 1:
        raise RuntimeError(
            f"evaluate_layup: no buckling factors parsed for seq={seq}; "
            f"see {buck_inp[:-4]}.log")

    n = len(seq)
    weight_g = mat["rho"] * (a * b * n * ply_t) * 1.0e6  # tonne -> g
    return dict(
        mode7_hz=freqs[6],
        pcr_n=buck[0] * REF_LOAD,
        freqs=freqs,
        buck=buck,
        weight_g=weight_g,
    )


# ----------------------------------------------------------------------------
# STEP + PNG
# ----------------------------------------------------------------------------
def export_step(path: str):
    import gmsh
    gmsh.initialize()
    try:
        gmsh.model.add("canale_plate")
        gmsh.model.occ.addBox(0, 0, 0, LX, LY, THICK)
        gmsh.model.occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()


def render_png(m: Mesh, path: str, freqs, buck, weight_g):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    segs = []
    for nn in m.elements.values():
        # outline using the 4 corners + midsides in order
        order = [0, 4, 1, 5, 2, 6, 3, 7, 0]
        pts = [(m.nodes[nn[k]][0], m.nodes[nn[k]][1]) for k in order]
        for k in range(len(pts) - 1):
            segs.append([pts[k], pts[k + 1]])

    fig, ax = plt.subplots(figsize=(10, 6))
    lc = LineCollection(segs, colors="#3367d6", linewidths=0.4)
    ax.add_collection(lc)
    xs = [c[0] for c in m.nodes.values()]
    ys = [c[1] for c in m.nodes.values()]
    ax.scatter(xs, ys, s=1.5, c="#d63333", zorder=3)
    ax.set_xlim(-5, LX + 5)
    ax.set_ylim(-15, LY + 5)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)  — compression / buckling direction")
    ax.set_ylabel("Y (mm)")
    title = (f"Canale CFRP plate {LX:.0f}x{LY:.0f} mm, {N_PLIES} plies x {PLY_T} mm "
             f"= {THICK} mm\nlayup [45/-45/0/90/0]_2s  |  S8R mesh "
             f"{NEL_X}x{NEL_Y} ({len(m.elements)} el, {len(m.nodes)} nodes)")
    ax.set_title(title, fontsize=10)
    note = (f"mode-7 freq = {freqs[6]:.1f} Hz   "
            f"buckling factor_1 = {buck[0]:.3f} -> Pcr = {buck[0]*REF_LOAD:.0f} N   "
            f"weight = {weight_g:.2f} g")
    ax.text(0.5, -0.16, note, transform=ax.transAxes, ha="center", fontsize=9)
    # dimension arrows
    ax.annotate("", xy=(LX, -8), xytext=(0, -8),
                arrowprops=dict(arrowstyle="<->", color="black"))
    ax.text(LX / 2, -11, f"{LX:.0f} mm", ha="center", fontsize=8)
    ax.annotate("", xy=(-3, LY), xytext=(-3, 0),
                arrowprops=dict(arrowstyle="<->", color="black"))
    ax.text(-7, LY / 2, f"{LY:.0f} mm", va="center", rotation=90, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    m = build_mesh(NEL_X, NEL_Y)

    # FE evaluation of the 20-ply baseline via the per-layup evaluator.
    ev = evaluate_layup(LAYUP, OUTDIR, mat=MAT, a=LX, b=LY, ply_t=PLY_T,
                        n_freq=N_FREQ, n_buck=N_BUCK)
    freqs = ev["freqs"]
    buck = ev["buck"]
    buck_inp = os.path.join(OUTDIR, _seq_hash(LAYUP), "plate.inp")
    freq_inp = os.path.join(OUTDIR, _seq_hash(LAYUP), "plate_freq.inp")

    # weight
    volume = LX * LY * THICK                # mm^3
    mass_tonne = MAT["rho"] * volume        # tonne
    mass_g = ev["weight_g"]

    # CAD + viz
    step_path = os.path.join(OUTDIR, "plate.step")
    png_path = os.path.join(OUTDIR, "plate.png")
    try:
        export_step(step_path)
    except Exception as e:  # noqa
        print(f"WARN: STEP export failed: {e}")
        step_path = "(STEP export failed)"
    render_png(m, png_path, freqs, buck, mass_g)

    # ---- report ----
    mode7 = freqs[6]
    pcr = buck[0] * REF_LOAD
    print("=" * 72)
    print("CANALE COMPOSITE PLATE — frequency + buckling baseline (CalculiX 2.21)")
    print("=" * 72)
    print(f"Geometry      : {LX:.0f} x {LY:.0f} mm flat plate, "
          f"{N_PLIES} plies x {PLY_T} mm = {THICK} mm laminate")
    print(f"Layup         : [45/-45/0/90/0]_2s  ->")
    print(f"                {LAYUP}")
    print(f"Material      : T300/epoxy orthotropic, E1={MAT['E1']:.0f} E2={MAT['E2']:.0f} "
          f"MPa, rho={MAT['rho']:.2e} tonne/mm^3")
    print(f"Units         : mm, N, MPa, tonne, s  (freq in Hz)")
    print(f"Mesh          : S8R {NEL_X}x{NEL_Y} = {len(m.elements)} elements, "
          f"{len(m.nodes)} nodes")
    print(f"BCs           : simply-supported all 4 edges (UZ=0) + minimal in-plane RBM restraint")
    print(f"Buckling load : uniaxial -X compression, ref total {REF_LOAD:.0f} N on x={LX:.0f} edge")
    print("-" * 72)
    print(f"All {len(freqs)} natural frequencies (Hz):")
    for i, fr in enumerate(freqs, 1):
        print(f"   mode {i:2d}: {fr:12.3f} Hz")
    print("-" * 72)
    print(f"All {len(buck)} buckling factors:")
    for i, b in enumerate(buck, 1):
        print(f"   {i}: factor={b:12.5f}  ->  Pcr = {b*REF_LOAD:12.2f} N")
    print("-" * 72)
    print(f"MODE-7 FREQUENCY : {mode7:.3f} Hz   "
          f"[{'PASS' if mode7 > 60 else 'FAIL'} vs > 60 Hz]")
    print(f"BUCKLING factor_1: {buck[0]:.5f}   "
          f"=> Pcr = {pcr:.2f} N   [{'PASS' if pcr > 5000 else 'FAIL'} vs > 5000 N]")
    print(f"WEIGHT           : volume={volume:.0f} mm^3, mass={mass_tonne:.3e} tonne "
          f"= {mass_g:.3f} g")
    print("-" * 72)
    print(f"Deck (buckle) : {buck_inp}")
    print(f"Deck (freq)   : {freq_inp}")
    print(f"STEP          : {step_path}")
    print(f"PNG           : {png_path}")
    print("=" * 72)

    return dict(freqs=freqs, buck=buck, mode7=mode7, pcr=pcr, mass_g=mass_g,
                inp=buck_inp, freq_inp=freq_inp, step=step_path, png=png_path)


if __name__ == "__main__":
    main()
