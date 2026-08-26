"""Gold-standard interlaminar (delamination/peel) assessment for the RR composite panel.

The naive point-max SZZ from a shell expansion is junk (free-edge interlaminar stresses are
SINGULAR — Pipes-Pagano). The rigorous route:

  1. Solve a 3D SOLID (one C3D8 element per ply) with a DISTRIBUTED tip traction (no lumped-force
     artifact) — gives a clean in-plane stress field sigma_xx, sigma_yy, sigma_xy on a structured
     (i,j,k) grid.
  2. EQUILIBRIUM-BASED stress recovery: integrate the 3D equilibrium equations through the
     thickness from the traction-free surface to recover the interlaminar stresses
       tau_xz(z) = -∫ (∂σxx/∂x + ∂σxy/∂y) dz
       tau_yz(z) = -∫ (∂σxy/∂x + ∂σyy/∂y) dz
       sigma_zz(z) = -∫ (∂τxz/∂x + ∂τyz/∂y) dz       (peel)
     (the standard recovery; FE in-plane stresses are accurate, the raw FE interlaminar are not).
  3. A FRACTURE-INFORMED delamination criterion: a quadratic interlaminar index at each ply
     interface, AVERAGED over a characteristic distance d0 from the free edge (Whitney-Nuismer
     averaged-stress, the practical regularisation of the singular field):
       Q = (<σzz>+/Zt)^2 + (<τxz>/S13)^2 + (<τyz>/S23)^2   ;  delamination onset if Q >= 1.

This module builds the solid, runs ccx, recovers the interlaminar stresses, and returns Q + the
recovered peak interlaminar stresses. Reuses the orthotropic `layer` material from the RR decks.
"""
from __future__ import annotations
import os, re, subprocess, tempfile, shutil

from fe.ccx_bin import CCX          # un solo default per il binario ccx (audit F12)
LX, LY = 100.0, 50.0
PLY_T = 0.1
# Orthotropic ply: SINGLE SOURCE OF TRUTH nel registro, non ribattuta qui. Questo modulo era
# la terza copia indipendente delle costanti (la prima e' il registro, la seconda era in
# optimisers/constrained_search.py, gia' agganciata al registro il 26/08): un ritocco a una
# delle copie faceva divergere IN SILENZIO il materiale del solido da quello del guscio, cioe'
# Q, peel e i pannelli B/C dalla campagna. L'assert sotto pinna i byte del deck spediti finora.
from fe.materials import get_material
_LAMINA = get_material("canale2018")
E1, E2, NU12, G12 = _LAMINA["E1"], _LAMINA["E2"], _LAMINA["nu12"], _LAMINA["G12"]
ZT, S13, S23 = 50.0, 90.0, 90.0   # peel (transverse tensile) + interlaminar shear strengths
D0 = 2.0                          # averaging characteristic distance from the free edge (mm)


def _orient_block(angles):
    out = []
    for a in sorted(set(angles)):
        nm = f"O{str(a).replace('-', 'm')}"
        out += [f"*ORIENTATION, NAME={nm}", "1.,0.,0.,0.,1.,0.", f"3, {a}."]
    return out


def make_solid_deck(seq, axial, side, nx=20, ny=10, ply_t=None):
    """Parametric C3D8 solid: nx x ny in-plane, len(seq) plies through z, each an oriented
    *SOLID SECTION layer; clamp x=0, DISTRIBUTED nodal traction on the x=Lx tip (no point load).

    `ply_t` overrides the module's PLY_T for this deck only. It exists so that a layer can be
    SPLIT into several elements through the thickness at CONSTANT total thickness (pass the same
    angle k times with ply_t=PLY_T/k): that is the only way to tell element locking, which relaxes
    under through-thickness refinement, from a spurious eigenvalue, which does not.
    """
    nz = len(seq)
    ply_t = PLY_T if ply_t is None else ply_t
    nid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k + 1
    L = ["*HEADING", "rr solid interlaminar", "*NODE"]
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                L.append(f"{nid(i,j,k)}, {i*LX/nx:.5f}, {j*LY/ny:.5f}, {k*ply_t:.5f}")
    # one element set per ply layer k
    L.append("*ELEMENT, TYPE=C3D8I, ELSET=EALL")
    eid = 1
    eoflayer = {k: [] for k in range(nz)}
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = [nid(i, j, k), nid(i+1, j, k), nid(i+1, j+1, k), nid(i, j+1, k),
                     nid(i, j, k+1), nid(i+1, j, k+1), nid(i+1, j+1, k+1), nid(i, j+1, k+1)]
                L.append(f"{eid}, " + ", ".join(map(str, c)))
                eoflayer[k].append(eid); eid += 1
    onm = {a: f"O{str(a).replace('-', 'm')}" for a in set(seq)}
    for k in range(nz):
        es = eoflayer[k]
        L.append(f"*ELSET, ELSET=LY{k}")
        L += [",".join(map(str, es[t:t+16])) + "," for t in range(0, len(es), 16)]
    L += _orient_block(seq)
    for k in range(nz):
        L.append(f"*SOLID SECTION, ELSET=LY{k}, ORIENTATION={onm[seq[k]]}, MATERIAL=layer")
        L.append("1.,")
    _mat = ["*MATERIAL, NAME=layer", "*ELASTIC, TYPE=ENGINEERING CONSTANTS",
            f"{E1},{E2},{E2},{NU12},{_LAMINA['nu13']},{_LAMINA['nu23']},{G12},"
            f"{_LAMINA['G13']:.0f}.", f"{_LAMINA['G23']:.0f}.,"]
    assert _mat == ["*MATERIAL, NAME=layer", "*ELASTIC, TYPE=ENGINEERING CONSTANTS",
                    "125100.0,7840.0,7840.0,0.3,0.15,0.15,4600.0,4000.", "4000.,"], _mat
    L += _mat
    # clamp x=0 face
    clamp = sorted({nid(0, j, k) for j in range(ny + 1) for k in range(nz + 1)})
    L.append("*NSET, NSET=CLAMP")
    L += [",".join(map(str, clamp[t:t+16])) for t in range(0, len(clamp), 16)]
    # distributed tip load: axial (X) + side (Y) spread over x=Lx face nodes
    tip = sorted({nid(nx, j, k) for j in range(ny + 1) for k in range(nz + 1)})
    L += ["*BOUNDARY", "CLAMP, 1, 3", "*STEP", "*STATIC", "*CLOAD"]
    for n in tip:
        L.append(f"{n}, 1, {axial/len(tip):.6f}")
        L.append(f"{n}, 2, {side/len(tip):.6f}")
    L += ["*NODE FILE", "U", "*EL FILE", "S", "*END STEP"]
    return "\n".join(L) + "\n", nx, ny, nz


def make_solid_buckle_deck(seq, axial, side, nx=20, ny=10, nbuck=2, ply_t=None):
    """Same C3D8I solid as `make_solid_deck`, but a *BUCKLE step instead of *STATIC.

    The module's own deck is STATIC (it exists to recover interlaminar stresses); panel (B)
    of the pitfalls figure is about BUCKLING on the same solid, so the two cannot be compared
    without this variant. Geometry, mesh, plies, material, clamp and tip traction are reused
    verbatim from `make_solid_deck` so that solid-vs-shell is a comparison of ELEMENT TYPE and
    nothing else. Buckling factors go to the .dat file, so the .frd output cards are dropped.
    """
    deck, nx, ny, nz = make_solid_deck(seq, axial, side, nx, ny, ply_t)
    deck = deck.replace("*STEP\n*STATIC\n", f"*STEP\n*BUCKLE\n{nbuck}\n")
    deck = deck.replace("*NODE FILE\nU\n*EL FILE\nS\n", "")
    return deck, nx, ny, nz


def solid_buckling_factor(seq, axial, side, nx=20, ny=10, nbuck=2, ply_t=None):
    """First buckling factor of the C3D8I solid. Returns None if ccx produced no eigenvalue.

    Counterpart of `optimisers.constrained_search.buckling_factor` (the validated S8R shell):
    same panel, same loads, different element type.
    """
    deck, nx, ny, nz = make_solid_buckle_deck(seq, axial, side, nx, ny, nbuck, ply_t)
    d = tempfile.mkdtemp()
    try:
        open(d + "/job.inp", "w").write(deck)
        subprocess.run([CCX, "-i", "job"], cwd=d, capture_output=True, text=True, timeout=3600,
                       env={**os.environ, "OMP_NUM_THREADS": "1"})  # ccx<2.21 threaded buckling is wrong
        dat = open(d + "/job.dat").read() if os.path.exists(d + "/job.dat") else ""
        f = re.findall(r"^\s*1\s+([\d.E+\-]+)\s*$", dat, re.M)
        return float(f[0]) if f else None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _parse_stress_grid(frd, nx, ny, nz):
    """Return sig[i][j][k] = (sxx,syy,szz,sxy,syz,szx) on the (nx+1,ny+1,nz+1) node grid.

    ⚠️ Questa funzione estraeva il numero del nodo con `int(line.split()[1])`, ed era SBAGLIATO
    in un modo che non si annunciava. Il .frd di CalculiX e' a colonne fisse (1X,A2,I10,6E12.5):
    un primo valore NEGATIVO riempie tutti i 12 caratteri del campo e si incolla al campo del
    nodo, cosi' che `split()[1]` restituisce "2556-1.26360E-01", `int()` alza ValueError e la
    riga veniva scartata con un `continue`. La cella restava lo 0.0 dell'inizializzazione, cioe'
    uno sforzo perfettamente legittimo, che il recupero per equilibrio a valle consuma senza
    accorgersene. In un blocco STRESS in compressione il primo valore e' sigma_xx: si perdevano
    esattamente i nodi che contano. Misurato su un .frd reale del bundle: 0 nodi estratti su
    50.000 dalle righe con primo valore negativo, contro 50.000 su 50.000 con lo slicing.
    `fe/frd_parse._read_block` documentava gia' questo trap e lo evitava: questa era una copia
    indipendente che non lo faceva. Ora la copia non esiste piu', si usa quella.
    """
    from fe.frd_parse import _read_block, _skip_minus5
    nid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k + 1
    inv = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                inv[nid(i, j, k)] = (i, j, k)
    import numpy as np
    sig = np.zeros((nx + 1, ny + 1, nz + 1, 6))
    lines = frd.splitlines()
    rows, i = [], 0
    while i < len(lines):
        u = lines[i].upper()
        if u.lstrip().startswith("-4") and "STRESS" in u:
            rows, _ = _read_block(lines, i + 1 + _skip_minus5(lines, i + 1))
            break
        i += 1
    seen = 0
    for node, v in rows:
        if len(v) >= 6 and node in inv:
            a, b, c = inv[node]
            sig[a, b, c] = v[:6]
            seen += 1
    # Copertura: se il blocco non copre la griglia, le celle mancanti resterebbero 0.0 e
    # sarebbero indistinguibili da uno sforzo nullo misurato. Meglio fermarsi.
    expected = len(inv)
    if seen != expected:
        raise RuntimeError(
            "interlaminar: blocco STRESS incompleto, %d nodi della griglia su %d "
            "(righe lette: %d). Un campo parzialmente azzerato produce numeri plausibili e "
            "falsi: la run si ferma invece di consegnarli." % (seen, expected, len(rows)))
    return sig


def interlaminar(seq, axial, side, nx=20, ny=10):
    """Run the solid, recover interlaminar stresses by equilibrium, return the fracture-informed
    delamination index Q and the recovered peak interlaminar stresses."""
    import numpy as np
    deck, nx, ny, nz = make_solid_deck(seq, axial, side, nx, ny)
    d = tempfile.mkdtemp()
    try:
        open(d + "/job.inp", "w").write(deck)
        r = subprocess.run([CCX, "-i", "job"], cwd=d, capture_output=True, text=True, timeout=900,
                           env={**os.environ, "OMP_NUM_THREADS": "1"})  # ccx<2.21 threaded solve unsafe
        frd = open(d + "/job.frd").read() if os.path.exists(d + "/job.frd") else ""
        # Il returncode e lo stderr non erano controllati: un .frd PARZIALE, non vuoto e senza
        # "ERROR" in stdout, passava il gate e finiva nel parser, dove i nodi mancanti valevano
        # zero. Ora un guasto e' un guasto, e porta con se' abbastanza per capirlo.
        if r.returncode != 0 or "ERROR" in r.stdout or not frd:
            return {"error": "ccx", "returncode": r.returncode,
                    "log": r.stdout[-400:], "stderr": (r.stderr or "")[-400:]}
        from fe.frd_parse import parse_frd_static
        disp = max((x["umag"] for x in parse_frd_static(frd)["disp"]), default=0.0)
        sig = _parse_stress_grid(frd, nx, ny, nz)             # (i,j,k,6): sxx,syy,szz,sxy,syz,szx
        dx, dy, dz = LX / nx, LY / ny, PLY_T

        def smooth(a):  # 3x3 in-plane box smoothing per z-layer (de-noise FE stresses before FD)
            b = a.copy()
            b[1:-1, :, :] = (a[:-2, :, :] + a[1:-1, :, :] + a[2:, :, :]) / 3
            c = b.copy()
            c[:, 1:-1, :] = (b[:, :-2, :] + b[:, 1:-1, :] + b[:, 2:, :]) / 3
            return c
        sxx, syy, sxy = smooth(sig[..., 0]), smooth(sig[..., 1]), smooth(sig[..., 3])
        # in-plane gradients (central differences over the in-plane grid)
        dsxx_dx = np.gradient(sxx, dx, axis=0)
        dsxy_dy = np.gradient(sxy, dy, axis=1)
        dsxy_dx = np.gradient(sxy, dx, axis=0)
        dsyy_dy = np.gradient(syy, dy, axis=1)
        # EQUILIBRIUM recovery: integrate through z from the bottom free surface (k=0, tau=0)
        txz = np.zeros_like(sxx); tyz = np.zeros_like(sxx)
        for k in range(1, nz + 1):
            txz[:, :, k] = txz[:, :, k-1] - 0.5*(dsxx_dx[:, :, k]+dsxy_dy[:, :, k]
                                                 + dsxx_dx[:, :, k-1]+dsxy_dy[:, :, k-1]) * dz
            tyz[:, :, k] = tyz[:, :, k-1] - 0.5*(dsxy_dx[:, :, k]+dsyy_dy[:, :, k]
                                                 + dsxy_dx[:, :, k-1]+dsyy_dy[:, :, k-1]) * dz
        dtxz_dx = np.gradient(txz, dx, axis=0)
        dtyz_dy = np.gradient(tyz, dy, axis=1)
        szz = np.zeros_like(sxx)
        for k in range(1, nz + 1):
            szz[:, :, k] = szz[:, :, k-1] - 0.5*(dtxz_dx[:, :, k]+dtyz_dy[:, :, k]
                                                 + dtxz_dx[:, :, k-1]+dtyz_dy[:, :, k-1]) * dz
        # AVERAGED (fracture-informed) criterion: average over a band d0 from the free edges
        # (y=0 and y=LY are the free edges; the clamped x=0 and loaded x=LX are not free).
        jband = max(1, int(D0 / dy))
        edge = np.r_[0:jband + 1, ny - jband:ny + 1]
        # AVERAGE over the characteristic edge band (mesh-CONVERGENT, unlike the singular point
        # max), keeping the through-thickness profile; then take the worst ply interface.
        szz_avg = np.maximum(szz[1:nx, edge, :].mean(axis=(0, 1)), 0.0)   # per-interface, >=0
        txz_avg = np.abs(txz[1:nx, edge, :]).mean(axis=(0, 1))
        tyz_avg = np.abs(tyz[1:nx, edge, :]).mean(axis=(0, 1))
        peel = float(szz_avg.max())
        ilss = float(np.sqrt(txz_avg**2 + tyz_avg**2).max())
        Q = float(((szz_avg/ZT)**2 + (txz_avg/S13)**2 + (tyz_avg/S23)**2).max())
        # check free-surface BC was respected (recovered tau ~ 0 at k=0 and k=nz)
        bc = float(max(np.max(np.abs(txz[:, :, 0])), np.max(np.abs(txz[:, :, nz]))))
        # The NAIVE peel: the point maximum of the recovered sigma_zz on the free edge, i.e. what
        # you get if you skip the Whitney-Nuismer averaging. It is the singular quantity: it does
        # NOT converge under mesh refinement (Pipes-Pagano), and that is the whole point of
        # panel (C) of the pitfalls figure. Returned alongside the averaged value so the two can
        # be swept together; nothing else in the module depends on it.
        peel_point = float(np.maximum(szz[:, edge, :], 0.0).max())
        return {"disp": disp, "peel": peel, "ilss": ilss, "Q": Q, "delam": Q >= 1.0,
                "bc_resid": bc, "nz": nz, "peel_point": peel_point,
                "mesh": {"nx": nx, "ny": ny, "nz": nz}}
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    import sys, json
    seq = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 45, -45, 90] * 6
    # NB: questi sono i carichi del CASO DEMO STORICO (il vecchio fattore 0.44), non quelli
    # della campagna: exp10/exp10b li usano deliberatamente e lo documentano, questo __main__ e'
    # solo una prova manuale del modulo. Il default divergente di LOAD_SCALE fra i moduli e'
    # stato eliminato il 26/08 (DEFAULT_LOAD_SCALE in optimisers/constrained_search.py); qui il
    # numero resta scritto in chiaro apposta, per non far credere che segua quella costante.
    print(json.dumps(interlaminar(seq, axial=-20000*0.44, side=5000*0.44), indent=2))
