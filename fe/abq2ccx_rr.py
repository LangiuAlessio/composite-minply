# Ported into this bundle 2026-07-20 (RS-005) from ingegneria/fe-batch-lab/cases/abq2ccx_rr.py,
# the FE dev lab from which code/ was extracted. Unchanged. It is the generator behind panel (A)
# of the pitfalls figure (experiments/exp15_panelA_weakchop.py). stdlib-only (re).
"""Translate the industrial composite panel Abaqus decks (Composite_*.inp) into a CalculiX (ccx) runnable deck.

These decks are full Abaqus: a single Part/Instance/Assembly (identity transform),
a C3D8I solid with 60 plies modelled as 60 oriented *Solid Section element layers,
loads applied to a single assembly reference node coupled to the loaded edge via a
node *Surface + *MPC BEAM, and a *Buckle / *Frequency / *Static step with
Abaqus-only options. ccx parses ~all of it but stops fatally on `*Surface, type=NODE`
and a few Abaqus-only options.

Translation (lossless for the physics, cantilever + reference-node load preserved):
- drop the Part/Instance/Assembly wrappers (identity instance -> node/element ids unchanged);
- renumber the assembly reference node (it collides with part node 1) to RP_NEW and
  update the _PickedSet14..17 nsets that point to it;
- replace `*Surface, type=NODE` + `*MPC BEAM, edge, RP` with `*RIGID BODY, NSET=edge,
  REF NODE=RP_NEW` (ccx rigid body: the ref node carries 6 DOF, so the torsion moment
  on DOF 4-6 still applies);
- strip Abaqus-only nset/elset params (`internal`, `instance=`) and output keywords
  (*Restart, *Output, *Preprint);
- regenerate the step as a ccx *BUCKLE / *FREQUENCY / *STATIC, keeping the original
  *Cload (on the reference node) and clamping the cantilever edge _PickedSet9 in 1-3.

This is a proof-of-concept translator for THIS deck family, not a general converter.
"""
from __future__ import annotations
import re

RP_NEW = 900001  # reference-node id, safely above the part node range


def _kw(line: str):
    s = line.strip()
    if not s.startswith("*") or s.startswith("**"):
        return None
    return s[1:].split(",")[0].strip().lower()


def _strip_params(line: str, drop=("internal",)):
    """Remove Abaqus-only parameters (internal, instance=...) from a keyword line."""
    parts = [p.strip() for p in line.rstrip("\r\n").split(",")]
    kept = [parts[0]]
    for p in parts[1:]:
        key = p.split("=")[0].strip().lower()
        if key in drop or key == "instance":
            continue
        kept.append(p)
    return ", ".join(kept)


def _parse_nodes(text: str) -> dict:
    """Parse the part *Node block into {id: (x,y,z)} (first *Node block only)."""
    lines = text.splitlines()
    nodes = {}
    inblk = False
    for ln in lines:
        kw = _kw(ln)
        if kw == "node":
            inblk = True
            continue
        if inblk and kw is not None:
            break  # stop at the first keyword after the part *Node block
        if inblk:
            f = [c.strip() for c in ln.split(",") if c.strip()]
            if len(f) >= 4:
                try:
                    nodes[int(f[0])] = (float(f[1]), float(f[2]), float(f[3]))
                except ValueError:
                    pass
    return nodes


def _parse_nset(text: str, name: str) -> list:
    """Parse the node ids of an assembly-level *Nset (explicit list, not generate)."""
    lines = text.splitlines()
    ids, inblk = [], False
    for ln in lines:
        s = ln.strip()
        if re.match(r"\*Nset,\s*nset=" + re.escape(name) + r"\b", s, re.I):
            inblk = True
            continue
        if inblk and s.startswith("*"):
            break
        if inblk:
            for tok in s.split(","):
                tok = tok.strip()
                if tok.isdigit():
                    ids.append(int(tok))
    return ids


def build_edge_cloads(coords: dict, edge: list, axial: float, side: float,
                      torsion: float) -> list:
    """Equivalent nodal *CLOAD on the loaded edge: axial (X) + side (Y) spread evenly,
    and torsion about the X-axis as a tangential force-couple about the edge centroid
    (Fy=-c*dz, Fz=c*dy, c=T/Sum(r^2)) -> pure torque, ~zero net force. Robust: real
    nodal forces ccx transmits, no rigid-body moment dependence."""
    pts = [(nid, coords[nid]) for nid in edge if nid in coords]
    n = len(pts)
    yc = sum(p[1][1] for p in pts) / n
    zc = sum(p[1][2] for p in pts) / n
    sumr2 = sum((p[1][1] - yc) ** 2 + (p[1][2] - zc) ** 2 for p in pts) or 1.0
    c = torsion / sumr2
    out = ["*CLOAD"]
    for nid, (x, y, z) in pts:
        dy, dz = y - yc, z - zc
        fx = axial / n
        fy = side / n - c * dz
        fz = c * dy
        if fx:
            out.append(f"{nid}, 1, {fx:.8g}")
        if fy:
            out.append(f"{nid}, 2, {fy:.8g}")
        if fz:
            out.append(f"{nid}, 3, {fz:.8g}")
    return out


def _extract_rp_loads(cloads: list) -> tuple:
    """From the captured RP *Cload blocks, sum DOF1 (axial), DOF2 (side), DOF4 (torsion)."""
    axial = side = torsion = 0.0
    for blk in cloads:
        for ln in blk.splitlines():
            f = [c.strip() for c in ln.split(",")]
            if len(f) >= 3 and f[0].lower().startswith("_picked"):
                try:
                    dof, val = int(f[1]), float(f[2])
                except ValueError:
                    continue
                if dof == 1:
                    axial += val
                elif dof == 2:
                    side += val
                elif dof == 4:
                    torsion += val
    return axial, side, torsion


def translate(text: str, step: str, edge_loads: bool = True) -> str:
    """step in {'buckle','frequency','static'}. Returns a ccx deck string.

    edge_loads=True (default): apply the loads as direct nodal forces on the loaded
    edge (_PickedSet13), converting the reference-node torsion MOMENT into a tangential
    force-couple — because ccx does NOT transmit a moment applied to a *RIGID BODY
    reference node (verified: torsion-only -> buckling factor ~1e9). The rigid body and
    the RP *Cloads are then dropped."""
    lines = text.splitlines()
    out = []
    cloads = []            # original *Cload blocks (RP loads) to re-emit in the step
    past_instance = False  # after *End Instance -> assembly-level *Node = reference node
    i, n = 0, len(lines)
    # keywords whose whole block (kw line + data until next kw) we DROP
    DROP_BLOCK = {"surface", "mpc", "output", "restart", "preprint", "step", "buckle",
                  "frequency", "static", "boundary"}
    # single lines we DROP
    DROP_LINE = {"part", "end part", "assembly", "end assembly", "instance",
                 "end instance", "end step", "heading"}

    while i < n:
        line = lines[i]
        kw = _kw(line)
        if kw == "end instance":
            past_instance = True
            i += 1
            continue
        if kw in DROP_LINE:
            i += 1
            continue
        if kw == "node" and past_instance:
            # assembly reference node block: renumber id -> RP_NEW
            out.append("*NODE")
            i += 1
            while i < n and _kw(lines[i]) is None:
                row = lines[i].strip()
                if row:
                    f = [c.strip() for c in row.split(",")]
                    f[0] = str(RP_NEW)
                    out.append(", ".join(f))
                i += 1
            continue
        if kw == "cload":
            # capture for re-emission inside the regenerated step
            blk = [line]
            i += 1
            while i < n and _kw(lines[i]) is None:
                blk.append(lines[i])
                i += 1
            cloads.append("\n".join(blk))
            continue
        if kw in ("surface", "mpc", "output", "restart", "preprint", "boundary"):
            i += 1
            while i < n and _kw(lines[i]) is None:
                i += 1
            continue
        if kw in ("step",):
            # skip the original step, but CAPTURE its *Cload blocks (the reference-node
            # loads live inside the step) so they can be re-emitted in the new step.
            i += 1
            while i < n and _kw(lines[i]) != "end step":
                if _kw(lines[i]) == "cload":
                    blk = [lines[i]]
                    i += 1
                    while i < n and _kw(lines[i]) is None:
                        blk.append(lines[i])
                        i += 1
                    cloads.append("\n".join(blk))
                else:
                    i += 1
            i += 1
            continue
        if kw in ("nset", "elset"):
            hdr = _strip_params(line)
            out.append(hdr)
            i += 1
            # rewrite RP-pointing nsets (content "1,") to RP_NEW
            name = ""
            m = re.search(r"nset=([^,]+)", hdr, re.I) or re.search(r"elset=([^,]+)", hdr, re.I)
            if m:
                name = m.group(1).strip()
            while i < n and _kw(lines[i]) is None:
                row = lines[i].rstrip("\r\n")
                if past_instance and name.startswith("_PickedSet1") and row.strip() == "1,":
                    out.append(f"{RP_NEW},")
                else:
                    out.append(row)
                i += 1
            continue
        # default: keep the line as-is (nodes, elements, sections, orientations, materials)
        out.append(line.rstrip("\r\n"))
        i += 1

    if not edge_loads:
        # legacy path: rigid-body coupling + RP loads (torsion moment NOT transmitted)
        out.append("*RIGID BODY, NSET=_PickedSet13, REF NODE=%d" % RP_NEW)

    # regenerate the step
    out.append("*STEP")
    if step == "buckle":
        out.append("*BUCKLE")
        out.append("4")               # extract 4 buckling factors
    elif step == "frequency":
        out.append("*FREQUENCY")
        out.append("8")               # first 8 natural frequencies
    else:
        out.append("*STATIC")
    out.append("*BOUNDARY")
    out.append("_PickedSet9, 1, 3")   # cantilever clamp (solid: DOF 1-3 = fully fixed)
    if edge_loads:
        axial, side, torsion = _extract_rp_loads(cloads)
        coords = _parse_nodes(text)
        edge = _parse_nset(text, "_PickedSet13")
        out.extend(build_edge_cloads(coords, edge, axial, side, torsion))
    else:
        for blk in cloads:
            out.append(blk)
    if step == "static":
        out.append("*NODE FILE\nU")
        out.append("*EL FILE\nS")
    out.append("*END STEP")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    import sys
    src, stp = sys.argv[1], sys.argv[2]
    sys.stdout.write(translate(open(src).read(), stp))


# ---- laminate override (the optimiser's lever) -------------------------------
def set_laminate(text: str, sequence, n_layers: int = 60) -> str:
    """Override the deck to the given stacking `sequence` (list of ply angles).

    Active plies = the first len(sequence) physical layers (Set/Ori 1..k), each set to
    material=layer with its angle; the remaining layers get material=layer_weak (the
    'dummy material' chop). Returns the modified Abaqus text (still to be translate()-d)."""
    import re as _re
    lines = text.splitlines()
    k = len(sequence)
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        mo = _re.match(r"\*Orientation,\s*name=Ori-(\d+)\s*$", ln.strip(), _re.I)
        ms = _re.match(r"\*Solid Section,\s*elset=Set-(\d+),", ln.strip(), _re.I)
        if mo:
            idx = int(mo.group(1))
            out.append(lines[i]); out.append(lines[i + 1])  # coord-system line unchanged
            ang = sequence[idx - 1] if idx <= k else 0.0
            out.append(f"3, {float(ang)}")                  # rewrite the rotation angle
            i += 3
            continue
        if ms:
            idx = int(ms.group(1))
            mat = "layer" if idx <= k else "layer_weak"
            out.append(_re.sub(r"material=\w+", f"material={mat}", lines[i]))
            i += 1
            continue
        out.append(lines[i]); i += 1
    return "\n".join(out) + "\n"


# ---- crop to a <1000-node sub-model (for ccx-vs-Abaqus LE calibration) --------
def crop_layers(text: str, k: int) -> str:
    """Keep only the bottom `k` plies (z <= k*0.1) of the panel -> a sub-model with
    <1000 nodes that Abaqus Learning Edition (1000-node cap) can ALSO run, enabling a
    direct ccx-vs-Abaqus comparison on the SAME geometry. In-plane mesh (length/width,
    clamp & loaded edge) is unchanged. The reference node is moved to the cropped
    mid-thickness so the load apparatus stays sensible."""
    zmax = k * 0.1 + 1e-6
    coords = _parse_nodes(text)
    keepn = {nid for nid, (x, y, z) in coords.items() if z <= zmax}
    # elements whose every node is kept
    lines = text.splitlines()
    keepe = set()
    i, inblk = 0, False
    while i < len(lines):
        kw = _kw(lines[i])
        if kw == "element":
            inblk = True; i += 1; continue
        if inblk and kw is not None:
            inblk = False
        if inblk:
            f = [c.strip() for c in lines[i].split(",") if c.strip()]
            if len(f) >= 2 and f[0].isdigit():
                eid = int(f[0]); ns = [int(x) for x in f[1:] if x.isdigit()]
                if ns and all(n in keepn for n in ns):
                    keepe.add(eid)
        i += 1

    # which Set-N layers survive (their generate-range intersects keepe), + the cropped range
    surv = {}
    for idx in range(len(lines)):
        m = re.match(r"\*Elset,\s*elset=Set-(\d+),\s*generate", lines[idx].strip(), re.I)
        if m:
            f = [int(float(x)) for x in lines[idx + 1].split(",") if x.strip()]
            rng = [e for e in range(f[0], f[1] + 1, f[2] if len(f) > 2 else 1) if e in keepe]
            if rng:
                surv[int(m.group(1))] = (min(rng), max(rng))

    out = []
    past_instance = False
    i = 0
    while i < len(lines):
        ln = lines[i]; kw = _kw(ln)
        if kw == "end instance":
            past_instance = True; out.append(ln); i += 1; continue
        if kw == "node":
            out.append(ln); i += 1
            while i < len(lines) and _kw(lines[i]) is None:
                f = [c.strip() for c in lines[i].split(",") if c.strip()]
                if len(f) >= 4 and int(f[0]) in (keepn if not past_instance else {int(f[0])}):
                    if past_instance:  # assembly RP: move to cropped mid-thickness
                        f[3] = f"{zmax/2:.4f}"
                        out.append(", ".join(f))
                    else:
                        out.append(lines[i].rstrip())
                i += 1
            continue
        if kw == "element":
            out.append(ln); i += 1
            while i < len(lines) and _kw(lines[i]) is None:
                f = [c.strip() for c in lines[i].split(",") if c.strip()]
                if len(f) >= 2 and int(f[0]) in keepe:
                    out.append(lines[i].rstrip())
                i += 1
            continue
        ms = re.match(r"\*Elset,\s*elset=Set-(\d+),\s*generate", ln.strip(), re.I)
        if ms:
            N = int(ms.group(1))
            if N in surv:
                out.append(ln); out.append(f" {surv[N][0]},  {surv[N][1]},      1")
            i += 2; continue
        mo = re.match(r"\*Orientation,\s*name=Ori-(\d+)", ln.strip(), re.I)
        if mo:
            N = int(mo.group(1))
            if N in surv:
                out.append(ln); out.append(lines[i + 1].rstrip()); out.append(lines[i + 2].rstrip())
            i += 3; continue
        mss = re.match(r"\*Solid Section,\s*elset=Set-(\d+)", ln.strip(), re.I)
        if mss:
            N = int(mss.group(1))
            if N in surv:
                out.append(ln); out.append(lines[i + 1].rstrip())
            i += 2; continue
        if kw == "elset":  # non Set-N elsets (e.g. EALL): keep, restrict to kept elements
            out.append(ln); i += 1
            while i < len(lines) and _kw(lines[i]) is None:
                toks = [t.strip() for t in lines[i].split(",")]
                kept = [t for t in toks if t.isdigit() and int(t) in keepe]
                if kept:
                    out.append(", ".join(kept) + ",")
                i += 1
            continue
        if kw == "nset":
            out.append(ln); i += 1
            while i < len(lines) and _kw(lines[i]) is None:
                toks = [t.strip() for t in lines[i].split(",")]
                kepttoks = [t for t in toks if t.isdigit() and int(t) in keepn]
                if kepttoks:
                    out.append(", ".join(kepttoks) + ",")
                i += 1
            continue
        out.append(ln.rstrip()); i += 1
    return "\n".join(out) + "\n"
