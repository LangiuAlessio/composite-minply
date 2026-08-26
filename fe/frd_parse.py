"""Parse CalculiX .frd nodal results: displacement + stress -> von-Mises.

Keys on block headers "-4  DISP" / "-4  STRESS"; reads "-1 <node> v..." rows
until block end. Tolerant: empty lists if a block is absent."""
from __future__ import annotations
import math


def von_mises(sxx, syy, szz, sxy, syz, szx) -> float:
    return math.sqrt(0.5*((sxx-syy)**2+(syy-szz)**2+(szz-sxx)**2)
                     + 3.0*(sxy**2+syz**2+szx**2))


def _skip_minus5(lines, start):
    n = 0
    while start + n < len(lines) and lines[start + n][:3].strip() == "-5":
        n += 1
    return n


def _read_block(lines, start):
    """Read data rows in CalculiX fixed-width Fortran format (1X,A2,I10,6E12.5).

    node = cols 3:13, each value is a 12-char field starting at col 13. A
    negative first value abuts the node field, so .split() is unsafe -- we
    slice by fixed columns instead. The block ends at a -3 (or any non
    -1/-2/-3 tag, or EOF)."""
    rows = []
    i = start
    while i < len(lines):
        line = lines[i]
        tag = line[:3].strip()
        if tag in ("-1", "-2"):
            try:
                node = int(line[3:13])
            except ValueError:
                # malformed data row: skip defensively, do NOT end the block
                i += 1
                continue
            vals = []
            k = 0
            while True:
                chunk = line[13 + 12 * k: 13 + 12 * (k + 1)]
                if not chunk.strip():
                    break
                try:
                    vals.append(float(chunk))
                except ValueError:
                    break
                k += 1
            rows.append((node, vals)); i += 1
        else:
            break
    return rows, i


def parse_frd_static(text: str) -> dict:
    lines = text.splitlines()
    disp, stress = [], []
    i = 0
    while i < len(lines):
        u = lines[i].upper()
        if u.lstrip().startswith("-4") and "DISP" in u:
            rows, i = _read_block(lines, i + 1 + _skip_minus5(lines, i + 1))
            for node, v in rows:
                if len(v) >= 3:
                    ux, uy, uz = v[0], v[1], v[2]
                    disp.append({"node": node, "ux": ux, "uy": uy, "uz": uz,
                                 "umag": math.sqrt(ux*ux+uy*uy+uz*uz)})
            continue
        if u.lstrip().startswith("-4") and "STRESS" in u:
            rows, i = _read_block(lines, i + 1 + _skip_minus5(lines, i + 1))
            for node, v in rows:
                if len(v) >= 6:
                    stress.append({"node": node,
                                   "vm": von_mises(v[0], v[1], v[2], v[3], v[4], v[5])})
            continue
        i += 1
    return {"disp": disp, "von_mises": stress}
