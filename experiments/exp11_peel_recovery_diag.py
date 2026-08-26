"""Diagnostica del bc_resid crescente in fe/interlaminar.py -- VERDETTO: benigno.

`interlaminar()` controlla che la tau recuperata sia ~0 sulle superfici libere
(`bc_resid`). Quel residuo CRESCE col raffinamento (5.79 -> 22.9 MPa su quattro
mesh), il che sembrava un recupero che si degrada proprio dove il pannello (C)
della figura dei pitfall fa il suo punto.

Non lo e'. Tre misure, 2026-07-20 (RS-005):

 1. il massimo del residuo sta SEMPRE a j=0, cioe' esattamente sul bordo libero
    dove la soluzione esatta e' singolare (Pipes-Pagano);
 2. nell'INTERNO, esclusa la banda di bordo, cresce molto meno (2.1 -> 5.88 MPa)
    e resta all'1.3% della scala degli sforzi in piano (|sxx|max ~ 460 MPa);
 3. non e' un artefatto del filtro: il modulo liscia con una finestra fissa di
    UNA cella, la cui larghezza fisica si dimezza a ogni raffinamento, ma
    tenendola a larghezza fisica costante (--physical) il residuo non cala
    (22.9 -> 21.5 MPa, -6%).

Cioe': il recupero e' sano dove il campo e' regolare, e il residuo cresce dove
la soluzione esatta non ha un valore finito. E' lo stesso fenomeno che il
pannello (C) racconta, non un bug che lo smentisce.

Uso:  CCX_BIN=ccx_2.21 python3 experiments/exp11_peel_recovery_diag.py [--physical]
"""
import os, subprocess, tempfile, shutil, sys
import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from fe.interlaminar import make_solid_deck, _parse_stress_grid, LX, LY, PLY_T, D0

from fe.ccx_bin import CCX          # un solo default per il binario ccx (audit F12)
SEQ = [0, 45, -45, 90] * 6
AXIAL, SIDE = -20000 * 0.44, 5000 * 0.44


def recover(nx, ny, smooth_physical=False):
    deck, nx, ny, nz = make_solid_deck(SEQ, AXIAL, SIDE, nx, ny)
    d = tempfile.mkdtemp()
    try:
        open(d + '/job.inp', 'w').write(deck)
        subprocess.run([CCX, '-i', 'job'], cwd=d, capture_output=True, text=True,
                       timeout=900, env={**os.environ, 'OMP_NUM_THREADS': '1'})
        frd = open(d + '/job.frd').read()
        sig = _parse_stress_grid(frd, nx, ny, nz)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    dx, dy, dz = LX / nx, LY / ny, PLY_T

    def smooth(a, w):
        """box 1D ripetuto; w = semi-larghezza in CELLE (il modulo usa w=1 fisso)"""
        b = a.copy()
        for _ in range(w):
            c = b.copy()
            c[1:-1, :, :] = (b[:-2, :, :] + b[1:-1, :, :] + b[2:, :, :]) / 3
            b = c
            c = b.copy()
            c[:, 1:-1, :] = (b[:, :-2, :] + b[:, 1:-1, :] + b[:, 2:, :]) / 3
            b = c
        return b

    # il modulo liscia con una finestra FISSA di 1 cella: la sua larghezza FISICA
    # si dimezza a ogni raffinamento. Con smooth_physical la si tiene costante.
    w = 1 if not smooth_physical else max(1, round(nx / 20))
    sxx, syy, sxy = smooth(sig[..., 0], w), smooth(sig[..., 1], w), smooth(sig[..., 3], w)
    dsxx_dx = np.gradient(sxx, dx, axis=0); dsxy_dy = np.gradient(sxy, dy, axis=1)
    dsxy_dx = np.gradient(sxy, dx, axis=0); dsyy_dy = np.gradient(syy, dy, axis=1)
    txz = np.zeros_like(sxx)
    for k in range(1, nz + 1):
        txz[:, :, k] = txz[:, :, k-1] - 0.5*(dsxx_dx[:, :, k]+dsxy_dy[:, :, k]
                                             + dsxx_dx[:, :, k-1]+dsxy_dy[:, :, k-1]) * dz
    jband = max(1, int(D0 / dy))
    top = np.abs(txz[:, :, nz])
    interior = top[1:nx, jband+1:ny-jband]          # esclusa la banda di bordo libero
    edge = np.r_[top[:, :jband+1].ravel(), top[:, ny-jband:].ravel()]
    imax = np.unravel_index(np.argmax(top), top.shape)
    return dict(nx=nx, ny=ny, jband=jband, w=w,
                bc_all=float(top.max()), bc_interior=float(interior.max()),
                bc_edge=float(edge.max()),
                argmax_j=int(imax[1]), argmax_i=int(imax[0]),
                sxx_scale=float(np.abs(sig[..., 0]).max()))


if __name__ == '__main__':
    phys = '--physical' in sys.argv
    print(f"smoothing a larghezza fisica costante: {phys}")
    print(f"{'mesh':>9} {'w':>2} {'jband':>5} {'bc_all':>9} {'bc_interno':>11} {'bc_bordo':>9} "
          f"{'argmax j':>8} {'|sxx|max':>9} {'bc/|sxx|':>9}")
    for nx, ny in [(10, 6), (20, 10), (30, 16), (40, 20)]:
        r = recover(nx, ny, phys)
        print(f"{r['nx']:3d} x{r['ny']:3d} {r['w']:2d} {r['jband']:5d} {r['bc_all']:9.3g} "
              f"{r['bc_interior']:11.3g} {r['bc_edge']:9.3g} {r['argmax_j']:8d} "
              f"{r['sxx_scale']:9.4g} {r['bc_all']/r['sxx_scale']:9.2%}")
