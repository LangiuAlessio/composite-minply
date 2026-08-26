#!/usr/bin/env python3
"""Figures 1-3: three views of the validated reference model, computed HERE with CalculiX.

Why this script exists. Until 2026-07-27 the three views in the paper were screenshots of
the coauthor's Abaqus session, extracted from his .docx: 1086x446, 1171x514 and 1188x819
pixels, i.e. 172, 186 and 189 dpi at the width the paper prints them, against the 600 dpi
Springer asks for combination/colour art. There was no higher-resolution copy on our side,
and one of them (the modal view) had its colour bar recomposed by hand, which the paper had
to disclose. Both problems dissolve if the views are OURS: this script solves the same three
cases on the validated decks in this bundle and draws them as vector PDF, so resolution stops
being a quantity at all, and the figures move from "not reproducible without a licence" to
reproducible by anyone with ccx.

They are drawn in the LOOK of the originals, not in a plotting-library style: same twelve-band
colour ramp (sampled from the coauthor's own legend, see figures/_abaqus_style.py), same
legend box, same axis triad, deformed surface and solid block in axonometric projection. The
first attempt, on 2026-07-27, replaced them with flat maps in millimetres; the coauthor asked
on 2026-07-28 to keep the views as they looked, so what changed is the resolution and the
provenance, not the picture.

What it draws, all on ONE laminate so the three read as one model -- the 60-ply symmetric
cross-ply [0/90]_15s that is already the cross-solver anchor of the validation table:

  (1) first buckling eigenvector, S8R shell, axial+shear at the FIRST load scheme
      (-1000 N axial, 5000 N shear) -- the row tab:validation compares against Abaqus,
      NOT a campaign case of tab:buckling, whose magnitudes were retuned later;
  (2) first flexural mode of the same shell (the frequency row of tab:validation);
  (3) in-plane sigma_11 and peel sigma_33 on the C3D8I solid under the static case S1.

Canaries. The buckling factor and the first frequency are printed in the paper (3.71 and
628.0 Hz on this laminate). The script REFUSES to draw if what it measures drifts from them
by more than 1%: a figure that no longer matches the table it illustrates is worse than no
figure. Pass --force to draw anyway (and then fix the table, not the canary).

Provenance of every number: `_out/fig_fe_views.json`, written next to the figures.

Usage:  python3 code/figures/fig_fe_views.py [--force] [--quick]
        (--quick coarsens the solid IN-PLANE mesh to 10x5: for eyeballing the layout,
         NOT for the paper -- it changes the field.)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # .../code
sys.path.insert(0, str(ROOT))

import numpy as np                                      # noqa: E402
import matplotlib                                       # noqa: E402
from figures._style import style, INK, MUTED, AXIS      # noqa: E402
import matplotlib.pyplot as plt                         # noqa: E402
from matplotlib.tri import Triangulation                # noqa: E402

import optimisers.constrained_search as R               # noqa: E402
from fe.interlaminar import make_solid_deck             # noqa: E402
from fe.frd_parse import _read_block                    # noqa: E402
from figures import _abaqus_style as A                  # noqa: E402

OUT = ROOT / 'figures' / '_out'
CCX = os.environ.get('CCX_BIN', R.CCX)

# the cross-solver anchor: [0/90]_15s, 60 plies, mid-plane symmetric (fe/reference_cases.py)
HALF = [0, 90] * 15
SEQ = HALF + HALF[::-1]

# The buckling view is drawn on the case the VALIDATION table anchors on, so that the number
# in the figure is one the paper prints against Abaqus: axial+shear at the coauthor's own
# reference magnitudes (axial -1000, side 5000 N), not the v2 campaign case -- the campaign
# cases are the ones the optimiser runs, and no Abaqus counterpart is published for them.
CASE_BUCK_NAME = 'axial+shear (validation anchor)'
CASE_BUCK_LOADS = dict(axial=-1000., side=5000., torsion=0., threshold=0.)

# printed in the paper for THIS laminate (tab:validation)
PAPER_BLF, PAPER_BLF_ABQ = 3.71, 3.70            # CalculiX / Abaqus
PAPER_FREQ_HZ, PAPER_FREQ_ABQ = 628.0, 626.5     # CalculiX / Abaqus
# 1% band. The reference is the number FROZEN IN THE PAPER plus the independent Abaqus
# column beside it, not another run of this same script: a canary that shares the failure
# mode of what it checks certifies nothing.
TOL = 0.01

# sequential colour map: monotone in lightness, so the field survives a greyscale printer
FIELD_CMAP = 'cividis'


# --------------------------------------------------------------------------- ccx plumbing
def run_deck(deck: str, threads: int = 1, timeout: int = 3600) -> dict:
    """Solve a deck in a temp dir; return {'frd':..., 'dat':..., 'seconds':...}."""
    d = tempfile.mkdtemp()
    try:
        (Path(d) / 'job.inp').write_text(deck)
        t0 = time.time()
        p = subprocess.run([CCX, '-i', 'job'], cwd=d, capture_output=True, text=True,
                           timeout=timeout,
                           env={**os.environ, 'OMP_NUM_THREADS': str(threads)})
        el = time.time() - t0
        frd = (Path(d) / 'job.frd')
        dat = (Path(d) / 'job.dat')
        return {'frd': frd.read_text() if frd.exists() else '',
                'dat': dat.read_text() if dat.exists() else '',
                'stdout': p.stdout[-1500:], 'seconds': round(el, 1)}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _blocks(frd: str, kind: str):
    """[(step_value, {node: values}), ...] for every '-4  <kind>' block, in file order.

    The step value is the number on the '100CL' line above the block: for a *BUCKLE run it
    IS the buckling factor of that mode, for *FREQUENCY it is the eigenvalue. It matters
    because ccx writes THREE DISP blocks for a two-mode buckling job -- the first is the
    static pre-solution at step value 0, and reading it as "mode 1" gives a field that is
    not the eigenvector at all (it is what made the first run of this script draw zeros).
    """
    lines = frd.splitlines()
    out = []
    for i, ln in enumerate(lines):
        u = ln.upper()
        if u.lstrip().startswith('-4') and kind in u:
            val = 0.0
            for back in lines[max(0, i - 8):i]:
                if back.startswith('  100CL'):
                    try:
                        val = float(back.split()[2])
                    except (IndexError, ValueError):
                        val = 0.0
            j = i + 1
            while j < len(lines) and lines[j][:3].strip() == '-5':
                j += 1
            rows, _ = _read_block(lines, j)
            out.append((val, {n: v for n, v in rows}))
    return out


def frd_coords(frd: str) -> dict:
    """{node: (x, y, z)} from the .frd's own coordinate block.

    Needed because ccx EXPANDS a composite shell for output: this 661-node S8R panel comes
    back as 93,180 nodes stacked through the thickness, so the deck's node ids cannot be
    used to place the field on the plane."""
    lines = frd.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith('2C'):
            rows, _ = _read_block(lines, i + 1)
            return {n: tuple(v[:3]) for n, v in rows if len(v) >= 3}
    return {}


def mode_field(frd: str, kind: str = 'DISP'):
    """(step_value, xy, U, magnitude) of the first real mode, on the panel plane.

    Through-thickness duplicates of the same (x, y) are reduced by taking the largest
    magnitude, which is the quantity the caption names; the displacement VECTOR of that
    same node comes back with it, because the Abaqus-style view draws the deformed
    surface, not a flat map. The domain is a rectangle, so the default Delaunay
    triangulation of the projected nodes is exact."""
    blocks = [(v, d) for v, d in _blocks(frd, kind) if abs(v) > 0.0]
    if not blocks:
        return None
    val, disp = blocks[0]
    coords = frd_coords(frd)
    acc = {}
    for n, v in disp.items():
        if n not in coords or len(v) < 3:
            continue
        x, y, _ = coords[n]
        key = (round(x, 4), round(y, 4))
        mag = float(np.linalg.norm(v[:3]))
        if mag > acc.get(key, (-1.0, None))[0]:
            acc[key] = (mag, np.asarray(v[:3], dtype=float))
    keys = sorted(acc)
    xy = np.array(keys, dtype=float)
    U = np.array([acc[k][1] for k in keys])
    m = np.array([acc[k][0] for k in keys])
    return val, xy, U, m


def parse_first_eigenvalue(dat: str, kind: str) -> float:
    """First buckling factor (*BUCKLE) or first frequency in Hz (*FREQUENCY) from the .dat."""
    if kind == 'buckle':
        f = re.findall(r'^\s*1\s+([\d.E+\-]+)\s*$', dat, re.M)
        return float(f[0]) if f else float('nan')
    inblk = False
    for ln in dat.splitlines():
        u = ln.upper()
        if 'MODE NO' in u and 'EIGENVALUE' in u:
            inblk = True
            continue
        if inblk:
            p = ln.split()
            if len(p) >= 5:
                try:
                    int(p[0])
                    return float(p[3])          # cycles/time = Hz
                except ValueError:
                    continue
    return float('nan')


# ------------------------------------------------------------------------------- drawing
def mode_figure(xy, U, mag, note: str | None = None, figsize=(4.19, 1.71),
                elev=A.ELEV, azim=A.AZIM, rect=(0.0291, -0.3947, 1.1155, 2.0117),
                roll=0.0, legend_xy=(0.0242, 0.2591), triad_xy=(0.3031, 0.2389),
                triad_len=0.0509, note_xy=(0.6162, 0.2088)):
    """Una vista modale come le disegnava il coautore in Abaqus: deformata in assonometria,
    dodici bande, legenda `U, Magnitude` normalizzata a uno, triade degli assi.

    La figura e' disegnata PICCOLA di proposito. Il manoscritto la stampa a larghezza di
    colonna (circa 250 pt in due colonne): una figura sorgente da 7 pollici verrebbe ridotta
    al 56%, e i corpi 7.5 pt della legenda arriverebbero al lettore a 4 pt. A questa
    dimensione la scala e' quasi 1:1 e la legenda resta leggibile.

    La TELA, pero', deve contenere l'oggetto. Fino al 2026-07-29 non lo conteneva: l'asse 3D
    deborda dalla figura di proposito (in prospettiva la camera allontana l'oggetto, e senza
    questo il pannello resterebbe un francobollo accanto alla legenda), ma il rect era tanto
    grande che il pannello stesso finiva oltre il bordo -- 11% del lato inferiore e 4% del
    destro coperti d'inchiostro nella vista di buckling, 5% del superiore in quella modale.
    `pdfcrop` non li recupera: taglia il bianco, non ricostruisce l'inchiostro mai disegnato.
    La correzione NON tocca vista, focale, prospettiva ne' il rapporto fra legenda e oggetto,
    che sono calibrati sulle figure del coautore: allarga la tela e trasla il contenuto,
    lasciando ogni elemento alla stessa dimensione ASSOLUTA in pollici. Le costanti qui sotto
    sono quelle vecchie riportate sulla tela nuova (misura in
    `.sessions/*/scratch/layout_probe.py`, ingombro reale 4.15 x 1.67 in su una tela che era
    4.1 x 1.72). Se si ritocca la vista, si rimisura l'ingombro e si rifanno: non si stringe
    la tela finche' "sembra a posto"."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(list(rect), projection='3d')
    LX, LY, zr = A.draw_deformed_surface(fig, ax, xy, U, mag, elev=elev, azim=azim,
                                        roll=roll)
    A.legend_box(fig, 'U, Magnitude', np.linspace(0, 1, A.NBANDS + 1),
                 xy=legend_xy, fontsize=5.0)
    A.axis_triad(fig, ax, (LX, LY, zr), xy=triad_xy, length=triad_len, figsize=figsize)
    if note:
        fig.text(note_xy[0], note_xy[1], note, fontsize=8, ha='center', va='center',
                 family=A.FONT, zorder=7)
    return fig, ax


def draw_field(ax, tri, values, label, cmap=FIELD_CMAP, levels=18, fmt=None):
    """Filled field on the panel, with the clamped edge marked. Returns the mappable."""
    tcf = ax.tricontourf(tri, values, levels=levels, cmap=cmap)
    ax.tricontour(tri, values, levels=levels, colors='white', linewidths=0.15, alpha=0.55)
    ax.plot([0, 0], [0, R.LY], color=INK, lw=2.2, solid_capstyle='butt', zorder=5)
    ax.annotate('clamped', xy=(0, R.LY / 2), xytext=(4, R.LY / 2 + 3.2),
                color=INK, fontsize=7.5, rotation=90, va='center', ha='left')
    ax.set_aspect('equal')
    ax.set_xlim(-2, R.LX + 2)
    ax.set_ylim(-2, R.LY + 2)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    cb = ax.figure.colorbar(tcf, ax=ax, fraction=0.032, pad=0.02,
                            format=fmt) if label else None
    if cb is not None:
        cb.set_label(label, fontsize=8)
        cb.outline.set_edgecolor(AXIS)
        cb.outline.set_linewidth(0.6)
    return tcf


def clip_loss(fig, dpi: int = 150) -> int:
    """Pixel di inchiostro che il RITAGLIO DEGLI ASSI sta mangiando.

    mplot3d ritaglia al proprio riquadro, e quel riquadro e' il quadrato inscritto nel rect
    passato ad `add_axes`, non il rect: un oggetto piu' largo che alto ne esce e viene tagliato
    IN MEZZO ALLA TELA. Un controllo sull'inchiostro al bordo della figura non lo vede -- ha
    detto "pulito" per due giri mentre gli angoli del solido sparivano davvero.
    Qui si disegna due volte, con e senza `clip_on`, e si contano i pixel di differenza.
    """
    import io
    from PIL import Image

    def raster():
        buf = io.BytesIO()
        fig.savefig(buf, dpi=dpi, format='png', facecolor='white')
        buf.seek(0)
        return np.asarray(Image.open(buf).convert('RGB')).astype(int).sum(axis=2) < 720

    on = raster()
    toggled = []
    for ax in fig.axes:
        for art in list(ax.collections) + list(ax.lines):
            if art.get_clip_on():
                art.set_clip_on(False)
                toggled.append(art)
    off = raster()
    for art in toggled:
        art.set_clip_on(True)
    return int((off & ~on).sum())


def save(fig, stem: str, paper_name: str | None = None, crop_margin: int = 7) -> None:
    """Scrive il PDF vettoriale (+ un PNG a 600 dpi), lo ritaglia e lo porta nel paper.

    Il ritaglio serve perche' una vista assonometrica lascia molto bianco attorno
    all'oggetto: senza, `\\includegraphics[width=\\linewidth]` scala il bianco e il pannello
    stampa piccolo. `pdfcrop` (TeX Live, gia' necessario per compilare il manoscritto) lo
    fa senza rasterizzare; se manca, si tiene il PDF intero e lo si dice.

    `crop_margin` e' in punti ed e' salito da 3 a 7 il 2026-07-29: a 3 pt il pannello
    arrivava praticamente al bordo dell'immagine e, accanto alle figure del coautore -- che
    hanno bianco su tutti i lati -- si leggeva come TAGLIATO anche quando era intero
    (rilievo dell'utente sulla figura 3). 7 pt sono i 0,1 pollici di margine con cui la tela
    e' dimensionata: PNG e PDF cosi' respirano uguale. Il margine si paga in dimensione di
    stampa, perche' `width=\\linewidth` scala anche il bianco: per questo la figura 3 ha in
    cambio i blocchi piu' grandi del 15%.
    """
    lost = clip_loss(fig)
    if lost and '--force' not in sys.argv:
        sys.exit(f'{stem}: il ritaglio degli assi mangia {lost} px di inchiostro -- '
                 'l\'oggetto esce dal quadrato del riquadro 3D. Alza `pad` in '
                 'draw_solid_block o allarga il rect. (--force per disegnare comunque)')
    print(f'  ritaglio degli assi: {lost} px persi' + ('  [IGNORATO --force]' if lost else ''))
    OUT.mkdir(parents=True, exist_ok=True)
    pdf, png = OUT / f'{stem}.pdf', OUT / f'{stem}.png'
    fig.savefig(pdf)                     # vector: resolution stops being a quantity
    fig.savefig(png, dpi=600)            # 600 dpi raster for anyone who wants pixels
    plt.close(fig)
    if shutil.which('pdfcrop'):
        cropped = OUT / f'{stem}_crop.pdf'
        subprocess.run(['pdfcrop', '--margins', str(crop_margin), str(pdf), str(cropped)],
                       capture_output=True, check=False)
        if cropped.exists():
            shutil.move(str(cropped), str(pdf))
    else:
        print('  (pdfcrop assente: PDF non ritagliato, la figura stampera\' piu\' piccola)')
    if paper_name:                       # il file che il manoscritto include davvero
        shutil.copyfile(pdf, ROOT.parent / paper_name)
    print(f'  wrote {pdf.name} + {png.name} (600 dpi)'
          + (f' -> {paper_name}' if paper_name else ''))


# ------------------------------------------------------------------------------- figures
def fig_buckling(prov: dict):
    case = CASE_BUCK_LOADS
    deck = R.make_ccx_deck(SEQ, case).replace('*END STEP', '*NODE FILE\nU\n*END STEP')
    r = run_deck(deck, threads=1)                    # ccx<2.21 threaded buckling is wrong
    blf_dat = parse_first_eigenvalue(r['dat'], 'buckle')
    got = mode_field(r['frd'])
    if got is None:
        sys.exit('fig_buckling: nessun modo nel .frd -- ' + r['stdout'][-400:])
    blf, xy, U, mag = got
    prov['buckling'] = {'case': CASE_BUCK_NAME, 'loads_N': case,
                        'blf_ccx_frd': round(blf, 4), 'blf_ccx_dat': round(blf_dat, 4),
                        'paper_ccx': PAPER_BLF, 'paper_abaqus': PAPER_BLF_ABQ,
                        'plane_nodes': int(len(xy)), 'seconds': r['seconds']}
    fig, ax = mode_figure(xy, U, mag)
    save(fig, 'fig_ref_buckling', 'RR_ref_buckling.pdf')
    return blf


def fig_frequency(prov: dict):
    from fe.reference_cases import make_freq_deck
    deck = make_freq_deck(SEQ, nfreq=4).replace('*END STEP', '*NODE FILE\nU\n*END STEP')
    r = run_deck(deck, threads=1)
    f1_dat = parse_first_eigenvalue(r['dat'], 'freq')
    got = mode_field(r['frd'])
    if got is None:
        sys.exit('fig_frequency: nessun modo nel .frd -- ' + r['stdout'][-400:])
    _, xy, U, mag = got
    prov['frequency'] = {'f1_hz_ccx': round(f1_dat, 2), 'paper_ccx': PAPER_FREQ_HZ,
                         'paper_abaqus': PAPER_FREQ_ABQ, 'plane_nodes': int(len(xy)),
                         'seconds': r['seconds']}
    # la frequenza annotata sotto il pannello, dove la stampa Abaqus: e' il nostro valore
    # arrotondato al decimale che il paper porta in tab:validation (628.0 Hz)
    # vista calibrata sulla SUA figura 2, che guarda il pannello quasi in vera forma
    # tela 3.87 x 1.865: l'ingombro misurato e' 3.83 x 1.825 in (era 4.1 x 1.95, e il modo
    # sbordava di 0.02 in in alto e a sinistra)
    fig, ax = mode_figure(xy, U, mag, note=f'Freq = {f1_dat:8.2f}    (cycles/time)',
                          figsize=(3.87, 1.865), elev=A.ELEV_FREQ, azim=A.AZIM_FREQ,
                          rect=(0.0739, -0.3389, 1.1018, 1.6938), roll=A.ROLL_FREQ,
                          legend_xy=(0.0262, 0.0898), triad_xy=(0.3282, 0.0689),
                          triad_len=0.0551, note_xy=(0.6672, 0.0375))
    save(fig, 'fig_ref_frequency', 'RR_ref_frequency.pdf')
    return f1_dat


def fig_stress(prov: dict, quick: bool = False):
    """sigma_11 on the loaded surface and peak peel sigma_33 through the thickness, S1."""
    from fe.interlaminar import _parse_stress_grid
    sload = R.STATIC_LOAD
    nx, ny = (10, 5) if quick else (20, 10)      # --quick: mesh IN PIANO piu' grossa
    deck, nx, ny, nz = make_solid_deck(SEQ, axial=sload['axial'], side=sload['side'],
                                       nx=nx, ny=ny)
    print(f'  solido C3D8I: {(nx+1)*(ny+1)*(nz+1)} nodi, {nx*ny*nz} elementi -- ccx...')
    r = run_deck(deck, threads=2)
    if not r['frd']:
        sys.exit('fig_stress: ccx non ha prodotto .frd -- ' + r['stdout'][-600:])
    sig = _parse_stress_grid(r['frd'], nx, ny, nz)            # (i,j,k,6): xx yy zz xy yz zx
    X = np.linspace(0, R.LX, nx + 1)
    Y = np.linspace(0, R.LY, ny + 1)
    s11_surf = sig[:, :, nz, 0]                              # superficie esterna z=+h
    s33_peak = sig[:, :, :, 2].max(axis=2)                    # worst tensile peel over z
    prov['stress'] = {
        'load_case': 'S1', 'loads_N': sload, 'nx': nx, 'ny': ny, 'nz': nz,
        'nodes': (nx + 1) * (ny + 1) * (nz + 1), 'quick': bool(quick),
        's11_surface_MPa': [round(float(s11_surf.min()), 1), round(float(s11_surf.max()), 1)],
        's33_peak_MPa': [round(float(s33_peak.min()), 2), round(float(s33_peak.max()), 2)],
        'allowables_MPa': {'in_plane': 700.0, 'peel': 10.0},
        'seconds': r['seconds']}
    # Due blocchi in assonometria, come li stampava il coautore: sopra sigma_11, sotto il
    # peel. Le facce esterne portano il campo VERO su ognuna, quindi la legenda copre il
    # min-max del volume; per sigma_11 coincide con quello di superficie (0 e 370.7 MPa),
    # per il peel il minimo di volume (-1.3) e' piu' basso di quello di superficie.
    from fe.interlaminar import PLY_T
    Zc = np.linspace(0, nz * PLY_T, nz + 1)
    # Gli assi 3D si SOVRAPPONGONO di proposito (sono trasparenti): un'assonometria lascia
    # molto vuoto sopra e sotto l'oggetto, e senza la sovrapposizione i due blocchi si
    # allontanano tanto da far crescere il reso di una pagina.
    # Storia di questa tela, perche' i numeri non si tocchino a occhio.
    # Era 4.1 x 2.45 in mentre l'ingombro reale e' 4.27 x 2.735: i due blocchi USCIVANO dalla
    # figura -- 22% del lato superiore, 20% dell'inferiore e 38% del destro coperti
    # d'inchiostro, cioe' gli angoli del solido erano troncati davvero (rilievo dell'utente,
    # 2026-07-29 mattina). Primo giro: tela a 4.31 x 2.775, che li conteneva -- ma di 0,017
    # pollici, e con `pdfcrop --margins 3` il solido arrivava al bordo dell'immagine. Accanto
    # alle figure del coautore, che hanno bianco su tutti i lati, continuava a leggersi come
    # tagliato (secondo rilievo dell'utente, stesso giorno), e stampava piu' piccolo del suo.
    # Secondo giro: blocchi +15% e margine bianco vero di 0.1 in, con `crop_margin=7 pt` in
    # `save()` perche' pdfcrop non se lo rimangi.
    # Terzo giro, quello attuale (richiesta dell'utente: figura piu' alta, legenda piu'
    # grande, blocchi un pelo piu' piccoli cosi' gli angoli restano larghi). Blocchi
    # riportati a +5%, legende da 4.2 a **5.6 pt**, e i due assi separati di **1.60 in**
    # invece di 1.37: e' la separazione a far crescere l'ALTEZZA, che a `width=\linewidth`
    # e' l'unico modo di occupare piu' spazio a parita' di colonna. Rapporto
    # larghezza/altezza da 1.55 a **1.40** (quello del coautore e' 1.45), e il blocco occupa
    # il 74% della larghezza invece dell'80% -- il suo ne occupa il 75%.
    # I rect NON sono stimati: li calcola `fig3_candidate.py` (copia in
    # review_meeting_2026-07-29/) misurando l'inchiostro su una tela quintupla e centrando
    # ogni legenda sul proprio blocco. Se si cambia la vista o il corpo delle legende si
    # rilancia quello e si riportano i numeri, non si aggiustano a occhio.
    # Quarto giro: **i rect sono QUADRATI, e non e' un dettaglio**. mplot3d ritaglia al
    # quadrato inscritto nel rect (lato = il lato minore), centrato: i rect larghi e bassi dei
    # giri precedenti non allargavano niente, spostavano il quadrato, e le estremita' del
    # blocco -- proiettato il 2,9% piu' largo del quadrato -- venivano tagliate in mezzo alla
    # tela. E' il taglio che l'utente vedeva e che i controlli sul bordo della figura non
    # potevano trovare. Ora `draw_solid_block(pad=0.10)` allontana la camera quel tanto che
    # basta perche' il blocco stia dentro (93,6% del lato, 3,2% di aria per parte) e `save()`
    # ha il gate che conta i pixel persi dal ritaglio: se ne perde uno, si ferma.
    # Numeri da `fig3_candidate.py` (copia in review_meeting_2026-07-29/): blocco largo
    # 3,25 in -- un altro 11% in meno, come chiesto -- e la tela cresce in altezza
    # (rapporto 1,29 contro 1,40 di prima; il coautore sta a 1,45).
    figsize = (4.310, 3.340)
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_axes([0.1819, 0.2154, 0.8054, 1.0393], projection='3d')
    b1 = A.draw_solid_block(ax1, sig[..., 0], X, Y, Zc)
    A.legend_box(fig, 'S, S11 [MPa]', b1, xy=(0.0464, 0.5456), fmt='{:.0f}', fontsize=5.6)
    ax2 = fig.add_axes([0.1819, -0.2696, 0.8054, 1.0393], projection='3d')
    b2 = A.draw_solid_block(ax2, sig[..., 2], X, Y, Zc)
    A.legend_box(fig, 'S, S33 [MPa]', b2, xy=(0.0464, 0.0605), fmt='{:.1f}', fontsize=5.6)
    save(fig, 'fig_ref_stress', 'RR_ref_stress.pdf')
    prov['stress']['s11_volume_MPa'] = [round(float(sig[..., 0].min()), 1),
                                        round(float(sig[..., 0].max()), 1)]
    prov['stress']['s33_volume_MPa'] = [round(float(sig[..., 2].min()), 2),
                                        round(float(sig[..., 2].max()), 2)]
    return prov['stress']


def main() -> None:
    force = '--force' in sys.argv
    quick = '--quick' in sys.argv
    prov = {'laminate': '[0/90]_15s (60-ply symmetric cross-ply)', 'seq': SEQ,
            'solver': 'CalculiX', 'ccx_bin': CCX, 'note':
            'sostituisce le tre viste Abaqus estratte dal .docx del coautore (172/186/189 dpi)'}
    print('(1/3) buckling, shell S8R')
    blf = fig_buckling(prov)
    print('(2/3) prima frequenza, stesso shell')
    f1 = fig_frequency(prov)
    print('(3/3) sforzi sul solido C3D8I, carico statico S1')
    st = fig_stress(prov, quick=quick)

    bad = []
    if not abs(blf - PAPER_BLF) <= TOL * PAPER_BLF:
        bad.append(f'BLF misurato {blf:.4f} contro {PAPER_BLF} stampato nel paper '
                   f'(Abaqus {PAPER_BLF_ABQ})')
    if not abs(f1 - PAPER_FREQ_HZ) <= TOL * PAPER_FREQ_HZ:
        bad.append(f'f1 misurata {f1:.2f} Hz contro {PAPER_FREQ_HZ} stampata nel paper')
    prov['canaries'] = {'ok': not bad, 'failed': bad, 'tol': TOL}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'fig_fe_views.json').write_text(json.dumps(prov, indent=1) + '\n')
    print(f'\nprovenienza -> {(OUT / "fig_fe_views.json").relative_to(ROOT)}')
    print(f'  BLF = {blf:.4f} (paper ccx {PAPER_BLF}, Abaqus {PAPER_BLF_ABQ})   '
          f'f1 = {f1:.2f} Hz (paper ccx {PAPER_FREQ_HZ}, Abaqus {PAPER_FREQ_ABQ})')
    print(f'  sigma_11 superficie {st["s11_surface_MPa"]} MPa (allowable 700), '
          f'peel sigma_33 {st["s33_peak_MPa"]} MPa (allowable 10)')
    if bad and not force:
        sys.exit('CANARY: ' + ' ; '.join(bad) + '\n(--force per disegnare comunque)')
    if bad:
        print('CANARY IGNORATA (--force): ' + ' ; '.join(bad))


if __name__ == '__main__':
    main()
