#!/usr/bin/env python3
"""Rendering in stile Abaqus/Viewer per le tre viste del modello di riferimento.

Perche' esiste. Fino al 2026-07-27 le Figure 1-3 erano screenshot della sessione Abaqus
del coautore, a 172/186/189 dpi contro i 600 che Springer chiede. Rifarle in casa risolveva
la risoluzione ma cambiava l'aspetto: mappe piane con assi in millimetri al posto delle
viste tridimensionali che il coautore aveva prodotto. Il coautore le preferiva com'erano
(2026-07-28), quindi le viste sono ricalcolate qui MA disegnate come le sue: stessa
scala di colori, stessa legenda a bande, stessa triade degli assi, blocco solido in
assonometria. Vettoriali, cosi' la risoluzione smette di essere una grandezza in gioco.

La rampa di colori NON e' inventata ne' approssimata da una colormap di matplotlib: i
dodici colori sono campionati pixel per pixel dalla legenda di `gc_buckling.png`, la
figura originale del coautore (ora in `archive/figure-gc-superate-2026-07-27/`). Sono
i valori RGB che Abaqus stampa, nell'ordine in cui li stampa.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

# dal massimo al minimo, come li elenca la legenda di Abaqus
_ABQ_TOP_DOWN = [(255, 0, 0), (255, 93, 0), (255, 185, 0), (232, 255, 0), (139, 255, 0),
                 (46, 255, 0), (0, 255, 46), (0, 255, 139), (0, 255, 232), (0, 185, 255),
                 (0, 93, 255), (0, 0, 255)]
BANDS = [tuple(c / 255 for c in rgb) for rgb in reversed(_ABQ_TOP_DOWN)]   # min -> max
CMAP = ListedColormap(BANDS)
NBANDS = 12
FONT = 'DejaVu Sans'

# Vista e focale calibrate sulla figura originale, non scelte a occhio. I quattro vertici
# del pannello in `gc_buckling.png` stanno a (443,144), (1066,141), (977,336), (322,409)
# pixel: i due bordi corti misurano 290 e 214 px, cioe' la vista del coautore ha
# PROSPETTIVA -- in ortografica sarebbero uguali, ed e' il motivo per cui il primo
# tentativo, ortografico, non poteva somigliare per quanto si girassero gli angoli.
#
# Il fit di Procrustes sui quattro vertici da' un residuo del 2,6% SOLO se si ammettono
# riflessioni: la sua vista e' specchiata rispetto ai nostri assi, perche' nel suo modello
# la larghezza cresce dall'altra parte. Specchiare il nostro disegno per farlo combaciare
# vorrebbe dire ribaltare un campo che NON e' simmetrico in y (il taglio lo sverga), cioe'
# mentire sulla figura. Fra le viste NON specchiate si e' presa la piu' somigliante --
# azimut isometrico di Abaqus, elevazione bassa, prospettiva marcata.
ELEV, AZIM, FOCAL = 20.0, -60.0, 1.5

# La vista NON e' la stessa nelle due figure del coautore, e va calibrata su ciascuna. Nella
# sua Figura 2 il pannello si vede quasi in vera forma: il rapporto fra i lati proiettati e'
# 1,94 contro il 2,0 reale (nella Figura 1 e' 2,53), cioe' la guarda quasi perpendicolarmente
# invece che di taglio. Da qui l'elevazione alta della vista modale.
ELEV_FREQ, AZIM_FREQ = 68.0, -80.0
# L'azimut e' stato ruotato da -96 a -80 per SEPARARE gli assi Y e Z della triade, che a -96
# si proiettavano quasi sovrapposti (2-6 gradi l'uno dall'altro; a -80 sono 10). Ruotare
# l'azimut inclina pero' il bordo inferiore del pannello: il `roll` -- rotazione nel piano
# immagine, che non tocca ne' vista ne' campo -- lo riporta orizzontale. Il valore non e' a
# occhio: e' la bisezione sull'angolo del bordo proiettato, che chiude a 0,00 gradi.
ROLL_FREQ = 9.4

# Amplificazione della deformata, in frazione della lunghezza del pannello. Un autovettore ha
# scala arbitraria, quindi questo e' un parametro di DISEGNO -- lo stesso "deformation scale
# factor" che Abaqus sceglie da solo. A 0,03-0,13 la piastra si legge come un piano inclinato:
# la flessione c'e' (il profilo di uz lungo la campata fa 0, -0,02, -0,18, -0,53, -1,00, cioe'
# una mensola inflessa) ma non si vede. A 0,20 la curvatura si legge, che e' il punto di una
# figura di modo.
AMPL = 0.20


def band_colors(values, vmin: float, vmax: float):
    """Colore di banda per ogni valore, con i dodici intervalli uguali di Abaqus."""
    bounds = np.linspace(vmin, vmax, NBANDS + 1)
    return CMAP(BoundaryNorm(bounds, NBANDS)(np.asarray(values))), bounds


def legend_box(fig, title: str, bounds, xy=(0.015, 0.42), size=None,
               fmt='{:.2f}', fontsize=7.5, lead=1.2) -> None:
    """La legenda di Abaqus: bande impilate, tacche, valori a destra, cornice nera.

    L'altezza NON e' una frazione fissa della figura: e' calcolata dalle tredici etichette
    che deve contenere, alla dimensione in punti che avranno. Con una frazione fissa la
    legenda stava finche' la figura era grande e collassava appena la si rimpiccioliva --
    che e' esattamente cio' che si e' fatto per stamparla a scala 1:1 in colonna.
    """
    W, H = (s * 72 for s in fig.get_size_inches())          # figura in punti
    fx, fy = (lambda p: p / W), (lambda p: p / H)           # punti -> frazione di figura
    lx, ly = xy
    # La cornice parte 5 pt a sinistra della barra colore: se `lx` e' piu' vicino al margine
    # di cosi', il lato SINISTRO del riquadro cade fuori dal canvas e sparisce -- gli altri
    # tre restano, e la legenda sembra aperta a sinistra. Successo davvero, con lx=0.015 su
    # una figura da 295 pt. Si rientra invece di disegnare un rettangolo tagliato.
    lx = max(lx, fx(6.0))
    if size is None:
        lw = fx(10.0)                                        # barra colore, 10 pt
        lh = fy((NBANDS + 1) * fontsize * lead)              # quanto serve alle 13 etichette
    else:
        lw, lh = size
    tick, gap = fx(4.0), fx(7.0)
    for i in range(NBANDS):
        fig.patches.append(plt.Rectangle((lx, ly + i * lh / NBANDS), lw, lh / NBANDS,
                                         transform=fig.transFigure, facecolor=BANDS[i],
                                         edgecolor='none', zorder=5))
    label_w = fx(fontsize * 0.62 * max(len(fmt.format(b)) for b in bounds))
    # Il TITOLO puo' essere piu' largo della colonna di etichette -- "S, S11 [MPa]" contro
    # "331" -- e allora sfonda la cornice a destra. Succedeva col corpo a 5.6 pt della
    # figura 3 (2026-07-29). La larghezza del titolo si MISURA sul renderer invece di
    # stimarla da una larghezza media di carattere: la stima sbaglia sui titoli con
    # parentesi e cifre, ed e' il tipo di errore che si vede solo stampato.
    title_art = fig.text(lx - fx(3), ly + lh + fy(fontsize * 1.4), title,
                         fontsize=fontsize + 0.5, va='center', ha='left', family=FONT,
                         zorder=6)
    try:
        rend = fig.canvas.get_renderer()
        title_w = title_art.get_window_extent(renderer=rend).width / (W / 72 * fig.dpi)
    except Exception:                        # backend senza renderer: si torna alla stima
        title_w = fx(fontsize * 0.62 * len(title))
    box_w = max(lw + tick + gap + label_w + fx(8), title_w + fx(5))
    fig.patches.append(plt.Rectangle((lx - fx(5), ly - fy(fontsize * 0.9)),
                                     box_w, lh + fy(fontsize * 3.2),
                                     transform=fig.transFigure, fill=False,
                                     edgecolor='black', lw=0.8, zorder=4))
    for i, b in enumerate(bounds):
        y = ly + i * lh / NBANDS
        fig.text(lx + lw + tick + gap, y, fmt.format(b), fontsize=fontsize, va='center',
                 ha='left', family=FONT, zorder=6)
        fig.add_artist(plt.Line2D([lx + lw, lx + lw + tick], [y, y],
                                  transform=fig.transFigure, color='black', lw=0.7,
                                  zorder=6))


def axis_triad(fig, ax, scale, xy=(0.30, 0.13), length=0.052, figsize=(7.2, 3.0)) -> None:
    """Triade x/y/z proiettata con la stessa vista, come in Abaqus.

    Le direzioni si prendono da uno spostamento PICCOLO attorno al centro della scena, non
    dai vettori interi degli assi: in proiezione prospettica la direzione proiettata dipende
    da dove sta il punto, e usare (LX,0,0) dava una triade che contraddiceva il disegno --
    X puntava in alto mentre il lato lungo del pannello stava orizzontale.
    """
    M = ax.get_proj()
    c = np.array([scale[0] / 2, scale[1] / 2, 0.0])
    eps = 0.01 * max(scale[0], scale[1])
    o = np.array(proj3d.proj_transform(*c, M)[:2])
    x0, y0 = xy
    ar = figsize[0] / figsize[1]
    # Le etichette stanno a distanze DIVERSE dall'origine (Z piu' fuori, Y piu' dentro): con
    # la vista quasi dall'alto della figura modale, Y e Z si proiettano a una decina di gradi
    # l'uno dall'altro e con lo stesso raggio i due testi finivano stampati uno sull'altro.
    for name, vec, col, lab_r in (('X', c + [eps, 0, 0], (0.85, 0, 0), 1.45),
                                  ('Y', c + [0, eps, 0], (0, 0.60, 0), 1.30),
                                  ('Z', c + [0, 0, eps], (0, 0, 0.85), 2.25)):
        p = np.array(proj3d.proj_transform(*vec, M)[:2]) - o
        d = p / np.linalg.norm(p)
        fig.add_artist(plt.matplotlib.patches.FancyArrow(
            x0, y0, d[0] * length, d[1] * length * ar, transform=fig.transFigure,
            color=col, width=0.004, head_width=0.014, head_length=0.016,
            length_includes_head=True, zorder=7))
        fig.text(x0 + d[0] * length * lab_r, y0 + d[1] * length * lab_r * ar, name,
                 fontsize=8, weight='bold', ha='center', va='center', family=FONT,
                 zorder=7)


def draw_deformed_surface(fig, ax, xy, U, mag, ampl=AMPL, elev=ELEV, azim=AZIM,
                          focal=FOCAL, subdiv=3, edge=True, roll=0.0) -> None:
    """Superficie deformata di un modo, colorata a bande sulla magnitudine normalizzata.

    `subdiv` raffina la maglia di DISEGNO (non quella FE): colorando per faccia, i bordi
    di banda seguirebbero altrimenti i triangoli, e si vedrebbero seghettati.

    Vista e focale non sono scelte a occhio: vengono dal fit sui quattro vertici del
    pannello nella figura originale del coautore (vedi ELEV/AZIM/FOCAL).
    """
    from matplotlib.tri import Triangulation, UniformTriRefiner, LinearTriInterpolator

    mn = mag / mag.max()
    LX, LY = xy[:, 0].max(), xy[:, 1].max()
    scale = ampl * LX / np.abs(U).max()
    P = np.column_stack([xy[:, 0] + scale * U[:, 0], xy[:, 1] + scale * U[:, 1],
                         scale * U[:, 2]])
    tri = Triangulation(xy[:, 0], xy[:, 1])
    ref = UniformTriRefiner(tri)
    rtri, rvals = ref.refine_field(mn, triinterpolator=LinearTriInterpolator(tri, mn),
                                   subdiv=subdiv)
    rP = [ref.refine_field(P[:, k], triinterpolator=LinearTriInterpolator(tri, P[:, k]),
                           subdiv=subdiv)[1] for k in range(3)]
    P = np.column_stack(rP)
    faces = P[rtri.triangles]
    cols, _ = band_colors(rvals[rtri.triangles].mean(1), 0.0, 1.0)
    ax.add_collection3d(Poly3DCollection(faces, facecolors=cols, edgecolors=cols,
                                         linewidths=0.25, shade=False, antialiased=False))
    zr = max(np.abs(P[:, 2]).max(), 1e-9)
    ax.set_xlim(0, LX)
    ax.set_ylim(0, LY)
    ax.set_zlim(-zr, zr)
    ax.set_box_aspect((LX, LY, 2 * zr))          # proporzioni fisiche vere
    ax.set_proj_type('persp', focal_length=focal)
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.set_axis_off()
    if edge:
        _outline(ax, xy, U, scale, LX, LY)
    return LX, LY, zr


def _outline(ax, xy, U, scale, LX, LY, n=60):
    """Spigolo esterno del modello, come il feature edge che Abaqus disegna."""
    def at(px, py):
        i = int(np.argmin((xy[:, 0] - px) ** 2 + (xy[:, 1] - py) ** 2))
        return (xy[i, 0] + scale * U[i, 0], xy[i, 1] + scale * U[i, 1], scale * U[i, 2])
    t = np.linspace(0, 1, n)
    for a, b in (((0, 0), (LX, 0)), ((LX, 0), (LX, LY)),
                 ((LX, LY), (0, LY)), ((0, LY), (0, 0))):
        pts = [at(a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])) for s in t]
        P = np.array(pts)
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color='black', lw=0.5, zorder=10)


def _refine2d(A, U, V, k):
    """Bilineare di A(len(U), len(V)) su una griglia k volte piu' fitta."""
    u = np.linspace(U[0], U[-1], (len(U) - 1) * k + 1)
    v = np.linspace(V[0], V[-1], (len(V) - 1) * k + 1)
    tmp = np.empty((len(u), len(V)))
    for j in range(len(V)):
        tmp[:, j] = np.interp(u, U, A[:, j])
    out = np.empty((len(u), len(v)))
    for i in range(len(u)):
        out[i, :] = np.interp(v, V, tmp[i, :])
    return out, u, v


def _face_quads(A, U, V, place, k):
    F, u, v = _refine2d(A, U, V, k)
    quads, vals = [], []
    for i in range(len(u) - 1):
        for j in range(len(v) - 1):
            quads.append([place(u[i], v[j]), place(u[i + 1], v[j]),
                          place(u[i + 1], v[j + 1]), place(u[i], v[j + 1])])
            vals.append(0.25 * (F[i, j] + F[i + 1, j] + F[i + 1, j + 1] + F[i, j + 1]))
    return quads, vals


def draw_solid_block(ax, field, X, Y, Z, elev=24, azim=-62, focal=2.5, k=6, pad=0.10):
    """Blocco solido in assonometria, facce esterne colorate a bande sul campo.

    Il fondo non si disegna: da questa vista non e' visibile e vale un quinto dei poligoni.

    `pad` allarga i limiti degli assi attorno al centro del modello, cioe' ALLONTANA la
    camera: il blocco resta identico, rimpicciolisce dentro il riquadro. Serve perche'
    mplot3d **ritaglia al proprio riquadro, che e' un QUADRATO**: `ax.get_window_extent()`
    non e' il rect che si passa a `add_axes` -- mplot3d ne prende il lato minore e lo centra.
    Il nostro blocco, largo e basso, proiettato risultava il **2,9% piu' largo del quadrato**
    e le due estremita' venivano tagliate LI' DENTRO, a meta' tela: invisibili a un controllo
    sull'inchiostro al bordo della figura, che infatti per due giri ha detto "pulito" mentre
    gli angoli sparivano davvero (rilievo dell'utente, 2026-07-29, tre volte prima che
    guardassimo nel posto giusto). Con `pad=0.10` il blocco occupa il 93,5% del lato e
    respira il 3,2% per parte. Il gate corrispondente e' in `fig3_candidate.py`: disegna
    con e senza `set_clip_on`, e i due inchiostri devono coincidere ESATTAMENTE.
    Le viste modali non hanno il problema (una superficie piana ci sta dentro): verificato
    con lo stesso confronto, zero pixel di differenza.
    """
    nx, ny, nz = len(X) - 1, len(Y) - 1, len(Z) - 1
    quads, vals = [], []
    for A, U, V, place in (
            (field[:, :, nz], X, Y, lambda a, b: (a, b, Z[-1])),        # superficie esterna
            (field[:, 0, :], X, Z, lambda a, b: (a, Y[0], b)),          # fianco y=0
            (field[:, ny, :], X, Z, lambda a, b: (a, Y[-1], b)),        # fianco y=LY
            (field[nx, :, :], Y, Z, lambda a, b: (X[-1], a, b)),        # estremita' caricata
            (field[0, :, :], Y, Z, lambda a, b: (X[0], a, b))):         # incastro
        q, v = _face_quads(A, U, V, place, k)
        quads += q
        vals += v
    vmin, vmax = float(field.min()), float(field.max())
    cols, bounds = band_colors(vals, vmin, vmax)
    ax.add_collection3d(Poly3DCollection(quads, facecolors=cols, edgecolors=cols,
                                         linewidths=0.1, shade=False, antialiased=False,
                                         zsort='average'))
    # limiti allargati di `pad` attorno al centro: proporzioni intatte (i tre intervalli
    # crescono dello stesso fattore, quindi `set_box_aspect` resta quello vero), oggetto piu'
    # piccolo dentro il quadrato del riquadro
    for lim, hi in ((ax.set_xlim, X[-1]), (ax.set_ylim, Y[-1]), (ax.set_zlim, Z[-1])):
        lim(hi / 2 - hi * (1 + pad) / 2, hi / 2 + hi * (1 + pad) / 2)
    ax.set_box_aspect((X[-1], Y[-1], Z[-1]))
    ax.set_proj_type('persp', focal_length=focal)   # come le viste modali, e come Abaqus
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    # Sfondo trasparente: i due pannelli della figura degli sforzi hanno assi SOVRAPPOSTI
    # (serve a farli grandi senza allontanarli), e il riquadro opaco del secondo tagliava
    # in orizzontale il blocco del primo.
    ax.patch.set_visible(False)
    return bounds
