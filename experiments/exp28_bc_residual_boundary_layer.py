"""Il residuo della condizione di superficie libera diverge, o si sposta soltanto?

Nasce il 2026-08-26, quando lo sweep di exp10 rigenerato col parser .frd corretto ha fatto scattare
il controllo interno del modulo: `bc_resid` -- il massimo di |tau_xz| sulle facce esterne, che il
recupero per equilibrio dovrebbe lasciare a zero -- CRESCE col raffinamento invece di calare, da
12,6 a 74,2 MPa. Sembrava un degrado del recupero, cioe' del metodo su cui poggia tutto il blocco
interlaminare.

Non lo e'. Il massimo sta, a ogni mesh, nell'angolo dove l'incastro incontra un bordo libero, e
cresce perche' il raffinamento risolve sempre meglio un campo singolare. Misurato a distanza
FISICA costante dai bordi -- invece che a un numero costante di NODI, che col raffinamento si
avvicina alla singolarita' -- il residuo non cresce affatto:

    margine  0 mm : 12,63 -> 74,24   (~h^-1,28, diverge)
    margine  5 mm :  3,43 ->  1,96   (cala)
    margine 10 mm :  0,80 ->  0,71   (piatto)
    margine 20 mm :  0,23 ->  0,06   (~h^+0,93, convergenza del prim'ordine)

Cioe' il recupero e' consistente ovunque il campo sia liscio, ed e' il criterio mediato a ereditare
la sensibilita' di strato limite, perche' la sua banda di media sta sul bordo libero per
costruzione. Questo esperimento e' la misura, non il ragionamento: rieseguilo se qualcuno rimette
mano allo smoothing, ai gradienti o all'integrazione attraverso lo spessore.

⚠️ Trappola incontrata scrivendo la diagnosi, e vale in generale: una prima versione di questa
sonda smussava anche le righe di bordo (con `np.roll`, quindi con avvolgimento periodico) mentre
il modulo le lascia intatte. Con quel trattamento il residuo sembrava convergere, e la conclusione
-- "e' un artefatto dello smoothing a finestra decrescente" -- era SBAGLIATA e sarebbe finita nel
paper. Una sonda che non riproduce il numero del modulo non sta misurando il modulo: il primo
controllo e' che la colonna a finestra 3x3 dia esattamente i bc_resid di exp10.

Uso:  CCX_BIN=ccx_2.21 PYTHONPATH=$PWD python3 -m experiments.exp28_bc_residual_boundary_layer
"""
import os, subprocess, tempfile, shutil, math
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fe.interlaminar as il

SEQ=[0,45,-45,90]*6; AXIAL,SIDE=10000.0,3000.0
MESHES=[(10,6),(20,10),(30,16),(40,20)]
MARGINI_MM=[0.0, 5.0, 10.0, 20.0]

def solve(nx,ny):
    deck,nx,ny,nz = il.make_solid_deck(SEQ,AXIAL,SIDE,nx,ny)
    d=tempfile.mkdtemp()
    try:
        open(d+"/job.inp","w").write(deck)
        subprocess.run([il.CCX,"-i","job"],cwd=d,capture_output=True,text=True,timeout=3600,
                       env={**os.environ,"OMP_NUM_THREADS":"1"})
        return il._parse_stress_grid(open(d+"/job.frd").read(),nx,ny,nz),nx,ny,nz
    finally: shutil.rmtree(d,ignore_errors=True)

def smooth(a):
    b=a.copy(); b[1:-1,:,:]=(a[:-2,:,:]+a[1:-1,:,:]+a[2:,:,:])/3
    c=b.copy(); c[:,1:-1,:]=(b[:,:-2,:]+b[:,1:-1,:]+b[:,2:,:])/3
    return c

rows={m:[] for m in MARGINI_MM}
print(f"{'mesh':>8} " + " ".join(f"{'|tau| a >'+str(int(m))+'mm':>16}" for m in MARGINI_MM))
for nx,ny in MESHES:
    sig,nx,ny,nz=solve(nx,ny)
    dx,dy,dz = il.LX/nx, il.LY/ny, il.PLY_T
    sxx,sxy = smooth(sig[...,0]), smooth(sig[...,3])
    a=np.gradient(sxx,dx,axis=0)+np.gradient(sxy,dy,axis=1)
    txz=np.zeros_like(sxx)
    for k in range(1,nz+1):
        txz[:,:,k]=txz[:,:,k-1]-0.5*(a[:,:,k]+a[:,:,k-1])*dz
    top=np.abs(txz[:,:,nz])
    out=[]
    for m in MARGINI_MM:
        mi=int(math.ceil(m/dx)); mj=int(math.ceil(m/dy))
        sub=top[mi:top.shape[0]-mi or None, mj:top.shape[1]-mj or None]
        v=float(sub.max()) if sub.size else float('nan')
        rows[m].append(v); out.append(f"{v:16.2f}")
    print(f"{nx:3d}x{ny:<4d} " + " ".join(out))
print()
for m in MARGINI_MM:
    v=rows[m]
    print(f"margine {int(m):2d} mm: {v[0]:7.2f} -> {v[-1]:7.2f}   scaling ~ h^-{math.log(v[-1]/v[0])/math.log(4):+.2f}")
