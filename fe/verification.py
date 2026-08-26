"""Il verification layer come COMPONENTE: uno screen deterministico su un design consegnato.

Perche' esiste (2026-08-26). Il manoscritto afferma che «every design an optimiser returns is
screened by a deterministic rule layer against physical invariants before it is accepted» e che
l'esito e' «a deterministic verdict (ok, suspect or rejected)». Fino a oggi **quel layer non
esisteva**: i controlli vivevano sparsi dentro `optimise_case_ga`, `exp13`, `exp15` e `exp18`, e
venivano applicati a mano ai design consegnati. Per un paper il cui contributo dichiarato numero
due e' proprio questo layer, era il divario peggiore possibile fra cio' che il testo afferma e cio'
che il codice contiene: un lettore che apre il bundle cerca questo file per primo.

I controlli **non sono riscritti qui**: sono gli stessi, spostati, cosi' che la campagna e lo screen
usino un solo codice e non due copie libere di divergere.

QUATTRO CONTROLLI SU CINQUE FAILURE MODE, e il quinto non e' un'omissione:

  1. chop a materiale debole      -> RISPOSTA AL LAYUP: due laminati che il chop rende
                                     indistinguibili ([0]_n contro [90]_n) devono dare fattori
                                     DIVERSI. Se coincidono, il modello non sta piu' vedendo il
                                     laminato.
  2. autovalore spurio nel solido -> INVARIANTE DI SCALA: BLF x |F| non dipende dal carico di
     sotto-risolto                  riferimento. Se disperde, il numero non e' l'autovalore del
                                     problema posto.
  4. secondo modo al posto del     -> lo STESSO invariante di scala: quando il solutore cambia modo,
     primo                           BLF x |F| salta di colpo.
  5. il lettore del .frd scarta    -> COPERTURA: i record letti devono essere quanti i nodi attesi.
     record in silenzio              Implementato dentro `interlaminar._parse_stress_grid`, che
                                     alza; qui si espone come controllo a se'.

  3. singolarita' di bordo libero  -> **NON e' un controllo che scatta**, ed e' giusto che non lo
                                     sia: e' la ragione per cui il criterio di delaminazione e'
                                     mediato su una banda invece che puntuale. Si registra nel
                                     verdetto come `by_construction`, perche' un layer che
                                     dichiarasse cinque rilevatori su cinque mentirebbe di uno.

COSTO. L'invariante di scala richiede DUE solve di buckling per design invece di una: lo screen e'
piu' caro di una valutazione, ed e' il motivo per cui gira sui design consegnati e non dentro il
ciclo di ricerca.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

#: Soglie dei controlli. Sono deliberatamente larghe: devono separare un guasto da un risultato,
#: non misurare un'accuratezza. La dispersione dell'invariante misurata sul guscio validato e'
#: 9e-5 %, quella sul crop patologico 9e4 %: fra i due c'e' spazio per otto ordini di grandezza.
INVARIANT_TOL_PCT = 1.0      # oltre l'1% il fattore non e' piu' lo stesso autovalore

#: Separazione minima fra due layup che il chop renderebbe indistinguibili.
#: ⚠️ CALIBRATA SU MISURA, non scelta a occhio, e la prima versione era sbagliata: valeva 1.0 e
#: NON mordeva, perche' i due valori col chop ([0]_24+36weak = 0.95, [90]_24+36weak = 0.99, exp15)
#: distano gia' il 4%, cioe' passavano. Misurati il 2026-08-26 sul caso assiale i due laminati VERI
#: a 24 ply: [0]_24 = 0.7401 e [90]_24 = 1.1527, che distano il **35.8%**. Fra il 4% del guasto e il
#: 35.8% del sano c'e' un fattore nove, e la soglia sta in mezzo con margine da entrambe le parti.
#: NB: la separazione dipende dal caso di carico — su taglio puro due layup possono stare piu'
#: vicini. Per un caso diverso si rimisura e si passa `tol_pct`, non si tiene questo numero per fede.
LAYUP_TOL_PCT = 15.0
LAYUP_CALIBRAZIONE = {"caso": "c1_axial", "N": 24, "sano_pct": 35.8, "col_chop_pct": 4.0}


@dataclass
class Verdict:
    """L'esito dello screen. `esito` in {ok, suspect, rejected}."""
    esito: str
    motivi: list = field(default_factory=list)
    misure: dict = field(default_factory=dict)
    by_construction: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.esito == "ok"

    def as_dict(self) -> dict:
        return asdict(self)


def _peggiore(a: str, b: str) -> str:
    ordine = {"ok": 0, "suspect": 1, "rejected": 2}
    return a if ordine[a] >= ordine[b] else b


def check_layup_response(bf_a: float, bf_b: float, tol_pct: float = LAYUP_TOL_PCT) -> tuple:
    """Controllo 1. Due laminati che il chop renderebbe identici devono dare fattori BEN diversi.

    Torna (esito, misura). Non decide quale sia «giusto»: decide se il modello sta ancora
    guardando il laminato. La soglia e' calibrata (vedi LAYUP_CALIBRAZIONE): col chop i due
    fattori collassano entrambi verso ~1 e distano pochi punti percentuali, senza il chop distano
    decine.
    """
    if bf_a <= 0 or bf_b <= 0:
        return "rejected", {"delta_pct": None, "nota": "un fattore non positivo: solve fallita"}
    delta = abs(bf_a - bf_b) / max(abs(bf_a), abs(bf_b)) * 100
    esito = "ok" if delta > tol_pct else "rejected"
    return esito, {"bf_a": bf_a, "bf_b": bf_b, "delta_pct": round(delta, 4)}


def check_scale_invariant(pairs, tol_pct: float = INVARIANT_TOL_PCT) -> tuple:
    """Controlli 2 e 4. `pairs` = [(carico_di_riferimento, BLF), ...], almeno due.

    L'autovalore di buckling lineare scala come 1/|F|, quindi BLF x |F| e' invariante. Se disperde,
    o il numero non e' l'autovalore del problema posto (modo locale spurio), oppure il solutore ha
    restituito un modo diverso: sono i failure mode 2 e 4, e il controllo e' lo stesso.
    """
    if len(pairs) < 2:
        return "suspect", {"nota": "meno di due punti: l'invariante non e' verificabile"}
    prodotti = [abs(f) * abs(blf) for f, blf in pairs]
    if any(p <= 0 for p in prodotti):
        return "rejected", {"prodotti": prodotti, "nota": "prodotto non positivo: solve fallita"}
    lo, hi = min(prodotti), max(prodotti)
    disp = (hi - lo) / lo * 100
    esito = "ok" if disp <= tol_pct else "rejected"
    return esito, {"prodotti": [round(p, 6) for p in prodotti],
                   "dispersione_pct": round(disp, 6)}


def check_coverage(letti: int, attesi: int) -> tuple:
    """Controllo 5. Il lettore deve riempire ogni nodo della griglia che dichiara.

    E' il controllo che il 26/08 ha scoperto un campo di sforzo riempito al 15,9%, con il resto a
    zero: uno zero e' uno sforzo legittimo, quindi l'assenza non si vede sul valore. Si vede sul
    CONTEGGIO, ed e' il motivo per cui questo controllo esiste in questa forma.
    """
    if attesi <= 0:
        return "suspect", {"nota": "nessun nodo atteso: non c'e' niente da verificare"}
    frazione = letti / attesi
    esito = "ok" if letti == attesi else "rejected"
    return esito, {"letti": letti, "attesi": attesi, "frazione": round(frazione, 6)}


def check_constraints(metriche: dict, limiti: dict) -> tuple:
    """Il gate sui vincoli fisici, che e' l'altra meta' del layer.

    ⚠️ Un vincolo NON VERIFICATO non e' un vincolo soddisfatto: se una misura manca, l'esito e'
    `suspect`, mai `ok`. E' esattamente il tranello del caso assiale della campagna, dove un design
    raggiungeva il target di buckling mentre la frequenza non era stata guardata.
    """
    motivi, esito = [], "ok"
    for nome, limite in limiti.items():
        valore = metriche.get(nome)
        if valore is None or (isinstance(valore, float) and math.isnan(valore)):
            esito = _peggiore(esito, "suspect")
            motivi.append(f"{nome}: non verificato")
            continue
        verso = limite.get("verso", "max")
        soglia = limite["soglia"]
        viola = valore > soglia if verso == "max" else valore < soglia
        if viola:
            esito = _peggiore(esito, "rejected")
            motivi.append(f"{nome}: {valore:g} viola il limite ({verso} {soglia:g})")
    return esito, motivi


def screen(*, layup_pair=None, scale_pairs=None, coverage=None,
           metriche=None, limiti=None) -> Verdict:
    """Lo screen completo. Ogni argomento assente e' un controllo NON eseguito, e si vede.

    Un controllo non eseguito non alza il verdetto a `rejected`, ma lo tiene lontano da `ok`: il
    verdetto dice cosa e' stato guardato, non solo cosa e' passato. Un layer che restituisse `ok`
    per un design su cui non ha girato niente sarebbe peggio di nessun layer.
    """
    v = Verdict(esito="ok", by_construction=[
        "free-edge singularity: il criterio di delaminazione e' mediato su una banda, non "
        "puntuale; non e' un rilevatore a runtime ma una scelta di criterio"])

    if layup_pair is not None:
        e, m = check_layup_response(*layup_pair)
        v.misure["layup_response"] = m
        v.esito = _peggiore(v.esito, e)
        if e != "ok":
            v.motivi.append("risposta al layup assente: fattore indipendente dal laminato "
                            "(chop a materiale debole)")
    else:
        v.esito = _peggiore(v.esito, "suspect")
        v.motivi.append("risposta al layup: NON verificata")

    if scale_pairs is not None:
        e, m = check_scale_invariant(scale_pairs)
        v.misure["scale_invariant"] = m
        v.esito = _peggiore(v.esito, e)
        if e != "ok":
            v.motivi.append("invariante di scala violato: il fattore non e' l'autovalore del "
                            "problema posto, o il solutore ha cambiato modo")
    else:
        v.esito = _peggiore(v.esito, "suspect")
        v.motivi.append("invariante di scala: NON verificato")

    if coverage is not None:
        e, m = check_coverage(*coverage)
        v.misure["coverage"] = m
        v.esito = _peggiore(v.esito, e)
        if e != "ok":
            v.motivi.append("copertura incompleta: il lettore ha lasciato nodi non riempiti")
    else:
        v.esito = _peggiore(v.esito, "suspect")
        v.motivi.append("copertura del campo: NON verificata")

    if metriche is not None and limiti:
        e, motivi = check_constraints(metriche, limiti)
        v.misure["constraints"] = {"metriche": metriche, "limiti": limiti}
        v.esito = _peggiore(v.esito, e)
        v.motivi.extend(motivi)
    else:
        v.esito = _peggiore(v.esito, "suspect")
        v.motivi.append("vincoli fisici: NON verificati")

    return v
