"""Il binario CalculiX: UN default, in un posto solo, piu' la sua risoluzione sul PATH.

Esiste per chiudere F12 dell'audit del 26/08/2026. Prima di allora il default viveva in tre
posti con tre valori diversi -- `/opt/homebrew/bin/ccx` in fe/plate_model.py, `ccx_2.21` in
optimisers/constrained_search.py e fe/interlaminar.py, e un terzo, `ccx`, ribattuto negli
esperimenti al momento di scrivere il campo di provenienza nei JSON. Due conseguenze, entrambe
silenziose: su una macchina con piu' versioni installate, guscio e plate_model potevano girare
DUE ccx diversi nella stessa campagna senza che nessun log lo dicesse; e il campo `ccx` dei
JSON poteva non essere il binario davvero eseguito (senza CCX_BIN settata, exp16 scriveva
"ccx" mentre constrained_search aveva usato "ccx_2.21"). E' lo stesso pattern -- default
divergenti fra moduli -- gia' pagato con LOAD_SCALE 1.0/0.44.

Il modulo e' volutamente senza dipendenze (nemmeno numpy): ogni modulo del bundle deve poterlo
importare senza tirarsi dietro nulla. Per questo il default NON sta in fe/ccx_runner.py, che
importa fe/plate_model.py e quindi numpy.
"""
from __future__ import annotations
import os
import shutil

#: Default storico dei moduli di campagna, cioe' quello con cui sono stati prodotti i numeri
#: pubblicati (CalculiX 2.21). Si sovrascrive con la variabile d'ambiente CCX_BIN.
DEFAULT = "ccx_2.21"

CCX = os.environ.get("CCX_BIN", DEFAULT)


def resolved() -> str:
    """Il path assoluto del binario che verra' davvero eseguito, o il nome se non e' sul PATH.

    E' questo che va registrato come provenienza nei JSON, non il nome richiesto: se il nome
    non si risolve, il campo lo dice invece di far credere a un binario che non c'e'.
    """
    return shutil.which(CCX) or f"{CCX} (NON TROVATO sul PATH)"
