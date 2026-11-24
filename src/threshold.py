"""
Alegerea pragului de decizie (§2.4).

Un prag scăzut prinde mai multe fraude, dar produce mai multe alarme false;
unul ridicat face invers. Pragul se stabilește pe setul de VALIDARE, după una
din metodele:
  * "cost"   – minimizează  FN · cost_fraudă_ratată + FP · cost_alarmă_falsă
               (asimetria costurilor erorilor);
  * "f1"     – maximizează F1-score;
  * "recall" – cel mai mare prag care atinge un recall-țintă.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve

from . import config as C


def tabel_praguri(y, scor):
    """Pentru fiecare prag candidat: TP, FP, FN, precision, recall."""
    y = np.asarray(y)
    precizie, recall, praguri = precision_recall_curve(y, scor)
    precizie, recall = precizie[:-1], recall[:-1]   # ultimul punct nu are prag asociat
    poz = y.sum()
    tp = recall * poz
    with np.errstate(divide="ignore", invalid="ignore"):
        fp = np.where(precizie > 0, tp / precizie - tp, 0)
    fn = poz - tp
    return praguri, precizie, recall, tp, fp, fn


def alege_prag(y, scor, metoda: str = C.METODA_PRAG,
               cost_fn: float = C.COST_FRAUDA_RATATA, cost_fp: float = C.COST_ALARMA_FALSA,
               recall_tinta: float = C.RECALL_TINTA) -> float:
    praguri, precizie, recall, tp, fp, fn = tabel_praguri(y, scor)
    if len(praguri) == 0:
        return 0.5
    if metoda == "cost":
        i = int(np.argmin(fn * cost_fn + fp * cost_fp))
    elif metoda == "f1":
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = np.nan_to_num(2 * precizie * recall / (precizie + recall))
        i = int(np.argmax(f1))
    elif metoda == "recall":
        ok = np.where(recall >= recall_tinta)[0]
        i = int(ok[-1]) if len(ok) else 0
    else:
        raise ValueError(metoda)
    return float(praguri[i])
