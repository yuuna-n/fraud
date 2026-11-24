"""
Evaluarea modelelor. Din cauza dezechilibrului extrem al claselor, acuratețea NU
se folosește (un model care declară totul „legitim" ar avea >99%). Se raportează:
precision, recall, F1-score, ROC-AUC, PR-AUC și matricea de confuzie.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)


def metrici(y, scor, prag: float, suma=None) -> dict:
    """Metricile la pragul dat; opțional și suma fraudelor prinse / ratate."""
    y = np.asarray(y).astype(int)
    pred = (np.asarray(scor) >= prag).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    doua_clase = len(np.unique(y)) == 2
    rez = {
        "prag": float(prag),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scor)) if doua_clase else float("nan"),
        "pr_auc": float(average_precision_score(y, scor)) if doua_clase else float("nan"),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "alerte": int(tp + fp),
        "rata_alerte": float((tp + fp) / len(y)),
    }
    if suma is not None:
        suma = np.asarray(suma, dtype=float)
        rez["suma_fraude"] = float(suma[y == 1].sum())
        rez["suma_prinsa"] = float(suma[(y == 1) & (pred == 1)].sum())
    return rez


def curbe(y, scor, puncte: int = 300) -> dict:
    """
    Curbele ROC și precision–recall, reeșantionate pe o grilă fixă, pentru grafice.
      * PR: precizia interpolată (maximul preciziei la recall ≥ r) pe o grilă de recall;
      * ROC: rata fraudelor prinse pe o grilă de rate ale alarmelor false, mai deasă lângă 0.
    """
    fpr, tpr, _ = roc_curve(y, scor)
    grila_fpr = np.unique(np.concatenate([[0.0], np.geomspace(1e-5, 1, puncte)]))
    tpr_g = np.interp(grila_fpr, fpr, tpr)

    prec, rec, _ = precision_recall_curve(y, scor)
    ordine = np.argsort(rec)
    rec_s, prec_s = rec[ordine], prec[ordine]
    prec_interp = np.maximum.accumulate(prec_s[::-1])[::-1]   # max. preciziei la recall >= r
    grila_rec = np.linspace(0, 1, puncte)
    idx = np.clip(np.searchsorted(rec_s, grila_rec, side="left"), 0, len(rec_s) - 1)
    return {"fpr": grila_fpr.round(6).tolist(), "tpr": tpr_g.round(5).tolist(),
            "precision": prec_interp[idx].round(5).tolist(), "recall": grila_rec.round(5).tolist()}


def tabel_comparativ(rezultate: dict[str, dict], denumiri: dict[str, str]) -> pd.DataFrame:
    """Tabel cu câte un rând pentru fiecare model."""
    t = pd.DataFrame(rezultate).T
    t.index = [denumiri.get(i, i) for i in t.index]
    return t
