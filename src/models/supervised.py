"""
Clasificarea supravegheată cu XGBoost (§1.2, §2.4).

XGBoost construiește secvențial arbori de decizie, fiecare corectând erorile
celor anteriori. Modelul învață din etichete combinația de criterii asociată
fraudei și produce o probabilitate de fraudă pentru fiecare tranzacție.
Scalarea nu este necesară (arborii sunt invarianți la transformări monotone).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from .. import config as C


def antreneaza_xgboost(X_train: pd.DataFrame, y_train, X_val: pd.DataFrame, y_val,
                       scale_pos_weight: float, parametri: dict | None = None) -> XGBClassifier:
    """Antrenează XGBoost cu oprire timpurie pe PR-AUC-ul setului de validare."""
    p = {**C.XGB_PARAMETRI, **(parametri or {})}
    model = XGBClassifier(scale_pos_weight=scale_pos_weight, **p)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def optimizeaza_xgboost(X_train, y_train, X_val, y_val, scale_pos_weight: float,
                        jurnal=print) -> tuple[XGBClassifier, dict, list[dict]]:
    """
    Optimizarea hiperparametrilor pe setul de validare (tabelul 2.5): se încearcă
    combinațiile din config.XGB_GRILA și se păstrează cea cu PR-AUC maxim.
    """
    rezultate, cel_mai_bun, parametri_buni, scor_bun = [], None, {}, -1.0
    for combinatie in C.XGB_GRILA:
        model = antreneaza_xgboost(X_train, y_train, X_val, y_val, scale_pos_weight, combinatie)
        scor = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
        rezultate.append({**combinatie, "arbori": int(model.best_iteration + 1), "pr_auc_validare": scor})
        jurnal(f"    {combinatie} -> PR-AUC validare = {scor:.4f}")
        if scor > scor_bun:
            cel_mai_bun, parametri_buni, scor_bun = model, combinatie, scor
    return cel_mai_bun, parametri_buni, rezultate


def scor_xgboost(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Probabilitatea de fraudă (0–1)."""
    return model.predict_proba(X)[:, 1]
