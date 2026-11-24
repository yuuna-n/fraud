"""
Tratarea dezechilibrului claselor (tabelul 2.2).

  * "ponderi"        – nu modifică datele; XGBoost penalizează mai mult erorile pe
                       clasa rară prin scale_pos_weight = nr. legitime / nr. fraude;
  * "subesantionare" – elimină aleatoriu tranzacții legitime;
  * "smote"          – generează fraude sintetice prin interpolare între vecini.

Rebalansarea se aplică NUMAI pe setul de antrenare; validarea și testarea
păstrează distribuția reală.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def pondere_clasa_pozitiva(y) -> float:
    """scale_pos_weight pentru XGBoost."""
    y = np.asarray(y)
    poz = max(int(y.sum()), 1)
    return float((len(y) - poz) / poz)


def rebalanseaza(X: pd.DataFrame, y: pd.Series, metoda: str = C.DEZECHILIBRU_IMPLICIT):
    """
    Întoarce (X, y, scale_pos_weight) pregătite pentru antrenarea XGBoost.
    Pentru subeșantionare și SMOTE ponderea rămasă se recalculează pe datele noi.
    """
    if metoda == "ponderi":
        return X, y, pondere_clasa_pozitiva(y)

    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
    except ImportError as e:  # pragma: no cover
        raise ImportError("Pentru „subesantionare” / „smote” instalați imbalanced-learn.") from e

    if metoda == "subesantionare":
        s = RandomUnderSampler(sampling_strategy=C.RAPORT_SUBESANTIONARE, random_state=C.SEED)
    elif metoda == "smote":
        s = SMOTE(sampling_strategy=C.RAPORT_SMOTE, k_neighbors=5, random_state=C.SEED)
    else:
        raise ValueError(f"Metodă necunoscută: {metoda}")

    X_r, y_r = s.fit_resample(X, y)
    # după rebalansare clasele sunt încă inegale; ponderea compensează restul
    return X_r, y_r, pondere_clasa_pozitiva(y_r)
