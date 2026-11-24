"""
Explicabilitatea modelului XGBoost prin valori SHAP (§5.5).

XGBoost calculează nativ valorile SHAP exacte pentru arbori (algoritmul
TreeSHAP) prin opțiunea pred_contribs=True, deci nu e nevoie de o bibliotecă
suplimentară. Pentru fiecare tranzacție, valoarea SHAP a unei caracteristici
arată cât a împins acea caracteristică scorul spre „fraudă" (pozitiv) sau spre
„legitim" (negativ), în unități log-odds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config as C


def valori_shap(model, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Întoarce (contribuțiile fiecărei caracteristici, valoarea de bază)."""
    contributii = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return pd.DataFrame(contributii[:, :-1], columns=X.columns, index=X.index), contributii[:, -1]


def _grupeaza_one_hot(serie: pd.Series) -> pd.Series:
    """Însumează coloanele one-hot (ex. category_*) într-o singură caracteristică."""
    def grup(col: str) -> str:
        for baza in C.CATEG_ONE_HOT:
            if col.startswith(baza + "_"):
                return baza
        return col
    return serie.groupby(serie.index.map(grup)).sum()


def importanta_globala(model, X: pd.DataFrame, n: int = 5000, seed: int = C.SEED) -> pd.Series:
    """Media |SHAP| pe un eșantion – importanța globală a caracteristicilor."""
    if len(X) > n:
        X = X.sample(n, random_state=seed)
    shap_df, _ = valori_shap(model, X)
    imp = _grupeaza_one_hot(shap_df.abs().mean())
    return imp.sort_values(ascending=False)


def explicatie_tranzactie(model, x: pd.DataFrame) -> pd.Series:
    """Contribuțiile SHAP pentru o singură tranzacție (rând), grupate și sortate."""
    shap_df, _ = valori_shap(model, x.iloc[[0]])
    s = _grupeaza_one_hot(shap_df.iloc[0])
    return s.reindex(s.abs().sort_values(ascending=False).index)


def denumire(col: str) -> str:
    return C.DENUMIRI.get(col, {"category": "Categoria comerțului", "gender": "Sexul"}.get(col, col))
