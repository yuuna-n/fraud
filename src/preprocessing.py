"""
Preprocesarea datelor (§2.3): valori lipsă, codificarea variabilelor categoriale
și scalarea.

Toți parametrii (mediane, categorii, rate țintă, frecvențe, medii și abateri
pentru scalare) se învață EXCLUSIV pe setul de antrenare și se aplică apoi
neschimbați pe validare, testare și pe datele noi din interfață. Astfel nu
există scurgere de informație din setul de testare.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, TargetEncoder

from . import config as C

NECUNOSCUT = "necunoscut"


class Preprocesor:
    """Transformă tabelul cu caracteristici în matrice numerice pentru modele."""

    def __init__(self) -> None:
        self.mediane: dict[str, float] = {}
        self.categorii_one_hot: dict[str, list[str]] = {}
        self.mapari_ordinale: dict[str, dict[str, int]] = {}
        self.frecvente: dict[str, dict[str, float]] = {}
        self.encoder_tinta: TargetEncoder | None = None
        self.scaler = StandardScaler()
        self.coloane_supravegheat: list[str] = []
        self.coloane_nesupravegheat = list(C.CARACTERISTICI_NESUPRAVEGHEATE)

    # ------------------------------------------------------------------ util
    @property
    def coloane_numerice(self) -> list[str]:
        return C.NUMERICE_INITIALE + C.CARACTERISTICI_DERIVATE

    @property
    def coloane_categoriale(self) -> list[str]:
        return C.CATEG_ONE_HOT + C.CATEG_ORDINAL + C.CATEG_TARGET + C.CATEG_FRECVENTA

    def _completeaza_lipsa(self, df: pd.DataFrame) -> pd.DataFrame:
        """Numerice lipsă -> mediana din train; categoriale lipsă -> „necunoscut"."""
        df = df.copy()
        for c in self.coloane_numerice:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(self.mediane.get(c, 0.0))
        for c in self.coloane_categoriale:
            df[c] = df[c].astype("string").fillna(NECUNOSCUT).astype(str)
        return df

    # ------------------------------------------------------------------ fit
    def fit_transform(self, train: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Învață parametrii pe setul de antrenare și îl transformă."""
        self.mediane = {c: float(pd.to_numeric(train[c], errors="coerce").median())
                        for c in self.coloane_numerice}
        df = self._completeaza_lipsa(train)

        for c in C.CATEG_ONE_HOT:
            self.categorii_one_hot[c] = sorted(df[c].unique().tolist())
        for c in C.CATEG_ORDINAL:
            self.mapari_ordinale[c] = {v: i for i, v in enumerate(sorted(df[c].unique()))}
        for c in C.CATEG_FRECVENTA:
            self.frecvente[c] = df[c].value_counts(normalize=True).to_dict()

        # Rata țintei: pe train se folosește validarea încrucișată internă a
        # TargetEncoder (cross-fitting), ca un rând să nu-și „vadă" propria etichetă.
        self.encoder_tinta = TargetEncoder(
            target_type="binary", cv=StratifiedKFold(5, shuffle=True, random_state=C.SEED))
        tinta = self.encoder_tinta.fit_transform(df[C.CATEG_TARGET], df[C.COL_ETICHETA])

        X = self._construieste(df, tinta)
        self.coloane_supravegheat = X.columns.tolist()
        X_nes = self.scaler.fit_transform(self._pentru_nesupravegheat(df))
        return X, X_nes

    # ------------------------------------------------------------------ transform
    def transform(self, date: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Aplică parametrii învățați pe un set nou (validare, testare, interfață)."""
        if self.encoder_tinta is None:
            raise RuntimeError("Preprocesorul nu a fost antrenat (apelați fit_transform).")
        df = self._completeaza_lipsa(date)
        tinta = self.encoder_tinta.transform(df[C.CATEG_TARGET])
        X = self._construieste(df, tinta).reindex(columns=self.coloane_supravegheat, fill_value=0)
        X_nes = self.scaler.transform(self._pentru_nesupravegheat(df))
        return X, X_nes

    # ------------------------------------------------------------------ intern
    def _construieste(self, df: pd.DataFrame, tinta: np.ndarray) -> pd.DataFrame:
        parti = [df[self.coloane_numerice].astype(float)]

        for c, valori in self.categorii_one_hot.items():
            cat = pd.Categorical(df[c], categories=valori)
            parti.append(pd.get_dummies(cat, prefix=c, dtype=float).set_index(df.index))
        for c, mapare in self.mapari_ordinale.items():
            parti.append(df[c].map(mapare).fillna(-1).astype(float).rename(c).to_frame())
        for c, frecv in self.frecvente.items():
            parti.append(df[c].map(frecv).fillna(0.0).astype(float).rename(c).to_frame())
        parti.append(pd.DataFrame(tinta, columns=C.CATEG_TARGET, index=df.index))
        return pd.concat(parti, axis=1)

    def _pentru_nesupravegheat(self, df: pd.DataFrame) -> np.ndarray:
        """Caracteristicile comportamentale; variabilele asimetrice trec prin log."""
        X = df[self.coloane_nesupravegheat].astype(float).copy()
        for c in ["amt_ratio_avg", "time_since_last", "avg_amt_30d", "distance_km"]:
            X[c] = np.log1p(X[c].clip(lower=0))
        return X.to_numpy()
