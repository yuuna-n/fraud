"""
Încărcarea datelor și verificarea calității lor (§2.1, §2.3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

# Coloanele minime de care are nevoie sistemul pentru a calcula caracteristicile.
COLOANE_NECESARE = [
    C.COL_TIMP, C.COL_CARD, "merchant", "category", C.COL_SUMA,
    "gender", "job", "city", "state", "city_pop",
    "lat", "long", "merch_lat", "merch_long",
]


def citeste_csv(sursa) -> pd.DataFrame:
    """Citește un fișier CSV (cale sau obiect încărcat din interfață)."""
    return pd.read_csv(sursa, dtype={C.COL_CARD: "int64"}, low_memory=False)


def verifica_coloane(df: pd.DataFrame, cu_eticheta: bool = False) -> list[str]:
    """Întoarce lista coloanelor necesare care lipsesc din tabel."""
    necesare = COLOANE_NECESARE + ([C.COL_ETICHETA] if cu_eticheta else [])
    return [c for c in necesare if c not in df.columns]


def curata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Curățarea de bază, independentă de setul de antrenare:
      * conversia datelor calendaristice citite ca șiruri de caractere;
      * eliminarea duplicatelor;
      * excluderea coloanelor fără valoare predictivă;
      * calculul vârstei din data nașterii (dacă există coloana dob).
    Funcția poate fi aplicată de mai multe ori pe același tabel.
    Completarea valorilor lipsă (mediana / „necunoscut") se face în
    `preprocessing.Preprocesor`, deoarece mediana se învață doar pe train.
    """
    df = df.copy()
    lipsa = verifica_coloane(df)
    if lipsa:
        raise ValueError("Lipsesc coloanele: " + ", ".join(lipsa))

    if "trans_num" in df.columns:
        df = df.drop_duplicates(subset="trans_num")
    else:
        df = df.drop_duplicates()

    df[C.COL_TIMP] = pd.to_datetime(df[C.COL_TIMP], errors="coerce")
    df = df.dropna(subset=[C.COL_TIMP, C.COL_CARD, C.COL_SUMA])
    if "dob" in df.columns:
        dob = pd.to_datetime(df["dob"], errors="coerce")
        df["age"] = ((df[C.COL_TIMP] - dob).dt.days / 365.25).round(1)
    elif "age" not in df.columns:
        df["age"] = np.nan  # se completează ulterior cu mediana din train

    df = df.drop(columns=[c for c in C.COLOANE_EXCLUSE + ["dob", "zip"] if c in df.columns])
    return df.reset_index(drop=True)


def raport_calitate(df: pd.DataFrame) -> pd.DataFrame:
    """Tabel cu valorile lipsă și tipul fiecărei coloane (pentru interfață)."""
    return pd.DataFrame({
        "tip": df.dtypes.astype(str),
        "lipsă": df.isna().sum(),
        "lipsă (%)": (df.isna().mean() * 100).round(2),
        "valori distincte": df.nunique(),
    })


def incarca_seturi(fisier_train: Path = C.FISIER_TRAIN, fisier_test: Path = C.FISIER_TEST,
                   esantion_clienti: int | None = None):
    """Încarcă fraudTrain.csv și fraudTest.csv și aplică curățarea."""
    train = curata(citeste_csv(fisier_train))
    test = curata(citeste_csv(fisier_test))
    if esantion_clienti:
        # Pentru încercări rapide: se păstrează doar o parte din carduri (istoricul rămâne complet).
        carduri = train[C.COL_CARD].drop_duplicates().sample(
            min(esantion_clienti, train[C.COL_CARD].nunique()), random_state=C.SEED)
        train = train[train[C.COL_CARD].isin(carduri)].reset_index(drop=True)
        test = test[test[C.COL_CARD].isin(carduri)].reset_index(drop=True)
    return train, test
