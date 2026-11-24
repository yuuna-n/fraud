"""
Împărțirea datelor (§2.5, tabelul 2.5).

  * Antrenare – prima parte cronologică din fraudTrain.csv;
  * Validare  – ultima parte cronologică din fraudTrain.csv;
  * Testare   – fraudTest.csv, folosit o singură dată, la evaluarea finală.

Împărțirea aleatorie este evitată: modelul ar învăța din tranzacții „viitoare"
și ar fi evaluat pe unele trecute, ceea ce ar umfla artificial metricile.
"""

from __future__ import annotations

import pandas as pd

from . import config as C


def imparte_cronologic(df: pd.DataFrame, pondere_validare: float = C.PONDERE_VALIDARE):
    """Separă ultima fracțiune (în timp) a tabelului ca set de validare."""
    df = df.sort_values(C.COL_TIMP, kind="mergesort").reset_index(drop=True)
    moment = df[C.COL_TIMP].quantile(1 - pondere_validare)
    train = df[df[C.COL_TIMP] < moment].reset_index(drop=True)
    val = df[df[C.COL_TIMP] >= moment].reset_index(drop=True)
    return train, val, moment


def descriere_seturi(seturi: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Tabel rezumat: perioadă, număr de tranzacții, ponderea fraudelor."""
    randuri = []
    for nume, d in seturi.items():
        randuri.append({
            "set": nume,
            "de la": d[C.COL_TIMP].min(),
            "până la": d[C.COL_TIMP].max(),
            "tranzacții": len(d),
            "fraude": int(d[C.COL_ETICHETA].sum()),
            "ponderea fraudelor (%)": round(100 * d[C.COL_ETICHETA].mean(), 3),
        })
    return pd.DataFrame(randuri)
