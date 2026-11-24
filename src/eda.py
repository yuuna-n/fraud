"""
Analiza exploratorie a datelor (§2.2): statistici agregate folosite de
interfață pentru figurile 2.1–2.4, matricea de corelații și valorile extreme (IQR).
Statisticile se calculează o dată, la antrenare, și se salvează – interfața nu
mai trebuie să încarce cele ~1,3 mil. de rânduri.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def statistici_eda(df: pd.DataFrame) -> dict:
    y = df[C.COL_ETICHETA].astype(int)
    n = len(df)

    # Figura 2.1 – distribuția claselor
    clase = {"legitime": int((y == 0).sum()), "frauduloase": int((y == 1).sum())}

    # Figura 2.2 – distribuția sumelor pe scară logaritmică, normalizată pe clasă
    margini = np.logspace(0, np.log10(max(df[C.COL_SUMA].max(), 10) + 1), 45)
    hist = {}
    for eticheta, nume in [(0, "legitime"), (1, "frauduloase")]:
        val = df.loc[y == eticheta, C.COL_SUMA]
        h, _ = np.histogram(val, bins=margini)
        hist[nume] = (h / max(h.sum(), 1)).round(5).tolist()

    # Figura 2.3 – ponderea fraudelor pe ore
    ore = df.groupby(df[C.COL_TIMP].dt.hour)[C.COL_ETICHETA].agg(["mean", "size"])
    ore = ore.reindex(range(24), fill_value=0)

    # Figura 2.4 – ponderea fraudelor pe categorii
    cat = df.groupby("category")[C.COL_ETICHETA].agg(["mean", "size"]).sort_values("mean")

    # Sume medii pe clasă
    sume = df.groupby(y)[C.COL_SUMA].agg(["mean", "median"]).rename(index={0: "legitime", 1: "frauduloase"})

    # Corelații între variabilele numerice (inițiale + derivate, dacă există)
    num = [c for c in ["amt", "city_pop", "age"] + C.CARACTERISTICI_DERIVATE + [C.COL_ETICHETA]
           if c in df.columns]
    corelatii = df[num].sample(min(n, 200_000), random_state=C.SEED).corr().round(3)

    # Valori extreme ale sumei – metoda intervalului intercuartilic (nu se elimină!)
    q1, q3 = df[C.COL_SUMA].quantile([0.25, 0.75])
    lim = q3 + 1.5 * (q3 - q1)
    extreme = df[C.COL_SUMA] > lim

    return {
        "n": n,
        "perioada": [str(df[C.COL_TIMP].min()), str(df[C.COL_TIMP].max())],
        "carduri": int(df[C.COL_CARD].nunique()),
        "comercianti": int(df["merchant"].nunique()),
        "clase": clase,
        "hist_sume": {"margini": margini.round(3).tolist(), **hist},
        "sume_medii": sume.round(2).to_dict(orient="index"),
        "ore": {"ora": list(range(24)), "rata": ore["mean"].round(5).tolist(),
                "numar": ore["size"].astype(int).tolist()},
        "categorii": {"categorie": cat.index.tolist(), "rata": cat["mean"].round(5).tolist(),
                      "numar": cat["size"].astype(int).tolist()},
        "corelatii": {"coloane": corelatii.columns.tolist(), "valori": corelatii.values.tolist()},
        "iqr": {"limita_superioara": float(lim), "extreme": int(extreme.sum()),
                "fraude_printre_extreme": float(df.loc[extreme, C.COL_ETICHETA].mean()) if extreme.any() else 0.0},
    }
