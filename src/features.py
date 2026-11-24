"""
Construirea caracteristicilor derivate (tabelul 2.3, §2.3).

Principiu: o tranzacție este suspectă în raport cu comportamentul ANTERIOR al
aceluiași card. De aceea toate calculele:
  * se fac pe fiecare card (cc_num), în ordine cronologică;
  * folosesc exclusiv tranzacțiile de dinaintea momentului curent
    (tranzacția curentă nu intră niciodată în propriul profil).
Astfel aceleași funcții pot fi folosite și în timp real.

Implementarea este vectorizată (fără bucle pe clienți), ca să ruleze în câteva
secunde pe ~1,3 mil. de tranzacții.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

RAZA_PAMANT_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distanța pe sferă dintre două puncte, formula Haversine (2.2)."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlmb = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
    return 2 * RAZA_PAMANT_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _ferestre(chei: np.ndarray, secunde: np.ndarray, durata_s: int):
    """
    Pentru fiecare tranzacție întoarce intervalul [stânga, dreapta) de indici ai
    tranzacțiilor ACELUIAȘI card din fereastra [t - durata, t), fără tranzacția curentă.

    Datele sunt sortate după (card, timp). Cheia combinată card*BAZA + timp permite
    o singură căutare binară pe întregul tabel, în locul unei bucle pe carduri.
    """
    baza = np.int64(10**10)  # mai mare decât orice marcă temporală Unix
    cheie = chei.astype(np.int64) * baza + secunde
    stanga = np.searchsorted(cheie, cheie - durata_s, side="left")
    dreapta = np.searchsorted(cheie, cheie, side="left")
    return stanga, dreapta


def construieste_caracteristici(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adaugă caracteristicile din tabelul 2.3. Istoricul se calculează DOAR din
    tranzacțiile prezente în `df` (de aceea fiecare set – antrenare, validare,
    testare – se procesează separat, conform §2.5).
    Ordinea inițială a rândurilor se păstrează.
    """
    df = df.copy()
    df["_ordine"] = np.arange(len(df))
    df = df.sort_values([C.COL_CARD, C.COL_TIMP, "_ordine"], kind="mergesort").reset_index(drop=True)

    card_cod = pd.factorize(df[C.COL_CARD])[0]
    # secunde de la 1970 – independent de rezoluția internă (ns / us) a coloanei de timp
    secunde = ((df[C.COL_TIMP] - pd.Timestamp(0)) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
    suma = df[C.COL_SUMA].to_numpy(dtype=float)
    g = df.groupby(C.COL_CARD, sort=False)

    # ---------------- Sumă ----------------
    df["log_amt"] = np.log1p(suma)

    # media și abaterea standard ale sumelor ANTERIOARE ale cardului (formula 2.1)
    n_ant = g.cumcount().to_numpy()                       # câte tranzacții au fost înainte
    s_ant = g[C.COL_SUMA].cumsum().to_numpy() - suma      # suma lor
    s2_ant = (df[C.COL_SUMA] ** 2).groupby(df[C.COL_CARD], sort=False).cumsum().to_numpy() - suma ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        media = np.where(n_ant > 0, s_ant / n_ant, np.nan)
        var = np.where(n_ant > 1, (s2_ant - n_ant * media ** 2) / (n_ant - 1), np.nan)
    sigma = np.sqrt(np.clip(var, 0, None))

    with np.errstate(divide="ignore", invalid="ignore"):
        z = (suma - media) / sigma
    # Pentru clienții cu sub 2 tranzacții σ nu e definită -> abatere 0 (neutru).
    z = np.where((n_ant > 1) & (sigma > 1e-9), z, 0.0)
    df["amt_zscore_client"] = np.clip(z, -50, 50)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["amt_ratio_avg"] = np.where(n_ant > 0, suma / media, 1.0)

    # ---------------- Frecvență ----------------
    st1, dr1 = _ferestre(card_cod, secunde, 3600)
    st24, dr24 = _ferestre(card_cod, secunde, 24 * 3600)
    df["tx_count_1h"] = dr1 - st1
    df["tx_count_24h"] = dr24 - st24

    dif = g[C.COL_TIMP].diff().dt.total_seconds().to_numpy() / 3600
    df["time_since_last"] = np.clip(np.nan_to_num(dif, nan=C.PLAFON_TIMP_ORE), 0, C.PLAFON_TIMP_ORE)

    # ---------------- Locație și timp ----------------
    df["distance_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    df["hour"] = df[C.COL_TIMP].dt.hour
    df["is_night"] = df["hour"].isin(C.ORE_NOAPTE).astype(int)
    df["day_of_week"] = df[C.COL_TIMP].dt.dayofweek

    # ---------------- Istoric ----------------
    df["new_merchant"] = (df.groupby([C.COL_CARD, "merchant"], sort=False).cumcount() == 0).astype(int)
    df["new_category"] = (df.groupby([C.COL_CARD, "category"], sort=False).cumcount() == 0).astype(int)

    # media sumelor din ultimele 30 de zile, prin sume prefix pe fereastră
    st30, dr30 = _ferestre(card_cod, secunde, 30 * 24 * 3600)
    prefix = np.concatenate([[0.0], np.cumsum(suma)])
    n30 = dr30 - st30
    with np.errstate(divide="ignore", invalid="ignore"):
        medie30 = (prefix[dr30] - prefix[st30]) / n30
    # fără tranzacții în ultimele 30 de zile -> media istorică, apoi suma curentă
    medie30 = np.where(n30 > 0, medie30, np.where(n_ant > 0, media, suma))
    df["avg_amt_30d"] = medie30

    df = df.sort_values("_ordine").drop(columns="_ordine").reset_index(drop=True)
    return df


def criterii_suspiciune(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluează criteriile din tabelul 2.4 pentru fiecare tranzacție.
    Întoarce un tabel boolean, câte o coloană pentru fiecare regulă din config.REGULI.
    """
    return pd.DataFrame({
        "suma_mare": (df["amt_zscore_client"] > 3) | (df["amt_ratio_avg"] > 4),
        "frecventa_mare": (df["tx_count_1h"] >= 3) | (df["time_since_last"] < 0.1),
        "distanta_mare": df["distance_km"] > 120,
        "ora_noapte": df["is_night"] == 1,
        "comerciant_nou": (df["new_merchant"] == 1) & (df["new_category"] == 1),
        "categorie_risc": df["category"].isin(C.CATEGORII_RISC),
    }, index=df.index)
