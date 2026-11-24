"""
Generator de date demonstrative.

Creează două fișiere cu ACEEAȘI structură ca setul Kaggle
„Credit Card Transactions Fraud Detection Dataset" (fraudTrain.csv / fraudTest.csv),
pentru a putea testa sistemul fără descărcarea celor ~1,85 mil. de tranzacții.

Tiparele de fraudă imită pe cele observate în analiza exploratorie (cap. 2.2):
  * sume mai mari decât profilul clientului;
  * activitate concentrată noaptea;
  * categorii cu risc ridicat (shopping_net, misc_net, grocery_pos);
  * serii de tranzacții într-un interval scurt.

ATENȚIE: datele sunt artificiale. Rezultatele obținute pe ele servesc doar
la verificarea funcționării codului, nu descriu performanța reală.

Utilizare:
    python tools/genereaza_date_demo.py --clienti 150 --iesire data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Categoriile de comerț din setul original și parametrii sumelor legitime
# (media și abaterea în scară logaritmică).
CATEGORII = {
    "gas_transport": (4.0, 0.5),
    "grocery_pos": (4.6, 0.4),
    "home": (3.9, 0.9),
    "shopping_pos": (3.8, 1.1),
    "kids_pets": (3.6, 0.9),
    "shopping_net": (3.9, 1.1),
    "entertainment": (3.7, 0.9),
    "food_dining": (3.6, 0.9),
    "personal_care": (3.2, 0.9),
    "health_fitness": (3.7, 0.7),
    "misc_pos": (3.5, 1.0),
    "misc_net": (3.5, 1.1),
    "grocery_net": (3.9, 0.4),
    "travel": (3.6, 1.5),
}
PONDERI_CATEGORII = np.array([10, 9, 9, 9, 9, 7, 7, 7, 7, 6, 6, 5, 4, 3], dtype=float)

# Categoriile și intervalele de sume folosite în scenariile de fraudă.
FRAUDA_CATEGORII = {
    "shopping_net": (700, 1200),
    "misc_net": (650, 1000),
    "grocery_pos": (270, 360),
    "shopping_pos": (600, 1100),
    "gas_transport": (5, 20),  # „testarea" cardului cu sume mici
}
FRAUDA_PONDERI = np.array([30, 22, 28, 12, 8], dtype=float)

STATE = ["TX", "NY", "PA", "CA", "OH", "MI", "IL", "FL", "AL", "MO", "MN", "AR", "NC", "VA", "WI"]
ORASE = ["Springfield", "Riverside", "Franklin", "Greenville", "Bristol", "Clinton", "Fairview",
         "Salem", "Madison", "Georgetown", "Arlington", "Ashland", "Dover", "Milton", "Newport"]
JOBURI = ["Engineer", "Teacher", "Nurse", "Accountant", "Lawyer", "Designer", "Pharmacist",
          "Electrician", "Chef", "Architect", "Librarian", "Surveyor", "Paramedic", "Analyst",
          "Editor", "Plumber", "Geologist", "Programmer", "Dentist", "Farmer"]
PRENUME = ["Anna", "John", "Maria", "David", "Laura", "James", "Emily", "Robert", "Sarah", "Mark"]
NUME = ["Smith", "Johnson", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor", "Clark", "Lewis"]


def _genereaza_clienti(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Profilurile clienților: card, date demografice, domiciliu."""
    oras_idx = rng.integers(0, len(ORASE), n)
    return pd.DataFrame({
        "cc_num": rng.integers(10**15, 10**16, n, dtype=np.int64),
        "first": rng.choice(PRENUME, n),
        "last": rng.choice(NUME, n),
        "gender": rng.choice(["F", "M"], n),
        "street": [f"{rng.integers(1, 9999)} Main St" for _ in range(n)],
        "city": np.array(ORASE)[oras_idx],
        "state": rng.choice(STATE, n),
        "zip": rng.integers(10000, 99999, n),
        "lat": rng.uniform(27, 47, n).round(4),
        "long": rng.uniform(-120, -72, n).round(4),
        "city_pop": rng.lognormal(8.5, 2.0, n).astype(int) + 100,
        "job": rng.choice(JOBURI, n),
        "dob": pd.to_datetime(rng.integers(
            pd.Timestamp("1940-01-01").value // 10**9,
            pd.Timestamp("2002-01-01").value // 10**9, n), unit="s").strftime("%Y-%m-%d"),
        "rata_zilnica": rng.uniform(0.8, 4.0, n),  # tranzacții pe zi
        "multiplicator_suma": rng.lognormal(0, 0.35, n),
    })


def _ore_legitime(n: int, rng: np.random.Generator) -> np.ndarray:
    """Orele tranzacțiilor legitime: mai multe ziua, puține noaptea."""
    ponderi = np.array([2, 1, 1, 1, 1, 2, 4, 6, 7, 7, 7, 8, 9, 9, 8, 8, 8, 8, 8, 7, 6, 5, 4, 3], float)
    return rng.choice(24, n, p=ponderi / ponderi.sum())


def genereaza(clienti: pd.DataFrame, inceput: str, sfarsit: str, rng: np.random.Generator,
              comercianti: list[str], rata_frauda: float = 0.006) -> pd.DataFrame:
    t0, t1 = pd.Timestamp(inceput), pd.Timestamp(sfarsit)
    zile = (t1 - t0).days
    categorii = list(CATEGORII)
    blocuri = []

    for _, c in clienti.iterrows():
        # --- tranzacții legitime ---
        n = rng.poisson(c.rata_zilnica * zile)
        zi = rng.integers(0, zile, n)
        ora = _ore_legitime(n, rng)
        secunde = rng.integers(0, 3600, n)
        momente = t0 + pd.to_timedelta(zi, "D") + pd.to_timedelta(ora, "h") + pd.to_timedelta(secunde, "s")
        cat = rng.choice(categorii, n, p=PONDERI_CATEGORII / PONDERI_CATEGORII.sum())
        mu = np.array([CATEGORII[k][0] for k in cat])
        sd = np.array([CATEGORII[k][1] for k in cat])
        suma = np.exp(rng.normal(mu, sd)) * c.multiplicator_suma
        # fiecare client are un grup restrâns de comercianți preferați
        preferati = rng.choice(comercianti, 40, replace=False)
        merchant = np.where(rng.random(n) < 0.85, rng.choice(preferati, n), rng.choice(comercianti, n))
        legit = pd.DataFrame({"trans_date_trans_time": momente, "category": cat,
                              "amt": suma, "merchant": merchant, "is_fraud": 0})

        # --- episoade de fraudă (serie scurtă de tranzacții) ---
        fraude = []
        # numărul de episoade e proporțional cu durata perioadei
        n_ep = rng.poisson(zile * rata_frauda / 4.5)
        for _ in range(n_ep):
            start = t0 + pd.Timedelta(days=int(rng.integers(5, max(6, zile - 3))))
            start += pd.Timedelta(hours=int(rng.choice([22, 23, 0, 1, 2, 3])))
            k = int(rng.integers(3, 12))
            cat_f = rng.choice(list(FRAUDA_CATEGORII), k, p=FRAUDA_PONDERI / FRAUDA_PONDERI.sum())
            lo = np.array([FRAUDA_CATEGORII[x][0] for x in cat_f])
            hi = np.array([FRAUDA_CATEGORII[x][1] for x in cat_f])
            # seria durează de la câteva ore la ~2 zile; orele rămân preponderent nocturne
            decalaj = np.sort(rng.uniform(0, rng.choice([3, 20, 48]), k))
            mom = start + pd.to_timedelta(decalaj, "h")
            fraude.append(pd.DataFrame({
                "trans_date_trans_time": mom, "category": cat_f,
                "amt": rng.uniform(lo, hi), "merchant": rng.choice(comercianti, k),
                "is_fraud": 1}))
        bloc = pd.concat([legit, *fraude], ignore_index=True)

        for col in ["cc_num", "first", "last", "gender", "street", "city", "state", "zip",
                    "lat", "long", "city_pop", "job", "dob"]:
            bloc[col] = c[col]
        # comerciantul se află în jurul domiciliului (±1 grad), ca în simulatorul Sparkov
        bloc["merch_lat"] = (c.lat + rng.uniform(-1, 1, len(bloc))).round(6)
        bloc["merch_long"] = (c.long + rng.uniform(-1, 1, len(bloc))).round(6)
        blocuri.append(bloc)

    df = pd.concat(blocuri, ignore_index=True)
    df = df[(df.trans_date_trans_time >= t0) & (df.trans_date_trans_time < t1)]
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)
    df["amt"] = df["amt"].round(2).clip(lower=1.0)
    df["unix_time"] = (df["trans_date_trans_time"] - pd.Timestamp(0)) // pd.Timedelta(seconds=1)
    df["trans_num"] = [f"{x:032x}" for x in rng.integers(0, 2**62, len(df), dtype=np.int64)]
    df["trans_date_trans_time"] = df["trans_date_trans_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df.insert(0, "Unnamed: 0", np.arange(len(df)))
    coloane = ["Unnamed: 0", "trans_date_trans_time", "cc_num", "merchant", "category", "amt",
               "first", "last", "gender", "street", "city", "state", "zip", "lat", "long",
               "city_pop", "job", "dob", "trans_num", "unix_time", "merch_lat", "merch_long", "is_fraud"]
    return df[coloane]


def main() -> None:
    p = argparse.ArgumentParser(description="Generează date demo cu structura setului Kaggle.")
    p.add_argument("--clienti", type=int, default=150, help="numărul de carduri (clienți)")
    p.add_argument("--iesire", default="data/raw", help="folderul în care se scriu fișierele")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    comercianti = [f"fraud_Merchant {i:03d}" for i in range(400)]
    out = Path(a.iesire)
    out.mkdir(parents=True, exist_ok=True)

    # Aceiași clienți și aceleași perioade ca în setul original.
    clienti = _genereaza_clienti(a.clienti, rng)
    train = genereaza(clienti, "2019-01-01", "2020-06-21", rng, comercianti)
    test = genereaza(clienti, "2020-06-21", "2021-01-01", rng, comercianti)
    train.to_csv(out / "fraudTrain.csv", index=False)
    test.to_csv(out / "fraudTest.csv", index=False)
    for nume, d in [("fraudTrain.csv", train), ("fraudTest.csv", test)]:
        print(f"{nume}: {len(d):,} tranzacții, fraudă {d.is_fraud.mean():.2%}")


if __name__ == "__main__":
    main()
