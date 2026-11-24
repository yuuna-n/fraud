"""
Configurația centrală a sistemului: căi, liste de coloane, parametri ai modelelor
și pragurile regulilor de referință. Toate celelalte module citesc valorile de aici,
astfel încât un experiment se modifică dintr-un singur loc.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Căi
# ---------------------------------------------------------------------------
RADACINA = Path(__file__).resolve().parents[1]
DATE_BRUTE = RADACINA / "data" / "raw"
FISIER_TRAIN = DATE_BRUTE / "fraudTrain.csv"
FISIER_TEST = DATE_BRUTE / "fraudTest.csv"
DIR_MODELE = RADACINA / "models"

SEED = 42

# ---------------------------------------------------------------------------
# Coloanele setului de date (tabelul 2.1)
# ---------------------------------------------------------------------------
COL_TIMP = "trans_date_trans_time"
COL_CARD = "cc_num"
COL_SUMA = "amt"
COL_ETICHETA = "is_fraud"

# Coloane fără valoare predictivă sau cu date personale directe – se exclud (§2.3).
COLOANE_EXCLUSE = ["Unnamed: 0", "first", "last", "street", "trans_num", "unix_time"]

# Codificarea variabilelor categoriale, după cardinalitate (§2.3).
CATEG_ONE_HOT = ["category", "gender"]      # puține valori
CATEG_ORDINAL = ["state"]                   # cardinalitate medie
CATEG_TARGET = ["merchant"]                 # cardinalitate mare -> rata țintei (doar pe train)
CATEG_FRECVENTA = ["job", "city"]           # cardinalitate mare -> frecvență

# Variabile inițiale numerice păstrate în model.
NUMERICE_INITIALE = ["amt", "city_pop", "age"]

# ---------------------------------------------------------------------------
# Caracteristicile derivate (tabelul 2.3)
# ---------------------------------------------------------------------------
CARACTERISTICI_DERIVATE = [
    # sumă
    "log_amt", "amt_zscore_client", "amt_ratio_avg",
    # frecvență
    "tx_count_1h", "tx_count_24h", "time_since_last",
    # locație și timp
    "distance_km", "hour", "is_night", "day_of_week",
    # istoric
    "new_merchant", "new_category", "avg_amt_30d",
]

# Caracteristicile comportamentale folosite de algoritmii nesupravegheați
# (bazați pe distanțe/izolare) – doar variabile numerice cu sens, standardizate.
CARACTERISTICI_NESUPRAVEGHEATE = [
    "log_amt", "amt_zscore_client", "amt_ratio_avg",
    "tx_count_1h", "tx_count_24h", "time_since_last",
    "distance_km", "is_night", "new_merchant", "new_category", "avg_amt_30d",
]

# Ore considerate „de noapte" (is_night = 1).
ORE_NOAPTE = {22, 23, 0, 1, 2, 3, 4, 5}

# Plafon pentru time_since_last (ore) – se folosește și pentru prima tranzacție a cardului.
PLAFON_TIMP_ORE = 24 * 30

# ---------------------------------------------------------------------------
# Împărțirea datelor (tabelul 2.5)
# ---------------------------------------------------------------------------
# Ultima parte cronologică din fraudTrain.csv devine set de validare.
PONDERE_VALIDARE = 0.20

# ---------------------------------------------------------------------------
# Modele
# ---------------------------------------------------------------------------
XGB_PARAMETRI = dict(
    n_estimators=600,
    learning_rate=0.06,
    max_depth=6,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    eval_metric="aucpr",
    early_stopping_rounds=50,
    n_jobs=-1,
    random_state=SEED,
)
# Grila mică pentru optimizarea hiperparametrilor pe validare (opțiunea --optimizare).
XGB_GRILA = [
    dict(max_depth=4, learning_rate=0.08),
    dict(max_depth=6, learning_rate=0.06),
    dict(max_depth=8, learning_rate=0.05),
]

IFOREST_PARAMETRI = dict(n_estimators=300, max_samples=256, contamination="auto",
                         n_jobs=-1, random_state=SEED)

# LOF e costisitor (distanțe între puncte), de aceea se antrenează pe un eșantion.
# n_neighbors=100 a dat cel mai bun ROC-AUC pe validare dintre 20, 35 și 100.
LOF_PARAMETRI = dict(n_neighbors=100, novelty=True, n_jobs=-1)
LOF_ESANTION = 40_000

# Metoda de tratare a dezechilibrului (tabelul 2.2): "ponderi", "subesantionare", "smote".
DEZECHILIBRU_IMPLICIT = "ponderi"
RAPORT_SUBESANTIONARE = 0.05   # fraude / legitime după subeșantionare
RAPORT_SMOTE = 0.10            # fraude / legitime după SMOTE

# ---------------------------------------------------------------------------
# Pragul de decizie (§2.4) – asimetria costurilor
# ---------------------------------------------------------------------------
# Cât „costă" o fraudă ratată față de o alarmă falsă.
COST_FRAUDA_RATATA = 10.0
COST_ALARMA_FALSA = 1.0
METODA_PRAG = "cost"   # "cost", "f1" sau "recall"
RECALL_TINTA = 0.80

# ---------------------------------------------------------------------------
# Regulile de referință (tabelul 2.4) – metoda clasică
# ---------------------------------------------------------------------------
CATEGORII_RISC = ["shopping_net", "misc_net", "grocery_pos"]
REGULI = {
    "suma_mare":        {"titlu": "Sumă mult peste profilul clientului",
                         "conditie": "amt_zscore_client > 3 sau amt_ratio_avg > 4", "pondere": 2},
    "frecventa_mare":   {"titlu": "Multe tranzacții într-un interval scurt",
                         "conditie": "tx_count_1h ≥ 3 sau time_since_last < 0,1 h", "pondere": 1},
    "distanta_mare":    {"titlu": "Distanță mare față de domiciliu",
                         "conditie": "distance_km > 120", "pondere": 1},
    "ora_noapte":       {"titlu": "Activitate în timpul nopții",
                         "conditie": "is_night = 1", "pondere": 1},
    "comerciant_nou":   {"titlu": "Comerciant și categorie noi pentru client",
                         "conditie": "new_merchant = 1 și new_category = 1", "pondere": 1},
    "categorie_risc":   {"titlu": "Categorie de comerț cu risc ridicat",
                         "conditie": "category ∈ " + ", ".join(CATEGORII_RISC), "pondere": 1},
}

# Denumiri în română ale caracteristicilor, pentru interfață și grafice.
DENUMIRI = {
    "amt": "Suma", "log_amt": "Suma (log)", "amt_zscore_client": "Abaterea sumei față de client",
    "amt_ratio_avg": "Suma / media clientului", "tx_count_1h": "Tranzacții în ultima oră",
    "tx_count_24h": "Tranzacții în ultimele 24 h", "time_since_last": "Ore de la tranzacția anterioară",
    "distance_km": "Distanța față de domiciliu (km)", "hour": "Ora", "is_night": "Noapte",
    "day_of_week": "Ziua săptămânii", "new_merchant": "Comerciant nou", "new_category": "Categorie nouă",
    "avg_amt_30d": "Suma medie pe 30 de zile", "city_pop": "Populația localității", "age": "Vârsta",
    "merchant": "Comerciant (rata țintei)", "job": "Profesia (frecvență)", "city": "Localitatea (frecvență)",
    "state": "Statul",
}

NUME_MODELE = {
    "xgboost": "XGBoost",
    "iforest": "Isolation Forest",
    "lof": "Local Outlier Factor",
    "reguli": "Reguli de referință",
}
