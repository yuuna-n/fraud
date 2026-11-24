"""
Antrenarea completă a sistemului.

Etape (cap. 2):
  1. încărcarea și curățarea fraudTrain.csv / fraudTest.csv;
  2. împărțirea cronologică: antrenare / validare (din fraudTrain) și testare (fraudTest);
  3. caracteristicile derivate, calculate SEPARAT pentru fiecare set;
  4. preprocesarea (învățată doar pe antrenare);
  5. XGBoost (+ tratarea dezechilibrului), Isolation Forest, LOF, reguli de referință;
  6. alegerea pragurilor pe validare;
  7. evaluarea finală pe testare (o singură dată) și importanța SHAP;
  8. salvarea sistemului și a rezultatelor pentru interfață.

Utilizare:
    python train.py
    python train.py --dezechilibru smote --optimizare
    python train.py --esantion-clienti 300        # încercare rapidă pe o parte din clienți
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.data_loading import incarca_seturi
from src.eda import statistici_eda
from src.evaluate import curbe, metrici
from src.explain import importanta_globala
from src.features import construieste_caracteristici
from src.imbalance import rebalanseaza
from src.models.rules import scor_reguli
from src.models.supervised import antreneaza_xgboost, optimizeaza_xgboost, scor_xgboost
from src.models.unsupervised import DetectorAnomalii
from src.predict import SistemFrauda
from src.preprocessing import Preprocesor
from src.split import descriere_seturi, imparte_cronologic
from src.threshold import alege_prag


def jurnal(mesaj: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {mesaj}", flush=True)


def _json(o):
    """Conversie pentru json.dump (numpy, pandas, NaN)."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main() -> None:
    p = argparse.ArgumentParser(description="Antrenează sistemul de detectare a fraudelor.")
    p.add_argument("--train", type=Path, default=C.FISIER_TRAIN)
    p.add_argument("--test", type=Path, default=C.FISIER_TEST)
    p.add_argument("--dezechilibru", choices=["ponderi", "subesantionare", "smote"],
                   default=C.DEZECHILIBRU_IMPLICIT)
    p.add_argument("--metoda-prag", choices=["cost", "f1", "recall"], default=C.METODA_PRAG)
    p.add_argument("--optimizare", action="store_true", help="caută hiperparametrii XGBoost pe validare")
    p.add_argument("--esantion-clienti", type=int, default=None,
                   help="folosește doar N carduri (pentru o rulare rapidă)")
    a = p.parse_args()
    t_start = time.time()

    # 1. Încărcare și curățare ------------------------------------------------
    jurnal("Încarc datele…")
    brut_train, brut_test = incarca_seturi(a.train, a.test, a.esantion_clienti)
    demo = bool(brut_train["merchant"].str.fullmatch(r"fraud_Merchant \d{3}").all())

    # 2. Împărțire cronologică -------------------------------------------------
    tr, va, moment = imparte_cronologic(brut_train)
    te = brut_test.sort_values(C.COL_TIMP, kind="mergesort").reset_index(drop=True)
    seturi = descriere_seturi({"Antrenare": tr, "Validare": va, "Testare": te})
    jurnal(f"Împărțire la {moment}:\n{seturi.to_string(index=False)}")

    # 3. Caracteristici derivate, separat pentru fiecare set (§2.5) -------------
    jurnal("Construiesc caracteristicile derivate…")
    tr_f, va_f, te_f = (construieste_caracteristici(d) for d in (tr, va, te))

    jurnal("Calculez statisticile pentru analiza exploratorie…")
    eda = statistici_eda(pd.concat([tr_f, va_f], ignore_index=True))

    # 4. Preprocesare -----------------------------------------------------------
    jurnal("Preprocesare (învățată doar pe antrenare)…")
    prep = Preprocesor()
    X_tr, Xn_tr = prep.fit_transform(tr_f)
    X_va, Xn_va = prep.transform(va_f)
    X_te, Xn_te = prep.transform(te_f)
    y_tr, y_va, y_te = (d[C.COL_ETICHETA].astype(int) for d in (tr_f, va_f, te_f))

    # 5. Modele -----------------------------------------------------------------
    jurnal(f"XGBoost (dezechilibru: {a.dezechilibru})…")
    X_bal, y_bal, spw = rebalanseaza(X_tr, y_tr, a.dezechilibru)
    grila = []
    if a.optimizare:
        xgb_model, parametri_xgb, grila = optimizeaza_xgboost(X_bal, y_bal, X_va, y_va, spw, jurnal)
    else:
        xgb_model, parametri_xgb = antreneaza_xgboost(X_bal, y_bal, X_va, y_va, spw), {}
    jurnal(f"    scale_pos_weight = {spw:.1f}, arbori = {xgb_model.best_iteration + 1}")

    jurnal("Isolation Forest…")
    iforest = DetectorAnomalii("iforest").antreneaza(Xn_tr)
    jurnal(f"Local Outlier Factor (eșantion {min(C.LOF_ESANTION, len(Xn_tr)):,})…")
    lof = DetectorAnomalii("lof").antreneaza(Xn_tr)

    def toate_scorurile(X, Xn, df):
        return {"xgboost": scor_xgboost(xgb_model, X), "iforest": iforest.scor(Xn),
                "lof": lof.scor(Xn), "reguli": scor_reguli(df)}

    # 6. Praguri pe validare -------------------------------------------------------
    jurnal(f"Aleg pragurile pe validare (metoda: {a.metoda_prag})…")
    s_va = toate_scorurile(X_va, Xn_va, va_f)
    praguri = {m: alege_prag(y_va, s, a.metoda_prag) for m, s in s_va.items()}

    # 7. Evaluare -------------------------------------------------------------------
    jurnal("Evaluez pe validare și pe testare…")
    s_te = toate_scorurile(X_te, Xn_te, te_f)
    rezultate = {"validare": {}, "testare": {}, "curbe_testare": {}, "curbe_validare": {}}
    for m in praguri:
        rezultate["validare"][m] = metrici(y_va, s_va[m], praguri[m], va_f[C.COL_SUMA])
        rezultate["testare"][m] = metrici(y_te, s_te[m], praguri[m], te_f[C.COL_SUMA])
        rezultate["curbe_testare"][m] = curbe(y_te, s_te[m])
        rezultate["curbe_validare"][m] = curbe(y_va, s_va[m])

    tabel = pd.DataFrame(rezultate["testare"]).T[["prag", "precision", "recall", "f1", "roc_auc", "pr_auc", "tp", "fp", "fn"]]
    jurnal("Rezultate pe setul de testare:\n" + tabel.round(4).to_string())

    jurnal("Importanța caracteristicilor (SHAP, eșantion din validare)…")
    shap_global = importanta_globala(xgb_model, X_va)

    # 8. Salvare ---------------------------------------------------------------------
    C.DIR_MODELE.mkdir(parents=True, exist_ok=True)
    info = {
        "antrenat_la": time.strftime("%Y-%m-%d %H:%M"),
        "date_demo": demo,
        "fisiere": [str(a.train.name), str(a.test.name)],
        "dezechilibru": a.dezechilibru,
        "metoda_prag": a.metoda_prag,
        "cost_fn": C.COST_FRAUDA_RATATA, "cost_fp": C.COST_ALARMA_FALSA,
        "scale_pos_weight": spw,
        "parametri_xgb": {**{k: v for k, v in C.XGB_PARAMETRI.items() if k in ("max_depth", "learning_rate")},
                          **parametri_xgb, "arbori": int(xgb_model.best_iteration + 1)},
        "grila_xgb": grila,
        "moment_validare": str(moment),
        "durata_s": round(time.time() - t_start, 1),
    }
    sistem = SistemFrauda(prep, xgb_model, iforest, lof, praguri, info)
    sistem.salveaza()

    rezultate.update({
        "info": info,
        "praguri": praguri,
        "seturi": seturi.to_dict(orient="records"),
        "eda": eda,
        "shap_global": shap_global.to_dict(),
    })
    with open(C.DIR_MODELE / "rezultate.json", "w", encoding="utf-8") as f:
        json.dump(rezultate, f, ensure_ascii=False, indent=1, default=_json)

    # Scorurile de validare – pentru pagina „Prag de decizie" din interfață.
    pd.DataFrame({"eticheta": y_va.values, "suma": va_f[C.COL_SUMA].values,
                  **{m: s.astype("float32") for m, s in s_va.items()}}
                 ).to_parquet(C.DIR_MODELE / "scoruri_validare.parquet", index=False)

    # Istoricul de referință pentru verificarea unei tranzacții (setul de testare, date curățate).
    brut_test.to_parquet(C.DIR_MODELE / "istoric_referinta.parquet", index=False)

    jurnal(f"Gata în {time.time() - t_start:.0f} s. Fișiere salvate în {C.DIR_MODELE}")


if __name__ == "__main__":
    main()
