"""
Analiza unui fișier CSV întreg: scorurile tuturor tranzacțiilor, lista celor
semnalate și, dacă fișierul are eticheta is_fraud, metricile pe acel fișier.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import ui
from src import config as C
from src.data_loading import citeste_csv, raport_calitate, verifica_coloane
from src.evaluate import metrici

COLOANE_AFISATE = [C.COL_TIMP, C.COL_CARD, "merchant", "category", C.COL_SUMA, "distance_km",
                   "tx_count_1h", "amt_zscore_client"]


@st.cache_data(show_spinner=False, max_entries=3)
def _analizeaza(brute: pd.DataFrame) -> pd.DataFrame:
    return ui.sistem().analizeaza_fisier(brute)


def pagina() -> None:
    ui.antet("Analiza unui fișier",
             "Încărcați un CSV cu aceeași structură ca setul Kaggle. Istoricul fiecărui client se "
             "construiește numai din tranzacțiile din fișier, deci fișierele mai lungi dau profiluri mai bune.")
    if not ui.cere_sistem():
        return
    ui.banda_demo()

    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    incarcat = c1.file_uploader("Fișier CSV cu tranzacții", type=["csv"])
    exemplu = c2.button("Folosește setul de testare", icon=":material/dataset:", use_container_width=True,
                        help="Analizează tranzacțiile din fraudTest.csv salvate la antrenare.")
    if exemplu:
        st.session_state["fis_sursa"] = "test"
    if incarcat is not None:
        st.session_state["fis_sursa"] = "upload"

    sursa = st.session_state.get("fis_sursa")
    if sursa is None:
        st.info("Coloane necesare: " + ", ".join(f"`{c}`" for c in
                [C.COL_TIMP, C.COL_CARD, "merchant", "category", C.COL_SUMA, "gender", "job", "city",
                 "state", "city_pop", "lat", "long", "merch_lat", "merch_long"]) +
                ". Opțional: `dob` (pentru vârstă) și `is_fraud` (pentru a calcula metricile).",
                icon=":material/info:")
        return

    if sursa == "upload" and incarcat is not None:
        brute = citeste_csv(incarcat)
        nume = incarcat.name
    else:
        brute = ui.istoric_referinta()
        nume = "setul de testare"

    lipsa = verifica_coloane(brute)
    if lipsa:
        st.error("Fișierul nu are coloanele: " + ", ".join(lipsa) + ". Verificați antetul CSV-ului.")
        return

    with st.expander(f"Calitatea datelor din {nume}"):
        st.dataframe(raport_calitate(brute), use_container_width=True)

    with st.spinner(f"Analizez {ui.nr(len(brute))} tranzacții…"):
        rez = _analizeaza(brute)
    praguri = ui.sistem().praguri

    model = st.segmented_control("Metoda după care se ordonează", list(praguri),
                                 format_func=lambda m: ui.NUME_MODELE[m], default="xgboost") or "xgboost"
    alerte = rez[f"alerta_{model}"]

    m = st.columns(4)
    m[0].metric("Tranzacții analizate", ui.nr(len(rez)))
    m[1].metric("Semnalate", ui.nr(int(alerte.sum())), ui.pct(alerte.mean(), 2), delta_color="off", delta_arrow="off")
    m[2].metric("Suma semnalată", ui.bani(rez.loc[alerte, C.COL_SUMA].sum()))
    if C.COL_ETICHETA in rez:
        y = rez[C.COL_ETICHETA].astype(int)
        m[3].metric("Fraude reale prinse", f"{int((alerte & (y == 1)).sum())} din {int(y.sum())}")

    criterii = [c for c in rez.columns if c.startswith("criteriu_")]
    rez["criterii_indeplinite"] = rez[criterii].sum(axis=1)
    doar = st.toggle("Doar tranzacțiile semnalate", value=True)
    vizibile = rez[alerte] if doar else rez
    vizibile = vizibile.sort_values(model, ascending=False)

    coloane = COLOANE_AFISATE + [model, "criterii_indeplinite"] + ([C.COL_ETICHETA] if C.COL_ETICHETA in rez else [])
    st.dataframe(
        vizibile[coloane].head(2000), hide_index=True, use_container_width=True, height=440,
        column_config={
            C.COL_TIMP: st.column_config.DatetimeColumn("Momentul", format="DD.MM.YYYY HH:mm"),
            C.COL_CARD: st.column_config.NumberColumn("Card", format="%d"),
            "merchant": "Comerciant", "category": "Categorie",
            C.COL_SUMA: st.column_config.NumberColumn("Suma", format="%.2f $"),
            "distance_km": st.column_config.NumberColumn("Distanța (km)", format="%.0f"),
            "tx_count_1h": st.column_config.NumberColumn("Tranz. în ultima oră"),
            "amt_zscore_client": st.column_config.NumberColumn("Abaterea sumei (σ)", format="%.1f"),
            model: st.column_config.ProgressColumn(f"Risc {ui.NUME_MODELE[model]}", min_value=0, max_value=1,
                                                   format="percent"),
            "criterii_indeplinite": st.column_config.NumberColumn("Criterii îndeplinite"),
            C.COL_ETICHETA: st.column_config.CheckboxColumn("Fraudă reală"),
        })
    if len(vizibile) > 2000:
        st.caption(f"Sunt afișate primele 2 000 din {ui.nr(len(vizibile))} rânduri; fișierul descărcat le conține pe toate.")

    st.download_button("Descarcă tranzacțiile semnalate (CSV)", rez[alerte].to_csv(index=False).encode("utf-8"),
                       file_name=f"tranzactii_semnalate_{model}.csv", mime="text/csv", icon=":material/download:")

    if C.COL_ETICHETA in rez:
        st.subheader("Metricile pe acest fișier")
        y = rez[C.COL_ETICHETA].astype(int)
        if y.nunique() < 2:
            st.caption("Fișierul are o singură clasă, deci metricile nu pot fi calculate.")
            return
        t = pd.DataFrame({ui.NUME_MODELE[k]: metrici(y, rez[k], p) for k, p in praguri.items()}).T
        t = t[["precision", "recall", "f1", "pr_auc", "roc_auc", "tp", "fp", "fn"]].rename(columns={
            "precision": "Precision", "recall": "Recall", "f1": "F1-score", "pr_auc": "PR-AUC",
            "roc_auc": "ROC-AUC", "tp": "Fraude prinse", "fp": "Alarme false", "fn": "Fraude ratate"})
        st.dataframe(t.style.format({c: "{:.3f}" for c in ["Precision", "Recall", "F1-score", "PR-AUC", "ROC-AUC"]}
                                    | {c: "{:.0f}" for c in ["Fraude prinse", "Alarme false", "Fraude ratate"]}),
                     use_container_width=True)
