"""Explicabilitatea modelului XGBoost: importanța globală a caracteristicilor (SHAP, §5.5)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui

GRUPURI = {
    "Sumă": ["amt", "log_amt", "amt_zscore_client", "amt_ratio_avg"],
    "Frecvență": ["tx_count_1h", "tx_count_24h", "time_since_last"],
    "Locație și timp": ["distance_km", "hour", "is_night", "day_of_week"],
    "Istoric": ["new_merchant", "new_category", "avg_amt_30d"],
}
CULORI_GRUP = {"Sumă": ui.ROSU, "Frecvență": ui.CHIHLIMBAR, "Locație și timp": "#0E9BA8",
               "Istoric": "#7A4FC9", "Altele": ui.LEGITIM}

DESCRIERI = [
    ("Sumă", "log_amt", "Logaritmul sumei"),
    ("Sumă", "amt_zscore_client", "Abaterea sumei față de media istorică a clientului, în abateri standard"),
    ("Sumă", "amt_ratio_avg", "Raportul dintre suma curentă și suma medie a clientului"),
    ("Frecvență", "tx_count_1h, tx_count_24h", "Numărul de tranzacții ale cardului în ultima oră / zi"),
    ("Frecvență", "time_since_last", "Timpul scurs de la tranzacția anterioară a aceluiași card"),
    ("Locație și timp", "distance_km", "Distanța Haversine dintre domiciliul clientului și comerciant"),
    ("Locație și timp", "hour, is_night, day_of_week", "Ora, indicatorul pentru noapte și ziua săptămânii"),
    ("Istoric", "new_merchant, new_category", "Clientul cumpără pentru prima dată de la acel comerciant / din acea categorie"),
    ("Istoric", "avg_amt_30d", "Suma medie pe ultimele 30 de zile"),
]


def _grup(col: str) -> str:
    for g, cols in GRUPURI.items():
        if col in cols:
            return g
    return "Altele"


def pagina() -> None:
    ui.antet("Explicabilitate",
             "Ce caracteristici folosește cel mai mult XGBoost. Importanța e media valorii absolute SHAP: "
             "cât mută, în medie, fiecare caracteristică scorul unei tranzacții.")
    if not ui.cere_sistem():
        return
    ui.banda_demo()
    imp = pd.Series(ui.rezultate()["shap_global"]).sort_values()
    imp = imp[imp > 0].tail(18)
    grupuri = [_grup(c) for c in imp.index]

    fig = go.Figure()
    for g in CULORI_GRUP:
        masca = [x == g for x in grupuri]
        if any(masca):
            fig.add_trace(go.Bar(y=[ui.denumire(c) for c, k in zip(imp.index, masca) if k],
                                 x=[v for v, k in zip(imp.values, masca) if k], orientation="h", name=g,
                                 marker_color=CULORI_GRUP[g], hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(barmode="overlay", yaxis=dict(categoryorder="array",
                                                     categoryarray=[ui.denumire(c) for c in imp.index]))
    fig.update_xaxes(title="Media |SHAP| (log-odds)")
    st.plotly_chart(ui.stil_grafic(fig, 560), use_container_width=True)

    impg = pd.Series(ui.rezultate()["shap_global"])
    pe_grup = impg.groupby([_grup(c) for c in impg.index]).sum()
    total = pe_grup.sum()
    comport = pe_grup.drop("Altele", errors="ignore").sum()
    st.markdown(f"Caracteristicile derivate din comportamentul clientului (sumă, frecvență, locație și timp, "
                f"istoric) aduc **{ui.pct(comport / total, 0)}** din importanța totală. Restul vine din variabilele "
                "de context: categoria, comerciantul, datele demografice. Pentru explicația unei singure "
                "tranzacții, folosiți pagina *Verificare tranzacție*.")

    st.subheader("Caracteristicile derivate (tabelul 2.3)")
    st.dataframe(pd.DataFrame(DESCRIERI, columns=["Grup", "Caracteristică", "Mod de calcul"]),
                 hide_index=True, use_container_width=True)
