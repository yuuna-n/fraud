"""Compararea celor patru metode (§5.4): metrici, curbe ROC și PR, matrice de confuzie."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui

METRICI = {"precision": "Precision", "recall": "Recall", "f1": "F1-score", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"}


def _matrice(m: dict, titlu: str, culoare: str) -> go.Figure:
    z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    text = [[f"{ui.nr(v)}" for v in rand] for rand in z]
    fig = go.Figure(go.Heatmap(z=[[1, 0], [0, 1]], x=["Legitimă", "Fraudă"],
                               y=["Legitimă", "Fraudă"], text=text, texttemplate="%{text}",
                               textfont=dict(size=15), showscale=False, hoverinfo="skip",
                               colorscale=[[0, "#F7F8FA"], [1, culoare]], opacity=.85))
    fig.update_yaxes(autorange="reversed", title="Reală")
    fig.update_xaxes(title="Prezisă", side="bottom")
    fig.update_layout(title=titlu)
    return ui.stil_grafic(fig, 230, legenda=False)


def pagina() -> None:
    ui.antet("Compararea modelelor",
             "Aceleași date, patru metode: clasificarea supravegheată, două metode de detectare a "
             "anomaliilor și regulile clasice. Pragurile au fost alese pe validare și aplicate neschimbate pe testare.")
    if not ui.cere_sistem():
        return
    ui.banda_demo()
    r = ui.rezultate()

    set_ = st.segmented_control("Setul de date", ["testare", "validare"], default="testare",
                                format_func=lambda s: "Testare (fraudTest.csv)" if s == "testare"
                                else "Validare (ultima parte din fraudTrain.csv)") or "testare"
    rez = r[set_]
    tabel = pd.DataFrame({ui.NUME_MODELE[m]: v for m, v in rez.items()}).T
    afis = tabel[list(METRICI) + ["tp", "fp", "fn", "prag"]].rename(
        columns={**METRICI, "tp": "Fraude prinse", "fp": "Alarme false", "fn": "Fraude ratate", "prag": "Prag"})
    st.dataframe(
        afis.style.format({**{v: "{:.3f}" for v in METRICI.values()}, "Prag": "{:.3f}",
                           "Fraude prinse": "{:.0f}", "Alarme false": "{:.0f}", "Fraude ratate": "{:.0f}"})
            .highlight_max(subset=list(METRICI.values()), color="#DDE7FF"),
        use_container_width=True)
    st.caption("Evidențiat: cea mai bună valoare din fiecare coloană. Acuratețea lipsește intenționat: cu sub 1% "
               "fraude, un model care spune mereu „legitimă” ar depăși 99%.")

    curbe = r[f"curbe_{set_}"]
    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = go.Figure()
        for m, cv in curbe.items():
            fig.add_trace(go.Scatter(x=cv["recall"], y=cv["precision"], name=ui.NUME_MODELE[m], mode="lines",
                                     line=dict(color=ui.CULORI_MODELE[m], width=2.2)))
        fig.update_xaxes(title="Recall (fraude prinse)", range=[0, 1])
        fig.update_yaxes(title="Precision", range=[0, 1.02])
        st.markdown("**Curba precision–recall**")
        st.plotly_chart(ui.stil_grafic(fig, 380), use_container_width=True)
    with c2:
        fig = go.Figure()
        for m, cv in curbe.items():
            fig.add_trace(go.Scatter(x=cv["fpr"], y=cv["tpr"], name=ui.NUME_MODELE[m], mode="lines",
                                     line=dict(color=ui.CULORI_MODELE[m], width=2.2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                                 line=dict(color=ui.LINIE, dash="dash")))
        fig.update_xaxes(title="Rata alarmelor false", range=[0, 1])
        fig.update_yaxes(title="Rata fraudelor prinse", range=[0, 1.02])
        st.markdown("**Curba ROC**")
        st.plotly_chart(ui.stil_grafic(fig, 380), use_container_width=True)

    st.subheader("Matricele de confuzie")
    for col, (m, v) in zip(st.columns(4), rez.items()):
        with col:
            st.plotly_chart(_matrice(v, ui.NUME_MODELE[m], ui.CULORI_MODELE[m]), use_container_width=True)

    st.subheader("Suma fraudelor oprite")
    st.markdown('<p class="nota">Ce parte din banii pierduți prin fraudă ar fi fost opriți la pragul ales.</p>',
                unsafe_allow_html=True)
    fig = go.Figure()
    modele = list(rez)
    fig.add_trace(go.Bar(x=[ui.NUME_MODELE[m] for m in modele], y=[rez[m]["suma_prinsa"] for m in modele],
                         marker_color=[ui.CULORI_MODELE[m] for m in modele], name="Oprită",
                         text=[ui.pct(rez[m]["suma_prinsa"] / max(rez[m]["suma_fraude"], 1), 0) for m in modele],
                         textposition="outside", hovertemplate="%{x}: %{y:,.0f} $<extra></extra>"))
    fig.add_hline(y=rez[modele[0]]["suma_fraude"], line_dash="dot", line_color=ui.ROSU,
                  annotation_text=f"Total fraude: {ui.bani(rez[modele[0]]['suma_fraude'])}",
                  annotation_position="top right")
    fig.update_yaxes(title="$", range=[0, rez[modele[0]]["suma_fraude"] * 1.18])
    st.plotly_chart(ui.stil_grafic(fig, 330, legenda=False), use_container_width=True)

    i = r["info"]
    with st.expander("Parametrii antrenării"):
        st.markdown(
            f"- Tratarea dezechilibrului: **{i['dezechilibru']}**, `scale_pos_weight` = {ui.nr(i['scale_pos_weight'], 1)}\n"
            f"- XGBoost: adâncime {i['parametri_xgb'].get('max_depth')}, rata de învățare "
            f"{i['parametri_xgb'].get('learning_rate')}, {i['parametri_xgb'].get('arbori')} arbori (oprire timpurie pe PR-AUC)\n"
            f"- Pragul: metoda „{i['metoda_prag']}”, o fraudă ratată costă cât {ui.nr(i['cost_fn'] / i['cost_fp'])} alarme false\n"
            f"- Validarea începe la {i['moment_validare']}; durata antrenării: {ui.nr(i['durata_s'])} s")
        if i.get("grila_xgb"):
            st.dataframe(pd.DataFrame(i["grila_xgb"]), hide_index=True)
