"""Explorarea datelor (§2.2): figurile 2.1–2.4, corelațiile și valorile extreme."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui


def pagina() -> None:
    ui.antet("Explorarea datelor",
             "Cum arată tranzacțiile din fraudTrain.csv (antrenare și validare) înainte de modelare: "
             "cât de rare sunt fraudele, ce sume au, la ce ore și în ce categorii apar.")
    if not ui.cere_sistem():
        return
    ui.banda_demo()
    e = ui.rezultate()["eda"]

    c = st.columns(4)
    c[0].metric("Tranzacții", ui.nr(e["n"]))
    c[1].metric("Carduri", ui.nr(e["carduri"]))
    c[2].metric("Comercianți", ui.nr(e["comercianti"]))
    p0, p1 = (pd.Timestamp(x) for x in e["perioada"])
    luni = (p1.year - p0.year) * 12 + p1.month - p0.month + 1
    c[3].metric("Perioada", f"{luni} luni", help=f"{p0:%d.%m.%Y} – {p1:%d.%m.%Y}")

    st1, st2 = st.columns(2, gap="large")
    with st1:
        st.subheader("Ponderea claselor")
        cl = e["clase"]
        tot = sum(cl.values())
        fig = go.Figure(go.Bar(
            x=["Legitime", "Frauduloase"], y=[cl["legitime"], cl["frauduloase"]],
            marker_color=[ui.LEGITIM, ui.ROSU],
            text=[ui.pct(cl["legitime"] / tot, 2), ui.pct(cl["frauduloase"] / tot, 2)], textposition="outside",
            hovertemplate="%{x}: %{y:,}<extra></extra>"))
        fig.update_yaxes(type="log", dtick=1, title="Număr de tranzacții (scară log.)")
        st.plotly_chart(ui.stil_grafic(fig, 330, legenda=False), use_container_width=True)
        st.caption("Figura 2.1. Fraudele sunt sub 1% din total, de aceea acuratețea nu e o metrică potrivită.")

    with st2:
        st.subheader("Distribuția sumelor")
        h = e["hist_sume"]
        m = np.array(h["margini"])
        centre = np.sqrt(m[:-1] * m[1:])
        fig = go.Figure()
        for nume, culoare in [("legitime", ui.LEGITIM), ("frauduloase", ui.ROSU)]:
            fig.add_trace(go.Scatter(x=centre, y=np.array(h[nume]) * 100, name=nume.capitalize(),
                                     mode="lines", line=dict(color=culoare, width=2, shape="hvh"),
                                     fill="tozeroy", opacity=.6,
                                     hovertemplate="~%{x:.0f} $: %{y:.2f}%<extra></extra>"))
        fig.update_xaxes(type="log", title="Suma tranzacției ($, scară log.)")
        fig.update_yaxes(title="% din tranzacțiile clasei")
        st.plotly_chart(ui.stil_grafic(fig, 330), use_container_width=True)
        s = e["sume_medii"]
        st.caption(f"Figura 2.2. Suma medie: {ui.bani(s['legitime']['mean'])} la tranzacțiile legitime, "
                   f"{ui.bani(s['frauduloase']['mean'])} la cele frauduloase.")

    st3, st4 = st.columns(2, gap="large")
    with st3:
        st.subheader("Fraudele pe ore")
        o = e["ore"]
        rate = np.array(o["rata"]) * 100
        noapte = [h in (22, 23, 0, 1, 2, 3, 4, 5) for h in o["ora"]]
        fig = go.Figure(go.Bar(x=o["ora"], y=rate, marker_color=[ui.ROSU if n else ui.LEGITIM for n in noapte],
                               customdata=o["numar"],
                               hovertemplate="Ora %{x}: %{y:.2f}% fraude din %{customdata:,} tranzacții<extra></extra>"))
        fig.update_xaxes(title="Ora din zi", dtick=2)
        fig.update_yaxes(title="Ponderea fraudelor (%)")
        st.plotly_chart(ui.stil_grafic(fig, 330, legenda=False), use_container_width=True)
        st.caption("Figura 2.3. Orele de noapte (22–5, în roșu) au o pondere mult mai mare a fraudelor.")

    with st4:
        st.subheader("Fraudele pe categorii")
        k = e["categorii"]
        fig = go.Figure(go.Bar(y=k["categorie"], x=np.array(k["rata"]) * 100, orientation="h",
                               marker_color=ui.ALBASTRU, customdata=k["numar"],
                               hovertemplate="%{y}: %{x:.2f}% din %{customdata:,}<extra></extra>"))
        fig.update_xaxes(title="Ponderea fraudelor (%)")
        st.plotly_chart(ui.stil_grafic(fig, 380, legenda=False), use_container_width=True)
        st.caption("Figura 2.4. Achizițiile online (_net) și unele plăți fizice au cel mai mare risc.")

    st.subheader("Corelațiile dintre variabile")
    cor = e["corelatii"]
    nume = [ui.denumire(c) if c != "is_fraud" else "Fraudă" for c in cor["coloane"]]
    fig = go.Figure(go.Heatmap(z=cor["valori"], x=nume, y=nume, zmin=-1, zmax=1,
                               colorscale=[[0, ui.ALBASTRU], [.5, "#FFFFFF"], [1, ui.ROSU]],
                               hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(ui.stil_grafic(fig, 620, legenda=False), use_container_width=True)
    st.caption("Corelațiile directe cu eticheta sunt slabe: informația utilă stă în combinațiile de variabile, "
               "pe care le învață modelele.")

    q = e["iqr"]
    st.subheader("Valori extreme")
    st.markdown(f"După metoda intervalului intercuartilic, sumele peste **{ui.bani(q['limita_superioara'])}** "
                f"sunt extreme: **{ui.nr(q['extreme'])}** tranzacții, dintre care **{ui.pct(q['fraude_printre_extreme'], 2)}** "
                "sunt fraude. Ele nu au fost eliminate, pentru că pot fi chiar tranzacțiile căutate.")
