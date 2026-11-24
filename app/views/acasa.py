"""Pagina de prezentare: starea sistemului și rezumatul rezultatelor."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import ui

PASI = [
    ("Datele", "Tranzacțiile sunt curățate, iar coloanele categoriale sunt codificate."),
    ("Profilul clientului", "Pentru fiecare card se calculează abaterea sumei, frecvența, "
                            "distanța și noutatea comerciantului, numai din istoricul anterior."),
    ("Patru metode", "XGBoost dă o probabilitate de fraudă; Isolation Forest și LOF dau un scor "
                     "de anomalie; regulile clasice dau puncte pentru fiecare criteriu."),
    ("Pragul", "Fiecare scor e comparat cu un prag ales pe setul de validare, după costul erorilor."),
    ("Decizia", "Tranzacțiile peste prag sunt semnalate analistului, împreună cu motivele."),
]


def pagina() -> None:
    ui.antet("Detectarea fraudelor bancare",
             "Sistemul compară fiecare tranzacție cu obiceiurile clientului și o semnalează "
             "când se abate de la ele. Patru metode lucrează în paralel, ca să le puteți compara.")
    if not ui.cere_sistem():
        return
    ui.banda_demo()
    r = ui.rezultate()
    seturi = {s["set"]: s for s in r["seturi"]}
    total = sum(s["tranzacții"] for s in r["seturi"])
    fraude = sum(s["fraude"] for s in r["seturi"])
    x = r["testare"]["xgboost"]

    c = st.columns(4)
    c[0].metric("Tranzacții în date", ui.nr(total))
    c[1].metric("Ponderea fraudelor", ui.pct(fraude / total, 2))
    c[2].metric("Fraude prinse de XGBoost (testare)", ui.pct(x["recall"]),
                help="Recall pe setul de testare, la pragul ales pe validare.")
    c[3].metric("Alarme false la 1 000 de tranzacții", ui.nr(1000 * x["fp"] / seturi["Testare"]["tranzacții"], 1))

    st.subheader("Cum ajunge o tranzacție la decizie")
    html = "".join(
        f'<div style="flex:1;min-width:170px;padding:.9rem 1rem;background:#fff;border:1px solid {ui.LINIE};'
        f'border-top:3px solid {ui.ALBASTRU if i < 4 else ui.ROSU};border-radius:10px">'
        f'<div style="font-weight:650;margin-bottom:.3rem">{i + 1}. {t}</div>'
        f'<div class="nota">{d}</div></div>'
        for i, (t, d) in enumerate(PASI))
    st.markdown(f'<div style="display:flex;gap:.7rem;flex-wrap:wrap">{html}</div>', unsafe_allow_html=True)

    st.subheader("Cât de bine separă fraudele fiecare metodă")
    st.markdown('<p class="nota">PR-AUC pe setul de testare (1 = perfect). Pentru date atât de '
                "dezechilibrate e mai relevantă decât acuratețea sau ROC-AUC.</p>", unsafe_allow_html=True)
    modele = list(r["testare"])[::-1]
    fig = go.Figure(go.Bar(
        y=[ui.NUME_MODELE[m] for m in modele], x=[r["testare"][m]["pr_auc"] for m in modele],
        orientation="h", marker_color=[ui.CULORI_MODELE[m] for m in modele],
        text=[ui.nr(r["testare"][m]["pr_auc"], 3) for m in modele], textposition="outside",
        hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(range=[0, 1.08], title="PR-AUC")
    st.plotly_chart(ui.stil_grafic(fig, 250, legenda=False), use_container_width=True)

    st.subheader("Ce puteți face")
    legaturi = [
        ("tranzactie", "Verificați o tranzacție",
         "Introduceți datele unei plăți și vedeți riscul, motivele și contribuția fiecărei caracteristici."),
        ("fisier", "Analizați un fișier", "Încărcați un CSV cu tranzacții și descărcați lista celor suspecte."),
        ("prag", "Reglați pragul", "Vedeți câte fraude prindeți și câte alarme false primiți la fiecare prag."),
    ]
    for col, (cheie, titlu, text) in zip(st.columns(3), legaturi):
        with col:
            st.page_link(ui.PAGINI[cheie], label=titlu, icon=":material/arrow_forward:")
            st.markdown(f'<p class="nota">{text}</p>', unsafe_allow_html=True)

    i = r["info"]
    st.caption(f"Antrenat la {i['antrenat_la']} pe {', '.join(i['fisiere'])}. Dezechilibru tratat prin: "
               f"{i['dezechilibru']}. Pragul ales după metoda „{i['metoda_prag']}”.")
