"""
Interfața sistemului de predicție a fraudelor și anomaliilor bancare.

Pornire (din folderul proiectului):
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ui  # noqa: E402
from views import acasa, eda, explicatii, fisier, modele, prag, tranzactie  # noqa: E402

st.set_page_config(page_title="Detectarea fraudelor bancare", page_icon=":material/shield:",
                   layout="wide", initial_sidebar_state="expanded")
ui.aplica_stil()

P = ui.PAGINI
P["acasa"] = st.Page(acasa.pagina, title="Prezentare", icon=":material/home:", url_path="prezentare", default=True)
P["tranzactie"] = st.Page(tranzactie.pagina, title="Verificare tranzacție", icon=":material/credit_card:", url_path="tranzactie")
P["fisier"] = st.Page(fisier.pagina, title="Analiza unui fișier", icon=":material/upload_file:", url_path="fisier")
P["date"] = st.Page(eda.pagina, title="Explorarea datelor", icon=":material/bar_chart:", url_path="date")
P["modele"] = st.Page(modele.pagina, title="Compararea modelelor", icon=":material/compare_arrows:", url_path="modele")
P["prag"] = st.Page(prag.pagina, title="Pragul de decizie", icon=":material/tune:", url_path="prag")
P["explicatii"] = st.Page(explicatii.pagina, title="Explicabilitate", icon=":material/lightbulb:", url_path="explicatii")

pagini = {
    "": [P["acasa"]],
    "Analiză": [P["tranzactie"], P["fisier"], P["date"]],
    "Modele": [P["modele"], P["prag"], P["explicatii"]],
}

with st.sidebar:
    st.markdown("### Detectarea fraudelor\n<span style='opacity:.7;font-size:.85rem'>"
                "Proiect de an, UTM FCIM</span>", unsafe_allow_html=True)

st.navigation(pagini).run()
