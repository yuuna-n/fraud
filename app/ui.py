"""
Elemente comune ale interfeței: culori, stil, încărcarea sistemului salvat,
formatarea numerelor în stil românesc și aspectul graficelor Plotly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RADACINA = Path(__file__).resolve().parents[1]
if str(RADACINA) not in sys.path:
    sys.path.insert(0, str(RADACINA))

from src import config as C  # noqa: E402

# ---------------------------------------------------------------- culori
CERNEALA = "#172033"
GRI = "#5B677A"
LINIE = "#DCE1E8"
FUNDAL = "#F3F5F8"
ALBASTRU = "#2F5BD3"
VERDE = "#1E8A6E"
CHIHLIMBAR = "#D68A00"
ROSU = "#C8372D"
LEGITIM = "#9AA8BC"

CULORI_MODELE = {"xgboost": "#2F5BD3", "iforest": "#7A4FC9", "lof": "#0E9BA8", "reguli": "#8C96A6"}
NUME_MODELE = C.NUME_MODELE

# Paginile aplicației (completate în app.py), pentru legături între pagini.
PAGINI: dict = {}

CSS = f"""
<style>
html, body, [class*="st-"] {{ font-variant-numeric: tabular-nums; }}
.block-container {{ max-width: 1180px; padding-top: 2.2rem; }}
h1 {{ letter-spacing: -0.02em; }}
.antet-desc {{ color: {GRI}; font-size: 1.02rem; max-width: 68ch; margin: -0.4rem 0 1.4rem 0; }}

/* panoul cu verdictul – elementul principal al paginii de verificare */
.verdict {{ border-radius: 14px; padding: 1.4rem 1.6rem 1.2rem; color: #fff; position: relative; }}
.verdict.ridicat {{ background: linear-gradient(135deg, #B42F27, #D2483A); }}
.verdict.moderat {{ background: linear-gradient(135deg, #B87400, #DB9A1A); }}
.verdict.scazut  {{ background: linear-gradient(135deg, #17735C, #24977A); }}
.verdict .eticheta {{ font-size: .95rem; opacity: .9; }}
.verdict .procent {{ font-size: 4.2rem; font-weight: 700; line-height: 1; letter-spacing: -0.03em; margin: .25rem 0 .4rem; }}
.verdict .decizie {{ font-size: 1.15rem; font-weight: 600; }}
.verdict .scara {{ position: relative; height: 8px; border-radius: 4px; background: rgba(255,255,255,.28); margin: 1rem 0 .35rem; }}
.verdict .scara .umplere {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 4px; background: #fff; }}
.verdict .scara .prag {{ position: absolute; top: -5px; width: 2px; height: 18px; background: {CERNEALA}; }}
.verdict .legenda {{ display: flex; justify-content: space-between; font-size: .8rem; opacity: .85; }}

/* rândurile „registru” cu scorurile celor patru metode */
.registru {{ background: #fff; border: 1px solid {LINIE}; border-radius: 12px; padding: .3rem 1rem; }}
.registru .rand {{ display: grid; grid-template-columns: 1.6fr 2.4fr .8fr 1fr; gap: .8rem; align-items: center;
                   padding: .62rem 0; border-bottom: 1px solid {LINIE}; font-size: .95rem; }}
.registru .rand:last-child {{ border-bottom: 0; }}
.registru .bara {{ height: 6px; background: {FUNDAL}; border-radius: 3px; position: relative; }}
.registru .bara > span {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; }}
.registru .bara > i {{ position: absolute; top: -4px; width: 2px; height: 14px; background: {CERNEALA}; opacity: .55; }}
.registru .val {{ text-align: right; font-weight: 600; }}
.eticheta-alerta {{ display: inline-block; padding: .1rem .55rem; border-radius: 999px; font-size: .8rem; font-weight: 600; }}
.eticheta-alerta.da {{ background: #FBE3E1; color: {ROSU}; }}
.eticheta-alerta.nu {{ background: #E2F2EC; color: {VERDE}; }}

/* lista criteriilor din tabelul 2.4 */
.criteriu {{ display: flex; gap: .7rem; align-items: baseline; padding: .45rem 0; border-bottom: 1px dashed {LINIE}; }}
.criteriu:last-child {{ border-bottom: 0; }}
.criteriu .punct {{ width: .65rem; height: .65rem; border-radius: 50%; flex: none; transform: translateY(1px); }}
.criteriu .detaliu {{ color: {GRI}; font-size: .85rem; }}

.nota {{ color: {GRI}; font-size: .88rem; }}
.banda-demo {{ background: #FFF4DC; border: 1px solid #F1D8A0; color: #6B4E00; border-radius: 10px;
               padding: .6rem .9rem; font-size: .9rem; margin-bottom: 1rem; }}
[data-testid="stMetricValue"] {{ font-weight: 650; }}
[data-testid="stSidebarNav"] a span {{ font-size: .95rem; }}
</style>
"""


def aplica_stil() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def antet(titlu: str, descriere: str | None = None) -> None:
    st.title(titlu)
    if descriere:
        st.markdown(f'<p class="antet-desc">{descriere}</p>', unsafe_allow_html=True)


# ---------------------------------------------------------------- formatare
def nr(x, zecimale: int = 0) -> str:
    """Număr în format românesc: 1 234 567,89."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "–"
    s = f"{x:,.{zecimale}f}"
    return s.replace(",", " ").replace(".", ",")


def pct(x, zecimale: int = 1) -> str:
    return "–" if x is None or pd.isna(x) else nr(100 * x, zecimale) + "%"


def bani(x) -> str:
    return nr(x, 2) + " $"


# ---------------------------------------------------------------- date salvate
def exista_sistem() -> bool:
    return (C.DIR_MODELE / "sistem.joblib").exists() and (C.DIR_MODELE / "rezultate.json").exists()


@st.cache_resource(show_spinner="Încarc modelele…")
def sistem():
    from src.predict import SistemFrauda
    return SistemFrauda.incarca()


@st.cache_data(show_spinner=False)
def rezultate() -> dict:
    with open(C.DIR_MODELE / "rezultate.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def scoruri_validare() -> pd.DataFrame:
    return pd.read_parquet(C.DIR_MODELE / "scoruri_validare.parquet")


@st.cache_data(show_spinner="Încarc istoricul clienților…")
def istoric_referinta() -> pd.DataFrame:
    return pd.read_parquet(C.DIR_MODELE / "istoric_referinta.parquet")


def cere_sistem() -> bool:
    """Afișează instrucțiunile dacă modelele nu au fost antrenate; întoarce True dacă sunt."""
    if exista_sistem():
        return True
    st.warning("Modelele nu sunt antrenate încă.")
    st.markdown(
        "Puneți `fraudTrain.csv` și `fraudTest.csv` în folderul `data/raw/`, apoi rulați "
        "din folderul proiectului:\n\n```bash\npython train.py\n```\n"
        "Pentru o încercare rapidă fără setul Kaggle:\n\n"
        "```bash\npython tools/genereaza_date_demo.py\npython train.py\n```")
    return False


def banda_demo() -> None:
    if exista_sistem() and rezultate()["info"].get("date_demo"):
        st.markdown('<div class="banda-demo">Modelele sunt antrenate pe <b>date demonstrative</b> '
                    "generate de <code>tools/genereaza_date_demo.py</code>. Cifrele arată că sistemul "
                    "funcționează, nu performanța pe setul Kaggle.</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- grafice
def stil_grafic(fig: go.Figure, inaltime: int = 340, legenda: bool = True) -> go.Figure:
    fig.update_layout(
        height=inaltime, margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
        font=dict(family="IBM Plex Sans, sans-serif", color=CERNEALA, size=13),
        title_font=dict(size=15), showlegend=legenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(font_family="IBM Plex Sans, sans-serif"),
        separators=", ",
    )
    fig.update_xaxes(gridcolor="#EEF1F5", linecolor=LINIE, zeroline=False)
    fig.update_yaxes(gridcolor="#EEF1F5", linecolor=LINIE, zeroline=False)
    return fig


def denumire(col: str) -> str:
    from src.explain import denumire as d
    return d(col)
