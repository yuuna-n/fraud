"""
Pragul de decizie (§2.4): cum se schimbă fraudele prinse și alarmele false
când pragul crește sau scade. Se folosește setul de VALIDARE, cel pe care s-a
ales pragul; setul de testare rămâne neatins.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import ui
from src.threshold import tabel_praguri


@st.cache_data(show_spinner=False)
def _curba(model: str):
    """Tabelul complet al pragurilor (pentru costul minim exact)."""
    d = ui.scoruri_validare()
    return tabel_praguri(d["eticheta"], d[model])


def _reduce(*v, n: int = 1500):
    """Reduce numărul de puncte doar pentru desenare."""
    idx = np.unique(np.linspace(0, len(v[0]) - 1, min(n, len(v[0]))).astype(int))
    return [np.asarray(x)[idx] for x in v]


def pagina() -> None:
    ui.antet("Pragul de decizie",
             "Un prag mai mic prinde mai multe fraude, dar trimite analiștilor mai multe alarme false. "
             "Mutați pragul și urmăriți compromisul pe setul de validare.")
    if not ui.cere_sistem():
        return
    ui.banda_demo()
    r = ui.rezultate()
    d = ui.scoruri_validare()
    y = d["eticheta"].to_numpy()

    c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    model = c1.selectbox("Metoda", list(r["praguri"]), format_func=lambda m: ui.NUME_MODELE[m])
    cost_fn = c2.number_input("Costul unei fraude ratate", 0.0, 10_000.0, float(r["info"]["cost_fn"]), step=1.0)
    cost_fp = c3.number_input("Costul unei alarme false", 0.0, 10_000.0, float(r["info"]["cost_fp"]), step=1.0)

    prag_ales = float(r["praguri"][model])
    scor = d[model].to_numpy()
    jos = float(np.floor(np.quantile(scor, 0.5) * 100) / 100)
    cheie = f"prag_{model}"
    st.session_state.setdefault(cheie, prag_ales)
    if st.button("Revino la pragul ales la antrenare", icon=":material/restart_alt:"):
        st.session_state[cheie] = prag_ales
    prag = st.slider("Pragul", min_value=min(jos, prag_ales), max_value=1.0, step=0.0005, format="%.4f", key=cheie)

    alerta = scor >= prag
    tp = int((alerta & (y == 1)).sum())
    fp = int((alerta & (y == 0)).sum())
    fn = int((~alerta & (y == 1)).sum())
    poz = int(y.sum())
    m = st.columns(4)
    m[0].metric("Fraude prinse", f"{tp} din {poz}", ui.pct(tp / max(poz, 1)), delta_color="off", delta_arrow="off")
    m[1].metric("Alarme false", ui.nr(fp), f"{ui.nr(1000 * fp / len(y), 1)} la 1 000 de tranzacții", delta_color="off", delta_arrow="off")
    m[2].metric("Precision", ui.pct(tp / max(tp + fp, 1)),
                help="Din tranzacțiile semnalate, câte sunt fraude reale.")
    m[3].metric("Costul total", ui.nr(fn * cost_fn + fp * cost_fp), help="fraude ratate × cost + alarme false × cost")

    praguri, prec, rec, tpv, fpv, fnv = _curba(model)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.nan_to_num(2 * prec * rec / (prec + rec))
    cost = fnv * cost_fn + fpv * cost_fp
    prag_cost = float(praguri[int(np.argmin(cost))])
    praguri, prec, rec, f1, cost = _reduce(praguri, prec, rec, f1, cost)

    st1, st2 = st.columns(2, gap="large")
    with st1:
        fig = go.Figure()
        for nume, v, cul in [("Recall", rec, ui.ROSU), ("Precision", prec, ui.ALBASTRU), ("F1", f1, ui.GRI)]:
            fig.add_trace(go.Scatter(x=praguri, y=v, name=nume, mode="lines", line=dict(color=cul, width=2)))
        fig.add_vline(x=prag, line_color=ui.CERNEALA, line_width=1.5)
        fig.update_xaxes(title="Prag", range=[min(jos, prag_ales), 1])
        fig.update_yaxes(range=[0, 1.02])
        st.markdown("**Metricile în funcție de prag**")
        st.plotly_chart(ui.stil_grafic(fig, 360), use_container_width=True)
    with st2:
        fig = go.Figure(go.Scatter(x=praguri, y=cost, mode="lines", line=dict(color=ui.CHIHLIMBAR, width=2),
                                   name="Cost", hovertemplate="prag %{x:.4f}: cost %{y:,.0f}<extra></extra>"))
        fig.add_vline(x=prag, line_color=ui.CERNEALA, line_width=1.5)
        fig.add_vline(x=prag_cost, line_color=ui.CHIHLIMBAR, line_dash="dot",
                      annotation_text="cost minim", annotation_position="top")
        fig.update_xaxes(title="Prag", range=[min(jos, prag_ales), 1])
        # limita de sus: costul când nu se semnalează nimic (toate fraudele ratate), cu o marjă
        fig.update_yaxes(title="Cost total", range=[0, 1.6 * max(y.sum() * cost_fn, cost.min() * 2, 1)])
        st.markdown("**Costul erorilor în funcție de prag**")
        st.plotly_chart(ui.stil_grafic(fig, 360, legenda=False), use_container_width=True)

    st.markdown(f"Cu costurile de mai sus, pragul optim pentru {ui.NUME_MODELE[model]} este "
                f"**{ui.nr(prag_cost, 4)}**. Pragul salvat la antrenare este **{ui.nr(prag_ales, 4)}** "
                f"(metoda „{r['info']['metoda_prag']}”). Pentru a-l schimba definitiv, modificați "
                "`COST_FRAUDA_RATATA` / `METODA_PRAG` în `src/config.py` și rulați din nou `train.py`.")
