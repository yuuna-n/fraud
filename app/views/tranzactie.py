"""
Verificarea unei singure tranzacții: formular, verdict, scorurile celor patru
metode, criteriile de suspiciune (tabelul 2.4) și explicația SHAP.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from src import config as C

CLIENT_NOU = "Client nou (fără istoric)"
CHEI = ["f_data", "f_ora", "f_suma", "f_categorie", "f_comerciant", "f_distanta"]


# ------------------------------------------------------------------ date
def _clienti(ist: pd.DataFrame) -> pd.DataFrame:
    g = ist.groupby(C.COL_CARD).agg(oras=("city", "first"), stat=("state", "first"),
                                     n=(C.COL_SUMA, "size"), ultima=(C.COL_TIMP, "max"))
    return g.sort_values("n", ascending=False)


def _eticheta_client(card: int, rand) -> str:
    return f"Card …{str(card)[-4:]}, {rand.oras} ({rand.stat}), {rand.n} tranzacții"


def _distanta(rand) -> float:
    from src.features import haversine_km
    return float(haversine_km(rand["lat"], rand["long"], rand["merch_lat"], rand["merch_long"]))


def _incarca_exemplu(ist: pd.DataFrame, frauda: bool) -> None:
    """Completează formularul cu o tranzacție reală din istoricul de referință."""
    cand = ist[ist[C.COL_ETICHETA] == int(frauda)]
    if frauda:
        cand = cand[cand[C.COL_SUMA] > 150] if (cand[C.COL_SUMA] > 150).any() else cand
    rand = cand.sample(1, random_state=int(np.random.default_rng().integers(1e6))).iloc[0]
    cl = _clienti(ist)
    st.session_state["f_client"] = _eticheta_client(rand[C.COL_CARD], cl.loc[rand[C.COL_CARD]])
    st.session_state["f_data"] = rand[C.COL_TIMP].date()
    st.session_state["f_ora"] = rand[C.COL_TIMP].time().replace(microsecond=0)
    st.session_state["f_suma"] = float(rand[C.COL_SUMA])
    st.session_state["f_categorie"] = rand["category"]
    st.session_state["f_comerciant"] = rand["merchant"]
    st.session_state["f_distanta"] = round(_distanta(rand), 1)
    st.session_state["f_eticheta_reala"] = "o fraudă" if rand[C.COL_ETICHETA] == 1 else "o plată legitimă"
    st.session_state["f_rezultat"] = None


def _muta_comerciant(lat, lon, km):
    """Plasează comerciantul la `km` de domiciliu (spre nord-est) pentru formular."""
    d = km / 6371.0 / math.sqrt(2)
    return lat + math.degrees(d), lon + math.degrees(d) / max(math.cos(math.radians(lat)), 1e-6)


# ------------------------------------------------------------------ afișare
def _verdict(scor: float, prag: float) -> None:
    if scor >= prag:
        clasa, decizie = "ridicat", "Suspectă: se recomandă blocarea și verificarea"
    elif scor >= prag / 3:
        clasa, decizie = "moderat", "De urmărit: sub prag, dar peste obiceiul clientului"
    else:
        clasa, decizie = "scazut", "Obișnuită pentru acest client"
    st.markdown(f"""
    <div class="verdict {clasa}">
      <div class="eticheta">Riscul de fraudă după XGBoost</div>
      <div class="procent">{ui.nr(100 * scor, 1)}%</div>
      <div class="decizie">{decizie}</div>
      <div class="scara"><div class="umplere" style="width:{100 * scor:.1f}%"></div>
           <div class="prag" style="left:calc({100 * prag:.1f}% - 1px)"></div></div>
      <div class="legenda"><span>0%</span><span>prag {ui.nr(100 * prag, 1)}%</span><span>100%</span></div>
    </div>""", unsafe_allow_html=True)


def _registru(scoruri: dict, alerte: dict, praguri: dict) -> None:
    randuri = []
    for m, s in scoruri.items():
        culoare = ui.CULORI_MODELE[m]
        tag = ('<span class="eticheta-alerta da">semnalată</span>' if alerte[m]
               else '<span class="eticheta-alerta nu">nesemnalată</span>')
        randuri.append(
            f'<div class="rand"><div>{ui.NUME_MODELE[m]}</div>'
            f'<div class="bara"><span style="width:{100 * s:.1f}%;background:{culoare}"></span>'
            f'<i style="left:{100 * praguri[m]:.1f}%"></i></div>'
            f'<div class="val">{ui.nr(100 * s, 1)}%</div><div>{tag}</div></div>')
    st.markdown(f'<div class="registru">{"".join(randuri)}</div>', unsafe_allow_html=True)
    st.markdown('<p class="nota">Linia verticală arată pragul fiecărei metode. Pentru Isolation Forest '
                "și LOF scorul e percentila tranzacției față de datele de antrenare.</p>", unsafe_allow_html=True)


def _criterii(criterii: dict, f: pd.Series) -> None:
    valori = {
        "suma_mare": f"abatere {ui.nr(f['amt_zscore_client'], 1)} σ, de {ui.nr(f['amt_ratio_avg'], 1)} ori media",
        "frecventa_mare": f"{int(f['tx_count_1h'])} în ultima oră, {ui.nr(f['time_since_last'], 2)} h de la cea anterioară",
        "distanta_mare": f"{ui.nr(f['distance_km'], 0)} km",
        "ora_noapte": f"ora {int(f['hour'])}",
        "comerciant_nou": ("comerciant nou" if f["new_merchant"] else "comerciant cunoscut") + ", " +
                          ("categorie nouă" if f["new_category"] else "categorie cunoscută"),
        "categorie_risc": f["category"],
    }
    html = ""
    for k, regula in C.REGULI.items():
        da = bool(criterii[k])
        html += (f'<div class="criteriu"><span class="punct" style="background:{ui.ROSU if da else ui.LINIE}"></span>'
                 f'<div><div style="font-weight:{600 if da else 400}">{regula["titlu"]}</div>'
                 f'<div class="detaliu">{valori[k]}</div></div></div>')
    st.markdown(html, unsafe_allow_html=True)


def _grafic_shap(shap: pd.Series) -> None:
    top = shap.head(10)[::-1]
    culori = [ui.ROSU if v > 0 else ui.ALBASTRU for v in top.values]
    fig = go.Figure(go.Bar(x=top.values, y=[ui.denumire(c) for c in top.index], orientation="h",
                           marker_color=culori, hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
    fig.update_xaxes(title="Contribuția la scor (log-odds)", zeroline=True, zerolinecolor=ui.GRI)
    st.plotly_chart(ui.stil_grafic(fig, 360, legenda=False), use_container_width=True)


# ------------------------------------------------------------------ pagina
def pagina() -> None:
    ui.antet("Verificare tranzacție",
             "Alegeți clientul și completați datele plății. Profilul se construiește din tranzacțiile "
             "anterioare ale cardului, din setul de testare.")
    if not ui.cere_sistem():
        return
    ui.banda_demo()
    sistem = ui.sistem()
    ist = ui.istoric_referinta()
    clienti = _clienti(ist)
    optiuni = [CLIENT_NOU] + [_eticheta_client(c, r) for c, r in clienti.iterrows()]
    harta = dict(zip(optiuni[1:], clienti.index))

    b1, b2, _ = st.columns([1, 1, 2])
    if b1.button("Încarcă o fraudă reală", icon=":material/warning:", use_container_width=True):
        _incarca_exemplu(ist, True)
    if b2.button("Încarcă o plată obișnuită", icon=":material/check_circle:", use_container_width=True):
        _incarca_exemplu(ist, False)

    stanga, dreapta = st.columns([5, 7], gap="large")

    with stanga:
        st.subheader("Datele tranzacției")
        st.session_state.setdefault("f_client", optiuni[1] if len(optiuni) > 1 else optiuni[0])
        client = st.selectbox("Clientul", optiuni, key="f_client")
        card = harta.get(client)
        profil = ist[ist[C.COL_CARD] == card].iloc[-1] if card is not None else None

        if profil is None:
            with st.expander("Datele clientului nou", expanded=True):
                c1, c2 = st.columns(2)
                gen = c1.selectbox("Sexul", ["F", "M"])
                varsta = c2.number_input("Vârsta", 18, 100, 40)
                oras = c1.text_input("Localitatea", "Springfield")
                stat = c2.text_input("Statul", "IL")
                pop = c1.number_input("Populația localității", 100, 5_000_000, 50_000, step=1000)
                job = c2.text_input("Profesia", "Engineer")
                lat = c1.number_input("Latitudine domiciliu", -90.0, 90.0, 39.78, format="%.4f")
                lon = c2.number_input("Longitudine domiciliu", -180.0, 180.0, -89.65, format="%.4f")
            demografic = {"cc_num": 1, "gender": gen, "city": oras, "state": stat, "city_pop": pop, "job": job,
                          "lat": lat, "long": lon, "dob": str(dt.date.today().year - varsta) + "-06-15"}
        else:
            demografic = {k: profil[k] for k in ["cc_num", "gender", "city", "state", "city_pop", "job",
                                                 "lat", "long", "dob"] if k in profil.index}
            if "dob" not in demografic and "age" in profil.index:
                demografic["age"] = profil["age"]
            st.caption(f"{profil['gender']}, {profil['job']}, {profil['city']} ({profil['state']}). "
                       f"Ultima tranzacție în istoric: {clienti.loc[card, 'ultima']:%d.%m.%Y %H:%M}.")

        implicit_data = (clienti.loc[card, "ultima"] + pd.Timedelta(hours=6)) if card is not None \
            else pd.Timestamp("2020-12-01 14:00")
        st.session_state.setdefault("f_data", implicit_data.date())
        st.session_state.setdefault("f_ora", dt.time(14, 30))
        st.session_state.setdefault("f_suma", 85.0)
        categorii = sistem.preprocesor.categorii_one_hot["category"]
        st.session_state.setdefault("f_categorie", "shopping_net" if "shopping_net" in categorii else categorii[0])
        comercianti = sorted(ist["merchant"].unique())
        st.session_state.setdefault("f_comerciant", comercianti[0])
        st.session_state.setdefault("f_distanta", 40.0)
        if st.session_state["f_comerciant"] not in comercianti:
            comercianti = [st.session_state["f_comerciant"]] + comercianti

        c1, c2 = st.columns(2)
        data = c1.date_input("Data", key="f_data", format="DD.MM.YYYY")
        ora = c2.time_input("Ora", key="f_ora", step=60)
        suma = st.number_input("Suma ($)", min_value=0.01, step=10.0, key="f_suma")
        categorie = st.selectbox("Categoria comerțului", categorii, key="f_categorie")
        comerciant = st.selectbox("Comerciantul", comercianti, key="f_comerciant")
        distanta = st.slider("Distanța de la domiciliu (km)", 0.0, 500.0, key="f_distanta", step=0.5)

        if st.button("Verifică tranzacția", type="primary", use_container_width=True):
            mlat, mlon = _muta_comerciant(float(demografic["lat"]), float(demografic["long"]), distanta)
            tranz = {**demografic, C.COL_TIMP: pd.Timestamp.combine(data, ora), C.COL_SUMA: suma,
                     "category": categorie, "merchant": comerciant, "merch_lat": mlat, "merch_long": mlon}
            with st.spinner("Calculez profilul și scorurile…"):
                st.session_state["f_rezultat"] = sistem.analizeaza_tranzactie(
                    tranz, ist if card is not None else None)

        if st.session_state.get("f_eticheta_reala"):
            st.caption(f"Exemplul încărcat este, în realitate, **{st.session_state['f_eticheta_reala']}**.")

    with dreapta:
        rez = st.session_state.get("f_rezultat")
        if not rez:
            st.info("Completați formularul și apăsați „Verifică tranzacția”. Puteți porni și de la "
                    "un exemplu real, cu butoanele de sus.", icon=":material/credit_card:")
            return
        _verdict(rez["scoruri"]["xgboost"], sistem.praguri["xgboost"])
        st.write("")
        st.markdown("**Toate metodele**")
        _registru(rez["scoruri"], rez["alerte"], sistem.praguri)

        st.write("")
        st.markdown("**Criteriile de suspiciune**")
        _criterii(rez["criterii"], rez["caracteristici"])
        st.caption(f"Profil construit din {rez['tranzactii_anterioare']} tranzacții anterioare ale cardului.")

        st.markdown("**De ce a decis așa XGBoost**")
        st.markdown('<p class="nota">Cele mai influente 10 caracteristici. Roșu împinge spre fraudă, '
                    "albastru spre plată legitimă.</p>", unsafe_allow_html=True)
        _grafic_shap(rez["shap"])

        with st.expander("Toate caracteristicile calculate"):
            f = rez["caracteristici"]
            t = pd.DataFrame({"caracteristica": [ui.denumire(c) for c in C.CARACTERISTICI_DERIVATE],
                              "valoare": [f[c] for c in C.CARACTERISTICI_DERIVATE]})
            t["valoare"] = t["valoare"].map(lambda v: ui.nr(float(v), 2))
            st.dataframe(t, hide_index=True, use_container_width=True)
