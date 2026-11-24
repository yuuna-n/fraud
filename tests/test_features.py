"""
Teste pentru caracteristicile derivate: valori calculate de mână și absența
scurgerii de informație (tranzacția curentă nu intră în propriul profil).

Rulare:  python -m pytest tests -q
"""

import numpy as np
import pandas as pd
import pytest

from src.features import construieste_caracteristici, haversine_km


def _date():
    t = pd.to_datetime(["2020-01-01 10:00", "2020-01-01 10:20", "2020-01-01 10:50",
                        "2020-01-01 12:00", "2020-01-03 09:00"])
    return pd.DataFrame({
        "trans_date_trans_time": t, "cc_num": [1] * 5, "amt": [10.0, 20.0, 30.0, 40.0, 500.0],
        "merchant": ["a", "b", "a", "c", "d"], "category": ["x", "x", "y", "x", "z"],
        "lat": [45.0] * 5, "long": [28.0] * 5, "merch_lat": [45.0] * 5, "merch_long": [28.0] * 5,
    })


@pytest.mark.parametrize("unitate", ["s", "us", "ns"])
def test_ferestre_si_istoric(unitate):
    df = _date()
    df["trans_date_trans_time"] = df["trans_date_trans_time"].dt.as_unit(unitate)
    f = construieste_caracteristici(df)
    assert f["tx_count_1h"].tolist() == [0, 1, 2, 0, 0]       # 12:00 – fereastra (11:00, 12:00)
    assert f["tx_count_24h"].tolist() == [0, 1, 2, 3, 0]
    assert f["time_since_last"].iloc[1] == pytest.approx(20 / 60)
    assert f["new_merchant"].tolist() == [1, 1, 0, 1, 1]
    assert f["new_category"].tolist() == [1, 0, 1, 0, 1]
    # media anterioară pentru a 4-a tranzacție: (10+20+30)/3 = 20 -> raport 2
    assert f["amt_ratio_avg"].iloc[3] == pytest.approx(2.0)
    # abaterea: sumele anterioare 10,20,30 -> medie 20, sigma 10 -> z = (40-20)/10
    assert f["amt_zscore_client"].iloc[3] == pytest.approx(2.0)
    # ultima tranzacție: nimic în ultimele 30 de zile? ba da, 4 tranzacții -> media 25
    assert f["avg_amt_30d"].iloc[4] == pytest.approx(25.0)


def test_fara_scurgere_din_viitor():
    df = _date()
    complet = construieste_caracteristici(df)
    partial = construieste_caracteristici(df.iloc[:3])
    # caracteristicile primelor 3 tranzacții nu depind de tranzacțiile ulterioare
    cols = ["tx_count_1h", "amt_zscore_client", "amt_ratio_avg", "avg_amt_30d", "new_merchant"]
    pd.testing.assert_frame_equal(complet.loc[:2, cols], partial[cols])


def test_haversine():
    # Chișinău – Iași: aproximativ 97 km în linie dreaptă
    assert haversine_km(47.0105, 28.8638, 47.1585, 27.6014) == pytest.approx(97, abs=5)
