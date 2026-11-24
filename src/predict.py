"""
Sistemul complet, gata de utilizare: preprocesorul, cele patru modele și
pragurile lor, salvate împreună într-un singur fișier. Interfața folosește
numai această clasă.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .data_loading import curata
from .explain import explicatie_tranzactie
from .features import construieste_caracteristici, criterii_suspiciune
from .models.rules import scor_reguli
from .models.supervised import scor_xgboost

FISIER_SISTEM = C.DIR_MODELE / "sistem.joblib"


@dataclass
class SistemFrauda:
    preprocesor: object
    xgboost: object
    iforest: object
    lof: object
    praguri: dict[str, float]
    info: dict = field(default_factory=dict)

    # ------------------------------------------------------------ salvare
    def salveaza(self, cale: Path = FISIER_SISTEM) -> None:
        cale.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, cale)

    @staticmethod
    def incarca(cale: Path = FISIER_SISTEM) -> "SistemFrauda":
        return joblib.load(cale)

    # ------------------------------------------------------------ scorare
    @staticmethod
    def pregateste(brute: pd.DataFrame) -> pd.DataFrame:
        """Curățare + caracteristici derivate (istoricul = tranzacțiile din tabel)."""
        return construieste_caracteristici(curata(brute))

    def scoruri(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pentru un tabel care are deja caracteristicile derivate, întoarce scorul
        fiecărui model (0–1) și decizia la pragul ales pe validare.
        """
        X, X_nes = self.preprocesor.transform(df)
        rez = pd.DataFrame(index=df.index)
        rez["xgboost"] = scor_xgboost(self.xgboost, X)
        rez["iforest"] = self.iforest.scor(X_nes)
        rez["lof"] = self.lof.scor(X_nes)
        rez["reguli"] = scor_reguli(df)
        for m, prag in self.praguri.items():
            rez[f"alerta_{m}"] = rez[m] >= prag
        return rez

    def analizeaza_fisier(self, brute: pd.DataFrame) -> pd.DataFrame:
        """Analiza unui fișier întreg: caracteristici + scoruri + criterii."""
        df = self.pregateste(brute)
        crit = criterii_suspiciune(df).add_prefix("criteriu_")
        return pd.concat([df, self.scoruri(df), crit], axis=1)

    def analizeaza_tranzactie(self, tranzactie: dict, istoric: pd.DataFrame | None = None) -> dict:
        """
        Analiza unei singure tranzacții. Profilul clientului se construiește din
        `istoric` (tranzacțiile ANTERIOARE ale aceluiași card); fără istoric,
        clientul este tratat ca nou.
        """
        noua = pd.DataFrame([tranzactie])
        noua[C.COL_TIMP] = pd.to_datetime(noua[C.COL_TIMP])
        if istoric is not None and len(istoric):
            ist = istoric[(istoric[C.COL_CARD] == tranzactie[C.COL_CARD])
                          & (istoric[C.COL_TIMP] < noua[C.COL_TIMP].iloc[0])].copy()
        else:
            ist = pd.DataFrame(columns=noua.columns)
        ist["_noua"] = False
        noua["_noua"] = True
        comun = [c for c in noua.columns if c in ist.columns]
        tot = pd.concat([ist[comun], noua[comun]], ignore_index=True)
        tot["trans_num"] = np.arange(len(tot)).astype(str)  # evită eliminarea ca duplicat

        df = construieste_caracteristici(curata(tot))
        rand = df[df["_noua"]].drop(columns="_noua").iloc[[-1]]
        scor = self.scoruri(rand).iloc[0]
        X, _ = self.preprocesor.transform(rand)
        return {
            "caracteristici": rand.iloc[0],
            "scoruri": {m: float(scor[m]) for m in self.praguri},
            "alerte": {m: bool(scor[f"alerta_{m}"]) for m in self.praguri},
            "criterii": criterii_suspiciune(rand).iloc[0].to_dict(),
            "shap": explicatie_tranzactie(self.xgboost, X),
            "tranzactii_anterioare": int(len(ist)),
        }
