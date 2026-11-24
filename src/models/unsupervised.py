"""
Detectarea nesupravegheată a anomaliilor (§1.2, §2.4): Isolation Forest și
Local Outlier Factor. Niciunul nu folosește etichetele la antrenare.

  * Isolation Forest izolează observațiile prin partiționări aleatorii; anomaliile
    sunt izolate mai repede (la o adâncime mai mică în arbori).
  * LOF compară densitatea locală a unui punct cu cea a vecinilor săi; punctele
    din zone mult mai puțin dense primesc un scor ridicat.

Scorurile brute ale celor doi algoritmi au scări diferite. Pentru a le putea
compara cu probabilitatea XGBoost și a le afișa ca „risc", fiecare scor se
transformă în percentila sa față de distribuția scorurilor din setul de
antrenare (0 = cea mai obișnuită tranzacție, aproape de 1 = cea mai neobișnuită).
Peste maximul din antrenare scorul continuă să crească lin spre 1, ca
anomaliile extreme să rămână ordonate între ele (altfel toate ar primi 1,0).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from .. import config as C


class DetectorAnomalii:
    """Învelitoare comună: antrenare, scor brut și scor normalizat (percentilă)."""

    def __init__(self, tip: str) -> None:
        if tip not in ("iforest", "lof"):
            raise ValueError(tip)
        self.tip = tip
        self.model = None
        self.referinta: np.ndarray | None = None  # scoruri sortate din train

    def antreneaza(self, X: np.ndarray, rng: np.random.Generator | None = None) -> "DetectorAnomalii":
        rng = rng or np.random.default_rng(C.SEED)
        if self.tip == "iforest":
            self.model = IsolationForest(**C.IFOREST_PARAMETRI).fit(X)
            X_ref = X if len(X) <= 200_000 else X[rng.choice(len(X), 200_000, replace=False)]
        else:
            # LOF pe un eșantion (calculul distanțelor e costisitor pe ~1 mil. de rânduri)
            idx = rng.choice(len(X), min(C.LOF_ESANTION, len(X)), replace=False)
            self.model = LocalOutlierFactor(**C.LOF_PARAMETRI).fit(X[idx])
            X_ref = X[rng.choice(len(X), min(50_000, len(X)), replace=False)]
        self.referinta = np.sort(self.scor_brut(X_ref))
        return self

    def scor_brut(self, X: np.ndarray) -> np.ndarray:
        """Scor de anomalie: cu cât e mai mare, cu atât tranzacția e mai neobișnuită."""
        return -self.model.score_samples(X)

    def scor(self, X: np.ndarray) -> np.ndarray:
        """Scorul normalizat în [0, 1] (percentila față de setul de antrenare)."""
        bruta = self.scor_brut(X)
        ref = self.referinta
        n = len(ref)
        # funcția de repartiție empirică, interpolată liniar: minim -> 0, maxim -> 1 - 1/n
        p = np.interp(bruta, ref, np.linspace(0, 1 - 1 / n, n))
        # coada: peste maxim, apropiere exponențială de 1 (continuă și strict crescătoare)
        scara = max(ref[-1] - ref[int(0.99 * (n - 1))], 1e-9)
        coada = 1 - (1 / n) * np.exp(-(bruta - ref[-1]) / scara)
        return np.where(bruta > ref[-1], coada, p)
