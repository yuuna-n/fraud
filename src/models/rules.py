"""
Regulile de referință (§1.2, §2.4) – abordarea clasică, folosită ca punct de
comparație. Fiecare criteriu din tabelul 2.4 adaugă un număr de puncte
(ponderea din config.REGULI); scorul final este suma punctelor raportată la
maximul posibil. Pragul (câte puncte declanșează alerta) se alege pe validare,
la fel ca pentru celelalte modele.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..features import criterii_suspiciune


def scor_reguli(df: pd.DataFrame) -> np.ndarray:
    crit = criterii_suspiciune(df)
    ponderi = pd.Series({k: v["pondere"] for k, v in C.REGULI.items()})
    puncte = crit[ponderi.index].astype(int).mul(ponderi, axis=1).sum(axis=1)
    return (puncte / ponderi.sum()).to_numpy(dtype=float)
