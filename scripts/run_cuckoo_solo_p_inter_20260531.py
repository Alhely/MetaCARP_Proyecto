"""
Approach solo_p_inter_20260531 para Cuckoo Search (CK).

Selector p_inter + seleccion uniforme de operador dentro del grupo, sin PR/kick/
AOS/budget, bajo el evaluador de costo greedy (nativo). Grid search en dos fases
(calibracion + final). Toda la logica vive en ``_solo_p_inter_20260531_common``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _solo_p_inter_20260531_common import main

if __name__ == "__main__":
    main(mh_fijo="cuckoo")
