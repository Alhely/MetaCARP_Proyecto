"""
Approach binario_capacidad_20260531 para Artificial Bee Colony simple (ABC).

Selector BINARIO determinista basado en capacidad (viola -> INTER, factible ->
INTRA) + seleccion uniforme de operador dentro del grupo, sin PR/kick/AOS/budget,
bajo el evaluador de costo greedy (nativo). Grid search en dos fases. Toda la
logica vive en ``_binario_capacidad_20260531_common``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _binario_capacidad_20260531_common import main

if __name__ == "__main__":
    main(mh_fijo="abc_simple")
