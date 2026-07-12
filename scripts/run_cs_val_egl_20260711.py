"""
Campaña de Cuckoo Search sobre las 64 instancias val/egl/gdb faltantes.

Reutiliza ÍNTEGRA la maquinaria validada del mini-grid
(``run_cs_minigrid_20260710``): warm-start Path-Scanning mejor-de-5,
Path Relinking (umbral 30), cap uniforme de 10^6 evaluaciones / 300 s,
semillas deterministas y consolidación de CSV parciales por instancia.

La única diferencia es la configuración: en lugar de la malla 3×3, se corre
ÚNICAMENTE la combinación ganadora del mini-grid del 2026-07-10
(gap medio -10.82% sobre val1A/val5C/egl-e1-A):

    factor_pasos = 0.25    (pasos de Lévy cortos)
    p_inter      = 0.60    (sesgo inter alto, consistente con la clase egl
                            de la campaña principal)

y el conjunto de instancias pasa de las 3 representativas a las 64 de la
campaña (mismas listas que ``run_val_egl_20260710``).

Implementación: importamos el módulo del mini-grid y sobreescribimos sus
constantes de malla ANTES de invocar su ``main()``. Las tareas viajan
autocontenidas (dataclass ``Tarea`` con instancia/factor/p_inter/semilla),
por lo que los workers no dependen de las constantes parcheadas; los
parámetros fijos (pa=0.15, beta=1.3, num_nidos=25, umbral PR=30) se heredan
sin cambios.

Uso:
    python scripts/run_cs_val_egl_20260711.py [--workers N] [--repeticiones R]
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import run_cs_minigrid_20260710 as minigrid  # noqa: E402
# Listas de instancias canónicas de la campaña (única fuente de verdad).
from run_val_egl_20260710 import INSTANCIAS_VAL_GDB, INSTANCIAS_EGL  # noqa: E402

# Configuración ganadora del mini-grid: malla degenerada de 1×1 combinación.
minigrid.INSTANCIAS = list(INSTANCIAS_VAL_GDB) + list(INSTANCIAS_EGL)
minigrid.FACTOR_PASOS_GRID = (0.25,)
minigrid.P_INTER_GRID = (0.6,)

if __name__ == "__main__":
    # Reps default 5 para igualar la campaña principal (el mini-grid usaba 2).
    argv = sys.argv[1:]
    if not any(a.startswith("--repeticiones") for a in argv):
        argv += ["--repeticiones", "5"]
    minigrid.main(argv)
