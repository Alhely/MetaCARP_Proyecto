"""
Campaña VDO sobre las 23 instancias pequeñas (gdb/kshs), 5 repeticiones.

Complemento de la campaña val/egl (``run_val_egl_20260710.py --solo-mh vdo``)
para que VDO tenga cobertura en las 87 instancias del corpus, como las demás
metaheurísticas. Reutiliza TODA la maquinaria del runner (Tarea, semillas
deterministas, PR aislado con umbral 30, consolidación de parciales); las
pequeñas usan la config de clase "val" (p_inter = 0.5), coherente con su
tamaño.

Presupuesto uniforme por corrida: 10^6 evaluaciones (max_niveles = 1e6/n²)
o 300 s de pared, lo primero que ocurra.

Salida: experimentos_vdo_20260714/small/corrida_<ts>/vdo/final/

Uso:
    python scripts/run_vdo_small_20260714.py [--workers N] [--repeticiones R]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))

from run_val_egl_20260710 import (  # noqa: E402
    MAX_EVALUACIONES_DEF, TIEMPO_LIMITE_DEF, _ejecutar_mh,
)
from _gen_tabla_cuckoo_20260712 import INSTANCIAS_SMALL  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--repeticiones", type=int, default=5)
    parser.add_argument("--semilla-base", type=int, default=0)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d-%H%M")
    salida = RAIZ / "experimentos_vdo_20260714" / "small" / f"corrida_{ts}"
    salida.mkdir(parents=True, exist_ok=True)

    print(f"Campaña VDO pequeñas | {len(INSTANCIAS_SMALL)} instancias × "
          f"{args.repeticiones} reps | workers={args.workers}")
    _ejecutar_mh(
        "vdo",
        salida_dir=salida,
        instancias_por_clase={"val": list(INSTANCIAS_SMALL)},
        repeticiones=args.repeticiones,
        workers=args.workers,
        root=None,
        semilla_base=args.semilla_base,
        max_evaluaciones=MAX_EVALUACIONES_DEF,
        tiempo_limite=TIEMPO_LIMITE_DEF,
    )
    print(f"Fin. Salida: {salida}")


if __name__ == "__main__":
    main()
