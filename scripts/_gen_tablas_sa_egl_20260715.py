"""
Tablas en inglés de RECOCIDO SIMULADO restringidas a la clase egl (24
instancias), para el paper centrado en SA:

  1. Mejor resultado/gap por instancia con parámetros de la corrida ganadora
     (misma estructura que sa_results_table_en_20260713.tex).
  2. Soluciones explícitas como arcos (u,v) — sin marcadores TR — en
     landscape con márgenes de 1 cm.

Todas las corridas egl provienen de la campaña de transferencia
(config de clase: alpha=0.95, p_inter=0.6, pr_aislado, 10^6 eval / 300 s).

Salidas:
    resultados/sa_egl_results_table_en_20260715.tex
    resultados/sa_egl_solutions_table_en_20260715.tex

Uso:
    python scripts/_gen_tablas_sa_egl_20260715.py
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))
from run_val_egl_20260710 import INSTANCIAS_EGL  # noqa: E402
from _gen_tabla_cuckoo_20260712 import _deltas, _mapas_tr  # noqa: E402
from _gen_tablas_mh_20260713 import (  # noqa: E402
    MHS, _mejores, _render_resultados, _render_soluciones,
)


def main() -> None:
    spec = dict(MHS["sa"])  # copia superficial: solo cambiamos metadatos
    spec["label_res"] = "tab:sa-egl-per-instance"
    spec["label_sol"] = "tab:sa-egl-solutions"
    # Todas las filas egl salen de la campaña de transferencia (T).
    spec["caption_cfg"] = (
        r" All rows come from the egl transfer campaign: class-tuned"
        r" configuration ($\alpha = 0.95$, $p_{inter} = 0.6$) with"
        r" \texttt{pr\_aislado} (Path Relinking on stagnation), budget of"
        r" $10^6$ evaluations / 300\,s per run, 5 repetitions.")

    deltas = _deltas()
    mapas = _mapas_tr()
    mejores = _mejores(spec)
    filas = [(i, mejores[i]) for i in INSTANCIAS_EGL if i in mejores]
    faltan = [i for i in INSTANCIAS_EGL if i not in mejores]
    if faltan:
        print(f"AVISO: sin corridas para {faltan}")

    _render_resultados(
        "sa-egl", spec, filas, deltas,
        RAIZ / "resultados" / "sa_egl_results_table_en_20260715.tex")
    _render_soluciones(
        "sa-egl", spec, filas, deltas, mapas,
        RAIZ / "resultados" / "sa_egl_solutions_table_en_20260715.tex")


if __name__ == "__main__":
    main()
