"""
Variante experimental p_inter_exp_2026050524 para Busqueda Abejas Simple (ABC).

Cambios vs. ``run_abc_simple_automatico.py``:

  1. **Restriccion posicional (parcial en ABC)**: ancla la 1a tarea
     servida tras el deposito en las fases de empleadas y observadoras
     (monkey-patch de ``generar_vecino_ids`` dentro de
     ``metacarp.abejas_simple``, unica MH que usa el dispatcher ``_ids``).
     **NO** afecta a la fase scout, que reemplaza fuentes agotadas con
     soluciones COMPLETAMENTE aleatorias (Karaboga 2005 estricto). Los
     scouts no usan el dispatcher de vecinos: arman la solucion desde
     cero a traves de ``_generar_solucion_aleatoria`` en
     ``metacarp/abejas_simple.py``. Por tanto la 1a tarea de la solucion
     inicial puede no preservarse si la fuente inicial es agotada y
     reemplazada por un scout aleatorio. Es un compromiso documentado.
  2. **p_inter estatico decidido por la solucion inicial**:
        factible    → p_inter = 0.5
        infactible  → p_inter = 0.95
     ABC Simple ya no expone ``alpha_inter``: deriva internamente
     ``alpha_inter = max(p_inter, 0.8)``. Con p_inter=0.95 (caso infactible)
     la politica queda exactamente como la pedimos; con p_inter=0.5 (caso
     factible) ABC subira automaticamente a 0.8 cuando aparezca una
     violacion intermedia. Es un compromiso documentado, no un bug.
  3. **Grid simplificado**: instancia x repeticion (115 corridas). Demas
     hiperparametros (num_fuentes, limite_abandono, factor_*) usan
     defaults instance-aware del wrapper.
  4. **Salida**: ``experimentos/p_inter_exp_2026050524_abc_simple/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _p_inter_exp_2026050524_common import (
    TareaExp,
    aplicar_patch_vecindario,
    calcular_violacion_inicial,
    correr_grid,
    decidir_p_inter,
)

MODULO_MH       = "metacarp.abejas_simple"
SIMBOLO_PATCH   = "generar_vecino_ids"   # ABC usa el dispatcher ids-based
PREFIJO_CSV     = "abc_simple"
SUBCARPETA      = "p_inter_exp_2026050524_abc_simple"
EXPERIMENTO_TAG = "abc_simple_p_inter_exp"


def _ejecutar_una(tarea: TareaExp) -> tuple[TareaExp, str, dict | None, str | None]:
    """Worker: una corrida de ABC con la politica experimental.

    ABC Simple solo acepta ``p_inter`` (el ``alpha_inter`` interno es
    ``max(p_inter, 0.8)`` por diseño). Aceptamos esto como compromiso.
    """
    try:
        aplicar_patch_vecindario(MODULO_MH, SIMBOLO_PATCH)
        viol_ini = calcular_violacion_inicial(tarea.instancia, tarea.root)
        p_inter_fijo = decidir_p_inter(viol_ini)

        from metacarp import busqueda_abejas_simple_desde_instancia

        res = busqueda_abejas_simple_desde_instancia(
            tarea.instancia,
            root=tarea.root,
            # Solo p_inter (ABC Simple eliminó alpha_inter de su firma).
            p_inter=p_inter_fijo,
            usar_gpu=True,                # ABC se beneficia mas de GPU (lote en fase observadoras)
            semilla=None,
            repeticion=tarea.repeticion,
            guardar_csv=True,
            ruta_csv=tarea.ruta_csv_parcial,
            guardar_historial=False,
            extra_csv={
                "experimento": "p_inter_exp_2026050524",
                "viol_inicial": float(viol_ini),
                "p_inter_fijo": p_inter_fijo,
                "inicial_factible": viol_ini <= 1e-12,
                "restriccion_posicional": True,
            },
        )
        info = {
            "costo":  res.mejor_costo,
            "tiempo": res.tiempo_segundos,
            "viol_inicial": viol_ini,
            "p_inter": p_inter_fijo,
        }
        return (tarea, "ok", info, None)
    except Exception as exc:  # noqa: BLE001
        return (tarea, "fail", None, f"{type(exc).__name__}: {exc}")


def main() -> None:
    correr_grid(
        label_mh="ABC Simple",
        prefijo_csv=PREFIJO_CSV,
        subcarpeta_destino=SUBCARPETA,
        wrapper_fn=None,
        modulo_patchear=MODULO_MH,
        simbolo_patchear=SIMBOLO_PATCH,
        ejecutar_una=_ejecutar_una,
        descripcion_cli=(
            "Variante experimental p_inter_exp_2026050524 para ABC Simple: "
            "preserva la 1a tarea tras el deposito y fija p_inter por la inicial."
        ),
        experimento=EXPERIMENTO_TAG,
    )


if __name__ == "__main__":
    main()
