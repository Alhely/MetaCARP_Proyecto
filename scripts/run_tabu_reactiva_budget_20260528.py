"""
Experimento budget_20260528 para Reactive Tabu Search (RTS).

Cambios sobre el baseline p_inter_pr_20260528:

  1. PRESUPUESTO x25: ``iteraciones_max`` ~400 → 10 000,
     ``max_iter_sin_mejora`` → 2 500.
  2. KICK PROPORCIONAL: ``max_iter_sin_mejora_kick`` 30 → 300.
  3. SIN TOPE DE RESETS: ``max_resets=None``.
  4. LAMBDA x10.
  5. Selector p_inter + PR conservados identicos.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _budget_20260528_common import (
    ALPHA_INTER,
    ITERS_TS,
    KICK_ITERS,
    LAMBDA_FACTOR,
    MAX_RESETS,
    MAX_SIN_MEJ_TS,
    P_INTER,
    P_PR,
    TareaExp,
    aplicar_patches_budget,
    correr_grid,
)

MODULO_MH       = "metacarp.busqueda_tabu_reactiva"
PREFIJO_CSV     = "tabu_reactiva"
SUBCARPETA      = "budget_20260528_tabu_reactiva"
EXPERIMENTO_TAG = "tabu_reactiva_budget"


def _ejecutar_una(tarea: TareaExp) -> tuple[TareaExp, str, dict | None, str | None]:
    """Worker: una corrida de RTS con presupuesto x25 y lambda x10."""
    try:
        aplicar_patches_budget(MODULO_MH, p_pr=P_PR, lambda_factor=LAMBDA_FACTOR)

        from metacarp import busqueda_tabu_reactiva_desde_instancia
        from metacarp.strict_intra_inter_20260524 import OPERADORES_STRICT_5

        res = busqueda_tabu_reactiva_desde_instancia(
            tarea.instancia,
            root=tarea.root,
            operadores=OPERADORES_STRICT_5,
            iteraciones_max=ITERS_TS,
            max_iter_sin_mejora=MAX_SIN_MEJ_TS,
            max_iter_sin_mejora_kick=KICK_ITERS,
            max_resets=MAX_RESETS,
            usar_gpu=False,
            semilla=None,
            repeticion=tarea.repeticion,
            guardar_csv=True,
            ruta_csv=tarea.ruta_csv_parcial,
            guardar_historial=False,
            extra_csv={
                "experimento":       "budget_20260528",
                "selector":          "p_inter_probabilistico",
                "p_inter":           str(P_INTER),
                "alpha_inter":       str(ALPHA_INTER),
                "p_pr":              str(P_PR),
                "iters_max":         str(ITERS_TS),
                "kick_iters":        str(KICK_ITERS),
                "lambda_factor":     str(LAMBDA_FACTOR),
                "operadores_activos": "5 (2intra+3inter)",
            },
        )
        info = {
            "costo":    res.mejor_costo,
            "tiempo":   res.tiempo_segundos,
            "n_resets": res.n_resets_kick,
            "factible": res.mejor_solucion_factible_final,
        }
        return (tarea, "ok", info, None)
    except Exception as exc:  # noqa: BLE001
        return (tarea, "fail", None, f"{type(exc).__name__}: {exc}")


def main() -> None:
    correr_grid(
        label_mh="RTS",
        prefijo_csv=PREFIJO_CSV,
        subcarpeta_destino=SUBCARPETA,
        modulo_patchear=MODULO_MH,
        ejecutar_una=_ejecutar_una,
        descripcion_cli=(
            "budget_20260528 para RTS: iteraciones_max=10000, "
            "kick_iters=300, max_resets=None, lambda x10, "
            "selector p_inter (0.20) + PR (p_pr=0.5)."
        ),
        experimento=EXPERIMENTO_TAG,
    )


if __name__ == "__main__":
    main()
