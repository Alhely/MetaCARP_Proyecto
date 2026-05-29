"""
Experimento warmstart_greedy_20260529 para TS Simple.

Apila sobre ``budget_20260528`` + ``p_inter_pr_20260528``:
  - PS warmstart (Path-Scanning como solucion inicial).
  - Evaluador greedy (orientacion dinamica por tarea).

Smoke test inicial (1 corrida): gdb19 -> BKS, kshs1 -> BKS,
gdb10 52% -> 3.27%, gdb3 49% -> BKS, kshs5 38% -> 8.63%.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _warmstart_greedy_20260529_common import (
    ALPHA_INTER,
    ITERS_TS,
    KICK_ITERS,
    LAMBDA_FACTOR,
    MAX_RESETS,
    MAX_SIN_MEJ_TS,
    P_INTER,
    P_PR,
    TareaExp,
    aplicar_patches_warmstart_greedy,
    correr_grid,
)

MODULO_MH       = "metacarp.busqueda_tabu_simple"
PREFIJO_CSV     = "tabu_simple"
SUBCARPETA      = "warmstart_greedy_20260529_tabu_simple"
EXPERIMENTO_TAG = "tabu_simple_warmstart_greedy"


def _ejecutar_una(tarea: TareaExp) -> tuple[TareaExp, str, dict | None, str | None]:
    """Worker: una corrida de TS Simple con PS warmstart + evaluador greedy."""
    try:
        aplicar_patches_warmstart_greedy(
            MODULO_MH,
            p_pr=P_PR,
            lambda_factor=LAMBDA_FACTOR,
            root_warmstart=tarea.root,
        )

        from metacarp import busqueda_tabu_simple_desde_instancia
        from metacarp.strict_intra_inter_20260524 import OPERADORES_STRICT_5

        res = busqueda_tabu_simple_desde_instancia(
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
                "experimento":       "warmstart_greedy_20260529",
                "warmstart":         "path_scanning",
                "evaluador":         "greedy_orientation",
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
        label_mh="TS Simple",
        prefijo_csv=PREFIJO_CSV,
        subcarpeta_destino=SUBCARPETA,
        modulo_patchear=MODULO_MH,
        ejecutar_una=_ejecutar_una,
        descripcion_cli=(
            "warmstart_greedy_20260529 para TS Simple: Path-Scanning warmstart + "
            "evaluador greedy + budget x25 + lambda x10 + p_inter + PR."
        ),
        experimento=EXPERIMENTO_TAG,
    )


if __name__ == "__main__":
    main()
