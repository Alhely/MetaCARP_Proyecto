"""
Experimento warmstart_greedy_20260529 para Cuckoo Search.
PS warmstart + evaluador greedy + budget x10 (factor_iter) + lambda x10 + p_inter + PR.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _warmstart_greedy_20260529_common import (
    ALPHA_INTER,
    FACTOR_ITER_POP,
    KICK_ITERS,
    LAMBDA_FACTOR,
    MAX_RESETS,
    MAX_SIN_MEJ_POP,
    P_INTER,
    P_PR,
    TareaExp,
    aplicar_patches_warmstart_greedy,
    correr_grid,
)

MODULO_MH       = "metacarp.cuckoo_search"
PREFIJO_CSV     = "cuckoo"
SUBCARPETA      = "warmstart_greedy_20260529_cuckoo"
EXPERIMENTO_TAG = "cuckoo_warmstart_greedy"


def _ejecutar_una(tarea: TareaExp) -> tuple[TareaExp, str, dict | None, str | None]:
    """Worker: una corrida de Cuckoo Search con PS warmstart + evaluador greedy."""
    try:
        aplicar_patches_warmstart_greedy(
            MODULO_MH,
            p_pr=P_PR,
            lambda_factor=LAMBDA_FACTOR,
            root_warmstart=tarea.root,
        )

        from metacarp import cuckoo_search_desde_instancia
        from metacarp.strict_intra_inter_20260524 import OPERADORES_STRICT_5

        res = cuckoo_search_desde_instancia(
            tarea.instancia,
            root=tarea.root,
            operadores=OPERADORES_STRICT_5,
            factor_iter=FACTOR_ITER_POP,
            max_iter_sin_mejora=MAX_SIN_MEJ_POP,
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
                "factor_iter":       str(FACTOR_ITER_POP),
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
        label_mh="Cuckoo Search",
        prefijo_csv=PREFIJO_CSV,
        subcarpeta_destino=SUBCARPETA,
        modulo_patchear=MODULO_MH,
        ejecutar_una=_ejecutar_una,
        descripcion_cli=(
            "warmstart_greedy_20260529 para Cuckoo: Path-Scanning warmstart + "
            "evaluador greedy + factor_iter=200 + lambda x10 + p_inter + PR."
        ),
        experimento=EXPERIMENTO_TAG,
    )


if __name__ == "__main__":
    main()
