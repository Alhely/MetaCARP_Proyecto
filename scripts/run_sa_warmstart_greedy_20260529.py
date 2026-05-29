"""
Experimento warmstart_greedy_20260529 para Recocido Simulado (SA).
PS warmstart + evaluador greedy + reheat extendido + lambda x10 + p_inter + PR.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _warmstart_greedy_20260529_common import (
    ALPHA_INTER,
    KICK_ITERS,
    LAMBDA_FACTOR,
    MAX_RESETS,
    P_INTER,
    P_PR,
    TareaExp,
    aplicar_patches_warmstart_greedy,
    correr_grid,
)

MODULO_MH       = "metacarp.recocido_simulado"
PREFIJO_CSV     = "sa"
SUBCARPETA      = "warmstart_greedy_20260529_sa"
EXPERIMENTO_TAG = "sa_warmstart_greedy"

# Mismos parametros que SA budget_20260528.
SA_KICK_ITERS  = 20
SA_PATIENCE    = 100
SA_MAX_REHEATS = 30
SA_TEMP_MIN    = 1e-4


def _ejecutar_una(tarea: TareaExp) -> tuple[TareaExp, str, dict | None, str | None]:
    """Worker: una corrida de SA con PS warmstart + evaluador greedy."""
    try:
        aplicar_patches_warmstart_greedy(
            MODULO_MH,
            p_pr=P_PR,
            lambda_factor=LAMBDA_FACTOR,
            root_warmstart=tarea.root,
        )

        from metacarp import recocido_simulado_desde_instancia
        from metacarp.strict_intra_inter_20260524 import OPERADORES_STRICT_5

        res = recocido_simulado_desde_instancia(
            tarea.instancia,
            root=tarea.root,
            operadores=OPERADORES_STRICT_5,
            temperatura_minima=SA_TEMP_MIN,
            patience=SA_PATIENCE,
            reheat_factor=0.5,
            max_reheats_sin_mejora=SA_MAX_REHEATS,
            max_iter_sin_mejora_kick=SA_KICK_ITERS,
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
                "sa_kick_iters":     str(SA_KICK_ITERS),
                "sa_patience":       str(SA_PATIENCE),
                "sa_max_reheats":    str(SA_MAX_REHEATS),
                "sa_temp_min":       str(SA_TEMP_MIN),
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
        label_mh="SA",
        prefijo_csv=PREFIJO_CSV,
        subcarpeta_destino=SUBCARPETA,
        modulo_patchear=MODULO_MH,
        ejecutar_una=_ejecutar_una,
        descripcion_cli=(
            "warmstart_greedy_20260529 para SA: Path-Scanning warmstart + "
            "evaluador greedy + reheat extendido + lambda x10 + p_inter + PR."
        ),
        experimento=EXPERIMENTO_TAG,
    )


if __name__ == "__main__":
    main()
