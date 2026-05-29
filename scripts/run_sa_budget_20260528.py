"""
Experimento budget_20260528 para Recocido Simulado (SA).

SA ya tiene ~27 000 iteraciones de Markov; el presupuesto extra viene de
extender el reheat: patience 10→100 y max_reheats_sin_mejora 5→30.
La temperatura minima baja de 1e-3 a 1e-4 para explorar mas niveles.

Cambios sobre el baseline p_inter_pr_20260528:

  1. REHEAT EXTENDIDO: ``patience=100``, ``max_reheats_sin_mejora=30``,
     ``temperatura_minima=1e-4``.
  2. KICK PROPORCIONAL: ``max_iter_sin_mejora_kick`` 5 → 20
     (mismo ratio respecto al nivel de temperatura).
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
    KICK_ITERS,
    LAMBDA_FACTOR,
    MAX_RESETS,
    P_INTER,
    P_PR,
    TareaExp,
    aplicar_patches_budget,
    correr_grid,
)

MODULO_MH       = "metacarp.recocido_simulado"
PREFIJO_CSV     = "sa"
SUBCARPETA      = "budget_20260528_sa"
EXPERIMENTO_TAG = "sa_budget"

# SA usa kicks basados en niveles de temperatura, no en iteraciones globales.
# El ratio 30/400 = 7.5 % del budget se preserva con 20/(5*reheat) ~= ajuste.
SA_KICK_ITERS       = 20     # era 5; mas niveles sin mejora antes de kick
SA_PATIENCE         = 100    # niveles de reheat sin mejora → reheat (era 10)
SA_MAX_REHEATS      = 30     # reheats consecutivos sin mejora → parar (era 5)
SA_TEMP_MIN         = 1e-4   # temperatura minima (era 1e-3; mas iteraciones)


def _ejecutar_una(tarea: TareaExp) -> tuple[TareaExp, str, dict | None, str | None]:
    """Worker: una corrida de SA con reheat extendido y lambda x10."""
    try:
        aplicar_patches_budget(MODULO_MH, p_pr=P_PR, lambda_factor=LAMBDA_FACTOR)

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
                "experimento":           "budget_20260528",
                "selector":              "p_inter_probabilistico",
                "p_inter":               str(P_INTER),
                "alpha_inter":           str(ALPHA_INTER),
                "p_pr":                  str(P_PR),
                "sa_kick_iters":         str(SA_KICK_ITERS),
                "sa_patience":           str(SA_PATIENCE),
                "sa_max_reheats":        str(SA_MAX_REHEATS),
                "sa_temp_min":           str(SA_TEMP_MIN),
                "lambda_factor":         str(LAMBDA_FACTOR),
                "operadores_activos":    "5 (2intra+3inter)",
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
            "budget_20260528 para SA: patience=100, max_reheats=30, "
            "T_min=1e-4, kick_iters=20, max_resets=None, lambda x10, "
            "selector p_inter (0.20) + PR (p_pr=0.5)."
        ),
        experimento=EXPERIMENTO_TAG,
    )


if __name__ == "__main__":
    main()
