"""
Variante experimental aos_pm_20260527 para Reactive Tabu Search (RTS).

Cambios vs. ``run_tabu_reactiva_strict_intra_inter_20260524.py``:

  1. **AOS Probability Matching**: dentro del grupo elegido (inter/intra),
     el operador se selecciona con pesos adaptativos en lugar de uniforme.
     Los pesos se actualizan cuando el operador genera una mejora global
     (via patch de ``ContadorOperadores.registrar_mejora``).

  2. **Selector de grupo**: sigue siendo BINARIO ESTRICTO (viola -> inter,
     no viola -> intra). La capa AOS solo opera DENTRO del grupo elegido.

  3. **Kick reactivo**: conservado (``max_iter_sin_mejora_kick=30``,
     ``max_resets=10``). Identico al experimento strict.

  4. **Operadores**: mismos 5 (2 intra + 3 inter) que strict.

  5. **Salida**: ``experimentos/aos_pm_20260527_tabu_reactiva/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _aos_pm_20260527_common import (
    ALPHA_AOS,
    TareaExp,
    aplicar_patch_aos,
    correr_grid,
)

MODULO_MH       = "metacarp.busqueda_tabu_reactiva"
PREFIJO_CSV     = "tabu_reactiva"
SUBCARPETA      = "aos_pm_20260527_tabu_reactiva"
EXPERIMENTO_TAG = "tabu_reactiva_aos_pm"


def _ejecutar_una(tarea: TareaExp) -> tuple[TareaExp, str, dict | None, str | None]:
    """Worker: una corrida de RTS con la politica experimental AOS PM."""
    try:
        aplicar_patch_aos(MODULO_MH)

        from metacarp import busqueda_tabu_reactiva_desde_instancia
        from metacarp.aos_pm_20260527 import OPERADORES_AOS_5

        res = busqueda_tabu_reactiva_desde_instancia(
            tarea.instancia,
            root=tarea.root,
            operadores=OPERADORES_AOS_5,
            # Kick reactivo: 30 iteraciones sin mejora -> perturbar; tope 10 kicks.
            max_iter_sin_mejora_kick=30,
            max_resets=10,
            usar_gpu=False,
            semilla=None,
            repeticion=tarea.repeticion,
            guardar_csv=True,
            ruta_csv=tarea.ruta_csv_parcial,
            guardar_historial=False,
            extra_csv={
                "experimento": "aos_pm_20260527",
                "selector": "aos_probability_matching",
                "operadores_activos": "5 (2intra+3inter)",
                "alpha_aos": str(ALPHA_AOS),
            },
        )
        info = {
            "costo":    res.mejor_costo,
            "tiempo":   res.tiempo_segundos,
            "n_resets": res.n_resets_kick,
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
            "Variante experimental aos_pm_20260527 para RTS: "
            "AOS Probability Matching + 5 operadores + kick reactivo."
        ),
        experimento=EXPERIMENTO_TAG,
    )


if __name__ == "__main__":
    main()
