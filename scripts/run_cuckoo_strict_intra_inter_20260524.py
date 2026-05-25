"""
Variante experimental strict_intra_inter_20260524 para Cuckoo Search.

Cambios vs. ``run_cuckoo_p_inter_exp_2026050524.py``:

  1. **Selector binario estricto**: monkey-patch de
     ``seleccionar_grupo_operadores_inter_intra`` dentro de
     ``metacarp.cuckoo_search`` por la version binaria DETERMINISTA de
     ``metacarp.strict_intra_inter_20260524``. Cuckoo consume el
     selector tanto en la fase de generacion de cuckoos (UNA llamada
     por nido) como en la fase de abandono (UNA llamada para el grupo
     de los peores nidos).

  2. **Subconjunto reducido de operadores (5 = 2 intra + 3 inter)**:
     ``OPERADORES_STRICT_5`` reemplaza al canonico ``OPERADORES_POPULARES``.

  3. **Kick reactivo (capa 2)**: tras ``max_iter_sin_mejora_kick=30``
     iteraciones consecutivas sin mejorar el mejor global se aplica una
     perturbacion inter-ruta POBLACIONAL (a todos los nidos) y se
     reinicia el contador. Tras el kick se invoca ``fusionar_desde_nidos``
     para sincronizar el mejor global con la poblacion perturbada (un
     nido kicked podria casualmente mejorar el mejor). Se tolera un
     maximo de ``max_resets=10`` kicks antes de parar la corrida.

  4. **Grid simplificado**: instancia x repeticion. Hiperparametros
     (num_nidos, pasos_levy, pa_abandono, beta_levy) usan defaults
     instance-aware del wrapper.

  5. **Salida**: ``experimentos/strict_intra_inter_20260524_cuckoo/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _strict_intra_inter_20260524_common import (
    TareaExp,
    aplicar_patch_selector,
    correr_grid,
)

MODULO_MH       = "metacarp.cuckoo_search"
PREFIJO_CSV     = "cuckoo"
SUBCARPETA      = "strict_intra_inter_20260524_cuckoo"
EXPERIMENTO_TAG = "cuckoo_strict_intra_inter"


def _ejecutar_una(tarea: TareaExp) -> tuple[TareaExp, str, dict | None, str | None]:
    """Worker: una corrida de Cuckoo Search con la politica experimental."""
    try:
        aplicar_patch_selector(MODULO_MH)

        from metacarp import cuckoo_search_desde_instancia
        from metacarp.strict_intra_inter_20260524 import OPERADORES_STRICT_5

        res = cuckoo_search_desde_instancia(
            tarea.instancia,
            root=tarea.root,
            operadores=OPERADORES_STRICT_5,
            # Kick reactivo: 30 iteraciones sin mejora -> perturbar TODOS los
            # nidos; tope 10 kicks antes de detener la corrida.
            max_iter_sin_mejora_kick=30,
            max_resets=10,
            usar_gpu=True,
            semilla=None,
            repeticion=tarea.repeticion,
            guardar_csv=True,
            ruta_csv=tarea.ruta_csv_parcial,
            guardar_historial=False,
            extra_csv={
                "experimento": "strict_intra_inter_20260524",
                "selector": "binario_estricto",
                "operadores_activos": "5 (2intra+3inter)",
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
        label_mh="Cuckoo Search",
        prefijo_csv=PREFIJO_CSV,
        subcarpeta_destino=SUBCARPETA,
        modulo_patchear=MODULO_MH,
        ejecutar_una=_ejecutar_una,
        descripcion_cli=(
            "Variante experimental strict_intra_inter_20260524 para Cuckoo Search: "
            "selector binario determinista + 5 operadores + kick reactivo."
        ),
        experimento=EXPERIMENTO_TAG,
    )


if __name__ == "__main__":
    main()
