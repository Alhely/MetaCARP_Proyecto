"""
Variante experimental strict_intra_inter_20260524 para ABC Simple.

Cambios vs. ``run_abc_simple_p_inter_exp_2026050524.py``:

  1. **Selector binario estricto**: monkey-patch de
     ``seleccionar_grupo_operadores_inter_intra`` dentro de
     ``metacarp.abejas_simple`` por la version binaria DETERMINISTA de
     ``metacarp.strict_intra_inter_20260524``. ABC consume el selector
     en las fases EMPLEADAS y OBSERVADORAS; los SCOUTS canonicos siguen
     generando soluciones COMPLETAMENTE aleatorias sin pasar por el
     selector (decision de diseno de la version Karaboga 2005 que se
     respeta tal cual).

  2. **Subconjunto reducido de operadores (5 = 2 intra + 3 inter)**:
     ``OPERADORES_STRICT_5`` reemplaza al canonico ``OPERADORES_POPULARES``.

  3. **Kick reactivo (capa 2)**: tras ``max_iter_sin_mejora_kick=30``
     CICLOS consecutivos sin mejorar el mejor global se aplica una
     perturbacion inter-ruta POBLACIONAL (a todas las fuentes) y se
     reinicia el contador. Se tolera un maximo de ``max_resets=10``
     kicks antes de parar la corrida. Los contadores de abandono
     (``trials[i]``) tambien se resetean tras el kick para que las
     fuentes recien perturbadas tengan oportunidad de explotar su
     nueva posicion antes de ser reemplazadas por scouts.

  4. **Grid simplificado**: instancia x repeticion. Hiperparametros
     (num_fuentes, limite_abandono) usan defaults instance-aware del
     wrapper.

  5. **Salida**: ``experimentos/strict_intra_inter_20260524_abc_simple/``.
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

MODULO_MH       = "metacarp.abejas_simple"
PREFIJO_CSV     = "abc_simple"
SUBCARPETA      = "strict_intra_inter_20260524_abc_simple"
EXPERIMENTO_TAG = "abc_simple_strict_intra_inter"


def _ejecutar_una(tarea: TareaExp) -> tuple[TareaExp, str, dict | None, str | None]:
    """Worker: una corrida de ABC Simple con la politica experimental."""
    try:
        aplicar_patch_selector(MODULO_MH)

        from metacarp import busqueda_abejas_simple_desde_instancia
        from metacarp.strict_intra_inter_20260524 import OPERADORES_STRICT_5

        res = busqueda_abejas_simple_desde_instancia(
            tarea.instancia,
            root=tarea.root,
            operadores=OPERADORES_STRICT_5,
            # Kick reactivo: 30 ciclos sin mejora -> perturbar TODAS las fuentes;
            # tope 10 kicks antes de detener la corrida.
            max_iter_sin_mejora_kick=30,
            max_resets=10,
            usar_gpu=True,                # ABC se beneficia de GPU en lote observadoras
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
        label_mh="ABC Simple",
        prefijo_csv=PREFIJO_CSV,
        subcarpeta_destino=SUBCARPETA,
        modulo_patchear=MODULO_MH,
        ejecutar_una=_ejecutar_una,
        descripcion_cli=(
            "Variante experimental strict_intra_inter_20260524 para ABC Simple: "
            "selector binario determinista + 5 operadores + kick reactivo."
        ),
        experimento=EXPERIMENTO_TAG,
    )


if __name__ == "__main__":
    main()
