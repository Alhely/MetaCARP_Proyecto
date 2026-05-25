"""
Variante experimental strict_intra_inter_20260524 para Recocido Simulado (SA).

Cambios vs. ``run_sa_p_inter_exp_2026050524.py``:

  1. **Selector binario estricto**: monkey-patch de
     ``seleccionar_grupo_operadores_inter_intra`` dentro de
     ``metacarp.recocido_simulado`` por la version binaria DETERMINISTA
     de ``metacarp.strict_intra_inter_20260524``. Cuando la solucion
     actual viola capacidad, SIEMPRE se elige el grupo INTER. Cuando es
     factible, SIEMPRE se elige el grupo INTRA. No hay aleatoriedad en
     esta capa.

  2. **Subconjunto reducido de operadores (5 = 2 intra + 3 inter)**:
     ``OPERADORES_STRICT_5`` reemplaza al canonico
     ``OPERADORES_POPULARES``. Excluye ``2opt_intra`` y ``relocate_inter``.

  3. **Kick reactivo (capa 2)**: tras ``max_iter_sin_mejora_kick=5``
     niveles consecutivos sin mejorar el mejor global se aplica una
     perturbacion inter-ruta sobre la solucion actual y se reinicia el
     contador. Se tolera un maximo de ``max_resets=10`` kicks antes de
     parar la corrida.

  4. **Grid simplificado**: instancia x repeticion. Hiperparametros de
     SA (temperatura_minima, patience, reheat_factor) replican los del
     experimento anterior para garantizar comparabilidad.

  5. **Salida**: ``experimentos/strict_intra_inter_20260524_sa/``.
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

MODULO_MH       = "metacarp.recocido_simulado"
PREFIJO_CSV     = "sa"
SUBCARPETA      = "strict_intra_inter_20260524_sa"
EXPERIMENTO_TAG = "sa_strict_intra_inter"


def _ejecutar_una(tarea: TareaExp) -> tuple[TareaExp, str, dict | None, str | None]:
    """Worker que ejecuta UNA corrida de SA en un proceso hijo.

    Pasos por tarea:
      1. Aplicar el monkey-patch del selector (necesario en cada worker).
      2. Llamar al wrapper ``recocido_simulado_desde_instancia`` con el
         subconjunto reducido de operadores y los kwargs del kick.
      3. Devolver el estado de la corrida (ok/fail).
    """
    try:
        # 1) Patch del selector (cada worker tiene su propio modulo importado).
        aplicar_patch_selector(MODULO_MH)

        # 2) Importacion diferida: el wrapper se resuelve aqui para que el
        #    monkey-patch ya este activo cuando la MH internamente llame a
        #    ``seleccionar_grupo_operadores_inter_intra``.
        from metacarp import recocido_simulado_desde_instancia
        from metacarp.strict_intra_inter_20260524 import OPERADORES_STRICT_5

        res = recocido_simulado_desde_instancia(
            tarea.instancia,
            root=tarea.root,
            # Subconjunto reducido de operadores (2 intra + 3 inter).
            operadores=OPERADORES_STRICT_5,
            # Kick reactivo: 5 niveles sin mejora -> perturbar; tope 10 kicks.
            # SA usa el contador ``niveles_sin_mejora_kick`` paralelo al
            # ``niveles_sin_mejora`` del reheat, por lo que la unidad aqui
            # es "niveles consecutivos sin mejorar el mejor reportable".
            max_iter_sin_mejora_kick=5,
            max_resets=10,
            # Parametros de parada: replican los de run_sa_p_inter_exp.
            temperatura_minima=1e-3,
            patience=10,
            reheat_factor=0.5,
            max_reheats_sin_mejora=5,
            usar_gpu=True,                # cae a CPU si CuPy no esta disponible
            semilla=None,                 # aleatoria del sistema
            repeticion=tarea.repeticion,
            guardar_csv=True,
            ruta_csv=tarea.ruta_csv_parcial,
            guardar_historial=False,
            # Metadatos del experimento (SA NO vuelca ``extra_csv`` a la fila
            # del CSV — los reportamos solo para trazabilidad en el dict de
            # devolucion). Las columnas ``n_resets_kick`` ya salen del wrapper.
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
        label_mh="SA",
        prefijo_csv=PREFIJO_CSV,
        subcarpeta_destino=SUBCARPETA,
        modulo_patchear=MODULO_MH,
        ejecutar_una=_ejecutar_una,
        descripcion_cli=(
            "Variante experimental strict_intra_inter_20260524 para SA: "
            "selector binario determinista + 5 operadores + kick reactivo."
        ),
        experimento=EXPERIMENTO_TAG,
    )


if __name__ == "__main__":
    main()
