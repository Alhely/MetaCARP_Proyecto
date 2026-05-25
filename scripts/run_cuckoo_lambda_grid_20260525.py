"""
Variante experimental lambda_grid_20260525 para Cuckoo Search.

Extiende ``run_cuckoo_strict_intra_inter_20260524`` anadiendo
``lambda_factor`` como dimension del grid. Para cada (instancia, repeticion)
se ejecutan 5 corridas, una por cada lambda_factor en
``LAMBDA_FACTORS = [0.5, 1.0, 2.0, 5.0, 10.0]``.

El valor concreto pasado al wrapper se calcula como::

    lambda_default = lambda_penal_capacidad_por_defecto(ctx)
    lambda_actual  = lambda_factor * lambda_default

Notas
-----
- Cuckoo SI consume ``lambda_capacidad`` en su funcion objetivo penalizada
  y ademas es la UNICA MH que vuelca ``extra_csv`` a la fila (ya escribe
  ``lambda_capacidad``, ``lambda_factor``, ``lambda_actual``, etc. via
  extra_csv). El paso de inyeccion del common es idempotente: si las
  columnas ya existen, sobreescribe con los mismos valores.
- Salida: ``experimentos/lambda_grid_20260525_cuckoo/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lambda_grid_20260525_common import (
    TareaExp,
    aplicar_patch_selector,
    calcular_lambda_default,
    correr_grid,
    inyectar_columnas_lambda,
)

MODULO_MH       = "metacarp.cuckoo_search"
PREFIJO_CSV     = "cuckoo"
SUBCARPETA      = "lambda_grid_20260525_cuckoo"
EXPERIMENTO_TAG = "cuckoo_lambda_grid"


def _ejecutar_una(tarea: TareaExp) -> tuple[TareaExp, str, dict | None, str | None]:
    """Worker: una corrida de Cuckoo Search con la politica experimental."""
    try:
        aplicar_patch_selector(MODULO_MH)

        # Calculamos el lambda efectivo de esta corrida a partir del
        # default de la instancia escalado por el factor del grid.
        lambda_default = calcular_lambda_default(tarea.instancia, tarea.root)
        lambda_actual  = tarea.lambda_factor * lambda_default

        from metacarp import cuckoo_search_desde_instancia
        from metacarp.strict_intra_inter_20260524 import OPERADORES_STRICT_5

        res = cuckoo_search_desde_instancia(
            tarea.instancia,
            root=tarea.root,
            operadores=OPERADORES_STRICT_5,
            # Lambda efectivo del grid (peso de la penalizacion por exceso).
            lambda_capacidad=lambda_actual,
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
                "experimento":        "lambda_grid_20260525",
                "lambda_factor":      tarea.lambda_factor,
                "lambda_actual":      round(lambda_actual, 4),
                "lambda_default":     round(lambda_default, 4),
                "selector":           "binario_estricto",
                "operadores_activos": "5 (2intra+3inter)",
            },
        )

        # Inyectamos las columnas lambda en el CSV parcial. Cuckoo ya las
        # escribe via ``extra_csv``, pero invocamos la inyeccion igualmente
        # para garantizar consistencia con las demas MH del experimento.
        # La operacion es idempotente cuando las columnas ya existen.
        inyectar_columnas_lambda(
            tarea.ruta_csv_parcial,
            lambda_factor=tarea.lambda_factor,
            lambda_actual=lambda_actual,
            lambda_default=lambda_default,
        )

        info = {
            "costo":          res.mejor_costo,
            "tiempo":         res.tiempo_segundos,
            "n_resets":       res.n_resets_kick,
            "lambda_factor":  tarea.lambda_factor,
            "lambda_actual":  lambda_actual,
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
            "Variante experimental lambda_grid_20260525 para Cuckoo Search: "
            "selector binario determinista + 5 operadores + kick reactivo + grid de lambda."
        ),
        experimento=EXPERIMENTO_TAG,
    )


if __name__ == "__main__":
    main()
