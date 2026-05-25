"""
Variante experimental lambda_grid_20260525 para Busqueda Tabu Simple.

Extiende ``run_tabu_simple_strict_intra_inter_20260524`` anadiendo
``lambda_factor`` como dimension del grid. Para cada (instancia, repeticion)
se ejecutan 5 corridas, una por cada lambda_factor en
``LAMBDA_FACTORS = [0.5, 1.0, 2.0, 5.0, 10.0]``.

El valor concreto pasado al wrapper se calcula como::

    lambda_default = lambda_penal_capacidad_por_defecto(ctx)
    lambda_actual  = lambda_factor * lambda_default

donde el ``ctx`` se construye una vez por corrida desde la instancia.

Notas
-----
- TS Simple NO tiene objetivo penalizado en su bucle principal (usa
  costo puro). ``lambda_capacidad`` afecta solo la seleccion de la
  solucion inicial (``seleccionar_mejor_inicial_rapido``). Las columnas
  ``lambda_factor`` / ``lambda_actual`` / ``lambda_default`` se inyectan
  en el CSV via post-procesado (``inyectar_columnas_lambda``) dado que
  ``extra_csv`` no es volcado a la fila por el nucleo de TS Simple.
- Salida: ``experimentos/lambda_grid_20260525_tabu_simple/``.
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

MODULO_MH       = "metacarp.busqueda_tabu_simple"
PREFIJO_CSV     = "tabu_simple"
SUBCARPETA      = "lambda_grid_20260525_tabu_simple"
EXPERIMENTO_TAG = "tabu_simple_lambda_grid"


def _ejecutar_una(tarea: TareaExp) -> tuple[TareaExp, str, dict | None, str | None]:
    """Worker: una corrida de TS Simple con la politica experimental."""
    try:
        aplicar_patch_selector(MODULO_MH)

        # Calculamos el lambda efectivo de esta corrida a partir del
        # default de la instancia escalado por el factor del grid.
        lambda_default = calcular_lambda_default(tarea.instancia, tarea.root)
        lambda_actual  = tarea.lambda_factor * lambda_default

        from metacarp import busqueda_tabu_simple_desde_instancia
        from metacarp.strict_intra_inter_20260524 import OPERADORES_STRICT_5

        res = busqueda_tabu_simple_desde_instancia(
            tarea.instancia,
            root=tarea.root,
            operadores=OPERADORES_STRICT_5,
            # Afecta seleccion de solucion inicial; el bucle TS usa costo puro.
            lambda_capacidad=lambda_actual,
            # Kick reactivo: 30 iteraciones sin mejora -> perturbar; tope 10 kicks.
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

        # Inyectamos las columnas lambda en el CSV parcial. TS Simple no
        # vuelca ``extra_csv`` a la fila, asi que sin este paso las
        # columnas no aparecerian en el CSV final.
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
        label_mh="TS Simple",
        prefijo_csv=PREFIJO_CSV,
        subcarpeta_destino=SUBCARPETA,
        modulo_patchear=MODULO_MH,
        ejecutar_una=_ejecutar_una,
        descripcion_cli=(
            "Variante experimental lambda_grid_20260525 para TS Simple: "
            "selector binario determinista + 5 operadores + kick reactivo + grid de lambda."
        ),
        experimento=EXPERIMENTO_TAG,
    )


if __name__ == "__main__":
    main()
