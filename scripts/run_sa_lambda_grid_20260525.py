"""
Variante experimental lambda_grid_20260525 para Recocido Simulado (SA).

Extiende ``run_sa_strict_intra_inter_20260524`` anadiendo ``lambda_factor``
como dimension del grid. Para cada (instancia, repeticion) se ejecutan 5
corridas, una por cada lambda_factor en
``LAMBDA_FACTORS = [0.5, 1.0, 2.0, 5.0, 10.0]``.

El valor concreto pasado al wrapper se calcula como::

    lambda_default = lambda_penal_capacidad_por_defecto(ctx)
    lambda_actual  = lambda_factor * lambda_default

donde el ``ctx`` se construye una vez por corrida desde la instancia.

Notas
-----
- SA SI consume ``lambda_capacidad`` en su funcion objetivo penalizada
  (lo aplica al exceso de demanda en el criterio de Metropolis).
  ``recocido_simulado_desde_instancia`` lo acepta como parametro publico.
- SA NO vuelca ``extra_csv`` ni escribe ``lambda_capacidad`` en su fila
  CSV, por eso el common inyecta a posteriori las columnas
  ``lambda_factor`` / ``lambda_actual`` / ``lambda_default`` para
  trazabilidad del experimento.
- Salida: ``experimentos/lambda_grid_20260525_sa/``.
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

MODULO_MH       = "metacarp.recocido_simulado"
PREFIJO_CSV     = "sa"
SUBCARPETA      = "lambda_grid_20260525_sa"
EXPERIMENTO_TAG = "sa_lambda_grid"


def _ejecutar_una(tarea: TareaExp) -> tuple[TareaExp, str, dict | None, str | None]:
    """Worker que ejecuta UNA corrida de SA en un proceso hijo.

    Pasos por tarea:
      1. Aplicar el monkey-patch del selector (necesario en cada worker).
      2. Calcular ``lambda_actual`` = ``lambda_factor`` * lambda_default.
      3. Llamar al wrapper ``recocido_simulado_desde_instancia`` con el
         subconjunto reducido de operadores, los kwargs del kick y el
         ``lambda_capacidad`` del grid.
      4. Inyectar columnas lambda en el CSV parcial resultante.
      5. Devolver el estado de la corrida (ok/fail).
    """
    try:
        # 1) Patch del selector (cada worker tiene su propio modulo importado).
        aplicar_patch_selector(MODULO_MH)

        # 2) Calculamos el lambda efectivo de esta corrida a partir del
        #    default de la instancia escalado por el factor del grid.
        lambda_default = calcular_lambda_default(tarea.instancia, tarea.root)
        lambda_actual  = tarea.lambda_factor * lambda_default

        # 3) Importacion diferida: el wrapper se resuelve aqui para que el
        #    monkey-patch ya este activo cuando la MH internamente llame a
        #    ``seleccionar_grupo_operadores_inter_intra``.
        from metacarp import recocido_simulado_desde_instancia
        from metacarp.strict_intra_inter_20260524 import OPERADORES_STRICT_5

        res = recocido_simulado_desde_instancia(
            tarea.instancia,
            root=tarea.root,
            # Subconjunto reducido de operadores (2 intra + 3 inter).
            operadores=OPERADORES_STRICT_5,
            # Lambda efectivo del grid (peso de la penalizacion por exceso).
            lambda_capacidad=lambda_actual,
            # Kick reactivo: 5 niveles sin mejora -> perturbar; tope 10 kicks.
            # SA usa el contador ``niveles_sin_mejora_kick`` paralelo al
            # ``niveles_sin_mejora`` del reheat, por lo que la unidad aqui
            # es "niveles consecutivos sin mejorar el mejor reportable".
            max_iter_sin_mejora_kick=5,
            max_resets=10,
            # Parametros de parada: replican los de run_sa_strict_intra_inter.
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
            extra_csv={
                "experimento":        "lambda_grid_20260525",
                "lambda_factor":      tarea.lambda_factor,
                "lambda_actual":      round(lambda_actual, 4),
                "lambda_default":     round(lambda_default, 4),
                "selector":           "binario_estricto",
                "operadores_activos": "5 (2intra+3inter)",
            },
        )

        # 4) Inyectamos las columnas lambda en el CSV parcial. SA no vuelca
        #    ``extra_csv`` a la fila, asi que sin este paso las columnas no
        #    aparecerian en el CSV final del experimento.
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
        label_mh="SA",
        prefijo_csv=PREFIJO_CSV,
        subcarpeta_destino=SUBCARPETA,
        modulo_patchear=MODULO_MH,
        ejecutar_una=_ejecutar_una,
        descripcion_cli=(
            "Variante experimental lambda_grid_20260525 para SA: "
            "selector binario determinista + 5 operadores + kick reactivo + grid de lambda."
        ),
        experimento=EXPERIMENTO_TAG,
    )


if __name__ == "__main__":
    main()
