"""
Variante experimental p_inter_exp_2026050524 para Recocido Simulado (SA).

Cambios vs. ``run_sa_automatico.py``:

  1. **Restriccion posicional**: la primera tarea servida tras el deposito
     en cada ruta queda anclada (no la mueven los operadores). Se logra
     via monkey-patch del simbolo ``generar_vecino`` dentro del modulo
     ``metacarp.recocido_simulado``, reemplazandolo por
     ``generar_vecino_exp`` (del modulo ``vecindarios_p_inter_exp_2026050524``).

  2. **p_inter estatico decidido por la solucion inicial**:
        - Inicial factible    → p_inter = alpha_inter = 0.5
        - Inicial infactible  → p_inter = alpha_inter = 0.95
     El mismo valor se pasa a ambos campos, asi el helper
     ``seleccionar_grupo_operadores_inter_intra`` aplica la misma
     probabilidad sin importar la violacion instantanea.

  3. **Grid simplificado**: solo instancia x repeticion (115 corridas).
     No barremos alpha ni p_inter. Otros hiperparametros (patience,
     reheat_factor, alpha) usan los defaults canonicos de
     ``recocido_simulado_desde_instancia``.

  4. **Salida**: ``experimentos/p_inter_exp_2026050524_sa/``.

NO se modifica ningun archivo de ``metacarp/`` ni los scripts existentes.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Permite importar el helper common al ejecutar el script directamente.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# El helper expone TareaExp, decidir_p_inter, calcular_violacion_inicial,
# aplicar_patch_vecindario y correr_grid (el bucle principal).
from _p_inter_exp_2026050524_common import (
    TareaExp,
    aplicar_patch_vecindario,
    calcular_violacion_inicial,
    correr_grid,
    decidir_p_inter,
)

# Etiquetas del experimento
MODULO_MH        = "metacarp.recocido_simulado"
SIMBOLO_PATCH    = "generar_vecino"      # SA usa el dispatcher labels-based
PREFIJO_CSV      = "sa"
SUBCARPETA       = "p_inter_exp_2026050524_sa"
EXPERIMENTO_TAG  = "sa_p_inter_exp"


def _ejecutar_una(tarea: TareaExp) -> tuple[TareaExp, str, dict | None, str | None]:
    """Worker que ejecuta UNA corrida de SA en un proceso hijo.

    Pasos por tarea:
      1. Aplicar el monkey-patch (necesario en cada worker).
      2. Calcular violacion de la solucion inicial y decidir p_inter.
      3. Llamar al wrapper ``recocido_simulado_desde_instancia`` con la
         politica p_inter = alpha_inter = valor decidido.
      4. Devolver el estado de la corrida (ok/fail).
    """
    try:
        # 1) Patch del dispatcher (cada worker tiene su propio modulo importado).
        aplicar_patch_vecindario(MODULO_MH, SIMBOLO_PATCH)

        # 2) Decidir p_inter en base a la inicial.
        viol_ini = calcular_violacion_inicial(tarea.instancia, tarea.root)
        p_inter_fijo = decidir_p_inter(viol_ini)

        # 3) Importacion diferida: el wrapper se resuelve aqui para que el
        #    monkey-patch ya este activo cuando la MH internamente llame a
        #    ``generar_vecino``.
        from metacarp import recocido_simulado_desde_instancia

        res = recocido_simulado_desde_instancia(
            tarea.instancia,
            root=tarea.root,
            # Politica experimental: mismo valor en ambos para estatica.
            alpha_inter=p_inter_fijo,
            p_inter=p_inter_fijo,
            # Parametros de parada: replican los de run_sa_automatico.py para
            # garantizar que el SA termine en tiempo razonable. Los defaults
            # del wrapper (patience=50, max_reheats_sin_mejora=0) pueden
            # provocar corridas indefinidas en algunas instancias (p.ej.
            # gdb17 con calibracion automatica de T_minima).
            temperatura_minima=1e-3,
            patience=10,
            reheat_factor=0.5,
            max_reheats_sin_mejora=5,
            usar_gpu=True,               # cae a CPU automaticamente si CuPy no esta disponible
            semilla=None,                # aleatoria del sistema (no determinista)
            repeticion=tarea.repeticion,
            guardar_csv=True,
            ruta_csv=tarea.ruta_csv_parcial,
            guardar_historial=False,
            # Metadatos del experimento para inspeccionar en el CSV.
            extra_csv={
                "experimento": "p_inter_exp_2026050524",
                "viol_inicial": float(viol_ini),
                "p_inter_fijo": p_inter_fijo,
                "inicial_factible": viol_ini <= 1e-12,
                "restriccion_posicional": True,
            },
        )
        info = {
            "costo":  res.mejor_costo,
            "tiempo": res.tiempo_segundos,
            "viol_inicial": viol_ini,
            "p_inter": p_inter_fijo,
        }
        return (tarea, "ok", info, None)
    except Exception as exc:  # noqa: BLE001
        return (tarea, "fail", None, f"{type(exc).__name__}: {exc}")


def main() -> None:
    correr_grid(
        label_mh="SA",
        prefijo_csv=PREFIJO_CSV,
        subcarpeta_destino=SUBCARPETA,
        wrapper_fn=None,             # no se usa: ejecutar_una sabe su MH
        modulo_patchear=MODULO_MH,
        simbolo_patchear=SIMBOLO_PATCH,
        ejecutar_una=_ejecutar_una,
        descripcion_cli=(
            "Variante experimental p_inter_exp_2026050524 para SA: "
            "preserva la 1a tarea tras el deposito y fija p_inter por la inicial."
        ),
        experimento=EXPERIMENTO_TAG,
    )


if __name__ == "__main__":
    main()
