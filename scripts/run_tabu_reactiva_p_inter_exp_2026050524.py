"""
Variante experimental p_inter_exp_2026050524 para Busqueda Tabu Reactiva (RTS).

Cambios vs. ``run_tabu_reactiva_automatico.py``:

  1. **Restriccion posicional**: ancla la 1a tarea servida tras el deposito
     (monkey-patch de ``generar_vecino`` dentro de
     ``metacarp.busqueda_tabu_reactiva``).
  2. **p_inter estatico decidido por la solucion inicial**:
        factible    → p_inter = alpha_inter = 0.5
        infactible  → p_inter = alpha_inter = 0.95
  3. **Grid simplificado**: instancia x repeticion (115 corridas). Demas
     parametros (tenure, factor_aumento/reduccion, escape) usan defaults
     instance-aware del wrapper.
  4. **Salida**: ``experimentos/p_inter_exp_2026050524_tabu_reactiva/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _p_inter_exp_2026050524_common import (
    TareaExp,
    aplicar_patch_vecindario,
    calcular_violacion_inicial,
    correr_grid,
    decidir_p_inter,
)

MODULO_MH       = "metacarp.busqueda_tabu_reactiva"
SIMBOLO_PATCH   = "generar_vecino"
PREFIJO_CSV     = "tabu_reactiva"
SUBCARPETA      = "p_inter_exp_2026050524_tabu_reactiva"
EXPERIMENTO_TAG = "tabu_reactiva_p_inter_exp"


def _ejecutar_una(tarea: TareaExp) -> tuple[TareaExp, str, dict | None, str | None]:
    """Worker: una corrida de RTS con la politica experimental."""
    try:
        aplicar_patch_vecindario(MODULO_MH, SIMBOLO_PATCH)
        viol_ini = calcular_violacion_inicial(tarea.instancia, tarea.root)
        p_inter_fijo = decidir_p_inter(viol_ini)

        from metacarp import busqueda_tabu_reactiva_desde_instancia

        res = busqueda_tabu_reactiva_desde_instancia(
            tarea.instancia,
            root=tarea.root,
            alpha_inter=p_inter_fijo,
            p_inter=p_inter_fijo,
            usar_gpu=True,                # cae a CPU automaticamente si CuPy no esta disponible
            semilla=None,
            repeticion=tarea.repeticion,
            guardar_csv=True,
            ruta_csv=tarea.ruta_csv_parcial,
            guardar_historial=False,
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
        label_mh="RTS Reactiva",
        prefijo_csv=PREFIJO_CSV,
        subcarpeta_destino=SUBCARPETA,
        wrapper_fn=None,
        modulo_patchear=MODULO_MH,
        simbolo_patchear=SIMBOLO_PATCH,
        ejecutar_una=_ejecutar_una,
        descripcion_cli=(
            "Variante experimental p_inter_exp_2026050524 para RTS Reactiva: "
            "preserva la 1a tarea tras el deposito y fija p_inter por la inicial."
        ),
        experimento=EXPERIMENTO_TAG,
    )


if __name__ == "__main__":
    main()
