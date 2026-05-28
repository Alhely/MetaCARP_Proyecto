"""
Variante experimental path_relinking_20260528 para Busqueda Tabu Simple.

Cambios vs. ``run_tabu_simple_strict_intra_inter_20260524.py``:

  1. **Path Relinking truncado** (capa 3): cada vez que se dispara el kick
     reactivo, con probabilidad ``p_pr=0.5`` se ejecuta PR desde la solucion
     perturbada hacia la mejor solucion global vigente. Con probabilidad
     ``1-p_pr`` se ejecuta solo el kick puro. PR usa el objetivo PENALIZADO
     (costo + lambda*viol) -- nunca acepta mejoras solo por costo puro.

  2. **Selector de grupo**: sigue siendo BINARIO ESTRICTO (viola -> inter,
     no viola -> intra). PR se monta encima del selector strict (NO del AOS PM).

  3. **Kick reactivo**: conservado (``max_iter_sin_mejora_kick=30``,
     ``max_resets=10``). Identico al experimento strict.

  4. **Operadores**: mismos 5 (2 intra + 3 inter) que strict.

  5. **Salida**: ``experimentos/path_relinking_20260528_tabu_simple/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _path_relinking_20260528_common import (
    P_PR,
    TareaExp,
    aplicar_patch_pr_para_modulo,
    correr_grid,
)

MODULO_MH       = "metacarp.busqueda_tabu_simple"
PREFIJO_CSV     = "tabu_simple"
SUBCARPETA      = "path_relinking_20260528_tabu_simple"
EXPERIMENTO_TAG = "tabu_simple_path_relinking"


def _ejecutar_una(tarea: TareaExp) -> tuple[TareaExp, str, dict | None, str | None]:
    """Worker: una corrida de TS Simple con la politica experimental PR."""
    try:
        # Patch combinado: selector binario estricto + PR sobre kick.
        aplicar_patch_pr_para_modulo(MODULO_MH, p_pr=P_PR)

        from metacarp import busqueda_tabu_simple_desde_instancia
        from metacarp.strict_intra_inter_20260524 import OPERADORES_STRICT_5

        res = busqueda_tabu_simple_desde_instancia(
            tarea.instancia,
            root=tarea.root,
            operadores=OPERADORES_STRICT_5,
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
                "experimento": "path_relinking_20260528",
                "selector": "binario_estricto",
                "p_pr": str(P_PR),
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
        label_mh="TS Simple",
        prefijo_csv=PREFIJO_CSV,
        subcarpeta_destino=SUBCARPETA,
        modulo_patchear=MODULO_MH,
        ejecutar_una=_ejecutar_una,
        descripcion_cli=(
            "Variante experimental path_relinking_20260528 para TS Simple: "
            "selector binario estricto + 5 operadores + kick reactivo "
            "+ Path Relinking truncado con p_pr=0.5."
        ),
        experimento=EXPERIMENTO_TAG,
    )


if __name__ == "__main__":
    main()
