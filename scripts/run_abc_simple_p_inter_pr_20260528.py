"""
Variante experimental p_inter_pr_20260528 para ABC Simple.

Cambios vs. ``run_abc_simple_path_relinking_20260528.py``:

  1. **Selector p_inter probabilistico (NUEVO)**: en lugar del binario
     estricto, el selector propone INTER con probabilidad ``P_INTER=0.20``
     en estado factible (y ``ALPHA_INTER=0.80`` cuando viola). En ABC el
     PR mostro la mayor ganancia (Delta -5.4 pp); el cambio de selector
     busca amplificarla aprovechando la alta tasa de mejora de los inter.

  2. **Path Relinking truncado**: identico al experimento PR (capa 3 con
     ``p_pr=0.5``). Captura de ``mejor_sol_ids`` via patch a
     ``ContadorOperadores.registrar_mejora`` (la misma estrategia del PR).

  3. **Kick reactivo POBLACIONAL**: conservado (``max_iter_sin_mejora_kick=30``,
     ``max_resets=10``). El kick perturba todas las fuentes.

  4. **Operadores**: mismos 5 (2 intra + 3 inter).

  5. **GPU**: DESHABILITADA (``usar_gpu=False``) por estabilidad
     (libnvrtc.so.12 indisponible).

  6. **Salida**: ``experimentos/p_inter_pr_20260528_abc_simple/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _p_inter_pr_20260528_common import (
    ALPHA_INTER,
    P_INTER,
    P_PR,
    TareaExp,
    aplicar_patch_completo,
    correr_grid,
)

MODULO_MH       = "metacarp.abejas_simple"
PREFIJO_CSV     = "abc_simple"
SUBCARPETA      = "p_inter_pr_20260528_abc_simple"
EXPERIMENTO_TAG = "abc_simple_p_inter_pr"


def _ejecutar_una(tarea: TareaExp) -> tuple[TareaExp, str, dict | None, str | None]:
    """Worker: una corrida de ABC Simple con la politica experimental p_inter+PR."""
    try:
        aplicar_patch_completo(MODULO_MH, p_pr=P_PR)

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
            usar_gpu=False,                 # CPU forzado (libnvrtc no disponible).
            semilla=None,
            repeticion=tarea.repeticion,
            guardar_csv=True,
            ruta_csv=tarea.ruta_csv_parcial,
            guardar_historial=False,
            extra_csv={
                "experimento": "p_inter_pr_20260528",
                "selector": "p_inter_probabilistico",
                "p_inter": str(P_INTER),
                "alpha_inter": str(ALPHA_INTER),
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
        label_mh="ABC Simple",
        prefijo_csv=PREFIJO_CSV,
        subcarpeta_destino=SUBCARPETA,
        modulo_patchear=MODULO_MH,
        ejecutar_una=_ejecutar_una,
        descripcion_cli=(
            "Variante experimental p_inter_pr_20260528 para ABC Simple: "
            "selector p_inter probabilistico (p_inter=0.20, alpha_inter=0.80) "
            "+ 5 operadores + kick reactivo + Path Relinking truncado con p_pr=0.5."
        ),
        experimento=EXPERIMENTO_TAG,
    )


if __name__ == "__main__":
    main()
