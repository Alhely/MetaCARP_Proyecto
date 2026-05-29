"""
Variante experimental p_inter_pr_20260528 para Recocido Simulado (SA).

Cambios vs. ``run_sa_path_relinking_20260528.py``:

  1. **Selector p_inter probabilistico (NUEVO)**: en lugar del binario
     estricto, el selector propone INTER con probabilidad ``P_INTER=0.20``
     en estado factible (y ``ALPHA_INTER=0.80`` cuando viola). Motivacion:
     los inter tienen tasa de mejora 4-7x superior a los intra incluso en
     estado factible (analisis Seccion 12 del notebook).

  2. **Path Relinking truncado**: identico al experimento PR (capa 3 con
     ``p_pr=0.5``). Tras el kick reactivo, con probabilidad 0.5 se ejecuta
     PR desde la solucion perturbada hacia la mejor solucion global.

  3. **Kick reactivo**: conservado (``max_iter_sin_mejora_kick=5`` niveles,
     ``max_resets=10``), identico a PR/strict para SA.

  4. **Operadores**: mismos 5 (2 intra + 3 inter) que strict/PR.

  5. **Salida**: ``experimentos/p_inter_pr_20260528_sa/``.

Esta variante NO toca archivos de las MH ni de PR. El monkey-patch combinado
en ``aplicar_patch_completo`` instala los dos parches en orden:
  (1) aplicar_patch_pr (instala strict + captura mejor_sol + kick+PR),
  (2) aplicar_patch_p_inter (SOBREESCRIBE el selector strict con p_inter).
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

MODULO_MH       = "metacarp.recocido_simulado"
PREFIJO_CSV     = "sa"
SUBCARPETA      = "p_inter_pr_20260528_sa"
EXPERIMENTO_TAG = "sa_p_inter_pr"


def _ejecutar_una(tarea: TareaExp) -> tuple[TareaExp, str, dict | None, str | None]:
    """Worker que ejecuta UNA corrida de SA en un proceso hijo.

    Pasos por tarea:
      1. Aplicar el patch combinado: PR + p_inter (en ese orden).
      2. Llamar al wrapper ``recocido_simulado_desde_instancia`` con el
         subconjunto reducido de operadores y los kwargs del kick.
      3. Devolver el estado de la corrida (ok/fail).
    """
    try:
        # 1) Patch combinado: PR (capa 3) + selector p_inter (capa 1).
        aplicar_patch_completo(MODULO_MH, p_pr=P_PR)

        # 2) Importacion diferida: el wrapper se resuelve aqui para que los
        #    monkey-patches ya esten activos cuando la MH internamente llame
        #    a ``seleccionar_grupo_operadores_inter_intra`` y al kick.
        from metacarp import recocido_simulado_desde_instancia
        from metacarp.strict_intra_inter_20260524 import OPERADORES_STRICT_5

        res = recocido_simulado_desde_instancia(
            tarea.instancia,
            root=tarea.root,
            operadores=OPERADORES_STRICT_5,
            # Kick reactivo: 5 niveles sin mejora -> perturbar; tope 10 kicks.
            max_iter_sin_mejora_kick=5,
            max_resets=10,
            # Parametros de parada: replican los del experimento PR para SA.
            temperatura_minima=1e-3,
            patience=10,
            reheat_factor=0.5,
            max_reheats_sin_mejora=5,
            usar_gpu=False,
            semilla=None,                 # aleatoria del sistema
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
        label_mh="SA",
        prefijo_csv=PREFIJO_CSV,
        subcarpeta_destino=SUBCARPETA,
        modulo_patchear=MODULO_MH,
        ejecutar_una=_ejecutar_una,
        descripcion_cli=(
            "Variante experimental p_inter_pr_20260528 para SA: "
            "selector p_inter probabilistico (p_inter=0.20, alpha_inter=0.80) "
            "+ 5 operadores + kick reactivo + Path Relinking truncado con p_pr=0.5."
        ),
        experimento=EXPERIMENTO_TAG,
    )


if __name__ == "__main__":
    main()
