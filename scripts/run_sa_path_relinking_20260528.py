"""
Variante experimental path_relinking_20260528 para Recocido Simulado (SA).

Cambios vs. ``run_sa_strict_intra_inter_20260524.py``:

  1. **Path Relinking truncado** (capa 3): cada vez que se dispara el kick
     reactivo, con probabilidad ``p_pr=0.5`` se ejecuta PR desde la solucion
     perturbada hacia la mejor solucion global vigente. Con probabilidad
     ``1-p_pr`` se ejecuta solo el kick puro (comportamiento canonico del
     experimento strict). PR usa el objetivo PENALIZADO (costo + lambda*viol).

  2. **Selector de grupo**: sigue siendo BINARIO ESTRICTO (viola -> inter,
     no viola -> intra). PR se monta encima del selector strict (NO del AOS PM).

  3. **Kick reactivo**: conservado (``max_iter_sin_mejora_kick=5`` niveles,
     ``max_resets=10``). El umbral se mantiene identico al strict para
     aislar el efecto de PR.

  4. **Operadores**: mismos 5 (2 intra + 3 inter) que strict.

  5. **Salida**: ``experimentos/path_relinking_20260528_sa/``.

Esta variante NO toca archivos de las MH. El monkey-patch reasigna
``aplicar_kick_labels`` en ``metacarp.strict_intra_inter_20260528`` por una
version aumentada con PR; las MH (que importan el kick DIFERIDO) reciben
automaticamente esa version.
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

MODULO_MH       = "metacarp.recocido_simulado"
PREFIJO_CSV     = "sa"
SUBCARPETA      = "path_relinking_20260528_sa"
EXPERIMENTO_TAG = "sa_path_relinking"


def _ejecutar_una(tarea: TareaExp) -> tuple[TareaExp, str, dict | None, str | None]:
    """Worker que ejecuta UNA corrida de SA en un proceso hijo.

    Pasos por tarea:
      1. Aplicar los monkey-patches: selector strict + PR (los dos en una
         sola llamada). Cada worker tiene su propio modulo importado.
      2. Llamar al wrapper ``recocido_simulado_desde_instancia`` con el
         subconjunto reducido de operadores y los kwargs del kick.
      3. Devolver el estado de la corrida (ok/fail).
    """
    try:
        # 1) Patch combinado: selector binario estricto + PR sobre kick.
        aplicar_patch_pr_para_modulo(MODULO_MH, p_pr=P_PR)

        # 2) Importacion diferida: el wrapper se resuelve aqui para que el
        #    monkey-patch ya este activo cuando la MH internamente llame a
        #    ``seleccionar_grupo_operadores_inter_intra`` y al kick.
        from metacarp import recocido_simulado_desde_instancia
        from metacarp.strict_intra_inter_20260524 import OPERADORES_STRICT_5

        res = recocido_simulado_desde_instancia(
            tarea.instancia,
            root=tarea.root,
            operadores=OPERADORES_STRICT_5,
            # Kick reactivo: 5 niveles sin mejora -> perturbar; tope 10 kicks.
            max_iter_sin_mejora_kick=5,
            max_resets=10,
            # Parametros de parada: replican los del experimento strict.
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
        label_mh="SA",
        prefijo_csv=PREFIJO_CSV,
        subcarpeta_destino=SUBCARPETA,
        modulo_patchear=MODULO_MH,
        ejecutar_una=_ejecutar_una,
        descripcion_cli=(
            "Variante experimental path_relinking_20260528 para SA: "
            "selector binario estricto + 5 operadores + kick reactivo "
            "+ Path Relinking truncado con p_pr=0.5."
        ),
        experimento=EXPERIMENTO_TAG,
    )


if __name__ == "__main__":
    main()
