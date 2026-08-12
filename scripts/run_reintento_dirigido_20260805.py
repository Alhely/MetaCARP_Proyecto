"""
Re-intento dirigido para llevar a <=1% de gap las instancias que quedaron por
encima de ese umbral (diagnóstico en resultados/instancias_gap_mayor_1pct_20260805.csv).

Cubre SA, TS, RTS, ABC y VDO reutilizando ÍNTEGRA la maquinaria de
``run_val_egl_20260710.py`` (Tarea, semillas deterministas, consolidación de
CSV parciales, Path Relinking); Cuckoo Search se maneja aparte en
``run_reintento_cs_20260805.py`` porque vive en un módulo de campaña propio.

Diseño (ver conversación 2026-08-05 para el diagnóstico completo):

  * Clase de cada instancia: "val" para val/gdb-transferencia(6)/gdb-
    calibración(17)/kshs(6) — la config "val" de CONFIG_POR_MH coincide
    numéricamente con la calibración de junio de las 23 pequeñas, así que
    cubre correctamente ese grupo sin añadir una clase nueva. "egl" para las
    24 egl.

  * Presupuesto por tier (multiplicador sobre el default 300s/10^6 evals):
      A (gap 1-3%):   x1   + 10 repeticiones nuevas, config actual sin cambios.
      B (gap 3-10%):  x2.5 + 10 repeticiones nuevas, config actual sin cambios.
      C (gap 10-25%): x7   + 10 repeticiones nuevas, con fix/mini-grid (ver abajo).
      D (gap >25%):   x7   + 10 repeticiones nuevas, igual que C.

  * Fixes de mecanismo aplicados SOLO donde el diagnóstico los respalda:
      - SA en clase egl (tiers B/C/D): p_inter baja de 0.6 a 0.3. Evidencia:
        el análisis de operadores mostró que en SA la razón aceptación
        inter/intra se DESPLOMA con el tamaño (gdb 0.25 -> egl 0.068),
        mientras que en val (0.22) se mantiene cercana a gdb — por eso el
        fix aplica solo a egl, no a val/small.
      - VDO (todas las clases, automático): gamma se recalibra por instancia
        en ``_construir_kwargs`` (fix ya aplicado en run_val_egl_20260710.py)
        para que el amortiguamiento coincida con el presupuesto real, en vez
        del gamma=0.05 fijo que dejaba la amplitud sin decaer a tiempo.
      - TS/RTS/ABC en tiers C/D: NO hay evidencia de que el mismo colapso
        inter/intra de SA se replique (verificado: sus razones inter/intra
        se mantienen aprox. estables entre gdb y egl). En lugar de adivinar
        un valor, se hace un mini-grid de 3 candidatos de p_inter
        (0.5x, 1.0x, 1.5x el valor de clase, acotado a <=0.9) con las 10
        repeticiones nuevas repartidas 4/3/3 entre candidatos — mismo total
        de cómputo que un solo config, mejor cobertura.

Salida: experimentos_reintento_20260805/<mh>/final/ (mismo formato que la
campaña original; NO sobreescribe experimentos_val_egl_20260710/).

Uso:
    python scripts/run_reintento_dirigido_20260805.py [--workers N] [--smoke]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))

from run_val_egl_20260710 import (  # noqa: E402
    CONFIG_POR_MH, INSTANCIAS_EGL, Tarea, _cargar_runner, _consolidar_por_instancia,
    _derivar_semilla, _ejecutar_tarea,
)

LISTA_PROBLEMA = RAIZ / "resultados" / "instancias_gap_mayor_1pct_20260805.csv"
SALIDA_RAIZ = RAIZ / "experimentos_reintento_20260805"

MHS_COMPATIBLES = ("sa", "ts", "rts", "abc", "vdo")
# Mapeo etiqueta de la lista de problema (SA/TS/...) -> clave interna del runner.
ETIQUETA_A_CLAVE = {"SA": "sa", "TS": "ts", "RTS": "rts", "ABC": "abc", "VDO": "vdo"}

# multiplicador de presupuesto y repeticiones nuevas por tier.
PRESUPUESTO_TIER = {
    "A(1-3%)":   (1.0, 10),
    "B(3-10%)":  (2.5, 10),
    "C(10-25%)": (7.0, 10),
    "D(>25%)":   (7.0, 10),
}
TIME_LIMITE_DEF = 300.0
MAX_EVAL_DEF = 1_000_000

# MHs con mini-grid dirigido en tiers C/D (sin evidencia de fix único).
MHS_MINIGRID = {"ts", "rts", "abc"}


def _clase_de(instancia: str) -> str:
    return "egl" if instancia in INSTANCIAS_EGL else "val"


def _leer_lista_problema() -> list[dict]:
    filas = []
    with open(LISTA_PROBLEMA, newline="") as fh:
        for fila in csv.DictReader(fh):
            mh = ETIQUETA_A_CLAVE.get(fila["metaheuristica"])
            if mh not in MHS_COMPATIBLES:
                continue
            filas.append({**fila, "mh": mh})
    return filas


def _construir_tareas(filas: list[dict], smoke: bool) -> list[Tarea]:
    tareas: list[Tarea] = []
    dir_parciales_por_mh = {mh: SALIDA_RAIZ / mh / "final" / "_partials"
                            for mh in MHS_COMPATIBLES}
    for d in dir_parciales_por_mh.values():
        d.mkdir(parents=True, exist_ok=True)

    for fila in filas:
        mh, inst, tier = fila["mh"], fila["instancia"], fila["tier"]
        clase = _clase_de(inst)
        factor, reps_nuevas = PRESUPUESTO_TIER[tier]
        if smoke:
            factor, reps_nuevas = 0.02, 1  # recorte drástico para validar

        max_eval = int(MAX_EVAL_DEF * factor)
        tiempo_lim = TIME_LIMITE_DEF * factor

        # --- Construcción de la lista de (overrides, n_reps) para esta fila ---
        configs: list[tuple[dict, int]] = []
        p_inter_clase = float(CONFIG_POR_MH[mh][clase].get("p_inter", 0.5))

        if mh == "sa" and clase == "egl" and tier != "A(1-3%)":
            # Fix diagnosticado: p_inter 0.6 -> 0.3 en SA/egl.
            configs = [({"p_inter": 0.3, "alpha_inter": 0.80}, reps_nuevas)]
        elif mh in MHS_MINIGRID and tier in ("C(10-25%)", "D(>25%)"):
            candidatos = sorted({round(p_inter_clase * 0.5, 3),
                                  p_inter_clase,
                                  round(min(0.9, p_inter_clase * 1.5), 3)})
            # Reparto proporcional a pesos [4,3,3,...] escalado a
            # reps_nuevas exactas (método de mayores restos: robusto para
            # cualquier reps_nuevas >= 1, incluido el recorte de --smoke).
            pesos = ([4, 3, 3] + [1] * len(candidatos))[:len(candidatos)]
            base_pesos = sum(pesos)
            crudo = [reps_nuevas * w / base_pesos for w in pesos]
            reparto = [int(x) for x in crudo]
            restos = sorted(range(len(candidatos)), key=lambda i: -(crudo[i] - reparto[i]))
            for i in restos[:reps_nuevas - sum(reparto)]:
                reparto[i] += 1
            for p_inter_cand, r in zip(candidatos, reparto):
                if r <= 0:
                    continue
                configs.append(({"p_inter": p_inter_cand}, r))
        else:
            configs = [({}, reps_nuevas)]

        base_semilla_tag = f"reintento{tier[0]}"
        for overrides, n_reps in configs:
            for rep in range(1, n_reps + 1):
                idx = len(tareas)
                parcial = (dir_parciales_por_mh[mh]
                           / f"{mh}_{inst}_{os.getpid()}_{idx}.csv")
                pid_semilla = f"{inst}{base_semilla_tag}{overrides.get('p_inter','-')}"
                tareas.append(Tarea(
                    mh=mh, instancia=inst, clase=clase, repeticion=rep,
                    semilla=_derivar_semilla(0, pid_semilla, mh, rep),
                    root=None, ruta_csv_parcial=str(parcial),
                    max_evaluaciones=max_eval, tiempo_limite_seg=tiempo_lim,
                    overrides=overrides,
                ))
    return tareas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--smoke", action="store_true",
                    help="Recorte drástico de presupuesto/reps para validar el flujo.")
    ap.add_argument("--solo-mh", type=str, default=None,
                    help="Filtrar a una sola MH (sa/ts/rts/abc/vdo) para pruebas.")
    args = ap.parse_args()

    filas = _leer_lista_problema()
    if args.solo_mh:
        filas = [f for f in filas if f["mh"] == args.solo_mh]
    if args.smoke:
        filas = filas[:6]

    tareas = _construir_tareas(filas, smoke=args.smoke)
    print(f"Filas de la lista de problema: {len(filas)}  ->  {len(tareas)} corridas")
    if not tareas:
        print("Nada que ejecutar.")
        return

    # Trazabilidad de la corrida.
    SALIDA_RAIZ.mkdir(parents=True, exist_ok=True)
    (SALIDA_RAIZ / "config_reintento.json").write_text(json.dumps({
        "generado": datetime.now().isoformat(timespec="seconds"),
        "smoke": args.smoke, "solo_mh": args.solo_mh,
        "n_filas_problema": len(filas), "n_corridas": len(tareas),
        "presupuesto_tier": PRESUPUESTO_TIER,
    }, indent=2), encoding="utf-8")

    ok = fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futuros = {ex.submit(_ejecutar_tarea, t): t for t in tareas}
        for i, fut in enumerate(as_completed(futuros), 1):
            tarea, estado, info, err = fut.result()
            if estado == "ok":
                ok += 1
                print(f"[{i}/{len(tareas)}] OK  {tarea.mh}/{tarea.instancia} "
                      f"costo={info['costo']:.1f} t={info['tiempo']:.0f}s "
                      f"ov={tarea.overrides}")
            else:
                fail += 1
                print(f"[{i}/{len(tareas)}] FAIL {tarea.mh}/{tarea.instancia}: {err}")

    print(f"\nOK={ok}  FAIL={fail}")
    for mh in MHS_COMPATIBLES:
        if not any(t.mh == mh for t in tareas):
            continue
        dir_final = SALIDA_RAIZ / mh / "final"
        n = _consolidar_por_instancia(dir_final / "_partials", dir_final, mh)
        print(f"[{mh}] consolidados {n} CSV finales -> {dir_final}")


if __name__ == "__main__":
    main()
