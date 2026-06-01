"""
Calibración del SEGUNDO knob por metaheurística (20260601).

Sobre la base ``solo_p_inter`` (selector p_inter nativo, sin PR/kick/AOS/budget,
operadores completos, lambda default instance-aware) y con ``p_inter`` y el knob
PRINCIPAL ya FIJOS a su mejor valor canónico por MH, este script calibra el
parámetro adicional MÁS INFLUYENTE de cada metaheurística que aún no se había
tocado:

  SA     -> max_reheats_sin_mejora ∈ {3, 5, 10}   (persistencia del reheat)
  TS     -> tam_vecindario         ∈ {15, 25, 40} (muestreo del vecindario)
  RTS    -> factor_reduccion       ∈ {0.85, 0.90, 0.95}
  ABC    -> limite_abandono        ∈ {15, 30, 60} (límite de scout)
  Cuckoo -> beta_levy              ∈ {1.3, 1.5, 1.7} (exponente de Lévy)

Para cada MH se corre el grid del 2º knob (3 valores) × 23 instancias × 3 reps
(sin semilla) y se elige el valor con MENOR gap medio. El resultado se imprime y
se guarda en ``<salida>/mejor_2knob.json`` para incorporarlo luego a la config
canónica de los tres approaches.

Es un script de calibración de un solo uso (análogo a la fase 1 del primer
ciclo), independiente de los orquestadores de los approaches.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# Limitar hilos BLAS/OpenMP a 1 ANTES de importar numpy (evita sobre-suscripción).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys as _sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from metacarp.vecindarios import OPERADORES_POPULARES

REPS_DEF = 3
ALPHA_INTER_FIJO = 0.80

INSTANCIAS = [
    "gdb19", "kshs1", "kshs2", "kshs3", "kshs4", "kshs5", "kshs6",
    "gdb4",  "gdb14", "gdb15", "gdb1",  "gdb20", "gdb3",  "gdb6",
    "gdb7",  "gdb12", "gdb10", "gdb2",  "gdb5",  "gdb13", "gdb16",
    "gdb17", "gdb21",
]

# Config canónica YA fijada (p_inter + knob principal), por MH.
P_INTER_FIJO = {"sa": 0.5, "tabu_simple": 0.4, "tabu_reactiva": 0.5,
                "abc_simple": 0.5, "cuckoo": 0.1}
KNOB1_FIJO = {  # (nombre, valor) del knob principal ya calibrado
    "sa":            ("alpha",          0.90),
    "tabu_simple":   ("tabu_tenure",    25),
    "tabu_reactiva": ("factor_aumento", 1.2),
    "abc_simple":    ("num_fuentes",    30),
    "cuckoo":        ("pa_abandono",    0.15),
}
# Segundo knob a calibrar (nombre, grid de valores).
KNOB2_GRID = {
    "sa":            ("max_reheats_sin_mejora", (3, 5, 10)),
    "tabu_simple":   ("tam_vecindario",         (15, 25, 40)),
    "tabu_reactiva": ("factor_reduccion",       (0.85, 0.90, 0.95)),
    "abc_simple":    ("limite_abandono",        (15, 30, 60)),
    "cuckoo":        ("beta_levy",              (1.3, 1.5, 1.7)),
}


@dataclass(frozen=True)
class Tarea:
    mh: str
    instancia: str
    repeticion: int
    knob2_valor: float
    root: str | None
    ruta_csv_parcial: str


def _cargar_runner(mh: str):
    if mh == "sa":
        from metacarp import recocido_simulado_desde_instancia as r
    elif mh == "tabu_simple":
        from metacarp import busqueda_tabu_simple_desde_instancia as r
    elif mh == "tabu_reactiva":
        from metacarp import busqueda_tabu_reactiva_desde_instancia as r
    elif mh == "abc_simple":
        from metacarp import busqueda_abejas_simple_desde_instancia as r
    elif mh == "cuckoo":
        from metacarp import cuckoo_search_desde_instancia as r
    else:
        raise ValueError(mh)
    return r


def construir_kwargs(tarea: Tarea) -> dict:
    """Kwargs de la base solo_p_inter (canónica) + el 2º knob de esta corrida."""
    mh = tarea.mh
    k1_nombre, k1_val = KNOB1_FIJO[mh]
    k2_nombre = KNOB2_GRID[mh][0]
    base = dict(
        root=tarea.root,
        operadores=OPERADORES_POPULARES,
        usar_gpu=False,
        semilla=None,
        repeticion=tarea.repeticion,
        guardar_csv=True,
        ruta_csv=tarea.ruta_csv_parcial,
        guardar_historial=False,
        lambda_capacidad=None,
        p_inter=P_INTER_FIJO[mh],
        max_iter_sin_mejora_kick=None,   # kick OFF (base solo_p_inter)
        max_resets=None,
    )
    base[k1_nombre] = k1_val          # knob principal fijo
    base[k2_nombre] = tarea.knob2_valor  # 2º knob a calibrar
    if mh == "sa":
        base.update(alpha_inter=ALPHA_INTER_FIJO,
                    temperatura_inicial=None, temperatura_minima=None,
                    patience=10, reheat_factor=0.5)  # resto del reheat fijo
    elif mh in ("tabu_simple", "tabu_reactiva"):
        base["alpha_inter"] = ALPHA_INTER_FIJO
    return base


def _leer_gap(ruta_csv: str) -> float:
    try:
        with open(ruta_csv, "r", encoding="utf-8", newline="") as f:
            fila = next(csv.DictReader(f), None)
        return float(fila.get("gap_bks_porcentaje", "")) if fila else math.nan
    except (OSError, StopIteration, TypeError, ValueError):
        return math.nan


def ejecutar_una(tarea: Tarea):
    try:
        runner = _cargar_runner(tarea.mh)
        res = runner(tarea.instancia, **construir_kwargs(tarea))
        return (tarea, "ok", {"gap": _leer_gap(tarea.ruta_csv_parcial),
                              "costo": res.mejor_costo}, None)
    except Exception as exc:  # noqa: BLE001
        return (tarea, "fail", None, f"{type(exc).__name__}: {exc}")


def calibrar_mh(mh: str, *, dir_parciales: Path, instancias, reps, workers, root):
    k2_nombre, k2_grid = KNOB2_GRID[mh]
    tareas = []
    for v in k2_grid:
        for inst in instancias:
            for rep in range(1, reps + 1):
                idx = len(tareas)
                parcial = dir_parciales / f"{mh}_{inst}_{os.getpid()}_{idx}.csv"
                tareas.append(Tarea(mh, inst, rep, v, root, str(parcial)))

    print(f"\n=== Calibrando {mh}: {k2_nombre} ∈ {list(k2_grid)} "
          f"({len(tareas)} corridas) ===")
    acum: dict[float, list[float]] = {}
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(ejecutar_una, t) for t in tareas]):
            tarea, estado, info, err = fut.result()
            if estado == "ok":
                ok += 1
                g = info["gap"]
                if g is not None and not math.isnan(g):
                    acum.setdefault(tarea.knob2_valor, []).append(g)
            else:
                fail += 1
                print(f"  FAIL {tarea.mh}/{tarea.instancia} {k2_nombre}="
                      f"{tarea.knob2_valor}: {err}")
    filas = []
    for v, gaps in acum.items():
        media = sum(gaps) / len(gaps)
        filas.append((v, round(media, 3), len(gaps)))
    filas.sort(key=lambda r: r[1])
    print(f"  OK={ok} FAIL={fail}. Gap medio por {k2_nombre}:")
    for v, media, n in filas:
        print(f"    {k2_nombre}={v:<6} -> gap_medio={media}% (n={n})")
    mejor = filas[0]
    print(f"  -> MEJOR {mh}: {k2_nombre}={mejor[0]} (gap_medio={mejor[1]}%)")
    return {"mh": mh, "knob2_nombre": k2_nombre, "knob2_valor": mejor[0],
            "gap_medio": mejor[1],
            "todos": {str(v): m for v, m, _ in filas}}


def main(argv=None):
    p = argparse.ArgumentParser(description="Calibración del 2º knob por MH.")
    p.add_argument("--mhs", nargs="*", default=list(KNOB2_GRID.keys()))
    p.add_argument("--reps", type=int, default=REPS_DEF)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--instancias", nargs="*", default=None)
    p.add_argument("--salida", type=str, default="experimentos_costo_fixed/_calibracion_2knob")
    args = p.parse_args(argv)

    instancias = args.instancias or INSTANCIAS
    salida = Path(args.salida).expanduser().resolve()
    dir_parciales = salida / "_partials"
    dir_parciales.mkdir(parents=True, exist_ok=True)

    resultados = {}
    for mh in args.mhs:
        resultados[mh] = calibrar_mh(
            mh, dir_parciales=dir_parciales, instancias=instancias,
            reps=args.reps, workers=args.workers, root=args.root,
        )
    (salida / "mejor_2knob.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== RESUMEN: mejor 2º knob por MH ===")
    for mh, r in resultados.items():
        print(f"  {mh:14} {r['knob2_nombre']}={r['knob2_valor']}  "
              f"(gap_medio={r['gap_medio']}%)")
    print(f"\nGuardado en {salida/'mejor_2knob.json'}")


if __name__ == "__main__":
    main()
