"""
Calibración de los parámetros restantes del programa experimental (20260601).

Sobre la base ``solo_p_inter`` con la config canónica fija (p_inter + knobs 1 y
2 ya calibrados), este script calibra los tres parámetros que no se habían
tocado en el programa experimental con costo corregido:

  1. FACTOR LAMBDA (transversal a todas las MH)
     Multiplica ``lambda_penal_capacidad_por_defecto(ctx)`` por un factor:
       {0.3, 1.0, 3.0}
     El factor 1.0 reproduce la config canónica actual; 0.3 la aligera y 3.0
     la endurece. La calibración usa SA como MH representativa (rápida,
     resultados claros) sobre las 23 instancias × 3 reps.

  2. ALPHA_INTER (solo MHs que lo aceptan: SA, TS, RTS)
     Probabilidad de proponer INTER cuando la solución es INFACTIBLE:
       {0.5, 0.7, 0.9}
     El valor canónico actual es 0.8. Calibración por MH sobre las 23
     instancias × 3 reps.

  3. UMBRAL PR (``max_iter_sin_mejora_kick``, solo approach 3 = pr_aislado)
     Número de niveles/iteraciones de estancamiento antes de disparar PR:
       {15, 30, 60}
     El valor canónico actual es 30. Calibración por MH sobre las 23
     instancias × 3 reps con selector p_inter.

Salida:
  <salida>/mejor_lambda.json       -> factor lambda ganador + gap medio
  <salida>/mejor_alpha_inter.json  -> alpha_inter ganador por MH + gap medio
  <salida>/mejor_umbral_pr.json    -> umbral PR ganador por MH + gap medio
  <salida>/_partials/              -> parciales CSV de las corridas

Uso:
  python scripts/_calibracion_restantes_20260601.py [--objetivo lambda|alpha|umbral|todos]
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

# Limitar hilos BLAS/OpenMP a 1 ANTES de importar numpy.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys as _sys
_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
if str(_SCRIPTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SCRIPTS_DIR))

from metacarp.vecindarios import OPERADORES_POPULARES

REPS_DEF = 3

INSTANCIAS = [
    "gdb19", "kshs1", "kshs2", "kshs3", "kshs4", "kshs5", "kshs6",
    "gdb4",  "gdb14", "gdb15", "gdb1",  "gdb20", "gdb3",  "gdb6",
    "gdb7",  "gdb12", "gdb10", "gdb2",  "gdb5",  "gdb13", "gdb16",
    "gdb17", "gdb21",
]

# Config canónica fija ya establecida.
P_INTER_FIJO = {"sa": 0.5, "tabu_simple": 0.4, "tabu_reactiva": 0.5,
                "abc_simple": 0.5, "cuckoo": 0.1}
KNOB1_FIJO   = {"sa": ("alpha", 0.90), "tabu_simple": ("tabu_tenure", 25),
                "tabu_reactiva": ("factor_aumento", 1.2),
                "abc_simple": ("num_fuentes", 30), "cuckoo": ("pa_abandono", 0.15)}
KNOB2_FIJO   = {"sa": ("max_reheats_sin_mejora", 10), "tabu_simple": ("tam_vecindario", 40),
                "tabu_reactiva": ("factor_reduccion", 0.95),
                "abc_simple": ("limite_abandono", 60), "cuckoo": ("beta_levy", 1.3)}

# Grids de calibración.
GRID_LAMBDA_FACTOR = (0.3, 1.0, 3.0)     # multiplicadores del lambda default
GRID_ALPHA_INTER   = (0.5, 0.7, 0.9)     # (MHs que lo aceptan: sa, ts, rts)
GRID_UMBRAL_PR     = (15, 30, 60)         # niveles/iters de estancamiento antes de PR

MH_ALPHA_INTER = ("sa", "tabu_simple", "tabu_reactiva")  # las que exponen alpha_inter
MH_MODULOS     = {
    "sa": "metacarp.recocido_simulado", "tabu_simple": "metacarp.busqueda_tabu_simple",
    "tabu_reactiva": "metacarp.busqueda_tabu_reactiva", "abc_simple": "metacarp.abejas_simple",
    "cuckoo": "metacarp.cuckoo_search",
}


# ============================================================
# Helpers compartidos
# ============================================================

@dataclass(frozen=True)
class Tarea:
    mh: str
    instancia: str
    repeticion: int
    param_nombre: str   # e.g. "lambda_factor", "alpha_inter", "umbral_pr"
    param_valor: float
    root: str | None
    ruta_csv: str


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


def _base_kwargs(mh: str) -> dict:
    """Kwargs canónicos fijos (config canónica completa con 2 knobs)."""
    k1, v1 = KNOB1_FIJO[mh]; k2, v2 = KNOB2_FIJO[mh]
    b = dict(operadores=OPERADORES_POPULARES, usar_gpu=False, semilla=None,
             guardar_csv=True, guardar_historial=False,
             lambda_capacidad=None, max_iter_sin_mejora_kick=None, max_resets=None,
             p_inter=P_INTER_FIJO[mh])
    b[k1] = v1; b[k2] = v2
    if mh == "sa":
        b.update(alpha_inter=0.80, temperatura_inicial=None, temperatura_minima=None,
                 patience=10, reheat_factor=0.5)
    elif mh in ("tabu_simple", "tabu_reactiva"):
        b["alpha_inter"] = 0.80
    return b


def _leer_gap(ruta: str) -> float:
    try:
        with open(ruta, "r", encoding="utf-8", newline="") as f:
            fila = next(csv.DictReader(f), None)
        return float(fila.get("gap_bks_porcentaje", "")) if fila else math.nan
    except (OSError, StopIteration, TypeError, ValueError):
        return math.nan


def _ejecutar_tarea_grid(t: Tarea):
    """Worker de nivel de módulo (serializable por ProcessPoolExecutor)."""
    try:
        runner = _cargar_runner(t.mh)
        kwargs = construir_kwargs_para(t)
        kwargs["ruta_csv"] = t.ruta_csv
        kwargs["root"] = t.root
        runner(t.instancia, **kwargs)
        return (t, "ok", _leer_gap(t.ruta_csv), None)
    except Exception as exc:  # noqa: BLE001
        return (t, "fail", math.nan, f"{type(exc).__name__}: {exc}")


def _correr_grid(
    tareas: list[Tarea], workers: int,
) -> dict[float, list[float]]:
    """Ejecuta las tareas y devuelve {param_valor: [gaps]}."""
    acum: dict[float, list[float]] = {}
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_ejecutar_tarea_grid, t) for t in tareas]):
            t, estado, gap, err = fut.result()
            if estado == "ok":
                ok += 1
                if not math.isnan(gap):
                    acum.setdefault(t.param_valor, []).append(gap)
            else:
                fail += 1
                print(f"  FAIL {t.mh}/{t.instancia} {t.param_nombre}="
                      f"{t.param_valor}: {err}")
    print(f"  OK={ok} FAIL={fail}")
    return acum


def _seleccionar_mejor(acum: dict[float, list[float]], nombre: str) -> tuple[float, float]:
    """Devuelve (mejor_valor, gap_medio) ordenado por gap_medio asc."""
    filas = [(v, sum(gs)/len(gs)) for v, gs in acum.items()]
    filas.sort(key=lambda r: r[1])
    print(f"  Gap medio por {nombre}:")
    for v, media in filas:
        print(f"    {nombre}={v:<6} -> {media:.3f}%")
    mejor = filas[0]
    print(f"  -> MEJOR: {nombre}={mejor[0]} (gap={mejor[1]:.3f}%)")
    return mejor


# ============================================================
# CALIBRACIÓN 1: FACTOR LAMBDA
# ============================================================

def _worker_lambda(tarea: Tarea):
    """Worker con lambda patched = factor × default."""
    try:
        import metacarp.evaluador_costo as _ec
        _orig_lam = _ec.lambda_penal_capacidad_por_defecto
        _factor = tarea.param_valor
        _ec.lambda_penal_capacidad_por_defecto = lambda ctx: _factor * _orig_lam(ctx)
        try:
            runner = _cargar_runner(tarea.mh)
            kw = _base_kwargs(tarea.mh)
            kw.update(ruta_csv=tarea.ruta_csv, root=tarea.root,
                      lambda_capacidad=None, guardar_csv=True, guardar_historial=False)
            res = runner(tarea.instancia, **kw)
            gap = _leer_gap(tarea.ruta_csv)
            return (tarea, "ok", gap, None)
        finally:
            _ec.lambda_penal_capacidad_por_defecto = _orig_lam
    except Exception as exc:  # noqa: BLE001
        return (tarea, "fail", math.nan, f"{type(exc).__name__}: {exc}")


def calibrar_lambda(dir_parciales: Path, instancias, reps, workers, root, mh_ref="sa"):
    """Calibra el factor lambda usando SA como MH de referencia."""
    print(f"\n=== CALIBRACIÓN 1: lambda_factor ∈ {list(GRID_LAMBDA_FACTOR)} "
          f"({mh_ref}, {len(instancias)} inst × {reps} reps) ===")
    tareas = []
    for factor in GRID_LAMBDA_FACTOR:
        for inst in instancias:
            for rep in range(1, reps + 1):
                idx = len(tareas)
                parcial = dir_parciales / f"lam_{mh_ref}_{inst}_{os.getpid()}_{idx}.csv"
                tareas.append(Tarea(mh_ref, inst, rep, "lambda_factor", factor, root, str(parcial)))

    acum: dict[float, list[float]] = {}
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_worker_lambda, t) for t in tareas]):
            t, estado, gap, err = fut.result()
            if estado == "ok":
                ok += 1
                if not math.isnan(gap):
                    acum.setdefault(t.param_valor, []).append(gap)
            else:
                fail += 1
                print(f"  FAIL {t.instancia} factor={t.param_valor}: {err}")
    print(f"  OK={ok} FAIL={fail}")
    mejor_factor, gap_medio = _seleccionar_mejor(acum, "lambda_factor")
    return {"param": "lambda_factor", "mh_ref": mh_ref,
            "mejor_valor": mejor_factor, "gap_medio": gap_medio,
            "todos": {str(v): round(sum(gs)/len(gs), 3)
                      for v, gs in acum.items()}}


# ============================================================
# CALIBRACIÓN 2: ALPHA_INTER
# ============================================================

def construir_kwargs_para(t: Tarea) -> dict:
    kw = _base_kwargs(t.mh)
    if t.param_nombre == "alpha_inter":
        kw["alpha_inter"] = t.param_valor
    elif t.param_nombre == "umbral_pr":
        # activar PR: hook + umbral variable
        from metacarp.path_relinking_limpio_20260531 import hook_pr_labels, hook_pr_ids
        kw["intensificador"] = hook_pr_ids if t.mh == "abc_simple" else hook_pr_labels
        kw["max_iter_sin_mejora_kick"] = int(t.param_valor)
        kw["max_resets"] = None
    return kw


def calibrar_alpha_inter(dir_parciales: Path, instancias, reps, workers, root):
    """Calibra alpha_inter para SA, TS simple y RTS."""
    print(f"\n=== CALIBRACIÓN 2: alpha_inter ∈ {list(GRID_ALPHA_INTER)} "
          f"(SA/TS/RTS) ===")
    resultados = {}
    for mh in MH_ALPHA_INTER:
        tareas = []
        for val in GRID_ALPHA_INTER:
            for inst in instancias:
                for rep in range(1, reps + 1):
                    idx = len(tareas)
                    parcial = dir_parciales / f"ai_{mh}_{inst}_{os.getpid()}_{idx}.csv"
                    tareas.append(Tarea(mh, inst, rep, "alpha_inter", val, root, str(parcial)))

        print(f"  [{mh}] {len(tareas)} corridas")
        acum = _correr_grid(tareas, workers)
        mejor_val, gap_medio = _seleccionar_mejor(acum, f"alpha_inter [{mh}]")
        resultados[mh] = {"mejor_valor": mejor_val, "gap_medio": gap_medio,
                          "todos": {str(v): round(sum(gs)/len(gs), 3)
                                    for v, gs in acum.items()}}
    return resultados


# ============================================================
# CALIBRACIÓN 3: UMBRAL PR
# ============================================================

def calibrar_umbral_pr(dir_parciales: Path, instancias, reps, workers, root):
    """Calibra el umbral de estancamiento que dispara el PR (todas las MH)."""
    print(f"\n=== CALIBRACIÓN 3: umbral_pr ∈ {list(GRID_UMBRAL_PR)} "
          f"(todas las MH, selector p_inter) ===")
    resultados = {}
    for mh in MH_MODULOS:
        tareas = []
        for val in GRID_UMBRAL_PR:
            for inst in instancias:
                for rep in range(1, reps + 1):
                    idx = len(tareas)
                    parcial = dir_parciales / f"upr_{mh}_{inst}_{os.getpid()}_{idx}.csv"
                    tareas.append(Tarea(mh, inst, rep, "umbral_pr", val, root, str(parcial)))

        print(f"  [{mh}] {len(tareas)} corridas")
        acum = _correr_grid(tareas, workers)
        mejor_val, gap_medio = _seleccionar_mejor(acum, f"umbral_pr [{mh}]")
        resultados[mh] = {"mejor_valor": int(mejor_val), "gap_medio": gap_medio,
                          "todos": {str(v): round(sum(gs)/len(gs), 3)
                                    for v, gs in acum.items()}}
    return resultados


# ============================================================
# CLI
# ============================================================

def main(argv=None):
    p = argparse.ArgumentParser(description="Calibración de parámetros restantes.")
    p.add_argument("--objetivo", choices=["lambda", "alpha", "umbral", "todos"],
                   default="todos")
    p.add_argument("--reps", type=int, default=REPS_DEF)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--instancias", nargs="*", default=None)
    p.add_argument("--salida", type=str,
                   default="experimentos_costo_fixed/_calibracion_restantes")
    args = p.parse_args(argv)

    instancias = args.instancias or INSTANCIAS
    salida = Path(args.salida).expanduser().resolve()
    dir_parciales = salida / "_partials"
    dir_parciales.mkdir(parents=True, exist_ok=True)

    objs = (["lambda", "alpha", "umbral"] if args.objetivo == "todos"
            else [args.objetivo])

    if "lambda" in objs:
        res = calibrar_lambda(dir_parciales, instancias, args.reps, args.workers, args.root)
        (salida / "mejor_lambda.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Guardado: {salida/'mejor_lambda.json'}")

    if "alpha" in objs:
        res = calibrar_alpha_inter(dir_parciales, instancias, args.reps, args.workers, args.root)
        (salida / "mejor_alpha_inter.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Guardado: {salida/'mejor_alpha_inter.json'}")

    if "umbral" in objs:
        res = calibrar_umbral_pr(dir_parciales, instancias, args.reps, args.workers, args.root)
        (salida / "mejor_umbral_pr.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Guardado: {salida/'mejor_umbral_pr.json'}")

    print("\n=== CALIBRACIÓN COMPLETA ===")
    for f in sorted(salida.glob("mejor_*.json")):
        print(f"  {f.name}: {f.read_text(encoding='utf-8')[:120].strip()}...")


if __name__ == "__main__":
    main()
