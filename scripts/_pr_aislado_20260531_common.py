"""
Approach ``pr_aislado_20260531`` — tercer approach: aislar Path Relinking.

Tercer approach del programa con el evaluador greedy nativo. Mide el efecto de
añadir PATH RELINKING (intensificación guiada hacia la mejor solución global)
sobre la base desnuda de un selector, SIN kick aleatorio, AOS ni budget.

PR LIMPIO E INDEPENDIENTE
=========================
A diferencia del PR del primer ciclo (``path_relinking_20260528``), aquí PR NO
es un "kick aumentado" ni usa frame-hacks. Es el módulo limpio
``metacarp.path_relinking_limpio_20260531`` invocado mediante un HOOK explícito
(``intensificador``) que las 5 MH aceptan como parámetro. En el punto de
estancamiento (mismo disparador que el kick, controlado por
``max_iter_sin_mejora_kick``) la MH ejecuta PR hacia la mejor solución global
EN LUGAR del kick aleatorio. Toda la transferencia de ``ctx``/``lambda``/mejor
solución es por argumentos explícitos.

SELECTOR PARAMETRIZABLE
=======================
El approach permite elegir el selector base con ``--selector``:
  - ``p_inter``: selector probabilístico nativo (como el approach 1), con el
    ``p_inter`` ya calibrado por MH en ``solo_p_inter_20260531``.
  - ``binario``: selector binario determinista por capacidad (como el approach
    2), instalado con ``aplicar_patch_selector`` en el worker.
El selector es una DIMENSIÓN del grid: se corren ambos para poder medir el
efecto de PR sobre cada base. El aislamiento se obtiene comparando el gap de
``[selector + PR]`` (este approach) contra ``[selector]`` (approach 1 / 2).

Diseño del grid (dos fases, sin semilla)
----------------------------------------
  selector (2) × parámetro libre por MH (3) = 6 configs por MH.
  Umbral de disparo de PR fijo (estancamiento), no se calibra.
  Fase 1 (3 reps): elige el mejor parámetro libre POR CADA selector.
  Fase 2 (5 reps): confirma la mejor config de cada selector (2 finales/MH).

  Parámetro libre por MH (idéntico a approaches 1/2):
    SA -> alpha {0.90,0.95,0.99}; TS -> tabu_tenure {7,15,25};
    RTS -> factor_aumento {1.1,1.2,1.3}; ABC -> num_fuentes {10,20,30};
    Cuckoo -> pa_abandono {0.15,0.25,0.35}.

Salida
------
  experimentos_costo_fixed/<mh>_pr_aislado_<YYYYMMDD-HHMM>/
    grid_parciales/        parciales fase 1
    calibracion_todas.csv  consolidado fase 1
    grid_resumen.csv        gap medio por (selector, parámetro libre)
    mejor_config.json       mejor config por selector
    final/                 CSVs fase 2 (5 reps) por (selector, instancia)
      _partials/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Limitar hilos BLAS/OpenMP a 1 ANTES de importar numpy (evita sobre-suscripción
# con ProcessPoolExecutor). Ver approaches 1/2.
for _var_hilos in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var_hilos, "1")

# Raíz del proyecto + ``scripts/`` en sys.path (este último para que los workers
# importen módulos hermanos: ``_strict_intra_inter_20260524_common``).
import sys as _sys
_ROOT_PROYECTO = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR   = Path(__file__).resolve().parent
if str(_ROOT_PROYECTO) not in _sys.path:
    _sys.path.insert(0, str(_ROOT_PROYECTO))
if str(_SCRIPTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SCRIPTS_DIR))

from metacarp.vecindarios import OPERADORES_POPULARES


# ============================================================
# Constantes del experimento
# ============================================================

ALPHA_INTER_FIJO: float = 0.80
REPS_CALIBRACION_DEF: int = 3
REPS_FINAL_DEF: int = 5

# Umbral de estancamiento que dispara el PR (en niveles para SA, en iteraciones
# para el resto). Fijo, no se calibra: queremos aislar PR, no su cadencia.
UMBRAL_PR_DEF: int = 30

# 23 instancias pequeñas del corpus (idénticas a los approaches 1/2).
INSTANCIAS = [
    "gdb19", "kshs1", "kshs2", "kshs3", "kshs4", "kshs5", "kshs6",
    "gdb4",  "gdb14", "gdb15", "gdb1",  "gdb20", "gdb3",  "gdb6",
    "gdb7",  "gdb12", "gdb10", "gdb2",  "gdb5",  "gdb13", "gdb16",
    "gdb17", "gdb21",
]

MH_MODULOS: dict[str, str] = {
    "sa":             "metacarp.recocido_simulado",
    "tabu_simple":    "metacarp.busqueda_tabu_simple",
    "tabu_reactiva":  "metacarp.busqueda_tabu_reactiva",
    "abc_simple":     "metacarp.abejas_simple",
    "cuckoo":         "metacarp.cuckoo_search",
}

# Parámetro libre por MH (igual que approaches 1/2).
GRID_LIBRE: dict[str, tuple[str, tuple[float, ...]]] = {
    "sa":            ("alpha",          (0.90, 0.95, 0.99)),
    "tabu_simple":   ("tabu_tenure",    (7, 15, 25)),
    "tabu_reactiva": ("factor_aumento", (1.1, 1.2, 1.3)),
    "abc_simple":    ("num_fuentes",    (10, 20, 30)),
    "cuckoo":        ("pa_abandono",    (0.15, 0.25, 0.35)),
}

# p_inter calibrado por MH en el approach 1 (solo_p_inter_20260531). Se usa
# cuando ``--selector p_inter``.
P_INTER_CALIBRADO: dict[str, float] = {
    "sa":            0.5,
    "tabu_simple":   0.4,
    "tabu_reactiva": 0.5,
    "abc_simple":    0.5,
    "cuckoo":        0.1,
}

SELECTORES = ("p_inter", "binario")


# ============================================================
# Dataclass de una corrida
# ============================================================

@dataclass(frozen=True)
class Tarea:
    """Una corrida del grid (sin semilla fija)."""
    mh: str
    instancia: str
    repeticion: int
    selector: str       # "p_inter" | "binario"
    libre_nombre: str
    libre_valor: float
    fase: str           # "calibracion" | "final"
    root: str | None
    ruta_csv_parcial: str


# ============================================================
# Construcción de kwargs por MH
# ============================================================

def construir_kwargs(tarea: Tarea) -> dict:
    """Arma los kwargs del runner para esta corrida (PR activo + selector).

    El PR se activa pasando el ``intensificador`` (hook limpio) y un umbral de
    estancamiento (``max_iter_sin_mejora_kick=UMBRAL_PR``): en el estancamiento
    la MH ejecuta PR hacia la mejor solución global en lugar del kick aleatorio.
    El selector se configura según ``tarea.selector`` (el binario se instala en
    el worker, ver ``ejecutar_una``).
    """
    # Importación diferida del hook (módulo PR limpio).
    from metacarp.path_relinking_limpio_20260531 import hook_pr_labels, hook_pr_ids
    hook = hook_pr_ids if tarea.mh == "abc_simple" else hook_pr_labels

    base = dict(
        root=tarea.root,
        operadores=OPERADORES_POPULARES,
        usar_gpu=False,
        semilla=None,
        repeticion=tarea.repeticion,
        guardar_csv=True,
        ruta_csv=tarea.ruta_csv_parcial,
        guardar_historial=False,
        lambda_capacidad=None,                  # λ default instance-aware
        # PR como respuesta al estancamiento (NO kick aleatorio):
        max_iter_sin_mejora_kick=UMBRAL_PR_DEF,
        max_resets=None,
        intensificador=hook,
        extra_csv={
            "experimento": "pr_aislado_20260531",
            "approach":    "pr_aislado",
            "selector":    tarea.selector,
            "intensificador": "path_relinking_limpio",
            "umbral_pr":   str(UMBRAL_PR_DEF),
            "fase":        tarea.fase,
            "lambda":      "default_instance_aware",
        },
    )

    # Selector probabilístico nativo: pasar el p_inter calibrado (SA/TS/RTS
    # aceptan alpha_inter; ABC/Cuckoo lo ignoran internamente).
    if tarea.selector == "p_inter":
        base["p_inter"] = P_INTER_CALIBRADO[tarea.mh]
        if tarea.mh in ("sa", "tabu_simple", "tabu_reactiva"):
            base["alpha_inter"] = ALPHA_INTER_FIJO
    # Si selector == "binario": el patch del selector se aplica en el worker;
    # no se pasa p_inter (el selector binario lo ignora de todas formas).

    mh = tarea.mh
    if mh == "sa":
        base.update(
            alpha=float(tarea.libre_valor),
            temperatura_inicial=None,
            temperatura_minima=None,
            # Reheat acotado (imprescindible para alpha alto; ver approaches 1/2).
            patience=10,
            reheat_factor=0.5,
            max_reheats_sin_mejora=5,
        )
    elif mh == "tabu_simple":
        base.update(tabu_tenure=int(tarea.libre_valor), tam_vecindario=25)
    elif mh == "tabu_reactiva":
        base.update(factor_aumento=float(tarea.libre_valor), factor_reduccion=0.9)
    elif mh == "abc_simple":
        base.update(num_fuentes=int(tarea.libre_valor))
    elif mh == "cuckoo":
        base.update(pa_abandono=float(tarea.libre_valor), beta_levy=1.5)
    else:
        raise ValueError(f"MH desconocida: {mh!r}")
    return base


def _cargar_runner(mh: str):
    if mh == "sa":
        from metacarp import recocido_simulado_desde_instancia as runner
    elif mh == "tabu_simple":
        from metacarp import busqueda_tabu_simple_desde_instancia as runner
    elif mh == "tabu_reactiva":
        from metacarp import busqueda_tabu_reactiva_desde_instancia as runner
    elif mh == "abc_simple":
        from metacarp import busqueda_abejas_simple_desde_instancia as runner
    elif mh == "cuckoo":
        from metacarp import cuckoo_search_desde_instancia as runner
    else:
        raise ValueError(f"MH desconocida: {mh!r}")
    return runner


def _leer_gap_del_parcial(ruta_csv: str) -> float:
    try:
        with open(ruta_csv, "r", encoding="utf-8", newline="") as f:
            fila = next(csv.DictReader(f), None)
        if not fila:
            return math.nan
        return float(fila.get("gap_bks_porcentaje", ""))
    except (OSError, StopIteration, TypeError, ValueError):
        return math.nan


# ============================================================
# Worker
# ============================================================

def _instalar_selector(nombre_modulo_mh: str, selector: str) -> None:
    """Instala el selector correcto en el modulo de la MH, de forma EXPLICITA.

    CRITICO: ``ProcessPoolExecutor`` REUTILIZA los procesos worker entre tareas,
    y el patch del selector binario (``seleccionar_grupo_strict``) reasigna el
    simbolo del modulo MH de forma PERMANENTE en ese proceso. Si un worker
    corre primero una tarea ``binario`` y luego una ``p_inter``, esta ultima
    heredaria el selector binario (contaminacion). Para evitarlo, en CADA tarea
    fijamos explicitamente el selector que corresponde:
      - ``binario``: instala ``seleccionar_grupo_strict``.
      - ``p_inter``: RESTAURA el selector probabilistico canonico de
        ``metaheuristicas_utils`` (deshace cualquier patch previo en el worker).
    """
    import importlib
    mh = importlib.import_module(nombre_modulo_mh)
    if selector == "binario":
        from metacarp.strict_intra_inter_20260524 import seleccionar_grupo_strict
        mh.seleccionar_grupo_operadores_inter_intra = seleccionar_grupo_strict
    else:  # "p_inter": selector probabilistico nativo (restaurado).
        from metacarp.metaheuristicas_utils import (
            seleccionar_grupo_operadores_inter_intra as _orig,
        )
        mh.seleccionar_grupo_operadores_inter_intra = _orig


def ejecutar_una(tarea: Tarea) -> tuple[Tarea, str, dict | None, str | None]:
    """Ejecuta una corrida instalando EXPLICITAMENTE el selector de la tarea."""
    try:
        # Fijamos el selector en CADA tarea (los workers se reutilizan: hay que
        # deshacer cualquier patch heredado de una tarea previa en este proceso).
        _instalar_selector(MH_MODULOS[tarea.mh], tarea.selector)

        runner = _cargar_runner(tarea.mh)
        kwargs = construir_kwargs(tarea)
        res = runner(tarea.instancia, **kwargs)
        info = {
            "costo":    res.mejor_costo,
            "tiempo":   res.tiempo_segundos,
            "gap":      _leer_gap_del_parcial(tarea.ruta_csv_parcial),
            "factible": getattr(res, "mejor_solucion_factible_final", None),
        }
        return (tarea, "ok", info, None)
    except Exception as exc:  # noqa: BLE001
        return (tarea, "fail", None, f"{type(exc).__name__}: {exc}")


# ============================================================
# Consolidación de parciales
# ============================================================

def _consolidar(parciales: list[Path], ruta_final: Path) -> int:
    filas: list[dict] = []
    columnas: list[str] = []
    vistas: set[str] = set()
    for parcial in sorted(parciales):
        with parcial.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for col in reader.fieldnames or []:
                if col not in vistas:
                    vistas.add(col)
                    columnas.append(col)
            filas.extend(reader)
    if not filas:
        return 0
    ruta_final.parent.mkdir(parents=True, exist_ok=True)
    with ruta_final.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(filas)
    return len(filas)


def _instancia_de_parcial(stem: str, mh: str) -> str | None:
    """Extrae la instancia de un stem ``{mh}_{selector}_{instancia}_{pid}_{idx}``.

    Robusto frente a MH/instancias con guion bajo: quita el prefijo
    ``{mh}_{selector}_`` y descarta los dos últimos tokens (pid, idx).
    """
    for sel in SELECTORES:
        prefix = f"{mh}_{sel}_"
        if stem.startswith(prefix):
            partes = stem[len(prefix):].split("_")
            if len(partes) < 3:
                return None
            return "_".join(partes[:-2])
    return None


# ============================================================
# Ejecución paralela
# ============================================================

def _correr_tareas(tareas: list[Tarea], workers: int) -> list[tuple[Tarea, dict]]:
    resultados: list[tuple[Tarea, dict]] = []
    ok = fail = 0
    if workers <= 1:
        for tarea in tareas:
            tarea, estado, info, err = ejecutar_una(tarea)
            ok, fail = _reportar(tarea, estado, info, err, ok, fail, resultados)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futuros = {ex.submit(ejecutar_una, t): t for t in tareas}
            for fut in as_completed(futuros):
                tarea, estado, info, err = fut.result()
                ok, fail = _reportar(tarea, estado, info, err, ok, fail, resultados)
    print("-" * 80)
    print(f"  OK={ok}  FAIL={fail}  (de {len(tareas)})")
    return resultados


def _reportar(tarea, estado, info, err, ok, fail, resultados) -> tuple[int, int]:
    if estado == "ok":
        gap = info.get("gap", math.nan)
        gap_str = "nan" if (gap is None or math.isnan(gap)) else f"{gap:.2f}%"
        print(f"  [{tarea.mh}/{tarea.selector}/{tarea.instancia}] "
              f"{tarea.libre_nombre}={tarea.libre_valor} rep={tarea.repeticion} "
              f"| costo={info['costo']:.4f} | gap={gap_str} | t={info['tiempo']:.2f}s")
        resultados.append((tarea, info))
        return ok + 1, fail
    print(f"  [{tarea.mh}/{tarea.selector}/{tarea.instancia}] "
          f"{tarea.libre_nombre}={tarea.libre_valor} rep={tarea.repeticion} | FAIL: {err}")
    return ok, fail + 1


# ============================================================
# Fase 1: calibración (grid selector × parámetro libre)
# ============================================================

def fase1_calibracion(
    mh: str, carpeta: Path, *, instancias, reps, workers, root,
    selectores, grid_libre,
) -> dict[str, dict]:
    """Barre (selector × parámetro libre) y elige el mejor libre POR selector.

    Devuelve {selector: mejor_config} y escribe grid_resumen.csv / parciales.
    """
    libre_nombre = GRID_LIBRE[mh][0]
    dir_parciales = carpeta / "grid_parciales"
    dir_parciales.mkdir(parents=True, exist_ok=True)

    tareas: list[Tarea] = []
    for selector in selectores:
        for libre_valor in grid_libre:
            for instancia in instancias:
                for rep in range(1, reps + 1):
                    idx = len(tareas)
                    parcial = dir_parciales / (
                        f"{mh}_{selector}_{instancia}_{os.getpid()}_{idx}.csv"
                    )
                    tareas.append(Tarea(
                        mh=mh, instancia=instancia, repeticion=rep,
                        selector=selector, libre_nombre=libre_nombre,
                        libre_valor=libre_valor, fase="calibracion",
                        root=root, ruta_csv_parcial=str(parcial),
                    ))

    print("=" * 80)
    print(f"FASE 1 — CALIBRACION  MH={mh}  approach=pr_aislado")
    print(f"  selectores={list(selectores)}  libre={libre_nombre}x{len(grid_libre)}  "
          f"instancias={len(instancias)}  reps={reps}  corridas={len(tareas)}")
    print("=" * 80)

    resultados = _correr_tareas(tareas, workers)
    _consolidar(sorted(dir_parciales.glob(f"{mh}_*.csv")),
                carpeta / "calibracion_todas.csv")

    # Gap medio por (selector, libre_valor).
    acum: dict[tuple[str, float], list[float]] = {}
    for tarea, info in resultados:
        gap = info.get("gap", math.nan)
        if gap is None or math.isnan(gap):
            continue
        acum.setdefault((tarea.selector, tarea.libre_valor), []).append(gap)

    filas_resumen: list[dict] = []
    for (selector, libre_valor), gaps in acum.items():
        media = sum(gaps) / len(gaps)
        var = sum((g - media) ** 2 for g in gaps) / len(gaps)
        filas_resumen.append({
            "mh": mh, "selector": selector, libre_nombre: libre_valor,
            "gap_medio": round(media, 4), "gap_std": round(math.sqrt(var), 4),
            "n_corridas": len(gaps),
        })
    filas_resumen.sort(key=lambda r: (r["selector"], r["gap_medio"], r["gap_std"]))

    with (carpeta / "grid_resumen.csv").open("w", encoding="utf-8", newline="") as f:
        campos = ["mh", "selector", libre_nombre, "gap_medio", "gap_std", "n_corridas"]
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas_resumen)

    if not filas_resumen:
        raise RuntimeError(f"FASE 1 sin gaps válidos para MH={mh}.")

    # Mejor config por selector (menor gap medio).
    mejor_por_sel: dict[str, dict] = {}
    for fila in filas_resumen:
        sel = fila["selector"]
        if sel not in mejor_por_sel:  # ya viene ordenado por gap dentro del selector
            mejor_por_sel[sel] = {
                "mh": mh, "selector": sel, "libre_nombre": libre_nombre,
                "libre_valor": fila[libre_nombre], "gap_medio": fila["gap_medio"],
                "gap_std": fila["gap_std"],
            }
    for sel, cfg in mejor_por_sel.items():
        print(f"  -> MEJOR {mh}/{sel}: {libre_nombre}={cfg['libre_valor']} "
              f"| gap_medio={cfg['gap_medio']}%")
    return mejor_por_sel


# ============================================================
# Fase 2: corrida final (mejor config por selector)
# ============================================================

def fase2_final(
    mh: str, carpeta: Path, mejor_por_sel: dict[str, dict], *,
    instancias, reps, workers, root,
) -> None:
    dir_final = carpeta / "final"
    dir_partials = dir_final / "_partials"
    dir_partials.mkdir(parents=True, exist_ok=True)

    tareas: list[Tarea] = []
    for selector, cfg in mejor_por_sel.items():
        for instancia in instancias:
            for rep in range(1, reps + 1):
                idx = len(tareas)
                parcial = dir_partials / (
                    f"{mh}_{selector}_{instancia}_{os.getpid()}_{idx}.csv"
                )
                tareas.append(Tarea(
                    mh=mh, instancia=instancia, repeticion=rep,
                    selector=selector, libre_nombre=cfg["libre_nombre"],
                    libre_valor=cfg["libre_valor"], fase="final",
                    root=root, ruta_csv_parcial=str(parcial),
                ))

    print("=" * 80)
    print(f"FASE 2 — FINAL  MH={mh}  configs por selector: "
          + "; ".join(f"{s}:{c['libre_nombre']}={c['libre_valor']}"
                      for s, c in mejor_por_sel.items()))
    print(f"  instancias={len(instancias)}  reps={reps}  corridas={len(tareas)}")
    print("=" * 80)

    _correr_tareas(tareas, workers)

    # Consolidar por (selector, instancia).
    grupos: dict[tuple[str, str], list[Path]] = {}
    for parcial in sorted(dir_partials.glob(f"{mh}_*.csv")):
        inst = _instancia_de_parcial(parcial.stem, mh)
        sel = next((s for s in SELECTORES if parcial.stem.startswith(f"{mh}_{s}_")), None)
        if inst is None or sel is None:
            continue
        grupos.setdefault((sel, inst), []).append(parcial)
    n = 0
    for (sel, inst), archivos in grupos.items():
        ruta = dir_final / f"{mh}_pr_{sel}_{inst}.csv"
        n += 1 if _consolidar(archivos, ruta) else 0
    print(f"  -> {n} CSV finales en {dir_final}")


# ============================================================
# Orquestación de una MH
# ============================================================

def ejecutar_experimento_mh(
    mh: str, *, salida_base, instancias, reps_calibracion, reps_final,
    workers, root, selectores, grid_libre, solo_fase,
) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    carpeta = Path(salida_base).expanduser().resolve() / f"{mh}_pr_aislado_{ts}"
    carpeta.mkdir(parents=True, exist_ok=True)
    print(f"\n### Experimento pr_aislado | MH={mh} | salida={carpeta}\n")

    mejor_por_sel: dict[str, dict] | None = None
    if solo_fase in ("ambas", "1"):
        mejor_por_sel = fase1_calibracion(
            mh, carpeta, instancias=instancias, reps=reps_calibracion,
            workers=workers, root=root, selectores=selectores, grid_libre=grid_libre,
        )
        (carpeta / "mejor_config.json").write_text(
            json.dumps(mejor_por_sel, indent=2, ensure_ascii=False), encoding="utf-8")

    if solo_fase in ("ambas", "2"):
        if mejor_por_sel is None:
            ruta_cfg = carpeta / "mejor_config.json"
            if not ruta_cfg.exists():
                raise RuntimeError("Fase 2 sin mejor_config.json; corre antes la fase 1.")
            mejor_por_sel = json.loads(ruta_cfg.read_text(encoding="utf-8"))
        fase2_final(mh, carpeta, mejor_por_sel, instancias=instancias,
                    reps=reps_final, workers=workers, root=root)
    return carpeta


# ============================================================
# CLI
# ============================================================

def _parse_instancias(items: list[str] | None) -> list[str]:
    if not items:
        return list(INSTANCIAS)
    bruto: list[str] = []
    for item in items:
        for tok in item.split(","):
            tok = tok.strip()
            if tok:
                bruto.append(tok)
    efectivas = [i for i in INSTANCIAS if i in bruto]
    for nombre in bruto:
        if nombre not in efectivas:
            efectivas.append(nombre)
    return efectivas


def main(argv: list[str] | None = None, *, mh_fijo: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Approach pr_aislado: Path Relinking limpio sobre selector p_inter/binario."
    )
    if mh_fijo is None:
        parser.add_argument("--mh", type=str, required=True, choices=list(MH_MODULOS.keys()))
    parser.add_argument("--salida-base", type=str, default="experimentos_costo_fixed")
    parser.add_argument("--selector", type=str, default="ambos",
                        choices=["p_inter", "binario", "ambos"])
    parser.add_argument("--reps-calibracion", type=int, default=REPS_CALIBRACION_DEF)
    parser.add_argument("--reps-final", type=int, default=REPS_FINAL_DEF)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--instancias", type=str, default=None, nargs="*")
    parser.add_argument("--solo-fase", type=str, default="ambas", choices=["ambas", "1", "2"])
    parser.add_argument("--smoke", action="store_true",
                        help="Malla mínima + 1 rep + 2 instancias (prueba rápida).")
    args = parser.parse_args(argv)

    mh = mh_fijo if mh_fijo is not None else args.mh
    instancias = _parse_instancias(args.instancias)
    selectores = SELECTORES if args.selector == "ambos" else (args.selector,)
    grid_libre = GRID_LIBRE[mh][1]
    reps_cal, reps_fin = args.reps_calibracion, args.reps_final

    if args.smoke:
        if not args.instancias:
            instancias = ["gdb19", "kshs1"]
        grid_libre = grid_libre[:2]
        reps_cal = reps_fin = 1

    ejecutar_experimento_mh(
        mh, salida_base=args.salida_base, instancias=instancias,
        reps_calibracion=reps_cal, reps_final=reps_fin, workers=args.workers,
        root=args.root, selectores=selectores, grid_libre=grid_libre,
        solo_fase=args.solo_fase,
    )


if __name__ == "__main__":
    main()
