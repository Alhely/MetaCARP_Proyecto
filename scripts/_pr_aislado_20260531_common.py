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
    ``p_inter`` canonico por MH definido en ``P_INTER_FIJO``.
  - ``binario``: selector binario determinista por capacidad (como el approach
    2), instalado con ``aplicar_patch_selector`` en el worker.
El selector es una DIMENSIÓN del experimento: se corren ambos por defecto para
poder medir el efecto de PR sobre cada base. El aislamiento se obtiene
comparando el gap de ``[selector + PR]`` (este approach) contra ``[selector]``
(approach 1 / 2).

Config canonica (sin grid, sin calibracion)
-------------------------------------------
  El approach corre UNA SOLA configuracion fija por MH × selector (2 configs/MH):
    p_inter por MH (solo para selector p_inter):
      sa=0.5, tabu_simple=0.4, tabu_reactiva=0.5, abc_simple=0.5, cuckoo=0.1
    Knob libre fijo por MH:
      sa->alpha=0.90, tabu_simple->tabu_tenure=25,
      tabu_reactiva->factor_aumento=1.2, abc_simple->num_fuentes=30,
      cuckoo->pa_abandono=0.15
  Umbral de disparo de PR fijo (estancamiento), no se calibra: queremos aislar
  PR, no su cadencia.

  La instalación EXPLÍCITA del selector en cada tarea (``_instalar_selector``)
  es CRÍTICA: ``ProcessPoolExecutor`` reutiliza workers entre tareas, y el patch
  binario contaminaría las tareas p_inter del mismo worker si no se restaura.

Salida
------
  experimentos_costo_fixed/<mh>_pr_aislado_<YYYYMMDD-HHMM>/
    final/                 CSVs por (selector, instancia)
      _partials/           parciales antes de consolidar
    config_fija.json       config usada (trazabilidad)
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
REPS_DEF: int = 5

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

# Config canonica: p_inter fijo por MH (usado solo con selector p_inter).
# Coincide con P_INTER_CALIBRADO del approach 1, ahora renombrado para claridad.
P_INTER_FIJO: dict[str, float] = {
    "sa":            0.5,
    "tabu_simple":   0.4,
    "tabu_reactiva": 0.5,
    "abc_simple":    0.5,
    "cuckoo":        0.1,
}

# Config canonica: knob libre fijo por MH (nombre_kwarg, valor).
# Identico al de los approaches 1/2; sin grid, se usa directamente.
CONFIG_FIJA: dict[str, tuple[str, float]] = {
    "sa":            ("alpha",          0.90),
    "tabu_simple":   ("tabu_tenure",    25),
    "tabu_reactiva": ("factor_aumento", 1.2),
    "abc_simple":    ("num_fuentes",    30),
    "cuckoo":        ("pa_abandono",    0.15),
}

SELECTORES = ("p_inter", "binario")


# ============================================================
# Dataclass de una corrida
# ============================================================

@dataclass(frozen=True)
class Tarea:
    """Una corrida de la config fija (sin semilla fija)."""
    mh: str
    instancia: str
    repeticion: int
    selector: str       # "p_inter" | "binario"
    libre_nombre: str   # nombre del parametro libre (de CONFIG_FIJA[mh])
    libre_valor: float  # valor del parametro libre (de CONFIG_FIJA[mh])
    root: str | None
    ruta_csv_parcial: str


# ============================================================
# Construcción de kwargs por MH
# ============================================================

def construir_kwargs(tarea: Tarea) -> dict:
    """Arma los kwargs del runner para esta corrida (PR activo + selector).

    El PR se activa pasando el ``intensificador`` (hook limpio) y un umbral de
    estancamiento (``max_iter_sin_mejora_kick=UMBRAL_PR_DEF``): en el
    estancamiento la MH ejecuta PR hacia la mejor solución global en lugar del
    kick aleatorio. El selector se configura según ``tarea.selector`` (el binario
    se instala en el worker via ``_instalar_selector``, ver ``ejecutar_una``).
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
            "lambda":      "default_instance_aware",
        },
    )

    # Selector probabilístico nativo: pasar el p_inter canonico (SA/TS/RTS
    # aceptan alpha_inter; ABC/Cuckoo lo ignoran internamente).
    if tarea.selector == "p_inter":
        base["p_inter"] = P_INTER_FIJO[tarea.mh]
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
# Orquestación de una MH (config fija, ambos selectores)
# ============================================================

def ejecutar_experimento_mh(
    mh: str,
    *,
    salida_base: str,
    instancias: list[str],
    reps: int,
    workers: int,
    root: str | None,
    selectores: tuple[str, ...],
) -> Path:
    """Crea la carpeta con timestamp, corre la config fija × selectores y consolida.

    No hay grid ni calibracion: se usa directamente CONFIG_FIJA[mh] y
    P_INTER_FIJO[mh]. Se generan tareas para cada selector en ``selectores``
    (por defecto ambos). La trazabilidad queda en ``config_fija.json``.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    carpeta = Path(salida_base).expanduser().resolve() / f"{mh}_pr_aislado_{ts}"
    carpeta.mkdir(parents=True, exist_ok=True)
    print(f"\n### Experimento pr_aislado | MH={mh} | salida={carpeta}\n")

    libre_nombre, libre_valor = CONFIG_FIJA[mh]

    dir_final = carpeta / "final"
    dir_partials = dir_final / "_partials"
    dir_partials.mkdir(parents=True, exist_ok=True)

    # Generamos tareas: selectores × instancias × repeticiones (knob libre fijo).
    tareas: list[Tarea] = []
    for selector in selectores:
        for instancia in instancias:
            for rep in range(1, reps + 1):
                idx = len(tareas)
                parcial = dir_partials / (
                    f"{mh}_{selector}_{instancia}_{os.getpid()}_{idx}.csv"
                )
                tareas.append(Tarea(
                    mh=mh, instancia=instancia, repeticion=rep,
                    selector=selector, libre_nombre=libre_nombre,
                    libre_valor=libre_valor,
                    root=root, ruta_csv_parcial=str(parcial),
                ))

    print("=" * 80)
    print(f"CONFIG FIJA  MH={mh}  approach=pr_aislado")
    print(f"  selectores={list(selectores)}  {libre_nombre}={libre_valor}")
    print(f"  umbral_pr={UMBRAL_PR_DEF}  intensificador=path_relinking_limpio")
    print(f"  instancias={len(instancias)}  reps={reps}  corridas={len(tareas)}")
    print(f"  workers={workers}")
    print("=" * 80)

    _correr_tareas(tareas, workers)

    # Consolidar por (selector, instancia) -> final/<mh>_pr_<selector>_<inst>.csv
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

    # Escribimos la config usada para trazabilidad y reproducibilidad.
    config_fija_doc = {
        "mh": mh,
        "approach": "pr_aislado",
        "selectores": list(selectores),
        "p_inter_fijo": {sel: P_INTER_FIJO[mh] for sel in selectores if sel == "p_inter"},
        "libre_nombre": libre_nombre,
        "libre_valor": libre_valor,
        "umbral_pr": UMBRAL_PR_DEF,
        "intensificador": "path_relinking_limpio",
        "instancias": instancias,
        "reps": reps,
    }
    (carpeta / "config_fija.json").write_text(
        json.dumps(config_fija_doc, indent=2, ensure_ascii=False), encoding="utf-8",
    )

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
        description="Approach pr_aislado: Path Relinking limpio con config canonica fija."
    )
    if mh_fijo is None:
        parser.add_argument("--mh", type=str, required=True, choices=list(MH_MODULOS.keys()))
    parser.add_argument("--salida-base", type=str, default="experimentos_costo_fixed")
    parser.add_argument("--selector", type=str, default="ambos",
                        choices=["p_inter", "binario", "ambos"])
    parser.add_argument("--reps", type=int, default=REPS_DEF)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--instancias", type=str, default=None, nargs="*")
    parser.add_argument("--smoke", action="store_true",
                        help="2 instancias + reps=1 (prueba rápida).")
    args = parser.parse_args(argv)

    mh = mh_fijo if mh_fijo is not None else args.mh
    instancias = _parse_instancias(args.instancias)
    selectores = SELECTORES if args.selector == "ambos" else (args.selector,)
    reps = args.reps

    if args.smoke:
        if not args.instancias:
            instancias = ["gdb19", "kshs1"]
        reps = 1

    ejecutar_experimento_mh(
        mh,
        salida_base=args.salida_base,
        instancias=instancias,
        reps=reps,
        workers=args.workers,
        root=args.root,
        selectores=selectores,
    )


if __name__ == "__main__":
    main()
