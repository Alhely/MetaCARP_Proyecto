"""
Approach ``binario_capacidad_20260531`` — segundo approach (selector binario).

Es el SEGUNDO approach del programa con el evaluador de costo corregido. Misma
logica que ``solo_p_inter_20260531`` (metaheuristica base, operadores completos,
lambda default instance-aware, sin PR/kick/AOS/budget) salvo por el SELECTOR:

  En lugar del selector PROBABILISTICO ``p_inter`` se usa un selector BINARIO
  DETERMINISTA basado en la capacidad:
    - Si la solucion VIOLA capacidad (violacion > 0): proponer el grupo
      INTER-ruta (reparacion).
    - Si la solucion es FACTIBLE (violacion <= 0): proponer el grupo
      INTRA-ruta (refinamiento).
  DENTRO del grupo elegido, ``generar_vecino`` selecciona un operador de forma
  UNIFORME (``pesos_operadores=None``), igual que en el primer approach.

Implementacion: el selector binario YA existe como
``metacarp.strict_intra_inter_20260524.seleccionar_grupo_strict`` y se instala
en el modulo de cada MH con ``aplicar_patch_selector`` (un monkey-patch que
reasigna ``seleccionar_grupo_operadores_inter_intra``). Es DETERMINISTA: no
consume el ``rng`` ni mira ``p_inter``/``alpha_inter`` (los absorbe e ignora).

A diferencia del experimento original ``strict_intra_inter_20260524``, aqui NO
se activa el kick reactivo (``max_iter_sin_mejora_kick=None``): queremos aislar
el efecto del SELECTOR binario puro, sin la capa de diversificacion. Tampoco se
usan los operadores reducidos (5): se mantiene el conjunto COMPLETO de 9 para
ser comparable con el primer approach.

Diferencia metodologica clave respecto al approach 1
====================================================
El selector binario es DETERMINISTA: no tiene el parametro ``p_inter``. Por eso
NO se pasa ``p_inter`` en este approach; solo se usa el knob libre canonico de
cada MH (CONFIG_FIJA).

Config canonica (sin grid, sin calibracion)
-------------------------------------------
  El approach corre UNA SOLA configuracion fija por MH (el selector es el mismo
  para todas: binario):
    Knob libre fijo: sa->alpha=0.90, tabu_simple->tabu_tenure=25,
                     tabu_reactiva->factor_aumento=1.2,
                     abc_simple->num_fuentes=30, cuckoo->pa_abandono=0.15
  Estos valores son la config canonica del proyecto. Se registran en
  ``config_fija.json`` para trazabilidad.

Salida
------
  experimentos_costo_fixed/<mh>_binario_capacidad_<YYYYMMDD-HHMM>/
    final/                 CSVs de la corrida (reps repeticiones, columnas completas)
      _partials/           parciales antes de consolidar
    config_fija.json       config usada (para trazabilidad y reproducibilidad)

Las columnas del CSV son las que escribe el nucleo (sin cambios): gap vs BKS,
mejor_costo, contadores de operadores (4 categorias x 9 operadores = 36
columnas), detalle de la solucion por ruta con nodos de deadheading y
costo_total_desde_reporte (consistente con mejor_costo bajo el evaluador greedy).
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

# IMPORTANTE: limitar a 1 los hilos de BLAS/OpenMP ANTES de importar numpy
# (lo hace ``metacarp`` al importarse). Cada corrida se ejecuta en su propio
# proceso (ProcessPoolExecutor con tantos workers como nucleos); si ademas cada
# worker dejara que MKL/OpenBLAS lance multiples hilos, habria
# sobre-suscripcion (nucleos^2 hilos) que degrada el rendimiento ~10x. Con un
# hilo por worker el paralelismo real lo da el pool de procesos. ``setdefault``
# permite override explicito desde el shell si se quisiera otra cosa.
for _var_hilos in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var_hilos, "1")

# Aseguramos que la raiz del proyecto y el directorio ``scripts/`` esten en
# sys.path. La raiz es necesaria para importar ``metacarp``; ``scripts/`` es
# necesaria para que los workers (procesos hijos de ProcessPoolExecutor)
# puedan importar ``_strict_intra_inter_20260524_common`` desde ``ejecutar_una``.
# Los workers heredan sys.path del padre: si el padre fue lanzado directamente
# (``python scripts/_binario_..._common.py``) solo tendra la raiz del proyecto
# si no anadimos ``scripts/`` aqui. Los runners ``run_*.py`` ya insertan
# ``scripts/`` antes de importar este modulo, pero la insercion directa es
# necesaria para el caso ``--smoke`` y cualquier invocacion directa del common.
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

# Repeticiones por defecto de la corrida final.
REPS_DEF: int = 5

# 23 instancias pequenas del corpus actual (identicas a solo_p_inter_20260531).
INSTANCIAS = [
    "gdb19", "kshs1", "kshs2", "kshs3", "kshs4", "kshs5", "kshs6",
    "gdb4",  "gdb14", "gdb15", "gdb1",  "gdb20", "gdb3",  "gdb6",
    "gdb7",  "gdb12", "gdb10", "gdb2",  "gdb5",  "gdb13", "gdb16",
    "gdb17", "gdb21",
]

# Modulos MH soportados (clave corta -> modulo).
MH_MODULOS: dict[str, str] = {
    "sa":             "metacarp.recocido_simulado",
    "tabu_simple":    "metacarp.busqueda_tabu_simple",
    "tabu_reactiva":  "metacarp.busqueda_tabu_reactiva",
    "abc_simple":     "metacarp.abejas_simple",
    "cuckoo":         "metacarp.cuckoo_search",
}

# Config canonica: knob libre fijo por MH (nombre_kwarg, valor).
# Valores derivados del analisis previo; se usan directamente sin grid.
# NO hay dimension p_inter: el selector binario es determinista.
CONFIG_FIJA: dict[str, tuple[str, float]] = {
    "sa":            ("alpha",          0.90),
    "tabu_simple":   ("tabu_tenure",    25),
    "tabu_reactiva": ("factor_aumento", 1.2),
    "abc_simple":    ("num_fuentes",    30),
    "cuckoo":        ("pa_abandono",    0.15),
}


# ============================================================
# Dataclass de una corrida
# ============================================================

@dataclass(frozen=True)
class Tarea:
    """Una corrida de la config fija (sin semilla fija: la variabilidad es estocastica).

    A diferencia del approach 1 NO hay campo ``p_inter``: el selector binario
    es determinista y no depende de ningun knob probabilistico.
    """
    mh: str             # clave de MH_MODULOS
    instancia: str
    repeticion: int
    libre_nombre: str   # nombre del parametro libre (de CONFIG_FIJA[mh])
    libre_valor: float  # valor del parametro libre (de CONFIG_FIJA[mh])
    root: str | None
    ruta_csv_parcial: str


# ============================================================
# Construccion de kwargs por MH
# ============================================================

def construir_kwargs(tarea: Tarea) -> dict:
    """Arma los kwargs del runner ``*_desde_instancia`` para esta corrida.

    Comun a todas las MH:
      - ``operadores=OPERADORES_POPULARES`` (9 completos, seleccion uniforme).
      - ``lambda_capacidad=None`` -> usa el lambda default instance-aware.
      - kick DESACTIVADO (``max_iter_sin_mejora_kick=None``, ``max_resets=None``).
      - ``guardar_csv=True`` para que cada corrida escriba su parcial con TODAS
        las columnas (operadores, gap, detalle de deadheading).

    NO se pasa ``p_inter`` ni ``alpha_inter``: el selector binario instalado por
    el worker (``seleccionar_grupo_strict``) los ignora. Se deja que cada MH use
    sus defaults para esos kwargs irrelevantes.

    Por MH se fija unicamente el parametro libre de CONFIG_FIJA.
    """
    base = dict(
        root=tarea.root,
        operadores=OPERADORES_POPULARES,
        usar_gpu=False,
        semilla=None,
        repeticion=tarea.repeticion,
        guardar_csv=True,
        ruta_csv=tarea.ruta_csv_parcial,
        guardar_historial=False,
        lambda_capacidad=None,            # lambda default (instance-aware)
        max_iter_sin_mejora_kick=None,    # kick reactivo desactivado
        max_resets=None,
        # extra_csv: solo Cuckoo lo escribe; lo incluimos por trazabilidad.
        extra_csv={
            "experimento": "binario_capacidad_20260531",
            "approach":    "binario_capacidad",
            "selector":    "binario_estricto_capacidad",
            "lambda":      "default_instance_aware",
        },
    )
    mh = tarea.mh
    if mh == "sa":
        base.update(
            alpha=float(tarea.libre_valor),
            temperatura_inicial=None,     # instance-aware
            temperatura_minima=None,      # instance-aware
            # Reheat ACOTADO (igual que la config SA validada del proyecto:
            # patience=10, reheat_factor=0.5, max_reheats_sin_mejora=10). Es
            # IMPRESCINDIBLE: con el reheat por defecto (max_reheats_sin_mejora=0
            # = sin tope) y alpha alto (0.99, enfriamiento lento) la corrida se
            # dispara y no termina en tiempo razonable. Con el tope, cada corrida
            # converge en ~5 s independientemente de alpha.
            patience=10,
            reheat_factor=0.5,
            max_reheats_sin_mejora=10,
        )
    elif mh == "tabu_simple":
        base.update(
            tabu_tenure=int(tarea.libre_valor),
            tam_vecindario=40,
        )
    elif mh == "tabu_reactiva":
        base.update(
            factor_aumento=float(tarea.libre_valor),
            factor_reduccion=0.95,
        )
    elif mh == "abc_simple":
        base.update(
            num_fuentes=int(tarea.libre_valor),
            limite_abandono=60,
        )
    elif mh == "cuckoo":
        base.update(
            pa_abandono=float(tarea.libre_valor),
            beta_levy=1.3,
        )
    else:
        raise ValueError(f"MH desconocida: {mh!r}")
    return base


def _cargar_runner(mh: str):
    """Importa y devuelve la funcion ``*_desde_instancia`` de la MH."""
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
    """Lee ``gap_bks_porcentaje`` de un CSV parcial de una sola fila.

    Devuelve ``nan`` si la columna no existe o no es parseable (p.ej. instancia
    sin BKS conocido), para que el promedio la ignore.
    """
    try:
        with open(ruta_csv, "r", encoding="utf-8", newline="") as f:
            fila = next(csv.DictReader(f), None)
        if not fila:
            return math.nan
        valor = fila.get("gap_bks_porcentaje", "")
        return float(valor)
    except (OSError, StopIteration, TypeError, ValueError):
        return math.nan


# ============================================================
# Worker
# ============================================================

def ejecutar_una(tarea: Tarea) -> tuple[Tarea, str, dict | None, str | None]:
    """Ejecuta una corrida y devuelve (tarea, estado, info, error).

    Antes de correr instala el SELECTOR BINARIO en el modulo de la MH. Debe
    hacerse dentro del worker porque cada proceso hijo importa el modulo de la
    MH desde cero y necesita aplicar su propio monkey-patch.
    """
    try:
        # 1) Instalar el selector binario determinista (reemplaza el selector
        #    probabilistico ``seleccionar_grupo_operadores_inter_intra``).
        from _strict_intra_inter_20260524_common import aplicar_patch_selector
        aplicar_patch_selector(MH_MODULOS[tarea.mh])

        # 2) Ejecutar la MH base con el resto de la configuracion del approach.
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
# Consolidacion de parciales (union de columnas, patron del proyecto)
# ============================================================

def _consolidar(parciales: list[Path], ruta_final: Path) -> int:
    """Une varios CSV parciales en uno solo, tomando la UNION de columnas.

    Devuelve el numero de filas escritas. Si no hay parciales, no escribe nada.
    """
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
    """Extrae el nombre de instancia de un stem ``{mh}_{instancia}_{pid}_{idx}``.

    Robusto frente a claves de MH con guion bajo (p.ej. ``tabu_simple``): se
    quita el prefijo ``{mh}_`` y se descartan los dos ultimos tokens (pid, idx),
    reuniendo lo intermedio (la instancia, que podria tambien tener guion bajo).
    """
    prefix = f"{mh}_"
    if not stem.startswith(prefix):
        return None
    partes = stem[len(prefix):].split("_")
    if len(partes) < 3:
        return None
    return "_".join(partes[:-2])


# ============================================================
# Ejecucion paralela de una lista de tareas
# ============================================================

def _correr_tareas(tareas: list[Tarea], workers: int) -> list[tuple[Tarea, dict]]:
    """Corre las tareas (paralelo o secuencial) y devuelve [(tarea, info_ok)]."""
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
    """Imprime el estado de una corrida y acumula los OK. Helper de _correr_tareas."""
    if estado == "ok":
        gap = info.get("gap", math.nan)
        gap_str = "nan" if (gap is None or math.isnan(gap)) else f"{gap:.2f}%"
        print(f"  [{tarea.mh}/{tarea.instancia}] {tarea.libre_nombre}="
              f"{tarea.libre_valor} rep={tarea.repeticion} "
              f"| costo={info['costo']:.4f} | gap={gap_str} | t={info['tiempo']:.2f}s")
        resultados.append((tarea, info))
        return ok + 1, fail
    print(f"  [{tarea.mh}/{tarea.instancia}] {tarea.libre_nombre}={tarea.libre_valor} "
          f"rep={tarea.repeticion} | FAIL: {err}")
    return ok, fail + 1


# ============================================================
# Orquestacion de una MH (config fija, una sola corrida)
# ============================================================

def ejecutar_experimento_mh(
    mh: str,
    *,
    salida_base: str,
    instancias: list[str],
    reps: int,
    workers: int,
    root: str | None,
) -> Path:
    """Crea la carpeta con timestamp, corre la config fija y consolida en final/.

    No hay grid ni calibracion: se usa directamente CONFIG_FIJA[mh]. El selector
    binario se aplica como patch dentro de cada worker (``aplicar_patch_selector``).
    La trazabilidad queda en ``config_fija.json``.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    carpeta = Path(salida_base).expanduser().resolve() / f"{mh}_binario_capacidad_{ts}"
    carpeta.mkdir(parents=True, exist_ok=True)
    print(f"\n### Experimento binario_capacidad | MH={mh} | salida={carpeta}\n")

    libre_nombre, libre_valor = CONFIG_FIJA[mh]

    dir_final = carpeta / "final"
    dir_partials = dir_final / "_partials"
    dir_partials.mkdir(parents=True, exist_ok=True)

    # Generamos tareas: config fija x instancias x repeticiones.
    tareas: list[Tarea] = []
    for instancia in instancias:
        for rep in range(1, reps + 1):
            idx = len(tareas)
            parcial = dir_partials / f"{mh}_{instancia}_{os.getpid()}_{idx}.csv"
            tareas.append(Tarea(
                mh=mh, instancia=instancia, repeticion=rep,
                libre_nombre=libre_nombre, libre_valor=libre_valor,
                root=root, ruta_csv_parcial=str(parcial),
            ))

    print("=" * 80)
    print(f"CONFIG FIJA  MH={mh}  approach=binario_capacidad")
    print(f"  selector=binario_estricto_capacidad  {libre_nombre}={libre_valor}")
    print(f"  instancias={len(instancias)}  reps={reps}  corridas={len(tareas)}")
    print(f"  workers={workers}")
    print("=" * 80)

    _correr_tareas(tareas, workers)

    # Consolidamos por instancia (un CSV final por instancia, columnas completas).
    grupos: dict[str, list[Path]] = {}
    for parcial in sorted(dir_partials.glob(f"{mh}_*.csv")):
        instancia = _instancia_de_parcial(parcial.stem, mh)
        if instancia is None:
            continue
        grupos.setdefault(instancia, []).append(parcial)
    n = 0
    for instancia, archivos in grupos.items():
        ruta = dir_final / f"{mh}_binario_capacidad_{instancia}.csv"
        n += 1 if _consolidar(archivos, ruta) else 0
    print(f"  -> {n} CSV finales por instancia en {dir_final}")

    # Escribimos la config usada para trazabilidad y reproducibilidad.
    config_fija_doc = {
        "mh": mh,
        "approach": "binario_capacidad",
        "selector": "binario_estricto_capacidad",
        "libre_nombre": libre_nombre,
        "libre_valor": libre_valor,
        "instancias": instancias,
        "reps": reps,
    }
    (carpeta / "config_fija.json").write_text(
        json.dumps(config_fija_doc, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return carpeta


# ============================================================
# CLI
# ============================================================

def _parse_instancias(items: list[str] | None) -> list[str]:
    """Normaliza --instancias (admite separadas por coma o espacio)."""
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
    """Punto de entrada. ``mh_fijo`` lo usan los runners por-MH."""
    parser = argparse.ArgumentParser(
        description="Approach binario_capacidad (selector binario) con config canonica fija."
    )
    if mh_fijo is None:
        parser.add_argument("--mh", type=str, required=True,
                            choices=list(MH_MODULOS.keys()))
    parser.add_argument("--salida-base", type=str, default="experimentos_costo_fixed")
    parser.add_argument("--reps", type=int, default=REPS_DEF)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--instancias", type=str, default=None, nargs="*")
    parser.add_argument("--smoke", action="store_true",
                        help="2 instancias + reps=1 (prueba rapida).")
    args = parser.parse_args(argv)

    mh = mh_fijo if mh_fijo is not None else args.mh
    instancias = _parse_instancias(args.instancias)
    reps = args.reps

    if args.smoke:
        # Prueba rapida: 2 instancias, 1 rep.
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
    )


if __name__ == "__main__":
    main()
