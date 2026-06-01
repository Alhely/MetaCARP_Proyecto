"""
Approach ``solo_p_inter_20260531`` — el approach MAS SIMPLE bajo el costo corregido.

Motivacion
==========
El evaluador de costo del proyecto estaba mal (forzaba la orientacion canonica
``u->v`` de cada tarea), lo que inflaba artificialmente los gaps y hacia
imposible aislar el efecto real de cada cambio (R1..R5). Ya se corrigio: la
orientacion GREEDY es ahora la unica logica nativa de ``metacarp.evaluador_costo``
y el reporte de deadheading es consistente con ``mejor_costo``.

Sobre esa base limpia vamos de lo MAS SIMPLE a lo mas complejo. Este, el primer
approach, usa UNICAMENTE el selector probabilistico ``p_inter``:

  1. Con probabilidad ``p_inter`` se propone el grupo INTER-ruta; si no, el
     grupo INTRA-ruta. En estado infactible la probabilidad sube a
     ``alpha_inter`` (reparacion agresiva).
  2. DENTRO del grupo elegido, ``generar_vecino`` selecciona un operador de
     forma UNIFORME (``pesos_operadores=None``).

NO hay Path Relinking, NI kick reactivo, NI AOS, NI presupuesto x25. Es la
metaheuristica base corriendo con kwargs especificos. Por eso este modulo NO
define ningun monkey-patch: el selector base
``metacarp.metaheuristicas_utils.seleccionar_grupo_operadores_inter_intra`` ya
implementa exactamente esta mecanica, compartida por las 5 MH (SA, TS, RTS,
ABC, Cuckoo).

Objetivo metodologico: si este approach simple ya da un buen gap con el costo
corregido, los demas (PR, kick, AOS, budget) pasan a justificarse solo por
reducir el tiempo de ejecucion, no por mejorar la calidad de la solucion.

Diseno del experimento
======================
- Operadores: conjunto COMPLETO ``OPERADORES_POPULARES`` (9 = 3 intra + 6 inter).
- lambda (penalizador de capacidad): default del modulo
  (``lambda_penal_capacidad_por_defecto`` ~ 10x la mediana del costo de arco,
  instance-aware). Se pasa ``lambda_capacidad=None`` -> sin patch.
- Parametros INSTANCE-AWARE: se conservan automaticos pasando ``None``
  (T_init/T_min/L en SA; iteraciones/num_nidos/limites en el resto).
- Repeticiones: 5 por instancia en la fase final.

Grid search en DOS FASES
------------------------
  Fase 1 (calibracion, pocas reps): se barre la malla
        p_inter  x  un parametro libre por MH
  y se elige, por MH, la config con MENOR gap medio sobre las 23 instancias.
  Fase 2 (final, 5 reps): se corre solo la mejor config x 23 instancias.

  Malla:
    p_inter (comun)         : {0.1, 0.2, 0.3, 0.4, 0.5}
    SA            -> alpha          : {0.90, 0.95, 0.99}
    TS simple     -> tabu_tenure    : {7, 15, 25}
    RTS           -> factor_aumento : {1.1, 1.2, 1.3}
    ABC           -> num_fuentes    : {10, 20, 30}
    Cuckoo        -> pa_abandono    : {0.15, 0.25, 0.35}

  -> 5 x 3 = 15 configs por MH.

Salida
------
  experimentos_costo_fixed/<mh>_solo_p_inter_<YYYYMMDD-HHMM>/
    grid_parciales/        CSVs parciales de la fase 1 (uno por corrida)
    calibracion_todas.csv  consolidado de la fase 1 (columnas completas)
    grid_resumen.csv       gap medio por config (criterio de seleccion)
    mejor_config.json      config ganadora + su gap medio
    final/                 CSVs de la fase 2 (5 reps, columnas completas)
      _partials/           parciales antes de consolidar

NOTA sobre ``extra_csv``: solo Cuckoo lo escribe al CSV; las otras 4 MH lo
aceptan en su firma pero lo descartan. Por eso NO dependemos de ``extra_csv``
para identificar la config: cada MH ya graba sus parametros de forma nativa
(``p_inter`` + el knob libre) y nosotros guardamos la config en el dataclass
``Tarea``. La fase queda codificada por la carpeta (grid_parciales vs final).
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

# Aseguramos que la raiz del proyecto este en sys.path para poder importar
# ``metacarp`` cuando el script se ejecuta directamente desde ``scripts/``.
import sys as _sys
_ROOT_PROYECTO = Path(__file__).resolve().parent.parent
if str(_ROOT_PROYECTO) not in _sys.path:
    _sys.path.insert(0, str(_ROOT_PROYECTO))

from metacarp.vecindarios import OPERADORES_POPULARES


# ============================================================
# Constantes del experimento
# ============================================================

# Probabilidad de proponer INTER cuando la solucion VIOLA capacidad. Se fija en
# 0.80 (reparacion agresiva, default del selector). No se calibra: el approach
# calibra p_inter (estado factible), no alpha_inter (estado infactible).
ALPHA_INTER_FIJO: float = 0.80

# Repeticiones por defecto de cada fase.
REPS_CALIBRACION_DEF: int = 3
REPS_FINAL_DEF: int = 5

# 23 instancias pequenas del corpus actual (identicas a correct_cost_20260529).
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

# Knob compartido: probabilidad de proponer INTER en estado factible.
GRID_P_INTER: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)

# Un parametro libre por MH: (nombre_kwarg, valores). El resto de parametros se
# dejan instance-aware (None) o en su default de literatura clasica.
GRID_LIBRE: dict[str, tuple[str, tuple[float, ...]]] = {
    "sa":            ("alpha",          (0.90, 0.95, 0.99)),
    "tabu_simple":   ("tabu_tenure",    (7, 15, 25)),
    "tabu_reactiva": ("factor_aumento", (1.1, 1.2, 1.3)),
    "abc_simple":    ("num_fuentes",    (10, 20, 30)),
    "cuckoo":        ("pa_abandono",    (0.15, 0.25, 0.35)),
}


# ============================================================
# Dataclass de una corrida
# ============================================================

@dataclass(frozen=True)
class Tarea:
    """Una corrida del grid (sin semilla fija: la variabilidad es estocastica)."""
    mh: str             # clave de MH_MODULOS
    instancia: str
    repeticion: int
    p_inter: float      # knob compartido
    libre_nombre: str   # nombre del parametro libre de esta MH
    libre_valor: float  # valor del parametro libre en esta corrida
    fase: str           # "calibracion" | "final"
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
      - ``p_inter`` = knob calibrado; kick DESACTIVADO (sin variantes).
      - ``guardar_csv=True`` para que cada corrida escriba su parcial con TODAS
        las columnas (operadores, gap, detalle de deadheading).

    Por MH se fija el parametro libre del grid y, en SA/TS/RTS, ``alpha_inter``
    (ABC y Cuckoo no exponen ese kwarg: internamente usan max(p_inter, 0.8)).
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
        p_inter=tarea.p_inter,
        max_iter_sin_mejora_kick=None,    # kick reactivo desactivado
        max_resets=None,
        # extra_csv: solo Cuckoo lo escribe; lo incluimos por trazabilidad.
        extra_csv={
            "experimento": "solo_p_inter_20260531",
            "approach":    "solo_p_inter",
            "fase":        tarea.fase,
            "lambda":      "default_instance_aware",
        },
    )
    mh = tarea.mh
    if mh == "sa":
        base.update(
            alpha=float(tarea.libre_valor),
            alpha_inter=ALPHA_INTER_FIJO,
            temperatura_inicial=None,     # instance-aware
            temperatura_minima=None,      # instance-aware
            # Reheat ACOTADO (igual que la config SA validada del proyecto:
            # patience=10, reheat_factor=0.5, max_reheats_sin_mejora=5). Es
            # IMPRESCINDIBLE: con el reheat por defecto (max_reheats_sin_mejora=0
            # = sin tope) y alpha alto (0.99, enfriamiento lento) la corrida se
            # dispara y no termina en tiempo razonable. Con el tope, cada corrida
            # converge en ~5 s independientemente de alpha.
            patience=10,
            reheat_factor=0.5,
            max_reheats_sin_mejora=5,
        )
    elif mh == "tabu_simple":
        base.update(
            tabu_tenure=int(tarea.libre_valor),
            tam_vecindario=25,
            alpha_inter=ALPHA_INTER_FIJO,
        )
    elif mh == "tabu_reactiva":
        base.update(
            factor_aumento=float(tarea.libre_valor),
            factor_reduccion=0.9,
            alpha_inter=ALPHA_INTER_FIJO,
        )
    elif mh == "abc_simple":
        base.update(
            num_fuentes=int(tarea.libre_valor),
        )
    elif mh == "cuckoo":
        base.update(
            pa_abandono=float(tarea.libre_valor),
            beta_levy=1.5,
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
    """Ejecuta una corrida y devuelve (tarea, estado, info, error)."""
    try:
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


# ============================================================
# Ejecucion paralela de una lista de tareas
# ============================================================

def _correr_tareas(
    tareas: list[Tarea],
    workers: int,
) -> list[tuple[Tarea, dict]]:
    """Corre las tareas (paralelo o secuencial) y devuelve [(tarea, info_ok)]."""
    resultados: list[tuple[Tarea, dict]] = []
    ok = fail = 0
    if workers <= 1:
        iterador = (ejecutar_una(t) for t in tareas)
        for tarea, estado, info, err in iterador:
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
              f"{tarea.libre_valor} p_inter={tarea.p_inter} rep={tarea.repeticion} "
              f"| costo={info['costo']:.4f} | gap={gap_str} | t={info['tiempo']:.2f}s")
        resultados.append((tarea, info))
        return ok + 1, fail
    print(f"  [{tarea.mh}/{tarea.instancia}] p_inter={tarea.p_inter} "
          f"{tarea.libre_nombre}={tarea.libre_valor} rep={tarea.repeticion} | FAIL: {err}")
    return ok, fail + 1


# ============================================================
# Fase 1: calibracion (grid search)
# ============================================================

def fase1_calibracion(
    mh: str,
    carpeta: Path,
    *,
    instancias: list[str],
    reps: int,
    workers: int,
    root: str | None,
    grid_p_inter: tuple[float, ...],
    grid_libre: tuple[float, ...],
) -> dict:
    """Barre la malla (p_inter x parametro_libre) y elige la mejor config.

    Escribe grid_parciales/, calibracion_todas.csv y grid_resumen.csv. Devuelve
    un dict con la mejor config: {p_inter, libre_nombre, libre_valor, gap_medio}.
    """
    libre_nombre = GRID_LIBRE[mh][0]
    dir_parciales = carpeta / "grid_parciales"
    dir_parciales.mkdir(parents=True, exist_ok=True)

    # Generamos la lista de tareas: cada (config x instancia x rep).
    tareas: list[Tarea] = []
    for p_inter in grid_p_inter:
        for libre_valor in grid_libre:
            for instancia in instancias:
                for rep in range(1, reps + 1):
                    idx = len(tareas)
                    parcial = dir_parciales / (
                        f"{mh}_{instancia}_{os.getpid()}_{idx}.csv"
                    )
                    tareas.append(Tarea(
                        mh=mh, instancia=instancia, repeticion=rep,
                        p_inter=p_inter, libre_nombre=libre_nombre,
                        libre_valor=libre_valor, fase="calibracion",
                        root=root, ruta_csv_parcial=str(parcial),
                    ))

    print("=" * 80)
    print(f"FASE 1 — CALIBRACION  MH={mh}  approach=solo_p_inter")
    print(f"  configs={len(grid_p_inter) * len(grid_libre)} "
          f"(p_inter x {libre_nombre})  instancias={len(instancias)}  reps={reps}")
    print(f"  corridas totales={len(tareas)}  workers={workers}")
    print("=" * 80)

    resultados = _correr_tareas(tareas, workers)

    # --- Consolidamos los parciales de la fase 1 (columnas completas) ---
    _consolidar(
        sorted(dir_parciales.glob(f"{mh}_*.csv")),
        carpeta / "calibracion_todas.csv",
    )

    # --- Agregamos el gap medio por config ---
    # clave: (p_inter, libre_valor) -> lista de gaps validos.
    acum: dict[tuple[float, float], list[float]] = {}
    for tarea, info in resultados:
        gap = info.get("gap", math.nan)
        if gap is None or math.isnan(gap):
            continue
        acum.setdefault((tarea.p_inter, tarea.libre_valor), []).append(gap)

    filas_resumen: list[dict] = []
    for (p_inter, libre_valor), gaps in acum.items():
        media = sum(gaps) / len(gaps)
        var = sum((g - media) ** 2 for g in gaps) / len(gaps)
        filas_resumen.append({
            "mh": mh,
            "p_inter": p_inter,
            libre_nombre: libre_valor,
            "gap_medio": round(media, 4),
            "gap_std": round(math.sqrt(var), 4),
            "n_corridas": len(gaps),
        })
    # Orden: menor gap medio primero (mejor config arriba).
    filas_resumen.sort(key=lambda r: (r["gap_medio"], r["gap_std"], r["p_inter"]))

    ruta_resumen = carpeta / "grid_resumen.csv"
    with ruta_resumen.open("w", encoding="utf-8", newline="") as f:
        campos = ["mh", "p_inter", libre_nombre, "gap_medio", "gap_std", "n_corridas"]
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas_resumen)

    if not filas_resumen:
        raise RuntimeError(
            f"FASE 1 sin gaps validos para MH={mh}: no se puede elegir config."
        )

    mejor = filas_resumen[0]
    mejor_config = {
        "mh": mh,
        "p_inter": mejor["p_inter"],
        "libre_nombre": libre_nombre,
        "libre_valor": mejor[libre_nombre],
        "gap_medio": mejor["gap_medio"],
        "gap_std": mejor["gap_std"],
    }
    print(f"  -> MEJOR CONFIG {mh}: p_inter={mejor_config['p_inter']} "
          f"{libre_nombre}={mejor_config['libre_valor']} "
          f"| gap_medio={mejor_config['gap_medio']}%")
    return mejor_config


# ============================================================
# Fase 2: corrida final con la mejor config
# ============================================================

def fase2_final(
    mh: str,
    carpeta: Path,
    mejor_config: dict,
    *,
    instancias: list[str],
    reps: int,
    workers: int,
    root: str | None,
) -> None:
    """Corre la mejor config con ``reps`` repeticiones y consolida en final/."""
    dir_final = carpeta / "final"
    dir_partials = dir_final / "_partials"
    dir_partials.mkdir(parents=True, exist_ok=True)

    libre_nombre = mejor_config["libre_nombre"]
    libre_valor = mejor_config["libre_valor"]
    p_inter = mejor_config["p_inter"]

    tareas: list[Tarea] = []
    for instancia in instancias:
        for rep in range(1, reps + 1):
            idx = len(tareas)
            parcial = dir_partials / f"{mh}_{instancia}_{os.getpid()}_{idx}.csv"
            tareas.append(Tarea(
                mh=mh, instancia=instancia, repeticion=rep,
                p_inter=p_inter, libre_nombre=libre_nombre,
                libre_valor=libre_valor, fase="final",
                root=root, ruta_csv_parcial=str(parcial),
            ))

    print("=" * 80)
    print(f"FASE 2 — FINAL  MH={mh}  config: p_inter={p_inter} "
          f"{libre_nombre}={libre_valor}  instancias={len(instancias)}  reps={reps}")
    print(f"  corridas totales={len(tareas)}  workers={workers}")
    print("=" * 80)

    _correr_tareas(tareas, workers)

    # Consolidamos por instancia (un CSV final por instancia, columnas completas).
    # El stem tiene el patron ``{mh}_{instancia}_{pid}_{idx}``. Derivamos la
    # instancia quitando el prefijo conocido ``{mh}_`` y luego descartando los
    # dos ultimos tokens (pid e idx), separados por ``_``. Esto es robusto aunque
    # el nombre de MH contenga guiones bajos (p.ej. "tabu_simple", "abc_simple").
    prefix = f"{mh}_"
    grupos: dict[str, list[Path]] = {}
    for parcial in sorted(dir_partials.glob(f"{mh}_*.csv")):
        stem = parcial.stem
        if not stem.startswith(prefix):
            continue
        # Tras quitar el prefijo queda ``{instancia}_{pid}_{idx}``;
        # eliminamos los dos ultimos campos (_pid y _idx).
        sin_prefix = stem[len(prefix):]
        partes = sin_prefix.split("_")
        if len(partes) < 3:
            # Patron inesperado; lo omitimos para no producir agrupaciones erroneas.
            continue
        instancia = "_".join(partes[:-2])
        grupos.setdefault(instancia, []).append(parcial)
    n = 0
    for instancia, archivos in grupos.items():
        ruta = dir_final / f"{mh}_solo_p_inter_{instancia}.csv"
        n += 1 if _consolidar(archivos, ruta) else 0
    print(f"  -> {n} CSV finales por instancia en {dir_final}")


# ============================================================
# Orquestacion de una MH (las dos fases)
# ============================================================

def ejecutar_experimento_mh(
    mh: str,
    *,
    salida_base: str,
    instancias: list[str],
    reps_calibracion: int,
    reps_final: int,
    workers: int,
    root: str | None,
    grid_p_inter: tuple[float, ...],
    grid_libre: tuple[float, ...],
    solo_fase: str,
) -> Path:
    """Crea la carpeta con timestamp y corre las fases pedidas para una MH."""
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    carpeta = Path(salida_base).expanduser().resolve() / f"{mh}_solo_p_inter_{ts}"
    carpeta.mkdir(parents=True, exist_ok=True)
    print(f"\n### Experimento solo_p_inter | MH={mh} | salida={carpeta}\n")

    mejor_config: dict | None = None
    if solo_fase in ("ambas", "1"):
        mejor_config = fase1_calibracion(
            mh, carpeta, instancias=instancias, reps=reps_calibracion,
            workers=workers, root=root, grid_p_inter=grid_p_inter,
            grid_libre=grid_libre,
        )
        (carpeta / "mejor_config.json").write_text(
            json.dumps(mejor_config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if solo_fase in ("ambas", "2"):
        if mejor_config is None:
            # Permite re-correr solo la fase 2 leyendo una mejor_config previa.
            ruta_cfg = carpeta / "mejor_config.json"
            if not ruta_cfg.exists():
                raise RuntimeError(
                    "Fase 2 sin mejor_config.json; corre antes la fase 1."
                )
            mejor_config = json.loads(ruta_cfg.read_text(encoding="utf-8"))
        fase2_final(
            mh, carpeta, mejor_config, instancias=instancias,
            reps=reps_final, workers=workers, root=root,
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
    # Respeta el orden de INSTANCIAS y agrega extras al final.
    efectivas = [i for i in INSTANCIAS if i in bruto]
    for nombre in bruto:
        if nombre not in efectivas:
            efectivas.append(nombre)
    return efectivas


def main(argv: list[str] | None = None, *, mh_fijo: str | None = None) -> None:
    """Punto de entrada. ``mh_fijo`` lo usan los runners por-MH."""
    parser = argparse.ArgumentParser(
        description="Approach solo_p_inter con grid search en dos fases."
    )
    if mh_fijo is None:
        parser.add_argument("--mh", type=str, required=True,
                            choices=list(MH_MODULOS.keys()))
    parser.add_argument("--salida-base", type=str, default="experimentos_costo_fixed")
    parser.add_argument("--reps-calibracion", type=int, default=REPS_CALIBRACION_DEF)
    parser.add_argument("--reps-final", type=int, default=REPS_FINAL_DEF)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--instancias", type=str, default=None, nargs="*")
    parser.add_argument("--solo-fase", type=str, default="ambas",
                        choices=["ambas", "1", "2"])
    parser.add_argument("--smoke", action="store_true",
                        help="Malla minima + 1 rep + 2 instancias (prueba rapida).")
    args = parser.parse_args(argv)

    mh = mh_fijo if mh_fijo is not None else args.mh
    instancias = _parse_instancias(args.instancias)
    grid_p_inter = GRID_P_INTER
    grid_libre = GRID_LIBRE[mh][1]
    reps_cal = args.reps_calibracion
    reps_fin = args.reps_final

    if args.smoke:
        # Prueba rapida: 2 instancias, malla 2x2, 1 rep por fase.
        if not args.instancias:
            instancias = ["gdb19", "kshs1"]
        grid_p_inter = grid_p_inter[:2]
        grid_libre = grid_libre[:2]
        reps_cal = 1
        reps_fin = 1

    ejecutar_experimento_mh(
        mh,
        salida_base=args.salida_base,
        instancias=instancias,
        reps_calibracion=reps_cal,
        reps_final=reps_fin,
        workers=args.workers,
        root=args.root,
        grid_p_inter=grid_p_inter,
        grid_libre=grid_libre,
        solo_fase=args.solo_fase,
    )


if __name__ == "__main__":
    main()
