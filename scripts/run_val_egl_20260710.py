"""
Campaña val + egl 20260710 — corrida ÚNICA por (MH, instancia).

Este runner cierra la brecha del corpus dejado fuera de la campaña de "costo
corregido" (23 instancias pequeñas gdb/kshs). Aquí atacamos las 64 instancias
restantes:

    * 34 instancias val*     (val1A..val10D)
    * 24 instancias egl-*    (egl-e{1..4}-{A,B,C}, egl-s{1..4}-{A,B,C})
    *  6 instancias gdb no cubiertas (gdb8, gdb9, gdb11, gdb18, gdb22, gdb23)

Agrupamos por CLASE de instancia para tener dos ecosistemas separados:

    - ``val-class``: 34 val + 6 gdb  (INSTANCIAS_VAL_GDB)
    - ``egl-class``: 24 egl-*        (INSTANCIAS_EGL)

Diseño (LEER PRIMERO ``run_sa_automatico.py`` y ``_pr_aislado_20260531_common.py``)
==============================================================================

Reglas heredadas de esos dos módulos:

  1. ProcessPoolExecutor + ``--workers``  (default os.cpu_count()).
  2. Semillas deterministas derivadas de
     ``(semilla_base, instancia, mh, repeticion)``.
  3. CSVs parciales por worker en ``final/_partials/``; el proceso principal
     los CONSOLIDA en un CSV final por (mh, instancia). Sin locks, sin race
     conditions, sin id_corrida/config_id.
  4. La config de cada MH se dumpea en ``config_fija.json`` (trazabilidad).
  5. Composición ``pr_aislado``: Path Relinking sobre la MH base, con
     ``max_iter_sin_mejora_kick=UMBRAL_PR`` e ``intensificador=hook_pr_*``.
     La captura de columnas PR en el CSV es la misma que _pr_aislado_20260531
     (`extra_csv` con selector, intensificador, umbral_pr, lambda_default).

Diferencias respecto a las campañas anteriores
==============================================

  * SIN GRID (una sola configuración por MH y por clase).
  * Los parámetros calibrados con el usuario dependen de la CLASE
    (val-class vs egl-class). Ver ``CONFIG_POR_MH``.
  * Presupuesto uniforme entre MH: ``max_evaluaciones ≈ 1e6`` OR
    ``tiempo_limite_segundos = 300 s`` (lo que ocurra primero).
    Detalle por MH:

        - SA: no admite ``iteraciones_max``. Se usa el cap por wall-clock
          exclusivamente (kwarg ``tiempo_limite_segundos``, añadido al
          módulo). El bucle externo de SA (niveles de temperatura) hace la
          comprobación una vez por nivel; suficientemente frecuente porque
          un nivel dura n² evaluaciones.
        - TS simple: admite ``iteraciones_max``. Con
          ``tam_vecindario = max(40, n_tareas)``, un cap de
          ``iteraciones_max = max(1, floor(1e6 / tam_vecindario))``
          aproxima 1e6 evaluaciones. Además, ``tiempo_limite_segundos=300``.
        - RTS: admite ``iteraciones_max`` (default ``20·n``). Aplicamos
          análogamente ``iteraciones_max = max(1, floor(1e6 / max(20, 2·n)))``,
          más ``tiempo_limite_segundos=300``.
        - ABC simple: admite ``iteraciones`` (default ``max(200, 20·n)``).
          Con ``num_fuentes = 30`` cada iteración cuesta ~ ``2 × 30 = 60``
          evaluaciones; ``iteraciones = max(200, floor(1e6 / 60)) ≈ 16666``.
          Más ``tiempo_limite_segundos=300``.

    Cuando el cap por tiempo dispara, se REPORTA en stdout junto al costo,
    para poder validar en el smoke que la corrida termina por presupuesto y
    NO por convergencia natural. Ver ``cap_disparado`` en la salida de tarea.

Salida
------
    <salida-dir>/
        sa/    ← final/  _partials/  config_fija.json
        ts/
        rts/
        abc/

CLI
---
    python scripts/run_val_egl_20260710.py --smoke
        Prueba tibia: 2 instancias {val1A, egl-e1-A}, 1 rep, presupuesto
        recortado (10 s wall-clock, iteraciones × 10x menos) para validar
        que TODO el pipeline funciona en <1 min.

    python scripts/run_val_egl_20260710.py --solo-clase val --solo-mh sa
    python scripts/run_val_egl_20260710.py --repeticiones 5 --workers 11
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# --- BLAS/OpenMP a 1 hilo por proceso (ver _pr_aislado_20260531_common.py) ---
# Ajustamos ANTES de importar numpy para evitar sobre-suscripción con
# ProcessPoolExecutor: cada worker aísla su BLAS a un solo hilo y así el
# paralelismo real coincide con el número de workers configurado.
for _var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

# Los workers deben ver el paquete metacarp instalado; el root del proyecto y
# el propio directorio scripts/ se añaden al path para poder importar módulos
# hermanos (mismo patrón que _pr_aislado_20260531_common.py).
_ROOT_PROYECTO = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_ROOT_PROYECTO) not in sys.path:
    sys.path.insert(0, str(_ROOT_PROYECTO))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from metacarp.vecindarios import OPERADORES_POPULARES  # noqa: E402
from metacarp.instances import load_instances  # noqa: E402


# ============================================================
# Corpus de instancias
# ============================================================

# 34 val + 6 gdb no cubiertos por la campaña de costo corregido.
# Los 6 gdb son los que quedaron fuera de la lista original de 23 pequeñas
# (gdb19, kshs1-6, gdb1-7, gdb10, gdb12-17, gdb20, gdb21).
INSTANCIAS_VAL_GDB: list[str] = [
    # val (34)
    "val1A", "val1B", "val1C",
    "val2A", "val2B", "val2C",
    "val3A", "val3B", "val3C",
    "val4A", "val4B", "val4C", "val4D",
    "val5A", "val5B", "val5C", "val5D",
    "val6A", "val6B", "val6C",
    "val7A", "val7B", "val7C",
    "val8A", "val8B", "val8C",
    "val9A", "val9B", "val9C", "val9D",
    "val10A", "val10B", "val10C", "val10D",
    # gdb no cubiertos (6)
    "gdb8", "gdb9", "gdb11", "gdb18", "gdb22", "gdb23",
]

# 24 egl-*.
INSTANCIAS_EGL: list[str] = [
    "egl-e1-A", "egl-e1-B", "egl-e1-C",
    "egl-e2-A", "egl-e2-B", "egl-e2-C",
    "egl-e3-A", "egl-e3-B", "egl-e3-C",
    "egl-e4-A", "egl-e4-B", "egl-e4-C",
    "egl-s1-A", "egl-s1-B", "egl-s1-C",
    "egl-s2-A", "egl-s2-B", "egl-s2-C",
    "egl-s3-A", "egl-s3-B", "egl-s3-C",
    "egl-s4-A", "egl-s4-B", "egl-s4-C",
]


# ============================================================
# Configuración por MH y por CLASE
# ============================================================

# Cada MH tiene UN par de configuraciones (una para cada clase). Los valores
# fueron acordados con el usuario para esta campaña. Todo lo demás
# (T0, L, lambda) queda en su default instance-aware.
#
# Presupuesto UNIFORME:
#   * ``max_evaluaciones ≈ 1e6``
#   * ``tiempo_limite_segundos = 300`` s wall-clock
#
# Ver docstring del módulo para el detalle por MH.
CONFIG_POR_MH: dict[str, dict[str, dict]] = {
    "sa": {
        "val": {"alpha": 0.90, "max_reheats_sin_mejora": 10, "p_inter": 0.5},
        "egl": {"alpha": 0.95, "max_reheats_sin_mejora": 10, "p_inter": 0.6},
    },
    "ts": {
        # ``tabu_tenure`` y ``tam_vecindario`` se resuelven por INSTANCIA
        # (dependen de n_tareas). ``p_inter`` sí queda fijo por clase.
        "val": {"p_inter": 0.4},
        "egl": {"p_inter": 0.6},
    },
    "rts": {
        "val": {"factor_aumento": 1.2, "factor_reduccion": 0.95, "p_inter": 0.5},
        "egl": {"factor_aumento": 1.2, "factor_reduccion": 0.95, "p_inter": 0.6},
    },
    "abc": {
        # ``limite_abandono`` se resuelve por INSTANCIA (max(60, n_tareas)).
        "val": {"num_fuentes": 30, "p_inter": 0.5},
        "egl": {"num_fuentes": 30, "p_inter": 0.6},
    },
}

# Umbral de PR (mismo que _pr_aislado_20260531_common.py). Fijo, no se calibra.
UMBRAL_PR_DEF: int = 30

# Cap uniforme de wall-clock (segundos). Aplica a las 4 MH.
TIEMPO_LIMITE_DEF: float = 300.0

# Cap uniforme de evaluaciones. Se traduce a ``iteraciones_max`` /
# ``iteraciones`` según la MH (SA no lo admite; usa sólo wall-clock).
MAX_EVALUACIONES_DEF: int = 1_000_000

# En modo --smoke recortamos drásticamente para que la validación complete
# en menos de un minuto por corrida.
TIEMPO_LIMITE_SMOKE: float = 10.0
MAX_EVALUACIONES_SMOKE: int = 100_000

# Correspondencia MH → módulo importable (para diagnóstico).
MH_MODULOS: dict[str, str] = {
    "sa":  "metacarp.recocido_simulado",
    "ts":  "metacarp.busqueda_tabu_simple",
    "rts": "metacarp.busqueda_tabu_reactiva",
    "abc": "metacarp.abejas_simple",
}


# ============================================================
# Dataclass de tarea
# ============================================================

@dataclass(frozen=True)
class Tarea:
    """Encapsula UNA corrida (una MH sobre una instancia, una repetición).

    Diseño idéntico al de ``_pr_aislado_20260531_common.py``: objeto plano y
    serializable que se envía al worker. Mantiene todos los argumentos en un
    solo lugar (``clase`` para elegir la config, ``semilla`` determinista).
    """
    mh: str
    instancia: str
    clase: str            # "val" | "egl"
    repeticion: int
    semilla: int
    root: str | None
    ruta_csv_parcial: str
    max_evaluaciones: int
    tiempo_limite_seg: float


# ============================================================
# Semillas deterministas
# ============================================================

def _derivar_semilla(base: int, instancia: str, mh: str, repeticion: int) -> int:
    """Semilla determinista de una corrida.

    Mismo esquema que ``run_sa_automatico._derivar_semilla`` pero adaptado a
    la dimensión (base, instancia, mh, repeticion). Los enteros se combinan
    con un factor primo y los caracteres del nombre se mezclan por XOR para
    que la salida dependa de TODAS las dimensiones. Cuantizamos los enteros
    directamente (no hay floats en la clave).
    """
    h = base
    for ch in instancia:
        h = (h * 1000003) ^ ord(ch)
    for ch in mh:
        h = (h * 1000003) ^ ord(ch)
    h = (h * 1000003) ^ int(repeticion)
    return h & 0x7FFFFFFF


# ============================================================
# Cálculo instance-aware de ``tam_vecindario`` / ``tabu_tenure`` / etc.
# ============================================================

def _n_tareas_instancia(nombre_instancia: str, root: str | None) -> int:
    """Devuelve el número de tareas requeridas (``|LISTA_ARISTAS_REQ|``)."""
    data = load_instances(nombre_instancia, root=root)
    return int(len(data["LISTA_ARISTAS_REQ"]))


def _cap_iters_por_lote(max_evaluaciones: int, lote: int) -> int:
    """Traduce un cap de evaluaciones a un cap de iteraciones para un tamaño
    de lote conocido. Cota inferior 1 para evitar iteraciones=0.
    """
    return max(1, max_evaluaciones // max(1, lote))


# ============================================================
# Composición de kwargs por MH (config fija + presupuesto)
# ============================================================

def _construir_kwargs(tarea: Tarea) -> dict:
    """Arma los kwargs del runner ``*_desde_instancia`` para esta corrida.

    Aplica:
      * La configuración de MH+clase de ``CONFIG_POR_MH``.
      * El cap por wall-clock (``tiempo_limite_segundos``, kwarg añadido a
        cada MH para esta campaña).
      * El cap por iteraciones/ciclos que aproxima 1e6 evaluaciones (excepto
        en SA que no admite iteraciones_max).
      * PR como intensificador (hook_pr_labels / hook_pr_ids), disparado en
        el estancamiento con ``max_iter_sin_mejora_kick=UMBRAL_PR_DEF``.
        Esto reproduce EXACTAMENTE la composición pr_aislado del approach
        _pr_aislado_20260531_common.
    """
    # Importación diferida del hook PR (misma estrategia que pr_aislado_common).
    from metacarp.path_relinking_limpio_20260531 import hook_pr_labels, hook_pr_ids
    hook = hook_pr_ids if tarea.mh == "abc" else hook_pr_labels

    cfg_mh = CONFIG_POR_MH[tarea.mh][tarea.clase]

    # Instance-aware: leemos n_tareas UNA vez por tarea (necesario para TS/RTS/ABC).
    n_tareas = _n_tareas_instancia(tarea.instancia, tarea.root)

    # extra_csv común a todas las MH: traza selector, intensificador, cap,
    # clase, umbral_pr. NO incluye id_corrida/config_id (convención del CSV).
    extra_csv = {
        "experimento":     "val_egl_20260710",
        "approach":        "pr_aislado",
        "selector":        "p_inter",
        "intensificador":  "path_relinking_limpio",
        "umbral_pr":       str(UMBRAL_PR_DEF),
        "lambda":          "default_instance_aware",
        "clase_instancia": tarea.clase,
        "cap_tiempo_seg":  f"{tarea.tiempo_limite_seg:.1f}",
        "cap_evaluaciones": str(tarea.max_evaluaciones),
        "n_tareas":        str(n_tareas),
    }

    base = dict(
        root=tarea.root,
        operadores=OPERADORES_POPULARES,
        usar_gpu=False,
        semilla=tarea.semilla,
        repeticion=tarea.repeticion,
        guardar_csv=True,
        ruta_csv=tarea.ruta_csv_parcial,
        guardar_historial=False,
        lambda_capacidad=None,             # default instance-aware
        # PR limpio como respuesta al estancamiento (no kick aleatorio).
        max_iter_sin_mejora_kick=UMBRAL_PR_DEF,
        max_resets=None,
        intensificador=hook,
        # Cap uniforme de wall-clock (kwarg añadido para esta campaña).
        tiempo_limite_segundos=float(tarea.tiempo_limite_seg),
        extra_csv=extra_csv,
    )

    # --- Parámetros específicos por MH ---
    if tarea.mh == "sa":
        # SA no admite iteraciones_max; se detiene por wall-clock o
        # max_reheats_sin_mejora. patience=10 y reheat_factor=0.5 heredados
        # de la campaña de calibración.
        base.update(
            alpha=float(cfg_mh["alpha"]),
            p_inter=float(cfg_mh["p_inter"]),
            alpha_inter=0.80,
            temperatura_inicial=None,
            temperatura_minima=None,
            patience=10,
            reheat_factor=0.5,
            max_reheats_sin_mejora=int(cfg_mh["max_reheats_sin_mejora"]),
        )
    elif tarea.mh == "ts":
        # tam_vecindario = max(40, n_tareas)  → un vecino por tarea al menos.
        # tabu_tenure    = max(25, round(n_tareas / 4))
        tam_vecindario = max(40, n_tareas)
        tabu_tenure = max(25, round(n_tareas / 4))
        iteraciones_max = _cap_iters_por_lote(tarea.max_evaluaciones, tam_vecindario)
        base.update(
            iteraciones_max=iteraciones_max,
            # Sin cap de estancamiento aparte del presupuesto: dejamos
            # un valor grande para que el cap real sea el wall-clock.
            max_iter_sin_mejora=iteraciones_max,
            tam_vecindario=tam_vecindario,
            tabu_tenure=tabu_tenure,
            p_inter=float(cfg_mh["p_inter"]),
            alpha_inter=0.80,
        )
    elif tarea.mh == "rts":
        # En RTS el tam_vecindario por defecto es max(20, 2·n).
        tam_vecindario = max(20, 2 * n_tareas)
        iteraciones_max = _cap_iters_por_lote(tarea.max_evaluaciones, tam_vecindario)
        base.update(
            iteraciones_max=iteraciones_max,
            max_iter_sin_mejora=iteraciones_max,
            tam_vecindario=tam_vecindario,
            factor_aumento=float(cfg_mh["factor_aumento"]),
            factor_reduccion=float(cfg_mh["factor_reduccion"]),
            p_inter=float(cfg_mh["p_inter"]),
            alpha_inter=0.80,
        )
    elif tarea.mh == "abc":
        # ABC no expone tam_vecindario directamente; cada iteración cuesta
        # aproximadamente 2·num_fuentes evaluaciones (empleadas + observadoras).
        num_fuentes = int(cfg_mh["num_fuentes"])
        limite_abandono = max(60, n_tareas)
        evals_por_ciclo = 2 * num_fuentes
        iteraciones = _cap_iters_por_lote(tarea.max_evaluaciones, evals_por_ciclo)
        base.update(
            iteraciones=iteraciones,
            num_fuentes=num_fuentes,
            limite_abandono=limite_abandono,
            max_iter_sin_mejora=iteraciones,
            p_inter=float(cfg_mh["p_inter"]),
        )
    else:
        raise ValueError(f"MH desconocida: {tarea.mh!r}")
    return base


# ============================================================
# Cargador dinámico del runner *_desde_instancia
# ============================================================

def _cargar_runner(mh: str):
    """Importa la función ``_desde_instancia`` correspondiente al MH."""
    if mh == "sa":
        from metacarp import recocido_simulado_desde_instancia as runner
    elif mh == "ts":
        from metacarp import busqueda_tabu_simple_desde_instancia as runner
    elif mh == "rts":
        from metacarp import busqueda_tabu_reactiva_desde_instancia as runner
    elif mh == "abc":
        from metacarp import busqueda_abejas_simple_desde_instancia as runner
    else:
        raise ValueError(f"MH desconocida: {mh!r}")
    return runner


# ============================================================
# Worker: ejecuta una corrida
# ============================================================

def _leer_gap_del_parcial(ruta_csv: str) -> float:
    """Extrae el gap % del CSV parcial (mismo helper que pr_aislado_common)."""
    try:
        with open(ruta_csv, "r", encoding="utf-8", newline="") as f:
            fila = next(csv.DictReader(f), None)
        if not fila:
            return math.nan
        return float(fila.get("gap_bks_porcentaje", ""))
    except (OSError, StopIteration, TypeError, ValueError):
        return math.nan


def _detectar_cap_disparado(
    tiempo_medido: float,
    tiempo_limite: float,
    iteraciones_totales: int | None,
    iteraciones_max_config: int | None,
) -> str:
    """Determina qué cap terminó la corrida (para diagnóstico y honestidad
    en la afirmación de "fairness" cross-MH).

    Devuelve una etiqueta corta:
      * "TIEMPO"   → la corrida se cortó por wall-clock.
      * "ITERS"    → la corrida se cortó por iteraciones_max (o equivalente).
      * "NATURAL"  → convergencia / estancamiento sin llegar a los caps.
    """
    # Umbral del 90 % del tiempo_limite: si estamos cerca del cap por tiempo
    # es casi seguro que ese fue el criterio de parada (una iteración adicional
    # habría superado el límite).
    if tiempo_limite > 0 and tiempo_medido >= 0.9 * tiempo_limite:
        return "TIEMPO"
    if (iteraciones_totales is not None
            and iteraciones_max_config is not None
            and iteraciones_totales >= iteraciones_max_config):
        return "ITERS"
    return "NATURAL"


def _ejecutar_tarea(tarea: Tarea) -> tuple[Tarea, str, dict | None, str | None]:
    """Ejecuta UNA corrida en un worker.

    Similar a ``ejecutar_una`` de _pr_aislado_common: carga el runner
    dinámicamente, construye los kwargs, ejecuta, extrae el gap del CSV
    parcial y devuelve la info al proceso principal.
    """
    try:
        runner = _cargar_runner(tarea.mh)
        kwargs = _construir_kwargs(tarea)
        res = runner(tarea.instancia, **kwargs)

        # Iteraciones reportadas (dependen de la MH). Usamos getattr para
        # tolerar diferencias entre dataclasses.
        iters_reportadas = getattr(res, "iteraciones_totales", None)
        if iters_reportadas is None:
            iters_reportadas = getattr(res, "iteraciones", None)

        info = {
            "costo": float(res.mejor_costo),
            "tiempo": float(res.tiempo_segundos),
            "gap": _leer_gap_del_parcial(tarea.ruta_csv_parcial),
            "factible": getattr(res, "mejor_solucion_factible_final", None),
            "iteraciones": int(iters_reportadas) if iters_reportadas is not None else None,
            "cap_disparado": _detectar_cap_disparado(
                tiempo_medido=float(res.tiempo_segundos),
                tiempo_limite=float(tarea.tiempo_limite_seg),
                iteraciones_totales=(
                    int(iters_reportadas) if iters_reportadas is not None else None
                ),
                iteraciones_max_config=kwargs.get(
                    "iteraciones_max", kwargs.get("iteraciones")
                ),
            ),
        }
        return (tarea, "ok", info, None)
    except Exception as exc:  # noqa: BLE001
        return (tarea, "fail", None, f"{type(exc).__name__}: {exc}")


# ============================================================
# Consolidación de CSVs parciales
# ============================================================

def _consolidar_por_instancia(
    dir_parciales: Path,
    dir_final: Path,
    mh: str,
) -> int:
    """Fusiona los CSVs parciales por instancia en un CSV final por instancia.

    Formato del nombre de parcial: ``{mh}_{instancia}_{pid}_{idx}.csv``.
    Formato del nombre de final:  ``{mh}_val_egl_20260710_{instancia}.csv``.
    """
    grupos: dict[str, list[Path]] = {}
    for parcial in sorted(dir_parciales.glob(f"{mh}_*.csv")):
        # Estructura: mh_<instancia>_<pid>_<idx>. Como instancias contienen
        # guiones bajos (val1A no, pero por robustez), tomamos todo lo del
        # medio quitando los dos últimos tokens (pid, idx).
        partes = parcial.stem.split("_")
        if len(partes) < 4:
            continue
        instancia = "_".join(partes[1:-2])
        grupos.setdefault(instancia, []).append(parcial)

    n_finales = 0
    for instancia, archivos in grupos.items():
        ruta_final = dir_final / f"{mh}_val_egl_20260710_{instancia}.csv"
        columnas: list[str] = []
        vistas: set[str] = set()
        filas: list[dict] = []
        for parcial in sorted(archivos):
            with parcial.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for col in reader.fieldnames or []:
                    if col not in vistas:
                        vistas.add(col)
                        columnas.append(col)
                filas.extend(reader)
        if not filas:
            continue
        with ruta_final.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columnas)
            writer.writeheader()
            writer.writerows(filas)
        n_finales += 1
    return n_finales


# ============================================================
# Orquestación por MH
# ============================================================

def _ejecutar_mh(
    mh: str,
    *,
    salida_dir: Path,
    instancias_por_clase: dict[str, list[str]],
    repeticiones: int,
    workers: int,
    root: str | None,
    semilla_base: int,
    max_evaluaciones: int,
    tiempo_limite: float,
) -> Path:
    """Corre TODAS las corridas de una MH sobre las clases pasadas.

    Reserva su propia subcarpeta ``<salida_dir>/<mh>/`` con ``final/`` y
    ``final/_partials/``, dumpea ``config_fija.json`` y consolida al final.
    """
    subdir = salida_dir / mh
    dir_final = subdir / "final"
    dir_parciales = dir_final / "_partials"
    dir_parciales.mkdir(parents=True, exist_ok=True)

    # Volcado de la config de esta MH (una fila por clase). Trazabilidad.
    config_doc = {
        "mh": mh,
        "modulo": MH_MODULOS[mh],
        "experimento": "val_egl_20260710",
        "approach": "pr_aislado",
        "umbral_pr": UMBRAL_PR_DEF,
        "max_evaluaciones_objetivo": max_evaluaciones,
        "tiempo_limite_segundos": tiempo_limite,
        "clases": {
            clase: {
                "instancias": instancias_por_clase[clase],
                "config_fija": CONFIG_POR_MH[mh][clase],
            }
            for clase in instancias_por_clase
        },
        "repeticiones": repeticiones,
        "semilla_base": semilla_base,
    }
    (subdir / "config_fija.json").write_text(
        json.dumps(config_doc, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # Construcción de tareas: (instancia, clase, repetición).
    tareas: list[Tarea] = []
    for clase, instancias in instancias_por_clase.items():
        for instancia in instancias:
            for rep in range(1, repeticiones + 1):
                idx = len(tareas)
                parcial = dir_parciales / (
                    f"{mh}_{instancia}_{os.getpid()}_{idx}.csv"
                )
                tareas.append(Tarea(
                    mh=mh,
                    instancia=instancia,
                    clase=clase,
                    repeticion=rep,
                    semilla=_derivar_semilla(semilla_base, instancia, mh, rep),
                    root=root,
                    ruta_csv_parcial=str(parcial),
                    max_evaluaciones=max_evaluaciones,
                    tiempo_limite_seg=tiempo_limite,
                ))

    print()
    print("=" * 80)
    print(f"MH={mh.upper()}  approach=pr_aislado  experimento=val_egl_20260710")
    for clase, instancias in instancias_por_clase.items():
        print(f"  Clase '{clase}': {len(instancias)} instancias  "
              f"cfg={CONFIG_POR_MH[mh][clase]}")
    print(f"  Corridas totales: {len(tareas)}   Workers: {workers}   "
          f"Reps: {repeticiones}")
    print(f"  Cap evaluaciones: {max_evaluaciones:,}   "
          f"Cap wall-clock: {tiempo_limite:.0f}s")
    print("=" * 80)

    ok = fail = 0
    if workers <= 1:
        for tarea in tareas:
            resultado = _ejecutar_tarea(tarea)
            ok, fail = _reportar_resultado(resultado, ok, fail)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futuros = {executor.submit(_ejecutar_tarea, t): t for t in tareas}
            for fut in as_completed(futuros):
                ok, fail = _reportar_resultado(fut.result(), ok, fail)

    # Consolidación de parciales por instancia.
    n_finales = _consolidar_por_instancia(dir_parciales, dir_final, mh)
    print("-" * 80)
    print(f"[{mh}] OK={ok}  FAIL={fail}  CSVs finales={n_finales}  "
          f"salida={dir_final}")
    return subdir


def _reportar_resultado(resultado: tuple, ok: int, fail: int) -> tuple[int, int]:
    """Imprime una línea por corrida (mismo estilo que pr_aislado_common)."""
    tarea, estado, info, err = resultado
    if estado == "ok" and info is not None:
        gap = info["gap"]
        gap_str = "nan" if (gap is None or math.isnan(gap)) else f"{gap:.2f}%"
        iters = info.get("iteraciones")
        iters_str = f"{iters}" if iters is not None else "?"
        print(f"  [{tarea.mh}/{tarea.clase}/{tarea.instancia}] rep={tarea.repeticion} "
              f"| costo={info['costo']:.4f} | gap={gap_str} "
              f"| t={info['tiempo']:.2f}s | iters={iters_str} "
              f"| cap={info['cap_disparado']} | fact={info['factible']}")
        return ok + 1, fail
    print(f"  [{tarea.mh}/{tarea.clase}/{tarea.instancia}] rep={tarea.repeticion} "
          f"| FAIL: {err}")
    return ok, fail + 1


# ============================================================
# CLI
# ============================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Campaña val + egl 20260710 (single-config-per-MH, PR aislado).",
    )
    parser.add_argument("--salida-dir", type=str,
                        default="experimentos_val_egl_20260710",
                        help="Directorio raíz de salida.")
    parser.add_argument("--repeticiones", type=int, default=5,
                        help="Repeticiones por (MH, instancia). Default 5.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="Grado de paralelismo. Default = os.cpu_count().")
    parser.add_argument("--semilla-base", type=int, default=0,
                        help="Semilla raíz para derivación determinista.")
    parser.add_argument("--root", type=str, default=None,
                        help="Root del proyecto para localizar los datos.")
    parser.add_argument("--solo-clase", type=str, default="todas",
                        choices=["val", "egl", "todas"],
                        help="Filtrar por CLASE de instancia.")
    parser.add_argument("--solo-mh", type=str, default="todas",
                        choices=["sa", "ts", "rts", "abc", "todas"],
                        help="Filtrar por metaheurística.")
    parser.add_argument("--smoke", action="store_true",
                        help="Prueba tibia: 2 instancias {val1A, egl-e1-A}, "
                             "1 rep, presupuesto recortado (10s, 100k evals).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    salida_dir = Path(args.salida_dir).expanduser().resolve()
    salida_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    salida_dir = salida_dir / f"corrida_{ts}"
    salida_dir.mkdir(parents=True, exist_ok=True)

    # Selección de MH y de clases.
    mhs = ["sa", "ts", "rts", "abc"] if args.solo_mh == "todas" else [args.solo_mh]
    clases_a_correr: dict[str, list[str]] = {}
    if args.solo_clase in ("val", "todas"):
        clases_a_correr["val"] = list(INSTANCIAS_VAL_GDB)
    if args.solo_clase in ("egl", "todas"):
        clases_a_correr["egl"] = list(INSTANCIAS_EGL)

    # Presupuesto: smoke usa el recortado; normal usa los defaults.
    if args.smoke:
        max_eval = MAX_EVALUACIONES_SMOKE
        tiempo_lim = TIEMPO_LIMITE_SMOKE
        reps = 1
        # Smoke solo 2 instancias representativas (una de cada clase).
        clases_a_correr = {}
        if args.solo_clase in ("val", "todas"):
            clases_a_correr["val"] = ["val1A"]
        if args.solo_clase in ("egl", "todas"):
            clases_a_correr["egl"] = ["egl-e1-A"]
    else:
        max_eval = MAX_EVALUACIONES_DEF
        tiempo_lim = TIEMPO_LIMITE_DEF
        reps = args.repeticiones

    print("=" * 80)
    print(f"Campaña val_egl_20260710  |  timestamp={ts}")
    print(f"Salida raíz         : {salida_dir}")
    print(f"MHs                  : {mhs}")
    print(f"Clases               : {list(clases_a_correr.keys())}")
    for clase, instancias in clases_a_correr.items():
        print(f"  {clase}: {len(instancias)} instancias")
    print(f"Repeticiones         : {reps}")
    print(f"Workers              : {args.workers}")
    print(f"Semilla base         : {args.semilla_base}")
    print(f"Cap evaluaciones     : {max_eval:,}")
    print(f"Cap wall-clock       : {tiempo_lim:.0f}s")
    print(f"Modo smoke           : {args.smoke}")
    print("=" * 80)

    subdirs: list[Path] = []
    for mh in mhs:
        subdir = _ejecutar_mh(
            mh,
            salida_dir=salida_dir,
            instancias_por_clase=clases_a_correr,
            repeticiones=reps,
            workers=args.workers,
            root=args.root,
            semilla_base=args.semilla_base,
            max_evaluaciones=max_eval,
            tiempo_limite=tiempo_lim,
        )
        subdirs.append(subdir)

    print()
    print("=" * 80)
    print("Fin de la campaña.")
    for sd in subdirs:
        print(f"  Salida MH: {sd}")
    print("=" * 80)


if __name__ == "__main__":
    main()
