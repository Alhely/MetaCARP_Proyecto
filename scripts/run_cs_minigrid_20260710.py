"""
Cuckoo Search — MINI-GRID de confirmación previo a su campaña val + egl.

Objetivo
========
Cuckoo Search entra a la campaña ``val_egl_20260710`` en un lote SEPARADO
porque su comportamiento en instancias grandes tiene margen de ajuste que
las 4 MH restantes ya cerraron. Este mini-grid valida DOS decisiones antes
de la campaña completa:

  1. ``factor_pasos`` — controla la escala del vuelo Lévy discreto
     (``pasos_levy_base = max(3, round(factor_pasos · √n))``).
  2. ``p_inter`` — sesgo inter/intra-ruta base (con piso interno 0.8 bajo
     violación de capacidad).

Se corren 3 × 3 = 9 combinaciones sobre 3 instancias representativas
(una pequeña, una media, una egl grande) con 2 repeticiones ⇒ 54 corridas.

Warm-start: los nidos se siembran con la solución de PATH-SCANNING mejor-de-5
(Golden, DeArmon, Baker 1983) EN LUGAR de leer el pickle aleatorio. Reusamos
``aplicar_patch_warmstart_ps`` de ``metacarp.warmstart_20260529`` — la
maquinaria de warm-start del proyecto (no reinventamos).

    * Como el patch REEMPLAZA la función ``cargar_solucion_inicial`` a nivel
      de módulo, DEBE aplicarse EN CADA WORKER antes de importar cuckoo.
      Ver ``_ejecutar_tarea`` para el punto exacto de instalación.

El resto de la configuración es fija por decisión del usuario:

    pa_abandono = 0.15
    beta_levy   = 1.3
    reps        = 2

Presupuesto (mismo cap que la campaña val_egl_20260710)
-------------------------------------------------------
    * ``tiempo_limite_segundos = 300`` s wall-clock.
    * ``iteraciones ≈ max_evaluaciones / (2 · num_nidos)``.

Salida
------
    experimentos_val_egl_20260710/cs_minigrid/<timestamp>/
        final/                     CSVs por (fp, pi, instancia)
            _partials/             parciales antes de consolidar
        config_grid.json           documentación del grid corrido

CLI
---
    python scripts/run_cs_minigrid_20260710.py --smoke
        Prueba tibia: 1 instancia (val1A), 1 combinación (factor_pasos=0.5,
        p_inter=0.4), 1 rep, presupuesto recortado.
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

# BLAS/OpenMP a 1 hilo por proceso (mismo motivo que run_val_egl_20260710).
for _var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

_ROOT_PROYECTO = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_ROOT_PROYECTO) not in sys.path:
    sys.path.insert(0, str(_ROOT_PROYECTO))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from metacarp.vecindarios import OPERADORES_POPULARES  # noqa: E402
from metacarp.instances import load_instances  # noqa: E402


# ============================================================
# Configuración
# ============================================================

INSTANCIAS: list[str] = ["val1A", "val5C", "egl-e1-A"]

FACTOR_PASOS_GRID: tuple[float, ...] = (0.25, 0.5, 1.0)
P_INTER_GRID: tuple[float, ...] = (0.1, 0.4, 0.6)

# Fijos por decisión del usuario.
PA_ABANDONO_FIJO: float = 0.15
BETA_LEVY_FIJO: float = 1.3

# num_nidos default de cuckoo es max(10, round(2·√n)); para presupuestar
# iteraciones necesitamos una estimación estable. Usamos num_nidos=25 fijo:
#   * cae dentro del rango típico del algoritmo,
#   * facilita el cálculo del cap de iteraciones (evals/ciclo = 2·num_nidos = 50),
#   * las diferencias entre nidos_default y 25 son pequeñas para el mini-grid.
NUM_NIDOS_FIJO: int = 25

UMBRAL_PR_DEF: int = 30
TIEMPO_LIMITE_DEF: float = 300.0
MAX_EVALUACIONES_DEF: int = 1_000_000

TIEMPO_LIMITE_SMOKE: float = 10.0
MAX_EVALUACIONES_SMOKE: int = 100_000


# ============================================================
# Dataclass de tarea
# ============================================================

@dataclass(frozen=True)
class Tarea:
    """Encapsula UNA corrida (factor_pasos, p_inter, instancia, repetición)."""
    instancia: str
    factor_pasos: float
    p_inter: float
    repeticion: int
    semilla: int
    root: str | None
    ruta_csv_parcial: str
    max_evaluaciones: int
    tiempo_limite_seg: float


# ============================================================
# Semilla determinista
# ============================================================

def _derivar_semilla(
    base: int,
    instancia: str,
    factor_pasos: float,
    p_inter: float,
    repeticion: int,
) -> int:
    """Semilla determinista para una corrida del mini-grid.

    Los floats se cuantizan × 100 (mismo tratamiento que run_sa_automatico
    y pr_aislado_common). Así la semilla no depende de la representación
    binaria del float y es reproducible entre versiones de Python.
    """
    h = base
    for ch in instancia:
        h = (h * 1000003) ^ ord(ch)
    h = (h * 1000003) ^ int(round(factor_pasos * 100))
    h = (h * 1000003) ^ int(round(p_inter * 100))
    h = (h * 1000003) ^ int(repeticion)
    return h & 0x7FFFFFFF


# ============================================================
# Warm-start (Path-Scanning)
# ============================================================

def _instalar_warmstart_path_scanning(root: str | None) -> None:
    """Reemplaza ``cargar_solucion_inicial`` por Path-Scanning mejor-de-5.

    Es OBLIGATORIO llamar a esta función en CADA worker antes de importar
    ``cuckoo_search`` (el patch se aplica sobre símbolos importados por
    valor en los módulos MH). Consultar el docstring de
    ``metacarp.warmstart_20260529.aplicar_patch_warmstart_ps`` para el
    orden crítico.

    Además, ``ProcessPoolExecutor`` REUTILIZA workers entre tareas, así
    que la reinstalación por tarea es idempotente (el patch simplemente
    reasigna el símbolo, sin efectos secundarios).
    """
    from metacarp.warmstart_20260529 import aplicar_patch_warmstart_ps
    aplicar_patch_warmstart_ps(root=root)


# ============================================================
# Cálculo de n_tareas por instancia (para el cap de iteraciones)
# ============================================================

def _n_tareas(nombre_instancia: str, root: str | None) -> int:
    data = load_instances(nombre_instancia, root=root)
    return int(len(data["LISTA_ARISTAS_REQ"]))


def _cap_iters(max_evaluaciones: int, evals_por_ciclo: int) -> int:
    return max(1, max_evaluaciones // max(1, evals_por_ciclo))


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
    """Etiqueta corta: TIEMPO | ITERS | NATURAL. Ver run_val_egl_20260710."""
    if tiempo_limite > 0 and tiempo_medido >= 0.9 * tiempo_limite:
        return "TIEMPO"
    if (iteraciones_totales is not None
            and iteraciones_max_config is not None
            and iteraciones_totales >= iteraciones_max_config):
        return "ITERS"
    return "NATURAL"


def _ejecutar_tarea(tarea: Tarea) -> tuple[Tarea, str, dict | None, str | None]:
    """Ejecuta una corrida de Cuckoo con Path-Scanning warm-start."""
    try:
        # 1) Warm-start ANTES de importar cuckoo (patch a nivel de módulo).
        _instalar_warmstart_path_scanning(tarea.root)

        # 2) Import diferido para asegurar que el patch de warm-start ya
        #    está instalado en el worker. Cuckoo lee cargar_solucion_inicial
        #    por nombre cualificado desde el módulo MH; ver el docstring
        #    de aplicar_patch_warmstart_ps para el orden crítico.
        from metacarp import cuckoo_search_desde_instancia
        from metacarp.path_relinking_limpio_20260531 import hook_pr_labels

        n_tareas = _n_tareas(tarea.instancia, tarea.root)
        # Cap de iteraciones ≈ evaluaciones / (2 · num_nidos). Usamos el
        # num_nidos fijo para tener un cap determinista independiente de n.
        iters = _cap_iters(tarea.max_evaluaciones, 2 * NUM_NIDOS_FIJO)

        extra_csv = {
            "experimento":     "cs_minigrid_20260710",
            "approach":        "pr_aislado",
            "selector":        "p_inter",
            "intensificador":  "path_relinking_limpio",
            "umbral_pr":       str(UMBRAL_PR_DEF),
            "lambda":          "default_instance_aware",
            "warmstart":       "path_scanning_mejor_de_5",
            "cap_tiempo_seg":  f"{tarea.tiempo_limite_seg:.1f}",
            "cap_evaluaciones": str(tarea.max_evaluaciones),
            "factor_pasos":    f"{tarea.factor_pasos:.2f}",
            "p_inter":         f"{tarea.p_inter:.2f}",
            "n_tareas":        str(n_tareas),
        }

        res = cuckoo_search_desde_instancia(
            tarea.instancia,
            root=tarea.root,
            operadores=OPERADORES_POPULARES,
            usar_gpu=False,
            semilla=tarea.semilla,
            repeticion=tarea.repeticion,
            guardar_csv=True,
            ruta_csv=tarea.ruta_csv_parcial,
            guardar_historial=False,
            lambda_capacidad=None,
            # Presupuesto: cap por iteraciones y por wall-clock.
            iteraciones=iters,
            max_iter_sin_mejora=iters,
            tiempo_limite_segundos=float(tarea.tiempo_limite_seg),
            # Grid del mini-experimento (los dos knobs que aislamos).
            factor_pasos=float(tarea.factor_pasos),
            p_inter=float(tarea.p_inter),
            # Fijos del mini-grid.
            num_nidos=NUM_NIDOS_FIJO,
            pa_abandono=PA_ABANDONO_FIJO,
            beta_levy=BETA_LEVY_FIJO,
            # PR aislado (misma composición que _pr_aislado_20260531_common).
            max_iter_sin_mejora_kick=UMBRAL_PR_DEF,
            max_resets=None,
            intensificador=hook_pr_labels,
            extra_csv=extra_csv,
        )

        iters_reportadas = getattr(res, "iteraciones_totales", None) \
            or getattr(res, "iteraciones", None)

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
                iteraciones_max_config=iters,
            ),
        }
        return (tarea, "ok", info, None)
    except Exception as exc:  # noqa: BLE001
        return (tarea, "fail", None, f"{type(exc).__name__}: {exc}")


# ============================================================
# Consolidación de CSVs parciales
# ============================================================

def _consolidar_por_combo(
    dir_parciales: Path,
    dir_final: Path,
) -> int:
    """Fusiona los parciales por (factor_pasos, p_inter, instancia).

    Formato del stem parcial:
        cs_fp{FF}_pi{PP}_{instancia}_{pid}_{idx}.csv
    Formato del final:
        cs_fp{FF}_pi{PP}_{instancia}.csv
    """
    grupos: dict[tuple[str, str, str], list[Path]] = {}
    for parcial in sorted(dir_parciales.glob("cs_*.csv")):
        # cs_fp025_pi010_val1A_<pid>_<idx>
        partes = parcial.stem.split("_")
        if len(partes) < 5 or not partes[0] == "cs":
            continue
        fp = partes[1]  # fp025
        pi = partes[2]  # pi010
        # Instancia = todo entre pi{PP} y los dos últimos (pid, idx).
        instancia = "_".join(partes[3:-2])
        grupos.setdefault((fp, pi, instancia), []).append(parcial)

    n_finales = 0
    for (fp, pi, instancia), archivos in grupos.items():
        ruta_final = dir_final / f"cs_{fp}_{pi}_{instancia}.csv"
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
# CLI y bucle principal
# ============================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cuckoo Search — mini-grid de confirmación (val_egl_20260710).",
    )
    parser.add_argument("--salida-dir", type=str,
                        default="experimentos_val_egl_20260710",
                        help="Directorio raíz de salida.")
    parser.add_argument("--repeticiones", type=int, default=2,
                        help="Repeticiones por combinación (default 2).")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="Grado de paralelismo (default = os.cpu_count()).")
    parser.add_argument("--semilla-base", type=int, default=0,
                        help="Semilla raíz para derivación determinista.")
    parser.add_argument("--root", type=str, default=None,
                        help="Root del proyecto para localizar los datos.")
    parser.add_argument("--smoke", action="store_true",
                        help="Prueba tibia: val1A × 1 combinación (fp=0.5, "
                             "pi=0.4) × 1 rep, presupuesto recortado.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    salida_raiz = Path(args.salida_dir).expanduser().resolve() / "cs_minigrid"
    salida_raiz.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    salida_dir = salida_raiz / f"corrida_{ts}"
    dir_final = salida_dir / "final"
    dir_parciales = dir_final / "_partials"
    dir_parciales.mkdir(parents=True, exist_ok=True)

    # Selección del grid (normal vs smoke).
    if args.smoke:
        instancias = ["val1A"]
        fps = (0.5,)
        pis = (0.4,)
        reps = 1
        max_eval = MAX_EVALUACIONES_SMOKE
        tiempo_lim = TIEMPO_LIMITE_SMOKE
    else:
        instancias = INSTANCIAS
        fps = FACTOR_PASOS_GRID
        pis = P_INTER_GRID
        reps = args.repeticiones
        max_eval = MAX_EVALUACIONES_DEF
        tiempo_lim = TIEMPO_LIMITE_DEF

    # Construcción de tareas del grid.
    tareas: list[Tarea] = []
    for instancia in instancias:
        for fp in fps:
            for pi in pis:
                for rep in range(1, reps + 1):
                    idx = len(tareas)
                    # Codificamos fp y pi como "fp025" / "pi010" para
                    # nombres de archivo ASCII-safe y ordenables.
                    fp_tag = f"fp{int(round(fp * 100)):03d}"
                    pi_tag = f"pi{int(round(pi * 100)):03d}"
                    parcial = dir_parciales / (
                        f"cs_{fp_tag}_{pi_tag}_{instancia}_"
                        f"{os.getpid()}_{idx}.csv"
                    )
                    tareas.append(Tarea(
                        instancia=instancia,
                        factor_pasos=fp,
                        p_inter=pi,
                        repeticion=rep,
                        semilla=_derivar_semilla(
                            args.semilla_base, instancia, fp, pi, rep,
                        ),
                        root=args.root,
                        ruta_csv_parcial=str(parcial),
                        max_evaluaciones=max_eval,
                        tiempo_limite_seg=tiempo_lim,
                    ))

    # Trazabilidad del grid.
    config_doc = {
        "experimento": "cs_minigrid_20260710",
        "instancias": instancias,
        "factor_pasos_grid": list(fps),
        "p_inter_grid": list(pis),
        "num_nidos_fijo": NUM_NIDOS_FIJO,
        "pa_abandono_fijo": PA_ABANDONO_FIJO,
        "beta_levy_fijo": BETA_LEVY_FIJO,
        "warmstart": "path_scanning_mejor_de_5",
        "umbral_pr": UMBRAL_PR_DEF,
        "intensificador": "path_relinking_limpio",
        "repeticiones": reps,
        "workers": args.workers,
        "cap_evaluaciones": max_eval,
        "cap_wallclock_segundos": tiempo_lim,
        "semilla_base": args.semilla_base,
        "corridas_totales": len(tareas),
    }
    (salida_dir / "config_grid.json").write_text(
        json.dumps(config_doc, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    print("=" * 80)
    print(f"Cuckoo Search — mini-grid val_egl_20260710  |  ts={ts}")
    print(f"Instancias        : {instancias}")
    print(f"factor_pasos      : {list(fps)}")
    print(f"p_inter           : {list(pis)}")
    print(f"Repeticiones      : {reps}")
    print(f"Workers           : {args.workers}")
    print(f"Corridas totales  : {len(tareas)}")
    print(f"pa_abandono/beta  : {PA_ABANDONO_FIJO} / {BETA_LEVY_FIJO}")
    print(f"num_nidos fijo    : {NUM_NIDOS_FIJO}")
    print(f"Cap evaluaciones  : {max_eval:,}")
    print(f"Cap wall-clock    : {tiempo_lim:.0f}s")
    print(f"Warm-start        : Path-Scanning mejor-de-5")
    print(f"Salida            : {salida_dir}")
    print("=" * 80)

    ok = fail = 0
    if args.workers <= 1:
        for tarea in tareas:
            ok, fail = _reportar(_ejecutar_tarea(tarea), ok, fail)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futuros = {executor.submit(_ejecutar_tarea, t): t for t in tareas}
            for fut in as_completed(futuros):
                ok, fail = _reportar(fut.result(), ok, fail)

    n_finales = _consolidar_por_combo(dir_parciales, dir_final)
    print("-" * 80)
    print(f"OK={ok}  FAIL={fail}  CSVs finales={n_finales}  salida={dir_final}")


def _reportar(resultado: tuple, ok: int, fail: int) -> tuple[int, int]:
    tarea, estado, info, err = resultado
    if estado == "ok" and info is not None:
        gap = info["gap"]
        gap_str = "nan" if (gap is None or math.isnan(gap)) else f"{gap:.2f}%"
        iters = info.get("iteraciones")
        iters_str = f"{iters}" if iters is not None else "?"
        print(f"  [cs/{tarea.instancia}] fp={tarea.factor_pasos:.2f} "
              f"pi={tarea.p_inter:.2f} rep={tarea.repeticion} "
              f"| costo={info['costo']:.4f} | gap={gap_str} "
              f"| t={info['tiempo']:.2f}s | iters={iters_str} "
              f"| cap={info['cap_disparado']} | fact={info['factible']}")
        return ok + 1, fail
    print(f"  [cs/{tarea.instancia}] fp={tarea.factor_pasos:.2f} "
          f"pi={tarea.p_inter:.2f} rep={tarea.repeticion} | FAIL: {err}")
    return ok, fail + 1


if __name__ == "__main__":
    main()
