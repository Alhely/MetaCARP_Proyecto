"""
Corrida SA para instancias seleccionadas.

Configuración:
    temperatura_inicial         = None  →  automática: 20·d_max/n por instancia
    temperatura_minima          = 1e-3  →  fija para todas las instancias
    iteraciones_por_temperatura = None  →  automático: n² por instancia
    patience                    = 10    →  niveles sin mejora antes de reheat
    reheat_factor               = 0.5   →  fracción de T_init a la que se recalienta
    max_reheats_sin_mejora      = 5     →  reheats consecutivos sin mejora antes de parar
    alpha   = [0.80, 0.82, ..., 0.98, 0.99]  →  11 valores
    p_inter = [0.4, 0.5, 0.6, 0.7, 0.8]      →  5 valores

Total: 11 alpha × 5 p_inter × 23 instancias × 2 repeticiones = 2,530 corridas.

Paralelismo (--workers N)
-------------------------
Cada combinación (instancia, alpha, p_inter, repetición) es una tarea
INDEPENDIENTE. Las paralelizamos con ``ProcessPoolExecutor`` siguiendo el
mismo patrón que los scripts de TS simple y RTS:
    --workers 1   → modo secuencial reproducible.
    --workers N   → N procesos en paralelo (default: os.cpu_count()).

Reproducibilidad:
    Cada tarea recibe una semilla determinista derivada de
    ``(semilla_base, instancia, alpha, p_inter, repeticion)``. Con la
    misma ``--semilla-base`` la grilla completa es reproducible bit-a-bit
    independientemente del orden de ejecución.

Concurrencia de E/S:
    Cada worker escribe a su propio CSV parcial en ``_partials/``. Al
    finalizar todas las tareas, el proceso principal consolida los
    parciales en los CSV finales canónicos (uno por instancia).

Uso:
    python scripts/run_sa_automatico.py
    python scripts/run_sa_automatico.py --salida-dir resultados_sa
    python scripts/run_sa_automatico.py --repeticiones 3
    python scripts/run_sa_automatico.py --workers 1   # secuencial
    python scripts/run_sa_automatico.py --workers 8   # 8 procesos
"""
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from metacarp import recocido_simulado_desde_instancia

INSTANCIAS = [
    "gdb19", "kshs1", "kshs2", "kshs3", "kshs4", "kshs5", "kshs6",
    "gdb4",  "gdb14", "gdb15", "gdb1",  "gdb20", "gdb3",  "gdb6",
    "gdb7",  "gdb12", "gdb10", "gdb2",  "gdb5",  "gdb13", "gdb16",
    "gdb17", "gdb21",
]

ALPHAS   = [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99]
P_INTERS = [0.4, 0.5, 0.6, 0.7, 0.8]


# --- DATACLASS DE TAREA ---
# Encapsula los parámetros de UNA corrida de SA. Objeto plano, serializable,
# que se envía al worker. Mantener todos los argumentos en un único objeto
# facilita extender el grid sin tocar las firmas de las funciones internas.
@dataclass(frozen=True)
class TareaSA:
    instancia: str
    alpha: float
    p_inter: float
    repeticion: int
    semilla: int                              # semilla determinista por tarea
    root: str | None
    ruta_csv_parcial: str                     # cada tarea escribe a su propio archivo


def _derivar_semilla(
    base: int,
    instancia: str,
    alpha: float,
    p_inter: float,
    repeticion: int,
) -> int:
    """
    Calcula una semilla determinista para una corrida concreta.

    Misma lógica que en run_tabu_simple_automatico y run_tabu_reactiva_automatico:
    hash de los parámetros de la tarea para que cada corrida sea reproducible
    de forma independiente, sin depender del orden de ejecución. Los floats se
    cuantizan ×100 antes de mezclar (los floats no hashean estable entre
    versiones de Python).
    """
    h = base
    for ch in instancia:
        h = (h * 1000003) ^ ord(ch)
    h = (h * 1000003) ^ int(round(alpha * 100))
    h = (h * 1000003) ^ int(round(p_inter * 100))
    h = (h * 1000003) ^ int(repeticion)
    return h & 0x7FFFFFFF


def _ejecutar_tarea(tarea: TareaSA) -> tuple[TareaSA, str, dict | None, str | None]:
    """
    Ejecuta UNA corrida de SA en el worker.

    Esta función se ejecuta dentro de un PROCESO HIJO. Devuelve una tupla
    con el estado del resultado; el proceso principal formatea la salida.

    Returns
    -------
    (tarea, "ok"|"fail", info_dict_o_None, mensaje_error_o_None)
    """
    try:
        res = recocido_simulado_desde_instancia(
            tarea.instancia,
            temperatura_inicial=None,
            temperatura_minima=1e-3,
            alpha=tarea.alpha,
            p_inter=tarea.p_inter,
            patience=10,
            reheat_factor=0.5,
            max_reheats_sin_mejora=5,
            semilla=tarea.semilla,
            repeticion=tarea.repeticion,
            guardar_csv=True,
            ruta_csv=tarea.ruta_csv_parcial,
            guardar_historial=False,
            root=tarea.root,
        )
        info = {
            "costo": res.mejor_costo,
            "mejora": res.mejora_porcentaje_inicial_vs_final,
            "tiempo": res.tiempo_segundos,
        }
        return (tarea, "ok", info, None)
    except Exception as exc:  # noqa: BLE001
        return (tarea, "fail", None, f"{type(exc).__name__}: {exc}")


def _consolidar_parciales(
    dir_parciales: Path,
    salida_dir: Path,
    experimento: str,
    ydmh: str,
) -> int:
    """
    Concatena los CSVs parciales por instancia en un único CSV final por
    instancia. Misma estrategia map-reduce que los scripts de TS:
    cada worker escribe un parcial, el principal fusiona al final sin locks.

    Devuelve el número de archivos finales generados.
    """
    grupos: dict[str, list[Path]] = {}
    for parcial in sorted(dir_parciales.glob("sa_*.csv")):
        # Nombre con formato ``sa_<instancia>_<pid>_<idx>.csv``.
        partes = parcial.stem.split("_")
        if len(partes) < 3:
            continue
        instancia = partes[1]
        grupos.setdefault(instancia, []).append(parcial)

    n_archivos_creados = 0
    for instancia, archivos in grupos.items():
        ruta_final = salida_dir / f"sa_{instancia}_{experimento}_{ydmh}.csv"
        filas: list[dict] = []
        columnas_union: list[str] = []
        col_vistas: set[str] = set()
        for parcial in archivos:
            with parcial.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for col in reader.fieldnames or []:
                    if col not in col_vistas:
                        col_vistas.add(col)
                        columnas_union.append(col)
                for fila in reader:
                    filas.append(fila)
        with ruta_final.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columnas_union)
            writer.writeheader()
            for fila in filas:
                writer.writerow(fila)
        n_archivos_creados += 1

    return n_archivos_creados


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SA grid search sobre alpha y p_inter.")
    parser.add_argument("--salida-dir",   type=str, default="experimentos")
    parser.add_argument("--repeticiones", type=int, default=2)
    parser.add_argument("--experimento",  type=str, default="sa_auto")
    parser.add_argument("--root",         type=str, default=None)
    # --- Paralelismo ---
    # --workers controla el grado de concurrencia. Default = todos los núcleos.
    # --workers 1 ejecuta secuencialmente (sin spawnear procesos hijo), útil
    # para depurar y mantener un stdout predecible.
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Número de procesos paralelos. 1 = modo secuencial. "
             "Default = os.cpu_count().",
    )
    # --semilla-base raíz para semillas deterministas por tarea.
    parser.add_argument(
        "--semilla-base",
        type=int,
        default=0,
        help="Semilla raíz usada para derivar semillas deterministas por tarea.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    salida_dir = Path(args.salida_dir).expanduser().resolve() / "sa_small_instances_reheat_w_patience"
    salida_dir.mkdir(parents=True, exist_ok=True)
    # Subcarpeta para los CSV parciales de los workers (siempre, aun en modo secuencial).
    dir_parciales = salida_dir / "_partials"
    dir_parciales.mkdir(parents=True, exist_ok=True)
    ydmh = datetime.now().strftime("%Y%d%m%H%M")

    # --- CONSTRUCCIÓN DE LA LISTA DE TAREAS ---
    # Una tarea por (instancia, alpha, p_inter, repeticion). Independientes
    # entre sí, aptas para paralelizar trivialmente.
    tareas: list[TareaSA] = []
    for instancia in INSTANCIAS:
        for alpha in ALPHAS:
            for p_inter in P_INTERS:
                for rep in range(1, args.repeticiones + 1):
                    idx = len(tareas)
                    parcial = dir_parciales / (
                        f"sa_{instancia}_{os.getpid()}_{idx}.csv"
                    )
                    tareas.append(TareaSA(
                        instancia=instancia,
                        alpha=alpha,
                        p_inter=p_inter,
                        repeticion=rep,
                        semilla=_derivar_semilla(
                            args.semilla_base, instancia, alpha, p_inter, rep
                        ),
                        root=args.root,
                        ruta_csv_parcial=str(parcial),
                    ))

    total = len(tareas)
    print("=" * 80)
    print("SA  —  grid search alpha × p_inter  (con reheat)")
    print("=" * 80)
    print(f"Instancias                  : {len(INSTANCIAS)}")
    print(f"T_ini                       : automática (20·d_max/n por instancia)")
    print(f"T_min                       : 1e-3 (fija)")
    print(f"iteraciones_por_temperatura : None (adaptativo: n²)")
    print(f"Patience                    : 10 niveles sin mejora")
    print(f"Reheat factor               : 0.5 (T sube al 50% de T_init)")
    print(f"max_reheats_sin_mejora      : 5 (parada temprana tras 5 reheats estériles)")
    print(f"Alpha values                : {ALPHAS}")
    print(f"p_inter vals                : {P_INTERS}")
    print(f"Semilla base                : {args.semilla_base} (derivada por tarea)")
    print(f"Repeticiones                : {args.repeticiones}")
    print(f"Workers                     : {args.workers}")
    print(f"Corridas                    : {total}")
    print(f"Salida CSV                  : {salida_dir}")
    print("-" * 80)

    total_ok = 0
    total_fail = 0

    # --- EJECUCIÓN ---
    # Rama secuencial (debug) vs paralela (productiva). Misma estructura que
    # los scripts de TS simple y RTS para mantener consistencia.
    if args.workers <= 1:
        for tarea in tareas:
            _, estado, info, err = _ejecutar_tarea(tarea)
            if estado == "ok" and info is not None:
                print(
                    f"  [{tarea.instancia}] alpha={tarea.alpha:.2f} p_inter={tarea.p_inter:.1f} "
                    f"rep={tarea.repeticion} | costo={info['costo']:.4f} "
                    f"| mejora={info['mejora']:.2f}% "
                    f"| t={info['tiempo']:.2f}s"
                )
                total_ok += 1
            else:
                print(
                    f"  [{tarea.instancia}] alpha={tarea.alpha:.2f} p_inter={tarea.p_inter:.1f} "
                    f"rep={tarea.repeticion} | FAIL: {err}"
                )
                total_fail += 1
    else:
        # MODO PARALELO con ProcessPoolExecutor.
        # Garantías equivalentes a los otros scripts:
        #   1. Tareas puras sin estado compartido.
        #   2. CSV parcial único por tarea (sin contención de E/S).
        #   3. Semillas deterministas independientes del orden de ejecución.
        # ``as_completed`` reporta progreso en cuanto las tareas terminan.
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futuros = {executor.submit(_ejecutar_tarea, t): t for t in tareas}
            for fut in as_completed(futuros):
                tarea, estado, info, err = fut.result()
                if estado == "ok" and info is not None:
                    print(
                        f"  [{tarea.instancia}] alpha={tarea.alpha:.2f} p_inter={tarea.p_inter:.1f} "
                        f"rep={tarea.repeticion} | costo={info['costo']:.4f} "
                        f"| mejora={info['mejora']:.2f}% "
                        f"| t={info['tiempo']:.2f}s"
                    )
                    total_ok += 1
                else:
                    print(
                        f"  [{tarea.instancia}] alpha={tarea.alpha:.2f} p_inter={tarea.p_inter:.1f} "
                        f"rep={tarea.repeticion} | FAIL: {err}"
                    )
                    total_fail += 1

    # --- FUSIÓN DE PARCIALES ---
    print("\n" + "-" * 80)
    print(f"Consolidando CSVs parciales en {salida_dir} ...")
    n_finales = _consolidar_parciales(dir_parciales, salida_dir, args.experimento, ydmh)
    print(f"CSVs finales generados      : {n_finales}")

    print("\n" + "-" * 80)
    print(f"OK   : {total_ok}")
    print(f"FAIL : {total_fail}")
    print(f"CSV  : {salida_dir}")
    print(f"Parciales en                : {dir_parciales}  (no se borran automáticamente)")
    print("-" * 80)


if __name__ == "__main__":
    main()
