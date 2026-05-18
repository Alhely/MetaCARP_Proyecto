"""
Corrida Reactive Tabu Search (Battiti & Tecchiolli 1994) para instancias seleccionadas.

Versión REACTIVA de TS:
- Tenure dinámico auto-ajustable dentro del rango [min, max].
- Memoria de soluciones visitadas (hash canónico) para detectar ciclos.
- Mecanismo de escape por movimientos aleatorios + limpieza de memoria.

Configuración (grid search AMPLIO sobre 4 parámetros de RTS):

    iteraciones_max                  = instance-aware (20·n) salvo override
    max_iter_sin_mejora              = instance-aware (5·n)  salvo override
    tam_vecindario                   = instance-aware (2·n, min 20)
    tabu_tenure_inicial              = instance-aware (sqrt(n), min 3)
    tabu_tenure_min                  = 3
    tabu_tenure_max                  = instance-aware (3·sqrt(n), min 15)
    iter_sin_repeticion_para_reducir = instance-aware (2·sqrt(n), min 5)
    num_movimientos_escape           = instance-aware (max(3, n//10))

    Grid (4 dimensiones):
        factor_aumento             ∈ {1.05, 1.1, 1.2, 1.3, 1.4}
        umbral_repeticiones_escape ∈ {2, 3, 5, 8}
        p_inter                    ∈ {0.4, 0.5, 0.6, 0.7, 0.8}
        factor_reduccion           ∈ {0.85, 0.9, 0.95}

    operadores  = OPERADORES_POPULARES (9 operadores)
    semilla     = ALEATORIA del sistema (NO determinista)
    repeticiones = 5 (configurable por CLI)

NOTA SOBRE LA AUSENCIA DE SEMILLA FIJA:
Las repeticiones EXISTEN para medir la robustez de cada configuración a la
aleatoriedad inherente de la metaheurística (solución inicial aleatoria,
selección de vecinos, movimientos de escape). Si fijáramos una semilla
determinista por tarea, las 5 repeticiones de cada combinación serían 5
mediciones IDÉNTICAS — no aportarían información estadística. Una buena
configuración debe converger a buenos óptimos SIN IMPORTAR el punto de
partida ni la trayectoria aleatoria que tome. Por eso aquí cada corrida
muestrea libremente del espacio aleatorio del sistema.

Total: 5 * 4 * 5 * 3 = 300 combos × 23 instancias × 5 reps = 34,500 corridas.

Para REDUCIR el grid, basta con pasar overrides single-value por CLI:
    --p-inter 0.6           → fija p_inter, solo barre las otras 3 dimensiones.
    --factor-reduccion 0.9  → fija factor_reduccion, solo barre las otras 3.
    --p-inter 0.6 --factor-reduccion 0.9 → vuelve al grid compacto 2D.

Paralelismo (--workers N)
-------------------------
Cada combinación (instancia, factor_aumento, umbral_escape, p_inter,
factor_reduccion, repetición) es una tarea INDEPENDIENTE: no comparte
estado con las demás y solo lee la instancia desde disco. Esto las hace
trivialmente paralelizables con ``ProcessPoolExecutor``. El flag
``--workers N`` controla cuántos procesos se lanzan en paralelo:
    --workers 1   → ejecución secuencial (útil para depurar y porque
                    stdout no se mezcla entre tareas).
    --workers N   → N procesos en paralelo (default: os.cpu_count()).

Reproducibilidad:
    NO hay reproducibilidad bit-a-bit por diseño (ver nota arriba sobre la
    ausencia de semilla fija). Sin embargo, sí hay reproducibilidad
    ESTADÍSTICA: con 5 repeticiones por configuración, las medias y
    desviaciones de costo, mejora, número de escapes, etc. convergen a
    valores estables que sí son comparables entre experimentos.

Concurrencia de E/S:
    Para evitar carreras al escribir el CSV, cada worker escribe su fila en
    un CSV TEMPORAL único (por PID + índice de tarea) dentro de una
    subcarpeta ``_partials/``. Al terminar todas las tareas, el proceso
    principal CONCATENA los parciales por instancia en el CSV final
    canónico. Sin locks, sin colas, sin riesgo de filas truncadas.

Uso:
    python scripts/run_tabu_reactiva_automatico.py
    python scripts/run_tabu_reactiva_automatico.py --salida-dir experimentos/reactive_tabu_grid_pinter
    python scripts/run_tabu_reactiva_automatico.py --repeticiones 5
    python scripts/run_tabu_reactiva_automatico.py --workers 8
"""
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Importamos la función pública del módulo metacarp.
from metacarp import busqueda_tabu_reactiva_desde_instancia

# Mismo conjunto de instancias pequeñas que usa el script de SA y el TS simple,
# para mantener la comparabilidad de los experimentos (gdb + kshs).
INSTANCIAS = [
    "gdb19", "kshs1", "kshs2", "kshs3", "kshs4", "kshs5", "kshs6",
    "gdb4",  "gdb14", "gdb15", "gdb1",  "gdb20", "gdb3",  "gdb6",
    "gdb7",  "gdb12", "gdb10", "gdb2",  "gdb5",  "gdb13", "gdb16",
    "gdb17", "gdb21",
]

# Grid search amplio sobre 4 dimensiones:
# - factor_aumento: cuán agresivo es el crecimiento del tenure al detectar ciclos.
# - umbral_repeticiones_escape: cuántas veces hay que ver la misma solución
#   antes de disparar el mecanismo de escape (movimientos aleatorios).
# - p_inter: probabilidad de elegir un operador INTER-ruta cuando la solución
#   actual es factible. Controla el balance exploración inter/intra-ruta.
#   Default canónico de SA es 0.6; barremos un espectro amplio para estudiar
#   si RTS responde mejor con más exploración entre rutas (>0.6) o con más
#   refinamiento intra-ruta (<0.6).
# - factor_reduccion: multiplicador para encoger el tenure cuando la búsqueda
#   no detecta repeticiones por mucho tiempo. Battiti & Tecchiolli usan 0.9,
#   pero 0.85 fuerza encogimiento más agresivo y 0.95 lo hace casi pasivo.
#
# Total grid: 5 * 4 * 5 * 3 = 300 combos × 23 instancias × N repeticiones.
FACTORES_AUMENTO   = [1.05, 1.1, 1.2, 1.3, 1.4]
UMBRALES_ESCAPE    = [2, 3, 5, 8]
P_INTERS           = [0.4, 0.5, 0.6, 0.7, 0.8]
FACTORES_REDUCCION = [0.85, 0.9, 0.95]


# --- DATACLASS DE TAREA ---
# Encapsula todos los parámetros de UNA corrida. Es un objeto plano,
# pickle-friendly, que se envía a un proceso worker. Mantener todos los
# parámetros en un solo objeto evita firmas con muchos argumentos y facilita
# extender el grid en el futuro.
@dataclass(frozen=True)
class TareaRTS:
    instancia: str
    factor_aumento: float
    umbral_escape: int
    repeticion: int
    # NOTA: NO hay campo ``semilla``. Cada corrida se ejecuta con semilla
    # aleatoria del sistema (la metaheurística internamente crea su propio
    # ``random.Random()``). Esto es deliberado: las "repeticiones" SOLO son
    # informativas si cada una muestrea una trayectoria distinta del espacio
    # aleatorio. Una semilla determinista por tarea las convertiría en
    # mediciones idénticas, sin valor estadístico.
    iteraciones_max: int | None
    max_iter_sin_mejora: int | None
    tam_vecindario: int | None
    factor_reduccion: float
    alpha_inter: float | None
    p_inter: float | None
    root: str | None
    ruta_csv_parcial: str                     # cada tarea escribe a su propio archivo


def _ejecutar_tarea(tarea: TareaRTS) -> tuple[TareaRTS, str, dict | None, str | None]:
    """
    Ejecuta UNA corrida de RTS en el worker.

    Esta función se ejecuta dentro de un PROCESO HIJO. Recibe la tarea,
    invoca el algoritmo y devuelve una tupla con el estado del resultado.
    No imprime: el proceso principal formatea y muestra el resumen al
    recibir cada futuro completado.

    Returns
    -------
    (tarea, "ok"|"fail", info_dict_o_None, mensaje_error_o_None)
    """
    try:
        res = busqueda_tabu_reactiva_desde_instancia(
            tarea.instancia,
            iteraciones_max=tarea.iteraciones_max,
            max_iter_sin_mejora=tarea.max_iter_sin_mejora,
            tam_vecindario=tarea.tam_vecindario,
            factor_aumento=tarea.factor_aumento,
            factor_reduccion=tarea.factor_reduccion,
            umbral_repeticiones_escape=tarea.umbral_escape,
            # semilla=None: cada repetición usa una trayectoria aleatoria
            # distinta del sistema. Esto es lo que permite que las
            # repeticiones tengan valor estadístico.
            semilla=None,
            repeticion=tarea.repeticion,
            guardar_csv=True,
            ruta_csv=tarea.ruta_csv_parcial,
            guardar_historial=False,
            root=tarea.root,
            alpha_inter=tarea.alpha_inter,
            p_inter=tarea.p_inter,
        )
        info = {
            "costo": res.mejor_costo,
            "mejora": res.mejora_porcentaje_inicial_vs_final,
            "iter": res.iteraciones_totales,
            "reps": res.num_repeticiones_detectadas,
            "esc": res.num_escapes_realizados,
            "ten_min": res.tenure_min_alcanzado,
            "ten_max": res.tenure_max_alcanzado,
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
    instancia. Misma estrategia map-reduce que el script de TS simple:
    cada worker escribe a su parcial, el principal fusiona al final.

    Devuelve el número de archivos finales generados.
    """
    grupos: dict[str, list[Path]] = {}
    for parcial in sorted(dir_parciales.glob("tabu_reactiva_*.csv")):
        # Nombre con formato ``tabu_reactiva_<instancia>_<pid>_<idx>.csv``.
        partes = parcial.stem.split("_")
        if len(partes) < 4:
            continue
        instancia = partes[2]
        grupos.setdefault(instancia, []).append(parcial)

    n_archivos_creados = 0
    for instancia, archivos in grupos.items():
        ruta_final = salida_dir / f"tabu_reactiva_{instancia}_{experimento}_{ydmh}.csv"
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
    """Define los argumentos de línea de comandos del script."""
    parser = argparse.ArgumentParser(
        description="Reactive Tabu Search — grid search sobre factor_aumento y umbral_escape."
    )
    parser.add_argument("--salida-dir",   type=str, default="experimentos")
    parser.add_argument("--repeticiones", type=int, default=5)
    parser.add_argument("--experimento",  type=str, default="tabu_reactiva_auto")
    parser.add_argument("--root",         type=str, default=None)
    # Overrides opcionales: si no se pasan, se usan los defaults instance-aware
    # del propio busqueda_tabu_reactiva (None -> calculado dentro de la función).
    parser.add_argument("--iteraciones-max",     type=int, default=None,
                        help="Si no se pasa, se calcula como 20·n por instancia.")
    parser.add_argument("--max-iter-sin-mejora", type=int, default=None,
                        help="Si no se pasa, se calcula como 5·n por instancia.")
    parser.add_argument("--tam-vecindario",      type=int, default=None,
                        help="Si no se pasa, se calcula como max(20, 2·n) por instancia.")
    # IMPORTANTE: default None significa "barrer la lista FACTORES_REDUCCION".
    # Si se pasa un valor (--factor-reduccion 0.9), ese valor FIJA la dimensión
    # y se ignora la lista. Mismo patrón que --p-inter.
    parser.add_argument("--factor-reduccion",    type=float, default=None,
                        help="Factor de reducción del tenure. "
                             "None = barrer lista [0.85, 0.9, 0.95].")
    # Parámetros del sesgo inter/intra (compatibilidad con SA y TS simple).
    # Default None = usar el valor canónico de SA (alpha_inter=0.8, p_inter=0.6).
    # Permitimos override por CLI por si se desea explorar otros valores en
    # experimentos posteriores, pero NO los incluimos en el grid actual:
    # el experimento base estudia factor_aumento y umbral_escape aislados,
    # manteniendo el sesgo igual al de SA y TS simple para que la diferencia
    # observada se atribuya exclusivamente a los mecanismos reactivos.
    parser.add_argument("--alpha-inter", type=float, default=None,
                        help="P(elegir inter) cuando la sol. actual viola capacidad. "
                             "None = default SA (0.8).")
    parser.add_argument("--p-inter",     type=float, default=None,
                        help="P(elegir inter) cuando la sol. actual es factible. "
                             "None = default SA (0.6).")
    # --- Paralelismo ---
    # --workers controla el grado de concurrencia. Default = todos los núcleos.
    # --workers 1 ejecuta secuencialmente (sin spawnear procesos hijo), útil
    # para depurar y para garantizar que el orden de stdout sea predecible.
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Número de procesos paralelos. 1 = modo secuencial. "
             "Default = os.cpu_count().",
    )
    # No exponemos ``--semilla-base``: las semillas son aleatorias del sistema
    # para que las repeticiones reflejen la variabilidad real de la
    # metaheurística (ver nota en la dataclass ``TareaRTS``).
    return parser.parse_args()


def main() -> None:
    """Punto de entrada principal del script de experimentos."""
    args = _parse_args()
    # ``--salida-dir`` apunta DIRECTAMENTE al folder final donde se escriben
    # los CSV consolidados (un archivo por instancia). Antes el script añadía
    # automáticamente una subcarpeta fija "tabu_reactive_small_<fecha>", lo
    # que dificultaba comparar experimentos lanzados en distintos días o
    # con distintos grids. Ahora el nombre del folder lo elige el usuario:
    #     --salida-dir experimentos/reactive_tabu_grid_pinter
    salida_dir = Path(args.salida_dir).expanduser().resolve()
    salida_dir.mkdir(parents=True, exist_ok=True)
    # Subcarpeta para los CSV parciales de los workers (siempre, aunque sea modo secuencial).
    dir_parciales = salida_dir / "_partials"
    dir_parciales.mkdir(parents=True, exist_ok=True)
    # Marca de tiempo: año-día-mes-hora-minuto, mismo formato que SA / TS simple.
    ydmh = datetime.now().strftime("%Y%d%m%H%M")

    # --- RESOLUCIÓN DE LAS DIMENSIONES DEL GRID ---
    # Si el usuario pasa un override single-value por CLI (--p-inter 0.6 o
    # --factor-reduccion 0.9), ese valor SUSTITUYE a la lista completa del
    # grid: la dimensión correspondiente queda fijada a un solo punto.
    # Esto permite dos modos de uso sin tocar el código:
    #   - Grid completo:        no pasar --p-inter ni --factor-reduccion.
    #   - Fijar p_inter a 0.6:  --p-inter 0.6 (entonces solo se barre el resto).
    # Para alpha_inter no hay grid (no es un parámetro central de RTS según
    # el reporte previo); se queda como un solo valor (override o default).
    p_inter_values  = [args.p_inter]        if args.p_inter        is not None else P_INTERS
    f_red_values    = [args.factor_reduccion] if args.factor_reduccion is not None else FACTORES_REDUCCION

    # --- CONSTRUCCIÓN DE LA LISTA DE TAREAS ---
    # Una tarea por (instancia, factor_aumento, umbral_escape, p_inter,
    # factor_reduccion, repeticion). Cada tarea es independiente: comparte
    # solo lectura de la instancia desde disco. Esto las hace trivialmente
    # paralelizables con ProcessPoolExecutor.
    tareas: list[TareaRTS] = []
    for instancia in INSTANCIAS:
        for f_aum in FACTORES_AUMENTO:
            for umbral in UMBRALES_ESCAPE:
                for p_int in p_inter_values:
                    for f_red in f_red_values:
                        for rep in range(1, args.repeticiones + 1):
                            idx = len(tareas)
                            parcial = dir_parciales / (
                                f"tabu_reactiva_{instancia}_{os.getpid()}_{idx}.csv"
                            )
                            tareas.append(TareaRTS(
                                instancia=instancia,
                                factor_aumento=f_aum,
                                umbral_escape=umbral,
                                repeticion=rep,
                                iteraciones_max=args.iteraciones_max,
                                max_iter_sin_mejora=args.max_iter_sin_mejora,
                                tam_vecindario=args.tam_vecindario,
                                factor_reduccion=f_red,
                                alpha_inter=args.alpha_inter,
                                p_inter=p_int,
                                root=args.root,
                                ruta_csv_parcial=str(parcial),
                            ))

    total = len(tareas)
    print("=" * 80)
    print("Reactive Tabu Search (Battiti & Tecchiolli 1994) — grid search")
    print("=" * 80)
    print(f"Instancias                  : {len(INSTANCIAS)}")
    print(f"iteraciones_max             : {args.iteraciones_max if args.iteraciones_max else 'instance-aware (20·n)'}")
    print(f"max_iter_sin_mejora         : {args.max_iter_sin_mejora if args.max_iter_sin_mejora else 'instance-aware (5·n)'}")
    print(f"tam_vecindario              : {args.tam_vecindario if args.tam_vecindario else 'instance-aware (2·n)'}")
    print(f"factor_aumento values       : {FACTORES_AUMENTO}")
    print(f"umbral_escape values        : {UMBRALES_ESCAPE}")
    print(f"p_inter values              : {p_inter_values}")
    print(f"factor_reduccion values     : {f_red_values}")
    print(f"Semilla                     : aleatoria del sistema (cada repetición es independiente)")
    print(f"Repeticiones                : {args.repeticiones}")
    print(f"Workers                     : {args.workers}")
    print(f"Corridas                    : {total}")
    print(f"Salida CSV                  : {salida_dir}")
    print("-" * 80)

    total_ok = 0
    total_fail = 0

    # --- EJECUCIÓN ---
    # Misma estructura que el script TS simple: rama secuencial explícita
    # cuando --workers <= 1, rama paralela con ProcessPoolExecutor en otro caso.
    if args.workers <= 1:
        for tarea in tareas:
            _, estado, info, err = _ejecutar_tarea(tarea)
            if estado == "ok" and info is not None:
                print(
                    f"  [{tarea.instancia}] f_aum={tarea.factor_aumento:.2f} "
                    f"umb={tarea.umbral_escape} "
                    f"pInt={tarea.p_inter:.2f} "
                    f"fRed={tarea.factor_reduccion:.2f} "
                    f"rep={tarea.repeticion} "
                    f"| costo={info['costo']:.4f} "
                    f"| mejora={info['mejora']:.2f}% "
                    f"| iter={info['iter']} "
                    f"| reps={info['reps']} "
                    f"| esc={info['esc']} "
                    f"| tenure[{info['ten_min']}-{info['ten_max']}] "
                    f"| t={info['tiempo']:.2f}s"
                )
                total_ok += 1
            else:
                print(
                    f"  [{tarea.instancia}] f_aum={tarea.factor_aumento:.2f} "
                    f"umb={tarea.umbral_escape} "
                    f"pInt={tarea.p_inter:.2f} "
                    f"fRed={tarea.factor_reduccion:.2f} "
                    f"rep={tarea.repeticion} | FAIL: {err}"
                )
                total_fail += 1
    else:
        # MODO PARALELO con ProcessPoolExecutor.
        # Mismas garantías que en el script de TS simple:
        #   1. Cada tarea es función pura (sin estado compartido).
        #   2. Cada tarea escribe a su PROPIO CSV parcial.
        #   3. Cada tarea tiene su propia semilla determinista.
        # ``as_completed`` reporta cada tarea EN CUANTO TERMINA, no en el
        # orden de envío, para que stdout muestre progreso fluido sin
        # esperar a las corridas más lentas.
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futuros = {executor.submit(_ejecutar_tarea, t): t for t in tareas}
            for fut in as_completed(futuros):
                tarea, estado, info, err = fut.result()
                if estado == "ok" and info is not None:
                    print(
                        f"  [{tarea.instancia}] f_aum={tarea.factor_aumento:.2f} "
                        f"umb={tarea.umbral_escape} rep={tarea.repeticion} "
                        f"| costo={info['costo']:.4f} "
                        f"| mejora={info['mejora']:.2f}% "
                        f"| iter={info['iter']} "
                        f"| reps={info['reps']} "
                        f"| esc={info['esc']} "
                        f"| tenure[{info['ten_min']}-{info['ten_max']}] "
                        f"| t={info['tiempo']:.2f}s"
                    )
                    total_ok += 1
                else:
                    print(
                        f"  [{tarea.instancia}] f_aum={tarea.factor_aumento:.2f} "
                        f"umb={tarea.umbral_escape} rep={tarea.repeticion} | FAIL: {err}"
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
