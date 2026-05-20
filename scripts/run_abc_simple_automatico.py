"""
Corrida ABC SIMPLE (Karaboga 2005 canónico) para instancias seleccionadas.

Versión SIMPLE de Artificial Bee Colony:
- Scouts puramente aleatorios (NO vecinos de la mejor).
- Sesgo inter/intra-ruta en empleadas y observadoras (mismo helper que SA/TS/RTS).
- Corrección del bug de imputación de ``registrar_mejora`` al operador real.
- Criterio de parada por estancamiento (siempre activo, calibrado por n_tareas).
- ``alpha_inter`` ELIMINADO: el algoritmo eleva automáticamente P(inter) a
  ``max(p_inter, 0.8)`` en presencia de violación de capacidad. El usuario
  solo configura ``p_inter`` (la probabilidad cuando la solución es factible).

Configuración (grid search 4D sobre FACTORES DE ESCALA en función de n_tareas):

Grid (4 dimensiones, factores de escala en función de n_tareas):
    factor_fuentes   ∈ {1.5, 2.0, 3.0, 4.0}   → num_fuentes   = max(10, round(f·√n))
    factor_abandono  ∈ {0.25, 0.5, 0.75, 1.0}  → lim_abandono  = max(15, round(f·n))
    p_inter          ∈ {0.4, 0.5, 0.6, 0.7, 0.8}
    factor_iter      ∈ {15, 20, 30}             → iteraciones  = max(200, f·n)

Total: 4 × 4 × 5 × 3 = 240 combos × 23 instancias × 5 reps = 27,600 corridas.

POR QUÉ FACTORES Y NO VALORES ABSOLUTOS:
El grid anterior usaba valores ABSOLUTOS (num_fuentes ∈ {10,20,30}, etc.) que
ignoraban el tamaño de la instancia. Una corrida con 20 fuentes sobre una
instancia pequeña (gdb1, n=22) explora el espacio de forma muy distinta que
sobre una grande (gdb20, n=...). Con factores de escala, el algoritmo
asigna recursos PROPORCIONALES a cada instancia, lo que hace mucho más
comparables los resultados entre instancias y mucho más limpia la lectura
del grid (un factor "óptimo" detectado en instancias pequeñas también es
"óptimo" en grandes, salvo efectos no lineales que ahora se aprecian sin
ruido del tamaño absoluto).

``max_iter_sin_mejora`` no se barre en el grid: se deja en ``None`` y la
función ``busqueda_abejas_simple`` lo calibra a ``max(50, 3·n_tareas)``. Esto
ya da un criterio de parada anticipada bien dimensionado para cada
instancia.

NOTA SOBRE LA AUSENCIA DE SEMILLA FIJA:
Las repeticiones EXISTEN para medir la robustez de cada configuración a la
aleatoriedad inherente de la metaheurística (siembra inicial de fuentes,
selección de vecinos por ruleta, scouts aleatorios). Si fijáramos una semilla
determinista por tarea, las repeticiones de cada combinación serían IDÉNTICAS
— no aportarían información estadística. Una buena configuración debe
converger a buenos óptimos SIN IMPORTAR la trayectoria aleatoria que tome.
Por eso aquí cada corrida muestrea libremente del espacio aleatorio del
sistema (igual decisión que en RTS y TS simple).

Para REDUCIR el grid, basta con pasar override single-value por CLI:
    --p-inter 0.6 → fija p_inter, solo barre el resto.

Paralelismo (--workers N)
-------------------------
Cada combinación (instancia, factor_fuentes, factor_abandono, p_inter,
factor_iter, repetición) es una tarea INDEPENDIENTE: no comparte estado con
las demás y solo lee la instancia desde disco. Esto las hace trivialmente
paralelizables con ``ProcessPoolExecutor``. El flag ``--workers N`` controla
cuántos procesos se lanzan en paralelo:
    --workers 1   → ejecución secuencial (útil para depurar y porque stdout
                    no se mezcla entre tareas).
    --workers N   → N procesos en paralelo (default: os.cpu_count()).

Reproducibilidad:
    NO hay reproducibilidad bit-a-bit por diseño (ver nota arriba sobre la
    ausencia de semilla fija). Sin embargo, sí hay reproducibilidad
    ESTADÍSTICA: con 5 repeticiones por configuración, las medias y
    desviaciones de costo, mejora, número de scouts, etc. convergen a
    valores estables que sí son comparables entre experimentos.

Concurrencia de E/S:
    Para evitar carreras al escribir el CSV, cada worker escribe su fila en
    un CSV TEMPORAL único (por PID + índice de tarea) dentro de una
    subcarpeta ``_partials/``. Al terminar todas las tareas, el proceso
    principal CONCATENA los parciales por instancia en el CSV final
    canónico. Sin locks, sin colas, sin riesgo de filas truncadas.

Uso:
    python scripts/run_abc_simple_automatico.py
    python scripts/run_abc_simple_automatico.py --salida-dir experimentos/abc_simple_grid
    python scripts/run_abc_simple_automatico.py --repeticiones 5
    python scripts/run_abc_simple_automatico.py --workers 8
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
from metacarp import busqueda_abejas_simple_desde_instancia

# Mismo conjunto de instancias pequeñas que usan los scripts de SA / TS / RTS,
# para mantener la comparabilidad de los experimentos (gdb + kshs).
INSTANCIAS = [
    "gdb19", "kshs1", "kshs2", "kshs3", "kshs4", "kshs5", "kshs6",
    "gdb4",  "gdb14", "gdb15", "gdb1",  "gdb20", "gdb3",  "gdb6",
    "gdb7",  "gdb12", "gdb10", "gdb2",  "gdb5",  "gdb13", "gdb16",
    "gdb17", "gdb21",
]

# Grid search 4D sobre FACTORES DE ESCALA en función de n_tareas:
#
# - factor_fuentes: el algoritmo calcula num_fuentes = max(10, round(f·√n)).
#   Barrer de 1.5 a 4.0 cubre desde poblaciones pequeñas (~1.5√n ≈ 7 para
#   gdb1) hasta poblaciones grandes (~4√n ≈ 19 para gdb1). La heurística
#   típica en ABC es proporcional a √n; aquí abarcamos un rango amplio
#   para detectar el punto óptimo.
#
# - factor_abandono: limite_abandono = max(15, round(f·n_tareas)). Cuanto
#   mayor el factor, menos rápido se manda una fuente al scout (más
#   intensificación, menos diversificación). 0.25 manda al scout muy
#   pronto (alta diversificación); 1.0 espera un ciclo "n" antes de
#   reiniciar.
#
# - p_inter: probabilidad de elegir el grupo INTER-RUTA cuando la solución
#   actual es factible. Bajo violación, el algoritmo eleva a max(p_inter, 0.8)
#   automáticamente. Mismo rango que en RTS para comparabilidad.
#
# - factor_iter: iteraciones = max(200, f·n_tareas). Tres puntos (15, 20, 30)
#   alineados con las heurísticas de SA y RTS instance-aware.
FACTORES_FUENTES   = [1.5, 2.0, 3.0, 4.0]
FACTORES_ABANDONO  = [0.25, 0.5, 0.75, 1.0]
P_INTERS           = [0.4, 0.5, 0.6, 0.7, 0.8]
FACTORES_ITER      = [15, 20, 30]


# --- DATACLASS DE TAREA ---
# Encapsula todos los parámetros de UNA corrida. Es un objeto plano,
# pickle-friendly, que se envía a un proceso worker. Mantener todos los
# parámetros en un solo objeto evita firmas con muchos argumentos y facilita
# extender el grid en el futuro.
@dataclass(frozen=True)
class TareaABC:
    instancia: str
    # Sustituye al ``num_fuentes`` absoluto del grid anterior: el algoritmo
    # calcula num_fuentes = max(10, round(factor_fuentes · √n_tareas)).
    factor_fuentes: float
    # Sustituye al ``limite_abandono`` absoluto: lim = max(15, round(factor · n)).
    factor_abandono: float
    p_inter: float
    # Sustituye al ``iteraciones`` absoluto: iter = max(200, factor · n).
    factor_iter: int
    repeticion: int
    # NOTA: NO hay campo ``semilla``. Cada corrida se ejecuta con semilla
    # aleatoria del sistema (la metaheurística internamente crea su propio
    # ``random.Random()``). Esto es deliberado: las "repeticiones" SOLO son
    # informativas si cada una muestrea una trayectoria distinta del espacio
    # aleatorio. Una semilla determinista por tarea las convertiría en
    # mediciones idénticas, sin valor estadístico.
    #
    # NOTA: ``max_iter_sin_mejora`` no se barre. Se deja en None y el
    # algoritmo lo calibra a max(50, 3·n_tareas) — criterio instance-aware.
    usar_gpu: bool                            # si True, evaluación en lote usa GPU (fase observadoras)
    root: str | None
    ruta_csv_parcial: str                     # cada tarea escribe a su propio archivo


def _ejecutar_tarea(tarea: TareaABC) -> tuple[TareaABC, str, dict | None, str | None]:
    """
    Ejecuta UNA corrida de ABC simple en el worker.

    Esta función se ejecuta dentro de un PROCESO HIJO. Recibe la tarea,
    invoca el algoritmo y devuelve una tupla con el estado del resultado.
    No imprime: el proceso principal formatea y muestra el resumen al
    recibir cada futuro completado.

    El script pasa los FACTORES de escala (no los valores absolutos): el
    algoritmo internamente convierte cada factor a un valor concreto en
    función de ``n_tareas`` de la instancia. Esto mantiene la comparabilidad
    entre instancias de distinto tamaño.

    Returns
    -------
    (tarea, "ok"|"fail", info_dict_o_None, mensaje_error_o_None)
    """
    try:
        res = busqueda_abejas_simple_desde_instancia(
            tarea.instancia,
            # Factores de escala: el algoritmo los traduce a valores concretos
            # usando n_tareas de la instancia. La precedencia es:
            # valor absoluto > factor > default por fórmula. Aquí no pasamos
            # ningún valor absoluto, así que mandan los factores.
            factor_fuentes=tarea.factor_fuentes,
            factor_abandono=tarea.factor_abandono,
            factor_iter=tarea.factor_iter,
            p_inter=tarea.p_inter,
            usar_gpu=tarea.usar_gpu,
            # semilla=None: cada repetición usa una trayectoria aleatoria
            # distinta del sistema. Esto es lo que permite que las
            # repeticiones tengan valor estadístico.
            semilla=None,
            repeticion=tarea.repeticion,
            guardar_csv=True,
            ruta_csv=tarea.ruta_csv_parcial,
            guardar_historial=False,
            root=tarea.root,
        )
        info = {
            "costo": res.mejor_costo,
            "mejora": res.mejora_porcentaje_inicial_vs_final,
            "iter": res.iteraciones_totales,
            "scouts": res.scouts_reinicios,
            "mejoras": res.mejoras,
            "iter_sin_mejora": res.iteraciones_sin_mejora_final,
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
    instancia. Misma estrategia map-reduce que el script de RTS:
    cada worker escribe a su parcial, el principal fusiona al final.

    Devuelve el número de archivos finales generados.
    """
    grupos: dict[str, list[Path]] = {}
    for parcial in sorted(dir_parciales.glob("abc_simple_*.csv")):
        # Nombre con formato ``abc_simple_<instancia>_<pid>_<idx>.csv``.
        partes = parcial.stem.split("_")
        if len(partes) < 4:
            continue
        # ``abc_simple`` ocupa partes[0] y partes[1]; la instancia es partes[2].
        instancia = partes[2]
        grupos.setdefault(instancia, []).append(parcial)

    n_archivos_creados = 0
    for instancia, archivos in grupos.items():
        ruta_final = salida_dir / f"abc_simple_{instancia}_{experimento}_{ydmh}.csv"
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
        description=(
            "ABC Simple (Karaboga 2005) — grid 4D sobre factores de escala "
            "(factor_fuentes, factor_abandono, p_inter, factor_iter)."
        )
    )
    parser.add_argument("--salida-dir",   type=str, default="experimentos")
    parser.add_argument("--repeticiones", type=int, default=5)
    parser.add_argument("--experimento",  type=str, default="abc_simple_auto")
    parser.add_argument("--root",         type=str, default=None)
    # Overrides opcionales que FIJAN una dimensión a un solo valor (reducen
    # el grid). Si no se pasan, se barre la lista completa de esa dimensión.
    parser.add_argument("--p-inter", type=float, default=None,
                        help="P(elegir inter) cuando la sol. actual es factible. "
                             "None = barrer lista [0.4, 0.5, 0.6, 0.7, 0.8].")
    parser.add_argument("--factor-fuentes", type=float, default=None,
                        help="Fija el factor de escala de num_fuentes. "
                             "None = barrer lista [1.5, 2.0, 3.0, 4.0].")
    parser.add_argument("--factor-abandono", type=float, default=None,
                        help="Fija el factor de escala de limite_abandono. "
                             "None = barrer lista [0.25, 0.5, 0.75, 1.0].")
    parser.add_argument("--factor-iter", type=int, default=None,
                        help="Fija el factor de escala de iteraciones. "
                             "None = barrer lista [15, 20, 30].")
    # --usar-gpu activa la evaluación en lote con GPU (CuPy) en la fase
    # observadoras. Solo aplica si CuPy está instalado y hay GPU disponible;
    # si no, el algoritmo cae silenciosamente a CPU. Para instancias small,
    # el speedup es marginal pero no introduce errores.
    parser.add_argument(
        "--usar-gpu",
        action="store_true",
        default=False,
        help="Activa evaluación en lote con GPU (CuPy) en fase observadoras.",
    )
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
    # metaheurística (ver nota en la dataclass ``TareaABC``).
    return parser.parse_args()


def main() -> None:
    """Punto de entrada principal del script de experimentos."""
    args = _parse_args()
    # ``--salida-dir`` apunta DIRECTAMENTE al folder final donde se escriben
    # los CSV consolidados (un archivo por instancia).
    salida_dir = Path(args.salida_dir).expanduser().resolve()
    salida_dir.mkdir(parents=True, exist_ok=True)
    # Subcarpeta para los CSV parciales de los workers (siempre, aunque sea modo secuencial).
    dir_parciales = salida_dir / "_partials"
    dir_parciales.mkdir(parents=True, exist_ok=True)
    # Marca de tiempo: año-día-mes-hora-minuto, mismo formato que SA / TS / RTS.
    ydmh = datetime.now().strftime("%Y%d%m%H%M")

    # --- RESOLUCIÓN DE LAS DIMENSIONES DEL GRID ---
    # Si el usuario pasa un override single-value por CLI (--p-inter 0.6),
    # ese valor SUSTITUYE a la lista completa del grid: la dimensión
    # correspondiente queda fijada a un solo punto.
    p_inter_values        = [args.p_inter]        if args.p_inter        is not None else P_INTERS
    factor_fuentes_values = [args.factor_fuentes] if args.factor_fuentes is not None else FACTORES_FUENTES
    factor_abandono_values = [args.factor_abandono] if args.factor_abandono is not None else FACTORES_ABANDONO
    factor_iter_values    = [args.factor_iter]    if args.factor_iter    is not None else FACTORES_ITER

    # --- CONSTRUCCIÓN DE LA LISTA DE TAREAS ---
    # Una tarea por (instancia, factor_fuentes, factor_abandono, p_inter,
    # factor_iter, repeticion). Cada tarea es independiente: solo lectura
    # de la instancia desde disco. Esto las hace trivialmente paralelizables
    # con ProcessPoolExecutor.
    tareas: list[TareaABC] = []
    for instancia in INSTANCIAS:
        for ff in factor_fuentes_values:
            for fa in factor_abandono_values:
                for p_int in p_inter_values:
                    for fi in factor_iter_values:
                        for rep in range(1, args.repeticiones + 1):
                            idx = len(tareas)
                            parcial = dir_parciales / (
                                f"abc_simple_{instancia}_{os.getpid()}_{idx}.csv"
                            )
                            tareas.append(TareaABC(
                                instancia=instancia,
                                factor_fuentes=ff,
                                factor_abandono=fa,
                                p_inter=p_int,
                                factor_iter=fi,
                                repeticion=rep,
                                usar_gpu=args.usar_gpu,
                                root=args.root,
                                ruta_csv_parcial=str(parcial),
                            ))

    total = len(tareas)
    print("=" * 80)
    print("ABC Simple (Karaboga 2005 canónico) — grid search 4D (factores de escala)")
    print("=" * 80)
    print(f"Instancias                  : {len(INSTANCIAS)}")
    print(f"factor_fuentes values       : {factor_fuentes_values}  → num_fuentes  = max(10, round(f·√n))")
    print(f"factor_abandono values      : {factor_abandono_values}  → lim_abandono = max(15, round(f·n))")
    print(f"p_inter values              : {p_inter_values}")
    print(f"factor_iter values          : {factor_iter_values}  → iteraciones  = max(200, f·n)")
    print(f"GPU (fase observadoras)     : {'activada' if args.usar_gpu else 'desactivada (CPU)'}")
    print(f"max_iter_sin_mejora         : instance-aware (max(50, 3·n) dentro del algoritmo)")
    print(f"alpha_inter                 : ELIMINADO (piso 0.8 automático bajo violación)")
    print(f"Semilla                     : aleatoria del sistema (cada repetición es independiente)")
    print(f"Repeticiones                : {args.repeticiones}")
    print(f"Workers                     : {args.workers}")
    print(f"Corridas                    : {total}")
    print(f"Salida CSV                  : {salida_dir}")
    print("-" * 80)

    total_ok = 0
    total_fail = 0

    # --- EJECUCIÓN ---
    # Misma estructura que el script RTS: rama secuencial explícita cuando
    # --workers <= 1, rama paralela con ProcessPoolExecutor en otro caso.
    if args.workers <= 1:
        for tarea in tareas:
            _, estado, info, err = _ejecutar_tarea(tarea)
            if estado == "ok" and info is not None:
                # Print con los FACTORES (no valores absolutos): así el lector
                # puede comparar entre instancias sin tener que conocer n_tareas.
                print(
                    f"  [{tarea.instancia}] "
                    f"ff={tarea.factor_fuentes:.2f} "
                    f"fa={tarea.factor_abandono:.2f} "
                    f"pInt={tarea.p_inter:.1f} "
                    f"fi={tarea.factor_iter} "
                    f"rep={tarea.repeticion} "
                    f"| costo={info['costo']:.4f} "
                    f"| mejora={info['mejora']:.2f}% "
                    f"| iter={info['iter']} "
                    f"| scouts={info['scouts']} "
                    f"| t={info['tiempo']:.2f}s"
                )
                total_ok += 1
            else:
                print(
                    f"  [{tarea.instancia}] "
                    f"ff={tarea.factor_fuentes:.2f} "
                    f"fa={tarea.factor_abandono:.2f} "
                    f"pInt={tarea.p_inter:.1f} "
                    f"fi={tarea.factor_iter} "
                    f"rep={tarea.repeticion} | FAIL: {err}"
                )
                total_fail += 1
    else:
        # MODO PARALELO con ProcessPoolExecutor.
        # Mismas garantías que en el script de RTS:
        #   1. Cada tarea es función pura (sin estado compartido).
        #   2. Cada tarea escribe a su PROPIO CSV parcial.
        #   3. Cada tarea es independiente (no comparte semilla con las demás).
        # ``as_completed`` reporta cada tarea EN CUANTO TERMINA, no en el
        # orden de envío, para que stdout muestre progreso fluido sin
        # esperar a las corridas más lentas.
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futuros = {executor.submit(_ejecutar_tarea, t): t for t in tareas}
            for fut in as_completed(futuros):
                tarea, estado, info, err = fut.result()
                if estado == "ok" and info is not None:
                    print(
                        f"  [{tarea.instancia}] "
                        f"ff={tarea.factor_fuentes:.2f} "
                        f"fa={tarea.factor_abandono:.2f} "
                        f"pInt={tarea.p_inter:.1f} "
                        f"fi={tarea.factor_iter} "
                        f"rep={tarea.repeticion} "
                        f"| costo={info['costo']:.4f} "
                        f"| mejora={info['mejora']:.2f}% "
                        f"| iter={info['iter']} "
                        f"| scouts={info['scouts']} "
                        f"| t={info['tiempo']:.2f}s"
                    )
                    total_ok += 1
                else:
                    print(
                        f"  [{tarea.instancia}] "
                        f"ff={tarea.factor_fuentes:.2f} "
                        f"fa={tarea.factor_abandono:.2f} "
                        f"pInt={tarea.p_inter:.1f} "
                        f"fi={tarea.factor_iter} "
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
