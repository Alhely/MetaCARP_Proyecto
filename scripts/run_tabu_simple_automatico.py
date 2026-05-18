"""
Corrida Búsqueda Tabú SIMPLE para instancias seleccionadas.

Versión más simple de TS (lista FIFO, best-non-tabú, aspiración clásica).
Pensada como referencia didáctica de TS para la tesis. No incluye:
movimientos compuestos, memoria por frecuencia, intensificación/diversificación,
path relinking, etc.

Configuración (parámetros que se barren):
    iteraciones_max      = 400      →  cota dura de iteraciones (igual al TS original)
    max_iter_sin_mejora  = 100      →  parada por estancamiento
    tam_vecindario       = 25       →  vecinos por iteración (best-improvement sobre el lote)
    tabu_tenure          ∈ {5,10,15,20,25,30}  →  6 valores (regla ≈ sqrt(n))
    operadores           = OPERADORES_POPULARES (9 operadores)
    semilla              = aleatoria (None)
    repeticiones         = 2 (configurable por CLI)

Total: 6 tenures × 23 instancias × 2 repeticiones = 276 corridas.

Paralelismo (--workers N)
-------------------------
Cada combinación (instancia, tenure, repetición) es una tarea INDEPENDIENTE:
no comparte estado con las demás y solo lee la instancia desde disco. Esto
las hace trivialmente paralelizables con ``ProcessPoolExecutor``. El flag
``--workers N`` controla cuántos procesos se lanzan en paralelo:
    --workers 1   → ejecución secuencial (modo reproducible "clásico"; útil
                    para debugging porque la salida por stdout no se mezcla).
    --workers N   → N procesos en paralelo (default: os.cpu_count()).

Reproducibilidad de cada corrida:
    Cada tarea recibe su propia ``semilla`` determinista derivada de
    ``(instancia, tenure, repeticion)``. Aunque el ORDEN de finalización
    en paralelo no sea determinista, el RESULTADO numérico de cada corrida
    individual sí lo es porque depende solo de su semilla. Si se pasa una
    ``--semilla-base`` distinta, todas las semillas derivadas cambian de
    forma coherente.

Concurrencia de E/S:
    Para evitar carreras al escribir el CSV, cada worker escribe su fila en
    un CSV TEMPORAL único (por PID + índice de tarea) dentro de una
    subcarpeta ``_partials/``. Al terminar TODAS las tareas, el proceso
    principal CONCATENA los parciales por instancia en el CSV final
    canónico (un archivo por instancia, mismo nombre que la versión
    secuencial). Esto es la opción más sencilla y robusta: sin locks, sin
    cola compartida, sin riesgo de filas truncadas.

Uso:
    python scripts/run_tabu_simple_automatico.py
    python scripts/run_tabu_simple_automatico.py --salida-dir resultados_tabu_simple
    python scripts/run_tabu_simple_automatico.py --repeticiones 3
    python scripts/run_tabu_simple_automatico.py --workers 1   # secuencial
    python scripts/run_tabu_simple_automatico.py --workers 8   # 8 procesos
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
from metacarp import busqueda_tabu_simple_desde_instancia

# Mismo conjunto de instancias pequeñas que usa el script de SA, para mantener
# la comparabilidad de los experimentos (gdb + kshs).
INSTANCIAS = [
    "gdb19", "kshs1", "kshs2", "kshs3", "kshs4", "kshs5", "kshs6",
    "gdb4",  "gdb14", "gdb15", "gdb1",  "gdb20", "gdb3",  "gdb6",
    "gdb7",  "gdb12", "gdb10", "gdb2",  "gdb5",  "gdb13", "gdb16",
    "gdb17", "gdb21",
]

# Valores de tenure tabú a explorar: barrido pequeño alrededor del rango clásico
# (≈ sqrt(n) en problemas combinatorios). Tenure muy chico → poca memoria,
# riesgo de ciclos; muy grande → exceso de prohibiciones, búsqueda lenta.
TABU_TENURES = [5, 10, 15, 20, 25, 30]


# --- DATACLASS DE TAREA ---
# Encapsulamos todos los parámetros necesarios para ejecutar UNA corrida
# completa. Es una clase de DATOS plana, serializable (pickle-friendly), que
# se envía al proceso worker a través del ProcessPoolExecutor. Mantener todos
# los parámetros en un único objeto evita firmas de función con 10+ argumentos
# y facilita extender el script en el futuro (basta con añadir un atributo).
@dataclass(frozen=True)
class TareaTS:
    instancia: str
    tenure: int
    repeticion: int
    semilla: int                          # semilla determinista por tarea
    iteraciones_max: int
    max_iter_sin_mejora: int
    tam_vecindario: int
    alpha_inter: float | None
    p_inter: float | None
    root: str | None
    ruta_csv_parcial: str                 # cada tarea escribe a su propio archivo


def _derivar_semilla(base: int, instancia: str, tenure: int, repeticion: int) -> int:
    """
    Calcula una semilla determinista para una corrida concreta.

    ¿Por qué un hash y no un contador global?
    - Un contador global obligaría a serializar la enumeración antes del
      paralelismo y rompería la reproducibilidad si se cambia el orden de
      lanzamiento. Usando un hash de ``(base, instancia, tenure, repeticion)``
      cada tarea es REPRODUCIBLE de forma independiente sin importar el
      orden en que se ejecute.
    - Limitamos el hash a 31 bits para evitar problemas con random.Random,
      que internamente acepta enteros pero algunos backends esperan ints
      no negativos de tamaño "razonable".
    """
    # hash() de Python varía entre procesos por PYTHONHASHSEED; por eso
    # construimos el entero a mano con operaciones determinísticas que no
    # dependen del estado de hash de strings.
    h = base
    for ch in instancia:
        h = (h * 1000003) ^ ord(ch)
    h = (h * 1000003) ^ int(tenure)
    h = (h * 1000003) ^ int(repeticion)
    return h & 0x7FFFFFFF  # mantenemos solo los 31 bits inferiores


def _ejecutar_tarea(tarea: TareaTS) -> tuple[TareaTS, str, dict | None, str | None]:
    """
    Ejecuta UNA corrida de TS simple en el worker.

    Esta función se ejecuta dentro de un PROCESO HIJO. Recibe la tarea,
    invoca el algoritmo y devuelve una tupla con el estado del resultado.
    No imprime ni escribe a stdout: el proceso principal se encarga de
    formatear y mostrar el resumen al recibir cada futuro completado.

    Returns
    -------
    (tarea, "ok"|"fail", info_dict_o_None, mensaje_error_o_None)
    """
    try:
        res = busqueda_tabu_simple_desde_instancia(
            tarea.instancia,
            iteraciones_max=tarea.iteraciones_max,
            max_iter_sin_mejora=tarea.max_iter_sin_mejora,
            tam_vecindario=tarea.tam_vecindario,
            tabu_tenure=tarea.tenure,
            semilla=tarea.semilla,
            repeticion=tarea.repeticion,
            guardar_csv=True,
            ruta_csv=tarea.ruta_csv_parcial,
            guardar_historial=False,
            root=tarea.root,
            alpha_inter=tarea.alpha_inter,
            p_inter=tarea.p_inter,
        )
        # Devolvemos un dict mínimo con la info que el principal imprimirá.
        info = {
            "costo": res.mejor_costo,
            "mejora": res.mejora_porcentaje_inicial_vs_final,
            "iter": res.iteraciones_totales,
            "tiempo": res.tiempo_segundos,
        }
        return (tarea, "ok", info, None)
    except Exception as exc:  # noqa: BLE001
        # Capturamos cualquier excepción para no abortar el grid completo.
        # El proceso principal usará el mensaje de error para reportar.
        return (tarea, "fail", None, f"{type(exc).__name__}: {exc}")


def _consolidar_parciales(
    dir_parciales: Path,
    salida_dir: Path,
    experimento: str,
    ydmh: str,
) -> int:
    """
    Concatena los CSVs parciales por instancia en un único CSV final por
    instancia. Esta es la "etapa de fusión" del patrón map-reduce simple
    que usamos para evitar locks de escritura.

    Estrategia: agrupar los parciales por nombre de instancia (extraído del
    prefijo del nombre de archivo) y escribir un encabezado único seguido
    de todas las filas. Si dos parciales tuvieran columnas distintas (no
    debería suceder porque todos provienen de la misma metaheurística),
    se unen tomando la unión de columnas.

    Devuelve el número de archivos finales generados.
    """
    # Agrupamos los parciales por instancia. El nombre tiene el formato
    # ``tabu_simple_<instancia>_<pid>_<idx>.csv`` (ver lanzamiento abajo).
    grupos: dict[str, list[Path]] = {}
    for parcial in sorted(dir_parciales.glob("tabu_simple_*.csv")):
        # Extraemos la instancia del nombre del archivo (tercer campo
        # separado por '_' desde el inicio).
        partes = parcial.stem.split("_")
        # ``tabu_simple_gdb1_12345_3`` -> partes = ['tabu', 'simple', 'gdb1', '12345', '3']
        if len(partes) < 4:
            continue
        instancia = partes[2]
        grupos.setdefault(instancia, []).append(parcial)

    n_archivos_creados = 0
    for instancia, archivos in grupos.items():
        # Nombre canónico del CSV final, MISMO formato que el modo secuencial.
        ruta_final = salida_dir / f"tabu_simple_{instancia}_{experimento}_{ydmh}.csv"
        # Leemos todas las filas de todos los parciales y las escribimos
        # en orden de tarea (por nombre de archivo, que ya viene ordenado).
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
        # Escribimos el CSV consolidado.
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
        description="TS simple grid search sobre tabu_tenure."
    )
    parser.add_argument("--salida-dir",   type=str, default="experimentos")
    parser.add_argument("--repeticiones", type=int, default=2)
    parser.add_argument("--experimento",  type=str, default="tabu_simple_auto")
    parser.add_argument("--root",         type=str, default=None)
    parser.add_argument("--iteraciones-max",     type=int, default=400)
    parser.add_argument("--max-iter-sin-mejora", type=int, default=100)
    parser.add_argument("--tam-vecindario",      type=int, default=25)
    # Parámetros del sesgo inter/intra (compatibilidad con SA y RTS).
    # Default None = usar el valor canónico de SA (alpha_inter=0.8, p_inter=0.6).
    # Permitimos override por CLI por si se desea explorar otros valores en
    # experimentos posteriores, pero NO los incluimos en el grid actual:
    # el experimento base estudia tabu_tenure aislado, manteniendo el sesgo
    # igual al de SA para garantizar comparabilidad directa entre algoritmos.
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
    # --semilla-base permite reproducir EXACTAMENTE un grid completo: con la
    # misma semilla base, todas las semillas derivadas son iguales y por tanto
    # cada corrida individual da el mismo resultado numérico. Default = 0 para
    # determinismo por defecto. Si se desea aleatoriedad real, basta con pasar
    # ``--semilla-base $(date +%s)`` u otro valor distinto en cada ejecución.
    parser.add_argument(
        "--semilla-base",
        type=int,
        default=0,
        help="Semilla raíz usada para derivar semillas deterministas por tarea.",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada principal del script de experimentos."""
    args = _parse_args()
    # Subcarpeta dedicada para distinguir resultados del TS simple del original.
    salida_dir = Path(args.salida_dir).expanduser().resolve() / "tabu_simple_small_20260517"
    salida_dir.mkdir(parents=True, exist_ok=True)
    # Subcarpeta para los CSV parciales de los workers. Se crea siempre, aunque
    # solo se use cuando hay paralelismo (workers > 1). En modo secuencial
    # también la usamos para mantener una única ruta de código (más simple).
    dir_parciales = salida_dir / "_partials"
    dir_parciales.mkdir(parents=True, exist_ok=True)
    # Marca de tiempo: año-día-mes-hora-minuto, mismo formato que SA.
    ydmh = datetime.now().strftime("%Y%d%m%H%M")

    # --- CONSTRUCCIÓN DE LA LISTA DE TAREAS ---
    # Generamos UNA tarea por (instancia, tenure, repeticion). Cada tarea es
    # un objeto pequeño, serializable, y completamente independiente: el
    # worker no necesita estado compartido para ejecutarla.
    tareas: list[TareaTS] = []
    for instancia in INSTANCIAS:
        for tenure in TABU_TENURES:
            for rep in range(1, args.repeticiones + 1):
                # Cada tarea tiene su propio CSV parcial. El nombre incluye
                # un índice creciente para garantizar unicidad incluso si dos
                # workers con el mismo PID terminan corriendo (no debería
                # pasar, pero el índice añade una capa de seguridad).
                idx = len(tareas)
                parcial = dir_parciales / (
                    f"tabu_simple_{instancia}_{os.getpid()}_{idx}.csv"
                )
                tareas.append(TareaTS(
                    instancia=instancia,
                    tenure=tenure,
                    repeticion=rep,
                    semilla=_derivar_semilla(args.semilla_base, instancia, tenure, rep),
                    iteraciones_max=args.iteraciones_max,
                    max_iter_sin_mejora=args.max_iter_sin_mejora,
                    tam_vecindario=args.tam_vecindario,
                    alpha_inter=args.alpha_inter,
                    p_inter=args.p_inter,
                    root=args.root,
                    ruta_csv_parcial=str(parcial),
                ))

    total = len(tareas)
    print("=" * 80)
    print("Búsqueda Tabú SIMPLE  —  grid search tabu_tenure")
    print("=" * 80)
    print(f"Instancias                  : {len(INSTANCIAS)}")
    print(f"iteraciones_max             : {args.iteraciones_max}")
    print(f"max_iter_sin_mejora         : {args.max_iter_sin_mejora}")
    print(f"tam_vecindario              : {args.tam_vecindario}")
    print(f"tabu_tenure values          : {TABU_TENURES}")
    print(f"Semilla base                : {args.semilla_base} (derivada por tarea)")
    print(f"Repeticiones                : {args.repeticiones}")
    print(f"Workers                     : {args.workers}")
    print(f"Corridas                    : {total}")
    print(f"Salida CSV                  : {salida_dir}")
    print("-" * 80)

    total_ok = 0
    total_fail = 0

    # --- EJECUCIÓN ---
    # Decidimos secuencial vs paralelo según --workers. Mantenemos la rama
    # secuencial explícita (no usamos ProcessPoolExecutor con max_workers=1)
    # para que el debugging sea más fácil: sin procesos hijo, los tracebacks
    # aparecen directamente y se puede usar pdb.set_trace() sin trucos.
    if args.workers <= 1:
        for tarea in tareas:
            _, estado, info, err = _ejecutar_tarea(tarea)
            if estado == "ok" and info is not None:
                print(
                    f"  [{tarea.instancia}] tenure={tarea.tenure:>3d} rep={tarea.repeticion} "
                    f"| costo={info['costo']:.4f} "
                    f"| mejora={info['mejora']:.2f}% "
                    f"| iter={info['iter']} "
                    f"| t={info['tiempo']:.2f}s"
                )
                total_ok += 1
            else:
                print(
                    f"  [{tarea.instancia}] tenure={tarea.tenure:>3d} rep={tarea.repeticion} "
                    f"| FAIL: {err}"
                )
                total_fail += 1
    else:
        # MODO PARALELO con ProcessPoolExecutor.
        # ProcessPoolExecutor crea ``args.workers`` procesos Python hijos.
        # ``executor.submit(fn, arg)`` envía una tarea y devuelve un Future.
        # ``as_completed(futuros)`` itera los futuros EN EL ORDEN EN QUE
        # TERMINAN (no en el orden en que se enviaron), lo cual es ideal
        # para reportar progreso en tiempo real sin esperar a los más lentos.
        # El paralelismo es seguro aquí porque:
        #   1. Cada tarea es una función pura (no hay estado global mutable
        #      compartido entre tareas).
        #   2. Cada tarea escribe a su PROPIO archivo CSV parcial (no hay
        #      contención de E/S).
        #   3. La semilla de cada tarea es determinista (no depende del
        #      orden de ejecución).
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futuros = {executor.submit(_ejecutar_tarea, t): t for t in tareas}
            for fut in as_completed(futuros):
                tarea, estado, info, err = fut.result()
                if estado == "ok" and info is not None:
                    print(
                        f"  [{tarea.instancia}] tenure={tarea.tenure:>3d} rep={tarea.repeticion} "
                        f"| costo={info['costo']:.4f} "
                        f"| mejora={info['mejora']:.2f}% "
                        f"| iter={info['iter']} "
                        f"| t={info['tiempo']:.2f}s"
                    )
                    total_ok += 1
                else:
                    print(
                        f"  [{tarea.instancia}] tenure={tarea.tenure:>3d} rep={tarea.repeticion} "
                        f"| FAIL: {err}"
                    )
                    total_fail += 1

    # --- FUSIÓN DE PARCIALES ---
    # Tras terminar todas las tareas, consolidamos los CSVs parciales en los
    # CSV finales canónicos (uno por instancia). Esto preserva la estructura
    # de archivos del modo secuencial original, manteniendo compatibilidad
    # con notebooks/scripts de análisis que esperan esa convención.
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
