"""
Corrida Vibration Damping Optimization (VDO, Mehdizadeh & Tavakkoli-Moghaddam
2008/2009 adaptado a CARP) para instancias seleccionadas.

Versión instance-aware de VDO alineada con SA / TS simple / RTS / ABC simple
/ Cuckoo Search:
- Parámetros calibrados automáticamente por ``n_tareas`` (A0, umbral mínimo,
  σ y L se derivan de d_max y n cuando el usuario los deja en ``None``).
- Sesgo inter/intra-ruta vía ``seleccionar_grupo_operadores_inter_intra``
  (mismo helper que SA / TS / RTS / ABC simple / Cuckoo).
- Ley de amortiguamiento canónica del oscilador amortiguado:
      A_{t+1} = A0 · exp(−γ · t / 2)
- Regla de aceptación basada en la CDF de Rayleigh:
      p = 1 − exp(−A² / (2·σ²))

Configuración (grid search 3D sobre parámetros vibratorios):

Grid (3 dimensiones):
    gamma        ∈ {0.02, 0.05, 0.1, 0.2}      → coeficiente de amortiguamiento
    factor_sigma ∈ {0.25, 0.5, 1.0, 2.0}       → σ = factor_sigma · A0_eff
    p_inter      ∈ {0.4, 0.5, 0.6, 0.7, 0.8}    → P(inter) en régimen factible

Total: 4 × 4 × 5 = 80 combos × 23 instancias × 5 reps = 9,200 corridas.

POR QUÉ ESTAS DIMENSIONES:
Los parámetros más influyentes de VDO son el par (γ, σ):
- γ controla la RAPIDEZ del decaimiento (mayor γ ⇒ menos niveles ⇒ menos
  exploración total). Es una constante adimensional, por lo que su valor
  óptimo no depende del tamaño de la instancia.
- σ controla la SENSIBILIDAD a la amplitud actual (mayor σ ⇒ p de aceptación
  más baja para una A dada, curva más plana). En este script σ NO se pasa
  como valor absoluto: se expresa como una fracción de la amplitud inicial
  efectiva (``factor_sigma · A0_eff``), lo que mantiene la calibración
  proporcional entre instancias de distinto tamaño y hace el grid
  legible / comparable.
- ``p_inter`` juega el mismo rol que en SA / TS / RTS / Cuckoo: el sesgo de
  operador. Se barre con el mismo rango para poder cruzar los resultados con
  los otros grids.

A0 y L se dejan en ``None`` para que el algoritmo aplique la calibración
instance-aware canónica (A0 = 20·d_max/n, L = n²). ``max_niveles`` también
se deja en ``None`` para que la parada dependa exclusivamente del umbral de
amplitud (analogía de T_min en SA).

NOTA SOBRE LA AUSENCIA DE SEMILLA FIJA:
Las repeticiones EXISTEN para medir la robustez de cada configuración a la
aleatoriedad inherente de la metaheurística (aceptación estocástica de
empeoramientos, selección de vecinos y de grupo inter/intra). Si fijáramos
una semilla determinista por tarea, las repeticiones de cada combinación
serían IDÉNTICAS — no aportarían información estadística. Una buena
configuración debe converger a buenos óptimos SIN IMPORTAR la trayectoria
aleatoria que tome. Por eso aquí cada corrida muestrea libremente del
espacio aleatorio del sistema (igual decisión que en RTS / TS simple /
ABC simple / Cuckoo).

Paralelismo (--workers N)
-------------------------
Cada combinación (instancia, gamma, factor_sigma, p_inter, repetición) es una
tarea INDEPENDIENTE: no comparte estado con las demás y solo lee la instancia
desde disco. Esto las hace trivialmente paralelizables con
``ProcessPoolExecutor``. El flag ``--workers N`` controla cuántos procesos se
lanzan en paralelo:
    --workers 1   → ejecución secuencial (útil para depurar y porque stdout
                    no se mezcla entre tareas).
    --workers N   → N procesos en paralelo (default: os.cpu_count()).

Reproducibilidad:
    NO hay reproducibilidad bit-a-bit por diseño. Sí hay reproducibilidad
    ESTADÍSTICA: con 5 repeticiones por configuración, las medias y
    desviaciones convergen a valores estables comparables entre experimentos.

Concurrencia de E/S:
    Para evitar carreras al escribir el CSV, cada worker escribe su fila en
    un CSV TEMPORAL único (por PID + índice de tarea) dentro de una subcarpeta
    ``_partials/``. Al terminar todas las tareas, el proceso principal
    CONCATENA los parciales por instancia en el CSV final canónico. Sin locks,
    sin colas, sin riesgo de filas truncadas.

Uso:
    python scripts/run_vdo_automatico.py
    python scripts/run_vdo_automatico.py --salida-dir experimentos/vdo_grid
    python scripts/run_vdo_automatico.py --repeticiones 5
    python scripts/run_vdo_automatico.py --workers 8
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
from metacarp import vibration_damping_desde_instancia

# Mismo conjunto de instancias pequeñas que usan los scripts de SA / TS / RTS /
# ABC simple / Cuckoo, para mantener la comparabilidad entre experimentos.
INSTANCIAS = [
    "gdb19", "kshs1", "kshs2", "kshs3", "kshs4", "kshs5", "kshs6",
    "gdb4",  "gdb14", "gdb15", "gdb1",  "gdb20", "gdb3",  "gdb6",
    "gdb7",  "gdb12", "gdb10", "gdb2",  "gdb5",  "gdb13", "gdb16",
    "gdb17", "gdb21",
]

# Grid search 3D sobre los parámetros vibratorios de VDO:
#
# - gamma: coeficiente de amortiguamiento (rapidez del decaimiento). Rango
#   típico en la literatura: 0.01 – 0.2. Un γ pequeño equivale a un sistema
#   con poca fricción (muchos niveles, exploración prolongada); un γ grande
#   equivale a mucha fricción (pocos niveles, convergencia rápida). Los 4
#   puntos elegidos cubren el rango de "casi sin amortiguamiento" (0.02) a
#   "convergencia agresiva" (0.2).
#
# - factor_sigma: se traduce dentro del script a ``sigma = factor · A0_eff``
#   ANTES de invocar al algoritmo, para que el valor absoluto de σ escale
#   proporcionalmente al tamaño de cada instancia (idéntica filosofía a los
#   factores de Cuckoo / ABC simple). factor_sigma = 0.5 reproduce el default
#   canónico (σ = A0/2). Valores más pequeños "endurecen" la aceptación
#   (curva Rayleigh más empinada); valores más grandes la relajan.
#
# - p_inter: P(elegir el grupo INTER-RUTA) cuando la solución actual es
#   factible. Bajo violación, el algoritmo eleva a max(p_inter, alpha_inter)
#   automáticamente. Mismo rango que en SA / ABC / Cuckoo para comparabilidad.
GAMMAS         = [0.02, 0.05, 0.1, 0.2]
FACTORES_SIGMA = [0.25, 0.5, 1.0, 2.0]
P_INTERS       = [0.4, 0.5, 0.6, 0.7, 0.8]


# --- DATACLASS DE TAREA ---
# Encapsula todos los parámetros de UNA corrida. Es un objeto plano,
# pickle-friendly, que se envía a un proceso worker. Mantener todos los
# parámetros en un solo objeto evita firmas con muchos argumentos y facilita
# extender el grid en el futuro.
@dataclass(frozen=True)
class TareaVDO:
    instancia: str
    # Coeficiente de amortiguamiento del oscilador vibratorio.
    gamma: float
    # Factor multiplicativo aplicado a A0_eff para derivar σ (se convierte a
    # valor absoluto dentro del worker, una vez el algoritmo conozca A0_eff).
    factor_sigma: float
    # P(elegir inter-ruta) cuando la solución es factible.
    p_inter: float
    # Repetición dentro del experimento (solo informativa: NO determina semilla).
    repeticion: int
    # NOTA: NO hay campo ``semilla``. Cada corrida se ejecuta con semilla
    # aleatoria del sistema (la metaheurística internamente crea su propio
    # ``random.Random()``). Esto es deliberado: las "repeticiones" SOLO son
    # informativas si cada una muestrea una trayectoria distinta del espacio
    # aleatorio. Una semilla determinista por tarea las convertiría en
    # mediciones idénticas, sin valor estadístico.
    #
    # NOTA: A0 y L NO se barren. Se dejan en None y el algoritmo los calibra
    # con sus fórmulas instance-aware default (A0 = 20·d_max/n, L = n²).
    usar_gpu: bool                            # si True, evaluación en lote usa GPU
    root: str | None
    ruta_csv_parcial: str                     # cada tarea escribe a su propio archivo


def _resolver_sigma_absoluto(instancia: str, factor_sigma: float, root: str | None) -> float:
    """
    Resuelve el valor absoluto de σ a partir del ``factor_sigma`` de la tarea.

    El algoritmo VDO calcula A0_eff internamente como ``20·d_max/n``. Para que
    el grid barra σ de forma proporcional a cada instancia, aquí replicamos
    ese cálculo con los mismos ingredientes (``d_max`` y ``n_tareas``) y
    devolvemos ``sigma = factor_sigma · A0_eff``. Es más simple que ejecutar
    dos veces la metaheurística (una para leer A0_eff y otra para pasar σ).

    Precondición: la instancia ya tiene su matriz Dijkstra en caché (todas las
    instancias del proyecto la tienen precomputada; si no, el contexto la
    computa una vez y la guarda para las corridas siguientes).
    """
    # Import diferido: mantenemos limpio el top-level del script.
    from metacarp.evaluador_costo import construir_contexto_desde_instancia
    import numpy as np

    ctx = construir_contexto_desde_instancia(instancia, root=root, usar_gpu=False)
    n = int(len(ctx.u_arr))
    _dist_finita = ctx.dist[ctx.dist < np.inf]
    d_max = float(_dist_finita.max()) if len(_dist_finita) > 0 else 1.0
    a0 = 20.0 * d_max / max(1, n)
    return float(factor_sigma) * a0


def _ejecutar_tarea(tarea: TareaVDO) -> tuple[TareaVDO, str, dict | None, str | None]:
    """
    Ejecuta UNA corrida de VDO en el worker.

    Esta función se ejecuta dentro de un PROCESO HIJO. Recibe la tarea,
    resuelve el valor absoluto de σ desde ``factor_sigma``, invoca el
    algoritmo y devuelve una tupla con el estado del resultado. No imprime:
    el proceso principal formatea y muestra el resumen al recibir cada futuro
    completado.

    Returns
    -------
    (tarea, "ok"|"fail", info_dict_o_None, mensaje_error_o_None)
    """
    try:
        # Traducimos el factor a valor absoluto de σ una sola vez por tarea.
        sigma_abs = _resolver_sigma_absoluto(tarea.instancia, tarea.factor_sigma, tarea.root)
        res = vibration_damping_desde_instancia(
            tarea.instancia,
            gamma=tarea.gamma,
            sigma=sigma_abs,
            p_inter=tarea.p_inter,
            # A0, umbral_amplitud_minima e iteraciones_por_nivel se dejan en
            # None ⇒ calibración instance-aware canónica.
            amplitud_inicial=None,
            umbral_amplitud_minima=None,
            iteraciones_por_nivel=None,
            max_niveles=None,
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
            "niveles": res.niveles_ejecutados,
            "aceptadas": res.aceptadas,
            "mejoras": res.mejoras,
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
    instancia. Misma estrategia map-reduce que los otros scripts (ABC / RTS
    / Cuckoo): cada worker escribe a su parcial, el principal fusiona al final.

    Devuelve el número de archivos finales generados.
    """
    grupos: dict[str, list[Path]] = {}
    for parcial in sorted(dir_parciales.glob("vdo_*.csv")):
        # Nombre con formato ``vdo_<instancia>_<pid>_<idx>.csv``.
        partes = parcial.stem.split("_")
        if len(partes) < 4:
            continue
        # ``vdo`` ocupa partes[0]; la instancia es partes[1].
        instancia = partes[1]
        grupos.setdefault(instancia, []).append(parcial)

    n_archivos_creados = 0
    for instancia, archivos in grupos.items():
        ruta_final = salida_dir / f"vdo_{instancia}_{experimento}_{ydmh}.csv"
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
            "Vibration Damping Optimization (instance-aware) — grid 3D sobre "
            "(gamma, factor_sigma, p_inter)."
        )
    )
    parser.add_argument("--salida-dir",   type=str, default="experimentos")
    parser.add_argument("--repeticiones", type=int, default=5)
    parser.add_argument("--experimento",  type=str, default="vdo_auto")
    parser.add_argument("--root",         type=str, default=None)
    # Overrides opcionales que FIJAN una dimensión a un solo valor (reducen
    # el grid). Si no se pasan, se barre la lista completa de esa dimensión.
    parser.add_argument("--gamma", type=float, default=None,
                        help="Fija el coeficiente de amortiguamiento. "
                             "None = barrer lista [0.02, 0.05, 0.1, 0.2].")
    parser.add_argument("--factor-sigma", type=float, default=None,
                        help="Fija el factor de escala de σ (σ = f·A0_eff). "
                             "None = barrer lista [0.25, 0.5, 1.0, 2.0].")
    parser.add_argument("--p-inter", type=float, default=None,
                        help="P(elegir inter) cuando la sol. actual es factible. "
                             "None = barrer lista [0.4, 0.5, 0.6, 0.7, 0.8].")
    # --usar-gpu se acepta por consistencia con los demás scripts, aunque
    # VDO evalúa una solución por iteración y no aprovecha GPU (idéntica
    # situación que en SA).
    parser.add_argument(
        "--usar-gpu",
        action="store_true",
        default=False,
        help="Activa evaluación en lote con GPU (CuPy). VDO no lo aprovecha "
             "(evalúa una solución por iteración), pero mantenemos el flag "
             "por consistencia con los demás scripts.",
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
    # metaheurística (ver nota en la dataclass ``TareaVDO``).
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
    # Marca de tiempo: año-día-mes-hora-minuto, mismo formato que SA / TS / RTS / ABC / Cuckoo.
    ydmh = datetime.now().strftime("%Y%d%m%H%M")

    # --- RESOLUCIÓN DE LAS DIMENSIONES DEL GRID ---
    # Si el usuario pasa un override single-value por CLI (--gamma 0.05), ese
    # valor SUSTITUYE a la lista completa del grid: la dimensión correspondiente
    # queda fijada a un solo punto.
    gammas         = [args.gamma]        if args.gamma        is not None else GAMMAS
    factores_sigma = [args.factor_sigma] if args.factor_sigma is not None else FACTORES_SIGMA
    p_inters       = [args.p_inter]      if args.p_inter      is not None else P_INTERS

    # --- CONSTRUCCIÓN DE LA LISTA DE TAREAS ---
    # Una tarea por (instancia, gamma, factor_sigma, p_inter, repeticion). Cada
    # tarea es independiente: solo lectura de la instancia desde disco. Esto
    # las hace trivialmente paralelizables con ProcessPoolExecutor.
    tareas: list[TareaVDO] = []
    for instancia in INSTANCIAS:
        for g in gammas:
            for fs in factores_sigma:
                for p_int in p_inters:
                    for rep in range(1, args.repeticiones + 1):
                        idx = len(tareas)
                        parcial = dir_parciales / (
                            f"vdo_{instancia}_{os.getpid()}_{idx}.csv"
                        )
                        tareas.append(TareaVDO(
                            instancia=instancia,
                            gamma=g,
                            factor_sigma=fs,
                            p_inter=p_int,
                            repeticion=rep,
                            usar_gpu=args.usar_gpu,
                            root=args.root,
                            ruta_csv_parcial=str(parcial),
                        ))

    total = len(tareas)
    print("=" * 80)
    print("Vibration Damping Optimization (instance-aware) — grid search 3D")
    print("=" * 80)
    print(f"Instancias                  : {len(INSTANCIAS)}")
    print(f"gammas                      : {gammas}")
    print(f"factores_sigma              : {factores_sigma}  → σ = factor · A0_eff")
    print(f"p_inter values              : {p_inters}")
    print(f"A0 / L                      : instance-aware (A0=20·d_max/n, L=n²)")
    print(f"umbral_amplitud_minima      : instance-aware (20·d_max/n²)")
    print(f"max_niveles                 : sin tope (parada por umbral de A)")
    print(f"GPU (evaluación en lote)    : {'activada' if args.usar_gpu else 'desactivada (CPU)'}")
    print(f"Semilla                     : aleatoria del sistema (cada repetición es independiente)")
    print(f"Repeticiones                : {args.repeticiones}")
    print(f"Workers                     : {args.workers}")
    print(f"Corridas                    : {total}")
    print(f"Salida CSV                  : {salida_dir}")
    print("-" * 80)

    total_ok = 0
    total_fail = 0

    # --- EJECUCIÓN ---
    # Misma estructura que los otros scripts: rama secuencial explícita cuando
    # --workers <= 1, rama paralela con ProcessPoolExecutor en otro caso.
    if args.workers <= 1:
        for tarea in tareas:
            _, estado, info, err = _ejecutar_tarea(tarea)
            if estado == "ok" and info is not None:
                print(
                    f"  [{tarea.instancia}] "
                    f"γ={tarea.gamma:.3f} "
                    f"fσ={tarea.factor_sigma:.2f} "
                    f"pInt={tarea.p_inter:.1f} "
                    f"rep={tarea.repeticion} "
                    f"| costo={info['costo']:.4f} "
                    f"| mejora={info['mejora']:.2f}% "
                    f"| iter={info['iter']} "
                    f"| niveles={info['niveles']} "
                    f"| acept={info['aceptadas']} "
                    f"| t={info['tiempo']:.2f}s"
                )
                total_ok += 1
            else:
                print(
                    f"  [{tarea.instancia}] "
                    f"γ={tarea.gamma:.3f} "
                    f"fσ={tarea.factor_sigma:.2f} "
                    f"pInt={tarea.p_inter:.1f} "
                    f"rep={tarea.repeticion} | FAIL: {err}"
                )
                total_fail += 1
    else:
        # MODO PARALELO con ProcessPoolExecutor.
        # Mismas garantías que en los scripts de ABC / RTS / Cuckoo:
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
                        f"γ={tarea.gamma:.3f} "
                        f"fσ={tarea.factor_sigma:.2f} "
                        f"pInt={tarea.p_inter:.1f} "
                        f"rep={tarea.repeticion} "
                        f"| costo={info['costo']:.4f} "
                        f"| mejora={info['mejora']:.2f}% "
                        f"| iter={info['iter']} "
                        f"| niveles={info['niveles']} "
                        f"| acept={info['aceptadas']} "
                        f"| t={info['tiempo']:.2f}s"
                    )
                    total_ok += 1
                else:
                    print(
                        f"  [{tarea.instancia}] "
                        f"γ={tarea.gamma:.3f} "
                        f"fσ={tarea.factor_sigma:.2f} "
                        f"pInt={tarea.p_inter:.1f} "
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
