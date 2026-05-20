"""
Corrida Cuckoo Search (Yang & Deb 2009 adaptado a CARP) para instancias seleccionadas.

Versión modernizada e instance-aware de Cuckoo Search alineada con SA / TS simple
/ RTS / ABC simple:
- Parámetros calibrados automáticamente por ``n_tareas`` (con factores de escala
  como override).
- Sesgo inter/intra-ruta vía ``seleccionar_grupo_operadores_inter_intra``
  (mismo helper que SA / TS / RTS / ABC simple).
- ``alpha_inter`` ELIMINADO: el algoritmo eleva automáticamente P(inter) a
  ``max(p_inter, 0.8)`` en presencia de violación de capacidad. El usuario
  solo configura ``p_inter`` (la probabilidad cuando la solución es factible).
- ``registrar_mejora`` imputada al operador del vuelo Lévy responsable.
- ``extra_csv`` ahora se vuelca a la fila CSV (antes se ignoraba).

Configuración (grid search 4D sobre FACTORES DE ESCALA en función de n_tareas):

Grid (4 dimensiones):
    factor_nidos   ∈ {1.5, 2.0, 3.0, 4.0}       → num_nidos       = max(10, round(f·√n))
    factor_pasos   ∈ {0.25, 0.5, 0.75, 1.0}     → pasos_levy_base = max(3,  round(f·√n))
    p_inter        ∈ {0.4, 0.5, 0.6, 0.7, 0.8}   → P(inter) en régimen factible
    pa_abandono    ∈ {0.15, 0.25, 0.35}          → fracción de peores nidos a abandonar

Total: 4 × 4 × 5 × 3 = 240 combos × 23 instancias × 5 reps = 27,600 corridas.

POR QUÉ FACTORES Y NO VALORES ABSOLUTOS:
El grid anterior usaba valores ABSOLUTOS (num_nidos ∈ {10,20,30}, etc.) que
ignoraban el tamaño de la instancia. Una corrida con 20 nidos sobre una
instancia pequeña (gdb1, n=22) explora el espacio de forma muy distinta que
sobre una grande (gdb20). Con factores de escala, el algoritmo asigna recursos
PROPORCIONALES a cada instancia, lo que hace mucho más comparables los
resultados entre instancias y mucho más limpia la lectura del grid (un factor
"óptimo" detectado en instancias pequeñas también es "óptimo" en grandes,
salvo efectos no lineales que ahora se aprecian sin ruido del tamaño absoluto).

``factor_iter`` no se barre en el grid principal (se deja en None y el algoritmo
calibra ``iteraciones = max(200, 20·n)``). El flag CLI ``--factor-iter`` permite
fijarlo como override single-value. ``max_iter_sin_mejora`` tampoco se barre: el
algoritmo lo calibra a ``max(50, 3·n_tareas)``.

NOTA SOBRE LA AUSENCIA DE SEMILLA FIJA:
Las repeticiones EXISTEN para medir la robustez de cada configuración a la
aleatoriedad inherente de la metaheurística (vuelo de Lévy, competencia con
nidos aleatorios, abandono de peores). Si fijáramos una semilla determinista
por tarea, las repeticiones de cada combinación serían IDÉNTICAS — no
aportarían información estadística. Una buena configuración debe converger a
buenos óptimos SIN IMPORTAR la trayectoria aleatoria que tome. Por eso aquí
cada corrida muestrea libremente del espacio aleatorio del sistema (igual
decisión que en RTS / TS simple / ABC simple).

Para REDUCIR el grid, basta con pasar overrides single-value por CLI:
    --p-inter 0.6 --factor-nidos 2.0  → fija dos dimensiones, solo barre el resto.

Paralelismo (--workers N)
-------------------------
Cada combinación (instancia, factor_nidos, factor_pasos, p_inter, pa_abandono,
repetición) es una tarea INDEPENDIENTE: no comparte estado con las demás y solo
lee la instancia desde disco. Esto las hace trivialmente paralelizables con
``ProcessPoolExecutor``. El flag ``--workers N`` controla cuántos procesos se
lanzan en paralelo:
    --workers 1   → ejecución secuencial (útil para depurar y porque stdout
                    no se mezcla entre tareas).
    --workers N   → N procesos en paralelo (default: os.cpu_count()).

Reproducibilidad:
    NO hay reproducibilidad bit-a-bit por diseño (ver nota arriba sobre la
    ausencia de semilla fija). Sin embargo, sí hay reproducibilidad
    ESTADÍSTICA: con 5 repeticiones por configuración, las medias y
    desviaciones de costo, mejora, número de reemplazos, etc. convergen a
    valores estables que sí son comparables entre experimentos.

Concurrencia de E/S:
    Para evitar carreras al escribir el CSV, cada worker escribe su fila en
    un CSV TEMPORAL único (por PID + índice de tarea) dentro de una subcarpeta
    ``_partials/``. Al terminar todas las tareas, el proceso principal
    CONCATENA los parciales por instancia en el CSV final canónico. Sin locks,
    sin colas, sin riesgo de filas truncadas.

Uso:
    python scripts/run_cuckoo_automatico.py
    python scripts/run_cuckoo_automatico.py --salida-dir experimentos/cuckoo_grid
    python scripts/run_cuckoo_automatico.py --repeticiones 5
    python scripts/run_cuckoo_automatico.py --workers 8
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
from metacarp import cuckoo_search_desde_instancia

# Mismo conjunto de instancias pequeñas que usan los scripts de SA / TS / RTS /
# ABC simple, para mantener la comparabilidad de los experimentos (gdb + kshs).
INSTANCIAS = [
    "gdb19", "kshs1", "kshs2", "kshs3", "kshs4", "kshs5", "kshs6",
    "gdb4",  "gdb14", "gdb15", "gdb1",  "gdb20", "gdb3",  "gdb6",
    "gdb7",  "gdb12", "gdb10", "gdb2",  "gdb5",  "gdb13", "gdb16",
    "gdb17", "gdb21",
]

# Grid search 4D sobre FACTORES DE ESCALA en función de n_tareas:
#
# - factor_nidos: el algoritmo calcula num_nidos = max(10, round(f·√n)). Para
#   gdb1 (n≈22) y f=1.5 da num_nidos≈7 (piso 10); para f=4.0 da ≈19. El rango
#   cubre poblaciones desde el mínimo recomendado por la literatura hasta
#   poblaciones grandes que aumentan diversidad pero también el costo por iter.
#
# - factor_pasos: pasos_levy_base = max(3, round(f·√n)). Determina la escala
#   base del vuelo Lévy discreto (más pasos = saltos más largos en promedio).
#   0.25 da saltos cortos (intensificación); 1.0 da saltos largos
#   (diversificación). El número real de pasos por vuelo varía aleatoriamente
#   según la distribución de Lévy entre 1 y 12.
#
# - p_inter: P(elegir el grupo INTER-RUTA) cuando la solución actual es
#   factible. Bajo violación, el algoritmo eleva a max(p_inter, 0.8)
#   automáticamente. Mismo rango que en ABC simple para comparabilidad.
#
# - pa_abandono: fracción de los PEORES nidos que se abandonan por iteración
#   (se reconstruyen desde el mejor nido con un vuelo Lévy). Valor más alto =
#   más diversificación por iteración. 0.25 es el clásico de la literatura.
FACTORES_NIDOS = [1.5, 2.0, 3.0, 4.0]
FACTORES_PASOS = [0.25, 0.5, 0.75, 1.0]
P_INTERS       = [0.4, 0.5, 0.6, 0.7, 0.8]
PA_ABANDONOS   = [0.15, 0.25, 0.35]


# --- DATACLASS DE TAREA ---
# Encapsula todos los parámetros de UNA corrida. Es un objeto plano,
# pickle-friendly, que se envía a un proceso worker. Mantener todos los
# parámetros en un solo objeto evita firmas con muchos argumentos y facilita
# extender el grid en el futuro.
@dataclass(frozen=True)
class TareaCuckoo:
    instancia: str
    # Factor de escala de num_nidos: num_nidos = max(10, round(f · √n_tareas)).
    factor_nidos: float
    # Factor de escala de pasos_levy_base: pasos = max(3, round(f · √n_tareas)).
    factor_pasos: float
    # P(elegir inter-ruta) cuando la solución es factible.
    p_inter: float
    # Fracción de los peores nidos a abandonar por iteración.
    pa_abandono: float
    # Repetición dentro del experimento (solo informativa: NO determina semilla).
    repeticion: int
    # NOTA: NO hay campo ``semilla``. Cada corrida se ejecuta con semilla
    # aleatoria del sistema (la metaheurística internamente crea su propio
    # ``random.Random()``). Esto es deliberado: las "repeticiones" SOLO son
    # informativas si cada una muestrea una trayectoria distinta del espacio
    # aleatorio. Una semilla determinista por tarea las convertiría en
    # mediciones idénticas, sin valor estadístico.
    #
    # NOTA: ``factor_iter`` y ``max_iter_sin_mejora`` no se barren. Se dejan
    # en None y el algoritmo los calibra con sus fórmulas instance-aware
    # default (max(200, 20·n) y max(50, 3·n) respectivamente).
    # ``factor_iter`` puede fijarse por CLI como override single-value.
    usar_gpu: bool                            # si True, evaluación en lote usa GPU
    # factor_iter opcional propagado desde CLI (None ⇒ default del algoritmo).
    factor_iter: float | None
    root: str | None
    ruta_csv_parcial: str                     # cada tarea escribe a su propio archivo


def _ejecutar_tarea(tarea: TareaCuckoo) -> tuple[TareaCuckoo, str, dict | None, str | None]:
    """
    Ejecuta UNA corrida de Cuckoo Search en el worker.

    Esta función se ejecuta dentro de un PROCESO HIJO. Recibe la tarea, invoca
    el algoritmo y devuelve una tupla con el estado del resultado. No imprime:
    el proceso principal formatea y muestra el resumen al recibir cada futuro
    completado.

    El script pasa los FACTORES de escala (no los valores absolutos): el
    algoritmo internamente convierte cada factor a un valor concreto en
    función de ``n_tareas`` de la instancia. Esto mantiene la comparabilidad
    entre instancias de distinto tamaño.

    Returns
    -------
    (tarea, "ok"|"fail", info_dict_o_None, mensaje_error_o_None)
    """
    try:
        res = cuckoo_search_desde_instancia(
            tarea.instancia,
            # Factores de escala: el algoritmo los traduce a valores concretos
            # usando n_tareas de la instancia. La precedencia es:
            # valor absoluto > factor > default por fórmula. Aquí no pasamos
            # ningún valor absoluto, así que mandan los factores.
            factor_nidos=tarea.factor_nidos,
            factor_pasos=tarea.factor_pasos,
            factor_iter=tarea.factor_iter,           # None ⇒ default del algoritmo
            p_inter=tarea.p_inter,
            pa_abandono=tarea.pa_abandono,
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
            "reemplazos": res.reemplazos_exitosos,
            "abandonos": res.abandonos_totales,
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
    instancia. Misma estrategia map-reduce que el script de ABC simple / RTS:
    cada worker escribe a su parcial, el principal fusiona al final.

    Devuelve el número de archivos finales generados.
    """
    grupos: dict[str, list[Path]] = {}
    for parcial in sorted(dir_parciales.glob("cuckoo_*.csv")):
        # Nombre con formato ``cuckoo_<instancia>_<pid>_<idx>.csv``.
        partes = parcial.stem.split("_")
        if len(partes) < 4:
            continue
        # ``cuckoo`` ocupa partes[0]; la instancia es partes[1].
        instancia = partes[1]
        grupos.setdefault(instancia, []).append(parcial)

    n_archivos_creados = 0
    for instancia, archivos in grupos.items():
        ruta_final = salida_dir / f"cuckoo_{instancia}_{experimento}_{ydmh}.csv"
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
            "Cuckoo Search (modernizado, instance-aware) — grid 4D sobre "
            "factores de escala (factor_nidos, factor_pasos, p_inter, pa_abandono)."
        )
    )
    parser.add_argument("--salida-dir",   type=str, default="experimentos")
    parser.add_argument("--repeticiones", type=int, default=5)
    parser.add_argument("--experimento",  type=str, default="cuckoo_auto")
    parser.add_argument("--root",         type=str, default=None)
    # Overrides opcionales que FIJAN una dimensión a un solo valor (reducen
    # el grid). Si no se pasan, se barre la lista completa de esa dimensión.
    parser.add_argument("--p-inter", type=float, default=None,
                        help="P(elegir inter) cuando la sol. actual es factible. "
                             "None = barrer lista [0.4, 0.5, 0.6, 0.7, 0.8].")
    parser.add_argument("--factor-nidos", type=float, default=None,
                        help="Fija el factor de escala de num_nidos. "
                             "None = barrer lista [1.5, 2.0, 3.0, 4.0].")
    parser.add_argument("--factor-pasos", type=float, default=None,
                        help="Fija el factor de escala de pasos_levy_base. "
                             "None = barrer lista [0.25, 0.5, 0.75, 1.0].")
    parser.add_argument("--pa-abandono", type=float, default=None,
                        help="Fija la fracción de nidos a abandonar. "
                             "None = barrer lista [0.15, 0.25, 0.35].")
    parser.add_argument("--factor-iter", type=float, default=None,
                        help="Fija el factor de escala de iteraciones (override). "
                             "None = usa el default instance-aware del algoritmo "
                             "(max(200, 20·n_tareas)).")
    # --usar-gpu activa la evaluación en lote con GPU (CuPy). Solo aplica si
    # CuPy está instalado y hay GPU disponible; si no, el algoritmo cae
    # silenciosamente a CPU. Para instancias small, el speedup es marginal
    # pero no introduce errores.
    parser.add_argument(
        "--usar-gpu",
        action="store_true",
        default=False,
        help="Activa evaluación en lote con GPU (CuPy).",
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
    # metaheurística (ver nota en la dataclass ``TareaCuckoo``).
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
    # Marca de tiempo: año-día-mes-hora-minuto, mismo formato que SA / TS / RTS / ABC.
    ydmh = datetime.now().strftime("%Y%d%m%H%M")

    # --- RESOLUCIÓN DE LAS DIMENSIONES DEL GRID ---
    # Si el usuario pasa un override single-value por CLI (--p-inter 0.6), ese
    # valor SUSTITUYE a la lista completa del grid: la dimensión correspondiente
    # queda fijada a un solo punto.
    p_inter_values      = [args.p_inter]      if args.p_inter      is not None else P_INTERS
    factor_nidos_values = [args.factor_nidos] if args.factor_nidos is not None else FACTORES_NIDOS
    factor_pasos_values = [args.factor_pasos] if args.factor_pasos is not None else FACTORES_PASOS
    pa_abandono_values  = [args.pa_abandono]  if args.pa_abandono  is not None else PA_ABANDONOS

    # --- CONSTRUCCIÓN DE LA LISTA DE TAREAS ---
    # Una tarea por (instancia, factor_nidos, factor_pasos, p_inter,
    # pa_abandono, repeticion). Cada tarea es independiente: solo lectura
    # de la instancia desde disco. Esto las hace trivialmente paralelizables
    # con ProcessPoolExecutor.
    tareas: list[TareaCuckoo] = []
    for instancia in INSTANCIAS:
        for fn in factor_nidos_values:
            for fp in factor_pasos_values:
                for p_int in p_inter_values:
                    for pa in pa_abandono_values:
                        for rep in range(1, args.repeticiones + 1):
                            idx = len(tareas)
                            parcial = dir_parciales / (
                                f"cuckoo_{instancia}_{os.getpid()}_{idx}.csv"
                            )
                            tareas.append(TareaCuckoo(
                                instancia=instancia,
                                factor_nidos=fn,
                                factor_pasos=fp,
                                p_inter=p_int,
                                pa_abandono=pa,
                                repeticion=rep,
                                usar_gpu=args.usar_gpu,
                                factor_iter=args.factor_iter,
                                root=args.root,
                                ruta_csv_parcial=str(parcial),
                            ))

    total = len(tareas)
    print("=" * 80)
    print("Cuckoo Search (modernizado, instance-aware) — grid search 4D (factores)")
    print("=" * 80)
    print(f"Instancias                  : {len(INSTANCIAS)}")
    print(f"factor_nidos values         : {factor_nidos_values}  → num_nidos       = max(10, round(f·√n))")
    print(f"factor_pasos values         : {factor_pasos_values}  → pasos_levy_base = max(3,  round(f·√n))")
    print(f"p_inter values              : {p_inter_values}")
    print(f"pa_abandono values          : {pa_abandono_values}")
    if args.factor_iter is not None:
        print(f"factor_iter (override)      : {args.factor_iter}  → iteraciones    = max(200, round(f·n))")
    else:
        print(f"factor_iter                 : default instance-aware (max(200, 20·n))")
    print(f"GPU (evaluación en lote)    : {'activada' if args.usar_gpu else 'desactivada (CPU)'}")
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
    # Misma estructura que el script ABC simple / RTS: rama secuencial explícita
    # cuando --workers <= 1, rama paralela con ProcessPoolExecutor en otro caso.
    if args.workers <= 1:
        for tarea in tareas:
            _, estado, info, err = _ejecutar_tarea(tarea)
            if estado == "ok" and info is not None:
                # Print con los FACTORES (no valores absolutos): así el lector
                # puede comparar entre instancias sin tener que conocer n_tareas.
                print(
                    f"  [{tarea.instancia}] "
                    f"fn={tarea.factor_nidos:.2f} "
                    f"fp={tarea.factor_pasos:.2f} "
                    f"pInt={tarea.p_inter:.1f} "
                    f"pa={tarea.pa_abandono:.2f} "
                    f"rep={tarea.repeticion} "
                    f"| costo={info['costo']:.4f} "
                    f"| mejora={info['mejora']:.2f}% "
                    f"| iter={info['iter']} "
                    f"| reempl={info['reemplazos']} "
                    f"| aband={info['abandonos']} "
                    f"| t={info['tiempo']:.2f}s"
                )
                total_ok += 1
            else:
                print(
                    f"  [{tarea.instancia}] "
                    f"fn={tarea.factor_nidos:.2f} "
                    f"fp={tarea.factor_pasos:.2f} "
                    f"pInt={tarea.p_inter:.1f} "
                    f"pa={tarea.pa_abandono:.2f} "
                    f"rep={tarea.repeticion} | FAIL: {err}"
                )
                total_fail += 1
    else:
        # MODO PARALELO con ProcessPoolExecutor.
        # Mismas garantías que en el script de ABC simple / RTS:
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
                        f"fn={tarea.factor_nidos:.2f} "
                        f"fp={tarea.factor_pasos:.2f} "
                        f"pInt={tarea.p_inter:.1f} "
                        f"pa={tarea.pa_abandono:.2f} "
                        f"rep={tarea.repeticion} "
                        f"| costo={info['costo']:.4f} "
                        f"| mejora={info['mejora']:.2f}% "
                        f"| iter={info['iter']} "
                        f"| reempl={info['reemplazos']} "
                        f"| aband={info['abandonos']} "
                        f"| t={info['tiempo']:.2f}s"
                    )
                    total_ok += 1
                else:
                    print(
                        f"  [{tarea.instancia}] "
                        f"fn={tarea.factor_nidos:.2f} "
                        f"fp={tarea.factor_pasos:.2f} "
                        f"pInt={tarea.p_inter:.1f} "
                        f"pa={tarea.pa_abandono:.2f} "
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
