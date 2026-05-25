"""
Utilidades compartidas por los 5 scripts ``run_<mh>_lambda_grid_20260525.py``.

Extiende ``_strict_intra_inter_20260524_common`` añadiendo ``lambda_factor``
como dimension del grid. Conserva TODO lo demas:

  - Monkey-patch del selector binario estricto
    (``seleccionar_grupo_operadores_inter_intra`` -> ``seleccionar_grupo_strict``).
  - Subconjunto reducido ``OPERADORES_STRICT_5``.
  - Kick reactivo via ``max_iter_sin_mejora_kick`` y ``max_resets``.
  - Repeticiones con semilla aleatoria del sistema (no determinista).
  - Paralelismo con ``ProcessPoolExecutor`` y consolidacion map-reduce.

La diferencia es el GRID: cada (instancia, repeticion) se replica por cada
``lambda_factor`` de la lista ``LAMBDA_FACTORS`` y ``lambda_capacidad`` se
pasa al wrapper de la MH como ``lambda_factor * lambda_default``, donde
``lambda_default`` proviene de
``lambda_penal_capacidad_por_defecto(ctx)`` para esa instancia concreta.

Total nominal por MH: 23 instancias x 3 reps x 5 lambdas = 345 corridas
(x 5 MHs = 1725 corridas en el grid completo).
"""
from __future__ import annotations

import argparse
import csv
import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

# Garantiza que ``metacarp`` sea importable cuando se ejecuta desde scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Las 23 instancias pequenas del corpus actual de MetaCARP (mismas que el
# experimento ``strict_intra_inter_20260524``). Orden alterado para repartir
# las kshs (mas pesadas) al principio y evitar que se acumulen al final
# en el mismo worker en modo paralelo.
INSTANCIAS = [
    "gdb19", "kshs1", "kshs2", "kshs3", "kshs4", "kshs5", "kshs6",
    "gdb4",  "gdb14", "gdb15", "gdb1",  "gdb20", "gdb3",  "gdb6",
    "gdb7",  "gdb12", "gdb10", "gdb2",  "gdb5",  "gdb13", "gdb16",
    "gdb17", "gdb21",
]

# Lista canonica de multiplicadores de lambda. El valor efectivo usado en
# cada corrida es ``lambda_actual = lambda_factor * lambda_default(ctx)``,
# donde ``lambda_default`` se calcula por instancia mediante
# ``lambda_penal_capacidad_por_defecto`` (mediana de distancias x 10, piso
# 10.0). Los factores cubren tanto sub-penalizacion (0.5) como sobre-
# penalizacion fuerte (10.0) respecto al valor canonico (1.0).
LAMBDA_FACTORS: list[float] = [0.5, 1.0, 2.0, 5.0, 10.0]


# ============================================================
# Dataclass de tarea
# ============================================================

@dataclass(frozen=True)
class TareaExp:
    """Una corrida del grid experimental lambda_grid_20260525.

    Igual que ``TareaExp`` del experimento ``strict_intra_inter_20260524``
    pero con el campo ``lambda_factor`` adicional. NO incluye ``semilla``:
    cada repeticion arranca con la semilla del sistema (aleatoria) para
    muestrear trayectorias independientes.
    """
    instancia: str
    repeticion: int
    lambda_factor: float
    root: str | None
    ruta_csv_parcial: str


# ============================================================
# Calculo del lambda efectivo por instancia
# ============================================================

def calcular_lambda_default(nombre_instancia: str, root: str | None) -> float:
    """Calcula el ``lambda_default`` de una instancia.

    Carga los datos y el grafo de la instancia y construye un contexto de
    evaluacion (``ContextoEvaluacion``) en modo CPU para invocar
    ``lambda_penal_capacidad_por_defecto``. El contexto se descarta al
    salir; el unico valor que importa aqui es el escalar ``lambda``.

    Importacion local: evita activar el backend GPU en el proceso padre y
    minimiza acoplamientos cuando ``correr_grid`` se ejecuta sin instancia
    cargada.
    """
    # Importaciones locales: nos aseguramos de que cualquier carga pesada
    # (NumPy, NetworkX, modulo evaluador) ocurra dentro del worker que la
    # necesita y no en el modulo padre.
    from metacarp.instances import load_instances
    from metacarp.cargar_grafos import cargar_objeto_gexf
    from metacarp.metaheuristicas_utils import construir_contexto_para_corrida
    from metacarp.evaluador_costo import lambda_penal_capacidad_por_defecto

    data = load_instances(nombre_instancia, root=root)
    G    = cargar_objeto_gexf(nombre_instancia, root=root)
    # Construimos el contexto en CPU (no necesitamos GPU solo para leer
    # un escalar derivado de la matriz de distancias).
    ctx  = construir_contexto_para_corrida(
        data, G, nombre_instancia=nombre_instancia, usar_gpu=False, root=root,
    )
    return float(lambda_penal_capacidad_por_defecto(ctx))


# ============================================================
# Monkey-patch del selector (capa 1 del experimento, identico al strict)
# ============================================================

def aplicar_patch_selector(nombre_modulo_mh: str) -> None:
    """Reemplaza ``seleccionar_grupo_operadores_inter_intra`` dentro del
    modulo de la MH por ``seleccionar_grupo_strict``.

    Identico al patch del experimento ``strict_intra_inter_20260524``: debe
    invocarse al inicio de cada worker porque cada proceso hijo importa el
    modulo de la MH desde cero y necesita aplicar su propio patch.

    Parametros
    ----------
    nombre_modulo_mh : str
        Modulo de la metaheuristica donde se reasigna el simbolo.
        Ejemplos: ``"metacarp.recocido_simulado"``,
        ``"metacarp.busqueda_tabu_simple"``,
        ``"metacarp.abejas_simple"``, etc.
    """
    mh     = importlib.import_module(nombre_modulo_mh)
    strict = importlib.import_module("metacarp.strict_intra_inter_20260524")
    mh.seleccionar_grupo_operadores_inter_intra = strict.seleccionar_grupo_strict


# ============================================================
# Post-proceso del CSV parcial: inyecta columnas lambda_*
# ============================================================

def inyectar_columnas_lambda(
    ruta_csv: str,
    *,
    lambda_factor: float,
    lambda_actual: float,
    lambda_default: float,
) -> None:
    """Anade columnas ``lambda_factor``, ``lambda_actual`` y
    ``lambda_default`` a TODAS las filas de un CSV parcial ya escrito.

    Justificacion: SA, TS Simple y TS Reactiva NO vuelcan ``extra_csv`` a
    la fila resultante (sus implementaciones solo aceptan ``extra_csv`` en
    la firma pero lo ignoran al escribir). ABC vuelca ``lambda_capacidad``
    pero no ``extra_csv``. Cuckoo es el unico que vuelca ambos.

    Para garantizar que las columnas del grid esten presentes en TODOS los
    CSV parciales (requisito de trazabilidad del experimento), las
    inyectamos aqui despues de que el wrapper escribio su fila. Si las
    columnas ya existen (caso Cuckoo via ``extra_csv``), se sobreescriben
    con los mismos valores (operacion idempotente).

    Si el archivo no existe (corrida fallida que no llego a guardar CSV),
    la funcion no hace nada y retorna en silencio.
    """
    path = Path(ruta_csv)
    if not path.is_file():
        # Caso defensivo: la corrida fallo antes de escribir el CSV.
        return

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columnas_originales = list(reader.fieldnames or [])
        filas = list(reader)

    # Construimos la lista final de columnas: las originales primero (mismo
    # orden), seguidas de las columnas lambda solo si no existen ya.
    nuevas = ["lambda_factor", "lambda_actual", "lambda_default"]
    columnas_finales = list(columnas_originales)
    for nombre in nuevas:
        if nombre not in columnas_finales:
            columnas_finales.append(nombre)

    # Inyectamos / sobreescribimos los tres valores en cada fila.
    for fila in filas:
        fila["lambda_factor"]  = lambda_factor
        fila["lambda_actual"]  = lambda_actual
        fila["lambda_default"] = lambda_default

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columnas_finales)
        writer.writeheader()
        for fila in filas:
            writer.writerow(fila)


# ============================================================
# Consolidacion de CSVs parciales (map-reduce)
# ============================================================

def consolidar_parciales(
    dir_parciales: Path,
    salida_dir: Path,
    prefijo_csv: str,
    experimento: str,
    ydmh: str,
) -> int:
    """Fusiona los CSV parciales por instancia en un CSV final por instancia.

    Cada worker escribio a su propio archivo (sin contencion de E/S) y le
    inyectamos las columnas ``lambda_*``. Aqui agrupamos por instancia
    respetando la union de columnas y producimos un CSV final por
    instancia, conteniendo TODAS sus repeticiones y TODOS los lambda
    factors juntos (lo que facilita el analisis posterior con pandas).

    El nombre de la instancia se extrae del nombre del archivo parcial
    quitando el ``prefijo_csv`` y tomando el primer token restante. Este
    metodo es robusto cuando el prefijo contiene guiones bajos (como
    ``"tabu_simple"`` o ``"abc_simple"``), a diferencia del ``split[1]``
    usado en el common de strict.
    """
    grupos: dict[str, list[Path]] = {}
    prefijo_con_guion = f"{prefijo_csv}_"
    for parcial in sorted(dir_parciales.glob(f"{prefijo_csv}_*.csv")):
        nombre = parcial.stem
        if not nombre.startswith(prefijo_con_guion):
            # Defensivo: el glob ya filtra por prefijo, pero validamos por
            # si la convencion de nombrado cambia en el futuro.
            continue
        resto = nombre[len(prefijo_con_guion):]
        # Tras quitar el prefijo, el primer token es siempre la instancia.
        partes = resto.split("_")
        if not partes:
            continue
        instancia = partes[0]
        grupos.setdefault(instancia, []).append(parcial)

    n_finales = 0
    for instancia, archivos in grupos.items():
        ruta_final = salida_dir / f"{prefijo_csv}_{instancia}_{experimento}_{ydmh}.csv"
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
        n_finales += 1
    return n_finales


# ============================================================
# CLI y bucle principal compartido
# ============================================================

def parse_args_comun(descripcion: str) -> argparse.Namespace:
    """Parser de CLI compartido. Identico al del experimento strict, pero
    documentado aqui para autosuficiencia del modulo.

    Flags soportados:
      --salida-dir   : carpeta raiz donde se crea la subcarpeta del experimento.
      --repeticiones : numero de repeticiones por (instancia, lambda).
      --root         : carpeta raiz alternativa para buscar instancias.
      --workers      : numero de procesos paralelos (1 = secuencial).
      --instancias   : lista separada por comas para restringir el grid.
    """
    parser = argparse.ArgumentParser(description=descripcion)
    parser.add_argument(
        "--salida-dir",
        type=str,
        default="experimentos",
        help="Carpeta raiz donde se crea la subcarpeta del experimento.",
    )
    parser.add_argument(
        "--repeticiones",
        type=int,
        default=3,
        help="Numero de repeticiones por (instancia, lambda_factor).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Carpeta raiz alternativa donde buscar instancias y soluciones iniciales.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Numero de procesos paralelos. 1 = secuencial. Default = os.cpu_count().",
    )
    parser.add_argument(
        "--instancias",
        type=str,
        default=None,
        help=(
            "Lista de instancias separadas por coma para restringir el grid "
            "(util para smoke tests). Si se omite, se corren las 23 default."
        ),
    )
    return parser.parse_args()


def correr_grid(
    *,
    label_mh: str,
    prefijo_csv: str,
    subcarpeta_destino: str,
    modulo_patchear: str,
    ejecutar_una: Callable[[TareaExp], tuple[TareaExp, str, dict | None, str | None]],
    descripcion_cli: str,
    experimento: str,
) -> None:
    """Bucle principal compartido por los 5 scripts del experimento
    ``lambda_grid_20260525``.

    Cada script aporta su propia ``ejecutar_una``, que sabe que parametros
    pasarle al wrapper de su MH. Esta funcion se encarga del resto: parseo
    de CLI, construccion del grid cartesiano (instancia x rep x lambda),
    ejecucion paralela o secuencial, consolidacion y resumen final.

    Parametros
    ----------
    label_mh : str
        Etiqueta humana de la MH (solo para imprimir en el banner).
    prefijo_csv : str
        Prefijo de los archivos CSV (p.ej. ``"sa"``, ``"tabu_simple"``).
    subcarpeta_destino : str
        Subcarpeta debajo de ``--salida-dir`` donde se guardan los CSVs.
    modulo_patchear : str
        Modulo de la MH donde aplicar el monkey-patch del selector.
    ejecutar_una : Callable
        Funcion del script especifico que ejecuta UNA corrida.
    descripcion_cli : str
        Descripcion mostrada por ``--help``.
    experimento : str
        Etiqueta usada como sufijo de los CSV finales consolidados.
    """
    args = parse_args_comun(descripcion_cli)

    salida_dir = Path(args.salida_dir).expanduser().resolve() / subcarpeta_destino
    salida_dir.mkdir(parents=True, exist_ok=True)
    dir_parciales = salida_dir / "_partials"
    dir_parciales.mkdir(parents=True, exist_ok=True)
    ydmh = datetime.now().strftime("%Y%d%m%H%M")
    # Timestamp compacto reutilizable en el nombre de cada parcial. Lo
    # generamos UNA vez aqui y lo pasamos a las tareas para que todas las
    # corridas del mismo lanzamiento compartan el mismo sufijo temporal.
    timestamp_corto = datetime.now().strftime("%Y%m%d%H%M%S")

    # --- FILTRO OPCIONAL DE INSTANCIAS ---
    # Permite al usuario restringir el grid via ``--instancias gdb19,kshs1``.
    # Util para smoke tests donde no queremos correr las 23 instancias.
    if args.instancias:
        seleccion = [s.strip() for s in args.instancias.split(",") if s.strip()]
        instancias_efectivas = [i for i in INSTANCIAS if i in seleccion]
        # Si el usuario pasa nombres que NO estan en la lista default, los
        # respetamos igual: puede querer correr una instancia adicional.
        for nombre in seleccion:
            if nombre not in instancias_efectivas:
                instancias_efectivas.append(nombre)
    else:
        instancias_efectivas = list(INSTANCIAS)

    # --- CONSTRUCCION DE TAREAS ---
    # Producto cartesiano: instancia x repeticion x lambda_factor. El orden
    # de los bucles importa: queremos que las tareas del mismo lambda_factor
    # NO se agrupen consecutivamente en el ProcessPoolExecutor (mejor
    # balance de carga si las distintas lambdas tienen costos distintos).
    tareas: list[TareaExp] = []
    for instancia in instancias_efectivas:
        for rep in range(1, args.repeticiones + 1):
            for lf in LAMBDA_FACTORS:
                parcial = dir_parciales / (
                    f"{prefijo_csv}_{instancia}_rep{rep}_lf{lf:.1f}_{timestamp_corto}.csv"
                )
                tareas.append(TareaExp(
                    instancia=instancia,
                    repeticion=rep,
                    lambda_factor=lf,
                    root=args.root,
                    ruta_csv_parcial=str(parcial),
                ))

    total = len(tareas)
    print("=" * 80)
    print(f"{label_mh}  -  Variante experimental lambda_grid_20260525")
    print("=" * 80)
    print(f"Instancias        : {len(instancias_efectivas)}")
    print(f"Repeticiones      : {args.repeticiones}")
    print(f"Lambda factors    : {LAMBDA_FACTORS}")
    print(f"Workers           : {args.workers}")
    print(f"Corridas          : {total}  ({len(instancias_efectivas)} x {args.repeticiones} x {len(LAMBDA_FACTORS)})")
    print(f"Selector          : binario estricto (intra cuando factible / inter cuando viola)")
    print(f"Operadores activos: 5 (2 intra + 3 inter)")
    print(f"Modulo patcheado  : {modulo_patchear}")
    print(f"Salida CSV        : {salida_dir}")
    print("-" * 80)

    total_ok = 0
    total_fail = 0

    if args.workers <= 1:
        # MODO SECUENCIAL: util para debugging. Aplicamos el patch UNA vez
        # antes del bucle (estamos en el mismo proceso para todas las tareas).
        aplicar_patch_selector(modulo_patchear)
        for tarea in tareas:
            _, estado, info, err = ejecutar_una(tarea)
            if estado == "ok" and info is not None:
                print(
                    f"  [{tarea.instancia}] rep={tarea.repeticion} "
                    f"lf={tarea.lambda_factor:.1f} "
                    f"| costo={info['costo']:.4f} "
                    f"| t={info['tiempo']:.2f}s "
                    f"| kicks={info.get('n_resets', 0)} "
                    f"| lam={info.get('lambda_actual', 0.0):.2f}"
                )
                total_ok += 1
            else:
                print(
                    f"  [{tarea.instancia}] rep={tarea.repeticion} "
                    f"lf={tarea.lambda_factor:.1f} | FAIL: {err}"
                )
                total_fail += 1
    else:
        # MODO PARALELO: cada worker aplica su propio patch al arrancar
        # (esa responsabilidad recae en ``ejecutar_una``, que llama a
        # ``aplicar_patch_selector`` dentro del proceso hijo).
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futuros = {executor.submit(ejecutar_una, t): t for t in tareas}
            for fut in as_completed(futuros):
                tarea, estado, info, err = fut.result()
                if estado == "ok" and info is not None:
                    print(
                        f"  [{tarea.instancia}] rep={tarea.repeticion} "
                        f"lf={tarea.lambda_factor:.1f} "
                        f"| costo={info['costo']:.4f} "
                        f"| t={info['tiempo']:.2f}s "
                        f"| kicks={info.get('n_resets', 0)} "
                        f"| lam={info.get('lambda_actual', 0.0):.2f}"
                    )
                    total_ok += 1
                else:
                    print(
                        f"  [{tarea.instancia}] rep={tarea.repeticion} "
                        f"lf={tarea.lambda_factor:.1f} | FAIL: {err}"
                    )
                    total_fail += 1

    # --- FUSION DE PARCIALES ---
    print("\n" + "-" * 80)
    print(f"Consolidando CSVs parciales en {salida_dir} ...")
    n_finales = consolidar_parciales(
        dir_parciales, salida_dir, prefijo_csv, experimento, ydmh
    )
    print(f"CSVs finales generados: {n_finales}")

    print("\n" + "-" * 80)
    print(f"OK   : {total_ok}")
    print(f"FAIL : {total_fail}")
    print(f"CSV  : {salida_dir}")
    print(f"Parciales en: {dir_parciales}  (no se borran automaticamente)")
    print("-" * 80)
