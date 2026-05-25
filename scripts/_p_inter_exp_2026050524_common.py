"""
Utilidades compartidas por los 5 scripts ``run_<mh>_p_inter_exp_2026050524.py``.

Centraliza:
  - El monkey-patch que reemplaza ``generar_vecino`` o ``generar_vecino_ids``
    dentro del modulo de la MH antes de ejecutar la corrida.
  - La decision de p_inter en base a la violacion de la solucion inicial:
        viol_inicial > 1e-12  → p_inter = 0.95
        viol_inicial == 0     → p_inter = 0.50
  - El dataclass ``TareaExp`` (sin semilla determinista: cada repeticion
    explora una trayectoria estocastica distinta, igual que en los grids
    recientes de RTS, ABC y Cuckoo).
  - Paralelismo con ProcessPoolExecutor y consolidacion map-reduce de los
    CSVs parciales por instancia.

No duplica ninguna logica de las 5 MH ni del modulo ``vecindarios``: las
MH se invocan tal cual via sus wrappers ``*_desde_instancia``, con los
parametros estandar y solo ``p_inter`` (y opcionalmente ``alpha_inter``)
ajustados a la politica experimental.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# Las 23 instancias pequenas del corpus actual de MetaCARP (mismas que los
# grids run_*_automatico.py recientes). Se mantiene el orden alfa-numerico
# alterado para que las kshs (mas pesadas) se distribuyan al principio y
# no terminen todas en el mismo worker en modo paralelo.
INSTANCIAS = [
    "gdb19", "kshs1", "kshs2", "kshs3", "kshs4", "kshs5", "kshs6",
    "gdb4",  "gdb14", "gdb15", "gdb1",  "gdb20", "gdb3",  "gdb6",
    "gdb7",  "gdb12", "gdb10", "gdb2",  "gdb5",  "gdb13", "gdb16",
    "gdb17", "gdb21",
]


# ============================================================
# Politica de p_inter por la solucion inicial
# ============================================================

# Umbral numerico para considerar una solucion "factible". Coincide con el
# que usa ``seleccionar_grupo_operadores_inter_intra`` en metacarp.
TOL_VIOLACION = 1e-12

# Valores fijos de la politica experimental.
P_INTER_FACTIBLE   = 0.5    # cuando la inicial es factible
P_INTER_INFACTIBLE = 0.95   # cuando la inicial viola capacidad


def decidir_p_inter(viol_inicial: float) -> float:
    """Devuelve el p_inter estatico a aplicar durante toda la corrida.

    La decision se toma UNA sola vez por corrida, en base a la violacion
    de capacidad de la solucion inicial cargada desde el pickle.
    """
    return P_INTER_INFACTIBLE if viol_inicial > TOL_VIOLACION else P_INTER_FACTIBLE


# ============================================================
# Dataclass de tarea
# ============================================================

@dataclass(frozen=True)
class TareaExp:
    """Una corrida del grid experimental p_inter_exp_2026050524.

    NO incluye ``semilla``: cada repeticion arranca con la semilla del
    sistema (aleatoria) para muestrear trayectorias independientes, igual
    que en los grids recientes de RTS, ABC y Cuckoo.
    """
    instancia: str
    repeticion: int
    root: str | None
    ruta_csv_parcial: str


# ============================================================
# Monkey-patch del generador de vecinos
# ============================================================

def aplicar_patch_vecindario(
    nombre_modulo_mh: str,
    simbolo: str,
) -> None:
    """Reemplaza ``generar_vecino`` o ``generar_vecino_ids`` dentro del
    modulo de la MH por su contraparte experimental.

    Debe llamarse al INICIO de cada worker (cada proceso hijo importa el
    modulo de la MH desde cero y necesita aplicar su propio patch).

    Parametros
    ----------
    nombre_modulo_mh : str
        Modulo de la metaheuristica donde se reasigna el simbolo.
        Ejemplos: ``"metacarp.recocido_simulado"``, ``"metacarp.abejas_simple"``.
    simbolo : str
        Nombre del simbolo a reemplazar: ``"generar_vecino"`` (labels-based,
        usado por SA/TS/RTS/Cuckoo) o ``"generar_vecino_ids"`` (ids-based,
        usado solo por ABC simple).
    """
    mod = importlib.import_module(nombre_modulo_mh)
    exp = importlib.import_module("metacarp.vecindarios_p_inter_exp_2026050524")
    if simbolo == "generar_vecino":
        nuevo = exp.generar_vecino_exp
    elif simbolo == "generar_vecino_ids":
        nuevo = exp.generar_vecino_ids_exp
    else:
        raise ValueError(
            f"simbolo desconocido: {simbolo!r}. "
            "Usa 'generar_vecino' o 'generar_vecino_ids'."
        )
    setattr(mod, simbolo, nuevo)


# ============================================================
# Evaluacion de la violacion de la solucion inicial
# ============================================================

def calcular_violacion_inicial(
    nombre_instancia: str,
    root: str | None,
) -> float:
    """Carga la solucion inicial y devuelve su exceso de capacidad.

    Usa los mismos helpers oficiales que las 5 MH
    (``construir_contexto_para_corrida`` + ``seleccionar_mejor_inicial_rapido``)
    para que la violacion calculada aqui coincida bit a bit con la que veria
    la MH al arrancar la corrida.
    """
    # Imports diferidos: solo se necesitan dentro del worker.
    from metacarp.cargar_grafos import cargar_objeto_gexf
    from metacarp.cargar_soluciones_iniciales import cargar_solucion_inicial
    from metacarp.instances import load_instances
    from metacarp.metaheuristicas_utils import (
        construir_contexto_para_corrida,
        seleccionar_mejor_inicial_rapido,
    )

    data = load_instances(nombre_instancia, root=root)
    G = cargar_objeto_gexf(nombre_instancia, root=root)
    inicial_obj = cargar_solucion_inicial(nombre_instancia, root=root)
    ctx = construir_contexto_para_corrida(
        data, G, nombre_instancia=nombre_instancia, usar_gpu=False, root=root
    )
    sel = seleccionar_mejor_inicial_rapido(inicial_obj, ctx)
    return float(sel.violacion_capacidad)


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

    Cada worker escribio a su propio archivo (sin contencion de E/S);
    aqui agregamos todas las filas por instancia respetando la union de
    columnas (en caso de que algun parcial tuviera columnas adicionales).
    """
    grupos: dict[str, list[Path]] = {}
    for parcial in sorted(dir_parciales.glob(f"{prefijo_csv}_*.csv")):
        # Nombre con formato ``<prefijo>_<instancia>_<pid>_<idx>.csv``
        partes = parcial.stem.split("_")
        if len(partes) < 3:
            continue
        instancia = partes[1]
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
    """Parser de CLI compartido. Cada script puede ampliarlo si necesita
    flags propios, pero por defecto basta con estos cuatro."""
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
        default=5,
        help="Numero de repeticiones por instancia (cada una con semilla aleatoria).",
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
    return parser.parse_args()


def correr_grid(
    *,
    label_mh: str,
    prefijo_csv: str,
    subcarpeta_destino: str,
    wrapper_fn: Callable[..., Any],
    modulo_patchear: str,
    simbolo_patchear: str,
    ejecutar_una: Callable[[TareaExp], tuple[TareaExp, str, dict | None, str | None]],
    descripcion_cli: str,
    experimento: str,
) -> None:
    """Bucle principal compartido por los 5 scripts.

    Cada script aporta su propia ``ejecutar_una``, que sabe que parametros
    pasarle al wrapper de su MH (alpha en SA, tenure en TS, etc.). Esta
    funcion se encarga del resto: parseo de CLI, construccion de tareas,
    ejecucion paralela o secuencial, consolidacion y resumen final.
    """
    args = parse_args_comun(descripcion_cli)

    salida_dir = Path(args.salida_dir).expanduser().resolve() / subcarpeta_destino
    salida_dir.mkdir(parents=True, exist_ok=True)
    dir_parciales = salida_dir / "_partials"
    dir_parciales.mkdir(parents=True, exist_ok=True)
    ydmh = datetime.now().strftime("%Y%d%m%H%M")

    # --- CONSTRUCCION DE TAREAS ---
    # Una tarea por (instancia, repeticion). Sin semilla determinista.
    tareas: list[TareaExp] = []
    for instancia in INSTANCIAS:
        for rep in range(1, args.repeticiones + 1):
            idx = len(tareas)
            parcial = dir_parciales / (
                f"{prefijo_csv}_{instancia}_{os.getpid()}_{idx}.csv"
            )
            tareas.append(TareaExp(
                instancia=instancia,
                repeticion=rep,
                root=args.root,
                ruta_csv_parcial=str(parcial),
            ))

    total = len(tareas)
    print("=" * 80)
    print(f"{label_mh}  —  Variante experimental p_inter_exp_2026050524")
    print("=" * 80)
    print(f"Instancias        : {len(INSTANCIAS)}")
    print(f"Repeticiones      : {args.repeticiones}")
    print(f"Workers           : {args.workers}")
    print(f"Corridas          : {total}")
    print(f"Politica p_inter  : {P_INTER_FACTIBLE} (factible) / "
          f"{P_INTER_INFACTIBLE} (infactible inicial)")
    print(f"Vecindario        : {simbolo_patchear} → variante exp (preserva 1a tarea)")
    print(f"Salida CSV        : {salida_dir}")
    print("-" * 80)

    total_ok = 0
    total_fail = 0

    if args.workers <= 1:
        # MODO SECUENCIAL: util para debugging. Aplicamos el patch UNA vez
        # antes del bucle (estamos en el mismo proceso para todas las tareas).
        aplicar_patch_vecindario(modulo_patchear, simbolo_patchear)
        for tarea in tareas:
            _, estado, info, err = ejecutar_una(tarea)
            if estado == "ok" and info is not None:
                print(
                    f"  [{tarea.instancia}] rep={tarea.repeticion} "
                    f"p_inter={info.get('p_inter', '?')} "
                    f"viol_ini={info.get('viol_inicial', 0.0):.2f} "
                    f"| costo={info['costo']:.4f} | t={info['tiempo']:.2f}s"
                )
                total_ok += 1
            else:
                print(f"  [{tarea.instancia}] rep={tarea.repeticion} | FAIL: {err}")
                total_fail += 1
    else:
        # MODO PARALELO: cada worker aplica su propio patch al arrancar.
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futuros = {executor.submit(ejecutar_una, t): t for t in tareas}
            for fut in as_completed(futuros):
                tarea, estado, info, err = fut.result()
                if estado == "ok" and info is not None:
                    print(
                        f"  [{tarea.instancia}] rep={tarea.repeticion} "
                        f"p_inter={info.get('p_inter', '?')} "
                        f"viol_ini={info.get('viol_inicial', 0.0):.2f} "
                        f"| costo={info['costo']:.4f} | t={info['tiempo']:.2f}s"
                    )
                    total_ok += 1
                else:
                    print(f"  [{tarea.instancia}] rep={tarea.repeticion} | FAIL: {err}")
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
