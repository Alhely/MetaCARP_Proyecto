"""
Utilidades compartidas por los 5 scripts ``run_<mh>_aos_pm_20260527.py``.

Centraliza:
  - El monkey-patch que activa AOS Probability Matching:
    (1) reemplaza ``seleccionar_grupo_operadores_inter_intra`` dentro del
        modulo de la MH por ``seleccionar_grupo_aos``;
    (2) envuelve ``ContadorOperadores.registrar_mejora`` para que actualice
        los pesos AOS cada vez que la MH detecta una mejora del mejor global.
  - El dataclass ``TareaExp`` (sin semilla determinista: cada repeticion
    explora una trayectoria estocastica distinta).
  - Paralelismo con ProcessPoolExecutor y consolidacion map-reduce de los
    CSVs parciales por instancia.

Diferencias clave con ``_strict_intra_inter_20260524_common``:
  - Se importa ``aplicar_patch_aos`` (no ``aplicar_patch_selector``): la
    nueva funcion aplica ambos patches y REINICIA el estado AOS por corrida.
  - Se expone ``ALPHA_AOS`` como constante de modulo para que los scripts
    puedan pasarla a ``correr_grid`` y reportarla en el banner.
  - El bucle secuencial llama a ``aplicar_patch_aos(modulo, alpha=alpha_aos)``
    UNA vez antes del bucle; el modo paralelo delega esa responsabilidad a
    ``ejecutar_una`` (cada worker hace su propio patch).
"""
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

# --- Ajuste de sys.path ---
# Aseguramos que la raiz del proyecto este en sys.path para que ``metacarp``
# sea importable cuando los scripts ``run_*`` se ejecutan directamente desde
# la carpeta ``scripts/``. Replicamos el patron que usan los scripts run_*.
import sys as _sys
_ROOT_PROYECTO = Path(__file__).resolve().parent.parent
if str(_ROOT_PROYECTO) not in _sys.path:
    _sys.path.insert(0, str(_ROOT_PROYECTO))

# Reexportamos ``aplicar_patch_aos`` para que los scripts ``run_*`` puedan
# importarlo desde este modulo comun (mismo patron que el strict).
from metacarp.aos_pm_20260527 import aplicar_patch_aos  # noqa: F401, E402

# Tasa de aprendizaje por defecto del Probability Matching. Se expone como
# constante de modulo para que los scripts ``run_*`` la incluyan en
# ``extra_csv`` y en el banner.
ALPHA_AOS: float = 0.2

# Las 23 instancias pequenas del corpus actual de MetaCARP (mismas que los
# grids strict y p_inter recientes). El orden alfa-numerico alterado coloca
# las kshs (mas pesadas) al principio para distribuirlas entre workers.
INSTANCIAS = [
    "gdb19", "kshs1", "kshs2", "kshs3", "kshs4", "kshs5", "kshs6",
    "gdb4",  "gdb14", "gdb15", "gdb1",  "gdb20", "gdb3",  "gdb6",
    "gdb7",  "gdb12", "gdb10", "gdb2",  "gdb5",  "gdb13", "gdb16",
    "gdb17", "gdb21",
]


# ============================================================
# Dataclass de tarea
# ============================================================

@dataclass(frozen=True)
class TareaExp:
    """Una corrida del grid experimental aos_pm_20260527.

    NO incluye ``semilla``: cada repeticion arranca con la semilla del
    sistema (aleatoria) para muestrear trayectorias independientes, igual
    que en los grids recientes de RTS, ABC, Cuckoo y los scripts strict.
    """
    instancia: str
    repeticion: int
    root: str | None
    ruta_csv_parcial: str


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
    flags propios, pero por defecto basta con estos cuatro mas el filtro
    opcional de instancias."""
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
    alpha_aos: float = ALPHA_AOS,
) -> None:
    """Bucle principal compartido por los 5 scripts.

    Cada script aporta su propia ``ejecutar_una``, que sabe que parametros
    pasarle al wrapper de su MH (alpha en SA, tenure en TS, etc.). Esta
    funcion se encarga del resto: parseo de CLI, construccion de tareas,
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
        Modulo de la MH donde aplicar el monkey-patch del selector AOS.
    ejecutar_una : Callable
        Funcion del script especifico que ejecuta UNA corrida.
    descripcion_cli : str
        Descripcion mostrada por ``--help``.
    experimento : str
        Etiqueta usada como sufijo de los CSV finales consolidados.
    alpha_aos : float
        Tasa de aprendizaje del Probability Matching usada al patchear en
        modo secuencial. En modo paralelo, cada ``ejecutar_una`` aplica su
        propio patch (con su propio alpha).
    """
    args = parse_args_comun(descripcion_cli)

    salida_dir = Path(args.salida_dir).expanduser().resolve() / subcarpeta_destino
    salida_dir.mkdir(parents=True, exist_ok=True)
    dir_parciales = salida_dir / "_partials"
    dir_parciales.mkdir(parents=True, exist_ok=True)
    ydmh = datetime.now().strftime("%Y%d%m%H%M")

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
    # Una tarea por (instancia, repeticion). Sin semilla determinista.
    tareas: list[TareaExp] = []
    for instancia in instancias_efectivas:
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
    print(f"{label_mh}  -  Variante experimental aos_pm_20260527")
    print("=" * 80)
    print(f"Instancias        : {len(instancias_efectivas)}")
    print(f"Repeticiones      : {args.repeticiones}")
    print(f"Workers           : {args.workers}")
    print(f"Corridas          : {total}")
    print(f"Selector          : AOS Probability Matching (alpha={alpha_aos})")
    print(f"Grupo (capa 1)    : binario estricto (intra factible / inter viola)")
    print(f"Operadores activos: 5 (2 intra + 3 inter)")
    print(f"Modulo patcheado  : {modulo_patchear}")
    print(f"Salida CSV        : {salida_dir}")
    print("-" * 80)

    total_ok = 0
    total_fail = 0

    if args.workers <= 1:
        # MODO SECUENCIAL: util para debugging. Aplicamos el patch UNA vez
        # antes del bucle (estamos en el mismo proceso para todas las tareas).
        # OJO: ``aplicar_patch_aos`` REINICIA el estado AOS en cada llamada,
        # pero aqui solo la invocamos una vez, asi que el estado se mantiene
        # vivo entre tareas. Si se desea trayectorias independientes 100%,
        # cada ``ejecutar_una`` reinvocara ``aplicar_patch_aos`` antes de
        # llamar al wrapper de su MH (es lo que hacen los scripts run_*).
        aplicar_patch_aos(modulo_patchear, alpha=alpha_aos)
        for tarea in tareas:
            _, estado, info, err = ejecutar_una(tarea)
            if estado == "ok" and info is not None:
                print(
                    f"  [{tarea.instancia}] rep={tarea.repeticion} "
                    f"| costo={info['costo']:.4f} "
                    f"| t={info['tiempo']:.2f}s "
                    f"| kicks={info.get('n_resets', 0)}"
                )
                total_ok += 1
            else:
                print(f"  [{tarea.instancia}] rep={tarea.repeticion} | FAIL: {err}")
                total_fail += 1
    else:
        # MODO PARALELO: cada worker aplica su propio patch al arrancar
        # (esa responsabilidad recae en ``ejecutar_una``, que llama a
        # ``aplicar_patch_aos`` dentro del proceso hijo).
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futuros = {executor.submit(ejecutar_una, t): t for t in tareas}
            for fut in as_completed(futuros):
                tarea, estado, info, err = fut.result()
                if estado == "ok" and info is not None:
                    print(
                        f"  [{tarea.instancia}] rep={tarea.repeticion} "
                        f"| costo={info['costo']:.4f} "
                        f"| t={info['tiempo']:.2f}s "
                        f"| kicks={info.get('n_resets', 0)}"
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
