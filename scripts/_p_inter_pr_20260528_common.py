"""
Utilidades compartidas por los 5 scripts ``run_<mh>_p_inter_pr_20260528.py``.

Este experimento combina:
  1. SELECTOR ``p_inter`` PROBABILISTICO (NUEVO) en lugar del binario estricto.
     - En estado factible (violacion <= 0): proponer INTER con probabilidad
       ``P_INTER=0.20`` y INTRA con probabilidad ``1-P_INTER=0.80``.
     - En estado infactible: proponer INTER con probabilidad
       ``ALPHA_INTER=0.80`` (reparacion agresiva, igual que baseline).
  2. KICK REACTIVO al estancamiento (igual que strict + PR previos).
  3. PATH RELINKING truncado tras el kick, con probabilidad ``P_PR=0.50``
     de PR-vs-kick-puro (igual que el experimento PR validado en Seccion 12).

Justificacion del experimento (Seccion 12 del notebook):
  - Los operadores INTER tienen tasa de mejora 4-7x superior a los INTRA
    incluso en estado factible.
  - El selector binario estricto NUNCA propone inter en estado factible,
    perdiendo esa ganancia.
  - Con p_inter=0.20 capturamos ~20% de propuestas inter en estado factible
    sin caer en el extremo agresivo (p_inter > 0.5) que rompio el experimento
    ``p_inter_exp_2026050524``.

Centraliza:
  - El monkey-patch ``aplicar_patch_completo`` que instala los 3 patches en
    el orden correcto:
      1) PR (instala strict + captura mejor_sol + kick aumentado)
      2) p_inter (sobreescribe el selector strict con el probabilistico)
    Los dos patches son ORTOGONALES (actuan sobre funciones distintas) salvo
    por el selector, donde el segundo patch gana.
  - El dataclass ``TareaExp`` (sin semilla determinista).
  - Paralelismo con ProcessPoolExecutor y consolidacion map-reduce de los
    CSVs parciales por instancia.
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
# la carpeta ``scripts/``.
import sys as _sys
_ROOT_PROYECTO = Path(__file__).resolve().parent.parent
if str(_ROOT_PROYECTO) not in _sys.path:
    _sys.path.insert(0, str(_ROOT_PROYECTO))

# --- Constantes experimentales ---
# Probabilidad de proponer INTER en estado factible. Moderada (0.20) para
# capturar la alta tasa de mejora de los inter sin caer en el extremo
# agresivo del experimento anterior fallido.
P_INTER: float = 0.20

# Probabilidad de proponer INTER cuando la solucion VIOLA capacidad. Mismo
# valor que el baseline original; preservamos la reparacion agresiva.
ALPHA_INTER: float = 0.80

# Probabilidad de disparar PR cada vez que se ejecuta un kick. Mismo valor
# que el experimento PR validado en la Seccion 12 del notebook.
P_PR: float = 0.50

# Las 23 instancias pequenas del corpus actual de MetaCARP (mismas que los
# grids strict, AOS, PR recientes). El orden alfa-numerico alterado coloca
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
    """Una corrida del grid experimental p_inter_pr_20260528.

    NO incluye ``semilla``: cada repeticion arranca con la semilla del
    sistema (aleatoria) para muestrear trayectorias independientes, igual
    que en los grids recientes strict, aos_pm y PR.
    """
    instancia: str
    repeticion: int
    root: str | None
    ruta_csv_parcial: str


# ============================================================
# Patch combinado: PR + selector p_inter probabilistico
# ============================================================

def aplicar_patch_completo(
    nombre_modulo_mh: str,
    p_pr: float = P_PR,
) -> None:
    """Instala los dos patches del experimento en el orden correcto.

    Orden CRITICO:
      1. ``aplicar_patch_pr(modulo, p_pr)`` (de ``path_relinking_20260528``).
         Internamente:
           - Reemplaza ``seleccionar_grupo_operadores_inter_intra`` en el
             modulo MH por ``seleccionar_grupo_strict`` (binario estricto).
           - Reemplaza ``aplicar_kick_labels``/``aplicar_kick_ids`` en
             ``metacarp.strict_intra_inter_20260524`` por versiones que
             con probabilidad ``p_pr`` ejecutan PR tras el kick.
           - Instala captura de la mejor solucion global (guia de PR)
             via patch a ``copiar_solucion_labels`` y a
             ``ContadorOperadores.registrar_mejora``.
      2. ``aplicar_patch_p_inter(modulo)`` (de ``p_inter_pr_20260528``).
         SOBREESCRIBE el selector binario estricto que dejo el paso 1
         con el wrapper probabilistico (p_inter=0.20, alpha_inter=0.80).

    Como los dos patches actuan sobre funciones DISTINTAS salvo por el
    selector (donde el segundo gana), esta composicion produce:
      - Selector p_inter probabilistico (paso 2 sobreescribio al strict).
      - Kick aumentado con PR (paso 1, no tocado por paso 2).
      - Captura de mejor_sol (paso 1, no tocado por paso 2).

    Parametros
    ----------
    nombre_modulo_mh : str
        Modulo de la MH donde reasignar el selector (p.ej.
        ``"metacarp.recocido_simulado"``).
    p_pr : float
        Probabilidad de disparar PR cada vez que se ejecuta un kick.
    """
    # Importacion diferida: solo cargamos los modulos de los patches cuando
    # se llaman desde el worker (cada proceso hijo importa lo suyo).
    from metacarp.path_relinking_20260528 import aplicar_patch_pr
    from metacarp.p_inter_pr_20260528 import aplicar_patch_p_inter

    # 1) PR: instala selector strict + captura mejor_sol + kick+PR.
    aplicar_patch_pr(nombre_modulo_mh, p_pr=p_pr)

    # 2) p_inter: sobreescribe el selector strict con el probabilistico.
    aplicar_patch_p_inter(nombre_modulo_mh)


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
    """Parser de CLI compartido. Identico al de los grids strict/PR."""
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
        nargs="*",
        help=(
            "Lista de instancias (separadas por espacios o coma) para "
            "restringir el grid (util para smoke tests). Si se omite, se "
            "corren las 23 default."
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
    p_pr: float = P_PR,
) -> None:
    """Bucle principal compartido por los 5 scripts.

    Cada script aporta su propia ``ejecutar_una``, que sabe que parametros
    pasarle al wrapper de su MH. Esta funcion se encarga del resto:
    parseo de CLI, construccion de tareas, ejecucion paralela o secuencial,
    consolidacion y resumen final.
    """
    args = parse_args_comun(descripcion_cli)

    salida_dir = Path(args.salida_dir).expanduser().resolve() / subcarpeta_destino
    salida_dir.mkdir(parents=True, exist_ok=True)
    dir_parciales = salida_dir / "_partials"
    dir_parciales.mkdir(parents=True, exist_ok=True)
    ydmh = datetime.now().strftime("%Y%d%m%H%M")

    # --- FILTRO OPCIONAL DE INSTANCIAS ---
    if args.instancias:
        bruto: list[str] = []
        for item in args.instancias:
            for tok in item.split(","):
                tok = tok.strip()
                if tok:
                    bruto.append(tok)
        # Respetamos primero el orden default, luego instancias nuevas.
        instancias_efectivas = [i for i in INSTANCIAS if i in bruto]
        for nombre in bruto:
            if nombre not in instancias_efectivas:
                instancias_efectivas.append(nombre)
    else:
        instancias_efectivas = list(INSTANCIAS)

    # --- CONSTRUCCION DE TAREAS ---
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
    print(f"{label_mh}  -  Variante experimental p_inter_pr_20260528")
    print("=" * 80)
    print(f"Instancias        : {len(instancias_efectivas)}")
    print(f"Repeticiones      : {args.repeticiones}")
    print(f"Workers           : {args.workers}")
    print(f"Corridas          : {total}")
    print(f"Selector          : p_inter probabilistico (p_inter={P_INTER}, alpha_inter={ALPHA_INTER})")
    print(f"PR (capa 3)       : truncated PR hacia mejor global, p_pr={p_pr}")
    print(f"Operadores activos: 5 (2 intra + 3 inter)")
    print(f"Modulo patcheado  : {modulo_patchear}")
    print(f"Salida CSV        : {salida_dir}")
    print("-" * 80)

    total_ok = 0
    total_fail = 0

    if args.workers <= 1:
        # MODO SECUENCIAL: util para debugging. Aplicamos el patch UNA vez
        # antes del bucle (estamos en el mismo proceso para todas las tareas).
        aplicar_patch_completo(modulo_patchear, p_pr=p_pr)
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
        # ``aplicar_patch_completo`` dentro del proceso hijo).
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
