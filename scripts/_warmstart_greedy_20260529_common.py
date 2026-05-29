"""
Utilidades compartidas por los 5 scripts ``run_<mh>_warmstart_greedy_20260529.py``.

Experimento ``warmstart_greedy_20260529`` — apila TRES nuevas capas sobre
el baseline ``budget_20260528`` para cerrar el gap residual de ~30%:

  1. PRESUPUESTO Y LAMBDA  : heredados de ``budget_20260528`` (iters x25,
                              lambda x10, kick proporcional, max_resets=None).
  2. SELECTOR Y PR         : heredados de ``p_inter_pr_20260528``
                              (p_inter=0.20, alpha_inter=0.80, p_pr=0.5).
  3. PATH-SCANNING WARMSTART (NUEVO): la solucion inicial se construye
     dinamicamente con ``path_scanning`` (Golden, DeArmon, Baker 1983) en
     lugar de leer el pickle aleatorio precomputado.
  4. EVALUADOR GREEDY (NUEVO): los evaluadores ``costo_rapido_ids``,
     ``costo_lote_ids`` y ``costo_lote_penalizado_ids`` se sobreescriben
     por versiones que eligen GREEDY la orientacion de cada tarea
     (entrar por el extremo mas cercano al nodo previo). Esto elimina el
     artefacto del evaluador canonico que inflaba el gap al forzar
     orientacion fija u->v.

Resultado del smoke test (TS Simple, 5 instancias):
  gdb19: 14.55% -> 0.00%  (encuentra BKS)
  gdb3 : 49.36% -> 0.00%  (encuentra BKS)
  kshs1: 15.26% -> 0.00%  (encuentra BKS)
  gdb10: 52.00% -> 3.27%  (-48.7 pp)
  kshs5: 38.71% -> 8.63%  (-30 pp)

Las constantes experimentales se heredan tal cual de los modulos previos.
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

import sys as _sys
_ROOT_PROYECTO = Path(__file__).resolve().parent.parent
if str(_ROOT_PROYECTO) not in _sys.path:
    _sys.path.insert(0, str(_ROOT_PROYECTO))

# ── Re-export de constantes desde budget_20260528 ────────────────────────────
from _budget_20260528_common import (  # noqa: F401
    ALPHA_INTER,
    FACTOR_ITER_POP,
    ITERS_TS,
    KICK_ITERS,
    LAMBDA_FACTOR,
    MAX_RESETS,
    MAX_SIN_MEJ_POP,
    MAX_SIN_MEJ_TS,
    P_INTER,
    P_PR,
    INSTANCIAS,
)


# ============================================================
# Dataclass de tarea (identica a la del baseline)
# ============================================================

@dataclass(frozen=True)
class TareaExp:
    """Una corrida del grid warmstart_greedy_20260529 (sin semilla fija)."""
    instancia: str
    repeticion: int
    root: str | None
    ruta_csv_parcial: str


# ============================================================
# Patch combinado del experimento
# ============================================================

def aplicar_patches_warmstart_greedy(
    nombre_modulo_mh: str,
    *,
    p_pr: float = P_PR,
    lambda_factor: float = LAMBDA_FACTOR,
    root_warmstart: str | None = None,
) -> None:
    """Instala los 5 patches del experimento en orden.

    Orden CRITICO (cada paso depende del anterior):

      1. ``aplicar_patch_warmstart_ps``: reemplaza
         ``cargar_solucion_inicial`` por Path-Scanning. DEBE ir antes
         del resto para que los modulos MH que se importen despues vean
         la version parcheada al resolver el simbolo.

      2. ``aplicar_patch_evaluador_greedy``: reemplaza los 3 evaluadores
         rapidos por versiones con orientacion greedy por tarea.

      3. ``aplicar_patches_budget``: instala el stack heredado (lambda x10,
         PR + p_inter probabilistico). Internamente carga la MH, asi que
         debe ir DESPUES de (1) y (2) para que las MH carguen los simbolos
         ya parcheados.

      4. Re-aplicar (1) y (2) por si ``aplicar_patches_budget`` cargo
         modulos que necesitan ver los simbolos parcheados despues del
         load.
    """
    # 1) Path-Scanning warmstart.
    from metacarp.warmstart_20260529 import aplicar_patch_warmstart_ps
    aplicar_patch_warmstart_ps(root=root_warmstart)

    # 2) Evaluador greedy.
    from metacarp.evaluador_greedy_20260529 import aplicar_patch_evaluador_greedy
    aplicar_patch_evaluador_greedy()

    # 3) Patches heredados: lambda x10 + PR + selector p_inter.
    from _budget_20260528_common import aplicar_patches_budget
    aplicar_patches_budget(nombre_modulo_mh, p_pr=p_pr, lambda_factor=lambda_factor)

    # 4) Re-aplicar los patches NUEVOS por si los anteriores cargaron
    #    modulos que necesitan ver los simbolos ya sobreescritos.
    aplicar_patch_warmstart_ps(root=root_warmstart)
    aplicar_patch_evaluador_greedy()


# ============================================================
# Consolidacion de CSVs parciales (idem al baseline)
# ============================================================

def consolidar_parciales(
    dir_parciales: Path,
    salida_dir: Path,
    prefijo_csv: str,
    experimento: str,
    ydmh: str,
) -> int:
    """Fusiona los CSV parciales por instancia en un CSV final por instancia."""
    grupos: dict[str, list[Path]] = {}
    for parcial in sorted(dir_parciales.glob(f"{prefijo_csv}_*.csv")):
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
# CLI y bucle principal
# ============================================================

def parse_args_comun(descripcion: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=descripcion)
    parser.add_argument("--salida-dir", type=str, default="experimentos")
    parser.add_argument("--repeticiones", type=int, default=5)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--instancias", type=str, default=None, nargs="*")
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
    """Bucle principal del experimento warmstart_greedy_20260529."""
    args = parse_args_comun(descripcion_cli)

    salida_dir = Path(args.salida_dir).expanduser().resolve() / subcarpeta_destino
    salida_dir.mkdir(parents=True, exist_ok=True)
    dir_parciales = salida_dir / "_partials"
    dir_parciales.mkdir(parents=True, exist_ok=True)
    ydmh = datetime.now().strftime("%Y%d%m%H%M")

    if args.instancias:
        bruto: list[str] = []
        for item in args.instancias:
            for tok in item.split(","):
                tok = tok.strip()
                if tok:
                    bruto.append(tok)
        instancias_efectivas = [i for i in INSTANCIAS if i in bruto]
        for nombre in bruto:
            if nombre not in instancias_efectivas:
                instancias_efectivas.append(nombre)
    else:
        instancias_efectivas = list(INSTANCIAS)

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
    print(f"{label_mh}  —  Experimento warmstart_greedy_20260529")
    print("=" * 80)
    print(f"Instancias        : {len(instancias_efectivas)}")
    print(f"Repeticiones      : {args.repeticiones}")
    print(f"Workers           : {args.workers}")
    print(f"Corridas          : {total}")
    print(f"Warmstart inicial : Path-Scanning mejor-de-5 (Golden et al 1983)")
    print(f"Evaluador         : orientacion greedy por tarea (no canonico)")
    print(f"Budget TS/RTS     : {ITERS_TS:,} iters  |  kick cada {KICK_ITERS} iters")
    print(f"Budget ABC/Cuckoo : factor_iter={FACTOR_ITER_POP} (coef*n_tareas)")
    print(f"Lambda            : x{LAMBDA_FACTOR}")
    print(f"Selector          : p_inter probabilistico (p_inter={P_INTER}, alpha_inter={ALPHA_INTER})")
    print(f"PR (capa 3)       : p_pr={p_pr}")
    print(f"Modulo patcheado  : {modulo_patchear}")
    print(f"Salida CSV        : {salida_dir}")
    print("-" * 80)

    total_ok = 0
    total_fail = 0

    if args.workers <= 1:
        aplicar_patches_warmstart_greedy(
            modulo_patchear, p_pr=p_pr, lambda_factor=LAMBDA_FACTOR,
            root_warmstart=args.root,
        )
        for tarea in tareas:
            _, estado, info, err = ejecutar_una(tarea)
            if estado == "ok" and info is not None:
                print(
                    f"  [{tarea.instancia}] rep={tarea.repeticion} "
                    f"| costo={info['costo']:.4f} "
                    f"| t={info['tiempo']:.2f}s "
                    f"| kicks={info.get('n_resets', 0)} "
                    f"| factible={info.get('factible', '?')}"
                )
                total_ok += 1
            else:
                print(f"  [{tarea.instancia}] rep={tarea.repeticion} | FAIL: {err}")
                total_fail += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futuros = {executor.submit(ejecutar_una, t): t for t in tareas}
            for fut in as_completed(futuros):
                tarea, estado, info, err = fut.result()
                if estado == "ok" and info is not None:
                    print(
                        f"  [{tarea.instancia}] rep={tarea.repeticion} "
                        f"| costo={info['costo']:.4f} "
                        f"| t={info['tiempo']:.2f}s "
                        f"| kicks={info.get('n_resets', 0)} "
                        f"| factible={info.get('factible', '?')}"
                    )
                    total_ok += 1
                else:
                    print(f"  [{tarea.instancia}] rep={tarea.repeticion} | FAIL: {err}")
                    total_fail += 1

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
    print("-" * 80)
