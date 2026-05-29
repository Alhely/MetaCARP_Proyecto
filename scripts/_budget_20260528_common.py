"""
Utilidades compartidas por los 5 scripts ``run_<mh>_budget_20260528.py``.

Experimento ``budget_20260528`` — dos cambios sobre el baseline p_inter_pr:

  1. PRESUPUESTO x25:
       - TS Simple / RTS   : ``iteraciones_max`` 400 → 10 000
       - ABC / Cuckoo      : ``factor_iter`` x10  (~300 → ~3 000-10 000 segun n)
       - SA                : reheat mas largo (patience 10→100, reheats 5→30,
                             T_min 1e-3→1e-4)
       - Kick proporcional : ``max_iter_sin_mejora_kick`` 30 → 300
       - Sin tope de resets: ``max_resets=None`` (el limite es la iteracion dura)

  2. LAMBDA x10:
       Monkey-patch sobre ``evaluador_costo.lambda_penal_capacidad_por_defecto``
       que devuelve 10x el valor actual (median_arco * 10 → median_arco * 100).
       Encarece mas la infactibilidad; si el presupuesto extra no cierra el gap,
       el lambda elevado garantiza que las soluciones finales sean factibles.

Todos los demas patches (selector p_inter probabilistico + Path Relinking
truncado) se conservan intactos desde ``_p_inter_pr_20260528_common.py``.
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

# ── Constantes heredadas del experimento p_inter_pr ───────────────────────────
P_INTER: float     = 0.20
ALPHA_INTER: float = 0.80
P_PR: float        = 0.50

# ── Nuevos parámetros de presupuesto ─────────────────────────────────────────
ITERS_TS: int         = 10_000   # iteraciones_max para TS Simple y RTS
MAX_SIN_MEJ_TS: int   = 2_500    # 25 % del presupuesto TS (era 100/400 = 25 %)

# factor_iter es el COEFICIENTE de la formula max(200, coef * n_tareas).
# El default del algoritmo es 20 (→ max(200, 20·n)).  Para obtener ~10x
# el default usamos 200 (→ max(200, 200·n)). NO es un multiplicador del
# total de iteraciones sino el coeficiente por n_tareas.
FACTOR_ITER_POP: int      = 200
# max_iter_sin_mejora para ABC/Cuckoo: escalar con el mismo coeficiente.
# Default: max(50, 3·n). Con coef x10: max(500, 30·n). Usamos 500 como
# minimo de la misma formula reescalada, apropiado para n <= 50.
MAX_SIN_MEJ_POP: int      = 500

KICK_ITERS: int       = 300      # max_iter_sin_mejora_kick (era 30)
MAX_RESETS            = None     # sin tope de resets; el limite es ITERS_TS

LAMBDA_FACTOR: float  = 10.0     # amplificador del lambda default instance-aware

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
    """Una corrida del grid experimental budget_20260528.

    Sin semilla fija: cada repeticion usa semilla del sistema para
    muestrear trayectorias independientes.
    """
    instancia: str
    repeticion: int
    root: str | None
    ruta_csv_parcial: str


# ============================================================
# Patches
# ============================================================

def aplicar_patch_lambda(factor: float = LAMBDA_FACTOR) -> None:
    """Sustituye lambda_penal_capacidad_por_defecto por una version x``factor``.

    Al llamarse antes de cualquier importacion de MH, todos los modulos que
    lean ``evaluador_costo.lambda_penal_capacidad_por_defecto`` obtendrán el
    valor amplificado automaticamente cuando su ``lambda_capacidad=None``.
    """
    import metacarp.evaluador_costo as _ec
    _orig = _ec.lambda_penal_capacidad_por_defecto

    def _lambda_amplificado(ctx):
        return _orig(ctx) * factor

    _ec.lambda_penal_capacidad_por_defecto = _lambda_amplificado


def aplicar_patches_budget(
    nombre_modulo_mh: str,
    p_pr: float = P_PR,
    lambda_factor: float = LAMBDA_FACTOR,
) -> None:
    """Instala los tres patches del experimento en orden:

      1. Lambda x``lambda_factor``: encarece infactibilidad globalmente.
      2. PR (``aplicar_patch_pr``): strict selector + captura mejor_sol
         + kick con PR con probabilidad ``p_pr``.
      3. p_inter (``aplicar_patch_p_inter``): sobreescribe el selector
         strict del paso 2 con el probabilistico (P_INTER=0.20).

    Mismo orden y logica que ``aplicar_patch_completo`` en
    ``_p_inter_pr_20260528_common``, mas el patch de lambda.
    """
    # 1) Lambda amplificado — debe ir primero para que los modulos MH lo
    #    lean cuando construyan su ContextoEvaluacion.
    aplicar_patch_lambda(lambda_factor)

    # 2) PR: selector strict + captura mejor_sol + kick+PR.
    from metacarp.path_relinking_20260528 import aplicar_patch_pr
    aplicar_patch_pr(nombre_modulo_mh, p_pr=p_pr)

    # 3) p_inter: sobreescribe el selector strict que dejo el paso 2.
    from metacarp.p_inter_pr_20260528 import aplicar_patch_p_inter
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
    """Fusiona los CSV parciales por instancia en un CSV final."""
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
# CLI y bucle principal compartido
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
    """Bucle principal compartido por los 5 scripts del experimento budget."""
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
    print(f"{label_mh}  —  Experimento budget_20260528")
    print("=" * 80)
    print(f"Instancias        : {len(instancias_efectivas)}")
    print(f"Repeticiones      : {args.repeticiones}")
    print(f"Workers           : {args.workers}")
    print(f"Corridas          : {total}")
    print(f"Budget TS/RTS     : {ITERS_TS:,} iters  |  kick cada {KICK_ITERS} iters sin mejora global")
    print(f"Budget ABC/Cuckoo : factor_iter={FACTOR_ITER_POP} (coef*n_tareas)  |  kick cada {KICK_ITERS} iters")
    print(f"max_resets        : {MAX_RESETS!r}  (None = sin tope)")
    print(f"Lambda            : x{LAMBDA_FACTOR} el default instance-aware")
    print(f"Selector          : p_inter probabilistico (p_inter={P_INTER}, alpha_inter={ALPHA_INTER})")
    print(f"PR (capa 3)       : p_pr={p_pr}")
    print(f"Modulo patcheado  : {modulo_patchear}")
    print(f"Salida CSV        : {salida_dir}")
    print("-" * 80)

    total_ok = 0
    total_fail = 0

    if args.workers <= 1:
        aplicar_patches_budget(modulo_patchear, p_pr=p_pr)
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
