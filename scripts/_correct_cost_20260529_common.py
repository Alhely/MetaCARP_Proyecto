"""
Corrida definitiva con COSTO CORRECTO (correct_cost, 20260529).

Re-ejecuta los 5 approaches de seleccion de vecinos con el evaluador greedy
nativo de ``metacarp.evaluador_costo`` y con el reporte de deadheading ya
consistente (mejor_costo == costo_total_desde_reporte). Salida en una carpeta
nueva: ``experimentos/corridas/correct_cost/``.

El evaluador greedy es ahora el comportamiento NATIVO de ``evaluador_costo``:
no se requieren patches de runtime. Cada tarea se recorre por el extremo mas
cercano al nodo previo (orientacion dinamica), lo cual elimina el artefacto
del evaluador canonico que forzaba siempre la orientacion u->v.

OBJETIVO METODOLOGICO: aislar el comportamiento de cada estrategia de
seleccion de vecinos. Para ello, los HIPERPARAMETROS BASE de cada
metaheuristica se FIJAN a valores de LITERATURA CLASICA (ver bloque
"PARAMETROS CLASICOS DE LITERATURA" mas abajo), identicos en las 5 variantes.
La unica diferencia entre approaches es la estrategia de seleccion /
intensificacion que cada uno implementa.

Variantes (los 5 approaches tal cual; R5 conserva su presupuesto x25):
  R1 strict     <-> strict_intra_inter_20260524 (selector binario estricto)
  R2 aos        <-> aos_pm_20260527            (probability matching)
  R3 pr         <-> path_relinking_20260528    (path relinking truncado)
  R4 p_inter_pr <-> p_inter_pr_20260528        (selector probabilistico + PR)
  R5 budget     <-> budget_20260528            (PR + p_inter + presupuesto x25 + lambda x10)

CADA approach PRESERVA el warmstart pickle aleatorio (no usa Path-Scanning)
para ser comparable con el trabajo previo.
"""
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import sys as _sys
_ROOT_PROYECTO = Path(__file__).resolve().parent.parent
if str(_ROOT_PROYECTO) not in _sys.path:
    _sys.path.insert(0, str(_ROOT_PROYECTO))

# ── Constantes heredadas de experimentos previos ─────────────────────────────
# R3/R4 (PR + p_inter): probabilidad de PR tras kick.
P_PR: float                  = 0.50
# R4 (p_inter_pr): selector probabilistico.
P_INTER: float               = 0.20
ALPHA_INTER: float           = 0.80
# R5 (budget): presupuestos y lambda x10.
ITERS_TS: int                = 10_000
MAX_SIN_MEJ_TS: int          = 2_500
FACTOR_ITER_POP: int         = 200
MAX_SIN_MEJ_POP: int         = 500
KICK_ITERS: int              = 300
MAX_RESETS_BUDGET: int | None = None
LAMBDA_FACTOR_BUDGET: float  = 10.0
# R5 (budget) - SA reheat extendido.
SA_KICK_ITERS_BUDGET: int    = 20
SA_PATIENCE_BUDGET: int      = 100
SA_MAX_REHEATS_BUDGET: int   = 30
SA_TEMP_MIN_BUDGET: float    = 1e-4

# R1/R2/R3/R4 (sin budget): parametros estandar de kick.
KICK_ITERS_STD: int          = 30
MAX_RESETS_STD: int          = 10
SA_KICK_ITERS_STD: int       = 5
SA_PATIENCE_STD: int         = 10
SA_MAX_REHEATS_STD: int      = 5
SA_TEMP_MIN_STD: float       = 1e-3

# ── PARAMETROS CLASICOS DE LITERATURA (fijos en las 5 variantes) ─────────────
# Estos hiperparametros BASE de cada metaheuristica se fijan a valores
# canonicos de la literatura para que la unica variable entre approaches sea
# la estrategia de seleccion de vecinos. NO dependen de la variante.
#
# SA  — Kirkpatrick, Gelatt & Vecchi (1983): enfriamiento geometrico.
SA_ALPHA: float              = 0.95   # razon de enfriamiento geometrico T_{k+1}=alpha*T_k
#       T_init, T_min y L=n^2 se calibran por instancia (Lourenco et al.), default del modulo.
# TS  — Glover (1986, 1989): lista tabu de longitud fija.
TS_TABU_TENURE: int          = 7      # tenencia tabu clasica recomendada por Glover
TS_TAM_VECINDARIO: int       = 25     # tamano de muestreo del vecindario por iteracion
# RTS — Battiti & Tecchiolli (1994): tenure reactivo (incrementa/reduce ante ciclos).
RTS_FACTOR_AUMENTO: float    = 1.1    # multiplicador del tenure al detectar repeticion
RTS_FACTOR_REDUCCION: float  = 0.9    # divisor del tenure tras periodo sin repeticion
RTS_UMBRAL_ESCAPE: int       = 3      # repeticiones antes de la perturbacion de escape
# ABC — Karaboga (2005): tamano de colonia (numero de fuentes de alimento).
ABC_NUM_FUENTES: int         = 20     # SN clasico (empleadas = observadoras = SN)
# Cuckoo — Yang & Deb (2009): poblacion, fraccion de abandono, exponente de Levy.
CK_NUM_NIDOS: int            = 15     # numero de nidos (poblacion) clasico
CK_PA: float                 = 0.25   # fraccion de peores nidos abandonados
CK_BETA: float               = 1.5    # exponente de la distribucion de Levy

# 23 instancias del corpus actual.
INSTANCIAS = [
    "gdb19", "kshs1", "kshs2", "kshs3", "kshs4", "kshs5", "kshs6",
    "gdb4",  "gdb14", "gdb15", "gdb1",  "gdb20", "gdb3",  "gdb6",
    "gdb7",  "gdb12", "gdb10", "gdb2",  "gdb5",  "gdb13", "gdb16",
    "gdb17", "gdb21",
]

# Modulos MH soportados.
MH_MODULOS: dict[str, str] = {
    "sa":             "metacarp.recocido_simulado",
    "tabu_simple":    "metacarp.busqueda_tabu_simple",
    "tabu_reactiva":  "metacarp.busqueda_tabu_reactiva",
    "abc_simple":     "metacarp.abejas_simple",
    "cuckoo":         "metacarp.cuckoo_search",
}

# Variantes R y sus prefijos de CSV.
VARIANTES_R: dict[str, dict[str, str]] = {
    "R1": {"nombre": "strict_greedy",     "tag": "strict_intra_inter_greedy"},
    "R2": {"nombre": "aos_pm_greedy",     "tag": "aos_pm_greedy"},
    "R3": {"nombre": "pr_greedy",         "tag": "path_relinking_greedy"},
    "R4": {"nombre": "p_inter_pr_greedy", "tag": "p_inter_pr_greedy"},
    "R5": {"nombre": "budget_greedy",     "tag": "budget_greedy"},
}


# ============================================================
# Dataclass de tarea
# ============================================================

@dataclass(frozen=True)
class TareaExp:
    """Una corrida del grid R-experimento (sin semilla fija)."""
    variante: str            # "R1".."R5"
    mh:       str            # "sa", "tabu_simple", ...
    instancia: str
    repeticion: int
    root: str | None
    ruta_csv_parcial: str


# ============================================================
# Aplicacion de patches por variante
# ============================================================

def aplicar_patches_variante(variante: str, modulo_mh: str) -> None:
    """Aplica los patches del baseline por variante.

    El evaluador greedy es nativo de ``metacarp.evaluador_costo``; no requiere
    patch adicional. Esta funcion instala solo los patches especificos de cada
    variante (selector, PR, lambda), que son ortogonales a la evaluacion.

    R1 strict_greedy   : selector binario estricto (OPERADORES_STRICT_5 + kicks).
                          La MH usa el selector strict sin monkey-patch de
                          p_inter/alpha_inter (semantica equivalente a
                          p_inter=0.0, alpha_inter=1.0).
    R2 aos_pm_greedy   : patch AOS probability matching.
    R3 pr_greedy       : patch Path Relinking (PR ya instala strict selector
                          internamente).
    R4 p_inter_pr      : PR + selector p_inter probabilistico.
    R5 budget_greedy   : PR + p_inter + lambda x10. El budget extra se controla
                          via kwargs (no via patch).
    """
    if variante == "R1":
        # Strict: kick reactivo + operadores reducidos.
        # No requiere patch de selector porque strict_intra_inter es la
        # MISMA semantica que tener p_inter=0.0 y alpha_inter=1.0 (siempre
        # inter si viola, siempre intra si no). Para garantizar el binario
        # estricto, sobreescribimos el selector con el de strict.
        from metacarp.strict_intra_inter_20260524 import seleccionar_grupo_strict
        import importlib
        mh = importlib.import_module(modulo_mh)
        mh.seleccionar_grupo_operadores_inter_intra = seleccionar_grupo_strict
    elif variante == "R2":
        from metacarp.aos_pm_20260527 import aplicar_patch_aos
        aplicar_patch_aos(modulo_mh)
    elif variante == "R3":
        from metacarp.path_relinking_20260528 import aplicar_patch_pr
        aplicar_patch_pr(modulo_mh, p_pr=P_PR)
    elif variante == "R4":
        from metacarp.path_relinking_20260528 import aplicar_patch_pr
        from metacarp.p_inter_pr_20260528 import aplicar_patch_p_inter
        aplicar_patch_pr(modulo_mh, p_pr=P_PR)
        aplicar_patch_p_inter(modulo_mh)
    elif variante == "R5":
        # Budget: PR + p_inter + lambda x10 (los presupuestos se aplican via
        # kwargs en el launcher, no via patch).
        import metacarp.evaluador_costo as _ec
        _orig_lam = _ec.lambda_penal_capacidad_por_defecto
        _factor = LAMBDA_FACTOR_BUDGET
        def _lam_amplif(ctx):
            return _orig_lam(ctx) * _factor
        _ec.lambda_penal_capacidad_por_defecto = _lam_amplif

        from metacarp.path_relinking_20260528 import aplicar_patch_pr
        from metacarp.p_inter_pr_20260528 import aplicar_patch_p_inter
        aplicar_patch_pr(modulo_mh, p_pr=P_PR)
        aplicar_patch_p_inter(modulo_mh)
    else:
        raise ValueError(f"Variante desconocida: {variante!r}")


# ============================================================
# Wrappers de invocacion por MH
# ============================================================

def _kwargs_kick(variante: str, mh: str) -> dict:
    """Devuelve los kwargs del kick reactivo segun (variante, MH)."""
    if variante == "R5":
        if mh == "sa":
            return dict(
                max_iter_sin_mejora_kick=SA_KICK_ITERS_BUDGET,
                max_resets=MAX_RESETS_BUDGET,
            )
        else:
            return dict(
                max_iter_sin_mejora_kick=KICK_ITERS,
                max_resets=MAX_RESETS_BUDGET,
            )
    # R1..R4: kick estandar.
    if mh == "sa":
        return dict(
            max_iter_sin_mejora_kick=SA_KICK_ITERS_STD,
            max_resets=MAX_RESETS_STD,
        )
    return dict(
        max_iter_sin_mejora_kick=KICK_ITERS_STD,
        max_resets=MAX_RESETS_STD,
    )


def _kwargs_budget(variante: str, mh: str) -> dict:
    """Devuelve kwargs de budget/parada SOLO para R5 (resto usa defaults)."""
    if variante != "R5":
        return {}
    if mh == "sa":
        return dict(
            temperatura_minima=SA_TEMP_MIN_BUDGET,
            patience=SA_PATIENCE_BUDGET,
            reheat_factor=0.5,
            max_reheats_sin_mejora=SA_MAX_REHEATS_BUDGET,
        )
    if mh in ("tabu_simple", "tabu_reactiva"):
        return dict(
            iteraciones_max=ITERS_TS,
            max_iter_sin_mejora=MAX_SIN_MEJ_TS,
        )
    # abc_simple, cuckoo
    return dict(
        factor_iter=FACTOR_ITER_POP,
        max_iter_sin_mejora=MAX_SIN_MEJ_POP,
    )


def _kwargs_sa_std(variante: str) -> dict:
    """Defaults SA para R1..R4 (mismos que los baselines originales)."""
    if variante == "R5":
        return {}
    return dict(
        temperatura_minima=SA_TEMP_MIN_STD,
        patience=SA_PATIENCE_STD,
        reheat_factor=0.5,
        max_reheats_sin_mejora=SA_MAX_REHEATS_STD,
    )


def _kwargs_clasicos(mh: str) -> dict:
    """Hiperparametros base de LITERATURA CLASICA, fijos en las 5 variantes.

    Son ortogonales al presupuesto (kick/budget): fijan el comportamiento
    nucleo de cada metaheuristica (enfriamiento, tenencia tabu, tamano de
    poblacion, etc.) para que la unica variable entre approaches sea la
    estrategia de seleccion de vecinos.
    """
    if mh == "sa":
        # Kirkpatrick et al. (1983): solo la razon de enfriamiento geometrico;
        # T_init/T_min/L=n^2 se calibran por instancia (Lourenco), default modulo.
        return dict(alpha=SA_ALPHA)
    if mh == "tabu_simple":
        # Glover (1986): tenencia tabu fija + muestreo del vecindario.
        return dict(tabu_tenure=TS_TABU_TENURE, tam_vecindario=TS_TAM_VECINDARIO)
    if mh == "tabu_reactiva":
        # Battiti & Tecchiolli (1994): factores reactivos del tenure + escape.
        return dict(
            factor_aumento=RTS_FACTOR_AUMENTO,
            factor_reduccion=RTS_FACTOR_REDUCCION,
            umbral_repeticiones_escape=RTS_UMBRAL_ESCAPE,
            tam_vecindario=TS_TAM_VECINDARIO,
        )
    if mh == "abc_simple":
        # Karaboga (2005): tamano de colonia SN.
        return dict(num_fuentes=ABC_NUM_FUENTES)
    if mh == "cuckoo":
        # Yang & Deb (2009): nidos, fraccion de abandono, exponente de Levy.
        return dict(num_nidos=CK_NUM_NIDOS, pa_abandono=CK_PA, beta_levy=CK_BETA)
    return {}


def _extra_csv(variante: str) -> dict[str, str]:
    """Metadatos comunes a cada R-experimento."""
    meta = VARIANTES_R[variante]
    base = {
        "experimento":   f"correct_cost_{meta['nombre']}",
        "variante":      variante,
        "baseline":      meta["nombre"],
        "evaluador":     "greedy_orientation",
        "reporte":       "greedy_consistente",
        "parametros":    "literatura_clasica",
    }
    if variante in ("R3", "R4", "R5"):
        base["p_pr"] = str(P_PR)
    if variante in ("R4", "R5"):
        base["selector"]     = "p_inter_probabilistico"
        base["p_inter"]      = str(P_INTER)
        base["alpha_inter"]  = str(ALPHA_INTER)
    if variante == "R1":
        base["selector"] = "binario_estricto"
    if variante == "R2":
        base["selector"] = "aos_probability_matching"
    if variante == "R5":
        base["lambda_factor"] = str(LAMBDA_FACTOR_BUDGET)
    return base


def _operadores(variante: str):
    """Conjunto de operadores activos por variante."""
    if variante == "R2":
        from metacarp.aos_pm_20260527 import OPERADORES_AOS_5
        return OPERADORES_AOS_5
    from metacarp.strict_intra_inter_20260524 import OPERADORES_STRICT_5
    return OPERADORES_STRICT_5


def ejecutar_una(tarea: TareaExp) -> tuple[TareaExp, str, dict | None, str | None]:
    """Worker generico que despacha por (variante, MH)."""
    try:
        modulo = MH_MODULOS[tarea.mh]
        aplicar_patches_variante(tarea.variante, modulo)
        operadores = _operadores(tarea.variante)

        kwargs_comunes = dict(
            root=tarea.root,
            operadores=operadores,
            usar_gpu=False,
            semilla=None,
            repeticion=tarea.repeticion,
            guardar_csv=True,
            ruta_csv=tarea.ruta_csv_parcial,
            guardar_historial=False,
            extra_csv=_extra_csv(tarea.variante),
        )
        # Hiperparametros base de literatura clasica (fijos en las 5 variantes).
        # Se aplican primero; kick/budget pueden anadir presupuesto encima sin
        # solaparse (son parametros ortogonales).
        kwargs_comunes.update(_kwargs_clasicos(tarea.mh))
        kwargs_comunes.update(_kwargs_kick(tarea.variante, tarea.mh))
        kwargs_comunes.update(_kwargs_budget(tarea.variante, tarea.mh))

        if tarea.mh == "sa":
            kwargs_comunes.update(_kwargs_sa_std(tarea.variante))
            from metacarp import recocido_simulado_desde_instancia as runner
        elif tarea.mh == "tabu_simple":
            from metacarp import busqueda_tabu_simple_desde_instancia as runner
        elif tarea.mh == "tabu_reactiva":
            from metacarp import busqueda_tabu_reactiva_desde_instancia as runner
        elif tarea.mh == "abc_simple":
            from metacarp import busqueda_abejas_simple_desde_instancia as runner
        elif tarea.mh == "cuckoo":
            from metacarp import cuckoo_search_desde_instancia as runner
        else:
            raise ValueError(f"MH desconocida: {tarea.mh!r}")

        res = runner(tarea.instancia, **kwargs_comunes)
        info = {
            "costo":    res.mejor_costo,
            "tiempo":   res.tiempo_segundos,
            "n_resets": getattr(res, "n_resets_kick", 0),
            "factible": getattr(res, "mejor_solucion_factible_final", None),
        }
        return (tarea, "ok", info, None)
    except Exception as exc:  # noqa: BLE001
        return (tarea, "fail", None, f"{type(exc).__name__}: {exc}")


# ============================================================
# Consolidacion de parciales (igual al patron previo)
# ============================================================

def consolidar_parciales(
    dir_parciales: Path,
    salida_dir: Path,
    prefijo_csv: str,
    experimento: str,
    ydmh: str,
) -> int:
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
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-corrida de baselines bajo evaluador greedy."
    )
    parser.add_argument("--variant", type=str, required=True,
                        choices=list(VARIANTES_R.keys()),
                        help="R1/R2/R3/R4/R5")
    parser.add_argument("--mh", type=str, required=True,
                        choices=list(MH_MODULOS.keys()),
                        help="MH a ejecutar")
    parser.add_argument("--salida-dir", type=str,
                        default="experimentos/corridas/correct_cost")
    parser.add_argument("--repeticiones", type=int, default=5)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--instancias", type=str, default=None, nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = VARIANTES_R[args.variant]
    prefijo_csv = args.mh  # "sa", "tabu_simple", etc.
    subcarpeta  = f"correct_cost_{meta['nombre']}_{args.mh}"
    experimento = f"{args.mh}_{meta['tag']}"
    salida_dir = Path(args.salida_dir).expanduser().resolve() / subcarpeta
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
                variante=args.variant,
                mh=args.mh,
                instancia=instancia,
                repeticion=rep,
                root=args.root,
                ruta_csv_parcial=str(parcial),
            ))

    total = len(tareas)
    print("=" * 80)
    print(f"{args.variant} ({meta['nombre']})  —  MH={args.mh}  —  re_greedy_20260529")
    print("=" * 80)
    print(f"Instancias        : {len(instancias_efectivas)}")
    print(f"Repeticiones      : {args.repeticiones}")
    print(f"Workers           : {args.workers}")
    print(f"Corridas          : {total}")
    print(f"Evaluador         : greedy (orientacion dinamica por tarea)")
    print(f"Salida CSV        : {salida_dir}")
    print("-" * 80)

    total_ok = total_fail = 0
    if args.workers <= 1:
        # Aplicar patches una sola vez en modo secuencial.
        aplicar_patches_variante(args.variant, MH_MODULOS[args.mh])
        for tarea in tareas:
            _, estado, info, err = ejecutar_una(tarea)
            if estado == "ok":
                print(f"  [{tarea.instancia}] rep={tarea.repeticion} "
                      f"| costo={info['costo']:.4f} | t={info['tiempo']:.2f}s "
                      f"| kicks={info.get('n_resets', 0)} "
                      f"| factible={info.get('factible', '?')}")
                total_ok += 1
            else:
                print(f"  [{tarea.instancia}] rep={tarea.repeticion} | FAIL: {err}")
                total_fail += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futuros = {executor.submit(ejecutar_una, t): t for t in tareas}
            for fut in as_completed(futuros):
                tarea, estado, info, err = fut.result()
                if estado == "ok":
                    print(f"  [{tarea.instancia}] rep={tarea.repeticion} "
                          f"| costo={info['costo']:.4f} | t={info['tiempo']:.2f}s "
                          f"| kicks={info.get('n_resets', 0)} "
                          f"| factible={info.get('factible', '?')}")
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


if __name__ == "__main__":
    main()
