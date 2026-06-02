"""
Experimento 0: algoritmos en forma CANÓNICA PURA (sin mecanismos de escape).

Sirve como LÍNEA DE BASE INFERIOR del programa experimental: mide la calidad
de cada metaheurística en su forma más básica (los artículos originales), sin
los mecanismos de diversificación/escape que normalmente se añaden para evitar
el estancamiento. Compara directamente con el Experimento 8 (solo_p_inter) para
cuantificar cuánto aporta cada mecanismo de escape.

CONFIGURACIÓN CANÓNICA PURA POR METAHEURÍSTICA:
  SA     (Kirkpatrick, Gelatt & Vecchi, 1983):
         Enfriamiento geométrico puro, SIN reheat.
         patience=0 deshabilita el reheat; la búsqueda enfría hasta T_min y para.
  TS     (Glover, 1986):
         Lista tabú de longitud fija, SIN parada anticipada por estancamiento.
         max_iter_sin_mejora=10_000 (efectivamente desactivado); corre las
         iteraciones_max completas.
  RTS    (Battiti & Tecchiolli, 1994):
         SIN reactividad del tenure: factor_aumento=1.0, factor_reduccion=1.0
         hacen que el tenure no cambie nunca → equivalente a TS con tenure fijo.
  ABC    (Karaboga, 2005):
         SIN fase scout (o scout prácticamente desactivado):
         limite_abandono=10_000 → las fuentes nunca se abandonan; la búsqueda
         es puramente de explotación (fases empleada + observadora).
  Cuckoo (Yang & Deb, 2009):
         SIN abandono de nidos: pa_abandono=0.0 → no se reemplazan los peores
         nidos por nuevas soluciones aleatorias de Lévy; puramente vuelos Lévy.

TODO LO DEMÁS usa los mejores valores calibrados (p_inter, knob 1, knob 2) para
que la única diferencia respecto al Exp. 8 sea el mecanismo de escape.
Operadores: OPERADORES_POPULARES (9). Lambda: default (None). 5 reps × 23 inst.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Limitar hilos BLAS/OpenMP a 1 ANTES de importar numpy.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys as _sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from metacarp.vecindarios import OPERADORES_POPULARES

REPS_DEF = 5

INSTANCIAS = [
    "gdb19", "kshs1", "kshs2", "kshs3", "kshs4", "kshs5", "kshs6",
    "gdb4",  "gdb14", "gdb15", "gdb1",  "gdb20", "gdb3",  "gdb6",
    "gdb7",  "gdb12", "gdb10", "gdb2",  "gdb5",  "gdb13", "gdb16",
    "gdb17", "gdb21",
]

MH_MODULOS = {
    "sa": "metacarp.recocido_simulado", "tabu_simple": "metacarp.busqueda_tabu_simple",
    "tabu_reactiva": "metacarp.busqueda_tabu_reactiva", "abc_simple": "metacarp.abejas_simple",
    "cuckoo": "metacarp.cuckoo_search",
}

# Config canónica calibrada (igual que Exp. 8, para comparabilidad exacta).
P_INTER_FIJO = {"sa": 0.5, "tabu_simple": 0.4, "tabu_reactiva": 0.5,
                "abc_simple": 0.5, "cuckoo": 0.1}
KNOB1_FIJO   = {"sa": ("alpha", 0.90), "tabu_simple": ("tabu_tenure", 25),
                "tabu_reactiva": ("factor_aumento", 1.2),
                "abc_simple": ("num_fuentes", 30), "cuckoo": ("pa_abandono", 0.15)}
KNOB2_FIJO   = {"sa": ("max_reheats_sin_mejora", 10), "tabu_simple": ("tam_vecindario", 40),
                "tabu_reactiva": ("factor_reduccion", 0.95),
                "abc_simple": ("limite_abandono", 60), "cuckoo": ("beta_levy", 1.3)}

# Parámetros que DESACTIVAN el mecanismo de escape en cada MH.
# Junto a la config canónica calibrada, forman la "versión canónica pura".
ESCAPE_OFF: dict[str, dict] = {
    "sa": {
        # patience=0 → reheat completamente desactivado (Kirkpatrick puro).
        # Los knobs 1 y 2 de SA incluyen max_reheats_sin_mejora=10, que aquí
        # se anula porque patience=0 hace que el reheat no llegue a activarse.
        "patience": 0,
    },
    "tabu_simple": {
        # max_iter_sin_mejora=10_000 → nunca para antes de iteraciones_max.
        # Glover (1986) puro: corre todas las iteraciones sin excepción.
        "max_iter_sin_mejora": 10_000,
    },
    "tabu_reactiva": {
        # factor_aumento=1.0, factor_reduccion=1.0 → tenure nunca cambia.
        # Convierte RTS en TS con tenure fijo (sin la capa reactiva).
        "factor_aumento": 1.0,
        "factor_reduccion": 1.0,
    },
    "abc_simple": {
        # limite_abandono=10_000 → las fuentes nunca se convierten en scouts.
        # ABC sin fase de exploración global (solo empleadas + observadoras).
        "limite_abandono": 10_000,
    },
    "cuckoo": {
        # pa_abandono=0.0 → no se reemplazan nidos con vuelos de Lévy largos.
        # Yang & Deb puro sin el mecanismo de abandono (solo vuelos normales).
        "pa_abandono": 0.0,
    },
}


@dataclass(frozen=True)
class Tarea:
    mh: str
    instancia: str
    repeticion: int
    root: str | None
    ruta_csv: str


def construir_kwargs(mh: str) -> dict:
    """Config canónica calibrada + mecanismo de escape DESACTIVADO."""
    k1, v1 = KNOB1_FIJO[mh]
    k2, v2 = KNOB2_FIJO[mh]
    base = dict(
        operadores=OPERADORES_POPULARES,
        usar_gpu=False, semilla=None,
        guardar_csv=True, guardar_historial=False,
        lambda_capacidad=None,
        p_inter=P_INTER_FIJO[mh],
        max_iter_sin_mejora_kick=None,   # kick OFF
        max_resets=None,
    )
    base[k1] = v1
    base[k2] = v2
    # Parámetros de la config canónica que complementan los knobs de SA.
    if mh == "sa":
        base.update(alpha_inter=0.80, temperatura_inicial=None, temperatura_minima=None,
                    reheat_factor=0.5)
    elif mh in ("tabu_simple", "tabu_reactiva"):
        base["alpha_inter"] = 0.80
    # Aplicar la "desactivación" del mecanismo de escape (sobreescribe los
    # valores del knob 2 que sean necesarios, p.ej. max_reheats_sin_mejora
    # pierde efecto porque patience=0 ya impide que el reheat se active).
    base.update(ESCAPE_OFF[mh])
    return base


def _cargar_runner(mh: str):
    if mh == "sa":
        from metacarp import recocido_simulado_desde_instancia as r
    elif mh == "tabu_simple":
        from metacarp import busqueda_tabu_simple_desde_instancia as r
    elif mh == "tabu_reactiva":
        from metacarp import busqueda_tabu_reactiva_desde_instancia as r
    elif mh == "abc_simple":
        from metacarp import busqueda_abejas_simple_desde_instancia as r
    elif mh == "cuckoo":
        from metacarp import cuckoo_search_desde_instancia as r
    else:
        raise ValueError(mh)
    return r


def _leer_gap(ruta: str) -> float:
    try:
        with open(ruta, "r", encoding="utf-8", newline="") as f:
            fila = next(csv.DictReader(f), None)
        return float(fila.get("gap_bks_porcentaje", "")) if fila else math.nan
    except (OSError, StopIteration, TypeError, ValueError):
        return math.nan


def ejecutar_una(t: Tarea):
    try:
        runner = _cargar_runner(t.mh)
        kw = construir_kwargs(t.mh)
        kw.update(root=t.root, ruta_csv=t.ruta_csv, repeticion=t.repeticion)
        res = runner(t.instancia, **kw)
        return (t, "ok", {"gap": _leer_gap(t.ruta_csv), "costo": res.mejor_costo,
                          "tiempo": res.tiempo_segundos}, None)
    except Exception as exc:  # noqa: BLE001
        return (t, "fail", None, f"{type(exc).__name__}: {exc}")


def _instancia_de_parcial(stem: str, mh: str) -> str | None:
    prefix = f"{mh}_canonico_"
    if not stem.startswith(prefix):
        return None
    partes = stem[len(prefix):].split("_")
    if len(partes) < 3:
        return None
    return "_".join(partes[:-2])


def _consolidar(parciales: list[Path], ruta_final: Path) -> int:
    filas: list[dict] = []; columnas: list[str] = []; vistas: set[str] = set()
    for parcial in sorted(parciales):
        with parcial.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for col in reader.fieldnames or []:
                if col not in vistas:
                    vistas.add(col); columnas.append(col)
            filas.extend(reader)
    if not filas:
        return 0
    ruta_final.parent.mkdir(parents=True, exist_ok=True)
    with ruta_final.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columnas)
        writer.writeheader(); writer.writerows(filas)
    return len(filas)


def ejecutar_mh(mh: str, carpeta: Path, *, instancias, reps, workers, root):
    dir_partials = carpeta / "_partials"
    dir_partials.mkdir(parents=True, exist_ok=True)
    tareas = []
    for inst in instancias:
        for rep in range(1, reps + 1):
            idx = len(tareas)
            parcial = dir_partials / f"{mh}_canonico_{inst}_{os.getpid()}_{idx}.csv"
            tareas.append(Tarea(mh, inst, rep, root, str(parcial)))

    escape_desc = {k: v for k, v in ESCAPE_OFF[mh].items()}
    print(f"\n=== {mh} (escape OFF: {escape_desc}) | "
          f"{len(instancias)} inst × {reps} reps = {len(tareas)} corridas ===")
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(ejecutar_una, t) for t in tareas]):
            t, estado, info, err = fut.result()
            if estado == "ok":
                ok += 1
                gap = info["gap"]
                print(f"  [{t.instancia}] rep={t.repeticion} gap="
                      f"{'nan' if math.isnan(gap) else f'{gap:.2f}%'} "
                      f"costo={info['costo']:.1f} t={info['tiempo']:.1f}s")
            else:
                fail += 1; print(f"  [{t.instancia}] FAIL: {err}")
    print(f"  OK={ok} FAIL={fail}")

    grupos: dict[str, list[Path]] = {}
    for p in sorted(dir_partials.glob(f"{mh}_canonico_*.csv")):
        inst = _instancia_de_parcial(p.stem, mh)
        if inst:
            grupos.setdefault(inst, []).append(p)
    n = 0
    for inst, archivos in grupos.items():
        ruta = carpeta / f"{mh}_canonico_{inst}.csv"
        n += 1 if _consolidar(archivos, ruta) else 0
    print(f"  -> {n} CSV finales en {carpeta}")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Exp. 0: algoritmos canónicos puros (sin mecanismo de escape).")
    p.add_argument("--mhs", nargs="*", default=list(MH_MODULOS.keys()))
    p.add_argument("--reps", type=int, default=REPS_DEF)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--instancias", nargs="*", default=None)
    p.add_argument("--salida-base", type=str, default="experimentos_costo_fixed")
    p.add_argument("--smoke", action="store_true",
                   help="2 instancias, 1 rep (prueba rápida).")
    args = p.parse_args(argv)

    instancias = args.instancias or INSTANCIAS
    reps = args.reps
    if args.smoke:
        instancias = ["gdb19", "kshs1"]
        reps = 1

    ts = datetime.now().strftime("%Y%m%d-%H%M")
    carpeta = Path(args.salida_base).expanduser().resolve() / f"canonico_puro_{ts}"
    carpeta.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Experimento 0: canónico puro (sin escape) — {len(instancias)} inst × {reps} reps")
    print(f"Salida: {carpeta}")
    print("=" * 72)

    for mh in args.mhs:
        ejecutar_mh(mh, carpeta, instancias=instancias, reps=reps,
                    workers=args.workers, root=args.root)


if __name__ == "__main__":
    main()
