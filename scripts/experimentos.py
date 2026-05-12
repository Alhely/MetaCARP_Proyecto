"""
Script de experimentación para metaheurísticas CARP.

Objetivo:
- Ejecutar SA / Tabu / Abejas / Cuckoo por instancia.
- Guardar CSV por metaheurística e instancia.
- Permitir corridas reproducibles con espacio de búsqueda tipo literatura.

Ejemplos:
    python metacarp/scripts/experimentos.py
    python metacarp/scripts/experimentos.py --instancias gdb19 --metaheuristicas sa tabu
    python metacarp/scripts/experimentos.py --seed 7 --salida-dir experimentos
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Callable, Sequence

from metacarp import (
    busqueda_abejas_desde_instancia,
    busqueda_tabu_desde_instancia,
    cuckoo_search_desde_instancia,
    nombres_soluciones_iniciales_disponibles,
    recocido_simulado_desde_instancia,
)

# Alias canónicos de metaheurísticas disponibles en este script.
METAHEURISTICAS_VALIDAS = ("sa", "tabu", "abejas", "cuckoo")


@dataclass(frozen=True, slots=True)
class MetaRunner:
    nombre: str
    run: Callable[..., object]
    parametros_default: dict[str, object]
    espacio_parametros: list[dict[str, object]]


def _grid(param_space: dict[str, list[object]]) -> list[dict[str, object]]:
    """
    Construye combinaciones cartesianas de un espacio de búsqueda.
    """
    keys = list(param_space.keys())
    combos: list[dict[str, object]] = []
    for vals in product(*(param_space[k] for k in keys)):
        combos.append({k: v for k, v in zip(keys, vals)})
    return combos


def _construir_runners() -> dict[str, MetaRunner]:
    """
    Define un espacio de búsqueda ampliado (modo literatura) para cada metaheurística.
    """
    sa_space = _grid(
        {
            "temperatura_inicial": [150.0, 300.0, 500.0, 800.0],
            "temperatura_minima": [1e-3],
            "alpha": [0.90, 0.93, 0.95, 0.97],
            "iteraciones_por_temperatura": [40, 80, 120],
            "max_enfriamientos": [60, 100],
        }
    )
    tabu_space = _grid(
        {
            "iteraciones": [500, 700, 1000],
            "tam_vecindario": [20, 30, 40, 60],
            "tenure_tabu": [10, 15, 20, 30],
        }
    )
    abejas_space = _grid(
        {
            "iteraciones": [500, 700, 1000],
            "num_fuentes": [10, 20, 30, 40],
            "limite_abandono": [15, 30, 45, 60],
        }
    )
    cuckoo_space = _grid(
        {
            "iteraciones": [500, 750, 1000],
            "num_nidos": [15, 25, 35],
            "pa_abandono": [0.15, 0.25, 0.35],
            "pasos_levy_base": [2, 3, 5],
            "beta_levy": [1.2, 1.5],
        }
    )

    return {
        "sa": MetaRunner(
            nombre="sa",
            run=recocido_simulado_desde_instancia,
            parametros_default=sa_space[0],
            espacio_parametros=sa_space,
        ),
        "tabu": MetaRunner(
            nombre="tabu",
            run=busqueda_tabu_desde_instancia,
            parametros_default=tabu_space[0],
            espacio_parametros=tabu_space,
        ),
        "abejas": MetaRunner(
            nombre="abejas",
            run=busqueda_abejas_desde_instancia,
            parametros_default=abejas_space[0],
            espacio_parametros=abejas_space,
        ),
        "cuckoo": MetaRunner(
            nombre="cuckoo",
            run=cuckoo_search_desde_instancia,
            parametros_default=cuckoo_space[0],
            espacio_parametros=cuckoo_space,
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta campañas de experimentación para metaheurísticas CARP."
    )
    parser.add_argument(
        "--instancias",
        nargs="*",
        default=["all"],
        help="Lista de instancias (por defecto: all = todas con solución inicial).",
    )
    parser.add_argument(
        "--metaheuristicas",
        nargs="*",
        default=["all"],
        help="Subconjunto a ejecutar: sa tabu abejas cuckoo (por defecto: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla base para reproducibilidad.",
    )
    parser.add_argument(
        "--salida-dir",
        type=str,
        default="experimentos",
        help="Carpeta raíz donde se guardan los CSV (default: experimentos).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root de datos opcional (si no, usa configuración por defecto del paquete).",
    )
    parser.add_argument(
        "--repeticiones",
        type=int,
        default=2,
        help="Número de repeticiones por configuración (por defecto: 2).",
    )
    parser.add_argument(
        "--experimento",
        type=str,
        default="tesis",
        help="Etiqueta del experimento para el nombre de archivo CSV.",
    )
    parser.add_argument(
        "--usar-gpu",
        action="store_true",
        help=(
            "Activa backend GPU (CuPy) para evaluación por lotes (Tabu/Abejas/Cuckoo). "
            "Si CuPy no está disponible se hace fallback automático a CPU rápido."
        ),
    )
    return parser.parse_args()


def _resolver_instancias(raw: Sequence[str], *, root: str | None) -> list[str]:
    if not raw or (len(raw) == 1 and raw[0].lower() == "all"):
        return sorted(nombres_soluciones_iniciales_disponibles(root=root))
    return list(raw)


def _resolver_metaheuristicas(raw: Sequence[str]) -> list[str]:
    if not raw or (len(raw) == 1 and raw[0].lower() == "all"):
        return list(METAHEURISTICAS_VALIDAS)

    seleccion = [x.lower() for x in raw]
    invalidas = [x for x in seleccion if x not in METAHEURISTICAS_VALIDAS]
    if invalidas:
        raise ValueError(
            f"Metaheurísticas inválidas: {invalidas}. "
            f"Válidas: {METAHEURISTICAS_VALIDAS}."
        )
    return seleccion


def _ruta_csv(
    salida_dir: Path,
    meta: str,
    instancia: str,
    *,
    experimento: str,
    ydmh: str,
) -> Path:
    """
    Un CSV por metaheurística e instancia, con nombre:
    <metaheuristica>_<instancia>_<experimento>_YDMhm.csv
    """
    nombre = f"{meta}_{instancia}_{experimento}_{ydmh}.csv"
    return salida_dir / meta / nombre


def _espacio_parametros(runner: MetaRunner) -> list[dict[str, object]]:
    """
    Devuelve el espacio de búsqueda de parámetros para un runner.
    Si no hay espacio definido, usa la configuración default como única opción.
    """
    if runner.espacio_parametros:
        return [dict(cfg) for cfg in runner.espacio_parametros]
    return [dict(runner.parametros_default)]


def main() -> None:
    args = _parse_args()
    salida_dir = Path(args.salida_dir).expanduser().resolve()
    salida_dir.mkdir(parents=True, exist_ok=True)

    runners = _construir_runners()
    instancias = _resolver_instancias(args.instancias, root=args.root)
    metas = _resolver_metaheuristicas(args.metaheuristicas)

    if not instancias:
        raise RuntimeError("No se encontraron instancias para ejecutar.")
    if args.repeticiones <= 0:
        raise ValueError("--repeticiones debe ser > 0.")
    if not str(args.experimento).strip():
        raise ValueError("--experimento no puede estar vacío.")

    # Timestamp fijo para toda la corrida (YDMhm: year-day-month-hour-minute).
    ydmh = datetime.now().strftime("%Y%d%m%H%M")

    total_planeadas = 0
    for meta in metas:
        total_planeadas += len(_espacio_parametros(runners[meta])) * args.repeticiones
    total_planeadas *= len(instancias)

    print("=" * 96)
    print("EXPERIMENTOS METAHEURÍSTICAS")
    print("=" * 96)
    print(f"Instancias       : {instancias}")
    print(f"Metaheurísticas  : {metas}")
    print(f"Seed base        : {args.seed}")
    print(f"Experimento      : {args.experimento}")
    print(f"Repeticiones     : {args.repeticiones}")
    print(f"Backend          : {'gpu (CuPy)' if args.usar_gpu else 'cpu (NumPy)'}")
    for meta in metas:
        print(f"Configs {meta:7s} : {len(_espacio_parametros(runners[meta]))}")
    print(f"Corridas planeadas: {total_planeadas}")
    print(f"Salida CSV       : {salida_dir}")
    print("-" * 96)

    total_ok = 0
    total_fail = 0

    for idx_inst, instancia in enumerate(instancias):
        print(f"\n[INSTANCIA] {instancia}")

        for idx_meta, meta in enumerate(metas):
            runner = runners[meta]
            ruta_csv = _ruta_csv(
                salida_dir,
                meta,
                instancia,
                experimento=str(args.experimento).strip(),
                ydmh=ydmh,
            )
            configs = _espacio_parametros(runner)

            print(f"  -> {meta:7s} | configs={len(configs)} | csv={ruta_csv}")

            for cfg_idx, cfg in enumerate(configs, start=1):
                for rep in range(1, args.repeticiones + 1):
                    # Semilla derivada para que cada corrida sea estable y distinta.
                    semilla = args.seed + (idx_inst * 100000) + (idx_meta * 10000) + (cfg_idx * 100) + rep

                    kwargs = dict(cfg)
                    config_id = f"{meta}-cfg{cfg_idx}"
                    id_corrida = f"{instancia}-{meta}-cfg{cfg_idx}-rep{rep}-seed{semilla}"
                    kwargs.update(
                        {
                            "root": args.root,
                            "semilla": semilla,
                            "id_corrida": id_corrida,
                            "config_id": config_id,
                            "repeticion": rep,
                            "guardar_csv": True,
                            "ruta_csv": str(ruta_csv),
                            "guardar_historial": False,
                            "usar_gpu": args.usar_gpu,
                        }
                    )

                    print(
                        f"     cfg={cfg_idx}/{len(configs)} "
                        f"| rep={rep}/{args.repeticiones} "
                        f"| semilla={semilla}"
                    )
                    try:
                        res = runner.run(instancia, **kwargs)
                        print(
                            "       OK "
                            f"| mejor_costo={res.mejor_costo:.6f} "
                            f"| mejora={res.mejora_porcentaje_inicial_vs_final:.4f}% "
                            f"| tiempo={res.tiempo_segundos:.4f}s"
                        )
                        total_ok += 1
                    except Exception as exc:  # noqa: BLE001 - queremos continuar campaña.
                        print(f"       FAIL | {type(exc).__name__}: {exc}")
                        total_fail += 1

    print("\n" + "-" * 96)
    print(f"Ejecuciones OK   : {total_ok}")
    print(f"Ejecuciones FAIL : {total_fail}")
    print(f"CSV raíz         : {salida_dir}")
    print("-" * 96)


if __name__ == "__main__":
    main()
