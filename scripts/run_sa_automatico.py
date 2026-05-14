"""
Corrida SA para instancias seleccionadas.

Configuración:
    temperatura_inicial = None  →  automática: 20·d_max/n por instancia
    temperatura_minima  = None  →  automática: 20·d_max/n² por instancia
    patience            = 5     →  niveles sin mejora antes de reheat (agresivo)
    reheat_factor       = 0.8   →  fracción de T_init a la que se recalienta (agresivo)
    alpha               = [0.90]   →  fijo
    p_inter             = [0.65]   →  fijo

Total: 1 alpha × 1 p_inter × 23 instancias × 2 repeticiones = 46 corridas.

Uso:
    python scripts/run_sa_automatico.py
    python scripts/run_sa_automatico.py --salida-dir resultados_sa
    python scripts/run_sa_automatico.py --repeticiones 3
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from metacarp import recocido_simulado_desde_instancia

INSTANCIAS = [
    "gdb19", "kshs1", "kshs2", "kshs3", "kshs4", "kshs5", "kshs6",
    "gdb4",  "gdb14", "gdb15", "gdb1",  "gdb20", "gdb3",  "gdb6",
    "gdb7",  "gdb12", "gdb10", "gdb2",  "gdb5",  "gdb13", "gdb16",
    "gdb17", "gdb21",
]

ALPHAS    = [0.90]
P_INTERS  = [0.65]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SA grid search sobre alpha y p_inter.")
    parser.add_argument("--salida-dir",   type=str, default="experimentos")
    parser.add_argument("--repeticiones", type=int, default=2)
    parser.add_argument("--experimento",  type=str, default="sa_auto")
    parser.add_argument("--root",         type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    salida_dir = Path(args.salida_dir).expanduser().resolve() / "sa_small_reheatv3"
    salida_dir.mkdir(parents=True, exist_ok=True)
    ydmh = datetime.now().strftime("%Y%d%m%H%M")

    total = len(INSTANCIAS) * len(ALPHAS) * len(P_INTERS) * args.repeticiones
    print("=" * 80)
    print("SA  —  grid search alpha × p_inter")
    print("=" * 80)
    print(f"Instancias   : {len(INSTANCIAS)}")
    print(f"T_ini        : automática (20·d_max/n por instancia)")
    print(f"T_min        : automática (20·d_max/n² por instancia)")
    print(f"Patience     : 5 niveles (reheat agresivo)")
    print(f"Reheat factor: 0.8 (reheat agresivo)")
    print(f"Alpha values : {ALPHAS}")
    print(f"p_inter vals : {P_INTERS}")
    print(f"Semilla      : aleatoria (None)")
    print(f"Repeticiones : {args.repeticiones}")
    print(f"Corridas     : {total}")
    print(f"Salida CSV   : {salida_dir}")
    print("-" * 80)

    total_ok = 0
    total_fail = 0

    for instancia in INSTANCIAS:
        ruta_csv = salida_dir / f"sa_{instancia}_{args.experimento}_{ydmh}.csv"
        print(f"\n[{instancia}]  csv → {ruta_csv.name}")

        for alpha in ALPHAS:
            for p_inter in P_INTERS:
                for rep in range(1, args.repeticiones + 1):
                    try:
                        res = recocido_simulado_desde_instancia(
                            instancia,
                            temperatura_inicial=None,
                            temperatura_minima=None,
                            alpha=alpha,
                            p_inter=p_inter,
                            patience=5,
                            reheat_factor=0.8,
                            semilla=None,
                            repeticion=rep,
                            guardar_csv=True,
                            ruta_csv=str(ruta_csv),
                            guardar_historial=False,
                            root=args.root,
                        )
                        print(
                            f"  alpha={alpha:.2f} p_inter={p_inter:.1f} rep={rep} "
                            f"| costo={res.mejor_costo:.4f} "
                            f"| mejora={res.mejora_porcentaje_inicial_vs_final:.2f}% "
                            f"| t={res.tiempo_segundos:.2f}s"
                        )
                        total_ok += 1
                    except Exception as exc:  # noqa: BLE001
                        print(f"  alpha={alpha:.2f} p_inter={p_inter:.1f} rep={rep} | FAIL: {type(exc).__name__}: {exc}")
                        total_fail += 1

    print("\n" + "-" * 80)
    print(f"OK   : {total_ok}")
    print(f"FAIL : {total_fail}")
    print(f"CSV  : {salida_dir}")
    print("-" * 80)


if __name__ == "__main__":
    main()
