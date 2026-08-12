"""
Re-intento dirigido de Cuckoo Search para las instancias con gap>1%
(resultados/instancias_gap_mayor_1pct_20260805.csv), por niveles de
severidad. Reutiliza ÍNTEGRA la maquinaria de ``run_cs_minigrid_20260710.py``
(igual que ``run_cs_val_egl_20260711.py``), parcheando sus constantes de
malla y presupuesto ANTES de invocar ``main()`` — mismo patrón ya validado
en la campaña de julio.

Tratamiento por tier:
  A (gap 1-3%,  5 inst.): config ganadora fija (factor_pasos=0.25,
     p_inter=0.6), +10 reps, presupuesto x1.
  B (gap 3-10%, 44 inst.): misma config fija, +10 reps, presupuesto x2.5.
  C (gap 10-25%, 26 inst.): mini-grid factor_pasos in {0.15, 0.25, 0.35}
     (p_inter=0.6 fijo) x 3 reps c/u (~9 corridas/instancia), presupuesto x7.
     No hay evidencia de que un solo valor sea mejor que 0.25 en instancias
     grandes, así que se re-explora en vez de adivinar.

No hay Tier D en CS (ninguna instancia con gap>25%).

Salida: experimentos_reintento_20260805/cs_minigrid/corrida_<tier>_<ts>/

Uso:
    python scripts/run_reintento_cs_20260805.py [--workers N] [--smoke]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))

import run_cs_minigrid_20260710 as minigrid  # noqa: E402

LISTA_PROBLEMA = RAIZ / "resultados" / "instancias_gap_mayor_1pct_20260805.csv"
SALIDA_DIR = str(RAIZ / "experimentos_reintento_20260805")

TIER_A, TIER_B, TIER_C = "A(1-3%)", "B(3-10%)", "C(10-25%)"


def _instancias_cs_por_tier() -> dict[str, list[str]]:
    grupos: dict[str, list[str]] = {TIER_A: [], TIER_B: [], TIER_C: []}
    with open(LISTA_PROBLEMA, newline="") as fh:
        for fila in csv.DictReader(fh):
            if fila["metaheuristica"] != "CS":
                continue
            if fila["tier"] in grupos:
                grupos[fila["tier"]].append(fila["instancia"])
    return grupos


def _correr_tier(tier: str, instancias: list[str], workers: int, smoke: bool) -> None:
    if not instancias:
        print(f"[{tier}] sin instancias, se omite.")
        return
    minigrid.INSTANCIAS = instancias
    if tier == TIER_A:
        minigrid.FACTOR_PASOS_GRID = (0.25,)
        minigrid.P_INTER_GRID = (0.6,)
        reps, factor = 10, 1.0
    elif tier == TIER_B:
        minigrid.FACTOR_PASOS_GRID = (0.25,)
        minigrid.P_INTER_GRID = (0.6,)
        reps, factor = 10, 2.5
    else:  # TIER_C
        minigrid.FACTOR_PASOS_GRID = (0.15, 0.25, 0.35)
        minigrid.P_INTER_GRID = (0.6,)
        reps, factor = 3, 7.0

    minigrid.MAX_EVALUACIONES_DEF = int(1_000_000 * factor)
    minigrid.TIEMPO_LIMITE_DEF = 300.0 * factor

    print(f"\n[{tier}] {len(instancias)} instancias, grid factor_pasos="
          f"{minigrid.FACTOR_PASOS_GRID}, reps={reps}, presupuesto x{factor}")

    argv = ["--salida-dir", SALIDA_DIR, "--repeticiones", str(reps),
            "--workers", str(workers)]
    if smoke:
        argv.append("--smoke")
    minigrid.main(argv)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    import os
    workers = args.workers or os.cpu_count() or 1

    grupos = _instancias_cs_por_tier()
    if args.smoke:
        grupos = {t: v[:2] for t, v in grupos.items()}
    for tier in (TIER_A, TIER_B, TIER_C):
        _correr_tier(tier, grupos[tier], workers, args.smoke)


if __name__ == "__main__":
    main()
