"""
Etiqueta cada solución inicial como factible/infactible y guarda los resultados.

Genera:
  - feasibility_labels.json  (resultado detallado por instancia)
  - feasibility_summary.csv  (tabla resumen para inspección rápida)

Uso:
  python etiquetar_soluciones_iniciales.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from MetaCARP_Proyecto.metacarp import (
    cargar_matriz_dijkstra,
    cargar_solucion_inicial,
    load_instance,
    nombres_soluciones_iniciales_disponibles,
    verificar_factibilidad,
)

# Raíz del proyecto: carpeta donde están PickleInstances/, Matrices/, InitialSolution/
_ROOT        = Path(__file__).parent
_SALIDA_JSON = _ROOT / "feasibility_labels.json"
_SALIDA_CSV  = _ROOT / "feasibility_summary.csv"

_CSV_CAMPOS = ["instancia", "factible", "c1_cobertura", "c2_conectividad",
               "c3_capacidad", "c4_vehiculos", "c5_deposito"]


def _etiquetar_instancia(nombre: str) -> dict:
    solucion = cargar_solucion_inicial(nombre, root=_ROOT)
    data     = load_instance(nombre, root=_ROOT)
    matriz   = cargar_matriz_dijkstra(nombre, root=_ROOT)
    result   = verificar_factibilidad(solucion, data, matriz)
    det      = result.details
    return {
        "factible":         result.ok,
        "c1_cobertura":     not bool(det.c1_tareas_requeridas),
        "c2_conectividad":  not bool(det.c2_consecutivas),
        "c3_capacidad":     not bool(det.c3_capacidad),
        "c4_vehiculos":     not bool(det.c4_vehiculos),
        "c5_deposito":      not bool(det.c5_deposito_extremos),
        "violaciones": {
            "c1": det.c1_tareas_requeridas,
            "c2": det.c2_consecutivas,
            "c3": det.c3_capacidad,
            "c4": det.c4_vehiculos,
            "c5": det.c5_deposito_extremos,
        },
    }


def main() -> None:
    nombres = nombres_soluciones_iniciales_disponibles(root=_ROOT)
    total   = len(nombres)
    print(f"Procesando {total} soluciones iniciales…")

    resultados: dict[str, dict] = {}

    for i, nombre in enumerate(nombres, 1):
        try:
            resultados[nombre] = _etiquetar_instancia(nombre)
            estado = "OK" if resultados[nombre]["factible"] else "NO FACTIBLE"
        except Exception as exc:
            resultados[nombre] = {"error": str(exc)}
            estado = f"ERROR: {exc}"
        print(f"  [{i:3}/{total}] {nombre:<25} → {estado}")

    # ── JSON ──────────────────────────────────────────────────────────────────
    with _SALIDA_JSON.open("w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # ── CSV ───────────────────────────────────────────────────────────────────
    with _SALIDA_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_CAMPOS, extrasaction="ignore")
        writer.writeheader()
        for nombre, r in resultados.items():
            if "error" not in r:
                writer.writerow({
                    "instancia":       nombre,
                    "factible":        r["factible"],
                    "c1_cobertura":    r["c1_cobertura"],
                    "c2_conectividad": r["c2_conectividad"],
                    "c3_capacidad":    r["c3_capacidad"],
                    "c4_vehiculos":    r["c4_vehiculos"],
                    "c5_deposito":     r["c5_deposito"],
                })

    # ── Resumen por consola ───────────────────────────────────────────────────
    factibles   = sum(1 for r in resultados.values() if r.get("factible"))
    infactibles = sum(1 for r in resultados.values() if r.get("factible") is False)
    errores     = sum(1 for r in resultados.values() if "error" in r)

    print(f"\nResultados guardados en:")
    print(f"  {_SALIDA_JSON}")
    print(f"  {_SALIDA_CSV}")
    print(f"\nResumen: {factibles} factibles | {infactibles} no factibles | {errores} errores")


if __name__ == "__main__":
    main()
