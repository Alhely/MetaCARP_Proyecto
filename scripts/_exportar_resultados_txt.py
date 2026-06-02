"""
Consolida los CSV de resultados de cada approach del programa experimental
(experimentos_costo_fixed/) en archivos .txt tabulados, colapsando las
columnas de hiperparámetros en una sola columna "parametros" y los conteos
de operadores en "conteo_operadores".

Salida: resultados/
  resultados_solo_p_inter.txt
  resultados_binario_capacidad.txt
  resultados_pr_aislado.txt

Uso:
  python scripts/_exportar_resultados_txt.py
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

# ── Raíz del proyecto ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "experimentos_costo_fixed"
SALIDA = ROOT / "resultados"
SALIDA.mkdir(exist_ok=True)

# ── Columnas que quedan como columnas independientes ───────────────────────
COLS_FIJAS = [
    "approach",
    "metaheuristica",
    "instancia",
    "bks_referencia",
    "bks_origen",
    "gap_bks_porcentaje",
    "repeticion",
    "semilla",
    "tiempo_segundos",
    "mejor_costo",
    "costo_solucion_inicial",
    "mejora_absoluta",
    "mejora_porcentaje",
    "iteraciones_totales",
    "mejoras",
    "mejor_solucion_factible_final",
    "n_resets_kick",
    "costo_total_desde_reporte",
    "n_tareas",
]

# ── Columnas de conteo de operadores (se colapsan en "conteo_operadores") ──
OPS = [
    "relocate_intra", "swap_intra", "2opt_intra",
    "relocate_inter", "swap_inter", "2opt_star",
    "cross_exchange", "or_opt_2", "or_opt_3",
]

# ── Columnas excluidas completamente (texto largo / redundante) ────────────
EXCLUIR_PREFIJOS = ("trayectoria_mejor_",)
EXCLUIR_EXACTAS = {
    "mejor_solucion_tr_legible",
    "reporte_detalle_deadheading",
    "experimento",
    "approach",   # se añade desde el nombre del directorio
    "lambda",     # redundante con lambda_capacidad
    "usar_penalizacion_capacidad",  # siempre True en todas las corridas
}

# ── Approaches y sus directorios ──────────────────────────────────────────
APPROACHES = {
    "solo_p_inter":        "solo_p_inter",
    "binario_capacidad":   "binario_capacidad",
    "pr_aislado":          "pr_aislado",
}

MHS = ["sa", "tabu_simple", "tabu_reactiva", "abc_simple", "cuckoo"]


def es_col_op(col: str) -> str | None:
    """Devuelve el nombre del operador si la columna es propuesto/aceptado/mejorado."""
    for pfx in ("propuesto_", "aceptado_", "mejoraron_"):
        if col.startswith(pfx):
            return col[len(pfx):]
    return None


def es_excluida(col: str) -> bool:
    if col in EXCLUIR_EXACTAS:
        return True
    return any(col.startswith(pfx) for pfx in EXCLUIR_PREFIJOS)


def colapsar_ops(row: dict) -> str:
    """op=P{propuesto}/A{aceptado}/M{mejorado}; ..."""
    partes = []
    for op in OPS:
        p = row.get(f"propuesto_{op}", "")
        a = row.get(f"aceptado_{op}", "")
        m = row.get(f"mejoraron_{op}", "")
        if p or a or m:
            partes.append(f"{op}=P{p}/A{a}/M{m}")
    return "; ".join(partes)


def colapsar_params(row: dict, cols_param: list[str]) -> str:
    """key=val; key=val; ..."""
    partes = []
    for col in cols_param:
        val = row.get(col, "")
        if val != "":
            partes.append(f"{col}={val}")
    return "; ".join(partes)


def leer_csvs_approach(mh: str, approach_key: str) -> list[dict]:
    """Lee todos los CSV finales de un MH+approach y devuelve lista de dicts."""
    patron = f"{mh}_{approach_key}"
    dirs = sorted(BASE.glob(f"{patron}_*"))
    filas = []
    for d in dirs:
        final = d / "final"
        if not final.is_dir():
            continue
        for csv_path in sorted(final.glob("*.csv")):
            # Ignorar subdirectorios _partials
            if csv_path.parent.name == "_partials":
                continue
            try:
                with open(csv_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row["approach"] = approach_key
                        filas.append(row)
            except Exception as e:
                print(f"  WARN {csv_path}: {e}")
    return filas


def exportar_approach(approach_key: str, nombre_archivo: str):
    print(f"\n── {approach_key} ──")
    todas_filas: list[dict] = []
    for mh in MHS:
        filas = leer_csvs_approach(mh, approach_key)
        print(f"   {mh}: {len(filas)} filas")
        todas_filas.extend(filas)

    if not todas_filas:
        print("   (sin resultados)")
        return

    # Determinar columnas de parámetros: todo lo que no es fijo, no es op,
    # no está excluido, visto en al menos una fila.
    todas_cols: set[str] = set()
    for fila in todas_filas:
        todas_cols.update(fila.keys())

    cols_fijas_presentes = [c for c in COLS_FIJAS if c in todas_cols or c == "approach"]
    # Columnas de parámetros: el resto
    cols_param = sorted(
        c for c in todas_cols
        if c not in set(COLS_FIJAS)
        and not es_excluida(c)
        and es_col_op(c) is None
    )

    # Cabecera del TXT
    encabezado = cols_fijas_presentes + ["parametros", "conteo_operadores"]

    salida_path = SALIDA / nombre_archivo
    with open(salida_path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(encabezado) + "\n")
        for row in todas_filas:
            linea = []
            for col in cols_fijas_presentes:
                linea.append(str(row.get(col, "")))
            linea.append(colapsar_params(row, cols_param))
            linea.append(colapsar_ops(row))
            f.write("\t".join(linea) + "\n")

    n = len(todas_filas)
    print(f"   → {salida_path} ({n} filas, {len(encabezado)} columnas)")
    print(f"     Columnas fijas: {len(cols_fijas_presentes)}")
    print(f"     Parámetros colapsados: {len(cols_param)}: {cols_param}")


def main():
    print(f"Base: {BASE}")
    print(f"Salida: {SALIDA}")
    exportar_approach("solo_p_inter",      "resultados_solo_p_inter.txt")
    exportar_approach("binario_capacidad", "resultados_binario_capacidad.txt")
    exportar_approach("pr_aislado",        "resultados_pr_aislado.txt")
    print("\nListo.")


if __name__ == "__main__":
    main()
