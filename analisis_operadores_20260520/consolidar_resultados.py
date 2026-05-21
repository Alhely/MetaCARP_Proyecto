"""
Consolida todos los CSV de resultados de las cinco metaheurísticas en un único
archivo. Las columnas comunes a TODAS las MH se preservan; el resto se colapsa
en una columna JSON `parametros_especificos`.

Carpetas fuente (definidas en CARPETAS):
  - sa_small_simple                       (SA)
  - tabu_simple_small_20260517            (TS Simple)
  - reactive_tabu_grid_pinter             (RTS Reactiva — grid completo con p_inter)
  - abc_simple_gpu_small_inst             (ABC Simple)
  - cuckoo_grid_small_instances_20260520  (Cuckoo Search)

Salida:
  analisis_operadores_20260520/resultados_consolidados.csv

Estructura del CSV consolidado (55 columnas):
  - 52 columnas comunes a TODAS las metaheurísticas (identidad, costos,
    contadores de los 9 operadores × 4 perspectivas, ejecución y salida)
  - 1 columna `parametros_especificos` (JSON string con las columnas
    específicas de cada algoritmo: temperatura, tenure, núm. fuentes, etc.)
  - 2 columnas de trazabilidad (`archivo_origen`, `carpeta_origen`)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# =====================================================================
# Configuración
# =====================================================================

# Raíz del proyecto: dos niveles arriba de este archivo
# (MetaCARP_Proyecto/analisis_operadores_20260520/consolidar_resultados.py)
RAIZ = Path(__file__).resolve().parent.parent
DIR_EXPERIMENTOS = RAIZ / "experimentos"

# Directorio de salida = misma carpeta que este script
DIR_SALIDA = Path(__file__).resolve().parent
ARCHIVO_SALIDA = DIR_SALIDA / "resultados_consolidados.csv"

# Carpetas de experimentos a consolidar, en el orden en que se concatenarán
CARPETAS = [
    "sa_small_simple",
    "tabu_simple_small_20260517",
    "reactive_tabu_grid_pinter",
    "abc_simple_gpu_small_inst",
    "cuckoo_grid_small_instances_20260520",
]

# Los nueve operadores de vecindario disponibles en MetaCARP
OPERADORES = [
    "relocate_intra", "swap_intra", "2opt_intra",
    "relocate_inter", "swap_inter", "2opt_star",
    "cross_exchange", "or_opt_2", "or_opt_3",
]

# Las 36 columnas que produce ContadorOperadores.resumen_csv() — comunes a todas
# las metaheurísticas modernizadas (SA, TS Simple, RTS, ABC Simple, Cuckoo).
COLUMNAS_OPERADORES = [
    f"{prefijo}_{op}"
    for prefijo in ("propuesto", "aceptado", "mejoraron", "trayectoria_mejor")
    for op in OPERADORES
]

# Columnas presentes en TODOS los CSV (intersección de los cinco encabezados).
# Total: 10 (identidad+costos) + 36 (operadores) + 6 (ejecución/salida) = 52
COLUMNAS_COMUNES = [
    # Identidad de la corrida
    "metaheuristica", "instancia", "bks_referencia", "bks_origen",
    "gap_bks_porcentaje", "repeticion", "semilla", "tiempo_segundos",
    # Costos
    "mejor_costo", "costo_solucion_inicial",
    # 36 columnas de contador de operadores
    *COLUMNAS_OPERADORES,
    # Ejecución y salida legible
    "iteraciones_totales", "mejoras", "mejor_solucion_factible_final",
    "mejor_solucion_tr_legible", "reporte_detalle_deadheading",
    "costo_total_desde_reporte",
]


# =====================================================================
# Utilidades
# =====================================================================

def _serializar_valor(v):
    """Convierte un valor numpy/pandas a un tipo JSON-serializable.

    Devuelve None para NaN (que luego se filtra al construir el dict).
    Necesario porque pandas devuelve np.int64/np.float64/np.bool_ que
    json.dumps no maneja por defecto.
    """
    if v is None:
        return None
    # Detectar NaN sin caer en errores con strings (los strings no son float)
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        # Doble chequeo: numpy.nan también es np.floating
        if np.isnan(v):
            return None
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    # str, int, float, bool nativos pasan tal cual
    return v


def colapsar_fila(row: pd.Series, columnas_especificas: list[str]) -> str:
    """Serializa columnas no-comunes de una fila a JSON ordenado por clave.

    Las claves con valor NaN/None se omiten para mantener el string compacto
    y reducir ruido al hacer json_normalize posteriormente.
    """
    d: dict[str, object] = {}
    for col in columnas_especificas:
        val = _serializar_valor(row.get(col))
        if val is not None:
            d[col] = val
    # sort_keys=True asegura que filas con las mismas claves produzcan el mismo
    # JSON (útil para deduplicación posterior por configuración)
    return json.dumps(d, ensure_ascii=False, sort_keys=True)


def procesar_archivo(ruta: Path, carpeta_nombre: str) -> pd.DataFrame:
    """Lee un CSV de resultados y lo transforma al esquema consolidado.

    - Conserva las columnas comunes en el orden definido por COLUMNAS_COMUNES.
    - Si el CSV carece de alguna columna común (raro), se rellena con NA.
    - Las columnas que no están en COLUMNAS_COMUNES se serializan a JSON en
      `parametros_especificos`.
    - Agrega trazabilidad: nombre del archivo y de la carpeta de origen.
    """
    # low_memory=False evita el warning de tipos mixtos en columnas
    # como `mejor_solucion_tr_legible` (strings largos con comas internas)
    df = pd.read_csv(ruta, low_memory=False)

    # Identificar columnas específicas = todo lo que no está en las comunes
    columnas_especificas = [c for c in df.columns if c not in COLUMNAS_COMUNES]

    # Construir DataFrame de salida respetando el orden de COLUMNAS_COMUNES
    salida = pd.DataFrame()
    for col in COLUMNAS_COMUNES:
        salida[col] = df[col] if col in df.columns else pd.NA

    # Colapsar columnas específicas a JSON (una sola pasada por fila)
    if columnas_especificas:
        salida["parametros_especificos"] = df.apply(
            lambda row: colapsar_fila(row, columnas_especificas), axis=1
        )
    else:
        salida["parametros_especificos"] = "{}"

    # Trazabilidad: permite saber de qué archivo / grid proviene cada fila
    salida["archivo_origen"] = ruta.name
    salida["carpeta_origen"] = carpeta_nombre

    return salida


# =====================================================================
# Pipeline principal
# =====================================================================

def main() -> None:
    """Concatena todos los CSV de las carpetas configuradas y escribe el consolidado."""
    fragmentos: list[pd.DataFrame] = []
    # Resumen por carpeta: (nombre, n_csv, n_filas)
    resumen: list[tuple[str, int, int]] = []

    for carpeta in CARPETAS:
        ruta_carpeta = DIR_EXPERIMENTOS / carpeta
        if not ruta_carpeta.is_dir():
            print(f"[!] Carpeta no encontrada: {ruta_carpeta}")
            continue

        # Solo CSVs directos en la carpeta (no recursivo → ignora _partials/)
        csvs = sorted(ruta_carpeta.glob("*.csv"))
        if not csvs:
            print(f"[!] Sin CSV en {ruta_carpeta}")
            continue

        n_filas_carpeta = 0
        for csv_path in csvs:
            print(f"  -> procesando {csv_path.name}")
            df = procesar_archivo(csv_path, carpeta)
            fragmentos.append(df)
            n_filas_carpeta += len(df)
        resumen.append((carpeta, len(csvs), n_filas_carpeta))
        print(f"[OK] {carpeta}: {len(csvs)} CSV, {n_filas_carpeta:,} filas")

    if not fragmentos:
        print("[ERR] No se procesó ningún CSV.")
        return

    print("\nConcatenando fragmentos...")
    consolidado = pd.concat(fragmentos, ignore_index=True)

    print(f"Escribiendo {ARCHIVO_SALIDA} ...")
    consolidado.to_csv(ARCHIVO_SALIDA, index=False)

    # Reporte final
    print("\n=========== RESUMEN ===========")
    total_csv = 0
    total_filas = 0
    for carpeta, n_csv, n_filas in resumen:
        print(f"  {carpeta:50s} {n_csv:3d} CSV  {n_filas:>10,d} filas")
        total_csv += n_csv
        total_filas += n_filas
    print(f"  {'TOTAL':50s} {total_csv:3d} CSV  {len(consolidado):>10,d} filas")
    print(f"\nColumnas: {len(consolidado.columns)}")
    print(f"Salida:   {ARCHIVO_SALIDA}")


if __name__ == "__main__":
    main()
