"""
combinar_resultados_sa_small.py
================================
Combina y analiza los CSV de resultados del experimento SA "small simple".

Cada CSV corresponde a una instancia (gdb1..gdb21, kshs1..kshs6) y contiene
múltiples corridas (una fila por combinación de alpha × p_inter × repeticion).
El script encuentra la mejor corrida por instancia (mínimo mejor_costo),
genera un CSV resumen y una tabla legible en consola.

Uso básico:
    python scripts/combinar_resultados_sa_small.py

Uso con rutas personalizadas:
    python scripts/combinar_resultados_sa_small.py \\
        --carpeta /ruta/experimentos/sa_small_simple \\
        --salida  /ruta/salida/resumen.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

# ---------------------------------------------------------------------------
# Constantes de rutas por defecto
# ---------------------------------------------------------------------------
# CARPETA_ENTRADA: carpeta donde viven los 23 CSV de resultados SA small simple.
# RUTA_SALIDA_CSV: archivo de salida con la mejor corrida por instancia.
CARPETA_ENTRADA: Path = Path(
    "/home/alhely/Desktop/MetaCARP_Proyecto/experimentos/sa_small_simple"
)
RUTA_SALIDA_CSV: Path = CARPETA_ENTRADA / "resumen_mejores_corridas.csv"

# ---------------------------------------------------------------------------
# Columnas que se priorizan al inicio en el CSV de salida.
# El resto de columnas del CSV original se añaden a continuación en su orden
# original, para no perder ningún dato.
# ---------------------------------------------------------------------------
COLUMNAS_PRIORIDAD: list[str] = [
    "instancia",
    "n_corridas_totales",
    "bks_referencia",
    "mejor_costo",
    "gap_bks_porcentaje",
    "repeticion",
    "semilla",
    "tiempo_segundos",
]


# ---------------------------------------------------------------------------
# cargar_corridas
# ---------------------------------------------------------------------------
def cargar_corridas(carpeta: Path) -> tuple[list[dict], list[str]]:
    """
    Lee todos los archivos *.csv de `carpeta` y devuelve:
        - lista de todas las filas válidas (dicts), de todos los archivos.
        - lista de columnas tal como aparecen en el primer CSV leído (orden original).

    Una fila es válida si su columna 'metaheuristica' == 'recocido_simulado'.
    Este filtro es defensivo: en una carpeta de experimentos SA no debería
    haber filas de otra metaheurística, pero lo verificamos por robustez.

    IMPORTANTE: se usa csv.DictReader con open(..., newline="", encoding="utf-8")
    para que los campos multilínea entrecomillados (mejor_solucion_tr_legible,
    reporte_detalle_deadheading) se parseen correctamente de forma automática.
    No se procesan línea por línea; el módulo csv maneja las comillas RFC 4180.
    """
    archivos = sorted(carpeta.glob("*.csv"))

    # Si no hay archivos CSV en la carpeta el experimento no se corrió todavía
    # o la ruta está mal. Informamos y salimos con código de error.
    if not archivos:
        print(
            f"[ERROR] No se encontraron archivos CSV en: {carpeta}",
            file=sys.stderr,
        )
        sys.exit(1)

    todas_las_corridas: list[dict] = []
    columnas_originales: list[str] = []  # se captura del primer archivo leído

    for ruta_csv in archivos:
        with ruta_csv.open(newline="", encoding="utf-8") as f:
            lector = csv.DictReader(f)

            # Capturamos el orden de columnas del primer CSV.
            # Asumimos que todos los CSV del mismo experimento tienen el mismo
            # encabezado (se generaron con guardar_resultado_csv del mismo código).
            if not columnas_originales and lector.fieldnames:
                columnas_originales = list(lector.fieldnames)

            for fila in lector:
                # Filtro defensivo: solo procesar filas de recocido_simulado.
                if fila.get("metaheuristica") == "recocido_simulado":
                    todas_las_corridas.append(fila)

    return todas_las_corridas, columnas_originales


# ---------------------------------------------------------------------------
# conteos_por_instancia
# ---------------------------------------------------------------------------
def conteos_por_instancia(corridas: list[dict]) -> dict[str, int]:
    """
    Devuelve un dict {nombre_instancia: número_de_corridas}.

    Contar cuántas corridas tiene cada instancia es útil para saber si el
    experimento se completó (debería haber 11 alpha × 5 p_inter × 2 reps = 110
    corridas por instancia en el experimento SA small simple).
    """
    conteos: dict[str, int] = defaultdict(int)
    for fila in corridas:
        conteos[fila["instancia"]] += 1
    return dict(conteos)


# ---------------------------------------------------------------------------
# mejor_por_instancia
# ---------------------------------------------------------------------------
def mejor_por_instancia(corridas: list[dict]) -> dict[str, dict]:
    """
    Agrupa las corridas por instancia y devuelve, para cada una, la fila
    con el menor valor de 'mejor_costo' (convertido a float para la comparación).

    Criterio de desempate: si dos corridas tienen el mismo mejor_costo, se
    prefiere la de menor tiempo_segundos (corrida más rápida con igual calidad).

    Devuelve: dict {nombre_instancia: fila_mejor (dict con todos los campos)}
    """
    # Agrupamos primero: {instancia -> [lista de filas]}
    grupos: dict[str, list[dict]] = defaultdict(list)
    for fila in corridas:
        grupos[fila["instancia"]].append(fila)

    mejores: dict[str, dict] = {}
    for instancia, filas in grupos.items():
        # Clave de ordenación: (mejor_costo ASC, tiempo_segundos ASC).
        # Convertimos a float en el momento de la comparación para no mutar
        # las filas originales (que son strings tal como vienen del CSV).
        mejor = min(
            filas,
            key=lambda f: (float(f["mejor_costo"]), float(f["tiempo_segundos"])),
        )
        mejores[instancia] = mejor

    return mejores


# ---------------------------------------------------------------------------
# clave_orden_natural
# ---------------------------------------------------------------------------
def clave_orden_natural(nombre: str) -> tuple:
    """
    Genera una clave de ordenación para nombres de instancias con formato
    <prefijo_alfa><número>, por ejemplo: 'gdb1', 'gdb10', 'kshs6'.

    El orden lexicográfico estándar ordenaría gdb1, gdb10, gdb11, gdb2, ...
    (porque '1' < '10' como string). Esta función separa el prefijo del número
    entero para que el orden sea: gdb1, gdb2, gdb3, ..., gdb10, ..., kshs1, ...

    Ejemplo de clave generada:
        'gdb1'  -> ('gdb', 1)
        'gdb10' -> ('gdb', 10)
        'kshs1' -> ('kshs', 1)
    """
    # re.match extrae el prefijo alfabético y el sufijo numérico.
    # Si el nombre no sigue el patrón esperado, usamos (nombre, 0) como fallback.
    m = re.match(r"^([a-zA-Z_]+)(\d+)$", nombre)
    if m:
        prefijo = m.group(1)
        numero = int(m.group(2))
        return (prefijo, numero)
    return (nombre, 0)


# ---------------------------------------------------------------------------
# construir_filas_resumen
# ---------------------------------------------------------------------------
def construir_filas_resumen(
    mejores: dict[str, dict],
    conteos: dict[str, int],
    columnas_originales: list[str],
) -> tuple[list[dict], list[str]]:
    """
    Construye la lista de filas del CSV de resumen y las columnas de salida.

    Cada fila de salida contiene TODA la información de la mejor corrida de
    una instancia, más la columna extra 'n_corridas_totales'.

    El orden de columnas en la salida es:
        1. Columnas de COLUMNAS_PRIORIDAD (instancia, n_corridas_totales, etc.)
        2. El resto de columnas del CSV original, en su orden original, sin repetir.

    Esta forma de construir el encabezado garantiza que:
        - Las columnas más relevantes para la tesis aparecen primero.
        - Ninguna columna del CSV original se pierde.
        - No hay duplicados en el encabezado.
    """
    # Construimos el orden de columnas de salida:
    # primero las de prioridad, luego las originales no incluidas aún.
    columnas_prioridad_set = set(COLUMNAS_PRIORIDAD)
    columnas_restantes = [
        col for col in columnas_originales if col not in columnas_prioridad_set
    ]
    columnas_salida = COLUMNAS_PRIORIDAD + columnas_restantes

    # Ordenamos las instancias con orden natural (gdb1 < gdb2 < ... < gdb21 < kshs1...).
    instancias_ordenadas = sorted(mejores.keys(), key=clave_orden_natural)

    filas_resumen: list[dict] = []
    for instancia in instancias_ordenadas:
        fila_original = mejores[instancia]

        # Construimos la fila de salida: copiamos todos los campos originales
        # y añadimos la columna extra n_corridas_totales.
        fila_salida: dict = dict(fila_original)
        fila_salida["n_corridas_totales"] = conteos[instancia]

        filas_resumen.append(fila_salida)

    return filas_resumen, columnas_salida


# ---------------------------------------------------------------------------
# escribir_resumen_csv
# ---------------------------------------------------------------------------
def escribir_resumen_csv(
    filas_resumen: list[dict],
    ruta_salida: Path,
    columnas_salida: list[str],
) -> None:
    """
    Escribe el CSV de resumen en `ruta_salida`.

    Se usa csv.DictWriter con extrasaction='ignore' para que si alguna fila
    tiene columnas adicionales no listadas en columnas_salida (improbable),
    no se lance una excepción.

    Los campos multilínea (mejor_solucion_tr_legible, reporte_detalle_deadheading)
    se escriben tal cual: csv.writer los entrecomillará automáticamente al
    detectar saltos de línea internos, cumpliendo RFC 4180. Al leer de nuevo
    con DictReader se recuperarán correctamente.
    """
    # Creamos el directorio padre si no existe (comportamiento idempotente).
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    with ruta_salida.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columnas_salida,
            extrasaction="ignore",  # ignorar columnas extra no listadas
        )
        writer.writeheader()
        writer.writerows(filas_resumen)


# ---------------------------------------------------------------------------
# imprimir_tabla
# ---------------------------------------------------------------------------
def imprimir_tabla(filas_resumen: list[dict]) -> None:
    """
    Imprime en consola una tabla alineada con las columnas más relevantes:
        instancia, n_corridas, bks, mejor_costo, gap %, semilla, tiempo_s.

    El formateo usa f-strings con especificadores de anchura fija para que
    la tabla quede alineada aunque los valores tengan longitudes distintas.
    No se usa ninguna librería externa (tabulate, rich, etc.).
    """
    # Anchuras de columna elegidas para acomodar los valores típicos del experimento.
    ancho = {
        "instancia":  10,
        "n_corridas":  9,
        "bks":        10,
        "mejor_costo": 12,
        "gap_pct":    10,
        "semilla":    10,
        "tiempo_s":   10,
    }

    # Encabezado de la tabla.
    header = (
        f"{'instancia':>{ancho['instancia']}}"
        f"  {'n_corridas':>{ancho['n_corridas']}}"
        f"  {'bks':>{ancho['bks']}}"
        f"  {'mejor_costo':>{ancho['mejor_costo']}}"
        f"  {'gap %':>{ancho['gap_pct']}}"
        f"  {'semilla':>{ancho['semilla']}}"
        f"  {'tiempo_s':>{ancho['tiempo_s']}}"
    )
    separador = "-" * len(header)

    print()
    print("  MEJORES CORRIDAS POR INSTANCIA — SA small simple")
    print(separador)
    print(header)
    print(separador)

    for fila in filas_resumen:
        # gap_bks_porcentaje puede ser vacío si la instancia no tiene BKS cargado.
        gap_str = fila.get("gap_bks_porcentaje", "")
        try:
            gap_fmt = f"{float(gap_str):>10.4f}"
        except (ValueError, TypeError):
            gap_fmt = f"{'N/A':>10}"

        tiempo_str = fila.get("tiempo_segundos", "")
        try:
            tiempo_fmt = f"{float(tiempo_str):>10.2f}"
        except (ValueError, TypeError):
            tiempo_fmt = f"{'N/A':>10}"

        linea = (
            f"{fila['instancia']:>{ancho['instancia']}}"
            f"  {str(fila['n_corridas_totales']):>{ancho['n_corridas']}}"
            f"  {fila.get('bks_referencia', ''):>{ancho['bks']}}"
            f"  {fila.get('mejor_costo', ''):>{ancho['mejor_costo']}}"
            f"  {gap_fmt}"
            f"  {str(fila.get('semilla', '')):>{ancho['semilla']}}"
            f"  {tiempo_fmt}"
        )
        print(linea)

    print(separador)
    print()


# ---------------------------------------------------------------------------
# imprimir_resumen_global
# ---------------------------------------------------------------------------
def imprimir_resumen_global(
    n_archivos: int,
    n_corridas_total: int,
    filas_resumen: list[dict],
) -> None:
    """
    Imprime estadísticas globales del experimento:
        - Total de archivos CSV leídos.
        - Total de corridas combinadas (todas las filas de todos los archivos).
        - Número de instancias distintas encontradas.
        - Gap promedio, mediano, mejor y peor sobre las mejores corridas.

    El gap se toma del campo 'gap_bks_porcentaje' de la mejor corrida de cada
    instancia. Si alguna instancia no tiene BKS definido (gap vacío), se excluye
    del cálculo estadístico y se avisa al usuario.
    """
    n_instancias = len(filas_resumen)

    # Recopilamos los gaps válidos (convertibles a float).
    gaps_validos: list[float] = []
    instancias_sin_gap: list[str] = []
    for fila in filas_resumen:
        gap_str = fila.get("gap_bks_porcentaje", "")
        try:
            gaps_validos.append(float(gap_str))
        except (ValueError, TypeError):
            instancias_sin_gap.append(fila["instancia"])

    print("=" * 60)
    print("  RESUMEN GLOBAL DEL EXPERIMENTO SA small simple")
    print("=" * 60)
    print(f"  Archivos CSV leídos       : {n_archivos}")
    print(f"  Total de corridas         : {n_corridas_total}")
    print(f"  Instancias distintas      : {n_instancias}")

    if gaps_validos:
        gap_prom = mean(gaps_validos)
        gap_med  = median(gaps_validos)
        gap_min  = min(gaps_validos)
        gap_max  = max(gaps_validos)
        print(f"  Gap promedio (mejores)    : {gap_prom:.4f}%")
        print(f"  Gap mediano  (mejores)    : {gap_med:.4f}%")
        print(f"  Mejor gap                 : {gap_min:.4f}%")
        print(f"  Peor gap                  : {gap_max:.4f}%")

        # Identificar la instancia con menor gap (mejor resultado relativo al BKS).
        mejor_fila = min(filas_resumen, key=lambda f: float(f.get("gap_bks_porcentaje", "inf")))
        print(f"  Instancia con menor gap   : {mejor_fila['instancia']} "
              f"(gap={float(mejor_fila['gap_bks_porcentaje']):.4f}%, "
              f"costo={mejor_fila['mejor_costo']}, BKS={mejor_fila['bks_referencia']})")
    else:
        print("  No hay gaps válidos para calcular estadísticas.")

    if instancias_sin_gap:
        print(f"  Instancias sin BKS/gap    : {', '.join(instancias_sin_gap)}")

    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    """
    Define los argumentos de línea de comandos del script.

    --carpeta : carpeta de entrada con los CSV (default: CARPETA_ENTRADA).
    --salida  : ruta del CSV de salida (default: RUTA_SALIDA_CSV).

    Esto permite reutilizar el script con otros conjuntos de experimentos
    sin modificar el código, solo pasando argumentos distintos.
    """
    parser = argparse.ArgumentParser(
        description="Combina CSV de resultados SA small simple y genera resumen.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--carpeta",
        type=Path,
        default=CARPETA_ENTRADA,
        help=f"Carpeta con los CSV de entrada (default: {CARPETA_ENTRADA})",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=RUTA_SALIDA_CSV,
        help=f"Ruta del CSV de resumen de salida (default: {RUTA_SALIDA_CSV})",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Punto de entrada principal del script.

    Flujo:
        1. Parsear argumentos CLI.
        2. Cargar todas las corridas de todos los CSV de la carpeta de entrada.
        3. Calcular conteos y mejores corridas por instancia.
        4. Construir las filas del resumen con las columnas en el orden deseado.
        5. Escribir el CSV de salida.
        6. Imprimir la tabla legible en consola.
        7. Imprimir el resumen global con estadísticas.
    """
    args = _parse_args()
    carpeta: Path = args.carpeta.resolve()
    ruta_salida: Path = args.salida.resolve()

    print(f"\n[INFO] Carpeta de entrada : {carpeta}")
    print(f"[INFO] Archivo de salida  : {ruta_salida}")

    # ------------------------------------------------------------------
    # Paso 1: Cargar todas las corridas de todos los CSV.
    # ------------------------------------------------------------------
    # cargar_corridas abre cada CSV con DictReader y concatena las filas válidas.
    # También devuelve el orden de columnas del primer archivo leído.
    corridas, columnas_originales = cargar_corridas(carpeta)

    # Contamos los archivos realmente leídos (excluyendo el CSV de salida si
    # ya existe en la misma carpeta, para evitar contabilizarlo como entrada).
    n_archivos = len(sorted(carpeta.glob("*.csv")))
    # Si el archivo de salida está dentro de la carpeta de entrada, descontarlo.
    if ruta_salida.parent.resolve() == carpeta.resolve():
        # Solo descontamos si ya existe (en la primera ejecución no existe aún).
        if ruta_salida.is_file():
            n_archivos -= 1

    print(f"[INFO] Corridas cargadas  : {len(corridas)} (de {n_archivos} archivo(s))")

    # ------------------------------------------------------------------
    # Paso 2: Calcular conteos y mejores corridas por instancia.
    # ------------------------------------------------------------------
    conteos = conteos_por_instancia(corridas)
    mejores = mejor_por_instancia(corridas)

    # ------------------------------------------------------------------
    # Paso 3: Construir las filas del resumen con el orden de columnas deseado.
    # ------------------------------------------------------------------
    filas_resumen, columnas_salida = construir_filas_resumen(
        mejores, conteos, columnas_originales
    )

    # ------------------------------------------------------------------
    # Paso 4: Escribir el CSV de resumen.
    # ------------------------------------------------------------------
    escribir_resumen_csv(filas_resumen, ruta_salida, columnas_salida)
    print(f"[INFO] CSV de resumen guardado en: {ruta_salida}")

    # ------------------------------------------------------------------
    # Paso 5: Imprimir tabla legible en consola.
    # ------------------------------------------------------------------
    imprimir_tabla(filas_resumen)

    # ------------------------------------------------------------------
    # Paso 6: Imprimir resumen global con estadísticas.
    # ------------------------------------------------------------------
    imprimir_resumen_global(n_archivos, len(corridas), filas_resumen)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
