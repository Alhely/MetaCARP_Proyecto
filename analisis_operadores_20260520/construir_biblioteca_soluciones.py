"""
Construye una biblioteca de soluciones unicas a partir de `resultados_consolidados.csv`.

Agrupa todas las corridas por (instancia, mejor_solucion_tr_legible) y guarda
un pickle por cada solucion unica encontrada. Si N corridas distintas
convergieron a la misma solucion, queda un solo pickle con la lista completa
de las corridas que la produjeron.

Salida:
  biblioteca_soluciones/
    <instancia>/
      <hash8>_c<costo>.pkl

Cada pickle es un dict con:
  - instancia, mejor_solucion_tr_legible (string), mejor_costo
  - bks_referencia, gap_bks_porcentaje, factible
  - n_corridas_origen, metaheuristicas_encontradoras
  - corridas_origen: lista de dicts con metadatos por corrida
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

DIR_ENTRADA = Path(__file__).resolve().parent
CSV = DIR_ENTRADA / "resultados_consolidados.csv"
DIR_SALIDA = DIR_ENTRADA / "biblioteca_soluciones"


def hash_solucion(s: str) -> str:
    """Hash MD5 truncado a 8 chars; usado como prefijo unico de nombre de archivo."""
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]


def parse_params(s) -> dict:
    """Parsea el JSON de parametros_especificos. Tolera NaN y errores."""
    if pd.isna(s) or s == "{}":
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {"_parse_error": True, "_raw": str(s)}


def _to_int_or_none(v):
    return int(v) if pd.notna(v) else None


def _to_float_or_none(v):
    return float(v) if pd.notna(v) else None


def main() -> None:
    print(f"Cargando {CSV.name}...")
    df = pd.read_csv(CSV, low_memory=False)
    print(f"Filas totales: {len(df):,}")

    # Filtrar filas que efectivamente tienen una solucion legible
    df = df.dropna(subset=["mejor_solucion_tr_legible"]).reset_index(drop=True)
    print(f"Filas con mejor_solucion_tr_legible: {len(df):,}")

    DIR_SALIDA.mkdir(exist_ok=True)

    # Agrupacion clave: cada (instancia, solucion) es una entrada de la biblioteca
    grupos = df.groupby(
        ["instancia", "mejor_solucion_tr_legible"], sort=False
    )
    n_unicas = grupos.ngroups
    print(f"Soluciones unicas (instancia x solucion): {n_unicas:,}")

    # Contador por instancia y bandera de costos inconsistentes
    contador_instancia: dict[str, int] = {}
    grupos_con_costos_distintos: list[tuple[str, str]] = []
    pickles_escritos = 0

    for (instancia, sol_tr), grp in grupos:
        instancia_dir = DIR_SALIDA / instancia
        instancia_dir.mkdir(exist_ok=True)

        # Sanity: la misma solucion legible deberia tener siempre el mismo costo.
        # Si no, lo registramos pero igual escribimos el pickle con el primer costo.
        costos_observados = grp["mejor_costo"].dropna().unique()
        if len(costos_observados) > 1:
            grupos_con_costos_distintos.append(
                (instancia, hash_solucion(sol_tr))
            )
        costo = float(costos_observados[0]) if len(costos_observados) > 0 else float("nan")

        primera = grp.iloc[0]

        # Lista detallada de las corridas que produjeron esta solucion
        corridas_origen = []
        for _, row in grp.iterrows():
            corridas_origen.append({
                "metaheuristica": row["metaheuristica"],
                "repeticion": _to_int_or_none(row["repeticion"]),
                "semilla": _to_int_or_none(row["semilla"]),
                "tiempo_segundos": _to_float_or_none(row["tiempo_segundos"]),
                "archivo_origen": row["archivo_origen"],
                "carpeta_origen": row["carpeta_origen"],
                "parametros_especificos": parse_params(row.get("parametros_especificos")),
            })

        # MHs distintas que llegaron a esta misma solucion
        mhs = sorted({c["metaheuristica"] for c in corridas_origen})

        entry = {
            "instancia": instancia,
            "mejor_solucion_tr_legible": sol_tr,
            "mejor_costo": costo,
            "bks_referencia": _to_float_or_none(primera["bks_referencia"]),
            "gap_bks_porcentaje": _to_float_or_none(primera["gap_bks_porcentaje"]),
            "factible": bool(primera["mejor_solucion_factible_final"]),
            "n_corridas_origen": len(grp),
            "metaheuristicas_encontradoras": mhs,
            "corridas_origen": corridas_origen,
        }

        # Nombre: <hash>_c<costo>.pkl  (costo redondeado para legibilidad)
        h = hash_solucion(sol_tr)
        costo_str = f"{int(round(costo))}" if not pd.isna(costo) else "nan"
        nombre = f"{h}_c{costo_str}.pkl"
        ruta = instancia_dir / nombre

        with open(ruta, "wb") as f:
            pickle.dump(entry, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickles_escritos += 1
        contador_instancia[instancia] = contador_instancia.get(instancia, 0) + 1

    # Reporte
    print(f"\nPickles escritos: {pickles_escritos:,}")
    print(f"\nSoluciones unicas por instancia:")
    for ins in sorted(contador_instancia):
        print(f"  {ins:8s}  {contador_instancia[ins]:>6,d}")
    if grupos_con_costos_distintos:
        print(
            f"\n[!] {len(grupos_con_costos_distintos)} grupos con costos distintos "
            "para la misma solucion legible (revisar inconsistencias)."
        )
    print(f"\nBiblioteca en: {DIR_SALIDA}")


if __name__ == "__main__":
    main()
