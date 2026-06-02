"""
Genera tres archivos de resultados para el capítulo de experimentación
de la tesis de maestría:

  resultados/
    01_corridas_detalle.txt          — una fila por corrida (5 765 filas)
    02_agregado_instancia.txt        — estadísticas por (approach, MH, instancia)
    03_comparativa_mh.txt            — resumen global por (approach, MH)

Fuentes incluidas (todas con evaluador de costo greedy):
  · experimentos_costo_fixed/  → enfoques canónicos (solo_p_inter, binario_capacidad,
                                  pr_aislado) – resultado final del programa experimental
  · warmstart_greedy_20260529/ → baseline con warm-start greedy (comparación)

Columnas de cada tabla:
  01 (detalle):  approach | metaheuristica | instancia | n_tareas | bks_referencia |
                 bks_origen | gap_bks | rep | semilla | mejor_costo | costo_inicial |
                 mejora_abs | mejora_pct | tiempo_s | iters_totales | mejoras |
                 factible | n_resets | costo_desde_reporte | parametros | conteo_ops

  02 (agregado): approach | metaheuristica | instancia | n_tareas | bks_referencia |
                 n_reps | media_gap | std_gap | mejor_gap | peor_gap | mediana_gap |
                 media_tiempo_s | std_tiempo_s | min_tiempo_s | max_tiempo_s |
                 tasa_factible | mejor_costo | media_costo | std_costo |
                 media_iters | media_mejoras | media_resets

  03 (comparativa): approach | metaheuristica | n_instancias | n_reps_total |
                    media_gap_global | std_gap_global | mejor_gap_global |
                    peor_gap_global | pct_gap_menor_5 | pct_gap_menor_1 |
                    tasa_factible_global | media_tiempo_s | std_tiempo_s |
                    media_iters | total_mejoras

Uso:
  python scripts/_tabla_tesis.py
"""
from __future__ import annotations

import csv
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
SALIDA = ROOT / "resultados"
SALIDA.mkdir(exist_ok=True)

# ── Operadores para conteo colapsado ──────────────────────────────────────
OPS = ["relocate_intra","swap_intra","2opt_intra",
       "relocate_inter","swap_inter","2opt_star",
       "cross_exchange","or_opt_2","or_opt_3"]

# ── Columnas que van separadas en la tabla de detalle ─────────────────────
COLS_FIJAS_DETALLE = [
    "approach","metaheuristica","instancia","n_tareas",
    "bks_referencia","bks_origen","gap_bks_porcentaje",
    "repeticion","semilla","mejor_costo","costo_solucion_inicial",
    "mejora_absoluta","mejora_porcentaje","tiempo_segundos",
    "iteraciones_totales","mejoras","mejor_solucion_factible_final",
    "n_resets_kick","costo_total_desde_reporte",
]

EXCLUIR = {
    "mejor_solucion_tr_legible","reporte_detalle_deadheading",
    "experimento","approach","lambda","usar_penalizacion_capacidad",
}
EXCLUIR_PFX = ("trayectoria_mejor_",)

# ── Fuentes de datos ──────────────────────────────────────────────────────
FUENTES = [
    # (approach_label, glob_patron, excluir_partials)
    ("solo_p_inter",      "experimentos_costo_fixed/sa_solo_p_inter_*",              True),
    ("solo_p_inter",      "experimentos_costo_fixed/tabu_simple_solo_p_inter_*",     True),
    ("solo_p_inter",      "experimentos_costo_fixed/tabu_reactiva_solo_p_inter_*",   True),
    ("solo_p_inter",      "experimentos_costo_fixed/abc_simple_solo_p_inter_*",      True),
    ("solo_p_inter",      "experimentos_costo_fixed/cuckoo_solo_p_inter_*",          True),
    ("binario_capacidad", "experimentos_costo_fixed/sa_binario_capacidad_*",         True),
    ("binario_capacidad", "experimentos_costo_fixed/tabu_simple_binario_capacidad_*",True),
    ("binario_capacidad", "experimentos_costo_fixed/tabu_reactiva_binario_capacidad_*",True),
    ("binario_capacidad", "experimentos_costo_fixed/abc_simple_binario_capacidad_*", True),
    ("binario_capacidad", "experimentos_costo_fixed/cuckoo_binario_capacidad_*",     True),
    ("pr_aislado",        "experimentos_costo_fixed/sa_pr_aislado_*",                True),
    ("pr_aislado",        "experimentos_costo_fixed/tabu_simple_pr_aislado_*",       True),
    ("pr_aislado",        "experimentos_costo_fixed/tabu_reactiva_pr_aislado_*",     True),
    ("pr_aislado",        "experimentos_costo_fixed/abc_simple_pr_aislado_*",        True),
    ("pr_aislado",        "experimentos_costo_fixed/cuckoo_pr_aislado_*",            True),
    ("warmstart_greedy",  "experimentos/warmstart_greedy_20260529_sa",               True),
    ("warmstart_greedy",  "experimentos/warmstart_greedy_20260529_tabu_simple",      True),
    ("warmstart_greedy",  "experimentos/warmstart_greedy_20260529_tabu_reactiva",    True),
    ("warmstart_greedy",  "experimentos/warmstart_greedy_20260529_abc_simple",       True),
    ("warmstart_greedy",  "experimentos/warmstart_greedy_20260529_cuckoo",           True),
]

# ── Helpers ───────────────────────────────────────────────────────────────
def es_col_op(col):
    for pfx in ("propuesto_","aceptado_","mejoraron_"):
        if col.startswith(pfx):
            return col[len(pfx):]
    return None

def es_excluida(col):
    if col in EXCLUIR: return True
    return any(col.startswith(p) for p in EXCLUIR_PFX)

def colapsar_ops(row):
    partes = []
    for op in OPS:
        p = row.get(f"propuesto_{op}","")
        a = row.get(f"aceptado_{op}","")
        m = row.get(f"mejoraron_{op}","")
        if any(x != "" for x in (p,a,m)):
            partes.append(f"{op}:P{p}/A{a}/M{m}")
    return " | ".join(partes)

def colapsar_params(row, cols_param):
    partes = []
    for c in cols_param:
        v = row.get(c,"")
        if v != "":
            partes.append(f"{c}={v}")
    return "; ".join(partes)

def fval(s, default=None):
    try: return float(s)
    except: return default

def ival(s, default=None):
    try: return int(float(s))
    except: return default

def safe_std(vals):
    if len(vals) < 2: return 0.0
    return statistics.stdev(vals)

def safe_median(vals):
    if not vals: return ""
    return statistics.median(vals)

def fmt(v, decimals=4):
    if v is None or v == "": return ""
    if isinstance(v, float): return f"{v:.{decimals}f}"
    return str(v)

# ── Leer todos los CSV ────────────────────────────────────────────────────
def leer_fuente(approach_label, patron, excluir_partials):
    dirs = sorted(ROOT.glob(patron))
    filas = []
    for d in dirs:
        # Buscar CSVs directamente o en final/
        for sub in [d, d/"final"]:
            if not sub.is_dir(): continue
            for csv_path in sorted(sub.glob("*.csv")):
                if excluir_partials and "_partials" in str(csv_path): continue
                try:
                    with open(csv_path, newline="", encoding="utf-8") as f:
                        for row in csv.DictReader(f):
                            row["_approach"] = approach_label
                            filas.append(row)
                except Exception as e:
                    print(f"  WARN {csv_path}: {e}")
    return filas

print("Cargando datos...")
TODAS: list[dict] = []
for approach_label, patron, excluir in FUENTES:
    filas = leer_fuente(approach_label, patron, excluir)
    print(f"  {approach_label} / {patron.split('/')[-1]}: {len(filas)} filas")
    TODAS.extend(filas)

print(f"\nTotal filas cargadas: {len(TODAS)}")

# Determinar columnas de parámetros (unión de todas las columnas vistas)
todas_cols: set[str] = set()
for r in TODAS: todas_cols.update(r.keys())
cols_fijas_set = set(COLS_FIJAS_DETALLE) | {"_approach"}
cols_param = sorted(
    c for c in todas_cols
    if c not in cols_fijas_set
    and not es_excluida(c)
    and es_col_op(c) is None
)

# ── TABLA 01: detalle por corrida ─────────────────────────────────────────
print("\nGenerando 01_corridas_detalle.txt ...")
cab01 = COLS_FIJAS_DETALLE + ["parametros","conteo_operadores"]
out01 = SALIDA / "01_corridas_detalle.txt"
with open(out01,"w",encoding="utf-8",newline="") as f:
    f.write("\t".join(cab01)+"\n")
    for row in TODAS:
        linea = []
        for c in COLS_FIJAS_DETALLE:
            if c == "approach":
                linea.append(row.get("_approach", row.get("approach","")))
            else:
                linea.append(str(row.get(c,"")))
        linea.append(colapsar_params(row, cols_param))
        linea.append(colapsar_ops(row))
        f.write("\t".join(linea)+"\n")
print(f"  → {out01} ({len(TODAS)} filas, {len(cab01)} columnas)")

# ── TABLA 02: agregado por (approach, MH, instancia) ─────────────────────
print("\nGenerando 02_agregado_instancia.txt ...")

# Agrupar
grupos: dict[tuple, list[dict]] = defaultdict(list)
for row in TODAS:
    approach = row.get("_approach", row.get("approach",""))
    mh = row.get("metaheuristica","")
    inst = row.get("instancia","")
    grupos[(approach, mh, inst)].append(row)

cab02 = [
    "approach","metaheuristica","instancia","n_tareas",
    "bks_referencia","bks_origen",
    "n_reps",
    "media_gap","std_gap","mejor_gap","peor_gap","mediana_gap",
    "reps_bks_alcanzado","pct_bks_alcanzado",
    "media_tiempo_s","std_tiempo_s","min_tiempo_s","max_tiempo_s",
    "tasa_factible_pct",
    "mejor_costo","media_costo","std_costo",
    "media_iters_totales","media_mejoras","media_n_resets",
    # rango se añade en postprocesado
    "rango_media_gap_en_approach",
]

filas02_raw = []  # lista de dicts para postprocesado de ranking
for (approach, mh, inst), grupo in sorted(grupos.items()):
    gaps   = [fval(r.get("gap_bks_porcentaje","")) for r in grupo]
    gaps   = [g for g in gaps if g is not None]
    tiempos= [fval(r.get("tiempo_segundos","")) for r in grupo]
    tiempos= [t for t in tiempos if t is not None]
    costos = [fval(r.get("mejor_costo","")) for r in grupo]
    costos = [c for c in costos if c is not None]
    iters  = [fval(r.get("iteraciones_totales","")) for r in grupo]
    iters  = [i for i in iters if i is not None]
    mejoras= [fval(r.get("mejoras","")) for r in grupo]
    mejoras= [m for m in mejoras if m is not None]
    resets = [fval(r.get("n_resets_kick","")) for r in grupo]
    resets = [r for r in resets if r is not None]
    factibles = [1 if str(r.get("mejor_solucion_factible_final","")).strip().lower()=="true"
                 else 0 for r in grupo]
    bks_alcanzado = sum(1 for g in gaps if g is not None and g < 0.01)

    n_tareas   = next((r.get("n_tareas","") for r in grupo if r.get("n_tareas","")), "")
    bks_ref    = next((r.get("bks_referencia","") for r in grupo if r.get("bks_referencia","")), "")
    bks_origen = next((r.get("bks_origen","") for r in grupo if r.get("bks_origen","")), "")
    media_gap_v = statistics.mean(gaps) if gaps else None

    filas02_raw.append({
        "approach": approach, "mh": mh, "inst": inst,
        "n_tareas": n_tareas, "bks_ref": bks_ref, "bks_origen": bks_origen,
        "n_reps": len(grupo),
        "media_gap": media_gap_v,
        "std_gap": safe_std(gaps),
        "mejor_gap": min(gaps) if gaps else None,
        "peor_gap": max(gaps) if gaps else None,
        "mediana_gap": safe_median(gaps) if gaps else None,
        "bks_alcanzado": bks_alcanzado,
        "pct_bks": 100.0*bks_alcanzado/len(gaps) if gaps else None,
        "media_t": statistics.mean(tiempos) if tiempos else None,
        "std_t": safe_std(tiempos),
        "min_t": min(tiempos) if tiempos else None,
        "max_t": max(tiempos) if tiempos else None,
        "tasa_fact": 100.0*sum(factibles)/len(factibles) if factibles else None,
        "mejor_costo": min(costos) if costos else None,
        "media_costo": statistics.mean(costos) if costos else None,
        "std_costo": safe_std(costos),
        "media_iters": statistics.mean(iters) if iters else None,
        "media_mejoras": statistics.mean(mejoras) if mejoras else None,
        "media_resets": statistics.mean(resets) if resets else None,
    })

# Postprocesado: ranking de media_gap por (approach, instancia)
from collections import defaultdict as _dd
rango_map: dict[tuple, int] = {}
por_ap_inst: dict[tuple, list] = _dd(list)
for d in filas02_raw:
    por_ap_inst[(d["approach"], d["inst"])].append(d)
for (ap, inst), grupo_ai in por_ap_inst.items():
    ordenados = sorted(
        grupo_ai,
        key=lambda x: (x["media_gap"] is None, x["media_gap"] if x["media_gap"] is not None else 1e9)
    )
    for rng_i, d in enumerate(ordenados, 1):
        rango_map[(ap, d["mh"], inst)] = rng_i

filas02 = []
for d in filas02_raw:
    rango = rango_map.get((d["approach"], d["mh"], d["inst"]), "")
    filas02.append([
        d["approach"], d["mh"], d["inst"],
        d["n_tareas"], d["bks_ref"], d["bks_origen"],
        d["n_reps"],
        fmt(d["media_gap"]),
        fmt(d["std_gap"]),
        fmt(d["mejor_gap"]),
        fmt(d["peor_gap"]),
        fmt(d["mediana_gap"]),
        d["bks_alcanzado"],
        fmt(d["pct_bks"], 1),
        fmt(d["media_t"], 2),
        fmt(d["std_t"], 2),
        fmt(d["min_t"], 2),
        fmt(d["max_t"], 2),
        fmt(d["tasa_fact"], 1),
        fmt(d["mejor_costo"], 1),
        fmt(d["media_costo"], 2),
        fmt(d["std_costo"], 2),
        fmt(d["media_iters"], 0),
        fmt(d["media_mejoras"], 1),
        fmt(d["media_resets"], 2),
        rango,
    ])

out02 = SALIDA / "02_agregado_instancia.txt"
with open(out02,"w",encoding="utf-8",newline="") as f:
    f.write("\t".join(cab02)+"\n")
    for fila in filas02:
        f.write("\t".join(str(x) for x in fila)+"\n")
print(f"  → {out02} ({len(filas02)} filas, {len(cab02)} columnas)")

# ── TABLA 03: comparativa global por (approach, MH) ───────────────────────
print("\nGenerando 03_comparativa_mh.txt ...")

grupos_mh: dict[tuple, list[dict]] = defaultdict(list)
for row in TODAS:
    approach = row.get("_approach", row.get("approach",""))
    mh = row.get("metaheuristica","")
    grupos_mh[(approach, mh)].append(row)

cab03 = [
    "tipo_experimento",   # "canonico" | "exploratorio"
    "approach","metaheuristica",
    "n_instancias","n_reps_total",
    "media_gap_global","std_gap_global",
    "mejor_gap_global","peor_gap_global","mediana_gap_global",
    "reps_bks_alcanzado","pct_bks_alcanzado",
    "pct_corridas_gap_lt5","pct_corridas_gap_lt2","pct_corridas_gap_lt1",
    "tasa_factible_global_pct",
    "media_tiempo_s","std_tiempo_s","max_tiempo_s",
    "media_iters_totales","total_mejoras","media_n_resets",
    "instancias_cubiertas",
]
APPROACHES_CANONICOS = {"solo_p_inter","binario_capacidad","pr_aislado"}

filas03 = []
for (approach, mh), grupo in sorted(grupos_mh.items()):
    gaps    = [fval(r.get("gap_bks_porcentaje","")) for r in grupo]
    gaps    = [g for g in gaps if g is not None]
    tiempos = [fval(r.get("tiempo_segundos","")) for r in grupo]
    tiempos = [t for t in tiempos if t is not None]
    iters   = [fval(r.get("iteraciones_totales","")) for r in grupo]
    iters   = [i for i in iters if i is not None]
    mejoras_sum = sum(fval(r.get("mejoras",""),0) or 0 for r in grupo)
    resets  = [fval(r.get("n_resets_kick","")) for r in grupo]
    resets  = [r for r in resets if r is not None]
    factibles = [1 if str(r.get("mejor_solucion_factible_final","")).strip().lower()=="true"
                 else 0 for r in grupo]
    instancias = sorted({r.get("instancia","") for r in grupo})

    bks_alc  = sum(1 for g in gaps if g is not None and g < 0.01)
    pct_bks  = fmt(100.0*bks_alc/len(gaps) if gaps else None, 1)
    pct_lt5  = fmt(100.0*sum(1 for g in gaps if g < 5)/len(gaps) if gaps else None, 1)
    pct_lt2  = fmt(100.0*sum(1 for g in gaps if g < 2)/len(gaps) if gaps else None, 1)
    pct_lt1  = fmt(100.0*sum(1 for g in gaps if g < 1)/len(gaps) if gaps else None, 1)
    tipo     = "canonico" if approach in APPROACHES_CANONICOS else "exploratorio"

    filas03.append([
        tipo, approach, mh,
        len(instancias), len(grupo),
        fmt(statistics.mean(gaps) if gaps else None),
        fmt(safe_std(gaps)),
        fmt(min(gaps) if gaps else None),
        fmt(max(gaps) if gaps else None),
        fmt(safe_median(gaps) if gaps else None),
        bks_alc, pct_bks,
        pct_lt5, pct_lt2, pct_lt1,
        fmt(100.0*sum(factibles)/len(factibles) if factibles else None, 1),
        fmt(statistics.mean(tiempos) if tiempos else None, 2),
        fmt(safe_std(tiempos), 2),
        fmt(max(tiempos) if tiempos else None, 2),
        fmt(statistics.mean(iters) if iters else None, 0),
        int(mejoras_sum),
        fmt(statistics.mean(resets) if resets else None, 2),
        " ".join(instancias),
    ])

out03 = SALIDA / "03_comparativa_mh.txt"
with open(out03,"w",encoding="utf-8",newline="") as f:
    f.write("\t".join(cab03)+"\n")
    for fila in filas03:
        f.write("\t".join(str(x) for x in fila)+"\n")
print(f"  → {out03} ({len(filas03)} filas, {len(cab03)} columnas)")
print("\nTodas las tablas generadas.")
