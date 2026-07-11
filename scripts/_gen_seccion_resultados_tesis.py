#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera una SECCION DE RESULTADOS limpia (nivel tesis) en LaTeX, con tablas
auto-explicativas, a partir de los CSV de ``experimentos_costo_fixed/``.

Tablas producidas:
  1. Gap medio (%) por (metaheuristica x configuracion).      [calidad media]
  2. Optimos alcanzados (gap=0 en la mejor de 5 reps) / 23.    [robustez]
  3. Efecto del Path Relinking: Delta gap (sin PR -> con PR).  [aislamiento]
  4. Detalle por instancia de la MEJOR configuracion.          [granular]

Salida: docs/seccion_resultados_tesis.tex (fragmento LaTeX, sin preambulo).
Solo stdlib.
"""
from __future__ import annotations

import csv
import glob
import os
import statistics

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(RAIZ, "experimentos_costo_fixed")
SALIDA = os.path.join(RAIZ, "docs", "seccion_resultados_tesis.tex")

ORDEN_INST = [
    "gdb19", "kshs1", "kshs2", "kshs3", "kshs4", "kshs5", "kshs6",
    "gdb4", "gdb14", "gdb15", "gdb1", "gdb20", "gdb3", "gdb6", "gdb7",
    "gdb12", "gdb10", "gdb2", "gdb5", "gdb13", "gdb16", "gdb17", "gdb21",
]
ORDEN_MH = ["sa", "tabu_simple", "tabu_reactiva", "abc_simple", "cuckoo"]
NOMBRE_MH = {
    "sa": "Recocido Simulado",
    "tabu_simple": "Tabu Simple",
    "tabu_reactiva": "Tabu Reactiva",
    "abc_simple": "Colonia de Abejas",
    "cuckoo": "Cuckoo Search",
}

# Configuracion -> (token de carpeta, funcion prefijo de fichero)
CONFIGS = [
    ("A1", "solo\\_p\\_inter", "solo_p_inter", lambda mh: f"{mh}_solo_p_inter"),
    ("A2", "binario", "binario_capacidad", lambda mh: f"{mh}_binario_capacidad"),
    ("A3p", "PR+p\\_inter", "pr_aislado", lambda mh: f"{mh}_pr_p_inter"),
    ("A3b", "PR+binario", "pr_aislado", lambda mh: f"{mh}_pr_binario"),
]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def leer(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _orden(inst):
    return ORDEN_INST.index(inst) if inst in ORDEN_INST else len(ORDEN_INST)


def dir_mh(token, mh):
    ds = sorted(glob.glob(os.path.join(BASE, f"{mh}_{token}_*")))
    return ds[0] if ds else None


def archivos(carpeta_final, prefijo):
    res = {}
    for path in glob.glob(os.path.join(carpeta_final, prefijo + "_*.csv")):
        inst = os.path.basename(path)[:-4][len(prefijo) + 1:]
        if inst.startswith("gdb") or inst.startswith("kshs"):
            res[inst] = path
    return res


def stats_variante(mh, token, prefijo_fn):
    """Devuelve (gap_media_medio, gap_best_medio, n_optimos, detalle_por_inst)."""
    carpeta = dir_mh(token, mh)
    if not carpeta:
        return None
    mapa = archivos(os.path.join(carpeta, "final"), prefijo_fn(mh))
    if not mapa:
        return None
    gaps_media, gaps_best, n_opt = [], [], 0
    detalle = {}
    for inst, path in mapa.items():
        filas = leer(path)
        costos = [_f(r["mejor_costo"]) for r in filas if _f(r["mejor_costo"]) is not None]
        gaps = [_f(r["gap_bks_porcentaje"]) for r in filas if _f(r["gap_bks_porcentaje"]) is not None]
        tiempos = [_f(r["tiempo_segundos"]) for r in filas if _f(r["tiempo_segundos"]) is not None]
        bks = _f(filas[0]["bks_referencia"]) if filas else None
        if not costos:
            continue
        gb = min(gaps); gm = statistics.mean(gaps)
        gaps_best.append(gb); gaps_media.append(gm)
        if gb <= 1e-9:
            n_opt += 1
        detalle[inst] = dict(
            bks=bks, best=min(costos), media=statistics.mean(costos),
            peor=max(costos), gap_b=gb, gap_m=gm,
            t=statistics.mean(tiempos) if tiempos else float("nan"))
    return (statistics.mean(gaps_media), statistics.mean(gaps_best), n_opt, detalle)


def main():
    # Recolectar stats de las 4 configuraciones x 5 MH.
    tabla = {}  # (mh, cfg_id) -> stats
    for cfg_id, _etq, token, fn in CONFIGS:
        for mh in ORDEN_MH:
            tabla[(mh, cfg_id)] = stats_variante(mh, token, fn)

    L = []
    a = L.append

    a(r"% === Seccion de resultados (autogenerada). Requiere booktabs. ===")
    a(r"\section{Resultados}")
    a(r"\label{sec:resultados}")
    a(r"Todos los gaps se calculan respecto al mejor valor conocido (BKS) como "
      r"$\mathrm{gap}=100\,(\,c-\mathrm{BKS}\,)/\mathrm{BKS}$, con el evaluador de "
      r"orientacion \emph{greedy} (costo corregido). Cada configuracion se ejecuto "
      r"sobre $23$ instancias est\'andar (\texttt{gdb}, \texttt{kshs}) con $5$ "
      r"repeticiones independientes. Las cuatro configuraciones comparadas son: "
      r"\textbf{A1} (selector probabil\'istico $p_{\text{inter}}$), \textbf{A2} "
      r"(selector binario determinista), y \textbf{A3} (Path Relinking sobre cada "
      r"base: \emph{PR+$p_{\text{inter}}$} y \emph{PR+binario}).")

    # ---- Tabla 1: gap medio ----
    a(r"")
    a(r"\begin{table}[htbp]\centering")
    a(r"\caption{Calidad media de la soluci\'on: gap medio respecto al BKS "
      r"(\%), promediado sobre las $5$ repeticiones y las $23$ instancias. "
      r"Menor es mejor; en \textbf{negrita} la mejor configuraci\'on de cada fila.}")
    a(r"\label{tab:gap-medio}")
    a(r"\begin{tabular}{l rrrr}")
    a(r"\toprule")
    a(r"Metaheur\'istica & A1 & A2 & A3: PR$+p_{\text{inter}}$ & A3: PR$+$binario \\")
    a(r"\midrule")
    for mh in ORDEN_MH:
        vals = []
        for cfg_id, *_ in CONFIGS:
            s = tabla[(mh, cfg_id)]
            vals.append(s[0] if s else None)
        # mejor (min) de la fila
        validos = [v for v in vals if v is not None]
        mejor = min(validos) if validos else None
        celdas = []
        for v in vals:
            if v is None:
                celdas.append("--")
            elif mejor is not None and abs(v - mejor) < 1e-9:
                celdas.append(r"\textbf{%.2f}" % v)
            else:
                celdas.append("%.2f" % v)
        a("%s & %s \\\\" % (NOMBRE_MH[mh], " & ".join(celdas)))
    a(r"\bottomrule")
    a(r"\end{tabular}")
    a(r"\end{table}")

    # ---- Tabla 2: optimos ----
    a(r"")
    a(r"\begin{table}[htbp]\centering")
    a(r"\caption{Robustez: n\'umero de instancias (de $23$) en las que la "
      r"\emph{mejor} de las $5$ repeticiones alcanza el \'optimo conocido "
      r"($\mathrm{gap}=0$). Mayor es mejor.}")
    a(r"\label{tab:optimos}")
    a(r"\begin{tabular}{l rrrr}")
    a(r"\toprule")
    a(r"Metaheur\'istica & A1 & A2 & A3: PR$+p_{\text{inter}}$ & A3: PR$+$binario \\")
    a(r"\midrule")
    for mh in ORDEN_MH:
        vals = []
        for cfg_id, *_ in CONFIGS:
            s = tabla[(mh, cfg_id)]
            vals.append(s[2] if s else None)
        mejor = max([v for v in vals if v is not None], default=None)
        celdas = []
        for v in vals:
            if v is None:
                celdas.append("--")
            elif mejor is not None and v == mejor:
                celdas.append(r"\textbf{%d}" % v)
            else:
                celdas.append("%d" % v)
        a("%s & %s \\\\" % (NOMBRE_MH[mh], " & ".join(celdas)))
    a(r"\bottomrule")
    a(r"\end{tabular}")
    a(r"\end{table}")

    # ---- Tabla 3: efecto PR (Delta gap) ----
    a(r"")
    a(r"\begin{table}[htbp]\centering")
    a(r"\caption{Aislamiento del Path Relinking: variaci\'on del gap medio (\%) "
      r"al a\~nadir PR sobre cada selector base. Valores negativos indican "
      r"mejora (gap menor) gracias a PR.}")
    a(r"\label{tab:efecto-pr}")
    a(r"\begin{tabular}{l rr rr}")
    a(r"\toprule")
    a(r"& \multicolumn{2}{c}{Base $p_{\text{inter}}$ (A1$\to$A3)} "
      r"& \multicolumn{2}{c}{Base binario (A2$\to$A3)} \\")
    a(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    a(r"Metaheur\'istica & sin PR & $\Delta_{\text{PR}}$ & sin PR & $\Delta_{\text{PR}}$ \\")
    a(r"\midrule")
    for mh in ORDEN_MH:
        a1 = tabla[(mh, "A1")]; a3p = tabla[(mh, "A3p")]
        a2 = tabla[(mh, "A2")]; a3b = tabla[(mh, "A3b")]
        def fmt(base, conpr):
            if not base or not conpr:
                return ("--", "--")
            d = conpr[0] - base[0]
            return ("%.2f" % base[0], "%+.2f" % d)
        b1, d1 = fmt(a1, a3p)
        b2, d2 = fmt(a2, a3b)
        a("%s & %s & %s & %s & %s \\\\" % (NOMBRE_MH[mh], b1, d1, b2, d2))
    a(r"\bottomrule")
    a(r"\end{tabular}")
    a(r"\end{table}")

    # ---- Determinar la mejor configuracion global (por gap medio) ----
    mejor_cfg = None; mejor_val = float("inf"); mejor_mh = None
    etq_cfg = {c[0]: c[1] for c in CONFIGS}
    for (mh, cfg_id), s in tabla.items():
        if s and s[0] < mejor_val:
            mejor_val = s[0]; mejor_cfg = cfg_id; mejor_mh = mh
    det = tabla[(mejor_mh, mejor_cfg)][3]

    # ---- Tabla 4: detalle por instancia de la mejor config ----
    a(r"")
    a(r"\begin{table}[htbp]\centering")
    a(r"\caption{Detalle por instancia de la configuraci\'on de menor gap medio "
      r"(\textbf{%s}, %s): mejor valor conocido (BKS), costo m\'inimo, medio "
      r"y m\'aximo sobre $5$ repeticiones, gap del mejor y gap medio, y tiempo "
      r"medio por corrida (s).}" % (NOMBRE_MH[mejor_mh], etq_cfg[mejor_cfg]))
    a(r"\label{tab:detalle-mejor}")
    a(r"\footnotesize")
    a(r"\begin{tabular}{l rrrr rr r}")
    a(r"\toprule")
    a(r"Instancia & BKS & M\'in & Media & M\'ax & "
      r"gap$_{\min}$ & gap$_{\text{med}}$ & $t$ (s) \\")
    a(r"\midrule")
    sgb = sgm = st = 0.0; n = 0
    for inst in sorted(det, key=_orden):
        d = det[inst]
        a(r"\texttt{%s} & %s & %s & %.1f & %s & %.2f & %.2f & %.2f \\" % (
            inst.replace("_", r"\_"),
            ("%.0f" % d["bks"]) if d["bks"] is not None else "--",
            "%.0f" % d["best"], d["media"], "%.0f" % d["peor"],
            d["gap_b"], d["gap_m"], d["t"]))
        sgb += d["gap_b"]; sgm += d["gap_m"]; st += d["t"]; n += 1
    a(r"\midrule")
    a(r"\textbf{Promedio} & & & & & \textbf{%.2f} & \textbf{%.2f} & \textbf{%.2f} \\"
      % (sgb / n, sgm / n, st / n))
    a(r"\bottomrule")
    a(r"\end{tabular}")
    a(r"\end{table}")

    # ---- Parrafo de discusion (con cifras reales, lectura honesta) ----
    s_sa_a1 = tabla[("sa", "A1")]; s_sa_a3p = tabla[("sa", "A3p")]
    d_abc = tabla[("abc_simple", "A3p")][0] - tabla[("abc_simple", "A1")][0]
    d_ck = tabla[("cuckoo", "A3p")][0] - tabla[("cuckoo", "A1")][0]
    a(r"")
    a((r"\paragraph{Discusi\'on.} El Recocido Simulado (SA) domina en todas las "
       r"configuraciones. Conviene separar dos m\'etricas. En \emph{calidad media} "
       r"(Tabla~\ref{tab:gap-medio}), el approach m\'as simple, A1, ya logra con SA "
       r"el menor gap del estudio (%.2f\%%); a\~nadir Path Relinking sobre esa base "
       r"(A3: PR$+p_{\text{inter}}$) deja el promedio de SA pr\'acticamente igual "
       r"(%.2f\%%, $\Delta_{\text{PR}}=%+.2f$), lo que confirma la hip\'otesis "
       r"metodol\'ogica: para SA la configuraci\'on simple basta y los mecanismos "
       r"adicionales se justificar\'ian por tiempo, no por calidad. En cambio, en "
       r"\emph{robustez} (Tabla~\ref{tab:optimos}) PR s\'i ayuda a SA: los \'optimos "
       r"alcanzados pasan de %d a %d sobre $23$, el mejor registro del estudio.")
      % (s_sa_a1[0], s_sa_a3p[0], s_sa_a3p[0] - s_sa_a1[0], s_sa_a1[2], s_sa_a3p[2]))
    a(r"")
    a((r"El valor de PR es m\'as claro en las metaheur\'isticas de base m\'as d\'ebil "
       r"(Tabla~\ref{tab:efecto-pr}): mejora el gap medio de la Colonia de Abejas "
       r"en $%.2f$ puntos y de Cuckoo Search en $%.2f$, adem\'as de la Tabu Simple "
       r"y la Reactiva. Es decir, PR act\'ua como un compensador: aporta poco cuando "
       r"la base ya es fuerte (SA) y mucho cuando la base explora peor. Cuckoo "
       r"Search es, en todo caso, la metaheur\'istica menos competitiva en este "
       r"corpus.") % (d_abc, d_ck))

    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    with open(SALIDA, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("Generado:", SALIDA)
    print("Mejor config global:", mejor_mh, mejor_cfg, "gap_medio=%.3f" % mejor_val)
    # Volcamos a stdout para inspeccion.
    print("\n".join(L))


if __name__ == "__main__":
    main()
