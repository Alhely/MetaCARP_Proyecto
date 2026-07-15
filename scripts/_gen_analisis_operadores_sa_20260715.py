"""
Análisis de operadores de vecindario para RECOCIDO SIMULADO (SA):
clase egl (grandes) vs baseline gdb pequeñas.

Fuentes (mismo protocolo: selector p_inter + Path Relinking aislado):
  - egl: experimentos_val_egl_20260710/corrida_*/sa/final/sa_*_egl-*.csv
    (24 instancias × 5 reps, presupuesto 10^6 eval / 300 s, alpha=0.95,
    p_inter=0.6)
  - gdb: experimentos_costo_fixed/sa_pr_aislado_*/final/sa_pr_p_inter_gdb*.csv
    (17 instancias × 5 reps, config calibrada alpha=0.9, p_inter=0.5)

Para cada operador se agregan los 4 contadores por corrida y se reportan:
  - Prop. %  : participación en las propuestas (refleja el sesgo p_inter).
  - Acc. %   : tasa de aceptación (aceptado / propuesto).
  - Imp. /M  : mejoras del mejor global por millón de propuestas.
  - Traj. %  : participación en la trayectoria de movimientos que produjo
               la mejor solución final (composición del "camino ganador").

Salidas:
  resultados/sa_operator_analysis_en_20260715.tex   (booktabs, artículo)
  stdout: resumen agregado intra vs inter y ranking por Traj. %.

Uso:
    python scripts/_gen_analisis_operadores_sa_20260715.py
"""
from __future__ import annotations

import csv
import glob
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "resultados" / "sa_operator_analysis_en_20260715.tex"

# (clave CSV, etiqueta LaTeX, grupo)
OPERADORES = [
    ("relocate_intra", r"\texttt{relocate}",       "intra"),
    ("swap_intra",     r"\texttt{swap}",           "intra"),
    ("2opt_intra",     r"\texttt{2-opt}",          "intra"),
    ("relocate_inter", r"\texttt{relocate}",       "inter"),
    ("swap_inter",     r"\texttt{swap}",           "inter"),
    ("2opt_star",      r"\texttt{2-opt$^{*}$}",    "inter"),
    ("cross_exchange", r"\texttt{cross-exchange}", "inter"),
    ("or_opt_2",       r"\texttt{Or-opt-2}",       "inter"),
    ("or_opt_3",       r"\texttt{Or-opt-3}",       "inter"),
]

FUENTES = {
    "egl": "experimentos_val_egl_20260710/corrida_*/sa/final/sa_*_egl-*.csv",
    "gdb": "experimentos_costo_fixed/sa_pr_aislado_*/final/sa_pr_p_inter_gdb*.csv",
}


def _agregar(patron: str) -> tuple[dict[str, dict[str, int]], int]:
    """Suma los 4 contadores por operador sobre todas las corridas del glob."""
    tot = {op: {"prop": 0, "acep": 0, "mej": 0, "tray": 0}
           for op, _l, _g in OPERADORES}
    corridas = 0
    for ruta in glob.glob(str(RAIZ / patron)):
        if "_partials" in ruta:
            continue
        with open(ruta, newline="") as fh:
            for fila in csv.DictReader(fh):
                if fila.get("metaheuristica") != "recocido_simulado":
                    continue
                corridas += 1
                for op, _l, _g in OPERADORES:
                    tot[op]["prop"] += int(float(fila[f"propuesto_{op}"]))
                    tot[op]["acep"] += int(float(fila[f"aceptado_{op}"]))
                    tot[op]["mej"] += int(float(fila[f"mejoraron_{op}"]))
                    tot[op]["tray"] += int(float(fila[f"trayectoria_mejor_{op}"]))
    return tot, corridas


def _metricas(tot: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    sum_prop = sum(v["prop"] for v in tot.values())
    sum_tray = sum(v["tray"] for v in tot.values())
    out = {}
    for op, v in tot.items():
        out[op] = {
            "prop_pct": 100.0 * v["prop"] / max(1, sum_prop),
            "acc_pct": 100.0 * v["acep"] / max(1, v["prop"]),
            "imp_pm": 1e6 * v["mej"] / max(1, v["prop"]),
            "tray_pct": 100.0 * v["tray"] / max(1, sum_tray),
        }
    return out


CAPTION = (
    r"Operator-level behavior of Simulated Annealing on the large egl class"
    r" (24 instances $\times$ 5 runs, $\alpha=0.95$, $p_{inter}=0.6$, budget"
    r" of $10^6$ evaluations / 300\,s per run) against the small gdb"
    r" baseline (17 instances $\times$ 5 runs, calibrated $\alpha=0.9$,"
    r" $p_{inter}=0.5$), both under the \texttt{pr\_aislado} protocol."
    r" Prop.\,\% is the share of proposed moves (it reflects the"
    r" $p_{inter}$ group bias); Acc.\,\% the acceptance rate of the"
    r" operator; Imp.\,/M the number of improvements of the incumbent per"
    r" million proposals; Traj.\,\% the share of the operator in the move"
    r" trajectory that produced the final best solution.")


def _render(m_egl: dict, m_gdb: dict, n_egl: int, n_gdb: int) -> None:
    L: list[str] = []
    ap = L.append
    ap(r"% Generado por scripts/_gen_analisis_operadores_sa_20260715.py")
    ap(r"% Requiere: \usepackage{booktabs, multirow}")
    ap(r"\begin{table}[htbp]")
    ap(r"\centering")
    ap(rf"\caption{{{CAPTION}}}")
    ap(r"\label{tab:sa-operator-analysis}")
    ap(r"\small")
    ap(r"\begin{tabular}{llrrrrcrrrr}")
    ap(r"\toprule")
    ap(r" & & \multicolumn{4}{c}{egl (large)} & \phantom{x} &"
       r" \multicolumn{4}{c}{gdb (small baseline)} \\")
    ap(r"\cmidrule{3-6} \cmidrule{8-11}")
    ap(r"Group & Operator & Prop.\,\% & Acc.\,\% & Imp.\,/M & Traj.\,\% & &"
       r" Prop.\,\% & Acc.\,\% & Imp.\,/M & Traj.\,\% \\")
    ap(r"\midrule")
    grupo_ant = None
    for op, etiqueta, grupo in OPERADORES:
        if grupo_ant is not None and grupo != grupo_ant:
            ap(r"\midrule")
        g = grupo if grupo != grupo_ant else ""
        grupo_ant = grupo
        e, b = m_egl[op], m_gdb[op]
        ap(rf"{g} & {etiqueta} &"
           rf" {e['prop_pct']:.1f} & {e['acc_pct']:.1f} &"
           rf" {e['imp_pm']:.1f} & {e['tray_pct']:.1f} & &"
           rf" {b['prop_pct']:.1f} & {b['acc_pct']:.1f} &"
           rf" {b['imp_pm']:.1f} & {b['tray_pct']:.1f} \\")
    ap(r"\bottomrule")
    ap(r"\end{tabular}")
    ap(r"\end{table}")
    DESTINO.write_text("\n".join(L), encoding="utf-8")
    print(f"Tabla ({n_egl} corridas egl, {n_gdb} corridas gdb) -> {DESTINO}")


def main() -> None:
    tot_egl, n_egl = _agregar(FUENTES["egl"])
    tot_gdb, n_gdb = _agregar(FUENTES["gdb"])
    m_egl, m_gdb = _metricas(tot_egl), _metricas(tot_gdb)
    _render(m_egl, m_gdb, n_egl, n_gdb)

    # Resumen en consola: agregado por grupo y ranking por trayectoria.
    for nombre, tot in (("egl", tot_egl), ("gdb", tot_gdb)):
        intra = {k: sum(tot[op][k] for op, _l, g in OPERADORES if g == "intra")
                 for k in ("prop", "acep", "mej", "tray")}
        inter = {k: sum(tot[op][k] for op, _l, g in OPERADORES if g == "inter")
                 for k in ("prop", "acep", "mej", "tray")}
        st = intra["tray"] + inter["tray"]
        print(f"\n[{nombre}] intra: acc={100*intra['acep']/intra['prop']:.1f}%"
              f" tray={100*intra['tray']/st:.1f}%  |  inter:"
              f" acc={100*inter['acep']/inter['prop']:.1f}%"
              f" tray={100*inter['tray']/st:.1f}%")
        m = _metricas(tot)
        rank = sorted(m.items(), key=lambda kv: -kv[1]["tray_pct"])
        print(f"[{nombre}] ranking Traj.%: "
              + ", ".join(f"{op}={v['tray_pct']:.1f}" for op, v in rank))


if __name__ == "__main__":
    main()
