"""
Tabla LaTeX: mejores resultados de CUCKOO SEARCH por instancia (87 instancias)
con los parámetros de la corrida ganadora.

Fuentes:
  - 23 pequeñas (gdb/kshs): campaña costo-corregido de junio
    (``experimentos_costo_fixed/cuckoo_*``, 3 approaches × 5 reps, config
    calibrada). El approach ganador se reporta por fila (P/B/R).
  - 64 val/gdb/egl: campaña de transferencia con warm-start Path-Scanning
    (``experimentos_val_egl_20260710/cs_campana``, config ganadora del
    mini-grid, 5 reps). Se marca con W.

En la familia val se reporta el costo comparable (ajuste de convención
CARPLIB: ``costo + delta``, delta = COSTE_TOTAL_REQ - suma de costos de
tránsito; ver _gen_mejores_val_egl_20260712.py) y el gap contra la BKS.

Los parámetros por fila se toman DEL CSV de la corrida ganadora
(num_nidos_efectivo, pa_abandono, beta_levy, factor_pasos, p_inter), no de
tablas manuales, para trazabilidad completa.

Salidas (longtable, mismos datos, dos idiomas):
    resultados/tabla_cuckoo_por_instancia_20260712.tex     (español)
    resultados/cuckoo_results_table_en_20260712.tex        (inglés, para el
                                                            artículo)

Uso:
    python scripts/_gen_tabla_cuckoo_20260712.py
"""
from __future__ import annotations

import csv
import glob
import pickle
import statistics
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))
from run_val_egl_20260710 import INSTANCIAS_VAL_GDB, INSTANCIAS_EGL  # noqa: E402

DESTINO = RAIZ / "resultados" / "tabla_cuckoo_por_instancia_20260712.tex"

# Las 23 pequeñas en su orden canónico.
INSTANCIAS_SMALL = [
    "gdb19", "kshs1", "kshs2", "kshs3", "kshs4", "kshs5", "kshs6",
    "gdb4",  "gdb14", "gdb15", "gdb1",  "gdb20", "gdb3",  "gdb6",
    "gdb7",  "gdb12", "gdb10", "gdb2",  "gdb5",  "gdb13", "gdb16",
    "gdb17", "gdb21",
]

LETRA_APPROACH = {
    "solo_p_inter": "P", "binario_capacidad": "B", "pr_aislado": "R",
}


def _deltas() -> dict[str, float]:
    """Delta CARPLIB por instancia (0 fuera de val)."""
    out: dict[str, float] = {}
    for ruta in glob.glob(str(RAIZ / "PickleInstances" / "*.pkl")):
        d = pickle.load(open(ruta, "rb"))
        nombre = d.get("NOMBRE") or Path(ruta).stem
        meta = d.get("COSTE_TOTAL_REQ")
        suma = sum(a["costo"] for a in d["LISTA_ARISTAS_REQ"])
        out[nombre] = float(meta) - float(suma) if meta is not None else 0.0
    return out


def _mejor_por_instancia(patrones: list[tuple[str, str]]) -> dict[str, dict]:
    """{(instancia): fila ganadora con origen}. patrones = [(glob, origen)]."""
    mejores: dict[str, dict] = {}
    for patron, origen in patrones:
        for ruta in glob.glob(str(RAIZ / patron)):
            if "_partials" in ruta:
                continue
            # El origen 'P/B/R' se deduce del nombre del directorio de junio.
            org = origen
            if origen == "?":
                for ap, letra in LETRA_APPROACH.items():
                    if ap in ruta:
                        org = letra
                        break
            with open(ruta, newline="") as fh:
                for fila in csv.DictReader(fh):
                    inst = fila["instancia"]
                    c = float(fila["mejor_costo"])
                    t = float(fila["tiempo_segundos"])
                    a = mejores.get(inst)
                    if (a is None or c < a["costo"]
                            or (c == a["costo"] and t < a["tiempo"])):
                        mejores[inst] = {
                            "costo": c, "tiempo": t, "origen": org,
                            "bks": float(fila["bks_referencia"]),
                            "nidos": fila.get("num_nidos_efectivo")
                                     or fila.get("num_nidos") or "--",
                            "pa": fila.get("pa_abandono", "--"),
                            "beta": fila.get("beta_levy", "--"),
                            "fpasos": fila.get("factor_pasos") or "--",
                            "p_inter": fila.get("p_inter") or "--",
                        }
    return mejores


# Textos por idioma para render dual (mismos datos, dos salidas).
TEXTOS = {
    "es": {
        "caption": (
            r"Cuckoo Search: mejor resultado por instancia y parámetros"
            r" de la corrida ganadora. Costos en la convención CARPLIB"
            r" (servicio + deadheading; en val se aplica el ajuste $\delta$)."
            r" Config: $P$~=~\texttt{solo\_p\_inter},"
            r" $B$~=~\texttt{binario\_capacidad}, $R$~=~\texttt{pr\_aislado},"
            r" $W$~=~warm-start Path-Scanning con Path Relinking (campaña"
            r" val/egl, presupuesto $10^6$ eval.\ / 300\,s). En"
            r" \textbf{negritas}, costos que igualan la BKS."),
        "label": "tab:cuckoo-por-instancia",
        "enc": (r"Instancia & BKS & Costo & Gap\,\% & $t$(s) & nidos & $p_a$"
                r" & $\beta$ & $f_{pasos}$ & $p_{inter}$/cfg \\"),
        "resumen": "Gap medio", "bks": "BKS alcanzadas",
    },
    "en": {
        "caption": (
            r"Cuckoo Search: best result per instance and parameters of the"
            r" winning run. Costs follow the CARPLIB convention (servicing"
            r" + deadheading; the constant $\delta$ adjustment is applied on"
            r" the val family). Config: $P$~=~\texttt{solo\_p\_inter},"
            r" $B$~=~\texttt{binario\_capacidad}, $R$~=~\texttt{pr\_aislado}"
            r" (Path Relinking), $W$~=~Path-Scanning warm start with Path"
            r" Relinking (val/egl campaign, budget of $10^6$ evaluations /"
            r" 300\,s per run). Costs matching the BKS are shown in"
            r" \textbf{bold}."),
        "label": "tab:cuckoo-per-instance",
        "enc": (r"Instance & BKS & Cost & Gap\,\% & $t$(s) & nests & $p_a$"
                r" & $\beta$ & $f_{steps}$ & $p_{inter}$/cfg \\"),
        "resumen": "Mean gap", "bks": "BKS reached",
    },
}


def _render(filas: list[tuple[str, dict]], deltas: dict[str, float],
            idioma: str, destino: Path) -> None:
    """Escribe la longtable en el idioma pedido a partir de las filas."""
    t = TEXTOS[idioma]
    L: list[str] = []
    ap = L.append
    ap(r"% Generado por scripts/_gen_tabla_cuckoo_20260712.py")
    ap(r"% Requiere: \usepackage{booktabs, longtable}")
    ap(r"\begingroup\scriptsize")
    ap(r"\begin{longtable}{lrrrrrrrrl}")
    ap(rf"\caption{{{t['caption']}}}\label{{{t['label']}}}\\")
    ap(r"\toprule"); ap(t["enc"]); ap(r"\midrule"); ap(r"\endfirsthead")
    ap(r"\toprule"); ap(t["enc"]); ap(r"\midrule"); ap(r"\endhead")
    ap(r"\bottomrule"); ap(r"\endfoot")

    gaps = []
    for inst, b in filas:
        delta = deltas.get(inst, 0.0)
        costo = b["costo"] + delta
        gap = (costo - b["bks"]) / b["bks"] * 100.0
        gaps.append(gap)
        ctxt = (rf"$\mathbf{{{costo:.0f}}}$" if costo <= b["bks"]
                else f"{costo:.0f}")
        fp = b["fpasos"] if b["fpasos"] not in ("", None) else "--"
        ap(rf"\texttt{{{inst}}} & {b['bks']:.0f} & {ctxt} & {gap:.2f} &"
           rf" {b['tiempo']:.0f} & {b['nidos']} & {b['pa']} & {b['beta']} &"
           rf" {fp} & {b['p_inter']}\,({b['origen']}) \\")
    ap(r"\midrule")
    ap(rf"\multicolumn{{3}}{{l}}{{{t['resumen']}: {statistics.mean(gaps):.2f}\%}}"
       rf" & \multicolumn{{7}}{{l}}{{{t['bks']}:"
       rf" {sum(1 for g in gaps if g <= 0.0001)}/{len(gaps)}}} \\")
    ap(r"\end{longtable}")
    ap(r"\endgroup")
    destino.write_text("\n".join(L), encoding="utf-8")
    print(f"[{idioma}] Filas: {len(filas)}  ->  {destino}")


def main() -> None:
    deltas = _deltas()
    small = _mejor_por_instancia(
        [("experimentos_costo_fixed/cuckoo_*/final/*.csv", "?")])
    grandes = _mejor_por_instancia(
        [("experimentos_val_egl_20260710/cs_campana/cs_minigrid/"
          "corrida_*/final/*.csv", "W")])

    filas: list[tuple[str, dict]] = (
        [(i, small[i]) for i in INSTANCIAS_SMALL if i in small]
        + [(i, grandes[i]) for i in INSTANCIAS_VAL_GDB if i in grandes]
        + [(i, grandes[i]) for i in INSTANCIAS_EGL if i in grandes]
    )

    _render(filas, deltas, "es", DESTINO)
    _render(filas, deltas, "en",
            RAIZ / "resultados" / "cuckoo_results_table_en_20260712.tex")


if __name__ == "__main__":
    main()
