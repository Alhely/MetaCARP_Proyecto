"""
Tabla LaTeX: MEJOR SOLUCIÓN ENCONTRADA por instancia (87 instancias), como
arreglo de arcos requeridos (u,v) — no marcadores TR.

Para cada instancia se toma la corrida de menor costo entre TODAS las
campañas y metaheurísticas disponibles (junio: 23 pequeñas × 5 MHs × 3
approaches; julio: 64 val/gdb/egl × SA/TS/RTS/ABC + CS warm-start), con
desempate por menor tiempo. La solución se lee de la columna
``mejor_solucion_tr_legible`` del CSV ganador y cada marcador TRk se traduce
a su arista requerida (u,v) con el campo ``nodos`` del pickle.

En la familia val el costo se reporta en la escala CARPLIB comparable
(``costo + delta``; ver _gen_mejores_val_egl_20260712.py). El delta es
constante por instancia, así que no altera qué corrida gana.

Salidas (longtable, mismos datos, dos idiomas):
    resultados/tabla_soluciones_por_instancia_20260713.tex   (español)
    resultados/best_solutions_table_en_20260713.tex          (inglés)

Variante SOLO CUCKOO (los mismos ganadores que la tabla de mejores
resultados de CS, ``_gen_tabla_cuckoo_20260712.py``; la columna cfg usa su
misma leyenda P/B/R/W):
    resultados/tabla_soluciones_cuckoo_20260713.tex          (español)
    resultados/cuckoo_solutions_table_en_20260713.tex        (inglés)

Uso:
    python scripts/_gen_tabla_soluciones_20260713.py
"""
from __future__ import annotations

import csv
import glob
import pickle
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))
from run_val_egl_20260710 import INSTANCIAS_VAL_GDB, INSTANCIAS_EGL  # noqa: E402
from _gen_tabla_cuckoo_20260712 import (  # noqa: E402
    INSTANCIAS_SMALL, _deltas, _mapas_tr, _mejor_por_instancia,
    _solucion_arcos,
)

# Todas las fuentes de corridas finales (se ignoran _partials).
PATRONES = [
    "experimentos_costo_fixed/*/final/*.csv",
    "experimentos_val_egl_20260710/corrida_*/*/final/*.csv",
    "experimentos_val_egl_20260710/cs_campana/cs_minigrid/corrida_*/final/*.csv",
    "experimentos_val_egl_20260710/cs_minigrid/corrida_*/final/*.csv",
    "experimentos_vdo_20260714/*/corrida_*/vdo/final/*.csv",
]

ETIQUETA_MH = {
    "recocido_simulado": "SA",
    "busqueda_tabu_simple": "TS",
    "tabu_reactiva": "RTS",
    "busqueda_abejas_simple": "ABC",
    "cuckoo_search": "CS",
    "vibration_damping": "VDO",
}


def _mejores() -> dict[str, dict]:
    """{instancia: corrida ganadora global} entre todas las fuentes."""
    mejores: dict[str, dict] = {}
    for patron in PATRONES:
        for ruta in glob.glob(str(RAIZ / patron)):
            if "_partials" in ruta or "_calibracion" in ruta:
                continue
            with open(ruta, newline="") as fh:
                for fila in csv.DictReader(fh):
                    inst = fila.get("instancia")
                    sol = fila.get("mejor_solucion_tr_legible")
                    if not inst or not sol:
                        continue
                    c = float(fila["mejor_costo"])
                    t = float(fila["tiempo_segundos"])
                    a = mejores.get(inst)
                    if (a is None or c < a["costo"]
                            or (c == a["costo"] and t < a["tiempo"])):
                        mejores[inst] = {
                            "costo": c, "tiempo": t, "sol_tr": sol,
                            "bks": float(fila["bks_referencia"]),
                            "mh": ETIQUETA_MH.get(
                                fila.get("metaheuristica", ""),
                                fila.get("metaheuristica", "?")),
                        }
    return mejores


TEXTOS = {
    "es": {
        "caption": (
            r"Mejor solución encontrada por instancia (ganadora entre todas"
            r" las metaheurísticas y campañas, presupuesto $10^6$"
            r" evaluaciones / 300\,s por corrida). Costos en la escala de la"
            r" referencia BKS (en la familia val se aplica el ajuste"
            r" constante $\delta$ de la convención CARPLIB). En"
            r" \textbf{negritas}, costos que igualan la BKS. Cada solución"
            r" se lista como arreglo de arcos requeridos $(u,v)$: rutas"
            r" separadas por $|$, depósito implícito en los extremos de"
            r" cada ruta."),
        "label": "tab:soluciones-por-instancia",
        "enc": r"Instancia & BKS & Costo & Gap\,\% & MH \\",
    },
    "en": {
        "caption": (
            r"Best solution found per instance (winner across all"
            r" metaheuristics and campaigns, budget of $10^6$ evaluations /"
            r" 300\,s per run). Costs are reported on the BKS reference"
            r" scale (the constant CARPLIB adjustment $\delta$ is applied"
            r" on the val family). Costs matching the BKS are shown in"
            r" \textbf{bold}. Each solution is listed as an array of"
            r" required edges $(u,v)$: routes are separated by $|$ and the"
            r" depot is implicit at both ends of every route."),
        "label": "tab:best-solutions-per-instance",
        "enc": r"Instance & BKS & Cost & Gap\,\% & MH \\",
    },
    "es_cuckoo": {
        "caption": (
            r"Cuckoo Search: mejor solución encontrada por instancia"
            r" (las mismas corridas ganadoras de la"
            r" Tabla~\ref{tab:cuckoo-por-instancia}). Costos en la escala"
            r" de la referencia BKS (en la familia val se aplica el ajuste"
            r" constante $\delta$ de la convención CARPLIB); en"
            r" \textbf{negritas}, costos que igualan la BKS."
            r" Cada solución se lista como arreglo de arcos requeridos"
            r" $(u,v)$: rutas separadas por $|$, depósito implícito en los"
            r" extremos de cada ruta."),
        "label": "tab:soluciones-cuckoo",
        "enc": r"Instancia & Costo & Solución \\",
    },
    "en_cuckoo": {
        "caption": (
            r"Cuckoo Search: best solution found per instance (the same"
            r" winning runs as Table~\ref{tab:cuckoo-per-instance}). Costs"
            r" are reported on the BKS reference scale (the constant"
            r" CARPLIB adjustment $\delta$ is applied on the val family);"
            r" costs matching the BKS are shown in \textbf{bold}."
            r" Each solution is listed as an array of required edges"
            r" $(u,v)$: routes are separated by $|$ and the depot is"
            r" implicit at both ends of every route."),
        "label": "tab:cuckoo-solutions",
        "enc": r"Instance & Cost & Solution \\",
    },
}


def _render(filas: list[tuple[str, dict]], deltas: dict[str, float],
            mapas: dict[str, dict[str, tuple[int, int]]],
            idioma: str, destino: Path) -> None:
    t = TEXTOS[idioma]
    L: list[str] = []
    ap = L.append
    ap(r"% Generado por scripts/_gen_tabla_soluciones_20260713.py")
    ap(r"% Requiere: \usepackage{booktabs, longtable}")
    ap(r"\begingroup\scriptsize")
    ap(r"\begin{longtable}{lrrrl}")
    ap(rf"\caption{{{t['caption']}}}\label{{{t['label']}}}\\")
    ap(r"\toprule"); ap(t["enc"]); ap(r"\midrule"); ap(r"\endfirsthead")
    ap(r"\toprule"); ap(t["enc"]); ap(r"\midrule"); ap(r"\endhead")
    ap(r"\bottomrule"); ap(r"\endfoot")

    for inst, b in filas:
        delta = deltas.get(inst, 0.0)
        costo = b["costo"] + delta
        gap = (costo - b["bks"]) / b["bks"] * 100.0
        ctxt = (rf"$\mathbf{{{costo:.0f}}}$" if costo <= b["bks"]
                else f"{costo:.0f}")
        ap(rf"\texttt{{{inst}}} & {b['bks']:.0f} & {ctxt} & {gap:.2f} &"
           rf" {b['mh']} \\")
        sol = _solucion_arcos(b["sol_tr"], mapas.get(inst, {}))
        sol = sol.replace("),(", "), (")
        ap(rf"\multicolumn{{5}}{{p{{0.96\linewidth}}}}"
           rf"{{\tiny\raggedright\arraybackslash {sol}}} \\[2pt]")
    ap(r"\end{longtable}")
    ap(r"\endgroup")
    destino.write_text("\n".join(L), encoding="utf-8")
    print(f"[{idioma}] Filas: {len(filas)}  ->  {destino}")


def _render_cs(filas: list[tuple[str, dict]], deltas: dict[str, float],
               mapas: dict[str, dict[str, tuple[int, int]]],
               idioma: str, destino: Path) -> None:
    """Variante de 3 columnas: instancia, costo y solución explícita.

    Se emite en apaisado (entorno ``landscape`` de pdflscape); dentro del
    entorno, ``\\linewidth`` ya es el ancho apaisado, así que la columna de
    solución se ensancha sola.
    """
    t = TEXTOS[idioma]
    L: list[str] = []
    ap = L.append
    ap(r"% Generado por scripts/_gen_tabla_soluciones_20260713.py")
    ap(r"% Requiere: \usepackage{booktabs, longtable, array, pdflscape,"
       r" geometry}")
    ap(r"% Márgenes mínimos solo para estas páginas apaisadas.")
    ap(r"\newgeometry{margin=1cm}")
    ap(r"\begin{landscape}")
    ap(r"\begingroup\scriptsize")
    ap(r"\begin{longtable}{lr>{\tiny\raggedright\arraybackslash}"
       r"p{0.78\linewidth}}")
    ap(rf"\caption{{{t['caption']}}}\label{{{t['label']}}}\\")
    ap(r"\toprule"); ap(t["enc"]); ap(r"\midrule"); ap(r"\endfirsthead")
    ap(r"\toprule"); ap(t["enc"]); ap(r"\midrule"); ap(r"\endhead")
    ap(r"\bottomrule"); ap(r"\endfoot")

    for inst, b in filas:
        costo = b["costo"] + deltas.get(inst, 0.0)
        ctxt = (rf"$\mathbf{{{costo:.0f}}}$" if costo <= b["bks"]
                else f"{costo:.0f}")
        sol = _solucion_arcos(b["sol_tr"], mapas.get(inst, {}))
        sol = sol.replace("),(", "), (")
        ap(rf"\texttt{{{inst}}} & {ctxt} & {sol} \\[2pt]")
    ap(r"\end{longtable}")
    ap(r"\endgroup")
    ap(r"\end{landscape}")
    ap(r"\restoregeometry")
    destino.write_text("\n".join(L), encoding="utf-8")
    print(f"[{idioma}] Filas: {len(filas)}  ->  {destino}")


def _mejores_cuckoo() -> dict[str, dict]:
    """Ganadores de CS con la MISMA selección que la tabla de resultados
    de Cuckoo (_gen_tabla_cuckoo_20260712.py); 'mh' lleva la letra P/B/R/W."""
    small = _mejor_por_instancia(
        [("experimentos_costo_fixed/cuckoo_*/final/*.csv", "?")])
    grandes = _mejor_por_instancia(
        [("experimentos_val_egl_20260710/cs_campana/cs_minigrid/"
          "corrida_*/final/*.csv", "W")])
    out = {}
    for fuente in (small, grandes):
        for inst, b in fuente.items():
            out[inst] = {"costo": b["costo"], "bks": b["bks"],
                         "sol_tr": b["sol_tr"], "mh": b["origen"]}
    return out


def main() -> None:
    deltas = _deltas()
    mapas = _mapas_tr()
    orden = INSTANCIAS_SMALL + INSTANCIAS_VAL_GDB + INSTANCIAS_EGL

    mejores = _mejores()
    filas = [(i, mejores[i]) for i in orden if i in mejores]
    faltan = [i for i in orden if i not in mejores]
    if faltan:
        print(f"AVISO: sin corridas para {faltan}")
    _render(filas, deltas, mapas, "es",
            RAIZ / "resultados" / "tabla_soluciones_por_instancia_20260713.tex")
    _render(filas, deltas, mapas, "en",
            RAIZ / "resultados" / "best_solutions_table_en_20260713.tex")

    cuckoo = _mejores_cuckoo()
    filas_cs = [(i, cuckoo[i]) for i in orden if i in cuckoo]
    _render_cs(filas_cs, deltas, mapas, "es_cuckoo",
               RAIZ / "resultados" / "tabla_soluciones_cuckoo_20260713.tex")
    _render_cs(filas_cs, deltas, mapas, "en_cuckoo",
               RAIZ / "resultados" / "cuckoo_solutions_table_en_20260713.tex")


if __name__ == "__main__":
    main()
