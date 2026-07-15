"""
Tablas LaTeX en inglés por metaheurística (SA, TS, RTS, ABC), 87 instancias.

Para cada MH genera dos longtables análogas a las de Cuckoo Search:

  1. RESULTADOS por instancia con los parámetros de la corrida ganadora
     (``resultados/<mh>_results_table_en_20260713.tex``): la corrida de menor
     costo (desempate por menor tiempo) entre la campaña de junio (23
     pequeñas, 3 approaches × 5 reps, config fija calibrada) y la campaña de
     transferencia val/egl de julio (64 instancias, config por clase,
     approach pr_aislado, 5 reps). Los parámetros por fila salen del CSV
     ganador. La leyenda cfg usa P/B/R (approach de junio) y T (campaña de
     transferencia val/egl).

  2. SOLUCIONES explícitas (``resultados/<mh>_solutions_table_en_20260713.tex``):
     landscape con márgenes de 1 cm, 3 columnas (instancia, costo, solución
     como arcos (u,v); rutas separadas por '|', depósito implícito). Mismas
     corridas ganadoras que la tabla 1.

En la familia val se reporta el costo comparable (ajuste CARPLIB
``costo + delta``; ver _gen_mejores_val_egl_20260712.py).

Uso:
    python scripts/_gen_tablas_mh_20260713.py
"""
from __future__ import annotations

import csv
import glob
import statistics
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))
from run_val_egl_20260710 import INSTANCIAS_VAL_GDB, INSTANCIAS_EGL  # noqa: E402
from _gen_tabla_cuckoo_20260712 import (  # noqa: E402
    INSTANCIAS_SMALL, LETRA_APPROACH, _deltas, _mapas_tr, _solucion_arcos,
)

# Formateadores de parámetros (los CSV traen strings).
_f0 = lambda v: f"{float(v):.0f}"      # noqa: E731  entero
_f1 = lambda v: f"{float(v):.1f}"      # noqa: E731  1 decimal
_fr = lambda v: str(v)                 # noqa: E731  literal


def _fmt(fila: dict, col: str, f) -> str:
    v = fila.get(col)
    return f(v) if v not in (None, "") else "--"


# Especificación por MH: globs de campañas, nombre del CSV, y columnas de
# parámetros (encabezado LaTeX, columna CSV, formateador).
MHS = {
    "sa": {
        "titulo": "Simulated Annealing",
        "csv_mh": "recocido_simulado",
        "small": "experimentos_costo_fixed/sa_*/final/*.csv",
        "big": "experimentos_val_egl_20260710/corrida_*/sa/final/*.csv",
        "params": [(r"$T_0$", "temperatura_inicial_efectiva", _f1),
                   (r"$\alpha$", "alpha", _fr),
                   (r"$L$", "L_cadena_markov", _f0)],
        "label_res": "tab:sa-per-instance",
        "label_sol": "tab:sa-solutions",
        "extra_caption": (
            r" $T_0$ is the effective instance-aware initial temperature"
            r" ($20\,d_{\max}/n$), $\alpha$ the geometric cooling factor and"
            r" $L$ the Markov-chain length per temperature level ($n^2$)."),
    },
    "ts": {
        "titulo": "Tabu Search",
        "csv_mh": "busqueda_tabu_simple",
        "small": "experimentos_costo_fixed/tabu_simple_*/final/*.csv",
        "big": "experimentos_val_egl_20260710/corrida_*/ts/final/*.csv",
        "params": [(r"$\theta$", "tabu_tenure", _f0),
                   (r"$B$", "tam_vecindario", _f0)],
        "label_res": "tab:ts-per-instance",
        "label_sol": "tab:ts-solutions",
        "extra_caption": (
            r" $\theta$ is the tabu tenure and $B$ the number of random"
            r" neighbors sampled and evaluated per iteration."),
    },
    "rts": {
        "titulo": "Reactive Tabu Search",
        "csv_mh": "tabu_reactiva",
        "small": "experimentos_costo_fixed/tabu_reactiva_*/final/*.csv",
        "big": "experimentos_val_egl_20260710/corrida_*/rts/final/*.csv",
        "params": [(r"$\theta_0$", "tenure_inicial", _f0),
                   (r"$f_{+}$", "factor_aumento", _fr),
                   (r"$f_{-}$", "factor_reduccion", _fr),
                   (r"$B$", "tam_vecindario", _f0)],
        "label_res": "tab:rts-per-instance",
        "label_sol": "tab:rts-solutions",
        "extra_caption": (
            r" $\theta_0$ is the initial tabu tenure, $f_{+}$/$f_{-}$ the"
            r" multiplicative tenure increase/decrease factors of the"
            r" reactive scheme, and $B$ the number of random neighbors"
            r" sampled per iteration."),
    },
    "abc": {
        "titulo": "Artificial Bee Colony",
        "csv_mh": "busqueda_abejas_simple",
        "small": "experimentos_costo_fixed/abc_simple_*/final/*.csv",
        "big": "experimentos_val_egl_20260710/corrida_*/abc/final/*.csv",
        "params": [(r"$SN$", "num_fuentes_efectivo", _f0),
                   (r"$\mathit{limit}$", "limite_abandono_efectivo", _f0)],
        "label_res": "tab:abc-per-instance",
        "label_sol": "tab:abc-solutions",
        "extra_caption": (
            r" $SN$ is the number of food sources and $\mathit{limit}$ the"
            r" abandonment counter of the scout phase."),
    },
    "vdo": {
        "titulo": "Vibration Damping Optimization",
        "csv_mh": "vibration_damping",
        # VDO no participó en la campaña de junio: ambas fuentes son la
        # campaña 20260714 (val/egl + pequeñas), config por clase, pr_aislado.
        "small": "experimentos_vdo_20260714/small/corrida_*/vdo/final/*.csv",
        "big": "experimentos_vdo_20260714/val_egl/corrida_*/vdo/final/*.csv",
        "small_origen": "T",
        "params": [(r"$A_0$", "amplitud_inicial_efectiva", _f1),
                   (r"$\sigma$", "sigma_efectivo", _f1),
                   (r"$\gamma$", "gamma", _fr),
                   (r"$L$", "iteraciones_por_nivel_L", _f0)],
        "label_res": "tab:vdo-per-instance",
        "label_sol": "tab:vdo-solutions",
        "extra_caption": (
            r" $A_0$ is the effective instance-aware initial amplitude"
            r" ($20\,d_{\max}/n$), $\sigma$ the Rayleigh scale ($A_0/2$),"
            r" $\gamma$ the damping coefficient and $L$ the number of"
            r" evaluations per amplitude level ($n^2$)."),
        # Todas las corridas de VDO son de la campaña de transferencia.
        "caption_cfg": (
            r" cfg: $T$~=~class-tuned configuration with"
            r" \texttt{pr\_aislado} (Path Relinking), budget of $10^6$"
            r" evaluations / 300\,s per run (all VDO rows; the metaheuristic"
            r" did not take part in the June calibration campaign)."),
    },
}

CAPTION_RES = (
    r"{titulo}: best result per instance and parameters of the winning run."
    r" Costs follow the CARPLIB convention (the constant $\delta$ adjustment"
    r" is applied on the val family); costs matching the BKS are shown in"
    r" \textbf{{bold}}.{extra}{cfg}")

CFG_LEYENDA_DEF = (
    r" cfg: $P$~=~\texttt{solo\_p\_inter},"
    r" $B$~=~\texttt{binario\_capacidad}, $R$~=~\texttt{pr\_aislado}"
    r" (Path Relinking) on the 23-instance calibration set;"
    r" $T$~=~val/egl transfer campaign (class-tuned configuration,"
    r" \texttt{pr\_aislado}, budget of $10^6$ evaluations / 300\,s per"
    r" run).")

CAPTION_SOL = (
    r"{titulo}: best solution found per instance (the same winning runs as"
    r" Table~\ref{{{label_res}}}). Costs are reported on the BKS reference"
    r" scale (the constant CARPLIB adjustment $\delta$ is applied on the val"
    r" family); costs matching the BKS are shown in \textbf{{bold}}. Each"
    r" solution is listed as an array of required edges $(u,v)$: routes are"
    r" separated by $|$ and the depot is implicit at both ends of every"
    r" route.")


def _mejores(spec: dict) -> dict[str, dict]:
    """{instancia: fila ganadora} para una MH entre ambas campañas."""
    mejores: dict[str, dict] = {}
    fuentes = [(spec["small"], spec.get("small_origen", "?")),
               (spec["big"], "T")]
    for patron, origen in fuentes:
        for ruta in glob.glob(str(RAIZ / patron)):
            if "_partials" in ruta or "_calibracion" in ruta:
                continue
            org = origen
            if origen == "?":
                for ap, letra in LETRA_APPROACH.items():
                    if ap in ruta:
                        org = letra
                        break
            with open(ruta, newline="") as fh:
                for fila in csv.DictReader(fh):
                    if fila.get("metaheuristica") != spec["csv_mh"]:
                        continue
                    inst = fila["instancia"]
                    c = float(fila["mejor_costo"])
                    t = float(fila["tiempo_segundos"])
                    a = mejores.get(inst)
                    if (a is None or c < a["_costo"]
                            or (c == a["_costo"] and t < a["_tiempo"])):
                        fila = dict(fila)
                        fila["_costo"], fila["_tiempo"] = c, t
                        fila["_origen"] = org
                        mejores[inst] = fila
    return mejores


def _render_resultados(mh: str, spec: dict, filas: list[tuple[str, dict]],
                       deltas: dict[str, float], destino: Path) -> None:
    n_par = len(spec["params"])
    n_col = 5 + n_par + 1  # Instance BKS Cost Gap t + params + p_inter/cfg
    enc = (r"Instance & BKS & Cost & Gap\,\% & $t$(s) & "
           + " & ".join(h for h, _c, _f in spec["params"])
           + r" & $p_{inter}$/cfg \\")
    caption = CAPTION_RES.format(titulo=spec["titulo"],
                                 extra=spec["extra_caption"],
                                 cfg=spec.get("caption_cfg", CFG_LEYENDA_DEF))
    L: list[str] = []
    ap = L.append
    ap(r"% Generado por scripts/_gen_tablas_mh_20260713.py")
    ap(r"% Requiere: \usepackage{booktabs, longtable}")
    ap(r"\begingroup\scriptsize")
    ap(r"\begin{longtable}{l" + "r" * (4 + n_par) + "l}")
    ap(rf"\caption{{{caption}}}\label{{{spec['label_res']}}}\\")
    ap(r"\toprule"); ap(enc); ap(r"\midrule"); ap(r"\endfirsthead")
    ap(r"\toprule"); ap(enc); ap(r"\midrule"); ap(r"\endhead")
    ap(r"\bottomrule"); ap(r"\endfoot")

    gaps = []
    for inst, b in filas:
        costo = b["_costo"] + deltas.get(inst, 0.0)
        bks = float(b["bks_referencia"])
        gap = (costo - bks) / bks * 100.0
        gaps.append(gap)
        ctxt = (rf"$\mathbf{{{costo:.0f}}}$" if costo <= bks
                else f"{costo:.0f}")
        pars = " & ".join(_fmt(b, c, f) for _h, c, f in spec["params"])
        p_int = b.get("p_inter") or "--"
        ap(rf"\texttt{{{inst}}} & {bks:.0f} & {ctxt} & {gap:.2f} &"
           rf" {b['_tiempo']:.0f} & {pars} & {p_int}\,({b['_origen']}) \\")
    ap(r"\midrule")
    n_bks = sum(1 for g in gaps if g <= 0.0001)
    ap(rf"\multicolumn{{3}}{{l}}{{Mean gap: {statistics.mean(gaps):.2f}\%}}"
       rf" & \multicolumn{{{n_col - 3}}}{{l}}{{BKS reached:"
       rf" {n_bks}/{len(gaps)}}} \\")
    ap(r"\end{longtable}")
    ap(r"\endgroup")
    destino.write_text("\n".join(L), encoding="utf-8")
    print(f"[{mh}] resultados: {len(filas)} filas  ->  {destino}")


def _render_soluciones(mh: str, spec: dict, filas: list[tuple[str, dict]],
                       deltas: dict[str, float],
                       mapas: dict[str, dict[str, tuple[int, int]]],
                       destino: Path) -> None:
    caption = CAPTION_SOL.format(titulo=spec["titulo"],
                                 label_res=spec["label_res"])
    enc = r"Instance & Cost & Solution \\"
    L: list[str] = []
    ap = L.append
    ap(r"% Generado por scripts/_gen_tablas_mh_20260713.py")
    ap(r"% Requiere: \usepackage{booktabs, longtable, array, pdflscape,"
       r" geometry}")
    ap(r"% Márgenes mínimos solo para estas páginas apaisadas.")
    ap(r"\newgeometry{margin=1cm}")
    ap(r"\begin{landscape}")
    ap(r"\begingroup\scriptsize")
    ap(r"\begin{longtable}{lr>{\tiny\raggedright\arraybackslash}"
       r"p{0.78\linewidth}}")
    ap(rf"\caption{{{caption}}}\label{{{spec['label_sol']}}}\\")
    ap(r"\toprule"); ap(enc); ap(r"\midrule"); ap(r"\endfirsthead")
    ap(r"\toprule"); ap(enc); ap(r"\midrule"); ap(r"\endhead")
    ap(r"\bottomrule"); ap(r"\endfoot")

    for inst, b in filas:
        costo = b["_costo"] + deltas.get(inst, 0.0)
        bks = float(b["bks_referencia"])
        ctxt = (rf"$\mathbf{{{costo:.0f}}}$" if costo <= bks
                else f"{costo:.0f}")
        sol = _solucion_arcos(b.get("mejor_solucion_tr_legible", ""),
                              mapas.get(inst, {}))
        sol = sol.replace("),(", "), (")
        ap(rf"\texttt{{{inst}}} & {ctxt} & {sol} \\[2pt]")
    ap(r"\end{longtable}")
    ap(r"\endgroup")
    ap(r"\end{landscape}")
    ap(r"\restoregeometry")
    destino.write_text("\n".join(L), encoding="utf-8")
    print(f"[{mh}] soluciones: {len(filas)} filas  ->  {destino}")


def main() -> None:
    deltas = _deltas()
    mapas = _mapas_tr()
    orden = INSTANCIAS_SMALL + INSTANCIAS_VAL_GDB + INSTANCIAS_EGL
    for mh, spec in MHS.items():
        mejores = _mejores(spec)
        filas = [(i, mejores[i]) for i in orden if i in mejores]
        faltan = [i for i in orden if i not in mejores]
        if faltan:
            print(f"[{mh}] AVISO: sin corridas para {faltan}")
        _render_resultados(
            mh, spec, filas, deltas,
            RAIZ / "resultados" / f"{mh}_results_table_en_20260713.tex")
        _render_soluciones(
            mh, spec, filas, deltas, mapas,
            RAIZ / "resultados" / f"{mh}_solutions_table_en_20260713.tex")


if __name__ == "__main__":
    main()
