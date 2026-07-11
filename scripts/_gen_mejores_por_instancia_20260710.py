"""
Genera la tabla de MEJORES VALORES POR INSTANCIA de la campaña costo-corregido.

Recorre TODOS los CSV finales de ``experimentos_costo_fixed/*/final/`` (las 3
variantes experimentales × 5 metaheurísticas × 23 instancias × 5 repeticiones)
y, para cada par (metaheurística, instancia), selecciona la corrida con MENOR
``mejor_costo``. Para cada mejor corrida registra:

  - el costo, el gap respecto a la BKS y el tiempo de esa corrida concreta;
  - el APPROACH ganador (solo_p_inter / binario_capacidad / pr_aislado);
  - los PARÁMETROS con los que se corrió. Nota metodológica importante: dentro
    de cada approach la configuración es FIJA (fue calibrada previamente en
    ``_calibracion_2knob``), de modo que "los parámetros del mejor" quedan
    completamente determinados por (metaheurística, approach) + la semilla de
    la repetición ganadora.

Salidas (en ``resultados/mejores_por_instancia_20260710/``):

  1. ``mejores_por_instancia.csv``  — tabla larga: una fila por (MH, instancia)
     con costo, BKS, gap, approach, parámetros expandidos, tiempo y semilla.
  2. ``tabla_mejores_por_instancia.tex`` — tabla compacta lista para Overleaf
     (booktabs): instancia × 5 MHs, costo con superíndice del approach ganador
     y negritas cuando el costo iguala la BKS. Incluye la leyenda de
     parámetros por metaheurística como tabla secundaria.

Uso:
    python scripts/_gen_mejores_por_instancia_20260710.py
"""
from __future__ import annotations

import csv
import glob
import json
import statistics
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "experimentos_costo_fixed"
SALIDA = RAIZ / "resultados" / "mejores_por_instancia_20260710"

# Orden canónico de las 23 instancias pequeñas (idéntico a los runners).
INSTANCIAS = [
    "gdb19", "kshs1", "kshs2", "kshs3", "kshs4", "kshs5", "kshs6",
    "gdb4",  "gdb14", "gdb15", "gdb1",  "gdb20", "gdb3",  "gdb6",
    "gdb7",  "gdb12", "gdb10", "gdb2",  "gdb5",  "gdb13", "gdb16",
    "gdb17", "gdb21",
]

# Nombre "bonito" y orden de columnas de la tabla LaTeX.
MHS = [
    ("recocido_simulado",       "SA"),
    ("busqueda_tabu_simple",    "TS"),
    ("tabu_reactiva",           "RTS"),
    ("busqueda_abejas_simple",  "ABC"),
    ("cuckoo_search",           "CS"),
]

# Letra de superíndice por approach (leyenda de la tabla LaTeX).
LETRA_APPROACH = {
    "solo_p_inter":      "P",
    "binario_capacidad": "B",
    "pr_aislado":        "R",
}

# Parámetros calibrados FIJOS por metaheurística (config_fija.json de cada
# variante + segundo knob de _calibracion_2knob/mejor_2knob.json). p_inter
# solo interviene en el approach solo_p_inter; en binario_capacidad el grupo
# inter/intra se decide de forma determinista por violación de capacidad y en
# pr_aislado se añade Path Relinking (umbral 30) sobre la base binaria.
PARAMS_CALIBRADOS = {
    "recocido_simulado":      "α=0.9, T₀=20·d_max/n, L=n², max_reheats=10, p_inter=0.5",
    "busqueda_tabu_simple":   "tenure=25, |V(s)|=40, p_inter=0.4",
    "tabu_reactiva":          "f_aumento=1.2, f_reducción=0.95, p_inter=0.5",
    "busqueda_abejas_simple": "num_fuentes=30, límite_abandono=60, p_inter=0.5",
    "cuckoo_search":          "p_a=0.15, β_Lévy=1.3, p_inter=0.1",
}


def _approach_desde_dir(nombre_dir: str) -> str:
    """Extrae el approach del nombre del directorio del experimento.

    Los directorios siguen el patrón ``<mh>_<approach>_<timestamp>``, p. ej.
    ``abc_simple_binario_capacidad_20260601-0219``. Basta con buscar cuál de
    los tres approaches conocidos aparece en el nombre.
    """
    for ap in LETRA_APPROACH:
        if ap in nombre_dir:
            return ap
    raise ValueError(f"Approach desconocido en: {nombre_dir}")


def recolectar_mejores() -> dict[tuple[str, str], dict]:
    """Devuelve {(mh, instancia): fila de la mejor corrida encontrada}."""
    mejores: dict[tuple[str, str], dict] = {}
    for ruta in glob.glob(str(ENTRADA / "*" / "final" / "*.csv")):
        approach = _approach_desde_dir(Path(ruta).parent.parent.name)
        with open(ruta, newline="") as fh:
            for fila in csv.DictReader(fh):
                mh, inst = fila["metaheuristica"], fila["instancia"]
                costo = float(fila["mejor_costo"])
                clave = (mh, inst)
                # Desempate: menor costo; a igualdad, menor tiempo (corrida
                # más barata que logró el mismo valor).
                actual = mejores.get(clave)
                t = float(fila["tiempo_segundos"])
                if (actual is None or costo < actual["mejor_costo"]
                        or (costo == actual["mejor_costo"] and t < actual["tiempo_segundos"])):
                    mejores[clave] = {
                        "metaheuristica": mh,
                        "instancia": inst,
                        "bks": float(fila["bks_referencia"]),
                        "mejor_costo": costo,
                        "gap_bks_porcentaje": float(fila["gap_bks_porcentaje"]),
                        "approach": approach,
                        "parametros": PARAMS_CALIBRADOS[mh],
                        "tiempo_segundos": t,
                        "repeticion": fila["repeticion"],
                        "semilla": fila["semilla"],
                    }
    return mejores


def escribir_csv(mejores: dict[tuple[str, str], dict]) -> Path:
    destino = SALIDA / "mejores_por_instancia.csv"
    campos = ["metaheuristica", "instancia", "bks", "mejor_costo",
              "gap_bks_porcentaje", "approach", "parametros",
              "tiempo_segundos", "repeticion", "semilla"]
    with open(destino, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        for mh, _ in MHS:
            for inst in INSTANCIAS:
                if (mh, inst) in mejores:
                    w.writerow(mejores[(mh, inst)])
    return destino


def escribir_latex(mejores: dict[tuple[str, str], dict]) -> Path:
    destino = SALIDA / "tabla_mejores_por_instancia.tex"
    lineas: list[str] = []
    ap = lineas.append
    ap(r"% Tabla generada por scripts/_gen_mejores_por_instancia_20260710.py")
    ap(r"% Campaña costo-corregido (experimentos_costo_fixed, 3 approaches x 5 reps).")
    ap(r"\begin{table}[htbp]")
    ap(r"\centering")
    ap(r"\caption{Mejor costo por instancia y metaheurística (campaña con costo"
       r" corregido). El superíndice indica el approach ganador:"
       r" $P$ = \texttt{solo\_p\_inter}, $B$ = \texttt{binario\_capacidad},"
       r" $R$ = \texttt{pr\_aislado} (Path Relinking). En \textbf{negritas},"
       r" costos que igualan la BKS.}")
    ap(r"\label{tab:mejores-por-instancia-costo-fixed}")
    ap(r"\small")
    ap(r"\begin{tabular}{lr" + "r" * len(MHS) + "}")
    ap(r"\toprule")
    ap("Instancia & BKS & " + " & ".join(n for _, n in MHS) + r" \\")
    ap(r"\midrule")
    for inst in INSTANCIAS:
        celdas = []
        for mh, _ in MHS:
            b = mejores.get((mh, inst))
            if b is None:
                celdas.append("--")
                continue
            costo = f"{b['mejor_costo']:.0f}"
            letra = LETRA_APPROACH[b["approach"]]
            # Negritas cuando iguala la mejor solución conocida.
            if b["mejor_costo"] <= b["bks"]:
                celdas.append(rf"$\mathbf{{{costo}}}^{{{letra}}}$")
            else:
                celdas.append(rf"${costo}^{{{letra}}}$")
        bks = mejores[(MHS[0][0], inst)]["bks"]
        ap(rf"\texttt{{{inst}}} & {bks:.0f} & " + " & ".join(celdas) + r" \\")
    ap(r"\midrule")
    # Resumen por columna: gap medio del mejor-por-instancia y #BKS alcanzadas.
    gaps_fila, bks_fila = [], []
    for mh, _ in MHS:
        gaps = [mejores[(mh, i)]["gap_bks_porcentaje"] for i in INSTANCIAS if (mh, i) in mejores]
        nbks = sum(1 for i in INSTANCIAS if (mh, i) in mejores
                   and mejores[(mh, i)]["mejor_costo"] <= mejores[(mh, i)]["bks"])
        gaps_fila.append(f"{statistics.mean(gaps):.2f}\\%")
        bks_fila.append(f"{nbks}/23")
    ap(r"Gap medio & -- & " + " & ".join(gaps_fila) + r" \\")
    ap(r"BKS alcanzadas & -- & " + " & ".join(bks_fila) + r" \\")
    ap(r"\bottomrule")
    ap(r"\end{tabular}")
    ap(r"\end{table}")
    ap("")
    # Tabla secundaria: parámetros calibrados fijos por metaheurística.
    ap(r"\begin{table}[htbp]")
    ap(r"\centering")
    ap(r"\caption{Parámetros calibrados usados en la campaña (fijos dentro de"
       r" cada approach; calibración de 2 knobs previa). En todos los casos"
       r" $\lambda = \max(10\cdot\text{mediana}(d), 10)$ y presupuesto"
       r" instance-aware alineado entre metaheurísticas.}")
    ap(r"\label{tab:parametros-calibrados-costo-fixed}")
    ap(r"\small")
    ap(r"\begin{tabular}{ll}")
    ap(r"\toprule")
    ap(r"Metaheurística & Parámetros \\")
    ap(r"\midrule")
    nombres = {"recocido_simulado": "Recocido Simulado (SA)",
               "busqueda_tabu_simple": "Búsqueda Tabú simple (TS)",
               "tabu_reactiva": "Búsqueda Tabú reactiva (RTS)",
               "busqueda_abejas_simple": "Colonia de Abejas (ABC)",
               "cuckoo_search": "Cuckoo Search (CS)"}
    latex_params = {
        "recocido_simulado": r"$\alpha=0.9$, $T_0=20\,d_{\max}/n$, $L=n^2$, reheats máx.\ $=10$, $p_{inter}=0.5$",
        "busqueda_tabu_simple": r"tenure $=25$, $|V(s)|=40$, $p_{inter}=0.4$",
        "tabu_reactiva": r"$f_{aumento}=1.2$, $f_{reducci\'on}=0.95$, $p_{inter}=0.5$",
        "busqueda_abejas_simple": r"fuentes $=30$, l\'imite de abandono $=60$, $p_{inter}=0.5$",
        "cuckoo_search": r"$p_a=0.15$, $\beta_{L\'evy}=1.3$, $p_{inter}=0.1$",
    }
    for mh, _ in MHS:
        ap(rf"{nombres[mh]} & {latex_params[mh]} \\")
    ap(r"\bottomrule")
    ap(r"\end{tabular}")
    ap(r"\end{table}")
    destino.write_text("\n".join(lineas), encoding="utf-8")
    return destino


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    mejores = recolectar_mejores()
    ruta_csv = escribir_csv(mejores)
    ruta_tex = escribir_latex(mejores)
    print(f"Filas: {len(mejores)} (esperadas {len(MHS) * len(INSTANCIAS)})")
    print(f"CSV  : {ruta_csv}")
    print(f"LaTeX: {ruta_tex}")


if __name__ == "__main__":
    main()
