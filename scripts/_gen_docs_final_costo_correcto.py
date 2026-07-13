#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador del documento LaTeX ``docs/latex/es/docs_final_costo_correcto_con_resultados.tex``.

Recorre TODOS los experimentos lanzados DESPUES de eliminar el calculador de
costo incorrecto (commit dab620b, "orientacion greedy como unica logica del
evaluador") y produce un unico .tex AUTOCONTENIDO y compilable en Overleaf
(pdflatex, >=2 pases) que:

  1. Narra el proceso experimental completo (correccion del costo, hook
     intensificador, calibracion de parametros, los 3 approaches).
  2. Vuelca los RESULTADOS EXTENSOS de cada corrida leidos de los CSV en
     ``experimentos_costo_fixed/``: nombres de los CSV de origen, tabla resumen
     por instancia (longtable), detalle por repeticion y la MEJOR solucion por
     instancia (rutas + desglose de deadheading).
  3. Embebe VERBATIM (como LaTeX real) los capitulos ya redactados de cada
     approach (docs/latex/es/experimento_*_costo_fixed.tex): tablas booktabs extensas,
     pseudocodigos en entorno algorithm y conclusiones.

No depende de pandas: usa unicamente la biblioteca estandar.
"""
from __future__ import annotations

import csv
import glob
import os
import json
import statistics
import unicodedata

# Raiz del proyecto = un nivel arriba de scripts/
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(RAIZ, "experimentos_costo_fixed")
SALIDA = os.path.join(RAIZ, "docs", "latex", "es", "docs_final_costo_correcto_con_resultados.tex")

# Capitulos LaTeX (fragmentos sin preambulo) ya redactados por approach.
TEX_POR_APPROACH = {
    "solo_p_inter": os.path.join(RAIZ, "docs", "latex", "es", "experimento_solo_p_inter_costo_fixed.tex"),
    "binario_capacidad": os.path.join(RAIZ, "docs", "latex", "es", "experimento_binario_capacidad_costo_fixed.tex"),
    "pr_aislado": os.path.join(RAIZ, "docs", "latex", "es", "experimento_pr_aislado_costo_fixed.tex"),
}

ORDEN_INST = [
    "gdb19", "kshs1", "kshs2", "kshs3", "kshs4", "kshs5", "kshs6",
    "gdb4", "gdb14", "gdb15", "gdb1", "gdb20", "gdb3", "gdb6", "gdb7",
    "gdb12", "gdb10", "gdb2", "gdb5", "gdb13", "gdb16", "gdb17", "gdb21",
]

NOMBRE_MH = {
    "sa": "Recocido Simulado (SA)",
    "tabu_simple": "Busqueda Tabu Simple (TS)",
    "tabu_reactiva": "Busqueda Tabu Reactiva (RTS)",
    "abc_simple": "Colonia Artificial de Abejas (ABC)",
    "cuckoo": "Cuckoo Search (CS)",
}
ORDEN_MH = ["sa", "tabu_simple", "tabu_reactiva", "abc_simple", "cuckoo"]


# ============================================================
# Utilidades LaTeX
# ============================================================

_ESC = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def T(s: str) -> str:
    """Escapa una cadena para texto normal LaTeX."""
    return "".join(_ESC.get(c, c) for c in str(s))


def fold(s: str) -> str:
    """Pliega a ASCII (quita acentos y caracteres no ASCII) para verbatim.

    El contenido verbatim (rutas, deadheading) procede de la salida del
    programa; plegarlo a ASCII garantiza que ``lstlisting`` compile en Overleaf
    sin depender de la configuracion de inputenc/fontenc.
    """
    nf = unicodedata.normalize("NFKD", str(s))
    sin_acentos = "".join(c for c in nf if not unicodedata.combining(c))
    return sin_acentos.encode("ascii", "ignore").decode("ascii")


def slug(s: str) -> str:
    """Etiqueta (\\label) segura a partir de un titulo."""
    base = fold(s).lower()
    return "".join(c if c.isalnum() else "-" for c in base).strip("-")


def lst(lineas: list[str]) -> list[str]:
    """Envuelve lineas en un entorno lstlisting (verbatim, ASCII)."""
    out = [r"\begin{lstlisting}[style=dump]"]
    out.extend(fold(l) for l in lineas)
    out.append(r"\end{lstlisting}")
    return out


# ============================================================
# Lectura de datos
# ============================================================

def _orden_inst_key(inst: str) -> int:
    return ORDEN_INST.index(inst) if inst in ORDEN_INST else len(ORDEN_INST)


def leer_csv_instancia(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(x: str) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def archivos_instancia(carpeta_final: str, prefijo: str) -> dict[str, str]:
    res: dict[str, str] = {}
    for path in glob.glob(os.path.join(carpeta_final, prefijo + "_*.csv")):
        base = os.path.basename(path)[:-4]
        inst = base[len(prefijo) + 1:]
        if inst.startswith("gdb") or inst.startswith("kshs"):
            res[inst] = path
    return res


def cargar_json(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def dir_por_mh_approach(approach_token: str) -> dict[str, str]:
    res: dict[str, str] = {}
    for d in sorted(glob.glob(os.path.join(BASE, "*_" + approach_token + "_*"))):
        nombre = os.path.basename(d)
        mh = nombre.split("_" + approach_token + "_")[0]
        res[mh] = d
    return res


# ============================================================
# Bloque LaTeX por variante (MH x approach)
# ============================================================

def bloque_variante(titulo: str, mapa_inst: dict[str, str]) -> list[str]:
    out: list[str] = []
    out.append(r"\subsection{%s}" % T(titulo))

    instancias = sorted(mapa_inst, key=_orden_inst_key)
    if not instancias:
        out.append(r"\emph{(sin datos)}")
        return out

    # --- Procedencia: CSV de origen ---
    carpeta_rel = os.path.relpath(os.path.dirname(next(iter(mapa_inst.values()))), RAIZ)
    out.append(r"\paragraph{Archivos CSV de origen.} Carpeta \texttt{%s/}:" % T(carpeta_rel))
    out.extend(lst(["{:<8} -> {}".format(inst, os.path.basename(mapa_inst[inst]))
                    for inst in instancias]))

    # --- Tabla resumen por instancia (longtable) ---
    etiqueta = slug(titulo)
    out.append(r"\paragraph{Resumen por instancia.}")
    out.append(r"\begingroup\footnotesize")
    out.append(r"\begin{longtable}{l r r r r r r r r}")
    out.append(r"\caption{Resumen por instancia --- %s. best/media/peor sobre 5 reps; "
               r"gap\_b/gap\_m vs BKS; t\_med tiempo medio; n\_reset reinicios medios.}"
               r"\label{tab:res-%s}\\" % (T(titulo), etiqueta))
    cab = (r"\toprule inst & BKS & best & media & peor & gap\_b\% & gap\_m\% & "
           r"t\_med(s) & n\_reset \\ \midrule")
    out.append(cab)
    out.append(r"\endfirsthead")
    out.append(r"\multicolumn{9}{l}{\footnotesize\itshape continuacion tabla \ref{tab:res-%s}}\\"
               % etiqueta)
    out.append(cab)
    out.append(r"\endhead")
    out.append(r"\bottomrule \endlastfoot")

    gaps_best_var: list[float] = []
    gaps_media_var: list[float] = []
    n_opt = 0
    detalle: list[tuple[str, list[dict]]] = []

    for inst in instancias:
        filas = leer_csv_instancia(mapa_inst[inst])
        detalle.append((inst, filas))
        costos = [_f(r["mejor_costo"]) for r in filas if _f(r["mejor_costo"]) is not None]
        gaps = [_f(r["gap_bks_porcentaje"]) for r in filas if _f(r["gap_bks_porcentaje"]) is not None]
        tiempos = [_f(r["tiempo_segundos"]) for r in filas if _f(r["tiempo_segundos"]) is not None]
        resets = [_f(r.get("n_resets_kick", "")) for r in filas]
        resets = [r for r in resets if r is not None]
        bks = _f(filas[0]["bks_referencia"]) if filas else None
        if not costos:
            continue
        best = min(costos); media = statistics.mean(costos); peor = max(costos)
        gap_b = min(gaps) if gaps else float("nan")
        gap_m = statistics.mean(gaps) if gaps else float("nan")
        t_med = statistics.mean(tiempos) if tiempos else float("nan")
        n_reset_med = statistics.mean(resets) if resets else 0.0
        gaps_best_var.append(gap_b); gaps_media_var.append(gap_m)
        if gap_b <= 1e-9:
            n_opt += 1
        out.append(r"%s & %s & %s & %.1f & %s & %.3f & %.3f & %.2f & %.1f \\" % (
            T(inst), f"{bks:.0f}" if bks is not None else "-",
            f"{best:.0f}", media, f"{peor:.0f}", gap_b, gap_m, t_med, n_reset_med))
    out.append(r"\end{longtable}\endgroup")

    if gaps_best_var:
        out.append(r"\noindent\textbf{Global:} gap\_best medio = %.3f\%%, "
                   r"gap\_media medio = %.3f\%%, optimos (best) = %d/%d." % (
                       statistics.mean(gaps_best_var), statistics.mean(gaps_media_var),
                       n_opt, len(instancias)))

    # --- Detalle por repeticion (verbatim) ---
    out.append(r"\paragraph{Detalle por repeticion.} "
               r"(rep $\mid$ costo\_ini $\to$ mejor\_costo $\mid$ gap\% $\mid$ tiempo s $\mid$ n\_reset)")
    det_lineas: list[str] = []
    for inst, filas in detalle:
        bks = _f(filas[0]["bks_referencia"]) if filas else None
        det_lineas.append("[{}]  BKS={}".format(inst, f"{bks:.0f}" if bks is not None else "-"))
        for r in sorted(filas, key=lambda x: int(x["repeticion"])):
            ci = _f(r["costo_solucion_inicial"]); mc = _f(r["mejor_costo"])
            gp = _f(r["gap_bks_porcentaje"]); tt = _f(r["tiempo_segundos"])
            nr = _f(r.get("n_resets_kick", ""))
            det_lineas.append("  rep {:>1} | {:>7} -> {:>7} | {:>7.3f}% | {:>8.2f} | {:>4}".format(
                r["repeticion"],
                f"{ci:.0f}" if ci is not None else "-",
                f"{mc:.0f}" if mc is not None else "-",
                gp if gp is not None else float("nan"),
                tt if tt is not None else float("nan"),
                f"{nr:.0f}" if nr is not None else "-"))
    out.extend(lst(det_lineas))

    # --- Mejor solucion por instancia (rutas + deadheading, verbatim) ---
    out.append(r"\paragraph{Mejor solucion por instancia.} "
               r"Repeticion de menor costo: rutas y desglose de deadheading "
               r"(el total coincide con mejor\_costo bajo el evaluador greedy).")
    for inst, filas in detalle:
        bks = _f(filas[0]["bks_referencia"]) if filas else None
        mejor_fila = min(filas, key=lambda r: (_f(r["mejor_costo"])
                                               if _f(r["mejor_costo"]) is not None else float("inf")))
        mc = _f(mejor_fila["mejor_costo"]); gp = _f(mejor_fila["gap_bks_porcentaje"])
        cuerpo: list[str] = []
        cuerpo.append("==== [{}]  BKS={}  mejor_costo={}  gap={:.3f}%  (rep {}) ====".format(
            inst, f"{bks:.0f}" if bks is not None else "-",
            f"{mc:.0f}" if mc is not None else "-",
            gp if gp is not None else float("nan"),
            mejor_fila.get("repeticion", "?")))
        rutas = (mejor_fila.get("mejor_solucion_tr_legible") or "").strip()
        if rutas:
            cuerpo.append("RUTAS:")
            for linea in rutas.replace(" || ", "\n").splitlines():
                cuerpo.append("  " + linea.strip())
        reporte = (mejor_fila.get("reporte_detalle_deadheading") or "").strip()
        if reporte:
            cuerpo.append("DESGLOSE DE DEADHEADING:")
            for linea in reporte.splitlines():
                cuerpo.append("  " + linea.rstrip())
        out.extend(lst(cuerpo))
    return out


def embeber_tex(token: str) -> list[str]:
    """Inserta el capitulo LaTeX del approach como fuente real (compila)."""
    path = TEX_POR_APPROACH.get(token)
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        contenido = fh.read().rstrip("\n")
    rel = os.path.relpath(path, RAIZ)
    out = [r"\clearpage",
           r"%% ===== Capitulo embebido (verbatim LaTeX): %s =====" % rel,
           contenido,
           r"%% ===== Fin capitulo embebido: %s =====" % rel]
    return out


# ============================================================
# Preambulo
# ============================================================

PREAMBULO = r"""\documentclass[11pt,a4paper]{article}

% ----- Codificacion e idioma -----
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish,es-nodecimaldot]{babel}

% ----- Matematicas y teoremas (necesarios para los capitulos embebidos) -----
\usepackage{amsmath, amssymb, amsthm}
\usepackage{mathtools}
\newtheorem{teorema}{Teorema}[section]
\newtheorem{lema}[teorema]{Lema}
\newtheorem{proposicion}[teorema]{Proposicion}
\theoremstyle{definition}
\newtheorem{definicion}[teorema]{Definicion}
\newtheorem{ejemplo}[teorema]{Ejemplo}
\theoremstyle{remark}
\newtheorem{observacion}[teorema]{Observacion}

% ----- Tablas profesionales -----
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{longtable}
\usepackage{multirow}

% ----- Figuras -----
\usepackage{graphicx}
\graphicspath{{../../figuras/}{docs/figuras/}{figuras/}{./}}

% ----- Pseudocodigo (capitulos embebidos) -----
\usepackage{algorithm}
\usepackage{algpseudocode}
\algrenewcommand\algorithmicrequire{\textbf{Entradas:}}
\algrenewcommand\algorithmicensure{\textbf{Devuelve:}}
\algrenewcommand\algorithmicif{\textbf{si}}
\algrenewcommand\algorithmicthen{\textbf{entonces}}
\algrenewcommand\algorithmicelse{\textbf{si no}}
\algrenewcommand\algorithmicend{\textbf{fin}}
\algrenewcommand\algorithmicwhile{\textbf{mientras}}
\algrenewcommand\algorithmicfor{\textbf{para}}
\algrenewcommand\algorithmicdo{\textbf{hacer}}
\algrenewcommand\algorithmicreturn{\textbf{devolver}}
\algrenewcommand\algorithmicrepeat{\textbf{repetir}}
\algrenewcommand\algorithmicuntil{\textbf{hasta}}

% ----- Color e hipervinculos -----
\usepackage{xcolor}
\definecolor{verdebueno}{RGB}{0,128,0}
\definecolor{rojomalo}{RGB}{200,0,0}
\definecolor{azulneutro}{RGB}{0,60,140}
\usepackage[colorlinks=true, linkcolor=blue!60!black,
            citecolor=blue!60!black, urlcolor=blue!60!black]{hyperref}

% ----- Margenes -----
\usepackage[a4paper,margin=2.2cm]{geometry}
\setlength{\parskip}{4pt}
\setlength{\parindent}{0pt}

% ----- Volcados verbatim (CSV, rutas, deadheading) -----
\usepackage{listings}
\lstdefinestyle{dump}{%
    basicstyle=\ttfamily\scriptsize,
    breaklines=true,
    columns=fullflexible,
    keepspaces=true,
    showstringspaces=false,
    frame=single,
    framesep=2pt,
    xleftmargin=2pt,
    aboveskip=4pt, belowskip=4pt,
}

\title{Proceso experimental y resultados extensos bajo el evaluador de costo
corregido\\[2pt]\large Proyecto MetaCARP --- Capacitated Arc Routing Problem}
\author{Generado automaticamente desde \texttt{experimentos\_costo\_fixed/}}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\clearpage
"""


# ============================================================
# Narrativa (LaTeX)
# ============================================================

def seccion_narrativa() -> list[str]:
    L: list[str] = []
    L.append(r"\section{El cambio raiz: eliminacion del calculador de costo incorrecto}")
    L.append(r"\textbf{Commit pivote:} \texttt{dab620b} --- "
             r"\emph{``refactor: integrar orientacion greedy como unica logica del "
             r"evaluador''} (30-may-2026).")
    L.append(r"\paragraph{Problema.} El evaluador de costo previo forzaba una "
             r"orientacion \emph{canonica} fija: cada tarea (arco) se recorria siempre "
             r"por su extremo $u\to v$, sin importar por donde llegaba el vehiculo. Eso "
             r"inflaba artificialmente el dead-heading (los traslados sin servicio) y, "
             r"por tanto, el costo total y el gap respecto al BKS. Con gaps inflados era "
             r"imposible aislar el efecto real de cada mejora.")
    L.append(r"\paragraph{Correccion.} La orientacion \emph{greedy} pasa a ser la unica "
             r"logica nativa del evaluador (\texttt{metacarp.evaluador\_costo}): cada "
             r"tarea se entra por el extremo mas cercano al nodo previo, minimizando el "
             r"dead-heading. Antes esto se aplicaba como \emph{monkey-patch} en tiempo de "
             r"ejecucion; ahora es el comportamiento base.")
    L.append(r"\begin{itemize}")
    L.append(r"\item \texttt{evaluador\_costo.py}: \texttt{costo\_rapido\_ids} y "
             r"\texttt{\_empaquetar\_lote\_ids} reescritos con orientacion greedy por "
             r"tarea (bucle secuencial prev$\to$orientacion).")
    L.append(r"\item \texttt{reporte\_solucion.py} y \texttt{metaheuristicas\_utils.py}: "
             r"se elimino el parametro \texttt{orientacion\_greedy} y el branch canonico; "
             r"el reporte usa siempre la regla greedy, de modo que \texttt{costo\_total} "
             r"coincide con \texttt{mejor\_costo} del CSV.")
    L.append(r"\item Se \textbf{elimino} el modulo \texttt{evaluador\_greedy\_20260529.py} "
             r"(patch ya redundante) y se limpiaron sus llamadas en los 3 scripts.")
    L.append(r"\end{itemize}")
    L.append(r"\emph{Verificacion:} import limpio, cero vestigios del patch, y "
             r"\texttt{costo\_rapido\_ids} $=$ \texttt{costo\_lote\_ids} sobre la misma "
             r"solucion, eligiendo la orientacion de minimo dead-heading (invierte la "
             r"tarea cuando conviene).")

    L.append(r"\section{Cambio funcional posterior: el hook \texttt{intensificador}}")
    L.append(r"Sobre la base ya corregida se introdujo un unico cambio funcional en las "
             r"5 metaheuristicas (commit \texttt{93cf736}): el parametro opcional "
             r"\texttt{intensificador: Callable | None = None}. Cuando se provee, en el "
             r"punto de estancamiento (mismo disparador que el kick) la metaheuristica "
             r"ejecuta Path Relinking \textbf{hacia} la mejor solucion global "
             r"\textbf{en lugar} del kick aleatorio, con la firma fija:")
    L.append(r"\begin{center}\ttfamily intensificador(sol\_actual, mejor\_global, ctx, "
             r"lam, rng, encoding, md)\end{center}")
    L.append(r"Reemplaza los \emph{frame-hacks} del PR del primer ciclo por una API "
             r"limpia. El nuevo modulo \texttt{metacarp/path\_relinking\_limpio\_20260531.py} "
             r"provee dos \emph{hooks} listos (\texttt{hook\_pr\_labels} para "
             r"SA/TS/RTS/Cuckoo y \texttt{hook\_pr\_ids} para ABC). PR es una funcion "
             r"\emph{pura}: recibe \texttt{ctx}/$\lambda$/mejor-solucion por argumentos "
             r"explicitos, sin robar nada del stack ni parchear nada, y devuelve siempre "
             r"una solucion con objetivo penalizado $\le$ al de partida (PR truncado "
             r"guarda el mejor intermedio). Afecta a las 5 MH (funcion principal y "
             r"\texttt{\_desde\_instancia}). En SA, Cuckoo y ABC la guia del PR es la "
             r"mejor solucion factible si existe; si no, la mejor sin restriccion.")

    L.append(r"\section{Programa experimental: de lo mas simple a lo mas complejo}")
    L.append(r"Filosofia metodologica: con el costo ya corregido se parte de la "
             r"configuracion \textbf{mas simple} y se anaden mecanismos uno a uno, para "
             r"que cada uno se justifique por si mismo. Si el approach simple ya da buen "
             r"gap, los demas (PR, kick, AOS, budget) solo se justificarian por reducir "
             r"tiempo, no por mejorar calidad.")
    L.append(r"\paragraph{Comun a los 3 approaches.}")
    L.append(r"\begin{itemize}")
    L.append(r"\item Operadores: conjunto \textbf{completo} (9 = 3 intra + 6 inter).")
    L.append(r"\item $\lambda$ (penalizador de capacidad): default instance-aware "
             r"($\sim 10\times$ la mediana del costo de arco).")
    L.append(r"\item Parametros instance-aware automaticos (T\_init/T\_min/L en SA, etc.).")
    L.append(r"\item 23 instancias (gdb/kshs) $\times$ 5 repeticiones.")
    L.append(r"\item Una sola configuracion canonica fija por MH (sin grid).")
    L.append(r"\end{itemize}")
    L.append(r"\paragraph{Approach 1 --- \texttt{solo\_p\_inter} (el mas simple).} "
             r"Selector \emph{probabilistico}: con probabilidad $p_{\text{inter}}$ se "
             r"propone el grupo inter-ruta, si no el grupo intra-ruta; dentro del grupo, "
             r"operador \emph{uniforme}. En estado infactible la probabilidad sube a "
             r"$\alpha_{\text{inter}}$ (reparacion agresiva). Sin PR, kick, AOS ni budget.")
    L.append(r"\paragraph{Approach 2 --- \texttt{binario\_capacidad} (determinista).} "
             r"Igual que el 1 salvo el selector: \emph{binario determinista} por "
             r"capacidad. Si la solucion viola capacidad $\to$ grupo inter (reparar); si "
             r"es factible $\to$ grupo intra (refinar). No consume el \texttt{rng} ni mira "
             r"$p_{\text{inter}}$. Sin kick reactivo.")
    L.append(r"\paragraph{Approach 3 --- \texttt{pr\_aislado} (aislar Path Relinking).} "
             r"Mide el efecto de anadir Path Relinking (via el hook "
             r"\texttt{intensificador}) sobre la base desnuda de un selector, sin kick "
             r"aleatorio/AOS/budget. El selector es una \emph{dimension}: se corre con "
             r"base \texttt{p\_inter} y con base \texttt{binario}. El aislamiento se "
             r"obtiene comparando [selector + PR] (este) contra [selector] (approach 1/2).")
    return L


def seccion_calibracion() -> list[str]:
    L: list[str] = []
    L.append(r"\section{Calibracion de parametros (previa a la config canonica)}")
    L.append(r"Antes de fijar la config canonica se calibro, por metaheuristica, el "
             r"segundo parametro mas influyente (2-knob) y los parametros restantes "
             r"($\alpha_{\text{inter}}$, $\lambda$). El criterio fue el gap medio "
             r"respecto al BKS. Resultados versionados:")

    cal2 = cargar_json(os.path.join(BASE, "_calibracion_2knob", "mejor_2knob.json"))
    if cal2:
        L.append(r"\paragraph{Calibracion 2-knob (segundo parametro mas influyente).}")
        L.append(r"\begin{center}\footnotesize\begin{tabular}{l l r l}")
        L.append(r"\toprule MH & parametro & mejor & gap medio (\%) [todos] \\ \midrule")
        for mh in ORDEN_MH:
            if mh in cal2:
                c = cal2[mh]
                todos = ", ".join(f"{k}:{v}" for k, v in c.get("todos", {}).items())
                L.append(r"%s & %s & %s & %s~[%s] \\" % (
                    T(mh), T(c["knob2_nombre"]), T(c["knob2_valor"]),
                    T(c["gap_medio"]), T(todos)))
        L.append(r"\bottomrule\end{tabular}\end{center}")

    calA = cargar_json(os.path.join(BASE, "_calibracion_restantes", "mejor_alpha_inter.json"))
    if calA:
        L.append(r"\paragraph{Calibracion $\alpha_{\text{inter}}$ "
                 r"(prob. inter en infactible).}")
        L.append(r"\begin{center}\footnotesize\begin{tabular}{l r l}")
        L.append(r"\toprule MH & mejor & gap medio (\%) [todos] \\ \midrule")
        for mh in ORDEN_MH:
            if mh in calA:
                c = calA[mh]
                todos = ", ".join(f"{k}:{v}" for k, v in c.get("todos", {}).items())
                L.append(r"%s & %s & %.3f~[%s] \\" % (
                    T(mh), T(c["mejor_valor"]), c["gap_medio"], T(todos)))
        L.append(r"\bottomrule\end{tabular}\end{center}")

    calL = cargar_json(os.path.join(BASE, "_calibracion_restantes", "mejor_lambda.json"))
    if calL:
        todos = ", ".join(f"{k}:{v}" for k, v in calL.get("todos", {}).items())
        L.append(r"\paragraph{Calibracion $\lambda$ (factor del penalizador de capacidad).} "
                 r"\texttt{%s} (ref \texttt{%s}): mejor = %s, gap medio = %.3f\%% [%s]." % (
                     T(calL["param"]), T(calL["mh_ref"]), T(calL["mejor_valor"]),
                     calL["gap_medio"], T(todos)))
    return L


# ============================================================
# Main
# ============================================================

def main() -> None:
    L: list[str] = [PREAMBULO]
    L.extend(seccion_narrativa())
    L.extend(seccion_calibracion())

    approaches = [
        ("solo_p_inter", "Resultados extensos --- Approach 1 (solo\\_p\\_inter)",
         lambda mh: f"{mh}_solo_p_inter"),
        ("binario_capacidad", "Resultados extensos --- Approach 2 (binario\\_capacidad)",
         lambda mh: f"{mh}_binario_capacidad"),
        ("pr_aislado", "Resultados extensos --- Approach 3 (pr\\_aislado)",
         None),
    ]

    for token, titulo_sec, prefijo_fn in approaches:
        L.append(r"\clearpage")
        L.append(r"\section{%s}" % titulo_sec)
        L.append(r"\noindent\emph{Leyenda:} best = mejor costo de las 5 reps; "
                 r"media/peor sobre las 5 reps; gap\_b\% = gap del best vs BKS; "
                 r"gap\_m\% = gap medio; t\_med = tiempo medio por corrida; "
                 r"n\_reset = media de reinicios (kick/PR).")
        dirs = dir_por_mh_approach(token)
        for mh in ORDEN_MH:
            if mh not in dirs:
                continue
            carpeta_final = os.path.join(dirs[mh], "final")
            if token == "pr_aislado":
                for sub, etiqueta in [("pr_binario", "PR + base binario"),
                                      ("pr_p_inter", "PR + base p_inter")]:
                    mapa = archivos_instancia(carpeta_final, f"{mh}_{sub}")
                    titulo = f"{NOMBRE_MH.get(mh, mh)} -- {etiqueta}"
                    L.extend(bloque_variante(titulo, mapa))
            else:
                mapa = archivos_instancia(carpeta_final, prefijo_fn(mh))
                L.extend(bloque_variante(NOMBRE_MH.get(mh, mh), mapa))
        # Capitulo LaTeX redactado de este approach (tablas + pseudocodigos).
        L.extend(embeber_tex(token))

    # Nota final
    L.append(r"\clearpage")
    L.append(r"\section{Nota final}")
    L.append(r"Todos los costos y gaps se calcularon con el evaluador de orientacion "
             r"greedy (costo corregido). Los CSV historicos previos al commit "
             r"\texttt{dab620b} (p.ej. \texttt{resultados\_lambda\_grid\_20260525.csv}) "
             r"se eliminaron del repo por corresponder al regimen de costo incorrecto y "
             r"no son comparables con lo de aqui.")
    L.append(r"\begin{itemize}")
    L.append(r"\item Fuente de datos: "
             r"\texttt{experimentos\_costo\_fixed/<mh>\_<approach>\_<fecha>/final/*.csv}")
    L.append(r"\item Capitulos LaTeX embebidos: "
             r"\texttt{docs/latex/es/experimento\_\{solo\_p\_inter,binario\_capacidad,pr\_aislado\}"
             r"\_costo\_fixed.tex}")
    L.append(r"\item Regenerar: \texttt{python3 scripts/\_gen\_docs\_final\_costo\_correcto.py}")
    L.append(r"\end{itemize}")
    L.append(r"\end{document}")

    with open(SALIDA, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    print(f"Documento LaTeX generado: {SALIDA}")
    print(f"Lineas: {len(chr(10).join(L).splitlines())}")


if __name__ == "__main__":
    main()
