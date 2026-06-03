#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador del documento ``docs_final_costo_correcto_con_resultados.txt``.

Recorre TODOS los experimentos lanzados DESPUES de eliminar el calculador de
costo incorrecto (commit dab620b, "orientacion greedy como unica logica del
evaluador") y produce un unico .txt que:

  1. Narra el proceso experimental completo (correccion del costo, hook
     intensificador, calibracion de parametros, los 3 approaches).
  2. Vuelca los RESULTADOS EXTENSOS de cada corrida leidos directamente de los
     CSV en ``experimentos_costo_fixed/`` (per-instancia, per-repeticion y
     resumenes agregados).

No depende de pandas: usa unicamente la biblioteca estandar (csv, glob, ...).
Es idempotente: re-ejecutarlo regenera el .txt desde los CSV actuales.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import statistics
from collections import defaultdict

# Raiz del proyecto = un nivel arriba de scripts/
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(RAIZ, "experimentos_costo_fixed")
SALIDA = os.path.join(RAIZ, "docs_final_costo_correcto_con_resultados.txt")

# Fuente LaTeX (capitulo de tesis) ya redactada para cada approach. Se embebe
# VERBATIM en el documento para no perder las tablas extensas, los pseudocodigos
# (algorithm) y las conclusiones ya escritas.
TEX_POR_APPROACH = {
    "solo_p_inter": os.path.join(RAIZ, "docs", "experimento_solo_p_inter_costo_fixed.tex"),
    "binario_capacidad": os.path.join(RAIZ, "docs", "experimento_binario_capacidad_costo_fixed.tex"),
    "pr_aislado": os.path.join(RAIZ, "docs", "experimento_pr_aislado_costo_fixed.tex"),
}

# Orden canonico de instancias (mismo que config_fija.json).
ORDEN_INST = [
    "gdb19", "kshs1", "kshs2", "kshs3", "kshs4", "kshs5", "kshs6",
    "gdb4", "gdb14", "gdb15", "gdb1", "gdb20", "gdb3", "gdb6", "gdb7",
    "gdb12", "gdb10", "gdb2", "gdb5", "gdb13", "gdb16", "gdb17", "gdb21",
]

# Nombre legible de cada metaheuristica.
NOMBRE_MH = {
    "sa": "Recocido Simulado (SA)",
    "tabu_simple": "Busqueda Tabu Simple (TS)",
    "tabu_reactiva": "Busqueda Tabu Reactiva (RTS)",
    "abc_simple": "Colonia Artificial de Abejas (ABC)",
    "cuckoo": "Cuckoo Search (CS)",
}

# Orden de presentacion de las MH.
ORDEN_MH = ["sa", "tabu_simple", "tabu_reactiva", "abc_simple", "cuckoo"]


def _orden_inst_key(inst: str) -> int:
    """Indice de una instancia en el orden canonico (al final si no esta)."""
    return ORDEN_INST.index(inst) if inst in ORDEN_INST else len(ORDEN_INST)


def leer_csv_instancia(path: str) -> list[dict]:
    """Lee un CSV de resultados (maneja campos multilinea entre comillas)."""
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(x: str) -> float | None:
    """Convierte a float tolerando vacios/None."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def archivos_instancia(carpeta_final: str, prefijo: str) -> dict[str, str]:
    """Mapea instancia -> ruta de su CSV para un prefijo dado.

    El prefijo identifica la variante (p.ej. ``sa_solo_p_inter`` o
    ``sa_pr_binario``). Solo se toman ficheros con sufijo de instancia
    (``_gdbN`` / ``_kshsN``); se descartan los agregados sin instancia.
    """
    res: dict[str, str] = {}
    for path in glob.glob(os.path.join(carpeta_final, prefijo + "_*.csv")):
        base = os.path.basename(path)[:-4]  # sin .csv
        inst = base[len(prefijo) + 1:]      # tras "prefijo_"
        # Solo nombres de instancia validos (gdbN / kshsN).
        if inst.startswith("gdb") or inst.startswith("kshs"):
            res[inst] = path
    return res


def bloque_variante(titulo: str, mapa_inst: dict[str, str]) -> list[str]:
    """Construye el bloque de texto extenso para una variante (MH x approach).

    Incluye: tabla resumen por instancia + detalle por repeticion + resumen
    global de la variante.
    """
    out: list[str] = []
    out.append("-" * 92)
    out.append(titulo)
    out.append("-" * 92)

    instancias = sorted(mapa_inst, key=_orden_inst_key)
    if not instancias:
        out.append("  (sin datos)")
        out.append("")
        return out

    # --- Procedencia: archivos CSV que alimentan este bloque ---
    # Carpeta comun (relativa a la raiz del proyecto) y los nombres de fichero
    # por instancia, para que cada cifra sea trazable a su CSV de origen.
    carpeta_rel = os.path.relpath(os.path.dirname(next(iter(mapa_inst.values()))), RAIZ)
    out.append("")
    out.append("  ARCHIVOS CSV DE ORIGEN  (carpeta: {}/)".format(carpeta_rel))
    for inst in instancias:
        out.append("    {:<8} -> {}".format(inst, os.path.basename(mapa_inst[inst])))

    # --- Tabla resumen por instancia ---
    out.append("")
    out.append("  RESUMEN POR INSTANCIA")
    out.append("  {:<8} {:>7} {:>9} {:>9} {:>9} {:>8} {:>8} {:>8} {:>9}".format(
        "inst", "BKS", "best", "media", "peor", "gap_b%", "gap_m%", "t_med(s)", "n_reset"))
    out.append("  " + "." * 86)

    gaps_best_var: list[float] = []
    gaps_media_var: list[float] = []
    n_opt = 0  # cuantas instancias alcanzan gap 0 en su mejor repeticion

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
        best = min(costos)
        media = statistics.mean(costos)
        peor = max(costos)
        gap_b = min(gaps) if gaps else float("nan")
        gap_m = statistics.mean(gaps) if gaps else float("nan")
        t_med = statistics.mean(tiempos) if tiempos else float("nan")
        n_reset_med = statistics.mean(resets) if resets else 0.0

        gaps_best_var.append(gap_b)
        gaps_media_var.append(gap_m)
        if gap_b <= 1e-9:
            n_opt += 1

        out.append("  {:<8} {:>7} {:>9} {:>9} {:>9} {:>8.3f} {:>8.3f} {:>8.2f} {:>9.1f}".format(
            inst,
            f"{bks:.0f}" if bks is not None else "-",
            f"{best:.0f}", f"{media:.1f}", f"{peor:.0f}",
            gap_b, gap_m, t_med, n_reset_med))

    # --- Resumen global de la variante ---
    out.append("  " + "." * 86)
    if gaps_best_var:
        out.append("  GLOBAL  gap_best medio = {:.3f}%   gap_media medio = {:.3f}%   "
                   "optimos(best) = {}/{}".format(
                       statistics.mean(gaps_best_var),
                       statistics.mean(gaps_media_var),
                       n_opt, len(instancias)))

    # --- Detalle por repeticion (extenso) ---
    out.append("")
    out.append("  DETALLE POR REPETICION  (rep | costo_ini -> mejor_costo | gap% | tiempo s | n_reset)")
    for inst, filas in detalle:
        bks = _f(filas[0]["bks_referencia"]) if filas else None
        out.append("    [{}]  BKS={}".format(inst, f"{bks:.0f}" if bks is not None else "-"))
        for r in sorted(filas, key=lambda x: int(x["repeticion"])):
            ci = _f(r["costo_solucion_inicial"])
            mc = _f(r["mejor_costo"])
            gp = _f(r["gap_bks_porcentaje"])
            tt = _f(r["tiempo_segundos"])
            nr = _f(r.get("n_resets_kick", ""))
            out.append("      rep {:>1} | {:>7} -> {:>7} | {:>7.3f}% | {:>8.2f} | {:>4}".format(
                r["repeticion"],
                f"{ci:.0f}" if ci is not None else "-",
                f"{mc:.0f}" if mc is not None else "-",
                gp if gp is not None else float("nan"),
                tt if tt is not None else float("nan"),
                f"{nr:.0f}" if nr is not None else "-"))
    out.append("")

    # --- Mejor solucion por instancia (rutas + desglose de deadheading) ---
    # Para cada instancia se toma la repeticion de MENOR mejor_costo y se vuelca
    # su solucion legible (cadena de rutas) y el reporte detallado de
    # deadheading, cuyo total coincide con mejor_costo bajo el evaluador greedy.
    out.append("  MEJOR SOLUCION POR INSTANCIA  (rep de menor costo: rutas + desglose de deadheading)")
    for inst, filas in detalle:
        bks = _f(filas[0]["bks_referencia"]) if filas else None
        # Repeticion con el menor mejor_costo (desempate: la primera).
        mejor_fila = min(
            filas,
            key=lambda r: (_f(r["mejor_costo"]) if _f(r["mejor_costo"]) is not None else float("inf")),
        )
        mc = _f(mejor_fila["mejor_costo"])
        gp = _f(mejor_fila["gap_bks_porcentaje"])
        out.append("")
        out.append("    ==== [{}]  BKS={}  mejor_costo={}  gap={:.3f}%  (rep {}) ====".format(
            inst,
            f"{bks:.0f}" if bks is not None else "-",
            f"{mc:.0f}" if mc is not None else "-",
            gp if gp is not None else float("nan"),
            mejor_fila.get("repeticion", "?")))
        # Cadena compacta de rutas (R1: D -> TRx -> ... -> D || R2: ...).
        rutas = (mejor_fila.get("mejor_solucion_tr_legible") or "").strip()
        if rutas:
            out.append("    RUTAS:")
            for linea in rutas.replace(" || ", "\n").splitlines():
                out.append("      " + linea.strip())
        # Reporte detallado de deadheading (multilinea; total == mejor_costo).
        reporte = (mejor_fila.get("reporte_detalle_deadheading") or "").strip()
        if reporte:
            out.append("    DESGLOSE DE DEADHEADING:")
            for linea in reporte.splitlines():
                out.append("      " + linea.rstrip())
    out.append("")
    return out


def cargar_json(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def dir_por_mh_approach(approach_token: str) -> dict[str, str]:
    """Mapea mh -> carpeta del experimento para un token de approach.

    ``approach_token`` es el fragmento que aparece en el nombre de carpeta:
    ``solo_p_inter``, ``binario_capacidad`` o ``pr_aislado``.
    """
    res: dict[str, str] = {}
    for d in sorted(glob.glob(os.path.join(BASE, "*_" + approach_token + "_*"))):
        nombre = os.path.basename(d)
        # mh = lo que precede a "_<approach_token>_"
        mh = nombre.split("_" + approach_token + "_")[0]
        res[mh] = d
    return res


def embeber_tex(token: str) -> list[str]:
    """Devuelve el contenido LaTeX del capitulo del approach, embebido verbatim.

    Incluye TODO el .tex (motivacion, diseno, pseudocodigos en entorno
    ``algorithm``, tablas ``table`` extensas, resultados y conclusiones), entre
    marcadores claros para que sea facil localizarlo y copiarlo a la tesis.
    """
    path = TEX_POR_APPROACH.get(token)
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        contenido = fh.read().rstrip("\n")
    rel = os.path.relpath(path, RAIZ)
    out: list[str] = []
    out.append("")
    out.append("." * 92)
    out.append("  FUENTE LaTeX DEL CAPITULO (tablas extensas + pseudocodigos), VERBATIM")
    out.append("  archivo: {}".format(rel))
    out.append("  >>> INICIO LaTeX >>>")
    out.append("." * 92)
    out.extend(contenido.splitlines())
    out.append("." * 92)
    out.append("  <<< FIN LaTeX <<<  ({})".format(rel))
    out.append("." * 92)
    out.append("")
    return out


def main() -> None:
    L: list[str] = []

    def add(*lineas: str) -> None:
        L.extend(lineas)

    # ================================================================
    # PORTADA Y NARRATIVA
    # ================================================================
    add(
        "=" * 92,
        "  PROCESO EXPERIMENTAL Y RESULTADOS EXTENSOS BAJO EL EVALUADOR DE COSTO CORREGIDO",
        "  Proyecto MetaCARP  --  Capacitated Arc Routing Problem",
        "=" * 92,
        "",
        "Este documento reune, en un solo lugar, (a) el proceso experimental completo",
        "ejecutado DESPUES de eliminar el calculador de costo incorrecto y (b) los",
        "resultados extensos -- corrida por corrida -- de cada experimento posterior a",
        "ese cambio. Se genera automaticamente desde los CSV en experimentos_costo_fixed/",
        "con scripts/_gen_docs_final_costo_correcto.py.",
        "",
        "Para maxima trazabilidad y detalle, cada bloque de resultados lista los NOMBRES",
        "EXACTOS de los CSV que lo alimentan, y al final de cada approach se embebe",
        "VERBATIM el capitulo LaTeX ya redactado (tablas extensas, pseudocodigos en",
        "entorno algorithm, y conclusiones) listo para copiar a la tesis.",
        "",
        "INDICE:",
        "  1. El cambio raiz (eliminacion del costo incorrecto)",
        "  2. El hook 'intensificador' (Path Relinking limpio)",
        "  3. El programa experimental (3 approaches)",
        "  4. Calibracion de parametros",
        "  5. Resultados + LaTeX -- Approach 1 (solo_p_inter)",
        "  6. Resultados + LaTeX -- Approach 2 (binario_capacidad)",
        "  7. Resultados + LaTeX -- Approach 3 (pr_aislado)",
        "  8. Nota final",
        "",
        "-" * 92,
        "1. EL CAMBIO RAIZ: ELIMINACION DEL CALCULADOR DE COSTO INCORRECTO",
        "-" * 92,
        "",
        "Commit pivote: dab620b -- 'refactor: integrar orientacion greedy como unica",
        "logica del evaluador' (30-may-2026).",
        "",
        "PROBLEMA. El evaluador de costo previo forzaba una orientacion CANONICA fija:",
        "cada tarea (arco) se recorria siempre por su extremo u->v, sin importar por",
        "donde llegaba el vehiculo. Eso inflaba artificialmente el dead-heading (los",
        "traslados sin servicio) y, por tanto, el costo total y el gap respecto al BKS.",
        "Con gaps inflados era imposible aislar el efecto real de cada mejora.",
        "",
        "CORRECCION. La orientacion GREEDY pasa a ser la unica logica NATIVA del",
        "evaluador (metacarp.evaluador_costo): cada tarea se entra por el extremo mas",
        "cercano al nodo previo, minimizando el dead-heading. Antes esto se aplicaba",
        "como monkey-patch en tiempo de ejecucion; ahora es el comportamiento base.",
        "",
        "Cambios concretos de ese commit:",
        "  - evaluador_costo.py: costo_rapido_ids y _empaquetar_lote_ids reescritos con",
        "    orientacion greedy por tarea (bucle secuencial prev->orientacion).",
        "  - reporte_solucion.py y metaheuristicas_utils.py: se elimino el parametro",
        "    orientacion_greedy y el branch canonico; el reporte usa siempre la regla",
        "    greedy, de modo que costo_total coincide con mejor_costo del CSV.",
        "  - Se ELIMINO el modulo evaluador_greedy_20260529.py (patch ya redundante).",
        "  - Se limpiaron las llamadas al patch en los 3 scripts de corrida.",
        "",
        "Verificacion: import limpio, cero vestigios del patch, y costo_rapido_ids ==",
        "costo_lote_ids sobre la misma solucion, eligiendo la orientacion de minimo",
        "dead-heading (invierte la tarea cuando conviene).",
        "",
        "-" * 92,
        "2. CAMBIO FUNCIONAL POSTERIOR: EL HOOK 'intensificador'",
        "-" * 92,
        "",
        "Sobre la base ya corregida se introdujo un unico cambio funcional en las 5",
        "metaheuristicas (commit 93cf736): el parametro opcional",
        "",
        "    intensificador: Callable | None = None",
        "",
        "Cuando se provee, en el punto de estancamiento (mismo disparador que el kick)",
        "la MH ejecuta Path Relinking HACIA la mejor solucion global EN LUGAR del kick",
        "aleatorio, con la firma fija:",
        "",
        "    intensificador(sol_actual, mejor_global, ctx, lam, rng, encoding, md)",
        "",
        "Reemplaza los frame-hacks del PR del primer ciclo por una API limpia. El nuevo",
        "modulo metacarp/path_relinking_limpio_20260531.py provee dos hooks listos",
        "(hook_pr_labels para SA/TS/RTS/Cuckoo y hook_pr_ids para ABC). PR es una",
        "funcion PURA: recibe ctx/lambda/mejor-solucion por argumentos explicitos, sin",
        "robar nada del stack ni parchear nada. Devuelve siempre una solucion con",
        "objetivo penalizado <= al de partida (PR truncado guarda el mejor intermedio).",
        "",
        "Las 5 MH afectadas (funcion principal + _desde_instancia, en ambas se propaga",
        "el parametro): recocido_simulado, busqueda_tabu_simple, busqueda_tabu_reactiva,",
        "cuckoo_search y busqueda_abejas_simple. En SA, Cuckoo y ABC la guia del PR es",
        "la mejor solucion factible si existe; si no, la mejor sin restriccion.",
        "",
        "-" * 92,
        "3. PROGRAMA EXPERIMENTAL: DE LO MAS SIMPLE A LO MAS COMPLEJO",
        "-" * 92,
        "",
        "Filosofia metodologica: con el costo ya corregido se parte de la configuracion",
        "MAS SIMPLE y se anaden mecanismos uno a uno, para que cada uno se justifique",
        "por si mismo. Si el approach simple ya da buen gap, los demas (PR, kick, AOS,",
        "budget) solo se justificarian por reducir tiempo, no por mejorar calidad.",
        "",
        "Comun a los 3 approaches:",
        "  - Operadores: conjunto COMPLETO (9 = 3 intra + 6 inter).",
        "  - lambda (penalizador de capacidad): default instance-aware (~10x la mediana",
        "    del costo de arco).",
        "  - Parametros instance-aware automaticos (T_init/T_min/L en SA, etc.).",
        "  - 23 instancias (gdb/kshs) x 5 repeticiones.",
        "  - Una sola configuracion canonica fija por MH (sin grid).",
        "",
        "APPROACH 1 -- solo_p_inter (el mas simple).",
        "  Selector PROBABILISTICO: con probabilidad p_inter se propone el grupo INTER,",
        "  si no el grupo INTRA; dentro del grupo, operador UNIFORME. En estado",
        "  infactible la probabilidad sube a alpha_inter (reparacion). Sin PR, sin kick,",
        "  sin AOS, sin budget. Es la MH base con kwargs especificos.",
        "",
        "APPROACH 2 -- binario_capacidad (selector determinista).",
        "  Igual que el 1 salvo el selector: BINARIO DETERMINISTA por capacidad. Si la",
        "  solucion viola capacidad -> grupo INTER (reparacion); si es factible -> grupo",
        "  INTRA (refinamiento). No consume rng ni mira p_inter. Sin kick reactivo.",
        "",
        "APPROACH 3 -- pr_aislado (aislar Path Relinking).",
        "  Mide el efecto de anadir Path Relinking (via el hook intensificador) sobre la",
        "  base desnuda de un selector, SIN kick aleatorio/AOS/budget. El selector es una",
        "  DIMENSION: se corre con base p_inter y con base binario. El aislamiento se",
        "  obtiene comparando [selector + PR] (este) contra [selector] (approach 1 / 2).",
        "",
    )

    # ================================================================
    # CALIBRACION DE PARAMETROS
    # ================================================================
    add(
        "-" * 92,
        "4. CALIBRACION DE PARAMETROS (previa a la config canonica)",
        "-" * 92,
        "",
        "Antes de fijar la config canonica se calibro, por metaheuristica, el segundo",
        "parametro mas influyente (2-knob) y los parametros restantes (alpha_inter,",
        "lambda). El criterio fue el gap medio respecto al BKS. Resultados versionados:",
        "",
    )

    cal2 = cargar_json(os.path.join(BASE, "_calibracion_2knob", "mejor_2knob.json"))
    if cal2:
        add("  CALIBRACION 2-KNOB (mejor_2knob.json) -- 2o parametro mas influyente por MH:")
        for mh in ORDEN_MH:
            if mh in cal2:
                c = cal2[mh]
                todos = "  ".join(f"{k}:{v}" for k, v in c.get("todos", {}).items())
                add("    {:<14} {} = {}  (gap_medio={}%)   [{}]".format(
                    mh, c["knob2_nombre"], c["knob2_valor"], c["gap_medio"], todos))
        add("")

    calA = cargar_json(os.path.join(BASE, "_calibracion_restantes", "mejor_alpha_inter.json"))
    if calA:
        add("  CALIBRACION alpha_inter (mejor_alpha_inter.json) -- prob. INTER en infactible:")
        for mh in ORDEN_MH:
            if mh in calA:
                c = calA[mh]
                todos = "  ".join(f"{k}:{v}" for k, v in c.get("todos", {}).items())
                add("    {:<14} mejor = {}  (gap_medio={:.3f}%)   [{}]".format(
                    mh, c["mejor_valor"], c["gap_medio"], todos))
        add("")

    calL = cargar_json(os.path.join(BASE, "_calibracion_restantes", "mejor_lambda.json"))
    if calL:
        todos = "  ".join(f"{k}:{v}" for k, v in calL.get("todos", {}).items())
        add("  CALIBRACION lambda (mejor_lambda.json) -- factor del penalizador de capacidad:",
            "    {} (ref {}): mejor = {}  (gap_medio={:.3f}%)   [{}]".format(
                calL["param"], calL["mh_ref"], calL["mejor_valor"], calL["gap_medio"], todos),
            "")

    # ================================================================
    # RESULTADOS EXTENSOS POR APPROACH
    # ================================================================
    approaches = [
        ("solo_p_inter", "APPROACH 1 -- solo_p_inter",
         lambda mh: f"{mh}_solo_p_inter"),
        ("binario_capacidad", "APPROACH 2 -- binario_capacidad",
         lambda mh: f"{mh}_binario_capacidad"),
        ("pr_aislado", "APPROACH 3 -- pr_aislado (PR sobre base p_inter y base binario)",
         None),  # caso especial: dos prefijos
    ]

    seccion = 5
    for token, titulo_sec, prefijo_fn in approaches:
        add(
            "",
            "=" * 92,
            f"  {seccion}. RESULTADOS EXTENSOS -- {titulo_sec}",
            "=" * 92,
            "",
            "Leyenda: best=mejor costo de las 5 reps; media/peor sobre las 5 reps;",
            "gap_b%=gap del best vs BKS; gap_m%=gap medio; t_med=tiempo medio por corrida;",
            "n_reset=media de reinicios (kick/PR) por corrida.",
            "",
        )
        dirs = dir_por_mh_approach(token)
        for mh in ORDEN_MH:
            if mh not in dirs:
                continue
            carpeta_final = os.path.join(dirs[mh], "final")
            if token == "pr_aislado":
                # Dos variantes: PR sobre base binario y PR sobre base p_inter.
                for sub, etiqueta in [("pr_binario", "PR + base binario"),
                                      ("pr_p_inter", "PR + base p_inter")]:
                    prefijo = f"{mh}_{sub}"
                    mapa = archivos_instancia(carpeta_final, prefijo)
                    titulo = f"{NOMBRE_MH.get(mh, mh)}  >>  {etiqueta}"
                    add(*bloque_variante(titulo, mapa))
            else:
                prefijo = prefijo_fn(mh)
                mapa = archivos_instancia(carpeta_final, prefijo)
                titulo = f"{NOMBRE_MH.get(mh, mh)}"
                add(*bloque_variante(titulo, mapa))
        # Tras los resultados numericos, embebemos el capitulo LaTeX completo de
        # este approach (tablas extensas, pseudocodigos y conclusiones ya listas).
        add(*embeber_tex(token))
        seccion += 1

    # ================================================================
    # CIERRE
    # ================================================================
    add(
        "",
        "=" * 92,
        f"  {seccion}. NOTA FINAL",
        "=" * 92,
        "",
        "Todos los costos y gaps de este documento se calcularon con el evaluador de",
        "orientacion greedy (costo corregido). Los CSV historicos previos al commit",
        "dab620b (p.ej. resultados_lambda_grid_20260525.csv) se eliminaron del repo por",
        "corresponder al regimen de costo incorrecto y no son comparables con lo de aqui.",
        "",
        "Fuente de datos: experimentos_costo_fixed/<mh>_<approach>_<fecha>/final/*.csv",
        "Capitulos LaTeX embebidos: docs/experimento_{solo_p_inter,binario_capacidad,",
        "pr_aislado}_costo_fixed.tex",
        "Regenerar: python3 scripts/_gen_docs_final_costo_correcto.py",
        "",
    )

    with open(SALIDA, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    print(f"Documento generado: {SALIDA}")
    print(f"Lineas: {len(L)}")


if __name__ == "__main__":
    main()
