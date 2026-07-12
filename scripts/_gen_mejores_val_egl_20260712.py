"""
Genera las tablas de MEJORES VALORES POR INSTANCIA de la campaña val/egl.

Análogo a ``_gen_mejores_por_instancia_20260710.py`` (23 pequeñas) pero para
las 64 instancias de la campaña de transferencia (34 val + 6 gdb + 24 egl):

  - SA / TS / RTS / ABC: ``experimentos_val_egl_20260710/corrida_*/<mh>/final/``
    (una configuración fija por MH y clase, 5 repeticiones).
  - CS: ``experimentos_val_egl_20260710/cs_campana/cs_minigrid/corrida_*/final/``
    (config ganadora del mini-grid: factor_pasos=0.25, p_inter=0.6,
    warm-start Path-Scanning mejor-de-5).

Para cada (MH, instancia) se toma la repetición de menor costo (desempate por
menor tiempo). El "parámetro" por celda queda determinado por (MH, clase);
la leyenda se documenta en el pie de tabla y en config_fija.json de cada MH.

IMPORTANTE — ajuste de escala de costos en la familia val (diagnóstico
2026-07-12): los ``.dat`` originales de val (bccm) son internamente
inconsistentes: su encabezado (``COSTE_TOTAL_REQ``, ``BKS``) está en una
escala de costos de servicio distinta a la de su propia lista de aristas
(p.ej. val1A: encabezado 220 vs aristas 146). Como toda solución factible
sirve todas las aristas exactamente una vez, ambas escalas difieren por la
CONSTANTE por instancia ``delta = COSTE_TOTAL_REQ - suma(costos aristas)``.
Verificación empírica: ``mejor_encontrado + delta`` reproduce el BKS exacto
en 12/34 val y nunca queda por debajo (34/34 coherentes). El evaluador de
costo del proyecto fue validado de forma independiente (DP de orientaciones
sobre el grafo crudo) y es CORRECTO; el ajuste aquí es solo de reporte:

    costo_comparable = mejor_costo + delta      (delta = 0 fuera de val)
    gap_comparable   = (costo_comparable - BKS) / BKS

Las negritas marcan el MEJOR VALOR POR FILA (ganador entre MHs; el ganador
no cambia con el ajuste porque delta es constante por instancia).

Salidas (en ``resultados/mejores_val_egl_20260712/``):
  1. ``mejores_val_egl.csv``            — tabla larga (MH × instancia).
  2. ``tabla_mejores_val_gdb.tex``      — tabla booktabs clase val+gdb (40 filas).
  3. ``tabla_mejores_egl.tex``          — tabla booktabs clase egl (24 filas).

Uso:
    python scripts/_gen_mejores_val_egl_20260712.py
"""
from __future__ import annotations

import csv
import glob
import statistics
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "resultados" / "mejores_val_egl_20260712"

# Única fuente de verdad de las listas: el runner de la campaña.
import sys
sys.path.insert(0, str(RAIZ / "scripts"))
from run_val_egl_20260710 import INSTANCIAS_VAL_GDB, INSTANCIAS_EGL  # noqa: E402

# (nombre en columna 'metaheuristica' del CSV) -> etiqueta de la tabla.
# El glob localiza los CSV finales de cada MH.
FUENTES = [
    ("recocido_simulado",      "SA",
     "experimentos_val_egl_20260710/corrida_*/sa/final/*.csv"),
    ("busqueda_tabu_simple",   "TS",
     "experimentos_val_egl_20260710/corrida_*/ts/final/*.csv"),
    ("tabu_reactiva",          "RTS",
     "experimentos_val_egl_20260710/corrida_*/rts/final/*.csv"),
    ("busqueda_abejas_simple", "ABC",
     "experimentos_val_egl_20260710/corrida_*/abc/final/*.csv"),
    ("cuckoo_search",          "CS",
     "experimentos_val_egl_20260710/cs_campana/cs_minigrid/corrida_*/final/*.csv"),
]


def _deltas_val() -> dict[str, float]:
    """Delta de escala por instancia: COSTE_TOTAL_REQ - suma(costos aristas).

    Solo es distinto de cero en la familia val (ver docstring del módulo).
    Se calcula desde los pickles para no depender de tablas manuales.
    """
    import pickle
    deltas: dict[str, float] = {}
    for ruta in glob.glob(str(RAIZ / "PickleInstances" / "*.pkl")):
        d = pickle.load(open(ruta, "rb"))
        nombre = d.get("NOMBRE") or Path(ruta).stem
        meta = d.get("COSTE_TOTAL_REQ")
        if meta is None:
            deltas[nombre] = 0.0
            continue
        suma = sum(a["costo"] for a in d["LISTA_ARISTAS_REQ"])
        deltas[nombre] = float(meta) - float(suma)
    return deltas


def recolectar() -> dict[tuple[str, str], dict]:
    """{(etiqueta_mh, instancia): mejor fila}. Ignora _partials."""
    deltas = _deltas_val()
    mejores: dict[tuple[str, str], dict] = {}
    for _mh_csv, etiqueta, patron in FUENTES:
        for ruta in glob.glob(str(RAIZ / patron)):
            if "_partials" in ruta:
                continue
            with open(ruta, newline="") as fh:
                for fila in csv.DictReader(fh):
                    inst = fila["instancia"]
                    costo = float(fila["mejor_costo"])
                    t = float(fila["tiempo_segundos"])
                    clave = (etiqueta, inst)
                    actual = mejores.get(clave)
                    if (actual is None or costo < actual["mejor_costo"]
                            or (costo == actual["mejor_costo"]
                                and t < actual["tiempo_segundos"])):
                        # Ajuste de escala val (delta = 0 en gdb/egl): el
                        # costo comparable vive en la misma escala que el
                        # BKS del encabezado del .dat (ver docstring).
                        delta = deltas.get(inst, 0.0)
                        bks = float(fila["bks_referencia"])
                        comparable = costo + delta
                        mejores[clave] = {
                            "metaheuristica": etiqueta,
                            "instancia": inst,
                            "referencia": bks,
                            "mejor_costo": costo,
                            "delta_escala_val": delta,
                            "costo_comparable": comparable,
                            "gap_referencia_porcentaje":
                                (comparable - bks) / bks * 100.0,
                            "tiempo_segundos": t,
                            "repeticion": fila["repeticion"],
                        }
    return mejores


def escribir_csv(mejores: dict) -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    campos = ["metaheuristica", "instancia", "referencia", "mejor_costo",
              "delta_escala_val", "costo_comparable",
              "gap_referencia_porcentaje", "tiempo_segundos", "repeticion"]
    with open(SALIDA / "mejores_val_egl.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        for (_mh, _inst), fila in sorted(mejores.items(),
                                         key=lambda kv: (kv[0][0], kv[0][1])):
            w.writerow(fila)


def _tabla(mejores: dict, instancias: list[str], titulo: str,
           etiqueta: str, destino: Path) -> None:
    """Tabla booktabs: filas=instancias, columnas=MHs; negrita = ganador fila."""
    mhs = [e for _c, e, _p in FUENTES]
    lineas = [
        rf"% Generado por scripts/_gen_mejores_val_egl_20260712.py",
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{titulo} Presupuesto uniforme $10^6$ evaluaciones /"
        r" 300\,s por corrida, 5 repeticiones; se reporta el mejor costo"
        r" en la escala de la referencia BKS (en la familia val se aplica el"
        r" ajuste constante de escala $\delta$ documentado en el texto)."
        r" En \textbf{negritas}, el ganador por instancia; con estrella"
        r" ($\star$), los que igualan la BKS.}",
        rf"\label{{{etiqueta}}}",
        r"\small",
        r"\begin{tabular}{lr" + "r" * len(mhs) + "}",
        r"\toprule",
        "Instancia & Ref. & " + " & ".join(mhs) + r" \\",
        r"\midrule",
    ]
    for inst in instancias:
        costos = {mh: mejores.get((mh, inst)) for mh in mhs}
        presentes = [b["costo_comparable"] for b in costos.values() if b]
        ganador = min(presentes) if presentes else None
        celdas = []
        for mh in mhs:
            b = costos[mh]
            if b is None:
                celdas.append("--")
                continue
            c = b["costo_comparable"]
            estrella = r"^{\star}" if c <= b["referencia"] else ""
            if c == ganador:
                celdas.append(rf"$\mathbf{{{c:.0f}}}{estrella}$")
            else:
                celdas.append(rf"${c:.0f}{estrella}$")
        ref = next((b["referencia"] for b in costos.values() if b), float("nan"))
        lineas.append(rf"\texttt{{{inst}}} & {ref:.0f} & "
                      + " & ".join(celdas) + r" \\")
    # Resumen: gap medio respecto a la referencia y # de victorias por MH.
    lineas.append(r"\midrule")
    gaps, wins = [], []
    for mh in mhs:
        g = [mejores[(mh, i)]["gap_referencia_porcentaje"]
             for i in instancias if (mh, i) in mejores]
        w = sum(1 for i in instancias
                if (mh, i) in mejores and mejores[(mh, i)]["costo_comparable"] ==
                min(mejores[(m2, i)]["costo_comparable"] for m2 in mhs
                    if (m2, i) in mejores))
        gaps.append(f"{statistics.mean(g):.1f}\\%" if g else "--")
        wins.append(str(w))
    lineas += [
        r"Gap medio vs ref. & -- & " + " & ".join(gaps) + r" \\",
        r"Victorias & -- & " + " & ".join(wins) + r" \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    destino.write_text("\n".join(lineas), encoding="utf-8")


def main() -> None:
    mejores = recolectar()
    escribir_csv(mejores)
    _tabla(mejores, INSTANCIAS_VAL_GDB,
           "Mejor costo por instancia (clase val + gdb restantes).",
           "tab:mejores-val-gdb", SALIDA / "tabla_mejores_val_gdb.tex")
    _tabla(mejores, INSTANCIAS_EGL,
           "Mejor costo por instancia (clase egl).",
           "tab:mejores-egl", SALIDA / "tabla_mejores_egl.tex")
    esperadas = len(FUENTES) * (len(INSTANCIAS_VAL_GDB) + len(INSTANCIAS_EGL))
    print(f"Filas: {len(mejores)} (esperadas {esperadas})")
    print(f"Salida: {SALIDA}")


if __name__ == "__main__":
    main()
