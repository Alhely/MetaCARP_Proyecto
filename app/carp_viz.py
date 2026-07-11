# -*- coding: utf-8 -*-
"""
Utilidades para la app Streamlit de MetaCARP.

Encapsula:
  - El catálogo de instancias y la especificación de parámetros por MH
    (Approach 1 ``solo_p_inter``, el de mejor desempeño).
  - La ejecución de las 5 metaheurísticas vía sus envoltorios
    ``*_desde_instancia``.
  - La reconstrucción de cada ruta A NIVEL DE NODO (servicio + dead-heading)
    usando la MISMA orientación greedy del evaluador, para que el dibujo
    coincida con ``mejor_costo``.
  - El cálculo de un layout 2D del grafo y las figuras Plotly (estática y por
    fotograma para la animación).

No escribe CSV (``guardar_csv=False``); todo ocurre en memoria.
"""
from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from metacarp.cargar_grafos import cargar_objeto_gexf
from metacarp.evaluador_costo import construir_contexto_desde_instancia
from metacarp.instances import load_instances
from metacarp.reporte_solucion import reporte_solucion_desde_instancia
from metacarp import (
    recocido_simulado_desde_instancia,
    busqueda_tabu_simple_desde_instancia,
    busqueda_tabu_reactiva_desde_instancia,
    busqueda_abejas_simple_desde_instancia,
    cuckoo_search_desde_instancia,
)

# --------------------------------------------------------------------------
# Catálogo de instancias (las 23 del corpus del estudio)
# --------------------------------------------------------------------------
INSTANCIAS: list[str] = [
    "gdb19", "kshs1", "kshs2", "kshs3", "kshs4", "kshs5", "kshs6",
    "gdb4", "gdb14", "gdb15", "gdb1", "gdb20", "gdb3", "gdb6", "gdb7",
    "gdb12", "gdb10", "gdb2", "gdb5", "gdb13", "gdb16", "gdb17", "gdb21",
]

# --------------------------------------------------------------------------
# Especificación de cada metaheurística (función + parámetros expuestos en UI)
# Defaults = configuración canónica del Approach 1 (solo_p_inter).
# Cada parámetro: (clave_kwarg, etiqueta, default, min, max, step, es_entero)
# --------------------------------------------------------------------------
MH_SPECS: dict[str, dict[str, Any]] = {
    "Recocido Simulado (SA)": {
        "func": recocido_simulado_desde_instancia,
        "params": [
            ("p_inter", "p_inter (sesgo a inter)", 0.5, 0.0, 1.0, 0.05, False),
            ("alpha", "alpha (enfriamiento)", 0.90, 0.50, 0.999, 0.01, False),
            ("alpha_inter", "alpha_inter (infactible)", 0.80, 0.0, 1.0, 0.05, False),
            ("patience", "patience (reheat)", 10, 1, 100, 1, True),
            ("reheat_factor", "reheat_factor", 0.50, 0.0, 1.0, 0.05, False),
            ("max_reheats_sin_mejora", "max_reheats_sin_mejora", 10, 1, 100, 1, True),
        ],
    },
    "Búsqueda Tabú Simple (TS)": {
        "func": busqueda_tabu_simple_desde_instancia,
        "params": [
            ("p_inter", "p_inter (sesgo a inter)", 0.4, 0.0, 1.0, 0.05, False),
            ("tabu_tenure", "tabu_tenure", 25, 1, 100, 1, True),
            ("tam_vecindario", "tam_vecindario", 40, 5, 200, 5, True),
            ("alpha_inter", "alpha_inter (infactible)", 0.80, 0.0, 1.0, 0.05, False),
            ("iteraciones_max", "iteraciones_max", 400, 50, 5000, 50, True),
        ],
    },
    "Búsqueda Tabú Reactiva (RTS)": {
        "func": busqueda_tabu_reactiva_desde_instancia,
        "params": [
            ("p_inter", "p_inter (sesgo a inter)", 0.5, 0.0, 1.0, 0.05, False),
            ("factor_aumento", "factor_aumento (>1)", 1.20, 1.0, 2.0, 0.05, False),
            ("factor_reduccion", "factor_reduccion (<1)", 0.95, 0.5, 1.0, 0.05, False),
            ("alpha_inter", "alpha_inter (infactible)", 0.80, 0.0, 1.0, 0.05, False),
        ],
    },
    "Colonia de Abejas (ABC)": {
        "func": busqueda_abejas_simple_desde_instancia,
        "params": [
            ("p_inter", "p_inter (sesgo a inter)", 0.5, 0.0, 1.0, 0.05, False),
            ("num_fuentes", "num_fuentes", 30, 5, 200, 5, True),
            ("limite_abandono", "limite_abandono", 60, 5, 300, 5, True),
        ],
    },
    "Cuckoo Search (CS)": {
        "func": cuckoo_search_desde_instancia,
        "params": [
            ("p_inter", "p_inter (sesgo a inter)", 0.1, 0.0, 1.0, 0.05, False),
            ("pa_abandono", "pa_abandono", 0.15, 0.0, 1.0, 0.05, False),
            ("beta_levy", "beta_levy", 1.30, 1.0, 2.0, 0.05, False),
        ],
    },
}

# Paleta de colores por ruta (se cicla si hay más rutas).
PALETA = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#17becf", "#e377c2", "#8c564b", "#bcbd22", "#7f7f7f",
]


# --------------------------------------------------------------------------
# Ejecución de la metaheurística
# --------------------------------------------------------------------------
def ejecutar_mh(nombre_mh: str, instancia: str, parametros: dict[str, Any],
                semilla: int | None) -> Any:
    """Corre la MH seleccionada y devuelve su objeto Result."""
    func = MH_SPECS[nombre_mh]["func"]
    kwargs = dict(parametros)
    kwargs.update(
        semilla=semilla,
        guardar_csv=False,          # nada de archivos: todo en memoria
        lambda_capacidad=None,      # lambda default instance-aware
    )
    return func(instancia, **kwargs)


# --------------------------------------------------------------------------
# Grafo + layout 2D
# --------------------------------------------------------------------------
def bks_instancia(instancia: str) -> float | None:
    """Mejor valor conocido (BKS) de la instancia, desde su pickle."""
    try:
        return float(load_instances(instancia)["BKS"])
    except Exception:
        return None


def cargar_grafo_y_contexto(instancia: str):
    """Devuelve (G_int, posiciones, ctx) para una instancia."""
    G = cargar_objeto_gexf(instancia)
    G = nx.relabel_nodes(G, {n: int(n) for n in G.nodes})
    ctx = construir_contexto_desde_instancia(instancia)
    pos = _layout(G)
    return G, pos, ctx


def _layout(G: nx.Graph) -> dict[int, tuple[float, float]]:
    """Layout 2D ponderado por costo (kamada-kawai si hay scipy; si no, spring)."""
    try:
        pos = nx.kamada_kawai_layout(G, weight="cost")
    except Exception:
        pos = nx.spring_layout(G, weight="cost", seed=42, iterations=200)
    return {int(n): (float(p[0]), float(p[1])) for n, p in pos.items()}


# --------------------------------------------------------------------------
# Reconstrucción de rutas a nivel de nodo (orientación greedy del evaluador)
# --------------------------------------------------------------------------
def reconstruir_rutas(solucion: list[list[str]], ctx: Any, G: nx.Graph) -> list[dict]:
    """Convierte la solución (etiquetas) en rutas con su recorrido de nodos.

    Para cada tarea aplica la regla greedy (entrar por el extremo más cercano
    al nodo previo) y rellena el dead-heading con el camino mínimo real sobre el
    grafo. Devuelve, por ruta: nodos en orden, segmentos (con tipo
    servicio/deadhead), demanda total y lista de tareas.
    """
    dist = ctx.dist
    u_arr, v_arr = ctx.u_arr, ctx.v_arr
    dem = ctx.demanda_arr
    depot = int(ctx.depot)
    md = ctx.marcador_depot.upper()
    l2id = ctx.encoding.label_to_id

    rutas: list[dict] = []
    for ruta in solucion:
        tareas = [t for t in ruta if str(t).upper() != md]
        if not tareas:
            continue
        prev = depot
        nodos: list[int] = [depot]
        segmentos: list[tuple[int, int, str]] = []
        demanda_total = 0.0
        for etq in tareas:
            tid = l2id[etq]
            u, v = int(u_arr[tid]), int(v_arr[tid])
            demanda_total += float(dem[tid])
            # Orientación greedy: entrar por el extremo más cercano a 'prev'.
            entrada, salida = (u, v) if dist[prev, u] <= dist[prev, v] else (v, u)
            # Dead-heading: camino mínimo real prev -> entrada.
            if prev != entrada:
                cam = nx.shortest_path(G, prev, entrada, weight="cost")
                for a, b in zip(cam[:-1], cam[1:]):
                    segmentos.append((a, b, "deadhead"))
                    nodos.append(b)
            # Servicio de la tarea (arco entrada -> salida).
            segmentos.append((entrada, salida, "servicio"))
            nodos.append(salida)
            prev = salida
        # Regreso al depósito.
        if prev != depot:
            cam = nx.shortest_path(G, prev, depot, weight="cost")
            for a, b in zip(cam[:-1], cam[1:]):
                segmentos.append((a, b, "deadhead"))
                nodos.append(b)
        rutas.append({
            "tareas": tareas,
            "nodos": nodos,
            "segmentos": segmentos,
            "demanda_total": demanda_total,
            "capacidad": float(ctx.capacidad_max),
        })
    return rutas


# --------------------------------------------------------------------------
# Figuras Plotly
# --------------------------------------------------------------------------
def _trazas_grafo_base(G: nx.Graph, pos: dict) -> list[go.Scatter]:
    """Aristas tenues + nodos del grafo, como fondo."""
    ex, ey = [], []
    for a, b in G.edges():
        ex += [pos[a][0], pos[b][0], None]
        ey += [pos[a][1], pos[b][1], None]
    fondo = go.Scatter(x=ex, y=ey, mode="lines",
                       line=dict(color="rgba(180,180,180,0.4)", width=1),
                       hoverinfo="skip", showlegend=False)
    nx_, ny_, txt = [], [], []
    for n in G.nodes():
        nx_.append(pos[n][0]); ny_.append(pos[n][1]); txt.append(str(n))
    nodos = go.Scatter(x=nx_, y=ny_, mode="markers+text", text=txt,
                       textposition="top center",
                       textfont=dict(size=9, color="#555"),
                       marker=dict(size=8, color="#cfd8dc",
                                   line=dict(color="#90a4ae", width=1)),
                       hoverinfo="text", showlegend=False)
    return [fondo, nodos]


def _depot_traza(pos: dict, depot: int) -> go.Scatter:
    return go.Scatter(x=[pos[depot][0]], y=[pos[depot][1]], mode="markers+text",
                      text=["D"], textposition="bottom center",
                      textfont=dict(size=13, color="black"),
                      marker=dict(symbol="star", size=20, color="black"),
                      name="Depósito", hoverinfo="text")


def figura_estatica(G, pos, rutas, depot) -> go.Figure:
    """Mapa completo: cada ruta en un color; servicio sólido, dead-heading punteado."""
    fig = go.Figure(_trazas_grafo_base(G, pos))
    for i, ruta in enumerate(rutas):
        color = PALETA[i % len(PALETA)]
        # Servicio (sólido grueso) y dead-heading (punteado) por separado.
        for kind, dash, width in [("servicio", "solid", 4), ("deadhead", "dot", 2)]:
            xs, ys = [], []
            for a, b, k in ruta["segmentos"]:
                if k != kind:
                    continue
                xs += [pos[a][0], pos[b][0], None]
                ys += [pos[a][1], pos[b][1], None]
            if xs:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(color=color, width=width, dash=dash),
                    name=f"Vehículo {i+1}" + ("" if kind == "servicio" else " (DH)"),
                    legendgroup=f"v{i}",
                    showlegend=(kind == "servicio"),
                    hoverinfo="skip"))
    fig.add_trace(_depot_traza(pos, depot))
    _estilo(fig)
    return fig


def reporte_detallado(instancia: str, solucion: list[list[str]]) -> str:
    """Reporte textual de la solución (servicio + dead-heading por vehículo).

    Usa la rutina oficial del proyecto (orientación greedy), por lo que el costo
    total del reporte coincide con ``mejor_costo``.
    """
    return reporte_solucion_desde_instancia(instancia, solucion).texto


def mapeo_tareas_arcos(ctx: Any) -> list[dict]:
    """Mapeo de cada tarea (TR#) a su arco (u--v) y su demanda."""
    l2id = ctx.encoding.label_to_id
    filas = []
    for etq, tid in sorted(l2id.items(), key=lambda kv: kv[1]):
        u, v = int(ctx.u_arr[tid]), int(ctx.v_arr[tid])
        filas.append({"Tarea": etq, "Arco (u–v)": f"{u} – {v}",
                      "Demanda": float(ctx.demanda_arr[tid])})
    return filas


def _estilo(fig: go.Figure) -> None:
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        plot_bgcolor="white", height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
