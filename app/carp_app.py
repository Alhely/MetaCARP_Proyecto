# -*- coding: utf-8 -*-
"""
App Streamlit --- MetaCARP (Approach 1: ``solo_p_inter``, el de mejor desempeño).

Tres pestañas:
  1. Resumen: elige instancia + metaheurística, ajusta parámetros y corre;
     muestra el costo/gap, la tabla de rutas, la solución detallada con
     dead-heading y el mapeo de tareas a arcos.
  2. Mejor ruta: visualización interactiva (zoom/hover) de la mejor solución.

Ejecutar con:
    streamlit run app/carp_app.py
desde la raíz del proyecto.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Permitir importar el paquete ``metacarp`` al lanzar desde cualquier carpeta.
RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from app import carp_viz as cv  # noqa: E402

st.set_page_config(page_title="MetaCARP --- solo_p_inter", layout="wide")

st.title("MetaCARP · Approach 1 (`solo_p_inter`)")
st.caption("Ejecuta las 5 metaheurísticas, ajusta sus parámetros y visualiza la "
           "mejor solución, con animación del recorrido de cada vehículo.")

# Estado persistente entre pestañas / reruns.
ss = st.session_state
ss.setdefault("resultado", None)   # dict con la última corrida

# ==========================================================================
# Barra lateral: configuración de la corrida
# ==========================================================================
with st.sidebar:
    st.header("Configuración")
    instancia = st.selectbox("Instancia", cv.INSTANCIAS, index=10)  # gdb1 por defecto
    nombre_mh = st.selectbox("Metaheurística", list(cv.MH_SPECS.keys()))
    semilla = st.number_input("Semilla (reproducibilidad)", min_value=0,
                              max_value=10_000, value=1, step=1)

    st.subheader("Parámetros")
    parametros: dict = {}
    for key, etiqueta, default, lo, hi, step, es_int in cv.MH_SPECS[nombre_mh]["params"]:
        if es_int:
            parametros[key] = st.number_input(etiqueta, min_value=int(lo),
                                              max_value=int(hi), value=int(default),
                                              step=int(step))
        else:
            parametros[key] = st.slider(etiqueta, float(lo), float(hi),
                                        float(default), float(step))

    correr = st.button("▶ Ejecutar", type="primary", use_container_width=True)

# ==========================================================================
# Ejecutar la metaheurística
# ==========================================================================
if correr:
    with st.spinner(f"Ejecutando {nombre_mh} sobre {instancia}…"):
        res = cv.ejecutar_mh(nombre_mh, instancia, parametros, int(semilla))
        G, pos, ctx = cv.cargar_grafo_y_contexto(instancia)
        rutas = cv.reconstruir_rutas(res.mejor_solucion, ctx, G)
        ss.resultado = {
            "instancia": instancia, "mh": nombre_mh, "parametros": parametros,
            "semilla": int(semilla),
            "mejor_costo": float(res.mejor_costo),
            "bks": cv.bks_instancia(instancia),
            "solucion": res.mejor_solucion,
            "rutas": rutas, "pos": pos, "depot": int(ctx.depot),
            "G_edges": list(G.edges()), "G_nodes": list(G.nodes()),
            "reporte": cv.reporte_detallado(instancia, res.mejor_solucion),
            "mapeo": cv.mapeo_tareas_arcos(ctx),
        }
    st.success("Listo. Revisa el detalle abajo y la pestaña «Mejor ruta».")

# ==========================================================================
# Pestañas
# ==========================================================================
tab_run, tab_ruta = st.tabs(["📊 Resumen", "🗺️ Mejor ruta"])

r = ss.resultado


def _reconstruir_grafo(r) -> "cv.nx.Graph":
    """Rearma un grafo ligero solo para dibujar (nodos + aristas guardadas)."""
    G = cv.nx.Graph()
    G.add_nodes_from(r["G_nodes"])
    G.add_edges_from(r["G_edges"])
    return G


# ---------------------- Pestaña 1: Resumen --------------------------------
with tab_run:
    if r is None:
        st.info("Configura la corrida en la barra lateral y pulsa **▶ Ejecutar**.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Instancia", r["instancia"])
        c2.metric("Mejor costo", f"{r['mejor_costo']:.0f}")
        if r["bks"]:
            gap = 100.0 * (r["mejor_costo"] - r["bks"]) / r["bks"]
            c3.metric("BKS", f"{r['bks']:.0f}")
            c4.metric("Gap vs BKS", f"{gap:.2f}%")
        st.markdown(f"**Metaheurística:** {r['mh']}  ·  **Semilla:** {r['semilla']}  "
                    f"·  **Vehículos usados:** {len(r['rutas'])}")
        st.markdown("**Parámetros:** " +
                    ", ".join(f"`{k}={v}`" for k, v in r["parametros"].items()))
        st.subheader("Rutas (resumen)")
        filas = []
        for i, ru in enumerate(r["rutas"]):
            filas.append({
                "Vehículo": i + 1,
                "# tareas": len(ru["tareas"]),
                "Demanda": f"{ru['demanda_total']:.0f} / {ru['capacidad']:.0f}",
                "Factible": "✅" if ru["demanda_total"] <= ru["capacidad"] + 1e-9 else "⚠️",
                "Secuencia": " → ".join(["D"] + ru["tareas"] + ["D"]),
            })
        st.dataframe(filas, use_container_width=True, hide_index=True)

        col_rep, col_map = st.columns([2, 1])
        with col_rep:
            st.subheader("Solución detallada (servicio + dead-heading)")
            st.caption("Cada vehículo: tareas servidas (arco y demanda) y los nodos "
                       "de dead-heading recorridos. El total coincide con el mejor costo.")
            st.code(r["reporte"], language="text")
        with col_map:
            st.subheader("Tareas → arcos")
            st.caption("Cada tarea requerida y el arco (par de nodos) que representa.")
            st.dataframe(r["mapeo"], use_container_width=True, hide_index=True,
                         height=420)

# ---------------------- Pestaña 2: Mejor ruta -----------------------------
with tab_ruta:
    if r is None:
        st.info("Aún no hay resultados. Ejecuta una corrida.")
    else:
        st.subheader(f"Mejor solución — {r['mh']} · {r['instancia']} "
                     f"(costo {r['mejor_costo']:.0f})")
        st.caption("Línea sólida = servicio de tarea · línea punteada = dead-heading "
                   "(traslado). ★ = depósito. Usa el mouse para zoom y hover.")
        G = _reconstruir_grafo(r)
        fig = cv.figura_estatica(G, r["pos"], r["rutas"], r["depot"])
        st.plotly_chart(fig, use_container_width=True)
