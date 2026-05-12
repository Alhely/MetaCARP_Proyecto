from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence

import networkx as nx

from .grafo_ruta import nodo_grafo, path_edges_and_cost, shortest_path_nodes
from .solucion_formato import construir_mapa_tareas_por_etiqueta, normalizar_rutas_etiquetas

__all__ = [
    "ReporteSolucionResult",
    "reporte_solucion",
    "reporte_solucion_desde_instancia",
]


@dataclass
class ReporteSolucionResult:
    texto: str
    costo_total: float
    costos_por_vehiculo: list[float]
    demandas_por_vehiculo: list[float]


def reporte_solucion(
    solucion: Sequence[Sequence[Hashable]],
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    marcador_depot_etiqueta: str | None = None,
    guardar: bool = False,
    carpeta_salida: str | os.PathLike[str] | None = None,
    nombre_archivo: str | None = None,
    nombre_instancia: str = "instancia",
    usar_gpu: bool = False,
) -> ReporteSolucionResult:
    """
    Genera un reporte textual detallado para una solución con formato por etiquetas:

      ['D', 'TR1', 'TR5', ..., 'D']

    - Muestra tramos [DEADHEADING] con el camino y **cada arco recorrido** (con costo).
    - Muestra [SERVICIO] para cada tarea (arco servido) con costo y demanda acumulada.
    - Muestra [RETORNO] al depósito y totales por vehículo y total de la solución.

    Si ``guardar=True``, escribe el reporte en disco.
    ``usar_gpu`` deja la API preparada para backend acelerado; hoy usa fallback CPU.
    """
    deposito = int(data.get("DEPOSITO", 1))
    capacidad_max = float(data.get("CAPACIDAD", 0) or 0)

    mapa = construir_mapa_tareas_por_etiqueta(data)
    rutas, err = normalizar_rutas_etiquetas(solucion, data, mapa, marcador_depot_etiqueta)
    if err:
        raise ValueError(err)

    lineas: list[str] = []
    costos_por_veh: list[float] = []
    demandas_por_veh: list[float] = []
    costo_total_sol = 0.0

    for i, ruta in enumerate(rutas):
        veh = i + 1
        lineas.append(f"VEHÍCULO #{veh}:")

        nodo_actual = deposito
        demanda_acum = 0.0
        costo_veh = 0.0

        if not ruta:
            lineas.append(f"  (Sin tareas) [RETORNO] {deposito} -> Depósito {deposito} (Costo: 0.0)")
            lineas.append(f"  TOTAL VEHÍCULO #{veh}: Costo = 0.0 | Demanda = 0.0 / {capacidad_max}\n")
            costos_por_veh.append(0.0)
            demandas_por_veh.append(0.0)
            continue

        for etiqueta in ruta:
            tarea = mapa.get(etiqueta)
            if not tarea:
                raise KeyError(f"Tarea {etiqueta!r} no existe en los datos de la instancia.")

            nodos = tarea.get("nodos")
            if not nodos or len(nodos) != 2:
                raise ValueError(f"Tarea {etiqueta!r}: falta par de nodos.")

            u, v = int(nodos[0]), int(nodos[1])
            costo_serv = float(tarea.get("costo", 0) or 0)
            dem_serv = float(tarea.get("demanda", 0) or 0)
            etiqueta_str = str(tarea.get("tarea", etiqueta))

            # Deadheading hasta u (si hace falta)
            if nodo_grafo(nodo_actual) != nodo_grafo(u):
                path = shortest_path_nodes(G, nodo_actual, u, usar_gpu=usar_gpu)
                edges, costo_dh = path_edges_and_cost(G, path)
                lineas.append(
                    f"  [DEADHEADING] {nodo_actual} -> {u} (para servir {etiqueta_str} {u}->{v}) "
                    f"Caminos: {path} (Costo: {costo_dh})"
                )
                for a, b, c in edges:
                    lineas.append(f"    - Arco {a} -> {b} | Costo: {c}")
                costo_veh += costo_dh
                nodo_actual = u

            # Servicio (arco servido u->v)
            demanda_acum += dem_serv
            costo_veh += costo_serv
            estado_cap = "OK" if demanda_acum <= capacidad_max else "EXCEDIDA"
            lineas.append(
                f"  [SERVICIO] {etiqueta_str} ({u},{v}) | Costo: {costo_serv} | "
                f"Demanda +{dem_serv} = {demanda_acum} / {capacidad_max} [{estado_cap}]"
            )
            nodo_actual = v

        # Retorno a depósito
        if nodo_grafo(nodo_actual) != nodo_grafo(deposito):
            path = shortest_path_nodes(G, nodo_actual, deposito, usar_gpu=usar_gpu)
            edges, costo_ret = path_edges_and_cost(G, path)
            lineas.append(
                f"  [RETORNO] {nodo_actual} -> Depósito {deposito} Caminos: {path} (Costo: {costo_ret})"
            )
            for a, b, c in edges:
                lineas.append(f"    - Arco {a} -> {b} | Costo: {c}")
            costo_veh += costo_ret
        else:
            lineas.append(f"  [RETORNO] {deposito} -> Depósito {deposito} (Costo: 0.0)")

        costos_por_veh.append(costo_veh)
        demandas_por_veh.append(demanda_acum)
        costo_total_sol += costo_veh

        estado_final = "OK" if demanda_acum <= capacidad_max else "EXCEDIDA"
        lineas.append(
            f"  TOTAL VEHÍCULO #{veh}: Costo = {costo_veh} | "
            f"Demanda = {demanda_acum} / {capacidad_max} [{estado_final}]\n"
        )

    lineas.append(f"COSTO TOTAL DE LA SOLUCIÓN: {costo_total_sol}")
    texto = "\n".join(lineas)

    if guardar:
        if carpeta_salida is None:
            raise ValueError("Si guardar=True, debes proveer carpeta_salida.")
        out_dir = Path(carpeta_salida)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = nombre_archivo or f"{nombre_instancia}_reporte_solucion.txt"
        (out_dir / fname).write_text(texto, encoding="utf-8")

    return ReporteSolucionResult(
        texto=texto,
        costo_total=costo_total_sol,
        costos_por_vehiculo=costos_por_veh,
        demandas_por_vehiculo=demandas_por_veh,
    )


def reporte_solucion_desde_instancia(
    nombre_instancia: str,
    solucion: Sequence[Sequence[Hashable]],
    *,
    root: str | os.PathLike[str] | None = None,
    usar_gpu: bool = False,
    **kwargs: Any,
) -> ReporteSolucionResult:
    """Carga ``data`` y ``G`` del paquete y genera el reporte."""
    from .cargar_grafos import cargar_objeto_gexf
    from .instances import load_instances

    data = load_instances(nombre_instancia, root=root)
    G = cargar_objeto_gexf(nombre_instancia, root=root)
    return reporte_solucion(
        solucion,
        data,
        G,
        nombre_instancia=nombre_instancia,
        usar_gpu=usar_gpu,
        **kwargs,
    )

