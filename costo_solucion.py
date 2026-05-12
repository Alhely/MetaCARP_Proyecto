from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence

import networkx as nx

from .grafo_ruta import costo_camino_minimo, nodo_grafo
from .solucion_formato import (
    construir_mapa_tareas_por_etiqueta,
    normalizar_rutas_etiquetas,
)

__all__ = [
    "CostoSolucionResult",
    "costo_solucion",
    "costo_solucion_desde_instancia",
]


@dataclass
class CostoSolucionResult:
    """Resultado de :func:`costo_solucion`."""

    costos_por_ruta: list[float]
    costo_total: float
    texto_detalle: str | None = None
    """Texto multilínea si ``detalle=True``; si no, ``None``."""

    demandas_por_ruta: list[float] = field(default_factory=list)
    """Demanda total atendida por ruta (solo aristas de servicio)."""


def costo_solucion(
    solucion: Sequence[Sequence[Hashable]],
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    detalle: bool = False,
    carpeta_salida: str | os.PathLike[str] | None = None,
    nombre_instancia: str = "instancia",
    nombre_archivo_detalle: str | None = None,
    marcador_depot_etiqueta: str | None = None,
    usar_gpu: bool = False,
) -> CostoSolucionResult:
    """
    Calcula el costo total de una solución en formato lista de rutas ``[[],[],...]``.

    **Formato esperado:** etiquetas como ``['D','TR1','TR5',...,'D']`` con ``D`` =
    depósito (nodo en ``DEPOSITO``) y ``TR*`` / ``TNR*`` según ``t['tarea']`` en
    los datos. El marcador de texto configurable con ``marcador_depot_etiqueta`` o
    ``data['MARCADOR_DEPOT_ETIQUETA']`` (por defecto ``\"D\"``).

    Por cada tarea se suma el DH (tránsito) y el costo/demanda de servicio; al final el
    regreso al depósito. El DH usa el mismo criterio que :func:`reporte_solucion`:
    camino mínimo con peso ``cost`` y costo como **suma de arcos** del camino.

    ``usar_gpu`` deja la API preparada para backend acelerado; hoy usa fallback CPU.
    """
    deposito = int(data.get("DEPOSITO", 1))
    capacidad_max = float(data.get("CAPACIDAD", 0) or 0)
    mapa_et = construir_mapa_tareas_por_etiqueta(data)
    rutas_proc, err = normalizar_rutas_etiquetas(
        solucion, data, mapa_et, marcador_depot_etiqueta
    )
    if err:
        raise ValueError(err)
    tiene_tarea: Mapping[Any, dict[str, Any]] = mapa_et

    costos_rutas: list[float] = []
    demandas_rutas: list[float] = []
    costo_total_solucion = 0.0
    reporte: list[str] = []

    if detalle:
        reporte.append("=" * 80)
        reporte.append("EVALUACIÓN DE RUTAS (DH + servicio)")
        reporte.append("=" * 80)

    for i, ruta in enumerate(rutas_proc):
        idx_ruta = i + 1
        ruta_list_orig = list(solucion[i]) if i < len(solucion) else list(ruta)
        ruta_list = list(ruta)
        if detalle:
            reporte.append(f"RUTA {idx_ruta} (entrada) {ruta_list_orig}")

        if not ruta_list:
            costos_rutas.append(0.0)
            demandas_rutas.append(0.0)
            if detalle:
                reporte.append(
                    f"  -> Vehículo vacío | Costo: 0 | Demanda: 0 / {capacidad_max}\n"
                )
            continue

        costo_vehiculo = 0.0
        demanda_vehiculo = 0.0
        nodo_actual = deposito

        for id_tarea in ruta_list:
            tarea = tiene_tarea.get(id_tarea)
            if not tarea:
                raise KeyError(f"Tarea {id_tarea!r} no existe en los datos de la instancia.")

            nodos = tarea.get("nodos")
            if not nodos or len(nodos) != 2:
                raise ValueError(f"Tarea {id_tarea!r}: falta par de nodos.")
            u, v = int(nodos[0]), int(nodos[1])
            costo_serv = float(tarea.get("costo", 0) or 0)
            dem_serv = float(tarea.get("demanda", 0) or 0)
            etiqueta = tarea.get("tarea", str(id_tarea))

            if nodo_grafo(nodo_actual) != nodo_grafo(u):
                costo_dh, camino_dh = costo_camino_minimo(
                    G, nodo_actual, u, usar_gpu=usar_gpu
                )
                str_dh = " -> ".join(camino_dh)
            else:
                costo_dh = 0.0
                str_dh = f"Ninguno (ya en {u})"

            costo_total_paso = costo_dh + costo_serv
            costo_vehiculo += costo_total_paso
            demanda_vehiculo += dem_serv

            if detalle:
                reporte.append(
                    f"  -> {etiqueta} ({u},{v}) | DH: [{str_dh}] | "
                    f"Demanda servicio: {dem_serv} | "
                    f"Costo (DH + serv): {costo_dh} + {costo_serv} = {costo_total_paso}"
                )

            nodo_actual = v

        if nodo_grafo(nodo_actual) != nodo_grafo(deposito):
            costo_ret, camino_ret = costo_camino_minimo(
                G, nodo_actual, deposito, usar_gpu=usar_gpu
            )
            str_ret = " -> ".join(camino_ret)
        else:
            costo_ret = 0.0
            str_ret = f"Ninguno (ya en {deposito})"

        costo_vehiculo += costo_ret
        if detalle:
            reporte.append(
                f"  -> REGRESO AL DEPÓSITO ({deposito}) | DH: [{str_ret}] | Costo: {costo_ret}"
            )
            estado_cap = "OK" if demanda_vehiculo <= capacidad_max else "EXCEDIDA"
            reporte.append(
                f"  => TOTAL RUTA {idx_ruta}: costo = {costo_vehiculo} | "
                f"demanda = {demanda_vehiculo} / {capacidad_max} [{estado_cap}]\n"
            )

        costos_rutas.append(costo_vehiculo)
        demandas_rutas.append(demanda_vehiculo)
        costo_total_solucion += costo_vehiculo

    if detalle:
        reporte.append("=" * 80)
        reporte.append(f"COSTO TOTAL DE LA SOLUCIÓN: {costo_total_solucion}")
        reporte.append("=" * 80 + "\n")

    texto_final = "\n".join(reporte) if detalle else None

    if carpeta_salida is not None and texto_final is not None:
        out_dir = Path(carpeta_salida)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = nombre_archivo_detalle or f"{nombre_instancia}_solucion_detalle.txt"
        (out_dir / fname).write_text(texto_final, encoding="utf-8")

    return CostoSolucionResult(
        costos_por_ruta=costos_rutas,
        costo_total=costo_total_solucion,
        texto_detalle=texto_final,
        demandas_por_ruta=demandas_rutas,
    )


def costo_solucion_desde_instancia(
    nombre_instancia: str,
    solucion: Sequence[Sequence[Hashable]],
    *,
    detalle: bool = False,
    carpeta_salida: str | os.PathLike[str] | None = None,
    root: str | os.PathLike[str] | None = None,
    marcador_depot_etiqueta: str | None = None,
    usar_gpu: bool = False,
    **kwargs: Any,
) -> CostoSolucionResult:
    """Carga ``data`` (pickle) y ``G`` (GEXF) del paquete y evalúa el costo."""
    from .cargar_grafos import cargar_objeto_gexf
    from .instances import load_instances

    data = load_instances(nombre_instancia, root=root)
    G = cargar_objeto_gexf(nombre_instancia, root=root)
    return costo_solucion(
        solucion,
        data,
        G,
        detalle=detalle,
        carpeta_salida=carpeta_salida,
        nombre_instancia=nombre_instancia,
        marcador_depot_etiqueta=marcador_depot_etiqueta,
        usar_gpu=usar_gpu,
        **kwargs,
    )
