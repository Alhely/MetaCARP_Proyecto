# ============================================================
# reporte_solucion.py
# ------------------------------------------------------------
# Módulo responsable de generar el REPORTE TEXTUAL de una
# solución CARP.
#
# Un reporte muestra, vehículo por vehículo:
#   - Los tramos sin servicio (DEADHEADING): arco por arco,
#     con su costo individual.
#   - Las tareas servidas (SERVICIO): arco atendido, costo y
#     demanda acumulada.
#   - El retorno al depósito al final de cada ruta.
#   - El costo total de toda la solución.
#
# Esto permite depurar soluciones y verificar que el cálculo
# de costo coincide con el módulo costo_solucion.
# ============================================================

from __future__ import annotations

import os
from dataclasses import dataclass   # Genera automáticamente __init__, __repr__ y __eq__
from pathlib import Path            # Manejo de rutas de archivos de forma portable (Windows/Linux/Mac)
from typing import Any, Hashable, Mapping, Sequence

import networkx as nx   # Biblioteca para trabajar con grafos (Graph, MultiGraph, caminos mínimos)

# Funciones del módulo grafo_ruta para navegar el grafo de la instancia
from .grafo_ruta import nodo_grafo, path_edges_and_cost, shortest_path_nodes
# Funciones del módulo solucion_formato para normalizar y validar la solución
from .solucion_formato import construir_mapa_tareas_por_etiqueta, normalizar_rutas_etiquetas

# Nombres públicos que este módulo exporta
__all__ = [
    "ReporteSolucionResult",
    "reporte_solucion",
    "reporte_solucion_desde_instancia",
]


# ------------------------------------------------------------
# CLASE: ReporteSolucionResult
# ------------------------------------------------------------
# OOP utilizado: DATACLASS (clase de datos simple).
# Agrupa en un solo objeto todos los resultados del reporte:
# el texto formateado y las métricas numéricas.
#
# Sin frozen=True: los campos SÍ pueden modificarse después de
# crear el objeto (a diferencia de MovimientoVecindario, que es
# inmutable). Aquí no se necesita inmutabilidad porque este
# objeto solo se crea una vez y se devuelve al llamador.
@dataclass
class ReporteSolucionResult:
    texto: str                       # Reporte legible completo en formato de texto
    costo_total: float               # Suma de costos de todos los vehículos
    costos_por_vehiculo: list[float] # Costo individual de cada vehículo (indexado por vehículo)
    demandas_por_vehiculo: list[float]  # Demanda total atendida por cada vehículo


# ============================================================
# FUNCIÓN PRINCIPAL: reporte_solucion
# ============================================================
def reporte_solucion(
    solucion: Sequence[Sequence[Hashable]],  # Lista de rutas, cada ruta es una lista de etiquetas
    data: Mapping[str, Any],                 # Diccionario con los datos de la instancia (CARP)
    G: nx.Graph,                             # Grafo de la red vial (nodos = cruces, arcos = calles)
    *,
    marcador_depot_etiqueta: str | None = None,  # Etiqueta que representa el depósito (p.ej. "D")
    guardar: bool = False,                        # Si True, guarda el reporte en disco
    carpeta_salida: str | os.PathLike[str] | None = None,  # Carpeta donde guardar el archivo
    nombre_archivo: str | None = None,            # Nombre del archivo de salida (opcional)
    nombre_instancia: str = "instancia",          # Nombre de la instancia para el archivo
    usar_gpu: bool = False,                       # Flag de backend GPU (hoy: fallback a CPU)
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
    # Nodo depósito: punto de inicio y fin de todas las rutas.
    # data.get("DEPOSITO", 1) devuelve el valor de "DEPOSITO" si existe, o 1 si no.
    deposito = int(data.get("DEPOSITO", 1))

    # Capacidad máxima de cada vehículo (límite de demanda acumulada por ruta).
    # El "or 0" protege contra el caso en que el valor sea None o cadena vacía.
    capacidad_max = float(data.get("CAPACIDAD", 0) or 0)

    # Construye el mapa etiqueta -> datos de la tarea
    # (p.ej. "TR1" -> {"nodos": [2, 5], "costo": 3, "demanda": 10})
    mapa = construir_mapa_tareas_por_etiqueta(data)

    # Normaliza las rutas: elimina el marcador "D" y valida que todas las etiquetas
    # existan en el mapa. Si hay un error (etiqueta desconocida), devuelve un mensaje.
    rutas, err = normalizar_rutas_etiquetas(solucion, data, mapa, marcador_depot_etiqueta)
    if err:
        raise ValueError(err)

    # Estructuras de resultados que se van llenando durante el recorrido
    lineas: list[str] = []          # Líneas de texto del reporte (se unen al final)
    costos_por_veh: list[float] = []    # Costo de cada vehículo
    demandas_por_veh: list[float] = []  # Demanda total de cada vehículo
    costo_total_sol = 0.0               # Acumulador del costo total de la solución

    # ---- Iterar sobre cada vehículo (ruta) ----
    # enumerate(rutas) devuelve pares (índice, ruta); i empieza en 0 por eso se suma 1
    for i, ruta in enumerate(rutas):
        veh = i + 1   # Número de vehículo legible (empieza en 1, no en 0)
        lineas.append(f"VEHÍCULO #{veh}:")

        nodo_actual = deposito  # El vehículo comienza en el depósito
        demanda_acum = 0.0      # Demanda acumulada en esta ruta
        costo_veh = 0.0         # Costo acumulado en esta ruta

        # Caso especial: ruta sin tareas asignadas
        if not ruta:
            lineas.append(f"  (Sin tareas) [RETORNO] {deposito} -> Depósito {deposito} (Costo: 0.0)")
            lineas.append(f"  TOTAL VEHÍCULO #{veh}: Costo = 0.0 | Demanda = 0.0 / {capacidad_max}\n")
            costos_por_veh.append(0.0)
            demandas_por_veh.append(0.0)
            continue   # Pasa al siguiente vehículo

        # ---- Procesar cada tarea de la ruta ----
        for etiqueta in ruta:
            # Busca la tarea en el mapa por su etiqueta
            tarea = mapa.get(etiqueta)
            if not tarea:
                raise KeyError(f"Tarea {etiqueta!r} no existe en los datos de la instancia.")

            # Obtiene los nodos del arco que representa esta tarea
            nodos = tarea.get("nodos")
            if not nodos or len(nodos) != 2:
                raise ValueError(f"Tarea {etiqueta!r}: falta par de nodos.")

            u, v = int(nodos[0]), int(nodos[1])              # Nodo origen y destino del arco a servir
            costo_serv = float(tarea.get("costo", 0) or 0)  # Costo de servir este arco
            dem_serv = float(tarea.get("demanda", 0) or 0)  # Demanda de este arco
            etiqueta_str = str(tarea.get("tarea", etiqueta)) # Etiqueta canónica para el reporte

            # ---- DEADHEADING: traslado vacío hasta el inicio del arco a servir ----
            # Si el vehículo no está ya en el nodo u (inicio del arco a servir),
            # debe viajar hasta allí SIN prestar servicio. Esto se llama "deadheading".
            # nodo_grafo() convierte el ID entero al formato de nodo del grafo GEXF.
            if nodo_grafo(nodo_actual) != nodo_grafo(u):
                # Calcula el camino más corto desde nodo_actual hasta u
                path = shortest_path_nodes(G, nodo_actual, u, usar_gpu=usar_gpu)
                # Desglosa el camino en arcos individuales y calcula el costo total del tramo
                edges, costo_dh = path_edges_and_cost(G, path)
                lineas.append(
                    f"  [DEADHEADING] {nodo_actual} -> {u} (para servir {etiqueta_str} {u}->{v}) "
                    f"Caminos: {path} (Costo: {costo_dh})"
                )
                # Imprime cada arco del tramo de traslado con su costo individual
                for a, b, c in edges:
                    lineas.append(f"    - Arco {a} -> {b} | Costo: {c}")
                costo_veh += costo_dh   # Suma el costo del traslado al total del vehículo
                nodo_actual = u         # El vehículo ahora está en u

            # ---- SERVICIO: el vehículo atiende el arco (u, v) ----
            demanda_acum += dem_serv   # Acumula la demanda de esta tarea
            costo_veh += costo_serv    # Suma el costo de servicio
            # Verifica si la capacidad ya fue excedida
            estado_cap = "OK" if demanda_acum <= capacidad_max else "EXCEDIDA"
            lineas.append(
                f"  [SERVICIO] {etiqueta_str} ({u},{v}) | Costo: {costo_serv} | "
                f"Demanda +{dem_serv} = {demanda_acum} / {capacidad_max} [{estado_cap}]"
            )
            nodo_actual = v   # Después de servir, el vehículo está en el nodo v

        # ---- RETORNO AL DEPÓSITO ----
        # Al terminar todas las tareas, el vehículo vuelve al depósito.
        # Si ya está en el depósito, el costo de retorno es 0.
        if nodo_grafo(nodo_actual) != nodo_grafo(deposito):
            path = shortest_path_nodes(G, nodo_actual, deposito, usar_gpu=usar_gpu)
            edges, costo_ret = path_edges_and_cost(G, path)
            lineas.append(
                f"  [RETORNO] {nodo_actual} -> Depósito {deposito} Caminos: {path} (Costo: {costo_ret})"
            )
            for a, b, c in edges:
                lineas.append(f"    - Arco {a} -> {b} | Costo: {c}")
            costo_veh += costo_ret   # Suma el costo del retorno
        else:
            # El vehículo ya terminó exactamente en el depósito: retorno sin costo
            lineas.append(f"  [RETORNO] {deposito} -> Depósito {deposito} (Costo: 0.0)")

        # Registra los totales de este vehículo
        costos_por_veh.append(costo_veh)
        demandas_por_veh.append(demanda_acum)
        costo_total_sol += costo_veh   # Acumula en el total de la solución

        # Estado final de capacidad para el resumen del vehículo
        estado_final = "OK" if demanda_acum <= capacidad_max else "EXCEDIDA"
        lineas.append(
            f"  TOTAL VEHÍCULO #{veh}: Costo = {costo_veh} | "
            f"Demanda = {demanda_acum} / {capacidad_max} [{estado_final}]\n"
        )

    # Línea final con el costo total de toda la solución
    lineas.append(f"COSTO TOTAL DE LA SOLUCIÓN: {costo_total_sol}")

    # Une todas las líneas del reporte en un único string con saltos de línea
    texto = "\n".join(lineas)

    # ---- Guardar en disco (opcional) ----
    if guardar:
        if carpeta_salida is None:
            raise ValueError("Si guardar=True, debes proveer carpeta_salida.")
        out_dir = Path(carpeta_salida)
        # mkdir con parents=True crea todos los directorios intermedios si no existen
        # exist_ok=True no lanza error si la carpeta ya existe
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = nombre_archivo or f"{nombre_instancia}_reporte_solucion.txt"
        # write_text escribe el string al archivo en la codificación indicada
        (out_dir / fname).write_text(texto, encoding="utf-8")

    # Devuelve el objeto de resultado con el texto y las métricas numéricas
    return ReporteSolucionResult(
        texto=texto,
        costo_total=costo_total_sol,
        costos_por_vehiculo=costos_por_veh,
        demandas_por_vehiculo=demandas_por_veh,
    )


# ============================================================
# FUNCIÓN AUXILIAR: reporte_solucion_desde_instancia
# ============================================================
# Versión de conveniencia: en lugar de pedir que el llamador
# cargue data y G por separado, esta función los carga
# internamente usando solo el nombre de la instancia.
# Reduce el código repetitivo en scripts de experimentación.
def reporte_solucion_desde_instancia(
    nombre_instancia: str,              # Nombre de la instancia (p.ej. "gdb19")
    solucion: Sequence[Sequence[Hashable]],
    *,
    root: str | os.PathLike[str] | None = None,  # Directorio raíz de datos (None = default del paquete)
    usar_gpu: bool = False,
    **kwargs: Any,   # Parámetros adicionales que se pasan directamente a reporte_solucion
) -> ReporteSolucionResult:
    """Carga ``data`` y ``G`` del paquete y genera el reporte."""
    # Importaciones locales (dentro de la función) para evitar dependencias circulares
    # al cargar el módulo y para que el overhead de importación solo ocurra cuando
    # realmente se llama esta función.
    from .cargar_grafos import cargar_objeto_gexf
    from .instances import load_instances

    data = load_instances(nombre_instancia, root=root)           # Carga los datos de la instancia
    G = cargar_objeto_gexf(nombre_instancia, root=root)          # Carga el grafo en formato GEXF
    return reporte_solucion(
        solucion,
        data,
        G,
        nombre_instancia=nombre_instancia,
        usar_gpu=usar_gpu,
        **kwargs,   # Pasa los argumentos adicionales (guardar, carpeta_salida, etc.)
    )
