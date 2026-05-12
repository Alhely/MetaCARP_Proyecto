# Módulo: costo_solucion.py
# Propósito: Calcula el costo total de una solución CARP dada en formato de
# etiquetas de texto. Es la función de evaluación "clásica" del proyecto:
# más legible que el evaluador vectorizado, pero más lenta porque llama a
# Dijkstra por cada par de tareas consecutivas. Se usa principalmente para
# verificar resultados, generar reportes detallados y en la evaluación de
# soluciones iniciales (donde la velocidad no es crítica).
#
# Fórmula de costo por ruta:
#   costo_ruta = sum_{t en ruta}[
#       dist_minima(nodo_actual → u_tarea)    ← deadheading (DH)
#       + costo_servicio(tarea)                ← costo de servir el arco
#   ] + dist_minima(v_ultima_tarea → deposito) ← regreso al depósito

from __future__ import annotations  # Permite anotaciones de tipo forward-reference

import os                                        # Para rutas del sistema de archivos
from dataclasses import dataclass, field         # Para definir clases de datos
from pathlib import Path                         # Manejo de rutas de archivos moderno
from typing import Any, Hashable, Mapping, Sequence  # Tipos genéricos

import networkx as nx  # Librería de grafos (se usa para el camino mínimo Dijkstra)

# Funciones de bajo nivel sobre el grafo: camino mínimo y conversión de nodos.
from .grafo_ruta import costo_camino_minimo, nodo_grafo

# Funciones del formato estándar de soluciones: construir el mapa de tareas
# y normalizar las rutas (eliminar marcadores de depósito, validar etiquetas).
from .solucion_formato import (
    construir_mapa_tareas_por_etiqueta,
    normalizar_rutas_etiquetas,
)

# API pública de este módulo: solo estos nombres quedan accesibles desde fuera.
__all__ = [
    "CostoSolucionResult",
    "costo_solucion",
    "costo_solucion_desde_instancia",
]


# ---------------------------------------------------------------------------
# Clase de datos: CostoSolucionResult
# ---------------------------------------------------------------------------
# @dataclass es un decorador que genera automáticamente __init__, __repr__ y
# __eq__ para esta clase. No usamos frozen=True aquí porque el resultado puede
# necesitar ser construido incremetalmente (por ej. añadir costos por ruta uno a uno).
@dataclass
class CostoSolucionResult:
    """
    Resultado del cálculo de costo de una solución CARP.

    Agrupa el costo global, los costos individuales por ruta y las demandas,
    más un texto de detalle opcional para reportes de depuración o auditoría.

    Atributos:
    - ``costos_por_ruta``: lista con el costo total de cada vehículo/ruta.
    - ``costo_total``: suma de todos los costos por ruta (costo de la solución).
    - ``texto_detalle``: string multilínea con el desglose completo paso a paso.
      Es None si se llamó con ``detalle=False``.
    - ``demandas_por_ruta``: demanda total atendida por cada ruta (suma de
      demandas de las aristas de servicio, sin contar deadheading).
    """

    # Lista de costos, uno por ruta (índice 0 = primera ruta, etc.).
    costos_por_ruta: list[float]

    # Suma total: costos_por_ruta[0] + costos_por_ruta[1] + …
    costo_total: float

    # Texto de reporte multilínea (solo si detalle=True). None si no se pidió.
    texto_detalle: str | None = None

    # field(default_factory=list) crea una nueva lista vacía para cada instancia.
    # Si escribiéramos demandas_por_ruta: list[float] = [], todas las instancias
    # compartirían la MISMA lista (bug clásico de Python con mutables por defecto).
    demandas_por_ruta: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Función principal: costo_solucion
# ---------------------------------------------------------------------------
def costo_solucion(
    solucion: Sequence[Sequence[Hashable]],           # Rutas de la solución
    data: Mapping[str, Any],                          # Datos de la instancia
    G: nx.Graph,                                      # Grafo de la instancia
    *,
    detalle: bool = False,                            # Si True, genera reporte textual
    carpeta_salida: str | os.PathLike[str] | None = None,  # Directorio para guardar reporte
    nombre_instancia: str = "instancia",              # Nombre para el archivo de reporte
    nombre_archivo_detalle: str | None = None,        # Nombre personalizado del archivo
    marcador_depot_etiqueta: str | None = None,       # Token de depósito (por defecto "D")
    usar_gpu: bool = False,                           # Reservado para futuro backend GPU
) -> CostoSolucionResult:
    """
    Calcula el costo total de una solución CARP dada por listas de rutas con etiquetas.

    **Formato esperado de la solución:**
    Lista de rutas, donde cada ruta es una lista de etiquetas como:
    ``['D', 'TR1', 'TR5', 'TNR2', 'D']``
    - ``D`` (o el valor de ``marcador_depot_etiqueta``) marca el depósito.
    - ``TR*`` y ``TNR*`` son etiquetas de tareas definidas en ``LISTA_ARISTAS_REQ``
      y ``LISTA_ARISTAS_NOREQ`` de los datos de la instancia.

    **Fórmula de costo:**
    Para cada ruta, se acumula:
    1. Deadheading (DH): camino mínimo desde el nodo actual hasta el nodo ``u``
       de la siguiente tarea (costo de transitar sin prestar servicio).
    2. Costo de servicio: costo de la arista que se sirve (atributo ``costo`` de la tarea).
    3. Al final de la ruta: DH de regreso al depósito desde el nodo ``v`` de la última tarea.

    El DH se calcula con Dijkstra (a través de :func:`costo_camino_minimo`) usando
    el atributo ``"cost"`` de las aristas del grafo.

    Args:
        solucion: Lista de rutas. Cada ruta es una secuencia de etiquetas hashables.
        data: Diccionario de datos de la instancia (resultado de ``load_instances``).
        G: Grafo NetworkX de la instancia (cargado del GEXF).
        detalle: Si True, genera un texto paso a paso con cada DH y costo de servicio.
        carpeta_salida: Si se especifica, guarda el texto de detalle en este directorio.
        nombre_instancia: Prefijo del nombre de archivo del reporte.
        nombre_archivo_detalle: Nombre completo del archivo. Si None, se genera automáticamente.
        marcador_depot_etiqueta: Token de texto del depósito. Si None, usa el de ``data``
            o el valor por defecto ("D").
        usar_gpu: Reservado para futuro backend GPU (actualmente sin efecto real).

    Returns:
        CostoSolucionResult con costos por ruta, costo total y texto de detalle opcional.

    Raises:
        ValueError: Si el formato de la solución es incorrecto o hay errores de normalización.
        KeyError: Si una etiqueta de tarea no existe en los datos de la instancia.
    """
    # Leemos el nodo depósito como entero (por defecto 1).
    deposito = int(data.get("DEPOSITO", 1))

    # Capacidad máxima del vehículo (para mostrar estado en el reporte de detalle).
    capacidad_max = float(data.get("CAPACIDAD", 0) or 0)

    # Construimos el mapa de tareas: {etiqueta: datos_tarea}.
    # Incluye tanto aristas requeridas como no requeridas.
    mapa_et = construir_mapa_tareas_por_etiqueta(data)

    # Normalizamos las rutas: eliminamos marcadores de depósito y validamos etiquetas.
    # 'rutas_proc' son las rutas limpias (solo etiquetas de tareas reales).
    # 'err' es None si todo está bien, o un string de error.
    rutas_proc, err = normalizar_rutas_etiquetas(
        solucion, data, mapa_et, marcador_depot_etiqueta
    )
    if err:
        raise ValueError(err)

    # Alias del mapa de tareas con anotación de tipo más precisa.
    tiene_tarea: Mapping[Any, dict[str, Any]] = mapa_et

    # Inicializamos los acumuladores de la solución.
    costos_rutas: list[float] = []    # Costo de cada ruta individual
    demandas_rutas: list[float] = []  # Demanda de cada ruta individual
    costo_total_solucion = 0.0        # Suma de todos los costos de ruta

    # Lista de líneas del reporte de detalle (solo se usa si detalle=True).
    reporte: list[str] = []

    if detalle:
        # Añadimos encabezado al reporte de detalle.
        reporte.append("=" * 80)
        reporte.append("EVALUACIÓN DE RUTAS (DH + servicio)")
        reporte.append("=" * 80)

    # Iteramos sobre cada ruta de la solución.
    for i, ruta in enumerate(rutas_proc):
        idx_ruta = i + 1  # Número de ruta 1-based para el reporte (más legible)

        # Ruta original (con depósito) para mostrar en el reporte de entrada.
        ruta_list_orig = list(solucion[i]) if i < len(solucion) else list(ruta)

        # Ruta procesada (sin depósito, solo tareas reales).
        ruta_list = list(ruta)

        if detalle:
            reporte.append(f"RUTA {idx_ruta} (entrada) {ruta_list_orig}")

        # Caso: ruta vacía (vehículo sin asignación de tareas).
        if not ruta_list:
            costos_rutas.append(0.0)
            demandas_rutas.append(0.0)
            if detalle:
                reporte.append(
                    f"  -> Vehículo vacío | Costo: 0 | Demanda: 0 / {capacidad_max}\n"
                )
            continue  # Pasamos a la siguiente ruta

        # Inicializamos los acumuladores de esta ruta.
        costo_vehiculo = 0.0    # Costo total del vehículo en esta ruta
        demanda_vehiculo = 0.0  # Demanda total atendida en esta ruta

        # 'nodo_actual' rastrea en qué nodo del grafo está el vehículo actualmente.
        # Al inicio de cada ruta, el vehículo está en el depósito.
        nodo_actual = deposito

        # Iteramos sobre cada tarea (arista de servicio) de la ruta.
        for id_tarea in ruta_list:
            # Buscamos los datos de esta tarea en el mapa.
            tarea = tiene_tarea.get(id_tarea)
            if not tarea:
                raise KeyError(f"Tarea {id_tarea!r} no existe en los datos de la instancia.")

            # Extraemos los nodos extremos del arco de servicio.
            nodos = tarea.get("nodos")
            if not nodos or len(nodos) != 2:
                raise ValueError(f"Tarea {id_tarea!r}: falta par de nodos.")

            # u = nodo inicio del arco; v = nodo fin del arco.
            u, v = int(nodos[0]), int(nodos[1])

            # Costo de servicio: el costo de recorrer este arco prestando servicio.
            costo_serv = float(tarea.get("costo", 0) or 0)

            # Demanda de servicio: la cantidad de residuos/recurso consumido en este arco.
            dem_serv = float(tarea.get("demanda", 0) or 0)

            # Etiqueta de la tarea para el reporte (ej. "TR1").
            etiqueta = tarea.get("tarea", str(id_tarea))

            # Calculamos el deadheading (DH): costo de moverse desde nodo_actual hasta u.
            # nodo_grafo() convierte el entero al formato string del GEXF ("3" en vez de 3).
            if nodo_grafo(nodo_actual) != nodo_grafo(u):
                # El vehículo no está en u: necesita moverse (deadheading).
                # costo_camino_minimo llama a Dijkstra en el grafo G.
                costo_dh, camino_dh = costo_camino_minimo(
                    G, nodo_actual, u, usar_gpu=usar_gpu
                )
                # Para el reporte: representamos el camino como "3 -> 7 -> 12".
                str_dh = " -> ".join(camino_dh)
            else:
                # El vehículo ya está en u: no hay deadheading (costo 0).
                costo_dh = 0.0
                str_dh = f"Ninguno (ya en {u})"

            # Costo total de este paso = deadheading + servicio.
            costo_total_paso = costo_dh + costo_serv

            # Acumulamos los costos y demanda de la ruta.
            costo_vehiculo += costo_total_paso
            demanda_vehiculo += dem_serv

            if detalle:
                reporte.append(
                    f"  -> {etiqueta} ({u},{v}) | DH: [{str_dh}] | "
                    f"Demanda servicio: {dem_serv} | "
                    f"Costo (DH + serv): {costo_dh} + {costo_serv} = {costo_total_paso}"
                )

            # Después de servir el arco (u→v), el vehículo está en v.
            nodo_actual = v

        # --- Regreso al depósito ---
        # Al terminar todas las tareas, el vehículo debe volver al depósito.
        if nodo_grafo(nodo_actual) != nodo_grafo(deposito):
            # El vehículo no está en el depósito: calculamos el costo de regreso.
            costo_ret, camino_ret = costo_camino_minimo(
                G, nodo_actual, deposito, usar_gpu=usar_gpu
            )
            str_ret = " -> ".join(camino_ret)
        else:
            # El vehículo ya está en el depósito (caso raro, pero posible).
            costo_ret = 0.0
            str_ret = f"Ninguno (ya en {deposito})"

        # Sumamos el costo de regreso al costo total del vehículo.
        costo_vehiculo += costo_ret

        if detalle:
            reporte.append(
                f"  -> REGRESO AL DEPÓSITO ({deposito}) | DH: [{str_ret}] | Costo: {costo_ret}"
            )
            # Verificamos si la ruta respeta la restricción de capacidad.
            estado_cap = "OK" if demanda_vehiculo <= capacidad_max else "EXCEDIDA"
            reporte.append(
                f"  => TOTAL RUTA {idx_ruta}: costo = {costo_vehiculo} | "
                f"demanda = {demanda_vehiculo} / {capacidad_max} [{estado_cap}]\n"
            )

        # Guardamos los resultados de esta ruta en las listas de la solución.
        costos_rutas.append(costo_vehiculo)
        demandas_rutas.append(demanda_vehiculo)
        costo_total_solucion += costo_vehiculo  # Acumulamos al total de la solución

    if detalle:
        # Añadimos el resumen final al reporte.
        reporte.append("=" * 80)
        reporte.append(f"COSTO TOTAL DE LA SOLUCIÓN: {costo_total_solucion}")
        reporte.append("=" * 80 + "\n")

    # Unimos todas las líneas del reporte con saltos de línea.
    # Si detalle=False, texto_final es None (no se generó reporte).
    texto_final = "\n".join(reporte) if detalle else None

    # Si se pidió guardar el reporte en disco, lo hacemos aquí.
    if carpeta_salida is not None and texto_final is not None:
        # Path() crea un objeto de ruta compatible con cualquier sistema operativo.
        out_dir = Path(carpeta_salida)

        # mkdir(parents=True, exist_ok=True): crea el directorio y todos sus padres
        # si no existen, sin lanzar error si ya existe.
        out_dir.mkdir(parents=True, exist_ok=True)

        # Nombre del archivo: personalizado si se especificó, o generado automáticamente.
        fname = nombre_archivo_detalle or f"{nombre_instancia}_solucion_detalle.txt"

        # Escribimos el texto del reporte en el archivo con codificación UTF-8.
        (out_dir / fname).write_text(texto_final, encoding="utf-8")

    # Construimos y devolvemos el resultado con todos los costos y el reporte.
    return CostoSolucionResult(
        costos_por_ruta=costos_rutas,
        costo_total=costo_total_solucion,
        texto_detalle=texto_final,
        demandas_por_ruta=demandas_rutas,
    )


# ---------------------------------------------------------------------------
# Función: costo_solucion_desde_instancia
# ---------------------------------------------------------------------------
def costo_solucion_desde_instancia(
    nombre_instancia: str,                              # Nombre de la instancia
    solucion: Sequence[Sequence[Hashable]],             # Rutas de la solución
    *,
    detalle: bool = False,
    carpeta_salida: str | os.PathLike[str] | None = None,
    root: str | os.PathLike[str] | None = None,        # Directorio raíz alternativo
    marcador_depot_etiqueta: str | None = None,
    usar_gpu: bool = False,
    **kwargs: Any,   # Argumentos extra que se pasan directamente a costo_solucion
) -> CostoSolucionResult:
    """
    Versión de conveniencia: carga la instancia por nombre y evalúa el costo.

    Carga automáticamente los datos de la instancia (archivo pickle) y el grafo
    (archivo GEXF) del paquete, luego llama a :func:`costo_solucion`.

    Args:
        nombre_instancia: Identificador de la instancia (ej. "EGL-E1-A").
        solucion: Lista de rutas con etiquetas de texto.
        detalle: Si True, genera reporte textual detallado.
        carpeta_salida: Directorio donde guardar el reporte (si detalle=True).
        root: Directorio raíz alternativo para buscar los archivos.
        marcador_depot_etiqueta: Token de texto del depósito.
        usar_gpu: Reservado para futuro backend GPU.
        **kwargs: Argumentos adicionales pasados directamente a ``costo_solucion``.

    Returns:
        CostoSolucionResult con costos por ruta, costo total y texto de detalle opcional.
    """
    # Importaciones diferidas: evitan ciclos de importación al cargar este módulo.
    # Se importan aquí (dentro de la función) y no al nivel del módulo.
    from .cargar_grafos import cargar_objeto_gexf  # Carga el grafo GEXF de la instancia
    from .instances import load_instances          # Carga el pickle de datos de la instancia

    # Cargamos los datos de la instancia desde el archivo pickle.
    data = load_instances(nombre_instancia, root=root)

    # Cargamos el grafo NetworkX desde el archivo GEXF de la instancia.
    G = cargar_objeto_gexf(nombre_instancia, root=root)

    # Delegamos al evaluador principal con todos los parámetros.
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
