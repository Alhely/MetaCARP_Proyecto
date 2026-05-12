# Módulo: grafo_ruta.py
# Propósito: Funciones de bajo nivel para operar sobre el grafo de la instancia CARP
# usando la librería NetworkX. Concentra toda la lógica de rutas mínimas (Dijkstra)
# y consulta de costos de arcos en un único lugar, para que el resto del código
# (costo_solucion, reporte_solucion, etc.) no tenga que repetir estas operaciones.
#
# Conceptos de grafos usados aquí:
# - Grafo ponderado: cada arco (u, v) tiene un atributo numérico "cost".
# - Camino mínimo (shortest path): la secuencia de nodos que minimiza la suma
#   de costos de los arcos recorridos. Se calcula con el algoritmo de Dijkstra.
# - Deadheading (DH): recorrido de un arco sin prestar servicio, solo para
#   moverse de un punto a otro. Su costo es el camino mínimo entre esos puntos.

from __future__ import annotations  # Permite anotaciones de tipo forward-reference

from typing import Any, Sequence  # Tipos genéricos para anotaciones

import networkx as nx                           # Librería principal de grafos en Python
from networkx.exception import NetworkXNoPath  # Excepción que lanza nx cuando no hay camino

# __all__ define la API pública de este módulo.
# Solo estas funciones quedan visibles al hacer "from grafo_ruta import *".
__all__ = [
    "nodo_grafo",
    "edge_cost",
    "shortest_path_nodes",
    "path_edges_and_cost",
    "costo_camino_minimo",
]


# ---------------------------------------------------------------------------
# Función interna: _aplicar_backend_gpu_placeholder
# ---------------------------------------------------------------------------
def _aplicar_backend_gpu_placeholder(usar_gpu: bool) -> tuple[str, str]:
    """
    Registra el backend solicitado y el backend real usado para trazabilidad.

    En el futuro, si se implementa un backend GPU real para cálculo de rutas,
    esta función se actualizará. Por ahora, siempre devuelve CPU como backend real.

    Returns:
        Tupla (backend_solicitado, backend_real). Ejemplo: ("gpu", "cpu").
    """
    if not usar_gpu:
        # El usuario no pidió GPU: ambos valores son "cpu".
        return "cpu", "cpu"
    # El usuario pidió GPU, pero aún no existe implementación real: fallback a CPU.
    return "gpu", "cpu"


# ---------------------------------------------------------------------------
# Función: nodo_grafo
# ---------------------------------------------------------------------------
def nodo_grafo(n: Any) -> str:
    """
    Convierte un ID de nodo de la instancia al tipo de dato usado en el GEXF.

    Los archivos GEXF almacenan los nodos como cadenas de texto ("1", "2", …),
    pero la instancia puede tener los nodos como enteros (1, 2, …) o flotantes
    (1.0). Esta función normaliza cualquier formato a string de entero.

    Ejemplo: nodo_grafo(3.0) → "3",  nodo_grafo("5") → "5"
    """
    # int(n) elimina decimales (3.0 → 3), str() convierte a texto ("3").
    return str(int(n))


# ---------------------------------------------------------------------------
# Función: edge_cost
# ---------------------------------------------------------------------------
def edge_cost(G: nx.Graph, a: str, b: str) -> float:
    """
    Devuelve el costo del arco ``a → b`` en el grafo ``G``.

    Soporta tanto Graph simple como MultiGraph (donde puede haber varias
    aristas entre el mismo par de nodos; en ese caso usa la primera con 'cost').

    Args:
        G: Grafo NetworkX cargado desde el GEXF de la instancia.
        a: Nodo de origen (como string, e.g. "3").
        b: Nodo de destino (como string, e.g. "7").

    Returns:
        Costo numérico del arco como float.

    Raises:
        KeyError: Si no existe arista entre a y b, o si no tiene el atributo 'cost'.
    """
    # get_edge_data devuelve los atributos de la arista (u, v) o None si no existe.
    data = G.get_edge_data(a, b)

    if not data:
        raise KeyError(f"No existe arista en el grafo entre {a} y {b}.")

    # Caso Graph simple: data es un dict {"cost": 5.0, "label": "…", …}
    if "cost" in data:
        return float(data["cost"])

    # Caso MultiGraph: data es un dict anidado {0: {"cost": 5.0}, 1: {"cost": 3.0}, …}
    # Iteramos las sub-aristas y devolvemos el primer 'cost' que encontremos.
    if isinstance(data, dict):
        for _k, attrs in data.items():
            if isinstance(attrs, dict) and "cost" in attrs:
                return float(attrs["cost"])

    raise KeyError(f"La arista {a}-{b} no tiene atributo 'cost'.")


# ---------------------------------------------------------------------------
# Función: shortest_path_nodes
# ---------------------------------------------------------------------------
def shortest_path_nodes(
    G: nx.Graph,           # Grafo de la instancia
    origen: Any,           # Nodo de origen (entero o string)
    destino: Any,          # Nodo de destino (entero o string)
    *,
    usar_gpu: bool = False,  # Reservado para futuro backend GPU
) -> list[str]:
    """
    Calcula la secuencia de nodos del camino mínimo entre ``origen`` y ``destino``.

    Usa el algoritmo de Dijkstra de NetworkX con el peso de arco ``"cost"``.
    El camino mínimo es el que minimiza la suma de costos de los arcos recorridos.

    Args:
        G: Grafo de la instancia CARP.
        origen: Nodo de partida (se normaliza con :func:`nodo_grafo`).
        destino: Nodo de llegada (se normaliza con :func:`nodo_grafo`).
        usar_gpu: Si True, se registra la solicitud GPU (sin efecto real aún).

    Returns:
        Lista de nodos como strings del camino mínimo, incluyendo origen y destino.
        Ejemplo: ["3", "7", "12", "5"]

    Raises:
        ValueError: Si algún nodo no existe en el grafo, o si no hay camino.
    """
    # Registramos el backend (por trazabilidad); los valores no se usan aún.
    _backend_solicitado, _backend_real = _aplicar_backend_gpu_placeholder(usar_gpu)

    # Normalizamos los nodos al formato string del GEXF.
    s, t = nodo_grafo(origen), nodo_grafo(destino)

    # Verificamos que ambos nodos existan en el grafo antes de llamar a Dijkstra.
    # Esto da un error más claro que el que lanzaría NetworkX internamente.
    if s not in G or t not in G:
        raise ValueError(f"Nodo {s} o {t} no está en el grafo.")

    try:
        # nx.shortest_path usa Dijkstra cuando se especifica weight.
        # Devuelve la lista de nodos del camino: [origen, nodo1, nodo2, …, destino].
        return nx.shortest_path(G, source=s, target=t, weight="cost")
    except NetworkXNoPath as e:
        # NetworkX lanza esta excepción específica cuando no existe ningún camino.
        raise ValueError(f"No hay camino en G entre {s} y {t}.") from e


# ---------------------------------------------------------------------------
# Función: path_edges_and_cost
# ---------------------------------------------------------------------------
def path_edges_and_cost(
    G: nx.Graph,            # Grafo de la instancia
    path: Sequence[str],    # Lista de nodos del camino (resultado de shortest_path_nodes)
) -> tuple[list[tuple[str, str, float]], float]:
    """
    Descompone un camino en sus arcos individuales con sus costos, y calcula el total.

    Dada la lista de nodos ["3", "7", "12"], produce:
        arcos: [("3","7", 2.5), ("7","12", 1.0)]
        total: 3.5

    Args:
        G: Grafo de la instancia.
        path: Secuencia de nodos del camino (al menos 2 elementos).

    Returns:
        Tupla (lista_de_arcos, costo_total).
        Cada arco es una tupla (nodo_origen, nodo_destino, costo_arco).
    """
    edges: list[tuple[str, str, float]] = []  # Lista de arcos con sus costos individuales
    total = 0.0                                # Acumulador del costo total del camino

    # zip(path, path[1:]) genera pares de nodos consecutivos:
    # Si path = [A, B, C] → [(A,B), (B,C)]
    for a, b in zip(path, path[1:]):
        c = edge_cost(G, a, b)    # Costo del arco entre este par de nodos
        edges.append((a, b, c))   # Guardamos el arco con su costo
        total += c                 # Acumulamos al total

    return edges, total


# ---------------------------------------------------------------------------
# Función principal: costo_camino_minimo
# ---------------------------------------------------------------------------
def costo_camino_minimo(
    G: nx.Graph,           # Grafo de la instancia
    origen: Any,           # Nodo de partida
    destino: Any,          # Nodo de llegada
    *,
    usar_gpu: bool = False,  # Reservado para futuro backend GPU
) -> tuple[float, list[str]]:
    """
    Calcula el costo total del camino mínimo entre ``origen`` y ``destino``.

    El costo es la **suma de los costos de los arcos** del camino mínimo,
    calculado con Dijkstra sobre el atributo ``"cost"`` de las aristas del grafo.

    Esta función es la que llaman ``costo_solucion`` y ``reporte_solucion``
    para calcular el deadheading (DH) entre tareas consecutivas de una ruta.

    Args:
        G: Grafo de la instancia CARP.
        origen: Nodo de partida (puede ser entero o string).
        destino: Nodo de llegada (puede ser entero o string).
        usar_gpu: Reservado para futuro backend GPU; hoy siempre usa CPU.

    Returns:
        Tupla ``(costo_total, camino)`` donde:
        - ``costo_total``: suma de costos de arcos del camino mínimo (float).
        - ``camino``: lista de nodos del camino como strings.

    Si ``origen == destino``, devuelve ``(0.0, [nodo])`` sin llamar a Dijkstra.
    """
    # Normalizamos ambos nodos al formato string del GEXF.
    s, t = nodo_grafo(origen), nodo_grafo(destino)

    # Caso trivial: origen y destino son el mismo nodo → costo cero, camino de un nodo.
    if s == t:
        return 0.0, [s]

    # Calculamos el camino mínimo como lista de nodos.
    path = shortest_path_nodes(G, origen, destino, usar_gpu=usar_gpu)

    # Calculamos el costo total sumando los arcos del camino.
    # Usamos _edges solo internamente; al llamador solo le interesa el total y el camino.
    _edges, total = path_edges_and_cost(G, path)

    return total, path
