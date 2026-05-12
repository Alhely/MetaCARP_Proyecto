from __future__ import annotations

from typing import Any, Sequence

import networkx as nx
from networkx.exception import NetworkXNoPath

__all__ = [
    "nodo_grafo",
    "edge_cost",
    "shortest_path_nodes",
    "path_edges_and_cost",
    "costo_camino_minimo",
]


def _aplicar_backend_gpu_placeholder(usar_gpu: bool) -> tuple[str, str]:
    """Devuelve (backend_solicitado, backend_real) para trazabilidad futura."""
    if not usar_gpu:
        return "cpu", "cpu"
    # Placeholder: aún no existe backend GPU real para rutas.
    return "gpu", "cpu"


def nodo_grafo(n: Any) -> str:
    """Convierte id de nodo de la instancia al tipo usado en el GEXF (str)."""
    return str(int(n))


def edge_cost(G: nx.Graph, a: str, b: str) -> float:
    """Costo ``cost`` de la arista ``a-b`` en ``G`` (Graph o MultiGraph)."""
    data = G.get_edge_data(a, b)
    if not data:
        raise KeyError(f"No existe arista en el grafo entre {a} y {b}.")
    if "cost" in data:
        return float(data["cost"])
    if isinstance(data, dict):
        for _k, attrs in data.items():
            if isinstance(attrs, dict) and "cost" in attrs:
                return float(attrs["cost"])
    raise KeyError(f"La arista {a}-{b} no tiene atributo 'cost'.")


def shortest_path_nodes(
    G: nx.Graph, origen: Any, destino: Any, *, usar_gpu: bool = False
) -> list[str]:
    """Secuencia de nodos del camino mínimo con peso ``cost``."""
    _backend_solicitado, _backend_real = _aplicar_backend_gpu_placeholder(usar_gpu)
    s, t = nodo_grafo(origen), nodo_grafo(destino)
    if s not in G or t not in G:
        raise ValueError(f"Nodo {s} o {t} no está en el grafo.")
    try:
        return nx.shortest_path(G, source=s, target=t, weight="cost")
    except NetworkXNoPath as e:
        raise ValueError(f"No hay camino en G entre {s} y {t}.") from e


def path_edges_and_cost(
    G: nx.Graph, path: Sequence[str]
) -> tuple[list[tuple[str, str, float]], float]:
    """Lista de arcos ``(a,b,costo)`` del camino y costo total."""
    edges: list[tuple[str, str, float]] = []
    total = 0.0
    for a, b in zip(path, path[1:]):
        c = edge_cost(G, a, b)
        edges.append((a, b, c))
        total += c
    return edges, total


def costo_camino_minimo(
    G: nx.Graph, origen: Any, destino: Any, *, usar_gpu: bool = False
) -> tuple[float, list[str]]:
    """
    Costo total del camino mínimo ``origen -> destino`` como **suma de costos de arcos**.

    Si origen y destino coinciden, devuelve ``(0.0, [nodo])``.
    """
    s, t = nodo_grafo(origen), nodo_grafo(destino)
    if s == t:
        return 0.0, [s]
    path = shortest_path_nodes(G, origen, destino, usar_gpu=usar_gpu)
    _edges, total = path_edges_and_cost(G, path)
    return total, path
