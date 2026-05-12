"""
Evaluador rápido de costo para metaheurísticas CARP.

Diseño
======
Las metaheurísticas evalúan miles/millones de soluciones por corrida. Cada
``costo_solucion`` clásico recorre rutas y, por cada salto, llama a
``nx.shortest_path`` sobre el grafo. Eso convierte la evaluación en el cuello de
botella absoluto del experimento.

Este módulo construye **una sola vez** un *contexto vectorizado* a partir de la
matriz Dijkstra precomputada y de la codificación entera de tareas, y expone
funciones de evaluación O(longitud_de_ruta) por solución (por etiquetas o IDs).
La fórmula de costo se preserva: por cada tarea se suma el deadheading
(camino mínimo) + costo de servicio, y al final el regreso al depósito.

Backends
--------
- ``cpu``: NumPy puro, óptimo para una solución a la vez.
- ``gpu``: CuPy si está disponible; útil para evaluar **lotes** de soluciones
  (poblaciones de Tabu/Abejas/Cuckoo). Si CuPy no está disponible se hace
  fallback transparente a CPU.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Hashable

import networkx as nx
import numpy as np

from .busqueda_indices import SearchEncoding, build_search_encoding
from .cargar_matrices import cargar_matriz_dijkstra
from .solucion_formato import (
    CLAVE_MARCADOR_DEPOSITO_DEFAULT,
    construir_mapa_tareas_por_etiqueta,
)

__all__ = [
    "ContextoEvaluacion",
    "construir_contexto",
    "construir_contexto_desde_instancia",
    "costo_rapido",
    "costo_rapido_ids",
    "costo_lote_ids",
    "gpu_disponible",
]


_INF = np.float64("inf")


def gpu_disponible() -> bool:
    """Detecta si CuPy es importable y hay un dispositivo CUDA accesible."""
    try:
        import cupy as cp  # type: ignore
        try:
            cp.cuda.runtime.getDeviceCount()
            return True
        except Exception:  # noqa: BLE001
            return False
    except Exception:  # noqa: BLE001
        return False


# Cache global por instancia (una sola vez por proceso) para no reconstruir
# matrices densas en cada corrida del grid search.
_CACHE_CONTEXTO: dict[tuple[str, str | None], "ContextoEvaluacion"] = {}


@dataclass(frozen=True, slots=True)
class ContextoEvaluacion:
    """
    Contexto inmutable y compartido para evaluar costos de soluciones.

    Atributos clave:
    - ``dist``: matriz densa (N+1, N+1) con distancias 1-indexed (0-fila/col vacía).
    - ``u_arr`` / ``v_arr``: nodos extremos por id de tarea.
    - ``costo_serv_arr`` / ``demanda_arr``: arrays paralelos a ids.
    - ``encoding``: SearchEncoding canónico de la instancia (etiquetas <-> ids).
    - ``backend_solicitado`` / ``backend_real``: trazabilidad GPU vs CPU.
    - ``dist_gpu``: copia en GPU (None si backend real es CPU).
    """

    encoding: SearchEncoding
    dist: np.ndarray
    u_arr: np.ndarray
    v_arr: np.ndarray
    costo_serv_arr: np.ndarray
    demanda_arr: np.ndarray
    depot: int
    marcador_depot: str
    backend_solicitado: str
    backend_real: str
    dist_gpu: Any | None = None

    @property
    def usar_gpu(self) -> bool:
        return self.backend_real == "gpu" and self.dist_gpu is not None


def _matriz_dijkstra_densa(
    dijkstra: Any,
    *,
    G: nx.Graph | None = None,
) -> np.ndarray:
    """
    Convierte el dict Dijkstra (``{u: {v: dist}}``) o cualquier mapping anidado
    a un ``np.ndarray`` denso 1-indexed.

    Si ``dijkstra`` es ``None`` y se proporciona ``G``, se computa con
    NetworkX (``all_pairs_dijkstra_path_length``).
    """
    if dijkstra is None:
        if G is None:
            raise ValueError("Falta dijkstra y G para reconstruir distancias.")
        # Computamos APSP una sola vez por instancia.
        nodos = sorted(int(n) for n in G.nodes())
        idx_max = max(nodos)
        D = np.full((idx_max + 1, idx_max + 1), _INF, dtype=np.float64)
        for u_str in G.nodes():
            length = nx.single_source_dijkstra_path_length(G, u_str, weight="cost")
            u = int(u_str)
            for v_str, d in length.items():
                D[u, int(v_str)] = float(d)
        return D

    if isinstance(dijkstra, np.ndarray):
        # Asumimos 1-indexed listo. Si fuera 0-indexed igual funciona si el
        # usuario consistentemente referencia nodos en ese rango.
        return dijkstra.astype(np.float64, copy=False)

    if isinstance(dijkstra, Mapping):
        keys = list(dijkstra.keys())
        try:
            idx_max = max(int(k) for k in keys)
            for k, fila in dijkstra.items():
                if isinstance(fila, Mapping):
                    if fila:
                        idx_max = max(idx_max, max(int(j) for j in fila.keys()))
        except (TypeError, ValueError) as exc:
            raise ValueError("Las claves de la matriz Dijkstra deben ser enteros.") from exc

        D = np.full((idx_max + 1, idx_max + 1), _INF, dtype=np.float64)
        for k, fila in dijkstra.items():
            i = int(k)
            if isinstance(fila, Mapping):
                for j, d in fila.items():
                    D[i, int(j)] = float(d)
            else:
                # fila tipo array indexado por entero.
                for j, d in enumerate(fila):
                    D[i, j] = float(d)
        return D

    raise TypeError(f"Formato de matriz Dijkstra no soportado: {type(dijkstra).__name__}")


def construir_contexto(
    data: Mapping[str, Any],
    *,
    dijkstra: Any | None = None,
    G: nx.Graph | None = None,
    usar_gpu: bool = False,
    encoding: SearchEncoding | None = None,
) -> ContextoEvaluacion:
    """
    Construye un contexto reutilizable de evaluación.

    Si ``dijkstra`` es ``None`` y se pasa ``G``, computa APSP con NetworkX una
    sola vez. Cuando ``usar_gpu=True`` y CuPy está disponible, copia la matriz
    a GPU para evaluación por lotes; si no, fallback a CPU.
    """
    enc = encoding or build_search_encoding(data)
    mapa = construir_mapa_tareas_por_etiqueta(data)
    if not mapa:
        raise ValueError("La instancia no tiene tareas (LISTA_ARISTAS_REQ/NOREQ vacías).")

    D = _matriz_dijkstra_densa(dijkstra, G=G)

    n = len(enc.id_to_label)
    u_arr = np.asarray(enc.u, dtype=np.int64)
    v_arr = np.asarray(enc.v, dtype=np.int64)
    costo_serv_arr = np.asarray(enc.costo_serv, dtype=np.float64)
    demanda_arr = np.asarray(enc.demanda, dtype=np.float64)

    if u_arr.shape[0] != n or v_arr.shape[0] != n:
        raise ValueError("Tamaño de arrays de tareas incoherente con el encoding.")

    depot = int(data.get("DEPOSITO", 1))
    marcador_depot = str(
        data.get("MARCADOR_DEPOT_ETIQUETA") or CLAVE_MARCADOR_DEPOSITO_DEFAULT
    ).strip().upper() or CLAVE_MARCADOR_DEPOSITO_DEFAULT

    backend_solicitado = "gpu" if usar_gpu else "cpu"
    backend_real = "cpu"
    dist_gpu: Any | None = None
    if usar_gpu and gpu_disponible():
        try:
            import cupy as cp  # type: ignore

            dist_gpu = cp.asarray(D)
            backend_real = "gpu"
        except Exception:  # noqa: BLE001
            backend_real = "cpu"
            dist_gpu = None

    return ContextoEvaluacion(
        encoding=enc,
        dist=D,
        u_arr=u_arr,
        v_arr=v_arr,
        costo_serv_arr=costo_serv_arr,
        demanda_arr=demanda_arr,
        depot=depot,
        marcador_depot=marcador_depot,
        backend_solicitado=backend_solicitado,
        backend_real=backend_real,
        dist_gpu=dist_gpu,
    )


def construir_contexto_desde_instancia(
    nombre_instancia: str,
    *,
    root: str | os.PathLike[str] | None = None,
    usar_gpu: bool = False,
) -> ContextoEvaluacion:
    """
    Construye el contexto cargando matriz Dijkstra + datos por nombre.
    Cachea por (nombre, backend) para evitar recomputar en grid search.
    """
    from .instances import load_instances

    backend = "gpu" if usar_gpu else "cpu"
    cache_key = (nombre_instancia, backend)
    cached = _CACHE_CONTEXTO.get(cache_key)
    if cached is not None:
        return cached

    data = load_instances(nombre_instancia, root=root)
    try:
        dijkstra = cargar_matriz_dijkstra(nombre_instancia, root=root)
    except FileNotFoundError:
        # Si no hay matriz precomputada, caemos a NetworkX una vez.
        from .cargar_grafos import cargar_objeto_gexf

        G = cargar_objeto_gexf(nombre_instancia, root=root)
        ctx = construir_contexto(data, dijkstra=None, G=G, usar_gpu=usar_gpu)
    else:
        ctx = construir_contexto(data, dijkstra=dijkstra, usar_gpu=usar_gpu)

    _CACHE_CONTEXTO[cache_key] = ctx
    return ctx


# ---------------------------------------------------------------------------
# Evaluadores rápidos
# ---------------------------------------------------------------------------


def _ruta_labels_a_ids(
    ruta: Sequence[Hashable],
    label_to_id: Mapping[str, int],
    marcador_depot_upper: str,
) -> list[int]:
    """Convierte una ruta por etiquetas a ids enteros (omite depot tokens)."""
    ids: list[int] = []
    for tok in ruta:
        s = str(tok).strip()
        if not s:
            continue
        if s.upper() == marcador_depot_upper:
            continue
        # Coincidencia rápida: directa o por mayúsculas.
        idx = label_to_id.get(s)
        if idx is None:
            idx = label_to_id.get(s.upper())
            if idx is None:
                # Búsqueda case-insensitive como último recurso.
                for k, vid in label_to_id.items():
                    if k.upper() == s.upper():
                        idx = vid
                        break
        if idx is None:
            raise KeyError(f"Etiqueta de tarea desconocida: {tok!r}")
        ids.append(idx)
    return ids


def costo_rapido_ids(
    solucion_ids: Sequence[Sequence[int]],
    ctx: ContextoEvaluacion,
) -> float:
    """
    Costo total de una solución dada por listas de ids (sin depot tokens).

    Equivalente exacto a ``costo_solucion`` para la fórmula:
    sum_rutas[ dist(depot, u_first) + sum_pares(dist(v_prev, u_curr) + costo_serv) +
               costo_serv_first + dist(v_last, depot) ].

    Implementación vectorizada: por cada ruta arma un array de origenes y
    destinos y resuelve con fancy-indexing en la matriz densa.
    """
    dist = ctx.dist
    u_arr = ctx.u_arr
    v_arr = ctx.v_arr
    cs_arr = ctx.costo_serv_arr
    depot = ctx.depot

    total = 0.0
    for ruta in solucion_ids:
        if not ruta:
            continue
        ids = np.asarray(ruta, dtype=np.int64)
        us = u_arr[ids]
        vs = v_arr[ids]
        # Origen previo: depot, luego v_prev de la tarea anterior.
        origen_prev = np.empty_like(us)
        origen_prev[0] = depot
        if us.shape[0] > 1:
            origen_prev[1:] = vs[:-1]
        # Suma DH por tarea (con dist=0 cuando coincide nodo).
        dh = dist[origen_prev, us]
        # Si origen_prev == us, el camino mínimo es 0 (la matriz lo refleja).
        total += float(dh.sum()) + float(cs_arr[ids].sum())
        # Regreso al depot desde el último v.
        total += float(dist[vs[-1], depot])
    return total


def costo_rapido(
    solucion_labels: Sequence[Sequence[Hashable]],
    ctx: ContextoEvaluacion,
) -> float:
    """
    Costo total de una solución por etiquetas (acepta ``D`` como marcador).

    Codifica labels -> ids con el mapeo del contexto (O(longitud)) y delega
    en ``costo_rapido_ids``.
    """
    md = ctx.marcador_depot.upper()
    label_to_id = ctx.encoding.label_to_id
    rutas_ids: list[list[int]] = []
    for ruta in solucion_labels:
        rutas_ids.append(_ruta_labels_a_ids(ruta, label_to_id, md))
    return costo_rapido_ids(rutas_ids, ctx)


# ---------------------------------------------------------------------------
# Evaluación por lotes (GPU opcional)
# ---------------------------------------------------------------------------


def _empaquetar_lote_ids(
    soluciones_ids: Sequence[Sequence[Sequence[int]]],
    ctx: ContextoEvaluacion,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Empaqueta un lote heterogéneo de soluciones (lista de rutas) a una matriz
    plana lista para evaluación vectorizada:

    Devuelve:
    - ``orig``: array (n_pasos,) con nodos origen previos.
    - ``dest``: array (n_pasos,) con nodos destino (u de cada tarea).
    - ``cs``:   array (n_pasos,) con costos de servicio (incluye también el
                tramo de regreso al depot, cuyo cs=0).
    - ``sol_idx``: array (n_pasos,) con el índice de la solución a la que
                pertenece cada paso (para reducción por solución).
    """
    u_arr = ctx.u_arr
    v_arr = ctx.v_arr
    cs_arr = ctx.costo_serv_arr
    depot = ctx.depot

    origs: list[int] = []
    dests: list[int] = []
    cs_l: list[float] = []
    sol_idx: list[int] = []

    for s_idx, sol in enumerate(soluciones_ids):
        for ruta in sol:
            if not ruta:
                continue
            ids = np.asarray(ruta, dtype=np.int64)
            us = u_arr[ids]
            vs = v_arr[ids]
            n = us.shape[0]
            # Pasos de servicio: origen previo -> u, costo_serv añadido aparte.
            origs.append(depot)
            origs.extend(vs[:-1].tolist() if n > 1 else [])
            dests.extend(us.tolist())
            cs_l.extend(cs_arr[ids].tolist())
            sol_idx.extend([s_idx] * n)
            # Paso de regreso al depot: cs=0.
            origs.append(int(vs[-1]))
            dests.append(depot)
            cs_l.append(0.0)
            sol_idx.append(s_idx)

    return (
        np.asarray(origs, dtype=np.int64),
        np.asarray(dests, dtype=np.int64),
        np.asarray(cs_l, dtype=np.float64),
        np.asarray(sol_idx, dtype=np.int64),
    )


def costo_lote_ids(
    soluciones_ids: Sequence[Sequence[Sequence[int]]],
    ctx: ContextoEvaluacion,
) -> np.ndarray:
    """
    Evalúa un lote de soluciones (todas con el mismo encoding) y devuelve
    un ``np.ndarray`` (n_soluciones,) con el costo total de cada una.

    Si el contexto fue construido con backend GPU real, la reducción se hace
    en GPU; el resultado regresa a host como NumPy.
    """
    n_sol = len(soluciones_ids)
    if n_sol == 0:
        return np.zeros((0,), dtype=np.float64)

    orig, dest, cs, sol_idx = _empaquetar_lote_ids(soluciones_ids, ctx)
    if orig.size == 0:
        return np.zeros((n_sol,), dtype=np.float64)

    if ctx.usar_gpu:
        import cupy as cp  # type: ignore

        d_gpu = ctx.dist_gpu
        orig_g = cp.asarray(orig)
        dest_g = cp.asarray(dest)
        cs_g = cp.asarray(cs)
        sol_g = cp.asarray(sol_idx)
        contrib = d_gpu[orig_g, dest_g] + cs_g
        out = cp.zeros((n_sol,), dtype=cp.float64)
        # bincount-like: suma por sol_idx.
        cp.scatter_add(out, sol_g, contrib)
        return cp.asnumpy(out)

    contrib = ctx.dist[orig, dest] + cs
    out = np.zeros((n_sol,), dtype=np.float64)
    np.add.at(out, sol_idx, contrib)
    return out
