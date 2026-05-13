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
    "costo_lote_penalizado_ids",
    "exceso_capacidad_rapido",
    "objectivo_penalizado",
    "lambda_penal_capacidad_por_defecto",
    "gpu_disponible",
]

_INF = np.float64("inf")

# Contextos ya construidos por (nombre_instancia, backend) para evitar
# recomputar la matriz Dijkstra en múltiples corridas del mismo proceso.
_CACHE_CONTEXTO: dict[tuple[str, str | None], "ContextoEvaluacion"] = {}


def gpu_disponible() -> bool:
    """
    Detecta si CuPy está instalado y hay al menos un dispositivo CUDA accesible.

    Returns:
        True si CuPy importa correctamente y hay un dispositivo CUDA disponible.
        False en cualquier otro caso (sin GPU, sin CuPy, error de driver…).
    """
    try:
        import cupy as cp  # type: ignore
        try:
            cp.cuda.runtime.getDeviceCount()
            return True
        except Exception:  # noqa: BLE001
            return False
    except Exception:  # noqa: BLE001
        return False


@dataclass(frozen=True, slots=True)
class ContextoEvaluacion:
    """
    Contexto inmutable y compartido para evaluar costos de soluciones CARP.

    Agrupa todos los datos pre-procesados que necesitan los evaluadores rápidos:
    la matriz de distancias, los arrays de tareas y la codificación entera.
    Se construye una vez por instancia y se reutiliza en todas las iteraciones.

    Atributos clave:
    - ``dist``: matriz NumPy densa (N+1 × N+1) con distancias precomputadas (1-indexed).
    - ``u_arr`` / ``v_arr``: nodos extremos de cada tarea, indexados por ID.
    - ``costo_serv_arr``: costo de servicio de cada tarea, indexado por ID.
    - ``demanda_arr``: demanda de cada tarea, indexada por ID.
    - ``encoding``: SearchEncoding canónico (etiquetas ↔ IDs) de la instancia.
    - ``depot``: número entero del nodo depósito en el grafo.
    - ``backend_solicitado`` / ``backend_real``: para trazabilidad CPU vs GPU.
    - ``dist_gpu``: copia de la matriz en GPU (None si se usa CPU).
    - ``capacidad_max``: capacidad máxima por vehículo (inf si no aplica).
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
    capacidad_max: float = float("inf")

    @property
    def usar_gpu(self) -> bool:
        """True si el backend real es GPU y la matriz está cargada en GPU."""
        return self.backend_real == "gpu" and self.dist_gpu is not None


def lambda_penal_capacidad_por_defecto(ctx: ContextoEvaluacion) -> float:
    """
    Calcula el factor lambda (λ) de penalización de capacidad para el objetivo penalizado.

    Calibrado automáticamente como ~10 veces la mediana del costo de los arcos
    de la instancia, para que escale apropiadamente con el tamaño del grafo.

    Returns:
        Valor de λ como float. Mínimo garantizado: 10.0.
    """
    D = ctx.dist
    fin = np.isfinite(D) & (D > 1e-12)
    if not np.any(fin):
        return 100.0
    mediana = float(np.median(D[fin]))
    return max(mediana * 10.0, 10.0)


def exceso_capacidad_sol_ids(
    solucion_ids: Sequence[Sequence[int]],
    ctx: ContextoEvaluacion,
) -> float:
    """
    Calcula el exceso total de demanda sobre la capacidad máxima de la solución.

    Para cada ruta: exceso = max(0, demanda_total_ruta - capacidad_max).
    Si ``capacidad_max`` no es finita o es 0, devuelve 0.

    Returns:
        Suma total de excesos de demanda (float ≥ 0).
    """
    cap = float(ctx.capacidad_max)
    if cap <= 0 or not np.isfinite(cap):
        return 0.0

    dem = ctx.demanda_arr
    total_exc = 0.0
    for ruta in solucion_ids:
        if not ruta:
            continue
        ids = np.asarray(ruta, dtype=np.int64)
        s = float(np.sum(dem[ids]))
        total_exc += max(0.0, s - cap)
    return total_exc


def exceso_capacidad_rapido(
    solucion_labels: Sequence[Sequence[Hashable]],
    ctx: ContextoEvaluacion,
) -> float:
    """
    Calcula el exceso de capacidad de una solución dada por etiquetas de texto.

    Versión de :func:`exceso_capacidad_sol_ids` que acepta rutas con etiquetas.

    Returns:
        Suma total de excesos de demanda (float ≥ 0).
    """
    md = ctx.marcador_depot.upper()
    label_to_id = ctx.encoding.label_to_id
    rutas_ids: list[list[int]] = [
        _ruta_labels_a_ids(ruta, label_to_id, md) for ruta in solucion_labels
    ]
    return exceso_capacidad_sol_ids(rutas_ids, ctx)


def objectivo_penalizado(
    costo_puro: float,
    violacion_cap: float,
    *,
    usar_penal: bool,
    lam: float,
) -> float:
    """
    Calcula el objetivo penalizado: ``costo_puro + λ × violación_capacidad``.

    Penalizar las violaciones permite al algoritmo explorar soluciones
    infactibles temporalmente mientras λ × violación las desincentiva.

    Args:
        costo_puro: Costo calculado sin penalización.
        violacion_cap: Exceso total de demanda sobre la capacidad.
        usar_penal: Si False, devuelve siempre el costo puro.
        lam: Factor de escala de la penalización (λ).
    """
    if not usar_penal or violacion_cap <= 0:
        return float(costo_puro)
    return float(costo_puro) + lam * float(violacion_cap)


def _matriz_dijkstra_densa(
    dijkstra: Any,
    *,
    G: nx.Graph | None = None,
) -> np.ndarray:
    """
    Convierte la matriz Dijkstra a un ``np.ndarray`` denso 1-indexed.

    Soporta tres formatos de entrada:
    1. ``None`` + grafo G: calcula APSP con NetworkX.
    2. ``np.ndarray``: devuelve el array tal cual.
    3. ``dict`` anidado {u: {v: dist}}: convierte al array denso.

    El resultado es una matriz 2D donde ``D[i, j]`` es el costo del camino
    mínimo del nodo i al nodo j (indexación 1-based).

    Raises:
        ValueError: Si dijkstra es None y G tampoco se proporciona.
        TypeError: Si el formato de dijkstra no es soportado.
    """
    if dijkstra is None:
        if G is None:
            raise ValueError("Falta dijkstra y G para reconstruir distancias.")
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
        return dijkstra.astype(np.float64, copy=False)

    if isinstance(dijkstra, Mapping):
        keys = list(dijkstra.keys())
        try:
            idx_max = max(int(k) for k in keys)
            for k, fila in dijkstra.items():
                if isinstance(fila, Mapping) and fila:
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
    Construye un contexto reutilizable de evaluación para una instancia CARP.

    Prepara todos los datos en formato NumPy para evaluación vectorizada O(1)
    por consulta de distancia. Este contexto se debe construir una sola vez y
    reutilizar en todas las iteraciones de la metaheurística.

    Si ``dijkstra`` es None y se pasa ``G``, computa APSP con NetworkX una sola vez.
    Cuando ``usar_gpu=True`` y CuPy está disponible, copia la matriz dist a GPU;
    si no, hace fallback transparente a CPU.

    Args:
        data: Diccionario de datos de la instancia.
        dijkstra: Matriz Dijkstra precomputada (dict o ndarray). Si None, se usa G.
        G: Grafo NetworkX de la instancia. Usado si dijkstra es None.
        usar_gpu: Si True, intenta usar CuPy/GPU para los arrays de distancias.
        encoding: SearchEncoding preexistente. Si None, se construye desde data.

    Returns:
        ContextoEvaluacion inmutable listo para evaluar soluciones.
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

    cap_raw = float(data.get("CAPACIDAD", 0) or 0)
    capacidad_max = cap_raw if cap_raw > 0 else float("inf")

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
        capacidad_max=capacidad_max,
    )


def construir_contexto_desde_instancia(
    nombre_instancia: str,
    *,
    root: str | os.PathLike[str] | None = None,
    usar_gpu: bool = False,
) -> ContextoEvaluacion:
    """
    Construye el contexto de evaluación cargando la instancia por nombre.

    Usa caché global para evitar recomputar la matriz Dijkstra en el mismo
    proceso, lo que es crítico en grid search con múltiples corridas.

    Args:
        nombre_instancia: Identificador de la instancia (ej. "EGL-E1-A").
        root: Directorio raíz donde buscar los archivos.
        usar_gpu: Si True, intenta construir el contexto con backend GPU.

    Returns:
        ContextoEvaluacion construido (posiblemente desde caché).
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
        from .cargar_grafos import cargar_objeto_gexf

        G = cargar_objeto_gexf(nombre_instancia, root=root)
        ctx = construir_contexto(data, dijkstra=None, G=G, usar_gpu=usar_gpu)
    else:
        ctx = construir_contexto(data, dijkstra=dijkstra, usar_gpu=usar_gpu)

    _CACHE_CONTEXTO[cache_key] = ctx
    return ctx


def _ruta_labels_a_ids(
    ruta: Sequence[Hashable],
    label_to_id: Mapping[str, int],
    marcador_depot_upper: str,
) -> list[int]:
    """
    Convierte una ruta de etiquetas de texto a una lista de IDs enteros.

    Omite silenciosamente el marcador de depósito. Busca la etiqueta primero
    de forma exacta, luego en mayúsculas, y finalmente con búsqueda lineal
    case-insensitive como último recurso.

    Raises:
        KeyError: Si una etiqueta no se encuentra en ningún formato.
    """
    ids: list[int] = []
    for tok in ruta:
        s = str(tok).strip()
        if not s:
            continue
        if s.upper() == marcador_depot_upper:
            continue

        idx = label_to_id.get(s)
        if idx is None:
            idx = label_to_id.get(s.upper())
        if idx is None:
            # Búsqueda lineal case-insensitive como último recurso (O(n), rara vez se ejecuta).
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
    Calcula el costo total de una solución dada por listas de IDs enteros.

    Es el evaluador principal de las metaheurísticas. Produce el mismo resultado
    que ``costo_solucion`` pero sin llamar a Dijkstra: usa la matriz dist
    precomputada para acceder a distancias en O(1).

    Fórmula por ruta:
        costo = dist(depot, u_0) + costo_serv_0
              + sum_{k=1}^{n-1} [ dist(v_{k-1}, u_k) + costo_serv_k ]
              + dist(v_{n-1}, depot)

    Implementación vectorizada: construye arrays de orígenes y destinos por ruta,
    luego extrae todas las distancias con fancy indexing en una sola operación.

    Returns:
        Costo total de la solución como float.
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

        origen_prev = np.empty_like(us)
        origen_prev[0] = depot
        if us.shape[0] > 1:
            origen_prev[1:] = vs[:-1]

        dh = dist[origen_prev, us]
        total += float(dh.sum()) + float(cs_arr[ids].sum())
        total += float(dist[vs[-1], depot])

    return total


def costo_rapido(
    solucion_labels: Sequence[Sequence[Hashable]],
    ctx: ContextoEvaluacion,
) -> float:
    """
    Calcula el costo total de una solución dada por etiquetas de texto.

    Acepta el formato estándar con marcador de depósito (ej. 'D').
    Convierte internamente las etiquetas a IDs y delega en :func:`costo_rapido_ids`.

    Returns:
        Costo total de la solución como float.
    """
    md = ctx.marcador_depot.upper()
    label_to_id = ctx.encoding.label_to_id
    rutas_ids: list[list[int]] = [
        _ruta_labels_a_ids(ruta, label_to_id, md) for ruta in solucion_labels
    ]
    return costo_rapido_ids(rutas_ids, ctx)


def _empaquetar_lote_ids(
    soluciones_ids: Sequence[Sequence[Sequence[int]]],
    ctx: ContextoEvaluacion,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Empaqueta un lote heterogéneo de soluciones en arrays planos para evaluación vectorizada.

    Las soluciones pueden tener diferente número de rutas y de tareas por ruta.
    Las "aplana" en arrays 1D paralelos, añadiendo un índice de solución para
    poder sumar los costos por solución con scatter_add al final.

    Returns:
        Tupla de 4 arrays NumPy de igual longitud (n_pasos_totales,):
        - ``orig``: nodo origen de cada paso.
        - ``dest``: nodo destino de cada paso.
        - ``cs``: costo de servicio de cada paso (0 para el regreso al depósito).
        - ``sol_idx``: índice de la solución a la que pertenece cada paso.
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

            origs.append(depot)
            origs.extend(vs[:-1].tolist() if n > 1 else [])
            dests.extend(us.tolist())
            cs_l.extend(cs_arr[ids].tolist())
            sol_idx.extend([s_idx] * n)

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
    Evalúa un lote completo de soluciones y devuelve un array de costos.

    Función estrella para metaheurísticas con población (Abejas, Cuckoo):
    evalúa todas las soluciones del lote en pocas operaciones NumPy/CuPy,
    evitando bucles Python por solución.

    Si el contexto usa backend GPU real, toda la reducción se realiza en GPU;
    el resultado regresa a memoria del host como NumPy.

    Returns:
        np.ndarray de shape (n_soluciones,) con el costo total de cada solución.
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

        try:
            cupyx_scatter = cp.scatter_add
        except AttributeError:
            import cupyx
            cupyx_scatter = cupyx.scatter_add

        cupyx_scatter(out, sol_g, contrib)
        return cp.asnumpy(out)

    contrib = ctx.dist[orig, dest] + cs
    out = np.zeros((n_sol,), dtype=np.float64)
    np.add.at(out, sol_idx, contrib)
    return out


def costo_lote_penalizado_ids(
    soluciones_ids: Sequence[Sequence[Sequence[int]]],
    ctx: ContextoEvaluacion,
    lam: float,
    *,
    usar_penal: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evalúa un lote con objetivo penalizado: ``costo_puro + λ × violación_capacidad``.

    Combina :func:`costo_lote_ids` con el cálculo de exceso de capacidad para
    producir el objetivo que guía la metaheurística cuando hay restricciones de
    capacidad activas.

    Args:
        soluciones_ids: Lote de soluciones, cada una como lista de rutas de IDs.
        ctx: Contexto de evaluación de la instancia.
        lam: Factor λ de penalización de capacidad.
        usar_penal: Si False, devuelve el costo puro sin penalización.

    Returns:
        Tupla de 3 arrays NumPy de shape (n_soluciones,):
        - ``objetivo``: costo_puro + λ × violación (el que minimiza la metaheurística).
        - ``costo_puro``: costo sin penalización.
        - ``violacion``: exceso total de capacidad de cada solución.
    """
    base = costo_lote_ids(soluciones_ids, ctx)
    n = len(soluciones_ids)

    if n == 0:
        z = np.zeros((0,), dtype=np.float64)
        return z, z, z

    if (
        not usar_penal
        or not np.isfinite(ctx.capacidad_max)
        or float(ctx.capacidad_max) <= 0
    ):
        z = np.zeros_like(base)
        return base.copy(), base, z

    exc = np.zeros((n,), dtype=np.float64)
    for i, sid in enumerate(soluciones_ids):
        exc[i] = exceso_capacidad_sol_ids(sid, ctx)

    obj = base + float(lam) * exc
    return obj, base, exc
