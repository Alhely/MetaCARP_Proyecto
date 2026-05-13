"""
Codificación entera de tareas para metaheurísticas CARP.

Convierte etiquetas de texto (e.g. "TR1", "TNR2") a identificadores enteros
consecutivos (0, 1, 2, …). El acceso a listas por índice entero es O(1), lo
que hace a los operadores de vecindario significativamente más rápidos que la
búsqueda por clave en diccionarios de texto.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Hashable, Sequence

from .solucion_formato import (
    CLAVE_MARCADOR_DEPOSITO_DEFAULT,
    construir_mapa_tareas_por_etiqueta,
    resolver_etiqueta_canonica,
)

__all__ = [
    "DEPOT_ID",
    "SearchEncoding",
    "build_search_encoding",
    "encode_solution",
    "decode_solution",
    "decode_task_ids",
]

# Centinela para el depósito en la codificación entera.
# -1 garantiza que nunca colisione con IDs de tareas (0, 1, 2, …).
DEPOT_ID = -1


@dataclass(frozen=True, slots=True)
class SearchEncoding:
    """
    Codificación compacta por instancia para búsqueda/metaheurísticas.

    Convierte las etiquetas de texto de las tareas CARP (como 'TR1', 'TNR3')
    a identificadores enteros consecutivos (0, 1, 2, …) para que los operadores
    de vecindario puedan usar indexación de arrays en lugar de búsquedas en
    diccionarios de texto.

    - ``label_to_id`` / ``id_to_label``: mapeo biyectivo entre etiquetas e IDs.
    - ``u``, ``v``: nodo de inicio y fin de cada arista-tarea, indexados por ID.
    - ``demanda``, ``costo_serv``: demanda y costo de servicio de cada tarea, por ID.
    - ``depot_marker``: token de texto que representa el depósito en las rutas.
    - ``depot_id``: valor centinela (-1) para el depósito en la codificación entera.
    """

    label_to_id: dict[str, int]
    id_to_label: list[str]
    u: list[int]
    v: list[int]
    demanda: list[float]
    costo_serv: list[float]
    depot_marker: str = CLAVE_MARCADOR_DEPOSITO_DEFAULT
    depot_id: int = DEPOT_ID

    def __len__(self) -> int:
        """Devuelve el número total de tareas codificadas en este encoding."""
        return len(self.id_to_label)

    def label_of(self, idx: int) -> str:
        """Devuelve la etiqueta de texto correspondiente al ID entero ``idx``."""
        return self.id_to_label[idx]

    def id_of(self, label: str) -> int:
        """Devuelve el ID entero correspondiente a la etiqueta de texto ``label``."""
        return self.label_to_id[label]


def build_search_encoding(
    data: Mapping[str, Any],
    *,
    marcador_depot: str | None = None,
) -> SearchEncoding:
    """
    Construye la codificación entera para acelerar operadores de vecindario.

    Lee las tareas de la instancia desde ``data`` y asigna a cada una un ID
    entero consecutivo (0, 1, 2, …). También extrae nodos extremos (u, v),
    demanda y costo de servicio en listas paralelas para acceso O(1).

    Args:
        data: Diccionario de datos de la instancia (resultado de ``load_instances``).
        marcador_depot: Token de texto para el depósito. Si es ``None``, se lee
            de ``data['MARCADOR_DEPOT_ETIQUETA']`` o se usa el valor por defecto.

    Returns:
        SearchEncoding inmutable con todos los arrays de tareas listos.

    Raises:
        ValueError: Si no hay tareas en la instancia o alguna tarea no tiene nodos.
    """
    mapa = construir_mapa_tareas_por_etiqueta(data)
    if not mapa:
        raise ValueError("No hay tareas para construir encoding (LISTA_ARISTAS_REQ/NOREQ vacías).")

    labels = list(mapa.keys())
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    id_to_label = labels[:]

    n = len(labels)
    u: list[int] = [0] * n
    v: list[int] = [0] * n
    demanda: list[float] = [0.0] * n
    costo_serv: list[float] = [0.0] * n

    for lab, idx in label_to_id.items():
        t = mapa[lab]
        nodos = t.get("nodos")
        if not nodos or len(nodos) != 2:
            raise ValueError(f"Tarea {lab!r}: falta par de nodos.")
        u[idx] = int(nodos[0])
        v[idx] = int(nodos[1])
        demanda[idx] = float(t.get("demanda", 0) or 0)
        costo_serv[idx] = float(t.get("costo", 0) or 0)

    if marcador_depot is None:
        raw = data.get("MARCADOR_DEPOT_ETIQUETA")
        depot_marker = str(raw).strip().upper() if raw is not None else CLAVE_MARCADOR_DEPOSITO_DEFAULT
    else:
        depot_marker = str(marcador_depot).strip().upper()

    if not depot_marker:
        depot_marker = CLAVE_MARCADOR_DEPOSITO_DEFAULT

    return SearchEncoding(
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        u=u,
        v=v,
        demanda=demanda,
        costo_serv=costo_serv,
        depot_marker=depot_marker,
    )


def encode_solution(
    solucion_labels: Sequence[Sequence[Hashable]],
    encoding: SearchEncoding,
    *,
    permitir_deposito: bool = True,
) -> list[list[int]]:
    """
    Convierte una solución por etiquetas a listas de IDs enteros.

    Transforma, por ejemplo:
        [['D', 'TR1', 'TR3', 'D'], ['D', 'TR2', 'D']]
    en:
        [[0, 2], [1]]

    El marcador de depósito se elimina; los algoritmos de búsqueda trabajan
    solo con las tareas reales.

    Args:
        solucion_labels: Lista de rutas, cada ruta es una lista de etiquetas.
        encoding: Codificación entera de la instancia.
        permitir_deposito: Si True, el token depósito se omite sin error.

    Returns:
        Lista de listas de IDs enteros (sin tokens de depósito).

    Raises:
        ValueError: Si hay etiquetas vacías o desconocidas.
    """
    rutas_ids: list[list[int]] = []
    md = encoding.depot_marker.upper()
    # mapa_dummy con solo las claves válidas, para que resolver_etiqueta_canonica
    # pueda normalizar alias de etiqueta (ej. "tr1" → "TR1").
    mapa_dummy = {k: {} for k in encoding.label_to_id}

    for r_idx, ruta in enumerate(solucion_labels):
        out: list[int] = []
        for x in ruta:
            s = str(x).strip()
            if not s:
                raise ValueError(f"Ruta {r_idx}: elemento vacío.")
            if permitir_deposito and s.upper() == md:
                continue
            can = resolver_etiqueta_canonica(s, mapa_dummy)
            if can is None:
                raise ValueError(f"Ruta {r_idx}: etiqueta de tarea desconocida {s!r}.")
            out.append(encoding.label_to_id[can])
        rutas_ids.append(out)
    return rutas_ids


def decode_solution(
    solucion_ids: Sequence[Sequence[int]],
    encoding: SearchEncoding,
    *,
    con_deposito: bool = True,
) -> list[list[str]]:
    """
    Convierte una solución por IDs enteros a etiquetas de texto.

    Operación inversa a :func:`encode_solution`. Transforma, por ejemplo:
        [[0, 2], [1]]
    en:
        [['D', 'TR1', 'TR3', 'D'], ['D', 'TR2', 'D']]

    Args:
        solucion_ids: Lista de rutas con IDs enteros de tareas.
        encoding: Codificación entera de la instancia.
        con_deposito: Si True, envuelve cada ruta con el marcador de depósito.

    Raises:
        ValueError: Si se encuentra un depot_id inesperado o un ID fuera de rango.
    """
    rutas_labels: list[list[str]] = []
    for r_idx, ruta in enumerate(solucion_ids):
        fila: list[str] = []
        if con_deposito:
            fila.append(encoding.depot_marker)
        for idx in ruta:
            if idx == encoding.depot_id:
                if con_deposito:
                    continue
                raise ValueError(
                    f"Ruta {r_idx}: depot_id ({encoding.depot_id}) inesperado cuando con_deposito=False."
                )
            if idx < 0 or idx >= len(encoding.id_to_label):
                raise ValueError(f"Ruta {r_idx}: id de tarea inválido {idx}.")
            fila.append(encoding.id_to_label[idx])
        if con_deposito:
            fila.append(encoding.depot_marker)
        rutas_labels.append(fila)
    return rutas_labels


def decode_task_ids(ids: Sequence[int], encoding: SearchEncoding) -> list[str]:
    """
    Devuelve las etiquetas de texto para una lista plana de IDs enteros.

    A diferencia de :func:`decode_solution`, trabaja con una sola lista (no
    anidada) y omite silenciosamente el ``depot_id`` si aparece.

    Raises:
        ValueError: Si un ID no es ni depot_id ni un ID de tarea válido.
    """
    out: list[str] = []
    for idx in ids:
        if idx == encoding.depot_id:
            continue
        if idx < 0 or idx >= len(encoding.id_to_label):
            raise ValueError(f"id de tarea inválido {idx}.")
        out.append(encoding.id_to_label[idx])
    return out
