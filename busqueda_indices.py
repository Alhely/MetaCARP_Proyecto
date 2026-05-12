# Módulo: busqueda_indices.py
# Propósito: Define la *codificación entera* de tareas que usan las metaheurísticas
# (Búsqueda Tabú, Abejas, Cuckoo Search) para operar con IDs numéricos en lugar de
# etiquetas de texto como "TR1" o "TNR2". Trabajar con enteros hace que los operadores
# de vecindario sean mucho más rápidos, porque el acceso a listas/arrays por índice
# es O(1) (tiempo constante) en Python.

from __future__ import annotations  # Permite usar tipos como list[int] en Python < 3.10

# 'dataclass' es un decorador que genera automáticamente __init__, __repr__, etc.
# 'field' permite personalizar atributos de un dataclass con valores por defecto complejos.
from dataclasses import dataclass

# Tipos de Python para anotaciones de tipo (mejoran la legibilidad y el autocompletado).
# 'Any' = cualquier tipo, 'Hashable' = cualquier valor que pueda ser clave de dict,
# 'Mapping' = cualquier objeto tipo diccionario (dict, OrderedDict…),
# 'Sequence' = cualquier secuencia ordenada (list, tuple…).
from collections.abc import Mapping
from typing import Any, Hashable, Sequence

# Importamos constantes y funciones del módulo que define el formato estándar de
# etiquetas y rutas en todo el proyecto (por ejemplo, "D" como marcador de depósito).
from .solucion_formato import (
    CLAVE_MARCADOR_DEPOSITO_DEFAULT,       # Valor por defecto del token depósito (e.g. "D")
    construir_mapa_tareas_por_etiqueta,    # Crea un dict {etiqueta: datos_tarea}
    resolver_etiqueta_canonica,            # Normaliza una etiqueta a su forma canónica
)

# __all__ lista los nombres que se exportan cuando alguien hace "from busqueda_indices import *".
# Es buena práctica declararlo explícitamente para controlar la API pública del módulo.
__all__ = [
    "DEPOT_ID",
    "SearchEncoding",
    "build_search_encoding",
    "encode_solution",
    "decode_solution",
    "decode_task_ids",
]

# Constante especial que representa el depósito en la codificación entera.
# Se usa -1 para que nunca colisione con los IDs de tareas (que son 0, 1, 2, …).
DEPOT_ID = -1


# ---------------------------------------------------------------------------
# Clase de datos: SearchEncoding
# ---------------------------------------------------------------------------
# En Python, un "dataclass" es una clase cuya responsabilidad principal es
# agrupar datos relacionados. El decorador @dataclass genera automáticamente
# métodos como __init__ (constructor) y __repr__ (representación en texto).
#
# frozen=True  → el objeto es INMUTABLE después de crearse (como una tupla).
#               Intentar asignar ctx.depot_id = 5 lanzará un FrozenInstanceError.
#               Esto es seguro para compartir entre hilos y evita mutaciones accidentales.
#
# slots=True   → Python reserva espacio fijo por atributo en lugar de usar un dict
#               interno (__dict__). Resultado: menos memoria y acceso más rápido.
@dataclass(frozen=True, slots=True)
class SearchEncoding:
    """
    Codificación compacta por instancia para búsqueda/metaheurísticas.

    Convierte las etiquetas de texto de las tareas CARP (como 'TR1', 'TNR3')
    a identificadores enteros consecutivos (0, 1, 2, …) para que los operadores
    de vecindario puedan usar indexación de arrays en lugar de búsquedas en
    diccionarios de texto.

    - ``label_to_id`` / ``id_to_label``: mapeo biyectivo (en ambas direcciones)
      entre etiquetas de texto e IDs enteros.
    - ``u``, ``v``: nodo de inicio y nodo de fin de cada arista-tarea, indexados
      por ID de tarea para acceso O(1).
    - ``demanda``, ``costo_serv``: demanda y costo de servicio de cada tarea,
      también indexados por ID.
    - ``depot_marker``: token de texto que representa el depósito en las rutas
      (por defecto "D"). Se omite al codificar a enteros.
    - ``depot_id``: valor centinela (-1) para el depósito en la codificación entera.
    """

    # dict que mapea etiqueta de texto → ID entero. Ejemplo: {"TR1": 0, "TR2": 1, …}
    label_to_id: dict[str, int]

    # list que mapea ID entero → etiqueta de texto. El índice es el ID. Ejemplo: ["TR1", "TR2", …]
    id_to_label: list[str]

    # Nodo de inicio (u) de cada tarea, indexado por ID de tarea.
    # Ejemplo: u[0] = 3 significa que la tarea 0 comienza en el nodo 3 del grafo.
    u: list[int]

    # Nodo de fin (v) de cada tarea, indexado por ID de tarea.
    v: list[int]

    # Demanda de cada tarea, indexada por ID de tarea.
    demanda: list[float]

    # Costo de servicio (traversal cost) de cada tarea, indexado por ID.
    costo_serv: list[float]

    # Token de texto que marca el depósito en las listas de rutas (por defecto "D").
    depot_marker: str = CLAVE_MARCADOR_DEPOSITO_DEFAULT

    # ID entero centinela para el depósito. Siempre -1 para distinguirlo de tareas reales.
    depot_id: int = DEPOT_ID

    def __len__(self) -> int:
        """Devuelve el número total de tareas codificadas en este encoding."""
        # len(encoding) da directamente cuántas tareas hay en la instancia.
        return len(self.id_to_label)

    def label_of(self, idx: int) -> str:
        """Devuelve la etiqueta de texto correspondiente al ID entero ``idx``."""
        # Acceso directo O(1) a la lista: igual que leer una celda de una tabla.
        return self.id_to_label[idx]

    def id_of(self, label: str) -> int:
        """Devuelve el ID entero correspondiente a la etiqueta de texto ``label``."""
        # Búsqueda O(1) en el diccionario inverso.
        return self.label_to_id[label]


# ---------------------------------------------------------------------------
# Función: build_search_encoding
# ---------------------------------------------------------------------------
def build_search_encoding(
    data: Mapping[str, Any],      # Datos completos de la instancia (leídos del pickle)
    *,
    marcador_depot: str | None = None,  # Permite sobreescribir el token de depósito
) -> SearchEncoding:
    """
    Construye la codificación entera para acelerar operadores de vecindario.

    Lee las tareas de la instancia desde ``data`` y asigna a cada una un ID
    entero consecutivo (0, 1, 2, …). También extrae los nodos extremos (u, v),
    demanda y costo de servicio de cada tarea en listas paralelas para acceso O(1).

    Args:
        data: Diccionario de datos de la instancia (resultado de ``load_instances``).
        marcador_depot: Token de texto para el depósito. Si es ``None``, se lee
            de ``data['MARCADOR_DEPOT_ETIQUETA']`` o se usa el valor por defecto.

    Returns:
        SearchEncoding inmutable con todos los arrays de tareas listos.

    Raises:
        ValueError: Si no hay tareas en la instancia o alguna tarea no tiene nodos.
    """
    # Construimos el diccionario {etiqueta: datos_tarea} desde los datos de la instancia.
    # Incluye tanto aristas requeridas (LISTA_ARISTAS_REQ) como no requeridas (NOREQ).
    mapa = construir_mapa_tareas_por_etiqueta(data)
    if not mapa:
        raise ValueError("No hay tareas para construir encoding (LISTA_ARISTAS_REQ/NOREQ vacías).")

    # Extraemos las etiquetas en orden determinista (el orden del dict en Python 3.7+ es FIFO).
    labels = list(mapa.keys())

    # Creamos el mapeo biyectivo:
    # label_to_id: {"TR1": 0, "TR2": 1, …}  ← dict comprensión con enumerate
    label_to_id = {lab: i for i, lab in enumerate(labels)}

    # id_to_label: ["TR1", "TR2", …]  ← copia de la lista para que sea independiente
    id_to_label = labels[:]

    # Preasignamos listas con ceros del tamaño correcto (evita append en bucle).
    # Estas listas serán los "arrays paralelos" indexados por ID de tarea.
    u: list[int] = [0] * len(labels)
    v: list[int] = [0] * len(labels)
    demanda: list[float] = [0.0] * len(labels)
    costo_serv: list[float] = [0.0] * len(labels)

    # Rellenamos cada posición con los datos de la tarea correspondiente.
    for lab, idx in label_to_id.items():
        t = mapa[lab]  # Diccionario con los datos de esta tarea (nodos, demanda, costo…)

        nodos = t.get("nodos")
        if not nodos or len(nodos) != 2:
            # Toda tarea CARP debe tener exactamente dos nodos extremos (arco dirigido u→v).
            raise ValueError(f"Tarea {lab!r}: falta par de nodos.")

        # Guardamos los nodos extremos como enteros en las posiciones correspondientes.
        u[idx] = int(nodos[0])
        v[idx] = int(nodos[1])

        # 'or 0' convierte None o valores ausentes a 0 antes de convertir a float.
        demanda[idx] = float(t.get("demanda", 0) or 0)
        costo_serv[idx] = float(t.get("costo", 0) or 0)

    # Determinamos el token de depósito que se usará en las rutas de texto.
    # Prioridad: argumento explícito > campo en data > valor por defecto.
    if marcador_depot is None:
        raw = data.get("MARCADOR_DEPOT_ETIQUETA")
        # Si el campo existe en data, lo normalizamos (strip + uppercase).
        depot_marker = str(raw).strip().upper() if raw is not None else CLAVE_MARCADOR_DEPOSITO_DEFAULT
    else:
        depot_marker = str(marcador_depot).strip().upper()

    # Si después de todo sigue vacío, usamos el valor por defecto del proyecto.
    if not depot_marker:
        depot_marker = CLAVE_MARCADOR_DEPOSITO_DEFAULT

    # Construimos y devolvemos el SearchEncoding inmutable.
    # Como es frozen=True, Python no dejará modificarlo después de esta línea.
    return SearchEncoding(
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        u=u,
        v=v,
        demanda=demanda,
        costo_serv=costo_serv,
        depot_marker=depot_marker,
        # depot_id usa el valor por defecto DEPOT_ID = -1
    )


# ---------------------------------------------------------------------------
# Función: encode_solution
# ---------------------------------------------------------------------------
def encode_solution(
    solucion_labels: Sequence[Sequence[Hashable]],  # Rutas con etiquetas de texto
    encoding: SearchEncoding,                        # Encoding de la instancia
    *,
    permitir_deposito: bool = True,  # Si True, el token depósito se omite silenciosamente
) -> list[list[int]]:
    """
    Convierte una solución por etiquetas a listas de IDs enteros.

    Transforma, por ejemplo:
        [['D', 'TR1', 'TR3', 'D'], ['D', 'TR2', 'D']]
    en:
        [[0, 2], [1]]

    El marcador de depósito (por defecto 'D') se elimina en la codificación,
    ya que los algoritmos de búsqueda trabajan solo con las tareas reales.

    Args:
        solucion_labels: Lista de rutas, cada ruta es una lista de etiquetas.
        encoding: Codificación entera de la instancia.
        permitir_deposito: Si True, el token depósito se omite sin error.

    Returns:
        Lista de listas de IDs enteros (sin tokens de depósito).

    Raises:
        ValueError: Si hay etiquetas vacías o desconocidas.
    """
    rutas_ids: list[list[int]] = []  # Resultado acumulado

    # Versión en mayúsculas del marcador de depósito para comparación case-insensitive.
    md = encoding.depot_marker.upper()

    # Creamos un mapeo auxiliar vacío solo con las claves válidas del encoding.
    # Se usa para llamar a resolver_etiqueta_canonica, que necesita un mapa de referencia.
    mapa_dummy = {k: {} for k in encoding.label_to_id}

    for r_idx, ruta in enumerate(solucion_labels):
        out: list[int] = []  # IDs de esta ruta
        for x in ruta:
            s = str(x).strip()  # Convertimos a texto y eliminamos espacios sobrantes

            if not s:
                raise ValueError(f"Ruta {r_idx}: elemento vacío.")

            # Si el elemento es el token depósito, lo saltamos (no es una tarea).
            if permitir_deposito and s.upper() == md:
                continue

            # Normalizamos la etiqueta a su forma canónica (ej. "tr1" → "TR1").
            # Esto garantiza que la búsqueda en label_to_id funcione siempre.
            can = resolver_etiqueta_canonica(s, mapa_dummy)
            if can is None:
                raise ValueError(f"Ruta {r_idx}: etiqueta de tarea desconocida {s!r}.")

            # Traducimos la etiqueta canónica a su ID entero.
            out.append(encoding.label_to_id[can])
        rutas_ids.append(out)
    return rutas_ids


# ---------------------------------------------------------------------------
# Función: decode_solution
# ---------------------------------------------------------------------------
def decode_solution(
    solucion_ids: Sequence[Sequence[int]],  # Rutas con IDs enteros
    encoding: SearchEncoding,               # Encoding de la instancia
    *,
    con_deposito: bool = True,  # Si True, añade el token depósito al inicio y final
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

    Returns:
        Lista de listas de etiquetas de texto.

    Raises:
        ValueError: Si se encuentra un depot_id inesperado o un ID fuera de rango.
    """
    rutas_labels: list[list[str]] = []  # Resultado acumulado

    for r_idx, ruta in enumerate(solucion_ids):
        fila: list[str] = []

        # Añadimos el token depósito al principio de la ruta si se solicitó.
        if con_deposito:
            fila.append(encoding.depot_marker)

        for idx in ruta:
            # Si encontramos el centinela de depósito en medio de la ruta…
            if idx == encoding.depot_id:
                if con_deposito:
                    # …lo ignoramos (ya lo añadimos manualmente al inicio/final).
                    continue
                raise ValueError(
                    f"Ruta {r_idx}: depot_id ({encoding.depot_id}) inesperado cuando con_deposito=False."
                )

            # Verificamos que el ID esté dentro del rango válido de tareas.
            if idx < 0 or idx >= len(encoding.id_to_label):
                raise ValueError(f"Ruta {r_idx}: id de tarea inválido {idx}.")

            # Traducimos el ID entero a su etiqueta de texto.
            fila.append(encoding.id_to_label[idx])

        # Añadimos el token depósito al final de la ruta si se solicitó.
        if con_deposito:
            fila.append(encoding.depot_marker)

        rutas_labels.append(fila)
    return rutas_labels


# ---------------------------------------------------------------------------
# Función: decode_task_ids
# ---------------------------------------------------------------------------
def decode_task_ids(ids: Sequence[int], encoding: SearchEncoding) -> list[str]:
    """
    Devuelve las etiquetas de texto para una lista plana de IDs enteros.

    A diferencia de :func:`decode_solution`, trabaja con una sola lista (no
    anidada) y omite silenciosamente el ``depot_id`` si aparece.

    Args:
        ids: Lista de IDs enteros de tareas (puede contener depot_id para ser ignorado).
        encoding: Codificación entera de la instancia.

    Returns:
        Lista de etiquetas de texto (sin el marcador de depósito).

    Raises:
        ValueError: Si un ID no es ni depot_id ni un ID de tarea válido.
    """
    out: list[str] = []
    for idx in ids:
        # Saltamos el centinela de depósito (-1) silenciosamente.
        if idx == encoding.depot_id:
            continue

        # Verificamos que el ID sea válido antes de indexar la lista.
        if idx < 0 or idx >= len(encoding.id_to_label):
            raise ValueError(f"id de tarea inválido {idx}.")

        out.append(encoding.id_to_label[idx])
    return out
