# =============================================================================
# solucion_formato.py — Conversión de soluciones entre formatos de etiquetas
#
# En CARP, una solución es un conjunto de rutas, donde cada ruta es una
# secuencia de tareas (aristas) que debe atender un vehículo. Las tareas se
# identifican con etiquetas de texto como:
#   - "TR1", "TR2", ...  (tareas requeridas, del inglés "Task Required")
#   - "TNR1", "TNR2", ... (tareas no requeridas, "Task Not Required")
#   - "D"                 (depósito: punto de inicio y fin de cada ruta)
#
# Este módulo proporciona funciones para:
#   1. Construir un diccionario rápido de etiqueta -> datos de arista.
#   2. Obtener el conjunto de etiquetas que son tareas requeridas.
#   3. Normalizar (limpiar) una solución: quitar marcadores de depósito y
#      validar que todas las etiquetas sean conocidas.
# =============================================================================

# Permite anotaciones de tipo como `str | None` en Python < 3.10
from __future__ import annotations

from typing import (
    Any,       # Cualquier tipo (usado cuando el tipo exacto no importa o varía)
    Hashable,  # Tipo abstracto: objetos que pueden usarse como clave de dict (str, int, etc.)
    Mapping,   # Tipo abstracto: cualquier cosa que se comporte como diccionario (solo lectura)
    Sequence,  # Tipo abstracto: cualquier secuencia ordenada (lista, tupla, etc.)
)

"""Formato de solución por etiquetas: ``TR*``/``TNR*`` con ``D`` = depósito."""

# Nombres públicos que este módulo expone al importarlo
__all__ = [
    "CLAVE_MARCADOR_DEPOSITO_DEFAULT",
    "construir_mapa_tareas_por_etiqueta",
    "etiquetas_tareas_requeridas",
    "normalizar_rutas_etiquetas",
    "resolver_etiqueta_canonica",
]

# Constante: cadena que identifica el depósito en una solución por etiquetas.
# Usar una constante en lugar del literal "D" permite cambiarla en un solo lugar
# si en algún proyecto se usa un marcador distinto.
CLAVE_MARCADOR_DEPOSITO_DEFAULT = "D"


# -----------------------------------------------------------------------------
# Funciones públicas
# -----------------------------------------------------------------------------

def construir_mapa_tareas_por_etiqueta(
    data: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Construye un diccionario que mapea etiqueta de tarea -> datos de la arista.

    Combina las listas de aristas requeridas y no requeridas de la instancia
    en un único diccionario de acceso rápido. Esto permite buscar los datos de
    cualquier tarea en O(1) (tiempo constante) dado su etiqueta.

    Parámetros
    ----------
    data : Mapping[str, Any]
        Diccionario de la instancia CARP. Debe contener las claves
        ``"LISTA_ARISTAS_REQ"`` y/o ``"LISTA_ARISTAS_NOREQ"``, cada una con
        una lista de diccionarios de arista, donde cada uno tiene la clave
        ``"tarea"`` con su etiqueta.

    Retorna
    -------
    dict[str, dict]
        Diccionario: etiqueta (str) -> diccionario de arista completo.
    """
    # Obtiene la lista de aristas requeridas; si no existe la clave, usa lista vacía
    lr = list(data.get("LISTA_ARISTAS_REQ") or [])

    # Obtiene la lista de aristas NO requeridas; mismo manejo de ausencia
    ln = list(data.get("LISTA_ARISTAS_NOREQ") or [])

    # Diccionario resultado: etiqueta -> dict de arista
    m: dict[str, dict[str, Any]] = {}

    # Itera sobre todas las aristas (requeridas primero, luego no requeridas)
    for t in lr + ln:
        k = t.get("tarea")  # Obtiene la etiqueta de esta tarea (ej. "TR1")
        if k is not None:
            m[str(k)] = t   # Indexa usando str() por si la clave no es cadena
    return m


def etiquetas_tareas_requeridas(data: Mapping[str, Any]) -> set[str]:
    """Devuelve el conjunto de etiquetas de todas las tareas REQUERIDAS.

    Una tarea requerida es una arista que el vehículo DEBE servir (tiene demanda
    asignada). Las no requeridas son aristas por las que el vehículo puede
    pasar sin servir (deadheading).

    Usa una "comprensión de conjunto" (set comprehension): sintaxis compacta de
    Python para construir un conjunto iterando sobre una colección.
    ``{expresion for item in iterable}`` genera un set con los resultados.

    Parámetros
    ----------
    data : Mapping[str, Any]
        Diccionario de la instancia CARP con la clave ``"LISTA_ARISTAS_REQ"``.
    """
    # Genera el conjunto de etiquetas convirtiendo cada valor de "tarea" a str
    return {str(t["tarea"]) for t in (data.get("LISTA_ARISTAS_REQ") or [])}


def _marcador_depot_str(data: Mapping[str, Any], override: str | None) -> str:
    """Determina el marcador de depósito a usar, en mayúsculas.

    Prioridad:
    1. Si ``override`` no es None, úsalo (el llamador lo especificó explícitamente).
    2. Si la instancia tiene la clave ``"MARCADOR_DEPOT_ETIQUETA"``, úsala.
    3. Si ninguna aplica, usa la constante por defecto ``"D"``.

    Siempre devuelve el marcador en mayúsculas para comparaciones consistentes.
    """
    if override is not None:
        # strip() elimina espacios accidentales; upper() convierte a mayúsculas
        return override.strip().upper()

    # Intenta leer el marcador de la propia instancia
    raw = data.get("MARCADOR_DEPOT_ETIQUETA")
    if raw is not None:
        return str(raw).strip().upper()

    # Valor por defecto si no hay ninguna configuración especial
    return CLAVE_MARCADOR_DEPOSITO_DEFAULT


def resolver_etiqueta_canonica(s: str, mapa: Mapping[str, dict[str, Any]]) -> str | None:
    """Busca la forma canónica (exacta) de una etiqueta en el mapa de tareas.

    Primero intenta coincidencia exacta. Si falla, intenta coincidencia
    insensible a mayúsculas/minúsculas. Esto permite que un usuario escriba
    ``"tr1"`` y se resuelva a ``"TR1"`` si esa es la forma guardada.

    Parámetros
    ----------
    s : str
        Etiqueta a buscar (tal como aparece en la solución del usuario).
    mapa : Mapping[str, dict]
        Diccionario construido con :func:`construir_mapa_tareas_por_etiqueta`.

    Retorna
    -------
    str | None
        La clave exacta en ``mapa`` si se encontró; ``None`` si no existe.
    """
    # Intento 1: coincidencia exacta (el caso más común y más rápido)
    if s in mapa:
        return s

    # Intento 2: comparación en mayúsculas (tolerancia a variaciones de capitalización)
    su = s.upper()
    for k in mapa:
        if str(k).upper() == su:
            return str(k)  # Devuelve la forma canónica (como está en el mapa)

    # Si no se encontró en ninguna forma, devuelve None
    return None


def normalizar_rutas_etiquetas(
    solucion: Sequence[Sequence[Hashable]],
    data: Mapping[str, Any],
    mapa: Mapping[str, dict[str, Any]],
    marcador_depot: str | None = None,
) -> tuple[list[list[str]], str | None]:
    """
    Limpia y valida una solución expresada como secuencias de etiquetas.

    Realiza dos operaciones:
    1. Elimina los marcadores de depósito (ej. ``"D"``) de cada ruta, ya que
       son solo puntos de inicio/fin y no son tareas a servir.
    2. Valida que cada token restante sea una etiqueta de tarea conocida en
       ``mapa``; si no lo es, devuelve un error descriptivo.

    Parámetros
    ----------
    solucion : Sequence[Sequence[Hashable]]
        Lista de rutas, donde cada ruta es una lista de tokens (etiquetas o "D").
        ``Sequence`` acepta listas, tuplas o cualquier secuencia ordenada.
        ``Hashable`` acepta str, int u otros tipos que puedan ser clave de dict.
    data : Mapping[str, Any]
        Diccionario de la instancia CARP (para leer ``MARCADOR_DEPOT_ETIQUETA``).
    mapa : Mapping[str, dict]
        Mapa etiqueta -> arista, construido con :func:`construir_mapa_tareas_por_etiqueta`.
    marcador_depot : str | None
        Marcador de depósito personalizado; si es None se lee de ``data`` o
        se usa el valor por defecto ``"D"``.

    Retorna
    -------
    tuple[list[list[str]], str | None]
        - Primer elemento: lista de rutas limpias (listas de etiquetas canónicas).
        - Segundo elemento: mensaje de error (str) si hay un problema, o None si todo fue bien.

        Convención: si hay error, la lista de rutas devuelta es vacía (``[]``).
    """
    # Determina el marcador de depósito a usar en este contexto
    md = _marcador_depot_str(data, marcador_depot)

    rutas: list[list[str]] = []  # Acumula las rutas limpias y validadas

    # Itera sobre cada ruta de la solución con su índice (enumerate proporciona el índice)
    for i, ruta in enumerate(solucion):
        fila: list[str] = []  # Tokens válidos de esta ruta (sin depósito)

        for x in ruta:
            # Convierte el token a string y elimina espacios al inicio/fin
            s = str(x).strip()

            # Token vacío: indica un problema en la solución de entrada
            if not s:
                return [], f"Ruta {i}: elemento vacío."

            # Compara con el marcador de depósito en mayúsculas
            su = s.upper()
            if su == md:
                # Es un marcador de depósito: se omite (no es una tarea)
                continue

            # Intenta resolver la etiqueta a su forma canónica en el mapa
            can = resolver_etiqueta_canonica(s, mapa)
            if can is None:
                # La etiqueta no corresponde a ninguna tarea conocida: error
                return [], f"Ruta {i}: etiqueta de tarea desconocida {s!r}."

            # Token válido: añade su forma canónica a la ruta limpia
            fila.append(can)

        rutas.append(fila)

    # Todo correcto: devuelve las rutas limpias y None como señal de "sin error"
    return rutas, None
