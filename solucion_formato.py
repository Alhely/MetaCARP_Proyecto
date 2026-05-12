from __future__ import annotations

from typing import Any, Hashable, Mapping, Sequence

"""Formato de solución por etiquetas: ``TR*``/``TNR*`` con ``D`` = depósito."""

__all__ = [
    "CLAVE_MARCADOR_DEPOSITO_DEFAULT",
    "construir_mapa_tareas_por_etiqueta",
    "etiquetas_tareas_requeridas",
    "normalizar_rutas_etiquetas",
    "resolver_etiqueta_canonica",
]

CLAVE_MARCADOR_DEPOSITO_DEFAULT = "D"


def construir_mapa_tareas_por_etiqueta(
    data: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """``t['tarea']`` -> dict de arista (LISTA_ARISTAS_REQ + LISTA_ARISTAS_NOREQ)."""
    lr = list(data.get("LISTA_ARISTAS_REQ") or [])
    ln = list(data.get("LISTA_ARISTAS_NOREQ") or [])
    m: dict[str, dict[str, Any]] = {}
    for t in lr + ln:
        k = t.get("tarea")
        if k is not None:
            m[str(k)] = t
    return m


def etiquetas_tareas_requeridas(data: Mapping[str, Any]) -> set[str]:
    return {str(t["tarea"]) for t in (data.get("LISTA_ARISTAS_REQ") or [])}


def _marcador_depot_str(data: Mapping[str, Any], override: str | None) -> str:
    if override is not None:
        return override.strip().upper()
    raw = data.get("MARCADOR_DEPOT_ETIQUETA")
    if raw is not None:
        return str(raw).strip().upper()
    return CLAVE_MARCADOR_DEPOSITO_DEFAULT


def resolver_etiqueta_canonica(s: str, mapa: Mapping[str, dict[str, Any]]) -> str | None:
    """Coincidencia exacta o solo por mayúsculas con alguna clave de ``mapa``."""
    if s in mapa:
        return s
    su = s.upper()
    for k in mapa:
        if str(k).upper() == su:
            return str(k)
    return None


def normalizar_rutas_etiquetas(
    solucion: Sequence[Sequence[Hashable]],
    data: Mapping[str, Any],
    mapa: Mapping[str, dict[str, Any]],
    marcador_depot: str | None = None,
) -> tuple[list[list[str]], str | None]:
    """
    Quita marcadores de depósito (p. ej. ``D``) y deja solo etiquetas de tareas
    conocidas en ``mapa``. Error si token desconocido.
    """
    md = _marcador_depot_str(data, marcador_depot)

    rutas: list[list[str]] = []
    for i, ruta in enumerate(solucion):
        fila: list[str] = []
        for x in ruta:
            s = str(x).strip()
            if not s:
                return [], f"Ruta {i}: elemento vacío."
            su = s.upper()
            if su == md:
                continue
            can = resolver_etiqueta_canonica(s, mapa)
            if can is None:
                return [], f"Ruta {i}: etiqueta de tarea desconocida {s!r}."
            fila.append(can)
        rutas.append(fila)
    return rutas, None
