"""
Utilidades comunes a todas las metaheurísticas (selección inicial, reporte y
exportación a CSV).

La evaluación de costo dentro de los bucles internos NO debe usar las funciones
de este módulo: para eso existe ``evaluador_costo.costo_rapido`` (10×–50× más
rápido que el evaluador clásico basado en NetworkX).
"""
from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Hashable

import networkx as nx

from .costo_solucion import costo_solucion
from .evaluador_costo import (
    ContextoEvaluacion,
    construir_contexto,
    costo_rapido,
)
from .reporte_solucion import reporte_solucion

__all__ = [
    "ContadorOperadores",
    "copiar_solucion_labels",
    "extraer_candidatas_desde_objeto",
    "evaluar_costo_solucion",
    "seleccionar_mejor_inicial",
    "seleccionar_mejor_inicial_rapido",
    "calcular_metricas_gap",
    "solucion_legible_humana",
    "generar_reporte_detallado",
    "guardar_resultado_csv",
    "construir_contexto_para_corrida",
]


@dataclass
class ContadorOperadores:
    """
    Lleva la cuenta de uso de operadores de vecindario durante una corrida.

    - ``propuestos``: cada vez que un operador se invoca para generar un vecino.
    - ``aceptados``: cada vez que el movimiento del operador se incorpora al
      estado actual (cambia la solución actual).
    - ``mejoraron``: subconjunto de ``aceptados`` que además bajó el mejor
      global histórico.
    - ``trayectoria_mejor``: snapshot de ``aceptados`` capturado en el momento
      en que se descubrió la mejor solución reportada al final. Responde
      directamente "qué operadores se usaron para construir la mejor".
    """

    propuestos: Counter = field(default_factory=Counter)
    aceptados: Counter = field(default_factory=Counter)
    mejoraron: Counter = field(default_factory=Counter)
    trayectoria_mejor: Counter = field(default_factory=Counter)

    def proponer(self, op: str | None) -> None:
        if op:
            self.propuestos[op] += 1

    def aceptar(self, op: str | None) -> None:
        if op:
            self.aceptados[op] += 1

    def registrar_mejora(self, op: str | None) -> None:
        """Marca una mejora del mejor global y congela el snapshot de aceptados."""
        if op:
            self.mejoraron[op] += 1
        # El snapshot incluye TODOS los operadores aceptados hasta ahora,
        # incluyendo el actual (asumimos que aceptar(op) ya fue llamado).
        self.trayectoria_mejor = Counter(self.aceptados)

    def como_dict_ordenado(self, contador: Counter) -> dict[str, int]:
        """Convierte un Counter a dict ordenado por valor descendente."""
        return dict(sorted(contador.items(), key=lambda kv: (-kv[1], kv[0])))

    def resumen_csv(self) -> dict[str, str]:
        """Cuatro columnas listas para serializar en CSV (como dicts ordenados)."""
        return {
            "operadores_propuestos": str(self.como_dict_ordenado(self.propuestos)),
            "operadores_aceptados": str(self.como_dict_ordenado(self.aceptados)),
            "operadores_mejoraron": str(self.como_dict_ordenado(self.mejoraron)),
            "operadores_trayectoria_mejor": str(
                self.como_dict_ordenado(self.trayectoria_mejor)
            ),
        }


def copiar_solucion_labels(sol: Sequence[Sequence[Hashable]]) -> list[list[str]]:
    """Copia ligera a formato de etiquetas string."""
    return [[str(x).strip() for x in ruta] for ruta in sol]


def _es_solucion_lista_de_rutas(obj: Any) -> bool:
    """Heurística: lista de listas con tokens hashables."""
    if not isinstance(obj, (list, tuple)) or isinstance(obj, (str, bytes)):
        return False
    for ruta in obj:
        if not isinstance(ruta, (list, tuple)) or isinstance(ruta, (str, bytes)):
            return False
        for token in ruta:
            if isinstance(token, Mapping):
                return False
    return True


def extraer_candidatas_desde_objeto(obj: Any, *, max_nodos: int = 20000) -> list[list[list[str]]]:
    """Recorre recursivamente y extrae candidatas con forma de solución CARP."""
    candidatas: list[list[list[str]]] = []
    visitados = 0

    def _walk(x: Any) -> None:
        nonlocal visitados
        visitados += 1
        if visitados > max_nodos:
            return
        if _es_solucion_lista_de_rutas(x):
            candidatas.append(copiar_solucion_labels(x))
            return
        if isinstance(x, Mapping):
            for v in x.values():
                _walk(v)
            return
        if isinstance(x, (list, tuple, set)):
            for v in x:
                _walk(v)

    _walk(obj)
    return candidatas


def evaluar_costo_solucion(
    solucion: Sequence[Sequence[Hashable]],
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    marcador_depot_etiqueta: str | None,
    usar_gpu: bool,
) -> float:
    """
    Evaluación lenta (NetworkX). Conservada solo para usos puntuales fuera de
    bucles internos. Para metaheurísticas, prefiere ``costo_rapido(sol, ctx)``.
    """
    return costo_solucion(
        solucion,
        data,
        G,
        detalle=False,
        marcador_depot_etiqueta=marcador_depot_etiqueta,
        usar_gpu=usar_gpu,
    ).costo_total


def construir_contexto_para_corrida(
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    nombre_instancia: str | None,
    usar_gpu: bool,
    root: str | None = None,
) -> ContextoEvaluacion:
    """
    Construye un contexto de evaluación rápida para una corrida.

    Si se proporciona ``nombre_instancia``, intenta cachear/usar la matriz
    Dijkstra precomputada de la instancia. Si no, computa APSP desde ``G``.
    """
    if nombre_instancia:
        from .evaluador_costo import construir_contexto_desde_instancia

        try:
            return construir_contexto_desde_instancia(
                nombre_instancia, root=root, usar_gpu=usar_gpu
            )
        except FileNotFoundError:
            pass
    return construir_contexto(data, G=G, usar_gpu=usar_gpu)


def seleccionar_mejor_inicial(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    marcador_depot_etiqueta: str | None,
    usar_gpu: bool,
) -> tuple[list[list[str]], float]:
    """
    Versión lenta (NetworkX). Mantenida para retrocompatibilidad. En código
    nuevo usa ``seleccionar_mejor_inicial_rapido`` con el contexto.
    """
    candidatas = extraer_candidatas_desde_objeto(inicial_obj)
    if not candidatas:
        raise ValueError(
            "No se encontraron soluciones candidatas en el objeto inicial. "
            "Se esperaba lista de rutas o estructura anidada que la contenga."
        )

    mejor_sol: list[list[str]] | None = None
    mejor_cost = float("inf")
    errores = 0
    for cand in candidatas:
        try:
            c = evaluar_costo_solucion(
                cand,
                data,
                G,
                marcador_depot_etiqueta=marcador_depot_etiqueta,
                usar_gpu=usar_gpu,
            )
        except Exception:  # noqa: BLE001
            errores += 1
            continue
        if c < mejor_cost:
            mejor_cost = c
            mejor_sol = cand

    if mejor_sol is None:
        raise ValueError(
            "Ninguna candidata inicial pudo evaluarse con costo_solucion. "
            f"Candidatas detectadas: {len(candidatas)} | inválidas: {errores}."
        )
    return mejor_sol, mejor_cost


def seleccionar_mejor_inicial_rapido(
    inicial_obj: Any,
    ctx: ContextoEvaluacion,
) -> tuple[list[list[str]], float]:
    """
    Selecciona la mejor candidata inicial usando el evaluador rápido (NumPy).
    Es 10×–50× más rápido que ``seleccionar_mejor_inicial``.
    """
    candidatas = extraer_candidatas_desde_objeto(inicial_obj)
    if not candidatas:
        raise ValueError(
            "No se encontraron soluciones candidatas en el objeto inicial."
        )

    mejor_sol: list[list[str]] | None = None
    mejor_cost = float("inf")
    errores = 0
    for cand in candidatas:
        try:
            c = costo_rapido(cand, ctx)
        except Exception:  # noqa: BLE001
            errores += 1
            continue
        if c < mejor_cost:
            mejor_cost = c
            mejor_sol = cand

    if mejor_sol is None:
        raise ValueError(
            "Ninguna candidata pudo evaluarse con el evaluador rápido. "
            f"Candidatas detectadas: {len(candidatas)} | inválidas: {errores}."
        )
    return mejor_sol, mejor_cost


def calcular_metricas_gap(costo_inicial: float, costo_mejor: float) -> tuple[float, float, float]:
    """
    Devuelve (gap_pct, mejora_abs, mejora_pct).
    Gap negativo indica mejora contra la referencia inicial.
    """
    if costo_inicial == 0:
        gap = 0.0 if costo_mejor == 0 else float("inf")
        return gap, costo_inicial - costo_mejor, 0.0
    gap = ((costo_mejor - costo_inicial) / costo_inicial) * 100.0
    mejora_abs = costo_inicial - costo_mejor
    mejora_pct = (mejora_abs / costo_inicial) * 100.0
    return gap, mejora_abs, mejora_pct


def solucion_legible_humana(solucion: Sequence[Sequence[Hashable]]) -> str:
    """Convierte una solución a texto legible: ``R1: D -> TR1 -> ... -> D || R2: ...``."""
    rutas_txt: list[str] = []
    for i, ruta in enumerate(solucion, start=1):
        seq = [str(x).strip() for x in ruta]
        rutas_txt.append(f"R{i}: " + " -> ".join(seq))
    return " || ".join(rutas_txt)


def generar_reporte_detallado(
    solucion: Sequence[Sequence[Hashable]],
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    nombre_instancia: str = "instancia",
    marcador_depot_etiqueta: str | None,
    usar_gpu: bool,
) -> tuple[str, float]:
    """
    Detalle textual con DH y costo total. Se ejecuta solo una vez (al final
    de la corrida) por lo que se mantiene en el evaluador clásico.
    """
    rep = reporte_solucion(
        solucion,
        data,
        G,
        nombre_instancia=nombre_instancia,
        marcador_depot_etiqueta=marcador_depot_etiqueta,
        usar_gpu=usar_gpu,
        guardar=False,
    )
    return rep.texto, rep.costo_total


def guardar_resultado_csv(
    *,
    fila: Mapping[str, Any],
    ruta_csv: str | Path,
) -> str:
    """
    Guarda una ejecución en CSV (una fila por corrida, una columna por item).
    Si el archivo no existe, escribe encabezado.
    """
    path = Path(ruta_csv).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    normalizada: dict[str, Any] = {}
    for k, v in fila.items():
        normalizada[k] = str(v) if isinstance(v, (list, tuple, set, dict)) else v

    existe = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(normalizada.keys()))
        if not existe:
            writer.writeheader()
        writer.writerow(normalizada)
    return str(path)
