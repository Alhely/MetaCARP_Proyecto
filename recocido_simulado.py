"""
Recocido Simulado clásico para CARP.

Optimización de evaluación
--------------------------
Construye un :class:`ContextoEvaluacion` (matriz Dijkstra densa + arrays por id
de tarea) **una sola vez** al inicio. Cada vecino se evalúa con
:func:`costo_rapido` (NumPy fancy-indexing): 10×–50× más rápido que el
evaluador clásico basado en NetworkX, sin alterar la fórmula de costo.

GPU
---
SA evalúa una solución por iteración, por lo que el flag ``usar_gpu`` se pasa
al contexto solo para trazabilidad: el cuello de botella ya está resuelto en
CPU y mover datos a GPU no aporta speedup en este caso.
"""
from __future__ import annotations

import math
import random
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import networkx as nx

from .busqueda_indices import build_search_encoding
from .cargar_grafos import cargar_objeto_gexf
from .cargar_soluciones_iniciales import cargar_solucion_inicial
from .evaluador_costo import costo_rapido
from .instances import load_instances
from .metaheuristicas_utils import (
    ContadorOperadores,
    calcular_metricas_gap,
    construir_contexto_para_corrida,
    copiar_solucion_labels,
    generar_reporte_detallado,
    guardar_resultado_csv,
    seleccionar_mejor_inicial_rapido,
    solucion_legible_humana,
)
from .vecindarios import MovimientoVecindario, OPERADORES_POPULARES, generar_vecino

__all__ = [
    "RecocidoSimuladoResult",
    "recocido_simulado",
    "recocido_simulado_desde_instancia",
]


@dataclass(frozen=True, slots=True)
class RecocidoSimuladoResult:
    """Resultado completo del recocido simulado clásico."""

    mejor_solucion: list[list[str]]
    mejor_costo: float
    solucion_inicial_referencia: list[list[str]]
    costo_inicial_referencia: float
    gap_porcentaje: float
    mejora_absoluta: float
    mejora_porcentaje: float
    tiempo_segundos: float
    iteraciones_totales: int
    enfriamientos_ejecutados: int
    aceptadas: int
    mejoras: int
    semilla: int | None
    backend_evaluacion: str = "cpu"
    historial_mejor_costo: list[float] = field(default_factory=list)
    historial_temperatura: list[float] = field(default_factory=list)
    ultimo_movimiento_aceptado: MovimientoVecindario | None = None
    operadores_propuestos: dict[str, int] = field(default_factory=dict)
    operadores_aceptados: dict[str, int] = field(default_factory=dict)
    operadores_mejoraron: dict[str, int] = field(default_factory=dict)
    operadores_trayectoria_mejor: dict[str, int] = field(default_factory=dict)
    archivo_csv: str | None = None


def recocido_simulado(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    temperatura_inicial: float = 1000.0,
    temperatura_minima: float = 1e-3,
    alpha: float = 0.95,
    iteraciones_por_temperatura: int = 120,
    max_enfriamientos: int = 250,
    semilla: int | None = None,
    operadores: Iterable[str] = OPERADORES_POPULARES,
    marcador_depot_etiqueta: str | None = None,
    usar_gpu: bool = False,
    backend_vecindario: Literal["labels", "ids"] = "labels",
    guardar_historial: bool = True,
    guardar_csv: bool = False,
    ruta_csv: str | None = None,
    nombre_instancia: str = "instancia",
    id_corrida: str | None = None,
    config_id: str | None = None,
    repeticion: int | None = None,
    root: str | None = None,
) -> RecocidoSimuladoResult:
    """Recocido simulado clásico para minimizar costo de solución CARP."""
    if temperatura_inicial <= 0:
        raise ValueError("temperatura_inicial debe ser > 0.")
    if temperatura_minima <= 0:
        raise ValueError("temperatura_minima debe ser > 0.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha debe estar en (0, 1).")
    if iteraciones_por_temperatura <= 0:
        raise ValueError("iteraciones_por_temperatura debe ser > 0.")
    if max_enfriamientos <= 0:
        raise ValueError("max_enfriamientos debe ser > 0.")

    rng = random.Random(semilla)
    t0 = time.perf_counter()

    ctx = construir_contexto_para_corrida(
        data,
        G,
        nombre_instancia=nombre_instancia if nombre_instancia != "instancia" else None,
        usar_gpu=usar_gpu,
        root=root,
    )

    sol_ref, costo_ref = seleccionar_mejor_inicial_rapido(inicial_obj, ctx)

    sol_actual = copiar_solucion_labels(sol_ref)
    costo_actual = costo_ref
    sol_mejor = copiar_solucion_labels(sol_ref)
    costo_mejor = costo_ref

    encoding = ctx.encoding if backend_vecindario == "ids" else None
    if backend_vecindario == "ids" and encoding is None:
        encoding = build_search_encoding(data)

    T = float(temperatura_inicial)
    iteraciones_totales = 0
    enfriamientos = 0
    aceptadas = 0
    mejoras = 0
    ultimo_mov_aceptado: MovimientoVecindario | None = None
    historial_best: list[float] = []
    historial_temp: list[float] = []
    contador = ContadorOperadores()

    md_op = marcador_depot_etiqueta or ctx.marcador_depot

    while T > temperatura_minima and enfriamientos < max_enfriamientos:
        if guardar_historial:
            historial_temp.append(T)
            historial_best.append(costo_mejor)

        for _ in range(iteraciones_por_temperatura):
            iteraciones_totales += 1

            vecino, mov = generar_vecino(
                sol_actual,
                rng=rng,
                operadores=operadores,
                marcador_depot=md_op,
                devolver_con_deposito=True,
                usar_gpu=usar_gpu,
                backend=backend_vecindario,
                encoding=encoding,
            )
            contador.proponer(mov.operador)

            costo_vecino = costo_rapido(vecino, ctx)
            delta = costo_vecino - costo_actual

            if delta <= 0:
                aceptar = True
            else:
                aceptar = rng.random() < math.exp(-delta / T)

            if aceptar:
                sol_actual = vecino
                costo_actual = costo_vecino
                aceptadas += 1
                ultimo_mov_aceptado = mov
                contador.aceptar(mov.operador)
                if costo_actual < costo_mejor:
                    costo_mejor = costo_actual
                    sol_mejor = copiar_solucion_labels(sol_actual)
                    mejoras += 1
                    contador.registrar_mejora(mov.operador)

        T *= alpha
        enfriamientos += 1

    elapsed = time.perf_counter() - t0
    gap, mejora_abs, mejora_pct = calcular_metricas_gap(costo_ref, costo_mejor)

    archivo_csv: str | None = None
    if guardar_csv:
        ruta = ruta_csv or f"resultados_recocido_simulado_{nombre_instancia}.csv"
        detalle_txt, costo_total_reporte = generar_reporte_detallado(
            sol_mejor, data, G,
            nombre_instancia=nombre_instancia,
            marcador_depot_etiqueta=marcador_depot_etiqueta,
            usar_gpu=False,  # reporte usa NetworkX para texto detallado
        )
        fila = {
            "metaheuristica": "recocido_simulado",
            "instancia": nombre_instancia,
            "id_corrida": id_corrida or "",
            "config_id": config_id or "",
            "repeticion": repeticion if repeticion is not None else "",
            "semilla": semilla,
            "backend_evaluacion_solicitado": ctx.backend_solicitado,
            "backend_evaluacion_real": ctx.backend_real,
            "tiempo_segundos": elapsed,
            "iteraciones_totales": iteraciones_totales,
            "enfriamientos_ejecutados": enfriamientos,
            "aceptadas": aceptadas,
            "mejoras": mejoras,
            "costo_inicial_referencia": costo_ref,
            "mejor_costo": costo_mejor,
            "gap_porcentaje": gap,
            "mejora_absoluta": mejora_abs,
            "mejora_porcentaje": mejora_pct,
            "mejor_solucion_tr_legible": solucion_legible_humana(sol_mejor),
            "reporte_detalle_deadheading": detalle_txt,
            "costo_total_desde_reporte": costo_total_reporte,
            **contador.resumen_csv(),
        }
        archivo_csv = guardar_resultado_csv(fila=fila, ruta_csv=ruta)

    return RecocidoSimuladoResult(
        mejor_solucion=sol_mejor,
        mejor_costo=costo_mejor,
        solucion_inicial_referencia=sol_ref,
        costo_inicial_referencia=costo_ref,
        gap_porcentaje=gap,
        mejora_absoluta=mejora_abs,
        mejora_porcentaje=mejora_pct,
        tiempo_segundos=elapsed,
        iteraciones_totales=iteraciones_totales,
        enfriamientos_ejecutados=enfriamientos,
        aceptadas=aceptadas,
        mejoras=mejoras,
        semilla=semilla,
        backend_evaluacion=ctx.backend_real,
        historial_mejor_costo=historial_best,
        historial_temperatura=historial_temp,
        ultimo_movimiento_aceptado=ultimo_mov_aceptado,
        operadores_propuestos=contador.como_dict_ordenado(contador.propuestos),
        operadores_aceptados=contador.como_dict_ordenado(contador.aceptados),
        operadores_mejoraron=contador.como_dict_ordenado(contador.mejoraron),
        operadores_trayectoria_mejor=contador.como_dict_ordenado(contador.trayectoria_mejor),
        archivo_csv=archivo_csv,
    )


def recocido_simulado_desde_instancia(
    nombre_instancia: str,
    *,
    root: str | None = None,
    temperatura_inicial: float = 1000.0,
    temperatura_minima: float = 1e-3,
    alpha: float = 0.95,
    iteraciones_por_temperatura: int = 120,
    max_enfriamientos: int = 250,
    semilla: int | None = None,
    operadores: Iterable[str] = OPERADORES_POPULARES,
    marcador_depot_etiqueta: str | None = None,
    usar_gpu: bool = False,
    backend_vecindario: Literal["labels", "ids"] = "labels",
    guardar_historial: bool = True,
    guardar_csv: bool = False,
    ruta_csv: str | None = None,
    id_corrida: str | None = None,
    config_id: str | None = None,
    repeticion: int | None = None,
) -> RecocidoSimuladoResult:
    """Helper que carga instancia + grafo + objeto inicial y ejecuta recocido."""
    data = load_instances(nombre_instancia, root=root)
    G = cargar_objeto_gexf(nombre_instancia, root=root)
    inicial_obj = cargar_solucion_inicial(nombre_instancia, root=root)
    return recocido_simulado(
        inicial_obj,
        data,
        G,
        temperatura_inicial=temperatura_inicial,
        temperatura_minima=temperatura_minima,
        alpha=alpha,
        iteraciones_por_temperatura=iteraciones_por_temperatura,
        max_enfriamientos=max_enfriamientos,
        semilla=semilla,
        operadores=operadores,
        marcador_depot_etiqueta=marcador_depot_etiqueta,
        usar_gpu=usar_gpu,
        backend_vecindario=backend_vecindario,
        guardar_historial=guardar_historial,
        guardar_csv=guardar_csv,
        ruta_csv=ruta_csv,
        nombre_instancia=nombre_instancia,
        id_corrida=id_corrida,
        config_id=config_id,
        repeticion=repeticion,
        root=root,
    )
