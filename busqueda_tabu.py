"""
Búsqueda Tabú clásica con memoria de corto plazo.

Optimización
------------
- Construye un :class:`ContextoEvaluacion` una sola vez (matriz Dijkstra densa).
- Cada vecino del lote se evalúa con :func:`costo_rapido` (10×–50× más rápido).

GPU (opcional)
--------------
Cuando ``usar_gpu=True`` y CuPy está disponible, **el lote completo de vecinos
por iteración** se evalúa en GPU con :func:`costo_lote_ids`. En instancias
pequeñas no aporta speedup (overhead PCIe), en instancias grandes con
``tam_vecindario >= 30`` sí compensa. Si CuPy no está disponible el código
hace fallback transparente a CPU rápido.
"""
from __future__ import annotations

import random
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import networkx as nx

from .busqueda_indices import build_search_encoding, encode_solution
from .cargar_grafos import cargar_objeto_gexf
from .cargar_soluciones_iniciales import cargar_solucion_inicial
from .evaluador_costo import costo_lote_ids, costo_rapido
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
    "BusquedaTabuResult",
    "busqueda_tabu",
    "busqueda_tabu_desde_instancia",
]


@dataclass(frozen=True, slots=True)
class BusquedaTabuResult:
    """Resultado de búsqueda tabú con métricas de calidad y tiempo."""

    mejor_solucion: list[list[str]]
    mejor_costo: float
    solucion_inicial_referencia: list[list[str]]
    costo_inicial_referencia: float
    gap_porcentaje: float
    mejora_absoluta: float
    mejora_porcentaje: float
    tiempo_segundos: float
    iteraciones_totales: int
    vecinos_evaluados: int
    movimientos_tabu_bloqueados: int
    mejoras: int
    semilla: int | None
    backend_evaluacion: str = "cpu"
    historial_mejor_costo: list[float] = field(default_factory=list)
    ultimo_movimiento_aceptado: MovimientoVecindario | None = None
    operadores_propuestos: dict[str, int] = field(default_factory=dict)
    operadores_aceptados: dict[str, int] = field(default_factory=dict)
    operadores_mejoraron: dict[str, int] = field(default_factory=dict)
    operadores_trayectoria_mejor: dict[str, int] = field(default_factory=dict)
    archivo_csv: str | None = None


def _clave_tabu(mov: MovimientoVecindario) -> tuple[Any, ...]:
    """Clave hashable del movimiento para memoria tabú."""
    return (
        mov.operador, mov.ruta_a, mov.ruta_b,
        mov.i, mov.j, mov.k, mov.l,
        tuple(mov.labels_movidos),
    )


def busqueda_tabu(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    iteraciones: int = 400,
    tam_vecindario: int = 25,
    tenure_tabu: int = 20,
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
) -> BusquedaTabuResult:
    """Búsqueda tabú clásica (short-term memory)."""
    if iteraciones <= 0:
        raise ValueError("iteraciones debe ser > 0.")
    if tam_vecindario <= 0:
        raise ValueError("tam_vecindario debe ser > 0.")
    if tenure_tabu <= 0:
        raise ValueError("tenure_tabu debe ser > 0.")

    rng = random.Random(semilla)
    t0 = time.perf_counter()

    ctx = construir_contexto_para_corrida(
        data, G,
        nombre_instancia=nombre_instancia if nombre_instancia != "instancia" else None,
        usar_gpu=usar_gpu, root=root,
    )

    sol_ref, costo_ref = seleccionar_mejor_inicial_rapido(inicial_obj, ctx)
    sol_actual = copiar_solucion_labels(sol_ref)
    costo_actual = costo_ref
    sol_mejor = copiar_solucion_labels(sol_ref)
    costo_mejor = costo_ref

    encoding = ctx.encoding
    if backend_vecindario == "ids" and encoding is None:
        encoding = build_search_encoding(data)

    tabu_hasta: dict[tuple[Any, ...], int] = {}
    vecinos_evaluados = 0
    bloqueados = 0
    mejoras = 0
    ultimo_mov_aceptado: MovimientoVecindario | None = None
    historial_best: list[float] = []
    contador = ContadorOperadores()

    md_op = marcador_depot_etiqueta or ctx.marcador_depot
    usar_gpu_lote = ctx.usar_gpu and tam_vecindario >= 16

    for it in range(iteraciones):
        if guardar_historial:
            historial_best.append(costo_mejor)

        # Generamos un lote de vecinos para evaluar.
        vecinos: list[list[list[str]]] = []
        movimientos: list[MovimientoVecindario] = []
        for _ in range(tam_vecindario):
            vecino, mov = generar_vecino(
                sol_actual, rng=rng, operadores=operadores,
                marcador_depot=md_op, devolver_con_deposito=True,
                usar_gpu=usar_gpu, backend=backend_vecindario, encoding=encoding,
            )
            vecinos.append(vecino)
            movimientos.append(mov)
            contador.proponer(mov.operador)

        # Evaluación: GPU por lote para vecindarios grandes; CPU rápido en otro caso.
        if usar_gpu_lote:
            sols_ids = [encode_solution(v, ctx.encoding) for v in vecinos]
            costos = costo_lote_ids(sols_ids, ctx).tolist()
        else:
            costos = [costo_rapido(v, ctx) for v in vecinos]
        vecinos_evaluados += len(vecinos)

        # Selección con criterio tabú + aspiración.
        mejor_admisible_idx = -1
        mejor_admisible_cost = float("inf")
        mejor_total_idx = 0
        mejor_total_cost = float("inf")

        for idx, c in enumerate(costos):
            if c < mejor_total_cost:
                mejor_total_cost = c
                mejor_total_idx = idx
            key = _clave_tabu(movimientos[idx])
            es_tabu = tabu_hasta.get(key, -1) > it
            aspiracion = c < costo_mejor
            if es_tabu and not aspiracion:
                continue
            if c < mejor_admisible_cost:
                mejor_admisible_cost = c
                mejor_admisible_idx = idx

        if mejor_admisible_idx == -1:
            elegido_idx = mejor_total_idx
            bloqueados += len(costos)
        else:
            elegido_idx = mejor_admisible_idx
            for idx, mov in enumerate(movimientos):
                key = _clave_tabu(mov)
                if tabu_hasta.get(key, -1) > it:
                    bloqueados += 1

        sol_actual = vecinos[elegido_idx]
        costo_actual = costos[elegido_idx]
        ultimo_mov_aceptado = movimientos[elegido_idx]
        contador.aceptar(ultimo_mov_aceptado.operador)

        tabu_hasta[_clave_tabu(ultimo_mov_aceptado)] = it + tenure_tabu

        if it % 25 == 0 and tabu_hasta:
            for k in [k for k, vence in tabu_hasta.items() if vence <= it]:
                del tabu_hasta[k]

        if costo_actual < costo_mejor:
            costo_mejor = costo_actual
            sol_mejor = copiar_solucion_labels(sol_actual)
            mejoras += 1
            contador.registrar_mejora(ultimo_mov_aceptado.operador)

    elapsed = time.perf_counter() - t0
    gap, mejora_abs, mejora_pct = calcular_metricas_gap(costo_ref, costo_mejor)

    archivo_csv: str | None = None
    if guardar_csv:
        ruta = ruta_csv or f"resultados_busqueda_tabu_{nombre_instancia}.csv"
        detalle_txt, costo_total_reporte = generar_reporte_detallado(
            sol_mejor, data, G,
            nombre_instancia=nombre_instancia,
            marcador_depot_etiqueta=marcador_depot_etiqueta,
            usar_gpu=False,
        )
        fila = {
            "metaheuristica": "busqueda_tabu",
            "instancia": nombre_instancia,
            "id_corrida": id_corrida or "",
            "config_id": config_id or "",
            "repeticion": repeticion if repeticion is not None else "",
            "semilla": semilla,
            "backend_evaluacion_solicitado": ctx.backend_solicitado,
            "backend_evaluacion_real": ctx.backend_real,
            "tiempo_segundos": elapsed,
            "iteraciones_totales": iteraciones,
            "vecinos_evaluados": vecinos_evaluados,
            "movimientos_tabu_bloqueados": bloqueados,
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

    return BusquedaTabuResult(
        mejor_solucion=sol_mejor,
        mejor_costo=costo_mejor,
        solucion_inicial_referencia=sol_ref,
        costo_inicial_referencia=costo_ref,
        gap_porcentaje=gap,
        mejora_absoluta=mejora_abs,
        mejora_porcentaje=mejora_pct,
        tiempo_segundos=elapsed,
        iteraciones_totales=iteraciones,
        vecinos_evaluados=vecinos_evaluados,
        movimientos_tabu_bloqueados=bloqueados,
        mejoras=mejoras,
        semilla=semilla,
        backend_evaluacion=ctx.backend_real,
        historial_mejor_costo=historial_best,
        ultimo_movimiento_aceptado=ultimo_mov_aceptado,
        operadores_propuestos=contador.como_dict_ordenado(contador.propuestos),
        operadores_aceptados=contador.como_dict_ordenado(contador.aceptados),
        operadores_mejoraron=contador.como_dict_ordenado(contador.mejoraron),
        operadores_trayectoria_mejor=contador.como_dict_ordenado(contador.trayectoria_mejor),
        archivo_csv=archivo_csv,
    )


def busqueda_tabu_desde_instancia(
    nombre_instancia: str,
    *,
    root: str | None = None,
    iteraciones: int = 400,
    tam_vecindario: int = 25,
    tenure_tabu: int = 20,
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
) -> BusquedaTabuResult:
    """Helper de ejecución tabú cargando recursos desde la instancia."""
    data = load_instances(nombre_instancia, root=root)
    G = cargar_objeto_gexf(nombre_instancia, root=root)
    inicial_obj = cargar_solucion_inicial(nombre_instancia, root=root)
    return busqueda_tabu(
        inicial_obj, data, G,
        iteraciones=iteraciones,
        tam_vecindario=tam_vecindario,
        tenure_tabu=tenure_tabu,
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
