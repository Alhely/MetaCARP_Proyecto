"""
Artificial Bee Colony (ABC) simplificada para CARP.

Optimización
------------
Construye un :class:`ContextoEvaluacion` una sola vez y evalúa cada vecino
con :func:`costo_rapido` (NumPy fancy-indexing). En las fases de empleadas y
observadoras (que generan ``num_fuentes`` vecinos por iteración), si
``usar_gpu=True`` y CuPy está disponible, las evaluaciones se hacen en lote
con :func:`costo_lote_ids`.
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
    "AbejasResult",
    "busqueda_abejas",
    "busqueda_abejas_desde_instancia",
]


@dataclass(frozen=True, slots=True)
class AbejasResult:
    """Resultado de la metaheurística Artificial Bee Colony (ABC) simplificada."""

    mejor_solucion: list[list[str]]
    mejor_costo: float
    solucion_inicial_referencia: list[list[str]]
    costo_inicial_referencia: float
    gap_porcentaje: float
    mejora_absoluta: float
    mejora_porcentaje: float
    tiempo_segundos: float
    iteraciones_totales: int
    fuentes_alimento: int
    scouts_reinicios: int
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


def _generar_vecinos_lote(
    sources: list[list[list[str]]],
    *,
    rng: random.Random,
    operadores: Iterable[str],
    marcador_depot: str,
    usar_gpu: bool,
    backend_vecindario: Literal["labels", "ids"],
    encoding: Any,
) -> tuple[list[list[list[str]]], list[MovimientoVecindario]]:
    """Aplica una perturbación local a cada solución fuente."""
    vecinos: list[list[list[str]]] = []
    movs: list[MovimientoVecindario] = []
    for sol in sources:
        v, m = generar_vecino(
            sol, rng=rng, operadores=operadores,
            marcador_depot=marcador_depot, devolver_con_deposito=True,
            usar_gpu=usar_gpu, backend=backend_vecindario, encoding=encoding,
        )
        vecinos.append(v)
        movs.append(m)
    return vecinos, movs


def _eval_costos(
    vecinos: list[list[list[str]]],
    ctx: Any,
    *,
    usar_gpu_lote: bool,
) -> list[float]:
    """Evalúa una lista de vecinos: GPU por lote si procede, si no CPU rápido."""
    if usar_gpu_lote and len(vecinos) >= 8:
        sols_ids = [encode_solution(v, ctx.encoding) for v in vecinos]
        return costo_lote_ids(sols_ids, ctx).tolist()
    return [costo_rapido(v, ctx) for v in vecinos]


def busqueda_abejas(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    iteraciones: int = 250,
    num_fuentes: int = 16,
    limite_abandono: int = 35,
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
) -> AbejasResult:
    """ABC simplificada con empleadas, observadoras y scouts."""
    if iteraciones <= 0:
        raise ValueError("iteraciones debe ser > 0.")
    if num_fuentes <= 1:
        raise ValueError("num_fuentes debe ser >= 2.")
    if limite_abandono <= 0:
        raise ValueError("limite_abandono debe ser > 0.")

    rng = random.Random(semilla)
    t0 = time.perf_counter()

    ctx = construir_contexto_para_corrida(
        data, G,
        nombre_instancia=nombre_instancia if nombre_instancia != "instancia" else None,
        usar_gpu=usar_gpu, root=root,
    )

    sol_ref, costo_ref = seleccionar_mejor_inicial_rapido(inicial_obj, ctx)

    encoding = ctx.encoding
    if backend_vecindario == "ids" and encoding is None:
        encoding = build_search_encoding(data)

    md_op = marcador_depot_etiqueta or ctx.marcador_depot
    usar_gpu_lote = ctx.usar_gpu

    # Inicialización de fuentes: base + perturbaciones.
    fuentes_sol: list[list[list[str]]] = [copiar_solucion_labels(sol_ref)]
    fuentes_cost: list[float] = [costo_ref]
    trials: list[int] = [0]
    while len(fuentes_sol) < num_fuentes:
        v, _m = generar_vecino(
            sol_ref, rng=rng, operadores=operadores,
            marcador_depot=md_op, devolver_con_deposito=True,
            usar_gpu=usar_gpu, backend=backend_vecindario, encoding=encoding,
        )
        fuentes_sol.append(v)
        fuentes_cost.append(costo_rapido(v, ctx))
        trials.append(0)

    best_idx = min(range(num_fuentes), key=fuentes_cost.__getitem__)
    sol_mejor = copiar_solucion_labels(fuentes_sol[best_idx])
    costo_mejor = fuentes_cost[best_idx]
    mejoras = 0
    scouts = 0
    historial_best: list[float] = []
    ultimo_mov_aceptado: MovimientoVecindario | None = None
    contador = ContadorOperadores()

    rango_fuentes = list(range(num_fuentes))

    for _it in range(iteraciones):
        if guardar_historial:
            historial_best.append(costo_mejor)

        # 1) Empleadas: una perturbación por fuente.
        vecinos, movs = _generar_vecinos_lote(
            fuentes_sol, rng=rng, operadores=operadores,
            marcador_depot=md_op, usar_gpu=usar_gpu,
            backend_vecindario=backend_vecindario, encoding=encoding,
        )
        for m in movs:
            contador.proponer(m.operador)
        costos = _eval_costos(vecinos, ctx, usar_gpu_lote=usar_gpu_lote)
        for i, c in enumerate(costos):
            if c < fuentes_cost[i]:
                fuentes_sol[i] = vecinos[i]
                fuentes_cost[i] = c
                trials[i] = 0
                ultimo_mov_aceptado = movs[i]
                contador.aceptar(movs[i].operador)
            else:
                trials[i] += 1

        # 2) Observadoras: selección por fitness.
        total_inv = sum(1.0 / (1.0 + max(c, 0.0)) for c in fuentes_cost)
        if total_inv <= 0:
            probs = [1.0 / num_fuentes] * num_fuentes
        else:
            probs = [(1.0 / (1.0 + max(c, 0.0))) / total_inv for c in fuentes_cost]
        idxs = rng.choices(rango_fuentes, weights=probs, k=num_fuentes)

        srcs = [fuentes_sol[i] for i in idxs]
        vecinos, movs = _generar_vecinos_lote(
            srcs, rng=rng, operadores=operadores,
            marcador_depot=md_op, usar_gpu=usar_gpu,
            backend_vecindario=backend_vecindario, encoding=encoding,
        )
        for m in movs:
            contador.proponer(m.operador)
        costos = _eval_costos(vecinos, ctx, usar_gpu_lote=usar_gpu_lote)
        for k, c in enumerate(costos):
            i = idxs[k]
            if c < fuentes_cost[i]:
                fuentes_sol[i] = vecinos[k]
                fuentes_cost[i] = c
                trials[i] = 0
                ultimo_mov_aceptado = movs[k]
                contador.aceptar(movs[k].operador)
            else:
                trials[i] += 1

        # 3) Scouts: reinicios alrededor del mejor.
        best_idx = min(rango_fuentes, key=fuentes_cost.__getitem__)
        base_best = fuentes_sol[best_idx]
        a_reiniciar = [i for i in rango_fuentes if trials[i] >= limite_abandono]
        if a_reiniciar:
            srcs = [base_best] * len(a_reiniciar)
            vecinos, movs_sc = _generar_vecinos_lote(
                srcs, rng=rng, operadores=operadores,
                marcador_depot=md_op, usar_gpu=usar_gpu,
                backend_vecindario=backend_vecindario, encoding=encoding,
            )
            for m in movs_sc:
                contador.proponer(m.operador)
            costos = _eval_costos(vecinos, ctx, usar_gpu_lote=usar_gpu_lote)
            for k, i in enumerate(a_reiniciar):
                fuentes_sol[i] = vecinos[k]
                fuentes_cost[i] = costos[k]
                trials[i] = 0
                scouts += 1
                # Los scouts reemplazan incondicionalmente: contamos la
                # operación como aceptada para análisis de uso por operador.
                contador.aceptar(movs_sc[k].operador)

        # Mejor global.
        best_idx = min(rango_fuentes, key=fuentes_cost.__getitem__)
        if fuentes_cost[best_idx] < costo_mejor:
            costo_mejor = fuentes_cost[best_idx]
            sol_mejor = copiar_solucion_labels(fuentes_sol[best_idx])
            mejoras += 1
            # Atribuimos la mejora al último operador aceptado que contribuyó.
            op_mejor = ultimo_mov_aceptado.operador if ultimo_mov_aceptado else None
            contador.registrar_mejora(op_mejor)

    elapsed = time.perf_counter() - t0
    gap, mejora_abs, mejora_pct = calcular_metricas_gap(costo_ref, costo_mejor)

    archivo_csv: str | None = None
    if guardar_csv:
        ruta = ruta_csv or f"resultados_busqueda_abejas_{nombre_instancia}.csv"
        detalle_txt, costo_total_reporte = generar_reporte_detallado(
            sol_mejor, data, G,
            nombre_instancia=nombre_instancia,
            marcador_depot_etiqueta=marcador_depot_etiqueta,
            usar_gpu=False,
        )
        fila = {
            "metaheuristica": "busqueda_abejas",
            "instancia": nombre_instancia,
            "id_corrida": id_corrida or "",
            "config_id": config_id or "",
            "repeticion": repeticion if repeticion is not None else "",
            "semilla": semilla,
            "backend_evaluacion_solicitado": ctx.backend_solicitado,
            "backend_evaluacion_real": ctx.backend_real,
            "tiempo_segundos": elapsed,
            "iteraciones_totales": iteraciones,
            "fuentes_alimento": num_fuentes,
            "scouts_reinicios": scouts,
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

    return AbejasResult(
        mejor_solucion=sol_mejor,
        mejor_costo=costo_mejor,
        solucion_inicial_referencia=sol_ref,
        costo_inicial_referencia=costo_ref,
        gap_porcentaje=gap,
        mejora_absoluta=mejora_abs,
        mejora_porcentaje=mejora_pct,
        tiempo_segundos=elapsed,
        iteraciones_totales=iteraciones,
        fuentes_alimento=num_fuentes,
        scouts_reinicios=scouts,
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


def busqueda_abejas_desde_instancia(
    nombre_instancia: str,
    *,
    root: str | None = None,
    iteraciones: int = 250,
    num_fuentes: int = 16,
    limite_abandono: int = 35,
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
) -> AbejasResult:
    """Helper de ABC cargando recursos desde la instancia."""
    data = load_instances(nombre_instancia, root=root)
    G = cargar_objeto_gexf(nombre_instancia, root=root)
    inicial_obj = cargar_solucion_inicial(nombre_instancia, root=root)
    return busqueda_abejas(
        inicial_obj, data, G,
        iteraciones=iteraciones,
        num_fuentes=num_fuentes,
        limite_abandono=limite_abandono,
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
