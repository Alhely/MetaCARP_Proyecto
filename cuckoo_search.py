"""
Cuckoo Search adaptado a espacio discreto para CARP.

Optimización
------------
Construye un :class:`ContextoEvaluacion` una vez y evalúa con
:func:`costo_rapido`. Cuando ``usar_gpu=True`` y CuPy está disponible, los
``num_nidos`` cuckoos generados por iteración se evalúan en lote en GPU vía
:func:`costo_lote_ids`.
"""
from __future__ import annotations

import math
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
    "CuckooSearchResult",
    "cuckoo_search",
    "cuckoo_search_desde_instancia",
]


@dataclass(frozen=True, slots=True)
class CuckooSearchResult:
    """Resultado de Cuckoo Search con reemplazo por nidos y abandono parcial."""

    mejor_solucion: list[list[str]]
    mejor_costo: float
    solucion_inicial_referencia: list[list[str]]
    costo_inicial_referencia: float
    gap_porcentaje: float
    mejora_absoluta: float
    mejora_porcentaje: float
    tiempo_segundos: float
    iteraciones_totales: int
    nidos: int
    abandonos_totales: int
    reemplazos_exitosos: int
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


def _vuelo_levy_discreto(
    base: list[list[str]],
    *,
    rng: random.Random,
    pasos_base: int,
    beta: float,
    operadores: Iterable[str],
    marcador_depot: str,
    usar_gpu: bool,
    backend_vecindario: Literal["labels", "ids"],
    encoding: Any,
) -> tuple[list[list[str]], list[MovimientoVecindario]]:
    """
    Aproximación discreta de vuelo Levy:
    ``n_pasos = 1 + floor(|N(0,1)|^(1/beta) * pasos_base)``, acotado a 12.

    Devuelve la solución resultante y la **lista completa** de movimientos
    aplicados, para que el llamador pueda contabilizar cada operador.
    """
    if beta <= 0:
        beta = 1.5
    x = abs(rng.gauss(0.0, 1.0))
    n_pasos = min(12, 1 + int((x ** (1.0 / beta)) * max(1, pasos_base)))

    sol = copiar_solucion_labels(base)
    movs_seq: list[MovimientoVecindario] = []
    for _ in range(n_pasos):
        sol, m = generar_vecino(
            sol, rng=rng, operadores=operadores,
            marcador_depot=marcador_depot, devolver_con_deposito=True,
            usar_gpu=usar_gpu, backend=backend_vecindario, encoding=encoding,
        )
        movs_seq.append(m)
    return sol, movs_seq


def _eval_costos(
    vecinos: list[list[list[str]]],
    ctx: Any,
    *,
    usar_gpu_lote: bool,
) -> list[float]:
    """GPU por lote si procede; si no, CPU rápido."""
    if usar_gpu_lote and len(vecinos) >= 8:
        sols_ids = [encode_solution(v, ctx.encoding) for v in vecinos]
        return costo_lote_ids(sols_ids, ctx).tolist()
    return [costo_rapido(v, ctx) for v in vecinos]


def cuckoo_search(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    iteraciones: int = 260,
    num_nidos: int = 20,
    pa_abandono: float = 0.25,
    pasos_levy_base: int = 3,
    beta_levy: float = 1.5,
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
) -> CuckooSearchResult:
    """Cuckoo Search clásico adaptado a espacio discreto."""
    if iteraciones <= 0:
        raise ValueError("iteraciones debe ser > 0.")
    if num_nidos <= 1:
        raise ValueError("num_nidos debe ser >= 2.")
    if not (0.0 < pa_abandono < 1.0):
        raise ValueError("pa_abandono debe estar en (0, 1).")
    if pasos_levy_base <= 0:
        raise ValueError("pasos_levy_base debe ser > 0.")

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

    contador = ContadorOperadores()

    # Inicialización de nidos: mejor referencia + perturbaciones.
    nidos_sol: list[list[list[str]]] = [copiar_solucion_labels(sol_ref)]
    nidos_cost: list[float] = [costo_ref]
    while len(nidos_sol) < num_nidos:
        cand, _m = generar_vecino(
            sol_ref, rng=rng, operadores=operadores,
            marcador_depot=md_op, devolver_con_deposito=True,
            usar_gpu=usar_gpu, backend=backend_vecindario, encoding=encoding,
        )
        nidos_sol.append(cand)
        nidos_cost.append(costo_rapido(cand, ctx))

    idx_best = min(range(num_nidos), key=nidos_cost.__getitem__)
    sol_mejor = copiar_solucion_labels(nidos_sol[idx_best])
    costo_mejor = nidos_cost[idx_best]
    mejoras = 0
    reemplazos = 0
    abandonos = 0
    historial_best: list[float] = []
    ultimo_mov_aceptado: MovimientoVecindario | None = None

    for _it in range(iteraciones):
        if guardar_historial:
            historial_best.append(costo_mejor)

        # Generación de cuckoos por vuelo Levy.
        cuckoos: list[list[list[str]]] = []
        movs_levy: list[list[MovimientoVecindario]] = []
        for i in range(num_nidos):
            cs, movs_seq = _vuelo_levy_discreto(
                nidos_sol[i], rng=rng,
                pasos_base=pasos_levy_base, beta=beta_levy,
                operadores=operadores, marcador_depot=md_op,
                usar_gpu=usar_gpu, backend_vecindario=backend_vecindario, encoding=encoding,
            )
            cuckoos.append(cs)
            movs_levy.append(movs_seq)
            # Cada paso del vuelo Levy es una propuesta del operador.
            for m in movs_seq:
                contador.proponer(m.operador)

        costos_cuckoos = _eval_costos(cuckoos, ctx, usar_gpu_lote=usar_gpu_lote)

        # Competencia con un nido aleatorio.
        for i in range(num_nidos):
            j = rng.randrange(num_nidos)
            if costos_cuckoos[i] < nidos_cost[j]:
                nidos_sol[j] = cuckoos[i]
                nidos_cost[j] = costos_cuckoos[i]
                reemplazos += 1
                # Aceptamos: atribuimos al último operador del vuelo (el que
                # consolida la solución entregada al nido).
                if movs_levy[i]:
                    ultimo_mov_aceptado = movs_levy[i][-1]
                    contador.aceptar(ultimo_mov_aceptado.operador)

        # Abandono de los peores nidos (reemplazo en torno al mejor).
        n_abandonar = max(1, int(math.floor(pa_abandono * num_nidos)))
        peores = sorted(range(num_nidos), key=nidos_cost.__getitem__, reverse=True)[:n_abandonar]
        idx_best = min(range(num_nidos), key=nidos_cost.__getitem__)
        base_best = nidos_sol[idx_best]
        nuevos: list[list[list[str]]] = []
        movs_abandono: list[list[MovimientoVecindario]] = []
        for _idx in peores:
            ns, ms = _vuelo_levy_discreto(
                base_best, rng=rng,
                pasos_base=pasos_levy_base, beta=beta_levy,
                operadores=operadores, marcador_depot=md_op,
                usar_gpu=usar_gpu, backend_vecindario=backend_vecindario, encoding=encoding,
            )
            nuevos.append(ns)
            movs_abandono.append(ms)
            for m in ms:
                contador.proponer(m.operador)
        costos_nuevos = _eval_costos(nuevos, ctx, usar_gpu_lote=usar_gpu_lote)
        for k, idx in enumerate(peores):
            nidos_sol[idx] = nuevos[k]
            nidos_cost[idx] = costos_nuevos[k]
            abandonos += 1
            # El abandono reemplaza incondicionalmente: contamos como aceptado.
            if movs_abandono[k]:
                contador.aceptar(movs_abandono[k][-1].operador)

        # Mejor global.
        idx_best = min(range(num_nidos), key=nidos_cost.__getitem__)
        if nidos_cost[idx_best] < costo_mejor:
            costo_mejor = nidos_cost[idx_best]
            sol_mejor = copiar_solucion_labels(nidos_sol[idx_best])
            mejoras += 1
            op_mejor = ultimo_mov_aceptado.operador if ultimo_mov_aceptado else None
            contador.registrar_mejora(op_mejor)

    elapsed = time.perf_counter() - t0
    gap, mejora_abs, mejora_pct = calcular_metricas_gap(costo_ref, costo_mejor)

    archivo_csv: str | None = None
    if guardar_csv:
        ruta = ruta_csv or f"resultados_cuckoo_search_{nombre_instancia}.csv"
        detalle_txt, costo_total_reporte = generar_reporte_detallado(
            sol_mejor, data, G,
            nombre_instancia=nombre_instancia,
            marcador_depot_etiqueta=marcador_depot_etiqueta,
            usar_gpu=False,
        )
        fila = {
            "metaheuristica": "cuckoo_search",
            "instancia": nombre_instancia,
            "id_corrida": id_corrida or "",
            "config_id": config_id or "",
            "repeticion": repeticion if repeticion is not None else "",
            "semilla": semilla,
            "backend_evaluacion_solicitado": ctx.backend_solicitado,
            "backend_evaluacion_real": ctx.backend_real,
            "tiempo_segundos": elapsed,
            "iteraciones_totales": iteraciones,
            "nidos": num_nidos,
            "abandonos_totales": abandonos,
            "reemplazos_exitosos": reemplazos,
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

    return CuckooSearchResult(
        mejor_solucion=sol_mejor,
        mejor_costo=costo_mejor,
        solucion_inicial_referencia=sol_ref,
        costo_inicial_referencia=costo_ref,
        gap_porcentaje=gap,
        mejora_absoluta=mejora_abs,
        mejora_porcentaje=mejora_pct,
        tiempo_segundos=elapsed,
        iteraciones_totales=iteraciones,
        nidos=num_nidos,
        abandonos_totales=abandonos,
        reemplazos_exitosos=reemplazos,
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


def cuckoo_search_desde_instancia(
    nombre_instancia: str,
    *,
    root: str | None = None,
    iteraciones: int = 260,
    num_nidos: int = 20,
    pa_abandono: float = 0.25,
    pasos_levy_base: int = 3,
    beta_levy: float = 1.5,
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
) -> CuckooSearchResult:
    """Helper que carga recursos y ejecuta Cuckoo Search."""
    data = load_instances(nombre_instancia, root=root)
    G = cargar_objeto_gexf(nombre_instancia, root=root)
    inicial_obj = cargar_solucion_inicial(nombre_instancia, root=root)
    return cuckoo_search(
        inicial_obj, data, G,
        iteraciones=iteraciones,
        num_nidos=num_nidos,
        pa_abandono=pa_abandono,
        pasos_levy_base=pasos_levy_base,
        beta_levy=beta_levy,
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
