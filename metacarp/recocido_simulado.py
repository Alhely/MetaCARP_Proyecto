"""
Recocido Simulado clásico para CARP.

Concepto algorítmico
--------------------
El Recocido Simulado (Simulated Annealing, SA) es una metaheurística inspirada
en el proceso de recocido de metales en metalurgia:

- Al enfriar un metal muy lentamente, sus átomos se reorganizan en estructuras
  de menor energía (cristales perfectos). Si se enfría rápido, queda con
  imperfecciones (mínimos locales).

Trasladado a optimización:
- La **temperatura T** controla cuánto se permite "empeorar" una solución.
- Al inicio (T alta) se aceptan soluciones peores con alta probabilidad,
  permitiendo explorar ampliamente el espacio de búsqueda (equivale a enfriar
  lentamente en el metal: mucho movimiento aleatorio de átomos).
- Gradualmente T baja (enfriamiento), y se aceptan cada vez menos soluciones
  peores, convergiendo hacia buenos mínimos (el metal se solidifica).

Regla de aceptación de Metropolis
----------------------------------
En cada iteración se genera un vecino con costo ``c_vecino``:
- Si el vecino es **mejor** (delta <= 0): se acepta siempre.
- Si el vecino es **peor** (delta > 0): se acepta con probabilidad
  ``P = exp(-delta / T)``

donde ``delta = c_vecino - c_actual``.

Cuando T es grande, P es cercana a 1 (casi cualquier empeoramiento se acepta).
Cuando T tiende a 0, P tiende a 0 (solo se aceptan mejoras, como una búsqueda
local voraz).

Enfriamiento geométrico
-----------------------
La temperatura se reduce en cada "nivel" multiplicando por alpha (0 < alpha < 1):
    T_nueva = alpha × T_actual

Valores típicos de alpha: 0.90 – 0.99. Un alpha más alto significa un
enfriamiento más lento (mejor calidad, más tiempo de cómputo).

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
# Permite usar `float | None` en Python < 3.10.
from __future__ import annotations

# math.exp calcula la función exponencial: necesaria para la regla de Metropolis.
import math
# Generador de números aleatorios controlado por semilla.
import random
# Medición de tiempo de alta resolución.
import time
# Tipos abstractos para firmas de funciones.
from collections.abc import Iterable, Mapping
# Soporte de dataclasses.
from dataclasses import dataclass, field
# Tipos de anotación.
from typing import Any, Literal

# Biblioteca de grafos.
import networkx as nx

# Importaciones internas del paquete metacarp:
from .busqueda_indices import build_search_encoding  # codificación para vecindario por ids
from .cargar_grafos import cargar_objeto_gexf         # carga el grafo desde archivo GEXF
from .cargar_soluciones_iniciales import cargar_solucion_inicial  # carga solución inicial
from .evaluador_costo import (
    costo_rapido,                        # evaluación rápida de una solución (NumPy)
    exceso_capacidad_rapido,             # calcula violación de capacidad rápido
    lambda_penal_capacidad_por_defecto,  # λ por defecto para penalización
    objectivo_penalizado,                # función objetivo: costo + λ × violación
)
from .instances import load_instances  # carga datos de la instancia CARP
from .metaheuristicas_utils import (
    ContadorOperadores,
    calcular_metricas_gap,
    construir_contexto_para_corrida,
    copiar_solucion_labels,
    generar_reporte_detallado,
    guardar_resultado_csv,
    pesos_intra_bias,
    resumen_bks_csv,
    seleccionar_mejor_inicial_rapido,
    solucion_legible_humana,
)
from .vecindarios import MovimientoVecindario, OPERADORES_POPULARES, generar_vecino

# API pública del módulo.
__all__ = [
    "RecocidoSimuladoResult",
    "recocido_simulado",
    "recocido_simulado_desde_instancia",
]


# --- CONCEPTO OOP: @dataclass(frozen=True, slots=True) ---
# frozen=True: objeto inmutable, sus atributos no se pueden cambiar tras su creación.
# slots=True: menor uso de memoria al fijar el conjunto de atributos en tiempo de compilación.
@dataclass(frozen=True, slots=True)
class RecocidoSimuladoResult:
    """
    Resultado completo del recocido simulado clásico.

    Agrupa la mejor solución encontrada, métricas de calidad, historial de
    temperatura, estadísticas de aceptaciones y operadores de vecindario.
    """

    # La mejor solución CARP encontrada durante toda la búsqueda.
    mejor_solucion: list[list[str]]
    # Costo de la mejor solución (objetivo minimizado).
    mejor_costo: float
    # Solución inicial de referencia (para medir la mejora).
    solucion_inicial_referencia: list[list[str]]
    # Costo de la solución inicial.
    costo_solucion_inicial: float
    # Diferencia absoluta: costo_inicial - mejor_costo (positivo = mejora).
    mejora_absoluta: float
    # Porcentaje de mejora respecto al costo inicial.
    mejora_porcentaje_inicial_vs_final: float
    # Tiempo total de ejecución en segundos.
    tiempo_segundos: float
    # Total de evaluaciones individuales de soluciones vecinas.
    iteraciones_totales: int
    # Cuántos ciclos de enfriamiento se ejecutaron (reducciones de temperatura).
    enfriamientos_ejecutados: int
    # Total de vecinos aceptados (mejores + peores aceptados por Metropolis).
    aceptadas: int
    # Veces que el mejor global mejoró.
    mejoras: int
    # Semilla del generador aleatorio.
    semilla: int | None
    # Dispositivo de evaluación: 'cpu' o 'gpu'.
    backend_evaluacion: str = "cpu"
    # Historial del mejor costo al inicio de cada nivel de temperatura.
    historial_mejor_costo: list[float] = field(default_factory=list)
    # Historial del valor de temperatura por nivel.
    historial_temperatura: list[float] = field(default_factory=list)
    # Último movimiento de vecindario aceptado.
    ultimo_movimiento_aceptado: MovimientoVecindario | None = None
    # Estadísticas de operadores de vecindario.
    operadores_propuestos: dict[str, int] = field(default_factory=dict)
    operadores_aceptados: dict[str, int] = field(default_factory=dict)
    operadores_mejoraron: dict[str, int] = field(default_factory=dict)
    operadores_trayectoria_mejor: dict[str, int] = field(default_factory=dict)
    # Si True, se usó penalización de capacidad.
    usar_penalizacion_capacidad: bool = True
    # Valor efectivo de λ.
    lambda_capacidad: float = 0.0
    # Estadísticas de la selección inicial.
    n_iniciales_evaluados: int = 0
    iniciales_infactibles_aceptadas: int = 0
    # Veces que se aceptó una solución que viola capacidad.
    aceptaciones_solucion_infactible: int = 0
    # True si la mejor solución final respeta todas las restricciones.
    mejor_solucion_factible_final: bool = True
    # Ruta del CSV guardado, o None si no se guardó.
    archivo_csv: str | None = None


def recocido_simulado(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    temperatura_inicial: float = 1000.0,      # temperatura de inicio (alta = mucha exploración)
    temperatura_minima: float = 1e-3,         # temperatura de parada (baja = solo mejoras)
    alpha: float = 0.95,                      # factor de enfriamiento geométrico (0 < alpha < 1)
    iteraciones_por_temperatura: int = 120,   # evaluaciones por nivel de temperatura
    max_enfriamientos: int = 250,             # límite de reducciones de temperatura
    semilla: int | None = None,               # semilla para reproducibilidad
    operadores: Iterable[str] = OPERADORES_POPULARES,  # operadores de vecindario habilitados
    marcador_depot_etiqueta: str | None = None,  # etiqueta del nodo depósito
    usar_gpu: bool = False,                   # flag de GPU (solo para trazabilidad en SA)
    backend_vecindario: Literal["labels", "ids"] = "labels",  # modo de generación de vecinos
    guardar_historial: bool = True,           # si True, guarda historial por nivel de temperatura
    guardar_csv: bool = False,                # si True, escribe resultados en CSV
    ruta_csv: str | None = None,              # ruta del CSV
    nombre_instancia: str = "instancia",     # nombre para el CSV
    id_corrida: str | None = None,
    config_id: str | None = None,
    repeticion: int | None = None,
    root: str | None = None,
    usar_penalizacion_capacidad: bool = True,  # si True, penaliza violaciones de capacidad
    lambda_capacidad: float | None = None,     # peso λ (None = automático)
    extra_csv: dict[str, object] | None = None,  # columnas adicionales para el CSV
    alpha_intra: float = 0.8,  # fracción de prob. asignada a ops intra-ruta cuando hay violación
) -> RecocidoSimuladoResult:
    """
    Recocido Simulado clásico para minimizar el costo de soluciones CARP.

    Estructura del algoritmo:
    - Bucle externo: niveles de temperatura (de T_inicial hasta T_minima).
    - Bucle interno: 'iteraciones_por_temperatura' evaluaciones por nivel.
    - En cada evaluación: genera un vecino, decide si aceptarlo (Metropolis).
    - Al finalizar cada nivel: T = alpha × T (enfriamiento geométrico).

    Criterio de parada: T < temperatura_minima O enfriamientos >= max_enfriamientos.
    """
    # Validaciones de parámetros.
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

    # Generador aleatorio reproducible.
    rng = random.Random(semilla)
    # Marca de tiempo de inicio.
    t0 = time.perf_counter()

    # Construcción del contexto de evaluación rápida (una sola vez para toda la corrida).
    ctx = construir_contexto_para_corrida(
        data,
        G,
        nombre_instancia=nombre_instancia if nombre_instancia != "instancia" else None,
        usar_gpu=usar_gpu,
        root=root,
    )

    # Lambda efectiva para penalización de capacidad.
    lam_eff = (
        float(lambda_capacidad)
        if lambda_capacidad is not None
        else lambda_penal_capacidad_por_defecto(ctx)
    )

    # Selección de la mejor solución inicial entre las candidatas.
    sel_ini = seleccionar_mejor_inicial_rapido(
        inicial_obj,
        ctx,
        usar_penalizacion_capacidad=usar_penalizacion_capacidad,
        lambda_capacidad=lambda_capacidad,
    )
    sol_ref = sel_ini.solucion          # solución de referencia para medir mejora
    costo_ref = sel_ini.costo_puro      # costo inicial de referencia
    ini_infact = sel_ini.n_candidatos_infactibles
    n_ini_ev = sel_ini.n_candidatos_evaluados

    # La solución "actual" es la que se modifica en cada iteración del recocido.
    sol_actual = copiar_solucion_labels(sol_ref)
    costo_actual = costo_ref
    viol_actual = sel_ini.violacion_capacidad  # violación de capacidad de la solución actual

    # Rastreo del mejor global y del mejor factible (sin violación de capacidad).
    mejor_any_c = float(costo_ref)
    mejor_any_s = copiar_solucion_labels(sol_ref)
    if viol_actual < 1e-12:
        mejor_fact_c: float | None = float(costo_ref)
        mejor_fact_s = copiar_solucion_labels(sol_ref)
    else:
        mejor_fact_c = None
        mejor_fact_s = None

    # Configuración del encoding para el backend de ids.
    encoding = ctx.encoding if backend_vecindario == "ids" else None
    if backend_vecindario == "ids" and encoding is None:
        encoding = build_search_encoding(data)

    # Temperatura inicial del recocido (variable que decrece con cada nivel).
    T = float(temperatura_inicial)
    iteraciones_totales = 0  # total de vecinos evaluados
    enfriamientos = 0        # número de reducciones de temperatura ejecutadas
    aceptadas = 0            # vecinos aceptados (mejores + peores)
    mejoras = 0              # veces que el mejor global mejoró
    ultimo_mov_aceptado: MovimientoVecindario | None = None
    historial_best: list[float] = []  # historial del mejor costo por nivel de T
    historial_temp: list[float] = []  # historial de temperaturas por nivel
    contador = ContadorOperadores()

    # Etiqueta del depósito para los operadores de vecindario.
    md_op = marcador_depot_etiqueta or ctx.marcador_depot

    def costo_para_reporte() -> float:
        """Retorna el mejor costo factible si existe; si no, el mejor general."""
        return float(mejor_fact_c) if mejor_fact_c is not None else mejor_any_c

    aceptaciones_sol_infactible = 0  # veces que se aceptó una solución infactible

    # Inicializamos sol_mejor y costo_mejor para uso dentro del bucle.
    sol_mejor = copiar_solucion_labels(
        mejor_fact_s if mejor_fact_s is not None else mejor_any_s
    )
    costo_mejor = costo_para_reporte()

    # === BUCLE EXTERNO: niveles de temperatura (enfriamiento) ===
    # Se repite mientras T no baje del mínimo Y no se excedan los enfriamientos máximos.
    while T > temperatura_minima and enfriamientos < max_enfriamientos:
        if guardar_historial:
            # Registramos la temperatura y el mejor costo al inicio de este nivel.
            historial_temp.append(T)
            historial_best.append(costo_para_reporte())

        # === BUCLE INTERNO: evaluaciones dentro del nivel de temperatura T ===
        pesos_ops = pesos_intra_bias(viol_actual, list(operadores), alpha_intra=alpha_intra)
        for _ in range(iteraciones_por_temperatura):
            iteraciones_totales += 1

            # Generamos un vecino aleatorio de la solución actual.
            vecino, mov = generar_vecino(
                sol_actual,
                rng=rng,
                operadores=operadores,
                pesos_operadores=pesos_ops,
                marcador_depot=md_op,
                devolver_con_deposito=True,
                usar_gpu=usar_gpu,
                backend=backend_vecindario,
                encoding=encoding,
            )
            contador.proponer(mov.operador)

            # Evaluamos el vecino: costo puro y violación de capacidad.
            costo_vec = costo_rapido(vecino, ctx)
            viol_vec = exceso_capacidad_rapido(vecino, ctx)

            # Calculamos los objetivos penalizados para comparación.
            # El objetivo incluye la penalización de capacidad si está activa.
            obj_actual = objectivo_penalizado(
                costo_actual,
                viol_actual,
                usar_penal=usar_penalizacion_capacidad,
                lam=lam_eff,
            )
            obj_vec = objectivo_penalizado(
                costo_vec,
                viol_vec,
                usar_penal=usar_penalizacion_capacidad,
                lam=lam_eff,
            )

            # delta = diferencia de objetivos (positivo = el vecino es peor).
            delta = obj_vec - obj_actual

            # --- Regla de aceptación de Metropolis ---
            if delta <= 0:
                # El vecino es igual o mejor: se acepta siempre.
                aceptar = True
            else:
                # El vecino es peor: se acepta con probabilidad exp(-delta / T).
                # Cuando T es grande: exp(-delta/T) ≈ 1 → se acepta casi siempre.
                # Cuando T es pequeño: exp(-delta/T) ≈ 0 → raramente se acepta.
                # rng.random() genera un número uniforme en [0, 1).
                # Si ese número es menor que la probabilidad, se acepta.
                aceptar = rng.random() < math.exp(-delta / T)

            if aceptar:
                # Actualizamos la solución actual al vecino aceptado.
                sol_actual = vecino
                costo_actual = costo_vec
                viol_actual = viol_vec
                aceptadas += 1
                ultimo_mov_aceptado = mov
                contador.aceptar(mov.operador)
                # Si la solución aceptada viola capacidad, lo registramos.
                if usar_penalizacion_capacidad and viol_vec > 1e-12:
                    aceptaciones_sol_infactible += 1

                # Verificamos si esta aceptación mejoró el mejor global.
                antes_rep = costo_para_reporte()
                # Actualizamos el mejor sin restricción de factibilidad.
                if costo_vec < mejor_any_c - 1e-15:
                    mejor_any_c = costo_vec
                    mejor_any_s = copiar_solucion_labels(sol_actual)
                # Actualizamos el mejor factible solo si el vecino no viola capacidad.
                if viol_vec < 1e-12:
                    lim_fact = mejor_fact_c if mejor_fact_c is not None else float("inf")
                    if costo_vec < lim_fact - 1e-15:
                        mejor_fact_c = float(costo_vec)
                        mejor_fact_s = copiar_solucion_labels(sol_actual)

                despues_rep = costo_para_reporte()
                # Si el costo reportable mejoró, registramos la mejora.
                if despues_rep < antes_rep - 1e-12:
                    mejoras += 1
                    contador.registrar_mejora(mov.operador)

                # Actualizamos sol_mejor y costo_mejor.
                sol_mejor = copiar_solucion_labels(
                    mejor_fact_s if mejor_fact_s is not None else mejor_any_s
                )
                costo_mejor = despues_rep

        # --- Enfriamiento geométrico ---
        # T_nueva = alpha × T_actual
        # alpha < 1, por lo que la temperatura decrece en cada nivel.
        T *= alpha
        enfriamientos += 1

    # === FIN DEL BUCLE EXTERNO ===

    # Tiempo total de la corrida.
    elapsed = time.perf_counter() - t0
    # Tomamos el mejor reportable final.
    costo_mejor = costo_para_reporte()
    sol_mejor = copiar_solucion_labels(
        mejor_fact_s if mejor_fact_s is not None else mejor_any_s
    )
    # True si la mejor solución final respeta capacidades.
    mejor_factible_final = mejor_fact_s is not None
    # _gap_descartado: calculamos el gap pero no lo incluimos en el resultado directamente.
    _gap_descartado, mejora_abs, mejora_pct = calcular_metricas_gap(costo_ref, costo_mejor)

    # --- Guardado en CSV (opcional) ---
    archivo_csv: str | None = None
    if guardar_csv:
        ruta = ruta_csv or f"resultados_recocido_simulado_{nombre_instancia}.csv"
        detalle_txt, costo_total_reporte = generar_reporte_detallado(
            sol_mejor, data, G,
            nombre_instancia=nombre_instancia,
            marcador_depot_etiqueta=marcador_depot_etiqueta,
            usar_gpu=False,  # reporte usa NetworkX para texto detallado
        )
        _bks = resumen_bks_csv(data, costo_mejor)
        fila = {
            "metaheuristica": "recocido_simulado",
            "instancia": nombre_instancia,
            "bks_referencia": _bks["bks_referencia"],
            "bks_origen": _bks["bks_origen"],
            "gap_bks_porcentaje": _bks["gap_bks_porcentaje"],
            "repeticion": repeticion if repeticion is not None else "",
            "semilla": semilla,
            "tiempo_segundos": elapsed,
            "mejor_costo": costo_mejor,
            "costo_solucion_inicial": costo_ref,
            "temperatura_inicial": temperatura_inicial,
            "temperatura_minima": temperatura_minima,
            "alpha": alpha,
            "iteraciones_por_temperatura": iteraciones_por_temperatura,
            "max_enfriamientos": max_enfriamientos,
            **contador.resumen_csv(),
            "aceptadas": aceptadas,
            "mejoras": mejoras,
        }
        archivo_csv = guardar_resultado_csv(fila=fila, ruta_csv=ruta)

    # Retornamos el objeto de resultado inmutable con todos los datos de la corrida.
    return RecocidoSimuladoResult(
        mejor_solucion=sol_mejor,
        mejor_costo=costo_mejor,
        solucion_inicial_referencia=sol_ref,
        costo_solucion_inicial=costo_ref,
        mejora_absoluta=mejora_abs,
        mejora_porcentaje_inicial_vs_final=mejora_pct,
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
        usar_penalizacion_capacidad=usar_penalizacion_capacidad,
        lambda_capacidad=lam_eff,
        n_iniciales_evaluados=n_ini_ev,
        iniciales_infactibles_aceptadas=ini_infact,
        aceptaciones_solucion_infactible=aceptaciones_sol_infactible,
        mejor_solucion_factible_final=mejor_factible_final,
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
    usar_penalizacion_capacidad: bool = True,
    lambda_capacidad: float | None = None,
    extra_csv: dict[str, object] | None = None,
    alpha_intra: float = 0.8,
) -> RecocidoSimuladoResult:
    """
    Función de conveniencia: carga todos los recursos necesarios desde el nombre
    de la instancia y ejecuta el recocido simulado completo.

    Equivalente a llamar manualmente a load_instances + cargar_objeto_gexf
    + cargar_solucion_inicial + recocido_simulado.
    """
    # Cargamos los datos de la instancia (capacidad, demandas, BKS, etc.).
    data = load_instances(nombre_instancia, root=root)
    # Cargamos el grafo de la instancia desde el archivo GEXF.
    G = cargar_objeto_gexf(nombre_instancia, root=root)
    # Cargamos la solución inicial desde el archivo pickle.
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
        usar_penalizacion_capacidad=usar_penalizacion_capacidad,
        lambda_capacidad=lambda_capacidad,
        extra_csv=extra_csv,
        alpha_intra=alpha_intra,
    )
