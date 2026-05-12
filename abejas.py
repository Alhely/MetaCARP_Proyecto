"""
Artificial Bee Colony (ABC) simplificada para CARP.

Concepto algorítmico
--------------------
ABC (Colonia Artificial de Abejas) es una metaheurística inspirada en el
comportamiento de búsqueda de alimento de las abejas melíferas. El algoritmo
divide la colonia en tres roles:

1. **Abejas empleadas**: cada una está asignada a una "fuente de alimento"
   (una solución candidata). Exploran el vecindario de su fuente y actualizan
   la fuente si encuentran una mejor. Analogía: abejas que ya conocen una flor
   y van a mejorar su explotación.

2. **Abejas observadoras**: observan la danza de las empleadas y eligen una
   fuente de alimento con probabilidad proporcional a su calidad. Luego también
   generan un vecino de esa fuente. Analogía: abejas que escuchan la "danza
   del meneo" y deciden qué fuente visitar.

3. **Abejas scout** (exploradoras): cuando una fuente no mejora después de
   ``limite_abandono`` intentos, se abandona y se reemplaza por una nueva
   solución generada en torno a la mejor actual. Analogía: la abeja deja de
   visitar una flor agotada y busca nuevas.

Optimización
------------
Construye un :class:`ContextoEvaluacion` una sola vez y evalúa cada vecino
con :func:`costo_rapido` (NumPy fancy-indexing). En las fases de empleadas y
observadoras (que generan ``num_fuentes`` vecinos por iteración), si
``usar_gpu=True`` y CuPy está disponible, las evaluaciones se hacen en lote
con :func:`costo_lote_ids`.
"""
# Permite escribir tipos como `float | None` en Python < 3.10.
from __future__ import annotations

# Generador de números aleatorios controlado por semilla.
import random
# Medición de tiempo de alta resolución.
import time
# Tipos abstractos para anotaciones de tipo en firmas de funciones.
from collections.abc import Iterable, Mapping
# @dataclass: genera código boilerplate de clases; field: configura atributos con defaults complejos.
from dataclasses import dataclass, field
# Any: cualquier tipo; Literal: conjunto fijo de valores permitidos.
from typing import Any, Literal

# Biblioteca para operaciones con grafos.
import networkx as nx

# Módulos internos del paquete metacarp:
from .busqueda_indices import build_search_encoding, encode_solution  # codificación para evaluación en lote
from .cargar_grafos import cargar_objeto_gexf                         # carga grafo desde GEXF
from .cargar_soluciones_iniciales import cargar_solucion_inicial      # carga solución inicial desde pickle
from .evaluador_costo import (
    costo_lote_penalizado_ids,           # evaluación en lote (GPU o CPU según disponibilidad)
    costo_rapido,                        # evaluación individual rápida con NumPy
    exceso_capacidad_rapido,             # calcula violación de capacidad rápidamente
    lambda_penal_capacidad_por_defecto,  # λ por defecto para penalización
    objectivo_penalizado,                # objetivo combinado: costo + λ × violación
)
from .instances import load_instances  # carga datos de la instancia CARP
from .metaheuristicas_utils import (
    ContadorOperadores,                  # estadísticas de uso de operadores
    calcular_metricas_gap,               # métricas de mejora respecto a la solución inicial
    construir_contexto_para_corrida,     # construye el contexto de evaluación rápida
    copiar_solucion_labels,              # copia soluciones a formato string uniforme
    generar_reporte_detallado,           # genera texto de reporte final
    guardar_resultado_csv,               # escribe fila de resultados en CSV
    resumen_bks_csv,                     # columnas de comparación con BKS
    seleccionar_mejor_inicial_rapido,    # selecciona la mejor solución inicial
    solucion_legible_humana,             # convierte solución a texto legible
)
from .vecindarios import MovimientoVecindario, OPERADORES_POPULARES, generar_vecino  # vecindarios

# API pública del módulo.
__all__ = [
    "AbejasResult",
    "busqueda_abejas",
    "busqueda_abejas_desde_instancia",
]


# --- CONCEPTO OOP: @dataclass(frozen=True, slots=True) ---
# frozen=True: el resultado es inmutable una vez creado (no se puede modificar).
# slots=True: reduce el uso de memoria al no crear __dict__ por instancia.
# Esto es apropiado para un objeto de resultado que solo se lee, nunca se modifica.
@dataclass(frozen=True, slots=True)
class AbejasResult:
    """
    Resultado completo de la metaheurística Artificial Bee Colony (ABC) simplificada.

    Agrupa la mejor solución encontrada, métricas de calidad, tiempo de ejecución
    y estadísticas detalladas de las fases de abejas (empleadas, observadoras, scouts).
    """

    # La mejor solución CARP encontrada (lista de rutas, cada ruta = lista de etiquetas).
    mejor_solucion: list[list[str]]
    # Costo total de la mejor solución (valor a minimizar).
    mejor_costo: float
    # Solución inicial de referencia (para medir la mejora lograda).
    solucion_inicial_referencia: list[list[str]]
    # Costo de la solución inicial.
    costo_solucion_inicial: float
    # Diferencia absoluta: costo_inicial - mejor_costo (positivo = mejora).
    mejora_absoluta: float
    # Porcentaje de mejora respecto al costo inicial.
    mejora_porcentaje_inicial_vs_final: float
    # Tiempo de ejecución en segundos.
    tiempo_segundos: float
    # Total de iteraciones ejecutadas.
    iteraciones_totales: int
    # Número de fuentes de alimento (soluciones candidatas mantenidas en paralelo).
    fuentes_alimento: int
    # Veces que una fuente fue abandonada y reemplazada (fase scout).
    scouts_reinicios: int
    # Veces que el mejor global mejoró.
    mejoras: int
    # Semilla del generador aleatorio.
    semilla: int | None
    # Dispositivo de evaluación usado: 'cpu' o 'gpu'.
    backend_evaluacion: str = "cpu"
    # Historial del mejor costo al inicio de cada iteración.
    historial_mejor_costo: list[float] = field(default_factory=list)
    # Último movimiento aceptado al finalizar la búsqueda.
    ultimo_movimiento_aceptado: MovimientoVecindario | None = None
    # Operadores de vecindario propuestos (generaron un vecino).
    operadores_propuestos: dict[str, int] = field(default_factory=dict)
    # Operadores aceptados (el vecino generado reemplazó la fuente).
    operadores_aceptados: dict[str, int] = field(default_factory=dict)
    # Operadores que mejoraron el mejor global.
    operadores_mejoraron: dict[str, int] = field(default_factory=dict)
    # Snapshot de aceptados en el momento de la mejor solución.
    operadores_trayectoria_mejor: dict[str, int] = field(default_factory=dict)
    # Si True, se usó penalización de capacidad.
    usar_penalizacion_capacidad: bool = True
    # Valor efectivo de λ (peso de penalización por violación de capacidad).
    lambda_capacidad: float = 0.0
    # Número de candidatas iniciales evaluadas.
    n_iniciales_evaluados: int = 0
    # Candidatas iniciales infactibles.
    iniciales_infactibles_aceptadas: int = 0
    # Veces que se aceptó una solución que viola restricciones de capacidad.
    aceptaciones_solucion_infactible: int = 0
    # True si la mejor solución final respeta todas las restricciones.
    mejor_solucion_factible_final: bool = True
    # Ruta del CSV guardado, o None si no se guardó.
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
    """
    Aplica una perturbación local (vecino) a cada solución de la lista 'sources'.

    Por cada fuente de alimento, genera exactamente un vecino usando un operador
    aleatorio. Retorna la lista de vecinos y la lista de movimientos aplicados,
    en el mismo orden que 'sources'.

    Esta función es usada tanto en la fase de empleadas como en la de observadoras
    y scouts, centralizando la lógica de generación de vecinos por lote.
    """
    vecinos: list[list[list[str]]] = []
    movs: list[MovimientoVecindario] = []
    for sol in sources:
        # Generamos un vecino aleatorio de la solución actual.
        v, m = generar_vecino(
            sol, rng=rng, operadores=operadores,
            marcador_depot=marcador_depot, devolver_con_deposito=True,
            usar_gpu=usar_gpu, backend=backend_vecindario, encoding=encoding,
        )
        vecinos.append(v)
        movs.append(m)
    return vecinos, movs


def busqueda_abejas(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    iteraciones: int = 250,          # número de ciclos completos (empleadas + observadoras + scouts)
    num_fuentes: int = 16,           # número de fuentes de alimento (soluciones activas en paralelo)
    limite_abandono: int = 35,       # intentos fallidos antes de abandonar una fuente (fase scout)
    semilla: int | None = None,      # semilla para reproducibilidad
    operadores: Iterable[str] = OPERADORES_POPULARES,  # operadores de vecindario habilitados
    marcador_depot_etiqueta: str | None = None,  # etiqueta del depósito en las rutas
    usar_gpu: bool = False,          # si True, intenta evaluar en GPU
    backend_vecindario: Literal["labels", "ids"] = "labels",  # modo de generación de vecinos
    guardar_historial: bool = True,  # si True, guarda el costo mejor por iteración
    guardar_csv: bool = False,       # si True, escribe resultados en CSV al finalizar
    ruta_csv: str | None = None,     # ruta del CSV (None = nombre automático)
    nombre_instancia: str = "instancia",  # nombre de la instancia para el CSV
    id_corrida: str | None = None,   # identificador de corrida
    config_id: str | None = None,    # identificador de configuración
    repeticion: int | None = None,   # número de repetición del experimento
    root: str | None = None,         # directorio raíz de datos
    usar_penalizacion_capacidad: bool = True,  # si True, penaliza violaciones de capacidad
    lambda_capacidad: float | None = None,     # peso λ de la penalización (None = automático)
    extra_csv: dict[str, object] | None = None,  # columnas adicionales para el CSV
) -> AbejasResult:
    """
    Implementación simplificada de ABC (Artificial Bee Colony) para CARP.

    El algoritmo mantiene 'num_fuentes' soluciones candidatas activas (las "fuentes")
    y en cada iteración las mejora mediante tres fases:

    1. **Empleadas**: generan y evalúan un vecino por cada fuente; actualizan
       la fuente si el vecino es mejor.
    2. **Observadoras**: seleccionan fuentes con probabilidad proporcional a su
       calidad (mejor fuente = más probabilidad) y generan más vecinos en ellas.
    3. **Scouts**: fuentes que no mejoraron en 'limite_abandono' intentos son
       reemplazadas por vecinos aleatorios de la mejor fuente actual.
    """
    # Validaciones rápidas de parámetros.
    if iteraciones <= 0:
        raise ValueError("iteraciones debe ser > 0.")
    if num_fuentes <= 1:
        raise ValueError("num_fuentes debe ser >= 2.")
    if limite_abandono <= 0:
        raise ValueError("limite_abandono debe ser > 0.")

    # Generador de números aleatorios con semilla reproducible.
    rng = random.Random(semilla)
    # Marca de tiempo de inicio de la corrida.
    t0 = time.perf_counter()

    # Construcción del contexto de evaluación rápida (matrices NumPy precomputadas).
    ctx = construir_contexto_para_corrida(
        data, G,
        nombre_instancia=nombre_instancia if nombre_instancia != "instancia" else None,
        usar_gpu=usar_gpu, root=root,
    )

    # λ efectiva: si no se especifica, usamos el valor calculado automáticamente.
    lam_eff = (
        float(lambda_capacidad)
        if lambda_capacidad is not None
        else lambda_penal_capacidad_por_defecto(ctx)
    )

    # Seleccionamos la mejor solución inicial entre todas las candidatas disponibles.
    sel_ini = seleccionar_mejor_inicial_rapido(
        inicial_obj,
        ctx,
        usar_penalizacion_capacidad=usar_penalizacion_capacidad,
        lambda_capacidad=lambda_capacidad,
    )
    sol_ref = sel_ini.solucion          # solución de referencia inicial
    costo_ref = sel_ini.costo_puro      # costo de la referencia inicial
    ini_infact = sel_ini.n_candidatos_infactibles
    n_ini_ev = sel_ini.n_candidatos_evaluados

    # Configuración del encoding para evaluación en lote por ids.
    encoding = ctx.encoding
    if backend_vecindario == "ids" and encoding is None:
        encoding = build_search_encoding(data)

    # Etiqueta del depósito: punto de inicio y fin de cada ruta.
    md_op = marcador_depot_etiqueta or ctx.marcador_depot

    # Rastreo del mejor global (cualquier solución) y del mejor factible.
    mejor_any_c = float(costo_ref)
    mejor_any_s = copiar_solucion_labels(sol_ref)
    # Si la solución inicial es factible, la aceptamos también como la mejor factible.
    if sel_ini.violacion_capacidad < 1e-12:
        mejor_fact_c: float | None = float(costo_ref)
        mejor_fact_s = copiar_solucion_labels(sol_ref)
    else:
        mejor_fact_c = None
        mejor_fact_s = None

    # Función auxiliar que retorna el costo que se reporta al usuario.
    # Preferimos el mejor factible; si no hay ninguno, retornamos el mejor general.
    def costo_para_reporte() -> float:
        return float(mejor_fact_c) if mejor_fact_c is not None else mejor_any_c

    # --- Inicialización de fuentes de alimento ---
    # La primera fuente es la solución inicial de referencia.
    fuentes_sol: list[list[list[str]]] = [copiar_solucion_labels(sol_ref)]
    fuentes_pure: list[float] = [costo_ref]          # costo puro de cada fuente
    fuentes_viol: list[float] = [float(sel_ini.violacion_capacidad)]  # violación de cada fuente
    # 'trials' cuenta cuántos intentos consecutivos ha fallado cada fuente (para fase scout).
    trials: list[int] = [0]

    # Llenamos las fuentes restantes con vecinos aleatorios de la referencia.
    while len(fuentes_sol) < num_fuentes:
        v, _m = generar_vecino(
            sol_ref,
            rng=rng,
            operadores=operadores,
            marcador_depot=md_op,
            devolver_con_deposito=True,
            usar_gpu=usar_gpu,
            backend=backend_vecindario,
            encoding=encoding,
        )
        cp = costo_rapido(v, ctx)            # costo puro del vecino generado
        vp = exceso_capacidad_rapido(v, ctx) # violación de capacidad del vecino
        fuentes_sol.append(v)
        fuentes_pure.append(cp)
        fuentes_viol.append(vp)
        trials.append(0)

    # Función auxiliar que calcula el objetivo penalizado de la fuente i.
    def obj_fuente(i: int) -> float:
        return float(
            objectivo_penalizado(
                fuentes_pure[i],
                fuentes_viol[i],
                usar_penal=usar_penalizacion_capacidad,
                lam=lam_eff,
            )
        )

    def fusionar_desde_fuentes() -> None:
        """
        Actualiza los mejores globales comparando todas las fuentes activas.

        Se llama al final de cada iteración y al cierre de la búsqueda para
        asegurar que el mejor global capture cualquier mejora producida en
        las fases de empleadas, observadoras o scouts.

        'nonlocal' declara que las variables modificadas aquí son las del
        ámbito externo (la función busqueda_abejas), no variables locales nuevas.
        """
        nonlocal mejor_any_c, mejor_any_s, mejor_fact_c, mejor_fact_s, sol_mejor, costo_mejor
        for i in rango_fuentes:
            cp = fuentes_pure[i]
            vv = fuentes_viol[i]
            sol = fuentes_sol[i]
            # Actualizamos el mejor global sin restricción de factibilidad.
            if cp < mejor_any_c - 1e-15:
                mejor_any_c = cp
                mejor_any_s = copiar_solucion_labels(sol)
            # Actualizamos el mejor factible solo si la fuente no viola capacidad.
            if vv < 1e-12:
                lim = mejor_fact_c if mejor_fact_c is not None else float("inf")
                if cp < lim - 1e-15:
                    mejor_fact_c = float(cp)
                    mejor_fact_s = copiar_solucion_labels(sol)
        # Actualizamos las variables de resultado.
        nr = costo_para_reporte()
        sol_mejor = copiar_solucion_labels(
            mejor_fact_s if mejor_fact_s is not None else mejor_any_s
        )
        costo_mejor = nr

    # Inicializamos sol_mejor y costo_mejor para su uso dentro del bucle.
    sol_mejor = copiar_solucion_labels(
        mejor_fact_s if mejor_fact_s is not None else mejor_any_s
    )
    costo_mejor = costo_para_reporte()

    # Contadores de estadísticas.
    mejoras = 0                # veces que el mejor global mejoró
    scouts = 0                 # veces que se ejecutó la fase scout
    historial_best: list[float] = []
    ultimo_mov_aceptado: MovimientoVecindario | None = None
    contador = ContadorOperadores()
    aceptaciones_sol_infactible = 0

    # Lista de índices [0, 1, ..., num_fuentes-1] para iterar sobre las fuentes.
    rango_fuentes = list(range(num_fuentes))

    # === BUCLE PRINCIPAL: cada iteración corresponde a un ciclo completo ABC ===
    for _it in range(iteraciones):
        mejor_rep_antes = costo_para_reporte()

        if guardar_historial:
            historial_best.append(costo_para_reporte())

        # =====================================================================
        # FASE 1: ABEJAS EMPLEADAS
        # Cada abeja empleada genera un vecino de su fuente asignada.
        # Si el vecino es mejor, reemplaza la fuente (exploración greedy local).
        # =====================================================================
        vecinos, movs = _generar_vecinos_lote(
            fuentes_sol,
            rng=rng,
            operadores=operadores,
            marcador_depot=md_op,
            usar_gpu=usar_gpu,
            backend_vecindario=backend_vecindario,
            encoding=encoding,
        )
        # Registramos que estos operadores fueron propuestos.
        for m in movs:
            contador.proponer(m.operador)

        # Evaluamos todos los vecinos de empleadas en lote.
        sols_em = [encode_solution(v, ctx.encoding) for v in vecinos]
        objs_np, bases_np, viols_np = costo_lote_penalizado_ids(
            sols_em,
            ctx,
            lam_eff,
            usar_penal=usar_penalizacion_capacidad,
        )
        # Para cada fuente, comparamos su vecino con la fuente actual.
        for i in range(num_fuentes):
            o_nei = float(objs_np[i])  # objetivo penalizado del vecino
            o_old = obj_fuente(i)      # objetivo penalizado de la fuente actual
            if o_nei < o_old - 1e-15:
                # El vecino es mejor: actualizamos la fuente y reseteamos el contador de intentos.
                fuentes_sol[i] = vecinos[i]
                fuentes_pure[i] = float(bases_np[i])
                fuentes_viol[i] = float(viols_np[i])
                trials[i] = 0
                ultimo_mov_aceptado = movs[i]
                contador.aceptar(movs[i].operador)
                if fuentes_viol[i] > 1e-12:
                    aceptaciones_sol_infactible += 1
            else:
                # El vecino no mejoró: incrementamos el contador de intentos fallidos.
                trials[i] += 1

        # =====================================================================
        # FASE 2: ABEJAS OBSERVADORAS
        # Las observadoras eligen fuentes con probabilidad proporcional a su
        # calidad (fitness). Las mejores fuentes reciben más visitas.
        #
        # Fitness = 1 / (1 + objetivo_penalizado)
        # Cuanto menor el objetivo, mayor el fitness, mayor la probabilidad.
        # =====================================================================
        objs_fit = [obj_fuente(i) for i in rango_fuentes]
        # Calculamos la suma de fitness inversos para normalizar a probabilidades.
        # max(o_f, 0.0) evita divisiones por negativo en casos excepcionales.
        total_inv = sum(1.0 / (1.0 + max(o_f, 0.0)) for o_f in objs_fit)
        if total_inv <= 0:
            # Fallback: probabilidades uniformes si no se puede calcular fitness.
            probs = [1.0 / num_fuentes] * num_fuentes
        else:
            # Probabilidad de cada fuente = su_fitness / suma_total_fitness
            # (distribución de probabilidad válida: suma = 1).
            probs = [(1.0 / (1.0 + max(o_f, 0.0))) / total_inv for o_f in objs_fit]
        # Seleccionamos 'num_fuentes' fuentes con reemplazo, ponderadas por probabilidad.
        # rng.choices permite seleccionar la misma fuente varias veces (sesgo hacia las buenas).
        idxs = rng.choices(rango_fuentes, weights=probs, k=num_fuentes)

        # Generamos vecinos de las fuentes seleccionadas por las observadoras.
        srcs = [fuentes_sol[i] for i in idxs]
        vecinos, movs = _generar_vecinos_lote(
            srcs,
            rng=rng,
            operadores=operadores,
            marcador_depot=md_op,
            usar_gpu=usar_gpu,
            backend_vecindario=backend_vecindario,
            encoding=encoding,
        )
        for m in movs:
            contador.proponer(m.operador)
        sols_ob = [encode_solution(v, ctx.encoding) for v in vecinos]
        objs_np, bases_np, viols_np = costo_lote_penalizado_ids(
            sols_ob,
            ctx,
            lam_eff,
            usar_penal=usar_penalizacion_capacidad,
        )
        # Comparamos cada vecino con su fuente original (no con la observadora).
        for k in range(num_fuentes):
            i_fuente = idxs[k]           # índice real de la fuente visitada
            o_nei = float(objs_np[k])
            o_old = obj_fuente(i_fuente)
            if o_nei < o_old - 1e-15:
                fuentes_sol[i_fuente] = vecinos[k]
                fuentes_pure[i_fuente] = float(bases_np[k])
                fuentes_viol[i_fuente] = float(viols_np[k])
                trials[i_fuente] = 0
                ultimo_mov_aceptado = movs[k]
                contador.aceptar(movs[k].operador)
                if fuentes_viol[i_fuente] > 1e-12:
                    aceptaciones_sol_infactible += 1
            else:
                trials[i_fuente] += 1

        # =====================================================================
        # FASE 3: ABEJAS SCOUT (EXPLORADORAS)
        # Fuentes que superaron el límite de intentos fallidos se abandonan y
        # se reemplazan con nuevas soluciones en torno a la mejor fuente actual.
        #
        # Analogía biológica: si una flor se agota, la abeja busca una nueva
        # fuente de alimento en una zona diferente (diversificación).
        # =====================================================================
        # Encontramos la fuente de mejor calidad (mínimo objetivo penalizado).
        best_idx = min(rango_fuentes, key=obj_fuente)
        base_best = fuentes_sol[best_idx]
        # Identificamos las fuentes que superaron el límite de intentos fallidos.
        a_reiniciar = [i for i in rango_fuentes if trials[i] >= limite_abandono]
        if a_reiniciar:
            # Generamos vecinos de la mejor fuente como "nuevas fuentes de alimento".
            srcs = [base_best] * len(a_reiniciar)
            vecinos, movs_sc = _generar_vecinos_lote(
                srcs,
                rng=rng,
                operadores=operadores,
                marcador_depot=md_op,
                usar_gpu=usar_gpu,
                backend_vecindario=backend_vecindario,
                encoding=encoding,
            )
            for m in movs_sc:
                contador.proponer(m.operador)
            sols_sc = [encode_solution(v, ctx.encoding) for v in vecinos]
            _, bases_np, viols_np = costo_lote_penalizado_ids(
                sols_sc,
                ctx,
                lam_eff,
                usar_penal=usar_penalizacion_capacidad,
            )
            # Reemplazamos cada fuente agotada por su nueva solución scout.
            for k, i in enumerate(a_reiniciar):
                fuentes_sol[i] = vecinos[k]
                fuentes_pure[i] = float(bases_np[k])
                fuentes_viol[i] = float(viols_np[k])
                trials[i] = 0  # reseteamos el contador al reiniciar la fuente
                scouts += 1
                contador.aceptar(movs_sc[k].operador)
                if fuentes_viol[i] > 1e-12:
                    aceptaciones_sol_infactible += 1

        # Actualizamos los mejores globales comparando todas las fuentes.
        fusionar_desde_fuentes()
        mejor_rep_despues = costo_para_reporte()
        # Si esta iteración produjo una mejora global, la registramos.
        if mejor_rep_despues + 1e-12 < mejor_rep_antes:
            mejoras += 1
            op_mejor = (
                ultimo_mov_aceptado.operador if ultimo_mov_aceptado else None
            )
            contador.registrar_mejora(op_mejor)

    # === FIN DEL BUCLE PRINCIPAL ===

    # Tiempo total de la corrida.
    elapsed = time.perf_counter() - t0
    # Fusión final para capturar cualquier mejora de la última iteración.
    fusionar_desde_fuentes()
    costo_mejor = costo_para_reporte()
    sol_mejor = copiar_solucion_labels(
        mejor_fact_s if mejor_fact_s is not None else mejor_any_s
    )
    # True si la mejor solución final es factible (respeta capacidades).
    mejor_factible_final = mejor_fact_s is not None
    # _gap_descartado: calculamos gap pero no lo usamos en el resultado (convención del módulo).
    _gap_descartado, mejora_abs, mejora_pct = calcular_metricas_gap(costo_ref, costo_mejor)

    # --- Guardado en CSV (opcional) ---
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
            "usar_penalizacion_capacidad": usar_penalizacion_capacidad,
            "lambda_capacidad": lam_eff,
            "n_iniciales_evaluados": n_ini_ev,
            "iniciales_infactibles_aceptadas": ini_infact,
            "aceptaciones_solucion_infactible": aceptaciones_sol_infactible,
            "mejor_solucion_factible_final": mejor_factible_final,
            "costo_solucion_inicial": costo_ref,
            "mejor_costo": costo_mejor,
            "mejora_absoluta": mejora_abs,
            "mejora_porcentaje_inicial_vs_final": mejora_pct,
            **resumen_bks_csv(data, costo_mejor),
            "mejor_solucion_tr_legible": solucion_legible_humana(sol_mejor),
            "reporte_detalle_deadheading": detalle_txt,
            "costo_total_desde_reporte": costo_total_reporte,
            **contador.resumen_csv(),
            **(extra_csv or {}),
        }
        archivo_csv = guardar_resultado_csv(fila=fila, ruta_csv=ruta)

    # Retornamos el objeto de resultado inmutable con todos los datos de la corrida.
    return AbejasResult(
        mejor_solucion=sol_mejor,
        mejor_costo=costo_mejor,
        solucion_inicial_referencia=sol_ref,
        costo_solucion_inicial=costo_ref,
        mejora_absoluta=mejora_abs,
        mejora_porcentaje_inicial_vs_final=mejora_pct,
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
        usar_penalizacion_capacidad=usar_penalizacion_capacidad,
        lambda_capacidad=lam_eff,
        n_iniciales_evaluados=n_ini_ev,
        iniciales_infactibles_aceptadas=ini_infact,
        aceptaciones_solucion_infactible=aceptaciones_sol_infactible,
        mejor_solucion_factible_final=mejor_factible_final,
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
    usar_penalizacion_capacidad: bool = True,
    lambda_capacidad: float | None = None,
    extra_csv: dict[str, object] | None = None,
) -> AbejasResult:
    """
    Función de conveniencia que carga todos los recursos necesarios desde el
    nombre de la instancia y ejecuta ABC completo.

    Equivalente a llamar manualmente a load_instances + cargar_objeto_gexf
    + cargar_solucion_inicial + busqueda_abejas.
    """
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
        usar_penalizacion_capacidad=usar_penalizacion_capacidad,
        lambda_capacidad=lambda_capacidad,
        extra_csv=extra_csv,
    )
