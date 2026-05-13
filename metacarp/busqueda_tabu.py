"""
Búsqueda Tabú clásica con memoria de corto plazo.

Concepto algorítmico
--------------------
La Búsqueda Tabú es una metaheurística que navega el espacio de soluciones
moviéndose siempre al mejor vecino disponible, incluso si ese vecino es peor que
la solución actual. Para evitar ciclos (regresar infinitamente a los mismos
estados), mantiene una "lista tabú": una memoria de movimientos recientes que
quedan prohibidos temporalmente.

Inspiración: el algoritmo evita quedarse atrapado en mínimos locales al "recordar"
que ciertos caminos ya fueron explorados, forzando la búsqueda hacia zonas nuevas.

Optimización de evaluación
--------------------------
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
# Permite usar anotaciones de tipo como `float | None` en Python < 3.10.
from __future__ import annotations

# Módulo estándar para generar números aleatorios de forma reproducible.
import random
# Módulo para medir el tiempo transcurrido con alta precisión.
import time
# Tipos abstractos para anotaciones de función.
from collections.abc import Iterable, Mapping
# @dataclass y field: para definir clases de datos sin escribir __init__ manualmente.
from dataclasses import dataclass, field
# Any = cualquier tipo; Literal = un conjunto fijo de valores permitidos.
from typing import Any, Literal

# NetworkX: biblioteca para manipular grafos (nodos y aristas).
import networkx as nx

# Importaciones internas del paquete:
from .busqueda_indices import build_search_encoding, encode_solution  # codificación de soluciones para evaluación en lote
from .cargar_grafos import cargar_objeto_gexf                         # carga el grafo desde archivo GEXF
from .cargar_soluciones_iniciales import cargar_solucion_inicial      # carga la solución inicial desde pickle
from .evaluador_costo import (
    costo_lote_penalizado_ids,           # evalúa un lote de soluciones en GPU o CPU rápido
    exceso_capacidad_rapido,             # calcula la violación de capacidad de una solución
    lambda_penal_capacidad_por_defecto,  # valor λ de penalización por defecto
)
from .instances import load_instances  # carga los datos de la instancia CARP
from .metaheuristicas_utils import (
    ContadorOperadores,                  # cuenta propuestas/aceptaciones por operador
    calcular_metricas_gap,               # calcula el gap porcentual de mejora
    construir_contexto_para_corrida,     # construye el contexto de evaluación rápida
    copiar_solucion_labels,              # copia una solución a formato de strings
    generar_reporte_detallado,           # genera el texto de reporte final
    guardar_resultado_csv,               # persiste la fila de resultados en CSV
    pesos_inter_bias,                    # pesos de sesgo inter-ruta por violación de capacidad
    resumen_bks_csv,                     # extrae columnas de comparación con BKS
    seleccionar_mejor_inicial_rapido,    # elige la mejor solución inicial
    solucion_legible_humana,             # convierte la solución a texto legible
)
from .vecindarios import MovimientoVecindario, OPERADORES_POPULARES, generar_vecino  # generación de vecinos

# Declara la API pública de este módulo.
__all__ = [
    "BusquedaTabuResult",
    "busqueda_tabu",
    "busqueda_tabu_desde_instancia",
]


# --- CONCEPTO OOP: @dataclass(frozen=True, slots=True) ---
# frozen=True: el objeto es inmutable una vez creado (no se puede modificar ningún atributo).
#              Esto es seguro porque el resultado de la búsqueda no debe cambiar después de
#              terminar la ejecución.
# slots=True: Python reserva exactamente los atributos declarados, reduciendo el uso de memoria.
@dataclass(frozen=True, slots=True)
class BusquedaTabuResult:
    """
    Resultado completo de la búsqueda tabú.

    Agrupa la solución encontrada, las métricas de calidad, información de
    tiempo y estadísticas de operadores. Se devuelve al finalizar la corrida.
    """

    # La mejor solución encontrada (lista de rutas, cada ruta = lista de etiquetas).
    mejor_solucion: list[list[str]]
    # Costo total de la mejor solución (objetivo a minimizar).
    mejor_costo: float
    # La solución inicial tal como fue seleccionada (para comparar mejora).
    solucion_inicial_referencia: list[list[str]]
    # Costo de la solución inicial (referencia de mejora).
    costo_solucion_inicial: float
    # Diferencia absoluta: costo_inicial - mejor_costo (positivo = mejora).
    mejora_absoluta: float
    # Mejora expresada como porcentaje del costo inicial.
    mejora_porcentaje_inicial_vs_final: float
    # Tiempo total de ejecución en segundos.
    tiempo_segundos: float
    # Número total de iteraciones ejecutadas.
    iteraciones_totales: int
    # Total de soluciones vecinas evaluadas durante toda la búsqueda.
    vecinos_evaluados: int
    # Veces que el mejor vecino disponible era tabú y hubo que ignorarlo.
    movimientos_tabu_bloqueados: int
    # Número de veces que el mejor global mejoró durante la búsqueda.
    mejoras: int
    # Semilla del generador de números aleatorios (para reproducibilidad).
    semilla: int | None
    # Dispositivo usado para la evaluación: 'cpu' o 'gpu'.
    backend_evaluacion: str = "cpu"
    # Historial del mejor costo registrado al inicio de cada iteración.
    historial_mejor_costo: list[float] = field(default_factory=list)
    # El último movimiento que fue aceptado al final de la búsqueda.
    ultimo_movimiento_aceptado: MovimientoVecindario | None = None
    # Conteo de veces que cada operador fue propuesto para generar un vecino.
    operadores_propuestos: dict[str, int] = field(default_factory=dict)
    # Conteo de veces que cada operador generó un vecino que fue aceptado.
    operadores_aceptados: dict[str, int] = field(default_factory=dict)
    # Conteo de operadores que mejoraron el mejor global.
    operadores_mejoraron: dict[str, int] = field(default_factory=dict)
    # Snapshot de operadores aceptados en el momento de la mejor solución.
    operadores_trayectoria_mejor: dict[str, int] = field(default_factory=dict)
    # Si True, se usó penalización de capacidad durante la búsqueda.
    usar_penalizacion_capacidad: bool = True
    # Valor efectivo de λ usado para penalizar violaciones de capacidad.
    lambda_capacidad: float = 0.0
    # Candidatas iniciales que fueron evaluadas.
    n_iniciales_evaluados: int = 0
    # De esas candidatas, cuántas violaban restricciones de capacidad.
    iniciales_infactibles_aceptadas: int = 0
    # Veces que se aceptó una solución infactible durante la búsqueda.
    aceptaciones_solucion_infactible: int = 0
    # True si la mejor solución final es factible (respeta capacidad).
    mejor_solucion_factible_final: bool = True
    # Ruta del archivo CSV donde se guardaron los resultados (o None si no se guardó).
    archivo_csv: str | None = None


def _clave_tabu(mov: MovimientoVecindario) -> tuple[Any, ...]:
    """
    Genera una clave hashable que identifica unívocamente un movimiento.

    La clave se usa como índice en el diccionario de memoria tabú.
    Un movimiento queda "tabú" cuando su clave está registrada en ese dict.

    Incluye: operador, índices de rutas afectadas, posiciones y etiquetas movidas.
    Dos movimientos son considerados el mismo si producen exactamente la misma
    perturbación estructural en la solución.
    """
    return (
        mov.operador,           # nombre del operador de vecindario (ej: 'swap_intra')
        mov.ruta_a, mov.ruta_b, # índices de las rutas involucradas
        mov.i, mov.j, mov.k, mov.l,  # posiciones dentro de las rutas
        tuple(mov.labels_movidos),   # etiquetas de las tareas que se mueven
    )


def busqueda_tabu(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    iteraciones: int = 400,          # número total de iteraciones de búsqueda
    tam_vecindario: int = 25,        # vecinos generados por iteración
    tenure_tabu: int = 20,           # cuántas iteraciones permanece un movimiento en la lista tabú
    semilla: int | None = None,      # semilla para reproducibilidad
    operadores: Iterable[str] = OPERADORES_POPULARES,  # operadores de vecindario habilitados
    marcador_depot_etiqueta: str | None = None,  # etiqueta del nodo depósito en la solución
    usar_gpu: bool = False,          # si True, intenta usar GPU para evaluación en lote
    backend_vecindario: Literal["labels", "ids"] = "labels",  # modo de generación de vecinos
    guardar_historial: bool = True,  # si True, guarda el historial de costos por iteración
    guardar_csv: bool = False,       # si True, escribe resultados al CSV al finalizar
    ruta_csv: str | None = None,     # ruta del archivo CSV (None = nombre automático)
    nombre_instancia: str = "instancia",  # nombre de la instancia para el CSV
    id_corrida: str | None = None,   # identificador de corrida para el CSV
    config_id: str | None = None,    # identificador de configuración para el CSV
    repeticion: int | None = None,   # número de repetición dentro de un experimento
    root: str | None = None,         # directorio raíz de los datos
    usar_penalizacion_capacidad: bool = True,  # si True, penaliza soluciones infactibles
    lambda_capacidad: float | None = None,     # peso λ de la penalización
    extra_csv: dict[str, object] | None = None,  # columnas adicionales para el CSV
    alpha_inter: float = 0.8,  # fracción de prob. asignada a ops inter-ruta cuando hay violación
) -> BusquedaTabuResult:
    """
    Búsqueda Tabú clásica con memoria de corto plazo (short-term memory).

    Estrategia general:
    1. Parte de la mejor solución inicial disponible.
    2. En cada iteración genera 'tam_vecindario' vecinos y elige el mejor
       que no sea tabú (o que cumpla el criterio de aspiración).
    3. El movimiento elegido se marca tabú por 'tenure_tabu' iteraciones.
    4. Se actualiza el mejor global si se encontró una solución mejorada.

    Criterio de aspiración: si el vecino tabú tiene un costo estrictamente mejor
    que el mejor histórico registrado, se acepta igualmente (se "aspira" a él).
    Esto evita que la lista tabú bloquee un salto al óptimo global.
    """
    # Validaciones de parámetros: fallan rápido si los valores son inválidos.
    if iteraciones <= 0:
        raise ValueError("iteraciones debe ser > 0.")
    if tam_vecindario <= 0:
        raise ValueError("tam_vecindario debe ser > 0.")
    if tenure_tabu <= 0:
        raise ValueError("tenure_tabu debe ser > 0.")

    # Generador de números aleatorios con semilla fija para reproducibilidad.
    rng = random.Random(semilla)
    # Marca de tiempo de inicio (perf_counter es más preciso que time.time).
    t0 = time.perf_counter()

    # Construimos el contexto de evaluación rápida (matrices NumPy precomputadas).
    ctx = construir_contexto_para_corrida(
        data, G,
        nombre_instancia=nombre_instancia if nombre_instancia != "instancia" else None,
        usar_gpu=usar_gpu, root=root,
    )

    # Lambda efectiva: si no se especificó, usamos el valor por defecto del contexto.
    lam_eff = (
        float(lambda_capacidad)
        if lambda_capacidad is not None
        else lambda_penal_capacidad_por_defecto(ctx)
    )

    # Seleccionamos la mejor solución inicial del objeto provisto (puede ser un dict con varias).
    sel_ini = seleccionar_mejor_inicial_rapido(
        inicial_obj,
        ctx,
        usar_penalizacion_capacidad=usar_penalizacion_capacidad,
        lambda_capacidad=lambda_capacidad,
    )
    sol_ref = sel_ini.solucion          # solución inicial de referencia (para calcular mejora)
    costo_ref = sel_ini.costo_puro      # costo de esa solución inicial
    ini_infact = sel_ini.n_candidatos_infactibles  # candidatas infactibles ignoradas
    n_ini_ev = sel_ini.n_candidatos_evaluados      # total de candidatas evaluadas

    # La solución "actual" es la que el algoritmo modifica en cada iteración.
    sol_actual = copiar_solucion_labels(sol_ref)
    costo_actual = costo_ref
    viol_actual = sel_ini.violacion_capacidad  # exceso de demanda (0 si es factible)

    # Rastreamos dos mejores: la mejor en general (cualquier) y la mejor factible.
    # Esto permite reportar la mejor factible al final incluso si se exploraron infactibles.
    mejor_any_c = float(costo_ref)
    mejor_any_s = copiar_solucion_labels(sol_ref)
    # Si la solución inicial es factible, la registramos también como la mejor factible.
    if viol_actual < 1e-12:  # umbral numérico: violación ≈ 0 significa factible
        mejor_fact_c: float | None = float(costo_ref)
        mejor_fact_s = copiar_solucion_labels(sol_ref)
    else:
        mejor_fact_c = None
        mejor_fact_s = None

    # Codificación para backend de ids (permite evaluación en lote más eficiente).
    encoding = ctx.encoding
    if backend_vecindario == "ids" and encoding is None:
        encoding = build_search_encoding(data)

    # --- ESTRUCTURA DE DATOS: Lista Tabú ---
    # Implementada como un diccionario: {clave_movimiento: iteración_en_que_vence_el_tabu}.
    # Un movimiento está tabú si tabu_hasta[clave] > iteración_actual.
    # Usar un dict es más eficiente que una lista porque la búsqueda es O(1).
    tabu_hasta: dict[tuple[Any, ...], int] = {}

    # Contadores de estadísticas para el resultado final.
    vecinos_evaluados = 0  # total de vecinos evaluados durante toda la búsqueda
    bloqueados = 0         # veces que el mejor admisible era tabú
    mejoras = 0            # veces que el mejor global mejoró
    ultimo_mov_aceptado: MovimientoVecindario | None = None
    historial_best: list[float] = []
    contador = ContadorOperadores()
    aceptaciones_sol_infactible = 0  # veces que se aceptó una solución que viola capacidad

    # Etiqueta del depósito: se usa para que los operadores de vecindario sepan
    # qué nodo es el punto de inicio/fin de cada ruta.
    md_op = marcador_depot_etiqueta or ctx.marcador_depot

    # Función auxiliar: retorna el costo que se reporta externamente.
    # Preferimos el mejor factible si existe; si no, usamos el mejor general.
    def costo_para_reporte() -> float:
        return float(mejor_fact_c) if mejor_fact_c is not None else mejor_any_c

    # Inicializamos sol_mejor y costo_mejor para usarlos dentro del bucle.
    sol_mejor = copiar_solucion_labels(
        mejor_fact_s if mejor_fact_s is not None else mejor_any_s
    )
    costo_mejor = costo_para_reporte()

    # === BUCLE PRINCIPAL DE BÚSQUEDA TABÚ ===
    for it in range(iteraciones):
        if guardar_historial:
            # Registramos el mejor costo al inicio de esta iteración.
            historial_best.append(costo_para_reporte())

        # --- Paso 1: Generación del lote de vecinos ---
        # Generamos 'tam_vecindario' vecinos y sus movimientos correspondientes.
        vecinos: list[list[list[str]]] = []
        movimientos: list[MovimientoVecindario] = []
        pesos_ops = pesos_inter_bias(viol_actual, list(operadores), alpha_inter=alpha_inter)
        for _ in range(tam_vecindario):
            vecino, mov = generar_vecino(
                sol_actual,
                rng=rng,
                operadores=operadores,
                pesos_operadores=pesos_ops,
                marcador_depot=md_op,
                devolver_con_deposito=True,   # incluye el nodo depósito al inicio y fin de ruta
                usar_gpu=usar_gpu,
                backend=backend_vecindario,
                encoding=encoding,
            )
            vecinos.append(vecino)
            movimientos.append(mov)
            contador.proponer(mov.operador)  # registra que este operador fue propuesto

        # --- Paso 2: Evaluación en lote ---
        # Convertimos las soluciones a formato de ids para la evaluación eficiente.
        sols_ids = [encode_solution(v, ctx.encoding) for v in vecinos]
        # Evaluamos todos los vecinos a la vez: objs (objetivo penalizado), bases (costo puro),
        # viols (violación de capacidad) son arrays NumPy.
        objs_np, bases_np, viols_np = costo_lote_penalizado_ids(
            sols_ids,
            ctx,
            lam_eff,
            usar_penal=usar_penalizacion_capacidad,
        )
        vecinos_evaluados += len(vecinos)

        # El costo de referencia para el criterio de aspiración es el mejor reportable actual.
        rep_best = costo_para_reporte()

        # --- Paso 3: Selección del mejor vecino admisible (que no sea tabú) ---
        mejor_admisible_idx = -1         # índice del mejor vecino no-tabú
        mejor_admisible_obj = float("inf")
        mejor_total_idx = 0              # índice del mejor vecino sin importar tabú
        mejor_total_obj = float("inf")

        for idx in range(len(objs_np)):
            o_obj = float(objs_np[idx])   # objetivo penalizado de este vecino
            c_p = float(bases_np[idx])    # costo puro de este vecino (sin penalización)

            # Actualizamos el mejor total (incluyendo tabú).
            if o_obj < mejor_total_obj:
                mejor_total_obj = o_obj
                mejor_total_idx = idx

            # Verificamos si este movimiento está en la lista tabú.
            key = _clave_tabu(movimientos[idx])
            es_tabu = tabu_hasta.get(key, -1) > it  # True si el movimiento está prohibido

            # Criterio de aspiración: se acepta un movimiento tabú si su costo puro
            # es estrictamente mejor que el mejor global conocido.
            aspiracion = c_p < rep_best - 1e-15

            # Si es tabú y no cumple la aspiración, lo saltamos.
            if es_tabu and not aspiracion:
                continue

            # Si llegamos aquí, el vecino es admisible (no tabú, o tabú pero con aspiración).
            if o_obj < mejor_admisible_obj:
                mejor_admisible_obj = o_obj
                mejor_admisible_idx = idx

        # --- Paso 4: Elección del movimiento a ejecutar ---
        if mejor_admisible_idx == -1:
            # Todos los vecinos eran tabú: elegimos el mejor global (forzamos salida del ciclo).
            elegido_idx = mejor_total_idx
            bloqueados += len(objs_np)
        else:
            elegido_idx = mejor_admisible_idx
            # Contamos cuántos movimientos tabú fueron ignorados en esta iteración.
            for idx, mov in enumerate(movimientos):
                key = _clave_tabu(mov)
                if tabu_hasta.get(key, -1) > it:
                    bloqueados += 1

        # Información sobre la solución elegida.
        viol_sel = float(viols_np[elegido_idx])
        if viol_sel > 1e-12:
            aceptaciones_sol_infactible += 1  # la solución elegida viola capacidad

        # Actualizamos el estado actual al vecino elegido.
        sol_actual = vecinos[elegido_idx]
        costo_actual = float(bases_np[elegido_idx])
        viol_actual = viol_sel
        ultimo_mov_aceptado = movimientos[elegido_idx]
        contador.aceptar(ultimo_mov_aceptado.operador)

        # --- Paso 5: Actualización de la lista tabú ---
        # El movimiento elegido se prohíbe por 'tenure_tabu' iteraciones.
        tabu_hasta[_clave_tabu(ultimo_mov_aceptado)] = it + tenure_tabu

        # Cada 25 iteraciones limpiamos entradas vencidas del dict tabú para ahorrar memoria.
        if it % 25 == 0 and tabu_hasta:
            for k in [k for k, vence in tabu_hasta.items() if vence <= it]:
                del tabu_hasta[k]

        # --- Paso 6: Actualización del mejor global ---
        antes_rep = costo_para_reporte()
        # Actualizamos el mejor sin restricción de factibilidad.
        if costo_actual < mejor_any_c - 1e-15:
            mejor_any_c = costo_actual
            mejor_any_s = copiar_solucion_labels(sol_actual)
        # Actualizamos el mejor factible solo si la solución no viola capacidad.
        if viol_actual < 1e-12:
            lim_fact = mejor_fact_c if mejor_fact_c is not None else float("inf")
            if costo_actual < lim_fact - 1e-15:
                mejor_fact_c = float(costo_actual)
                mejor_fact_s = copiar_solucion_labels(sol_actual)
        despues_rep = costo_para_reporte()
        # Si el mejor reportable mejoró, lo registramos como una mejora.
        if despues_rep < antes_rep - 1e-12:
            mejoras += 1
            contador.registrar_mejora(ultimo_mov_aceptado.operador)
        # Actualizamos sol_mejor y costo_mejor al final de cada iteración.
        sol_mejor = copiar_solucion_labels(
            mejor_fact_s if mejor_fact_s is not None else mejor_any_s
        )
        costo_mejor = despues_rep

    # === FIN DEL BUCLE PRINCIPAL ===

    # Calculamos el tiempo total transcurrido.
    elapsed = time.perf_counter() - t0
    # Tomamos el mejor reportable final.
    costo_mejor = costo_para_reporte()
    sol_mejor = copiar_solucion_labels(
        mejor_fact_s if mejor_fact_s is not None else mejor_any_s
    )
    # Indicamos si la mejor solución final es factible.
    mejor_factible_final = mejor_fact_s is not None
    # _gap_descartado: el gap porcentual no se usa en el resultado, pero sí mejora_abs y mejora_pct.
    _gap_descartado, mejora_abs, mejora_pct = calcular_metricas_gap(costo_ref, costo_mejor)

    # --- Guardado en CSV (opcional) ---
    archivo_csv: str | None = None
    if guardar_csv:
        ruta = ruta_csv or f"resultados_busqueda_tabu_{nombre_instancia}.csv"
        # Generamos el reporte detallado solo al final (costoso, no en el bucle).
        detalle_txt, costo_total_reporte = generar_reporte_detallado(
            sol_mejor, data, G,
            nombre_instancia=nombre_instancia,
            marcador_depot_etiqueta=marcador_depot_etiqueta,
            usar_gpu=False,
        )
        # Construimos el diccionario de la fila con todas las columnas del CSV.
        _bks = resumen_bks_csv(data, costo_mejor)
        fila = {
            "metaheuristica": "busqueda_tabu",
            "instancia": nombre_instancia,
            "bks_referencia": _bks["bks_referencia"],
            "bks_origen": _bks["bks_origen"],
            "gap_bks_porcentaje": _bks["gap_bks_porcentaje"],
            "repeticion": repeticion if repeticion is not None else "",
            "semilla": semilla,
            "tiempo_segundos": elapsed,
            "mejor_costo": costo_mejor,
            "costo_solucion_inicial": costo_ref,
            "iteraciones": iteraciones,
            "tam_vecindario": tam_vecindario,
            "tenure_tabu": tenure_tabu,
            **contador.resumen_csv(),
            "aceptadas": sum(contador.aceptados.values()),
            "mejoras": mejoras,
            "mejor_solucion_factible_final": mejor_factible_final,
            "mejor_solucion_tr_legible": solucion_legible_humana(sol_mejor),
            "reporte_detalle_deadheading": detalle_txt,
            "costo_total_desde_reporte": costo_total_reporte,
        }
        archivo_csv = guardar_resultado_csv(fila=fila, ruta_csv=ruta)

    # Construimos y retornamos el objeto de resultado inmutable.
    return BusquedaTabuResult(
        mejor_solucion=sol_mejor,
        mejor_costo=costo_mejor,
        solucion_inicial_referencia=sol_ref,
        costo_solucion_inicial=costo_ref,
        mejora_absoluta=mejora_abs,
        mejora_porcentaje_inicial_vs_final=mejora_pct,
        tiempo_segundos=elapsed,
        iteraciones_totales=iteraciones,
        vecinos_evaluados=vecinos_evaluados,
        movimientos_tabu_bloqueados=bloqueados,
        mejoras=mejoras,
        semilla=semilla,
        backend_evaluacion=ctx.backend_real,
        historial_mejor_costo=historial_best,
        ultimo_movimiento_aceptado=ultimo_mov_aceptado,
        # como_dict_ordenado convierte Counter a dict ordenado por frecuencia descendente.
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
    usar_penalizacion_capacidad: bool = True,
    lambda_capacidad: float | None = None,
    extra_csv: dict[str, object] | None = None,
    alpha_inter: float = 0.8,
) -> BusquedaTabuResult:
    """
    Función de conveniencia: carga todos los recursos necesarios desde el nombre
    de la instancia y ejecuta la búsqueda tabú completa.

    Es la forma más simple de usar la búsqueda tabú: solo se necesita el nombre
    de la instancia (ej: 'gdb1') y los parámetros del algoritmo.
    """
    # Cargamos los datos de la instancia (capacidad, demandas, BKS, etc.).
    data = load_instances(nombre_instancia, root=root)
    # Cargamos el grafo de la instancia desde el archivo GEXF.
    G = cargar_objeto_gexf(nombre_instancia, root=root)
    # Cargamos el objeto de solución inicial desde el archivo pickle.
    inicial_obj = cargar_solucion_inicial(nombre_instancia, root=root)
    # Ejecutamos la búsqueda tabú con todos los parámetros.
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
        usar_penalizacion_capacidad=usar_penalizacion_capacidad,
        lambda_capacidad=lambda_capacidad,
        extra_csv=extra_csv,
        alpha_inter=alpha_inter,
    )
