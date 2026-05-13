"""
Cuckoo Search adaptado a espacio discreto para CARP.

Concepto algorítmico
--------------------
Cuckoo Search es una metaheurística inspirada en el parasitismo de nidificación
del cucú (cuckoo): el cucú pone sus huevos en los nidos de otras aves. Si el
huevo es detectado como extraño, el huésped lo abandona y construye un nido nuevo.

Trasladado a optimización:
- Los **nidos** representan soluciones candidatas.
- Los **huevos de cucú** son nuevas soluciones generadas mediante "vuelo Lévy".
- Si la nueva solución (huevo) es mejor que la del nido que compite, **reemplaza**
  al nido (el huésped "acepta" el huevo).
- Con probabilidad ``pa_abandono``, los **peores nidos** son abandonados y
  reemplazados por nuevas soluciones en torno al mejor nido (el cucú construye
  un nuevo nido en otra zona).

Vuelo de Lévy
-------------
El vuelo de Lévy es un tipo de movimiento aleatorio con pasos de longitud
variable. A diferencia de un paseo aleatorio ordinario (pasos cortos uniformes),
el vuelo de Lévy combina muchos pasos cortos con saltos ocasionales muy largos.
Esta característica lo hace ideal para exploración: permite escapar de mínimos
locales con "saltos" hacia zonas lejanas del espacio de búsqueda.

En espacio continuo: el paso sigue una distribución de Lévy (heavy-tail).
En espacio discreto (rutas CARP): lo aproximamos aplicando múltiples
perturbaciones locales consecutivas, donde el número de perturbaciones sigue
una distribución Lévy discretizada.

Optimización
------------
Construye un :class:`ContextoEvaluacion` una vez y evalúa con
:func:`costo_rapido`. Cuando ``usar_gpu=True`` y CuPy está disponible, los
``num_nidos`` cuckoos generados por iteración se evalúan en lote en GPU vía
:func:`costo_lote_penalizado_ids`.
"""
# Permite usar `float | None` en Python < 3.10.
from __future__ import annotations

# math.floor y math.floor se usan para el cálculo de vuelo Lévy discreto.
import math
# Generador de números aleatorios controlado.
import random
# Medición de tiempo de alta precisión.
import time
# Tipos abstractos para firmas de funciones.
from collections.abc import Iterable, Mapping
# Soporte de dataclasses con atributos de valor por defecto complejos.
from dataclasses import dataclass, field
# Any: cualquier tipo; Literal: conjunto fijo de valores.
from typing import Any, Literal

# Biblioteca de grafos.
import networkx as nx

# Importaciones internas del paquete metacarp:
from .busqueda_indices import build_search_encoding, encode_solution  # codificación para lote
from .cargar_grafos import cargar_objeto_gexf                         # carga grafo GEXF
from .cargar_soluciones_iniciales import cargar_solucion_inicial      # carga solución inicial
from .evaluador_costo import (
    costo_lote_penalizado_ids,           # evaluación en lote eficiente
    costo_rapido,                        # evaluación individual rápida (NumPy)
    exceso_capacidad_rapido,             # calcula violación de capacidad
    lambda_penal_capacidad_por_defecto,  # λ automático para penalización
    objectivo_penalizado,                # objetivo = costo + λ × violación
)
from .instances import load_instances  # datos de la instancia CARP
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
    "CuckooSearchResult",
    "cuckoo_search",
    "cuckoo_search_desde_instancia",
]


# --- CONCEPTO OOP: @dataclass(frozen=True, slots=True) ---
# frozen=True: objeto inmutable; slots=True: menor consumo de memoria.
# Apropiado para un objeto de resultado que solo se consulta, nunca se modifica.
@dataclass(frozen=True, slots=True)
class CuckooSearchResult:
    """
    Resultado completo de Cuckoo Search con reemplazo por nidos y abandono parcial.

    Agrupa la mejor solución encontrada, métricas de calidad y estadísticas
    específicas de Cuckoo Search: número de nidos, abandonos y reemplazos.
    """

    # La mejor solución CARP encontrada.
    mejor_solucion: list[list[str]]
    # Costo de la mejor solución (objetivo minimizado).
    mejor_costo: float
    # Solución inicial de referencia para calcular mejora.
    solucion_inicial_referencia: list[list[str]]
    # Costo de la solución inicial.
    costo_solucion_inicial: float
    # Diferencia absoluta de costo: costo_inicial - mejor_costo.
    mejora_absoluta: float
    # Mejora expresada como porcentaje del costo inicial.
    mejora_porcentaje_inicial_vs_final: float
    # Tiempo total de ejecución en segundos.
    tiempo_segundos: float
    # Total de iteraciones ejecutadas.
    iteraciones_totales: int
    # Número de nidos (soluciones candidatas activas en paralelo).
    nidos: int
    # Total de veces que se ejecutó el abandono de peores nidos.
    abandonos_totales: int
    # Veces que un cuckoo reemplazó exitosamente un nido.
    reemplazos_exitosos: int
    # Veces que el mejor global mejoró.
    mejoras: int
    # Semilla del generador aleatorio.
    semilla: int | None
    # Dispositivo de evaluación: 'cpu' o 'gpu'.
    backend_evaluacion: str = "cpu"
    # Historial del mejor costo por iteración.
    historial_mejor_costo: list[float] = field(default_factory=list)
    # Último movimiento aceptado al final de la búsqueda.
    ultimo_movimiento_aceptado: MovimientoVecindario | None = None
    # Estadísticas de operadores de vecindario.
    operadores_propuestos: dict[str, int] = field(default_factory=dict)
    operadores_aceptados: dict[str, int] = field(default_factory=dict)
    operadores_mejoraron: dict[str, int] = field(default_factory=dict)
    operadores_trayectoria_mejor: dict[str, int] = field(default_factory=dict)
    # Configuración de penalización usada.
    usar_penalizacion_capacidad: bool = True
    lambda_capacidad: float = 0.0
    # Estadísticas de la selección inicial.
    n_iniciales_evaluados: int = 0
    iniciales_infactibles_aceptadas: int = 0
    aceptaciones_solucion_infactible: int = 0
    # True si la mejor solución final es factible.
    mejor_solucion_factible_final: bool = True
    # Ruta del CSV guardado, o None.
    archivo_csv: str | None = None


def _vuelo_levy_discreto(
    base: list[list[str]],
    *,
    rng: random.Random,
    pasos_base: int,
    beta: float,
    operadores: Iterable[str],
    pesos_operadores: list[float] | None = None,
    marcador_depot: str,
    usar_gpu: bool,
    backend_vecindario: Literal["labels", "ids"],
    encoding: Any,
) -> tuple[list[list[str]], list[MovimientoVecindario]]:
    """
    Aproximación discreta del vuelo de Lévy para espacio de soluciones CARP.

    En el Cuckoo Search original (espacio continuo), el vuelo de Lévy genera
    pasos cuya longitud sigue una distribución de cola pesada (heavy-tail):
    la mayoría de los pasos son cortos, pero ocasionalmente aparecen saltos
    muy largos que permiten explorar zonas alejadas del espacio.

    Aquí lo adaptamos a espacio discreto calculando el **número de
    perturbaciones consecutivas** según la fórmula:

        n_pasos = 1 + floor(|N(0,1)|^(1/beta) * pasos_base)

    donde N(0,1) es una muestra de la distribución normal estándar y
    beta controla la forma de la distribución (1.5 es el valor clásico).

    Un beta pequeño genera más saltos largos (mayor exploración).
    Un beta grande tiende a pasos cortos (mayor explotación local).

    Cada "paso" es una perturbación local: se aplica un operador de vecindario
    a la solución actual, generando una solución cercana.

    Retorna la solución resultante y la lista completa de movimientos aplicados.
    """
    # Si beta es inválido, usamos el valor clásico de Cuckoo Search.
    if beta <= 0:
        beta = 1.5
    # Muestreamos el valor absoluto de una normal estándar (|N(0,1)|).
    x = abs(rng.gauss(0.0, 1.0))
    # Calculamos el número de pasos: mínimo 1, máximo 12 (cota para evitar explosión).
    # x^(1/beta) transforma la distribución para imitar la cola pesada de Lévy.
    n_pasos = min(12, 1 + int((x ** (1.0 / beta)) * max(1, pasos_base)))

    # Comenzamos desde una copia de la solución base para no modificarla.
    sol = copiar_solucion_labels(base)
    movs_seq: list[MovimientoVecindario] = []
    # Aplicamos n_pasos perturbaciones consecutivas.
    for _ in range(n_pasos):
        sol, m = generar_vecino(
            sol, rng=rng, operadores=operadores,
            pesos_operadores=pesos_operadores,
            marcador_depot=marcador_depot, devolver_con_deposito=True,
            usar_gpu=usar_gpu, backend=backend_vecindario, encoding=encoding,
        )
        movs_seq.append(m)
    return sol, movs_seq


def _eval_penalizado_lote(
    vecinos: list[list[list[str]]],
    ctx: Any,
    lam: float,
    *,
    usar_penal: bool,
) -> tuple[list[float], list[float], list[float]]:
    """
    Evalúa un lote de soluciones y devuelve tres listas de Python (float).

    Retorna:
    - objs: objetivo penalizado de cada solución (costo + λ × violación).
    - puros: costo sin penalización de cada solución.
    - viols: exceso de demanda (violación de capacidad) de cada solución.

    Si la lista de vecinos está vacía, retorna tres listas vacías para
    evitar errores en la llamada a las funciones de evaluación.
    """
    if not vecinos:
        return [], [], []
    # Convertimos las soluciones a formato de ids para evaluación eficiente.
    sols_ids = [encode_solution(v, ctx.encoding) for v in vecinos]
    objs, puros, viols = costo_lote_penalizado_ids(
        sols_ids, ctx, lam, usar_penal=usar_penal
    )
    # .astype(float).tolist() convierte arrays NumPy a listas de Python float.
    return (
        objs.astype(float).tolist(),
        puros.astype(float).tolist(),
        viols.astype(float).tolist(),
    )


def cuckoo_search(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    iteraciones: int = 260,          # número de ciclos del algoritmo
    num_nidos: int = 20,             # número de nidos (soluciones activas en paralelo)
    pa_abandono: float = 0.25,       # fracción de peores nidos a abandonar por iteración (0 < pa < 1)
    pasos_levy_base: int = 3,        # escala base del vuelo Lévy discreto
    beta_levy: float = 1.5,          # parámetro de forma de la distribución Lévy (valor clásico: 1.5)
    semilla: int | None = None,      # semilla para reproducibilidad
    operadores: Iterable[str] = OPERADORES_POPULARES,  # operadores de vecindario habilitados
    marcador_depot_etiqueta: str | None = None,  # etiqueta del nodo depósito
    usar_gpu: bool = False,          # si True, evalúa en GPU cuando está disponible
    backend_vecindario: Literal["labels", "ids"] = "labels",  # modo de generación de vecinos
    guardar_historial: bool = True,  # si True, guarda historial de costo por iteración
    guardar_csv: bool = False,       # si True, escribe fila de resultados en CSV
    ruta_csv: str | None = None,     # ruta del CSV (None = nombre automático)
    nombre_instancia: str = "instancia",  # nombre de la instancia
    id_corrida: str | None = None,
    config_id: str | None = None,
    repeticion: int | None = None,
    root: str | None = None,
    usar_penalizacion_capacidad: bool = True,
    lambda_capacidad: float | None = None,
    extra_csv: dict[str, object] | None = None,
    alpha_intra: float = 0.8,  # fracción de prob. asignada a ops intra-ruta cuando hay violación
) -> CuckooSearchResult:
    """
    Cuckoo Search clásico adaptado a espacio discreto de rutas CARP.

    Estrategia por iteración:
    1. Cada nido genera un cuckoo mediante vuelo Lévy discreto.
    2. Cada cuckoo compite con un nido aleatorio; si es mejor, lo reemplaza.
    3. Los ``floor(pa_abandono × num_nidos)`` peores nidos son abandonados
       y reemplazados por nuevas soluciones generadas en torno al mejor nido.
    4. Se actualizan los mejores global y factible.
    """
    # Validaciones de parámetros.
    if iteraciones <= 0:
        raise ValueError("iteraciones debe ser > 0.")
    if num_nidos <= 1:
        raise ValueError("num_nidos debe ser >= 2.")
    if not (0.0 < pa_abandono < 1.0):
        raise ValueError("pa_abandono debe estar en (0, 1).")
    if pasos_levy_base <= 0:
        raise ValueError("pasos_levy_base debe ser > 0.")

    # Generador aleatorio controlado.
    rng = random.Random(semilla)
    # Marca de tiempo de inicio.
    t0 = time.perf_counter()

    # Contexto de evaluación rápida: precomputa matrices de distancias una sola vez.
    ctx = construir_contexto_para_corrida(
        data,
        G,
        nombre_instancia=nombre_instancia if nombre_instancia != "instancia" else None,
        usar_gpu=usar_gpu,
        root=root,
    )

    # Lambda efectiva para penalizar violaciones de capacidad.
    lam_eff = (
        float(lambda_capacidad)
        if lambda_capacidad is not None
        else lambda_penal_capacidad_por_defecto(ctx)
    )

    # Selección de la mejor solución inicial entre las candidatas disponibles.
    sel_ini = seleccionar_mejor_inicial_rapido(
        inicial_obj,
        ctx,
        usar_penalizacion_capacidad=usar_penalizacion_capacidad,
        lambda_capacidad=lambda_capacidad,
    )
    sol_ref = sel_ini.solucion
    costo_ref = sel_ini.costo_puro
    ini_infact = sel_ini.n_candidatos_infactibles
    n_ini_ev = sel_ini.n_candidatos_evaluados

    # Configuración del encoding para evaluación por ids.
    encoding = ctx.encoding
    if backend_vecindario == "ids" and encoding is None:
        encoding = build_search_encoding(data)

    # Etiqueta del nodo depósito.
    md_op = marcador_depot_etiqueta or ctx.marcador_depot

    # Rastreo del mejor global y del mejor factible.
    # Partimos de infinito para que cualquier solución real sea mejor.
    mejor_any_c = float("inf")
    mejor_any_s = copiar_solucion_labels(sol_ref)
    mejor_fact_c: float | None = None
    mejor_fact_s: list[list[str]] | None = None

    def costo_para_reporte() -> float:
        """Retorna el costo factible si existe; si no, el mejor general."""
        return float(mejor_fact_c) if mejor_fact_c is not None else mejor_any_c

    # Contador de estadísticas de operadores.
    contador = ContadorOperadores()
    # Lista de índices de nidos [0, 1, ..., num_nidos-1].
    rango_nidos = list(range(num_nidos))

    # --- Inicialización de nidos ---
    # El primer nido es la solución de referencia inicial.
    nidos_sol: list[list[list[str]]] = [copiar_solucion_labels(sol_ref)]
    nidos_pure: list[float] = [costo_ref]                    # costo puro de cada nido
    nidos_viol: list[float] = [float(sel_ini.violacion_capacidad)]  # violación de capacidad

    # Los nidos restantes se generan como vecinos aleatorios de la referencia.
    while len(nidos_sol) < num_nidos:
        cand, _m = generar_vecino(
            sol_ref,
            rng=rng,
            operadores=operadores,
            marcador_depot=md_op,
            devolver_con_deposito=True,
            usar_gpu=usar_gpu,
            backend=backend_vecindario,
            encoding=encoding,
        )
        cp = costo_rapido(cand, ctx)            # costo puro del candidato
        vp = exceso_capacidad_rapido(cand, ctx) # violación de capacidad del candidato
        nidos_sol.append(cand)
        nidos_pure.append(cp)
        nidos_viol.append(vp)

    def objeto_nido(k: int) -> float:
        """Calcula el objetivo penalizado del nido k: costo + λ × violación."""
        return float(
            objectivo_penalizado(
                nidos_pure[k],
                nidos_viol[k],
                usar_penal=usar_penalizacion_capacidad,
                lam=lam_eff,
            )
        )

    def fusionar_desde_nidos() -> None:
        """
        Actualiza los mejores globales revisando todos los nidos.

        Se llama al final de cada iteración y al cierre de la búsqueda.
        'nonlocal' declara que las variables referenciadas pertenecen al
        ámbito de cuckoo_search, no son variables locales nuevas.
        """
        nonlocal mejor_any_c, mejor_any_s, mejor_fact_c, mejor_fact_s
        nonlocal sol_mejor, costo_mejor
        for k in rango_nidos:
            cp = nidos_pure[k]
            vv = nidos_viol[k]
            sol = nidos_sol[k]
            # Actualizamos el mejor sin restricción.
            if cp < mejor_any_c - 1e-15:
                mejor_any_c = cp
                mejor_any_s = copiar_solucion_labels(sol)
            # Actualizamos el mejor factible solo si el nido no viola capacidad.
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

    # Primera fusión: inicializamos sol_mejor y costo_mejor con los nidos iniciales.
    fusionar_desde_nidos()

    # Inicializamos sol_mejor y costo_mejor para el bucle.
    sol_mejor = copiar_solucion_labels(
        mejor_fact_s if mejor_fact_s is not None else mejor_any_s
    )
    costo_mejor = costo_para_reporte()
    mejoras = 0        # contador de mejoras del mejor global
    reemplazos = 0     # veces que un cuckoo reemplazó un nido
    abandonos = 0      # veces que un nido fue abandonado (fase de abandono)
    historial_best: list[float] = []
    ultimo_mov_aceptado: MovimientoVecindario | None = None
    aceptaciones_sol_infactible = 0

    # === BUCLE PRINCIPAL DE CUCKOO SEARCH ===
    for _it in range(iteraciones):
        rep_antes_it = costo_para_reporte()
        if guardar_historial:
            historial_best.append(rep_antes_it)

        # =====================================================================
        # PASO 1: GENERACIÓN DE CUCKOOS MEDIANTE VUELO DE LÉVY
        # Cada nido genera un "cuckoo" (solución candidata) aplicando el vuelo
        # de Lévy discreto: múltiples perturbaciones con longitud variable.
        # =====================================================================
        cuckoos: list[list[list[str]]] = []                      # soluciones cuckoo generadas
        movs_levy: list[list[MovimientoVecindario]] = []          # movimientos aplicados por cuckoo
        for i in range(num_nidos):
            pesos_i = pesos_intra_bias(nidos_viol[i], list(operadores), alpha_intra=alpha_intra)
            cs, movs_seq = _vuelo_levy_discreto(
                nidos_sol[i],                    # partimos del nido i
                rng=rng,
                pasos_base=pasos_levy_base,      # escala del vuelo Lévy
                beta=beta_levy,                  # parámetro de forma de la distribución
                operadores=operadores,
                pesos_operadores=pesos_i,
                marcador_depot=md_op,
                usar_gpu=usar_gpu,
                backend_vecindario=backend_vecindario,
                encoding=encoding,
            )
            cuckoos.append(cs)
            movs_levy.append(movs_seq)
            # Registramos cada movimiento del vuelo Lévy como "propuesto".
            for m in movs_seq:
                contador.proponer(m.operador)

        # Evaluamos todos los cuckoos en lote (eficiente en CPU/GPU).
        objs_cu, pure_cu, viol_cu = _eval_penalizado_lote(
            cuckoos,
            ctx,
            lam_eff,
            usar_penal=usar_penalizacion_capacidad,
        )

        # =====================================================================
        # PASO 2: COMPETENCIA CUCKOO vs NIDO ALEATORIO
        # Cada cuckoo compite con un nido elegido al azar. Si el cuckoo es
        # mejor (objetivo menor), reemplaza al nido.
        #
        # Analogía: el cucú pone su huevo en el nido de otra ave. Si el ave
        # huésped no detecta el intruso (nuestro "criterio de mejora"), el
        # huevo permanece y el cucú "gana" ese nido.
        # =====================================================================
        for i in range(num_nidos):
            # Elegimos un nido al azar (puede ser el mismo nido i).
            j = rng.randrange(num_nidos)
            o_c = float(objs_cu[i])      # objetivo del cuckoo i
            o_dest = objeto_nido(j)      # objetivo del nido j (destino)
            if o_c < o_dest - 1e-15:
                # El cuckoo es mejor: reemplaza al nido j.
                nidos_sol[j] = cuckoos[i]
                nidos_pure[j] = pure_cu[i]
                nidos_viol[j] = viol_cu[i]
                reemplazos += 1
                if viol_cu[i] > 1e-12:
                    aceptaciones_sol_infactible += 1
                # Registramos el último movimiento del vuelo como aceptado.
                if movs_levy[i]:
                    ultimo_mov_aceptado = movs_levy[i][-1]
                    contador.aceptar(ultimo_mov_aceptado.operador)

        # =====================================================================
        # PASO 3: ABANDONO DE PEORES NIDOS
        # Una fracción 'pa_abandono' de los peores nidos es descartada y
        # reemplazada por nuevas soluciones en torno al mejor nido.
        #
        # Analogía: algunas aves huéspedes detectan el huevo del cucú y
        # abandonan el nido, construyendo uno nuevo en otro lugar.
        # Esto introduce diversidad: las peores soluciones se rejuvenecen.
        # =====================================================================
        # Número de nidos a abandonar en esta iteración (mínimo 1).
        n_abandonar = max(1, int(math.floor(pa_abandono * num_nidos)))
        # Ordenamos los nidos de peor a mejor y tomamos los primeros n_abandonar.
        peores = sorted(rango_nidos, key=objeto_nido, reverse=True)[:n_abandonar]
        # El mejor nido (menor objetivo penalizado) sirve de base para los nuevos.
        idx_best = min(rango_nidos, key=objeto_nido)
        base_best = nidos_sol[idx_best]

        # Generamos nuevas soluciones para reemplazar los peores nidos.
        nuevos: list[list[list[str]]] = []
        movs_abandono: list[list[MovimientoVecindario]] = []
        pesos_best = pesos_intra_bias(nidos_viol[idx_best], list(operadores), alpha_intra=alpha_intra)
        for _idx in peores:
            # Cada nuevo nido se genera con un vuelo Lévy desde el mejor nido actual.
            ns, ms = _vuelo_levy_discreto(
                base_best,
                rng=rng,
                pasos_base=pasos_levy_base,
                beta=beta_levy,
                operadores=operadores,
                pesos_operadores=pesos_best,
                marcador_depot=md_op,
                usar_gpu=usar_gpu,
                backend_vecindario=backend_vecindario,
                encoding=encoding,
            )
            nuevos.append(ns)
            movs_abandono.append(ms)
            for m in ms:
                contador.proponer(m.operador)
        # Evaluamos los nuevos nidos en lote.
        _objs_nv, pure_nv, viol_nv = _eval_penalizado_lote(
            nuevos,
            ctx,
            lam_eff,
            usar_penal=usar_penalizacion_capacidad,
        )
        # Reemplazamos cada nido abandonado por su nuevo vecino.
        for k, idx in enumerate(peores):
            nidos_sol[idx] = nuevos[k]
            nidos_pure[idx] = pure_nv[k]
            nidos_viol[idx] = viol_nv[k]
            abandonos += 1
            if viol_nv[k] > 1e-12:
                aceptaciones_sol_infactible += 1
            if movs_abandono[k]:
                # Registramos el último movimiento del vuelo de abandono.
                contador.aceptar(movs_abandono[k][-1].operador)

        # Actualizamos los mejores globales con todos los nidos del ciclo.
        fusionar_desde_nidos()
        # Si esta iteración mejoró el costo reportable, lo registramos.
        if costo_para_reporte() + 1e-12 < rep_antes_it:
            mejoras += 1
            op_mejor = ultimo_mov_aceptado.operador if ultimo_mov_aceptado else None
            contador.registrar_mejora(op_mejor)

    # === FIN DEL BUCLE PRINCIPAL ===

    # Tiempo total de ejecución.
    elapsed = time.perf_counter() - t0
    # Fusión final para capturar mejoras de la última iteración.
    fusionar_desde_nidos()
    costo_mejor = costo_para_reporte()
    sol_mejor = copiar_solucion_labels(
        mejor_fact_s if mejor_fact_s is not None else mejor_any_s
    )
    mejor_factible_final = mejor_fact_s is not None
    _gap_descartado, mejora_abs, mejora_pct = calcular_metricas_gap(costo_ref, costo_mejor)

    # --- Guardado en CSV (opcional) ---
    archivo_csv: str | None = None
    if guardar_csv:
        ruta = ruta_csv or f"resultados_cuckoo_search_{nombre_instancia}.csv"
        detalle_txt, costo_total_reporte = generar_reporte_detallado(
            sol_mejor, data, G,
            nombre_instancia=nombre_instancia,
            marcador_depot_etiqueta=marcador_depot_etiqueta,
            usar_gpu=False,
        )
        _bks = resumen_bks_csv(data, costo_mejor)
        fila = {
            "metaheuristica": "cuckoo_search",
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
            "num_nidos": num_nidos,
            "pa_abandono": pa_abandono,
            "pasos_levy_base": pasos_levy_base,
            "beta_levy": beta_levy,
            **contador.resumen_csv(),
            "aceptadas": sum(contador.aceptados.values()),
            "mejoras": mejoras,
            "mejor_solucion_factible_final": mejor_factible_final,
            "mejor_solucion_tr_legible": solucion_legible_humana(sol_mejor),
            "reporte_detalle_deadheading": detalle_txt,
            "costo_total_desde_reporte": costo_total_reporte,
        }
        archivo_csv = guardar_resultado_csv(fila=fila, ruta_csv=ruta)

    # Retornamos el objeto de resultado inmutable.
    return CuckooSearchResult(
        mejor_solucion=sol_mejor,
        mejor_costo=costo_mejor,
        solucion_inicial_referencia=sol_ref,
        costo_solucion_inicial=costo_ref,
        mejora_absoluta=mejora_abs,
        mejora_porcentaje_inicial_vs_final=mejora_pct,
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
        usar_penalizacion_capacidad=usar_penalizacion_capacidad,
        lambda_capacidad=lam_eff,
        n_iniciales_evaluados=n_ini_ev,
        iniciales_infactibles_aceptadas=ini_infact,
        aceptaciones_solucion_infactible=aceptaciones_sol_infactible,
        mejor_solucion_factible_final=mejor_factible_final,
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
    usar_penalizacion_capacidad: bool = True,
    lambda_capacidad: float | None = None,
    extra_csv: dict[str, object] | None = None,
    alpha_intra: float = 0.8,
) -> CuckooSearchResult:
    """
    Función de conveniencia: carga todos los recursos desde el nombre de la
    instancia y ejecuta Cuckoo Search completo.

    Equivalente a llamar manualmente a load_instances + cargar_objeto_gexf
    + cargar_solucion_inicial + cuckoo_search.
    """
    # Cargamos los datos de la instancia (parámetros CARP, BKS, etc.).
    data = load_instances(nombre_instancia, root=root)
    # Cargamos el grafo de la instancia desde el archivo GEXF.
    G = cargar_objeto_gexf(nombre_instancia, root=root)
    # Cargamos la solución inicial desde el archivo pickle.
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
        usar_penalizacion_capacidad=usar_penalizacion_capacidad,
        lambda_capacidad=lambda_capacidad,
        extra_csv=extra_csv,
        alpha_intra=alpha_intra,
    )
