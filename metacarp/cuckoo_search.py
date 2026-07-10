"""
Cuckoo Search adaptado a espacio discreto para CARP (versión modernizada e
instance-aware, alineada con SA / TS simple / RTS / ABC simple).

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

Modernización (mayo 2026) — alineación con SA / TS simple / RTS / ABC simple
----------------------------------------------------------------------------
Esta versión incorpora los mismos cambios estructurales que ya se aplicaron a
``busqueda_abejas_simple``:

* **Parámetros instance-aware**: ``iteraciones``, ``num_nidos``, ``pasos_levy_base``
  y ``max_iter_sin_mejora`` aceptan ``None`` y se calibran automáticamente en
  función de ``n_tareas`` (el número de arcos requeridos de la instancia). El
  usuario también puede pasar **factores de escala** (``factor_iter``,
  ``factor_nidos``, ``factor_pasos``) y el algoritmo los traduce a valores
  concretos. La precedencia es siempre: **valor absoluto > factor > default**.

* **Sesgo inter/intra-ruta uniforme**: se reemplaza el helper local
  ``pesos_inter_bias`` por el helper compartido
  ``seleccionar_grupo_operadores_inter_intra`` (mismo que SA / TS simple /
  RTS / ABC simple). El parámetro ``alpha_inter`` se elimina de la firma: el
  algoritmo eleva automáticamente ``P(inter)`` a ``max(p_inter, 0.8)`` cuando
  hay violación de capacidad. El usuario solo configura ``p_inter`` (la
  probabilidad inter en régimen factible); el piso 0.8 bajo violación es una
  garantía interna del algoritmo.

* **Bugs corregidos**:
  - ``extra_csv`` ahora se vuelca a la fila CSV (antes se ignoraba).
  - ``registrar_mejora`` se invoca dentro del MISMO ``if`` que detecta el
    cuckoo mejor que el global, con el operador del vuelo de Lévy responsable
    (no con un movimiento posterior). Esto garantiza que
    ``sum(operadores_mejoraron.values())`` coincide con ``mejoras`` (módulo los
    casos en que la mejora vino sin operador asociado).
  - ``historial_mejor_costo`` se registra al FINAL de cada iteración (tras
    competencia y abandono), no al inicio, para reflejar la mejor solución
    real del ciclo.

* **CSV unificado**: se añaden las columnas ``n_tareas``, ``*_efectivo``,
  ``factor_*``, ``mejora_absoluta``, ``mejora_porcentaje``,
  ``usar_penalizacion_capacidad``, ``lambda_capacidad``,
  ``iteraciones_sin_mejora_final``, ``iteraciones_con_violacion``,
  ``fraccion_iter_con_violacion``, ``p_inter`` y ``p_inter_max_efectivo``. Se
  elimina la columna agregada ``aceptadas`` (ya está desglosada por operador
  vía ``contador.resumen_csv()``).
"""
# Permite usar `float | None` en Python < 3.10.
from __future__ import annotations

# math.floor y math.sqrt se usan para el cálculo de vuelo Lévy discreto y para
# las fórmulas instance-aware.
import math
# Generador de números aleatorios controlado.
import random
# Medición de tiempo de alta precisión.
import time
# Tipos abstractos para firmas de funciones.
from collections.abc import Iterable, Mapping
# Soporte de dataclasses con atributos de valor por defecto complejos.
from dataclasses import dataclass, field
# Any: cualquier tipo; Callable: tipo para funciones/callables; Literal: conjunto fijo de valores.
from typing import Any, Callable, Literal

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
    resumen_bks_csv,
    seleccionar_grupo_operadores_inter_intra,  # helper compartido con SA / TS / RTS / ABC simple
    seleccionar_mejor_inicial_rapido,
    solucion_legible_humana,
)
from .vecindarios import (
    MovimientoVecindario,
    OPERADORES_INTER,                    # subset oficial inter-ruta (canon del proyecto)
    OPERADORES_INTRA,                    # subset oficial intra-ruta (canon del proyecto)
    OPERADORES_POPULARES,
    generar_vecino,
)

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
    # ---- Campos instance-aware (calibración automática por n_tareas) ----
    # Número de tareas requeridas de la instancia (= len(ctx.u_arr)). Es la
    # variable "n" usada en las fórmulas de escala.
    n_tareas: int = 0
    # Valores realmente usados por el bucle. Pueden venir del valor absoluto
    # que pasó el usuario, del factor de escala, o de la fórmula default.
    # Reportarlos en el resultado y en el CSV permite reproducir exactamente
    # la corrida posteriormente.
    iteraciones_efectivas: int = 0
    num_nidos_efectivo: int = 0
    pasos_levy_efectivo: int = 0
    max_iter_sin_mejora_efectivo: int | None = None
    # P(inter) máxima aplicada cuando hay violación (= max(p_inter, 0.8)). Si
    # la corrida no tuvo violación, este valor documenta lo que el algoritmo
    # HABRÍA usado si la hubiera tenido.
    p_inter_max_efectivo: float = 0.8
    # Factores de escala efectivamente usados (None si no se pasaron). Los
    # exponemos en el resultado para que el CSV y el análisis posterior
    # puedan distinguir corridas configuradas por factor vs por absoluto.
    factor_iter: float | None = None
    factor_nidos: float | None = None
    factor_pasos: float | None = None
    # Diagnóstico de estancamiento y de violación.
    iteraciones_sin_mejora_final: int = 0
    iteraciones_con_violacion: int = 0
    fraccion_iter_con_violacion: float = 0.0
    # Numero de kicks (perturbaciones inter-ruta) aplicados durante la corrida.
    # Solo es > 0 cuando se pasa max_iter_sin_mejora_kick al wrapper de la
    # variante experimental strict_intra_inter_20260524.
    n_resets_kick: int = 0


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

    Nota: a diferencia de la versión anterior, este helper NO recibe
    ``pesos_operadores``. El sesgo inter/intra se aplica AGUAS ARRIBA mediante
    ``seleccionar_grupo_operadores_inter_intra``, que devuelve el subconjunto de
    operadores ya filtrado. Dentro del vuelo, ``generar_vecino`` elige
    uniformemente dentro de ese grupo. Esta separación de responsabilidades es
    la misma que usan SA / TS simple / RTS / ABC simple.
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
            pesos_operadores=None,            # selección uniforme dentro del grupo ya filtrado
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
    # ---- Parámetros instance-aware (None ⇒ se calculan a partir de n_tareas) ----
    # Reglas de precedencia (idénticas a busqueda_abejas_simple):
    #   1. Si el usuario pasa el valor ABSOLUTO, manda.
    #   2. Si pasa el FACTOR de escala, se usa la fórmula del factor.
    #   3. Si no pasa nada, se usa la fórmula DEFAULT en función de n_tareas.
    iteraciones: int | None = None,          # None → max(200, 20·n)
    num_nidos: int | None = None,            # None → max(10, round(2·√n))
    pasos_levy_base: int | None = None,      # None → max(3, round(√n / 2))
    max_iter_sin_mejora: int | None = None,  # None → max(50, 3·n)
    # pa_abandono NO es instance-aware: es una RATIO (fracción de peores nidos
    # a abandonar por iteración). Su valor sensato vive en (0, 1) y no escala
    # con n. Lo mantenemos como float fijo con el default clásico de la literatura.
    pa_abandono: float = 0.25,
    beta_levy: float = 1.5,                  # parámetro de forma de la distribución Lévy
    # ---- Factores de escala alternativos (sobrescriben la fórmula default) ----
    # Permiten que el script de experimentos barra POR FACTOR sin tener que
    # calcular él mismo el valor absoluto (que depende de cada instancia).
    # Precedencia: valor absoluto > factor > default calculado por fórmula.
    factor_iter: float | None = None,        # iteraciones      = max(200, round(f · n))
    factor_nidos: float | None = None,       # num_nidos        = max(10,  round(f · √n))
    factor_pasos: float | None = None,       # pasos_levy_base  = max(3,   round(f · √n))
    semilla: int | None = None,              # semilla para reproducibilidad
    operadores: Iterable[str] = OPERADORES_POPULARES,  # operadores de vecindario habilitados
    marcador_depot_etiqueta: str | None = None,  # etiqueta del nodo depósito
    usar_gpu: bool = False,                  # si True, evalúa en GPU cuando está disponible
    backend_vecindario: Literal["labels", "ids"] = "labels",  # modo de generación de vecinos
    guardar_historial: bool = True,          # si True, guarda historial de costo por iteración
    guardar_csv: bool = False,               # si True, escribe fila de resultados en CSV
    ruta_csv: str | None = None,             # ruta del CSV (None = nombre automático)
    nombre_instancia: str = "instancia",     # nombre de la instancia
    repeticion: int | None = None,
    root: str | None = None,
    usar_penalizacion_capacidad: bool = True,
    lambda_capacidad: float | None = None,
    # p_inter BASE: P(elegir el grupo INTER-RUTA) cuando la solución actual es
    # FACTIBLE. Cuando hay violación de capacidad, el algoritmo eleva
    # automáticamente la P(inter) a max(p_inter, 0.8) — el parámetro
    # ``alpha_inter`` fue eliminado para que el usuario configure UN solo punto
    # de la curva y el piso 0.8 bajo violación esté garantizado por construcción
    # (misma decisión que en busqueda_abejas_simple).
    p_inter: float = 0.6,
    metodo_seleccion: str = "canonico",  # método de combinación inter/intra: canonico|p_inter|binario|random
    # --- Kick reactivo (variante experimental strict_intra_inter_20260524) ---
    # Cuando ``iter_sin_mejora`` alcanza este umbral se aplica una perturbacion
    # INTER-RUTA disruptiva a TODOS los nidos y se reinicia el contador.
    # None = mecanismo desactivado (comportamiento clasico).
    max_iter_sin_mejora_kick: int | None = None,
    # Cota dura del numero de kicks consecutivos. Cuando se alcanza, la corrida
    # termina. None = sin tope (los kicks se permiten indefinidamente).
    max_resets: int | None = None,
    # Hook de intensificacion OPCIONAL (p.ej. Path Relinking limpio). Si se
    # provee, en el punto de estancamiento (mismo disparador que el kick) se
    # ejecuta PR HACIA la mejor solucion global EN LUGAR del kick aleatorio.
    # None = comportamiento clasico (kick aleatorio). API limpia que reemplaza
    # los frame-hacks del PR del primer ciclo.
    intensificador: Callable | None = None,
    extra_csv: dict[str, object] | None = None,
    **_ignorado_kwargs: object,  # absorbe kwargs heredados (p.ej. id_corrida, config_id)
) -> CuckooSearchResult:
    """
    Cuckoo Search clásico adaptado a espacio discreto de rutas CARP.

    Estrategia por iteración:
    1. Cada nido genera un cuckoo mediante vuelo Lévy discreto.
    2. Cada cuckoo compite con un nido aleatorio; si es mejor, lo reemplaza.
    3. Los ``floor(pa_abandono × num_nidos)`` peores nidos son abandonados
       y reemplazados por nuevas soluciones generadas en torno al mejor nido.
    4. Se actualizan los mejores global y factible.

    Parámetros clave (instance-aware)
    --------------------------------
    iteraciones : int | None
        Número de ciclos del algoritmo. ``None`` ⇒ ``max(200, 20·n_tareas)``.
    num_nidos : int | None
        Soluciones candidatas en paralelo. ``None`` ⇒ ``max(10, round(2·√n))``.
    pasos_levy_base : int | None
        Escala base del vuelo Lévy discreto. ``None`` ⇒ ``max(3, round(√n/2))``.
    max_iter_sin_mejora : int | None
        Criterio de parada por estancamiento. ``None`` ⇒ ``max(50, 3·n)``.
    factor_iter, factor_nidos, factor_pasos : float | None
        Factores de escala alternativos: se usan solo si el respectivo
        parámetro absoluto vale ``None``. Pensados para el grid del script de
        experimentos. Permiten barrer un mismo factor sobre instancias de
        distinto tamaño manteniendo la proporcionalidad de recursos.
    p_inter : float (0..1)
        P(elegir grupo INTER-RUTA) cuando la solución es FACTIBLE. Bajo
        violación de capacidad se aplica el piso ``max(p_inter, 0.8)``.
    """
    # ----------------------------------------------------------
    # 1) Validaciones tempranas: solo validamos los valores que el usuario pasó
    #    explícitamente. Los parámetros instance-aware (None) quedan validados
    #    implícitamente tras el cálculo (sección 3.bis).
    # ----------------------------------------------------------
    if iteraciones is not None and iteraciones <= 0:
        raise ValueError("iteraciones debe ser > 0.")
    if num_nidos is not None and num_nidos <= 1:
        raise ValueError("num_nidos debe ser >= 2.")
    if not (0.0 < pa_abandono < 1.0):
        raise ValueError("pa_abandono debe estar en (0, 1).")
    if pasos_levy_base is not None and pasos_levy_base <= 0:
        raise ValueError("pasos_levy_base debe ser > 0.")
    if max_iter_sin_mejora is not None and max_iter_sin_mejora <= 0:
        raise ValueError("max_iter_sin_mejora debe ser > 0 si se especifica.")
    if not (0.0 <= p_inter <= 1.0):
        raise ValueError("p_inter debe estar en [0, 1].")
    # Validación de los factores de escala: solo si el usuario los pasó.
    if factor_iter is not None and factor_iter <= 0:
        raise ValueError("factor_iter debe ser > 0 si se especifica.")
    if factor_nidos is not None and factor_nidos <= 0:
        raise ValueError("factor_nidos debe ser > 0 si se especifica.")
    if factor_pasos is not None and factor_pasos <= 0:
        raise ValueError("factor_pasos debe ser > 0 si se especifica.")

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

    # ----------------------------------------------------------
    # 3.bis) Calibración instance-aware de los parámetros del algoritmo.
    #
    # ``n_tareas`` es el número de arcos REQUERIDOS de la instancia. Lo
    # tomamos del array ``u_arr`` del contexto (longitud = número de tareas),
    # con la misma convención que usan SA y ABC simple.
    #
    # Reglas de precedencia para cada parámetro escalable:
    #   1. Valor ABSOLUTO pasado por el usuario → manda.
    #   2. FACTOR de escala pasado por el usuario → se aplica la fórmula del factor.
    #   3. Ningún valor pasado → fórmula DEFAULT en función de n_tareas.
    #
    # Fórmulas default (alineadas con ABC simple):
    #   iteraciones      → max(200, 20 · n)
    #   num_nidos        → max(10, round(2 · √n))
    #   pasos_levy_base  → max(3, round(√n / 2))
    #   max_sin_mejora   → max(50, 3 · n)
    # Los pisos (200, 10, 3, 50) garantizan que instancias muy pequeñas no
    # ejecuten corridas degeneradas con tamaños insignificantes.
    # ----------------------------------------------------------
    n_tareas = int(len(ctx.u_arr))
    sqrt_n = math.sqrt(n_tareas) if n_tareas > 0 else 1.0

    # iteraciones efectivas: absoluto > factor > default.
    if iteraciones is not None:
        iteraciones_eff = int(iteraciones)
    elif factor_iter is not None:
        iteraciones_eff = max(200, int(round(float(factor_iter) * n_tareas)))
    else:
        iteraciones_eff = max(200, 20 * n_tareas)

    # num_nidos efectivo: absoluto > factor (escala con √n) > default.
    if num_nidos is not None:
        num_nidos_eff = int(num_nidos)
    elif factor_nidos is not None:
        num_nidos_eff = max(10, int(round(float(factor_nidos) * sqrt_n)))
    else:
        num_nidos_eff = max(10, int(round(2.0 * sqrt_n)))

    # pasos_levy_base efectivo: absoluto > factor (escala con √n) > default (√n/2).
    if pasos_levy_base is not None:
        pasos_levy_eff = int(pasos_levy_base)
    elif factor_pasos is not None:
        pasos_levy_eff = max(3, int(round(float(factor_pasos) * sqrt_n)))
    else:
        pasos_levy_eff = max(3, int(round(sqrt_n / 2.0)))

    # max_iter_sin_mejora efectivo: si el usuario lo dejó en None, lo ACTIVAMOS
    # automáticamente con un valor instance-aware (al contrario que antes, donde
    # None significaba "desactivado"). Justificación: queremos que toda corrida
    # tenga criterio de parada anticipada para evitar gastar tiempo en plateaus
    # largos. Mantenemos el check de None en el bucle para que un futuro flag
    # ``desactivar_estancamiento`` siga siendo trivial de añadir.
    if max_iter_sin_mejora is not None:
        max_sin_mejora_eff: int | None = int(max_iter_sin_mejora)
    else:
        max_sin_mejora_eff = max(50, 3 * n_tareas)

    # P(inter) máxima efectiva bajo violación: piso garantizado de 0.8.
    # Documenta lo que el algoritmo USARÁ en presencia de violación,
    # independientemente del valor base que pase el usuario.
    p_inter_max_efectivo = max(float(p_inter), 0.8)

    # Guardamos los VALORES BASE que pasó el usuario (cadena vacía en CSV si
    # son None). Estos valores muestran qué configuró el usuario, no qué usó
    # realmente el algoritmo.
    iteraciones_base_usuario = iteraciones
    num_nidos_base_usuario = num_nidos
    pasos_levy_base_usuario = pasos_levy_base
    max_iter_sin_mejora_base_usuario = max_iter_sin_mejora

    # ----------------------------------------------------------
    # 4) Precomputación UNA VEZ de las particiones intra/inter/fallback.
    #    Se pasan al helper compartido en cada paso del bucle sin reconstruirlas
    #    (microoptimización idéntica a SA / RTS / ABC simple).
    # ----------------------------------------------------------
    ops_list = list(operadores)
    ops_intra_list = [op for op in ops_list if op in OPERADORES_INTRA]
    ops_inter_list = [op for op in ops_list if op in OPERADORES_INTER]
    ops_fallback_list = list(ops_list)

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
    # Lista de índices de nidos [0, 1, ..., num_nidos_eff-1].
    rango_nidos = list(range(num_nidos_eff))

    # --- Inicialización de nidos ---
    # El primer nido es la solución de referencia inicial.
    nidos_sol: list[list[list[str]]] = [copiar_solucion_labels(sol_ref)]
    nidos_pure: list[float] = [costo_ref]                    # costo puro de cada nido
    nidos_viol: list[float] = [float(sel_ini.violacion_capacidad)]  # violación de capacidad

    # Los nidos restantes se generan como vecinos aleatorios de la referencia.
    while len(nidos_sol) < num_nidos_eff:
        cand, _m = generar_vecino(
            sol_ref,
            rng=rng,
            operadores=ops_list,
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
    # Contadores de diagnóstico instance-aware:
    iter_sin_mejora = 0
    # Numero de kicks (perturbaciones inter-ruta) aplicados durante la corrida.
    # Solo se incrementa si ``max_iter_sin_mejora_kick`` esta activo.
    n_resets_kick: int = 0
    iteraciones_con_violacion = 0
    # Número de ciclos efectivamente ejecutados (puede ser < iteraciones_eff si paramos antes).
    ciclo_final = 0

    # === BUCLE PRINCIPAL DE CUCKOO SEARCH ===
    for _it in range(iteraciones_eff):
        ciclo_final = _it + 1

        # Criterio de parada anticipada por estancamiento.
        # ``max_sin_mejora_eff`` es el valor instance-aware: nunca es None en
        # la nueva versión. Mantenemos el check de None por seguridad defensiva.
        if max_sin_mejora_eff is not None and iter_sin_mejora >= max_sin_mejora_eff:
            # Restamos 1 porque ciclo_final ya se incrementó al entrar pero
            # esta iteración no llegó a ejecutarse. Esto mantiene
            # ``iteraciones_totales`` igual al número REAL de ciclos completos.
            ciclo_final -= 1
            break

        # Snapshot del mejor al inicio del ciclo (para detectar mejora del ciclo
        # y actualizar el contador de estancamiento).
        rep_antes_it = costo_para_reporte()

        # Violación media de los nidos ANTES de la generación de cuckoos (driver
        # del sesgo inter/intra). Cuando hay violación media > 0, contamos esta
        # iteración para el diagnóstico ``fraccion_iter_con_violacion``.
        viol_media = sum(nidos_viol) / num_nidos_eff
        if viol_media > 0:
            iteraciones_con_violacion += 1

        # p_inter dinámico: cuando hay violación de capacidad, subimos a >=0.8
        # automáticamente para favorecer operadores inter-ruta que reequilibran
        # cargas. El usuario elige p_inter base; el algoritmo garantiza el piso
        # 0.8 bajo violación. Cuando NO hay violación, se respeta p_inter tal
        # cual lo pasó el usuario.
        p_efectivo = p_inter_max_efectivo if viol_media > 0 else float(p_inter)

        # =====================================================================
        # PASO 1: GENERACIÓN DE CUCKOOS MEDIANTE VUELO DE LÉVY
        # Cada nido genera un "cuckoo" (solución candidata) aplicando el vuelo
        # de Lévy discreto: múltiples perturbaciones con longitud variable.
        # =====================================================================
        cuckoos: list[list[list[str]]] = []                       # soluciones cuckoo generadas
        movs_levy: list[list[MovimientoVecindario]] = []           # movimientos aplicados por cuckoo
        grupos_por_nido: list[list[str]] = []                      # grupo de operadores elegido por cuckoo i
        for i in range(num_nidos_eff):
            # Para cada nido decidimos el grupo de operadores (inter o intra)
            # con el mismo helper compartido. Pasamos ``p_efectivo`` tanto a
            # ``alpha_inter`` (prob. bajo violación) como a ``p_inter`` (prob.
            # sin violación): como la violación de ESTE nido determina el
            # régimen, ambos parámetros apuntan al mismo valor para mantener
            # la decisión coherente con el comportamiento de ABC simple.
            viol_i = nidos_viol[i]
            grupo_i, _ = seleccionar_grupo_operadores_inter_intra(
                rng,
                viol_i,
                ops_intra_list,
                ops_inter_list,
                ops_fallback_list,
                alpha_inter=p_efectivo,
                p_inter=p_efectivo,
                metodo=metodo_seleccion,
            )
            grupos_por_nido.append(grupo_i)
            cs, movs_seq = _vuelo_levy_discreto(
                nidos_sol[i],                    # partimos del nido i
                rng=rng,
                pasos_base=pasos_levy_eff,       # escala del vuelo Lévy (instance-aware)
                beta=beta_levy,                  # parámetro de forma de la distribución
                operadores=grupo_i,              # subset ya filtrado (inter o intra)
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
        for i in range(num_nidos_eff):
            # Elegimos un nido al azar (puede ser el mismo nido i).
            j = rng.randrange(num_nidos_eff)
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
                # Esto representa el "operador responsable" del cuckoo aceptado.
                if movs_levy[i]:
                    ultimo_mov_aceptado = movs_levy[i][-1]
                    contador.aceptar(ultimo_mov_aceptado.operador)
                # CORRECCIÓN DEL BUG: detectamos la mejora del mejor global EN
                # EL MISMO IF que acepta el cuckoo. Imputamos la mejora al
                # operador del último movimiento del vuelo Lévy de ESTE cuckoo
                # (no a un movimiento posterior). Esta es la misma corrección
                # que se aplicó a busqueda_abejas_simple. Como consecuencia,
                # sum(operadores_mejoraron.values()) coincide con ``mejoras``
                # (módulo casos sin operador asociado).
                if pure_cu[i] < mejor_any_c - 1e-15 or (
                    viol_cu[i] < 1e-12
                    and (mejor_fact_c is None or pure_cu[i] < mejor_fact_c - 1e-15)
                ):
                    # Determinamos si esta aceptación es la que efectivamente
                    # mueve el costo reportable hacia abajo. Si lo es, lo
                    # contamos como mejora del mejor global.
                    if pure_cu[i] < mejor_any_c - 1e-15:
                        mejor_any_c = float(pure_cu[i])
                        mejor_any_s = copiar_solucion_labels(cuckoos[i])
                    if viol_cu[i] < 1e-12:
                        lim = mejor_fact_c if mejor_fact_c is not None else float("inf")
                        if pure_cu[i] < lim - 1e-15:
                            mejor_fact_c = float(pure_cu[i])
                            mejor_fact_s = copiar_solucion_labels(cuckoos[i])
                    # Recalculamos el costo reportable y, si bajó respecto al
                    # snapshot del inicio del ciclo, registramos la mejora con
                    # el operador del cuckoo aceptado.
                    nuevo_rep = costo_para_reporte()
                    if nuevo_rep + 1e-12 < rep_antes_it:
                        op_mejor = (
                            movs_levy[i][-1].operador if movs_levy[i] else None
                        )
                        contador.registrar_mejora(op_mejor)
                        mejoras += 1
                        # Actualizamos rep_antes_it para que mejoras sucesivas
                        # dentro del mismo ciclo se cuenten por separado.
                        rep_antes_it = nuevo_rep

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
        n_abandonar = max(1, int(math.floor(pa_abandono * num_nidos_eff)))
        # Ordenamos los nidos de peor a mejor y tomamos los primeros n_abandonar.
        peores = sorted(rango_nidos, key=objeto_nido, reverse=True)[:n_abandonar]
        # El mejor nido (menor objetivo penalizado) sirve de base para los nuevos.
        idx_best = min(rango_nidos, key=objeto_nido)
        base_best = nidos_sol[idx_best]
        # Para el grupo de operadores del paso de abandono usamos la violación
        # del mejor nido (idéntica lógica que la fase de generación: el sesgo
        # se decide por la solución desde la que parte el vuelo).
        viol_best = nidos_viol[idx_best]
        grupo_abandono, _ = seleccionar_grupo_operadores_inter_intra(
            rng,
            viol_best,
            ops_intra_list,
            ops_inter_list,
            ops_fallback_list,
            alpha_inter=p_efectivo,
            p_inter=p_efectivo,
            metodo=metodo_seleccion,
        )

        # Generamos nuevas soluciones para reemplazar los peores nidos.
        nuevos: list[list[list[str]]] = []
        movs_abandono: list[list[MovimientoVecindario]] = []
        for _idx in peores:
            # Cada nuevo nido se genera con un vuelo Lévy desde el mejor nido actual.
            ns, ms = _vuelo_levy_discreto(
                base_best,
                rng=rng,
                pasos_base=pasos_levy_eff,
                beta=beta_levy,
                operadores=grupo_abandono,
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
                ultimo_mov_aceptado = movs_abandono[k][-1]
                contador.aceptar(ultimo_mov_aceptado.operador)
                # Si el nido abandonado mejora el mejor global, lo registramos
                # con el operador del último movimiento del vuelo (mismo
                # criterio que en la fase de competencia).
                if pure_nv[k] < mejor_any_c - 1e-15 or (
                    viol_nv[k] < 1e-12
                    and (mejor_fact_c is None or pure_nv[k] < mejor_fact_c - 1e-15)
                ):
                    if pure_nv[k] < mejor_any_c - 1e-15:
                        mejor_any_c = float(pure_nv[k])
                        mejor_any_s = copiar_solucion_labels(nuevos[k])
                    if viol_nv[k] < 1e-12:
                        lim = mejor_fact_c if mejor_fact_c is not None else float("inf")
                        if pure_nv[k] < lim - 1e-15:
                            mejor_fact_c = float(pure_nv[k])
                            mejor_fact_s = copiar_solucion_labels(nuevos[k])
                    nuevo_rep = costo_para_reporte()
                    if nuevo_rep + 1e-12 < rep_antes_it:
                        op_mejor = movs_abandono[k][-1].operador
                        contador.registrar_mejora(op_mejor)
                        mejoras += 1
                        rep_antes_it = nuevo_rep

        # Fusión final del ciclo: asegura que cualquier mejora no detectada
        # arriba (por ejemplo, si pasos posteriores empeoraron y luego el mejor
        # se recuperó) queda capturada. También sincroniza sol_mejor y costo_mejor.
        fusionar_desde_nidos()

        # ----- Cierre del ciclo: estancamiento e historial -----
        # Actualizamos el contador de iteraciones sin mejora comparando el
        # costo reportable AL FINAL del ciclo con el snapshot del inicio.
        rep_final_it = costo_para_reporte()
        if rep_final_it + 1e-15 < rep_antes_it:
            iter_sin_mejora = 0
        else:
            iter_sin_mejora += 1

            # --- Kick por estancamiento global (strict_intra_inter_20260524) ---
            # En Cuckoo el kick es POBLACIONAL: aplicamos la rafaga inter-ruta
            # a TODOS los nidos para diversificar el conjunto. Tras el kick
            # invocamos ``fusionar_desde_nidos`` para sincronizar el mejor
            # global con la nueva poblacion (un nido kicked podria casualmente
            # mejorar el mejor, igual que un scout en ABC).
            if (max_iter_sin_mejora_kick is not None
                    and iter_sin_mejora >= max_iter_sin_mejora_kick):
                # Import diferido del kick clasico: solo se carga si la corrida
                # activa el kick Y no se provee un intensificador externo.
                if intensificador is None:
                    from metacarp.strict_intra_inter_20260524 import aplicar_kick_labels
                # La guia del intensificador es la mejor factible si existe;
                # si no, la mejor sin restriccion de factibilidad.
                guia = mejor_fact_s if mejor_fact_s is not None else mejor_any_s
                for k in rango_nidos:
                    if intensificador is not None:
                        # Respuesta de intensificacion (p.ej. Path Relinking limpio)
                        # hacia la mejor solucion global, en lugar del kick aleatorio.
                        nidos_sol[k] = intensificador(
                            nidos_sol[k], guia, ctx, lam_eff, rng, encoding, md_op
                        )
                    else:
                        nidos_sol[k] = aplicar_kick_labels(
                            nidos_sol[k], rng, md_op, encoding=encoding
                        )
                    # Recalculamos costo y violacion para ambas ramas (comun).
                    nidos_pure[k] = float(costo_rapido(nidos_sol[k], ctx))
                    nidos_viol[k] = float(exceso_capacidad_rapido(nidos_sol[k], ctx))
                # Re-fusionamos para capturar mejoras introducidas por el kick o PR.
                fusionar_desde_nidos()
                iter_sin_mejora = 0
                n_resets_kick += 1
                if max_resets is not None and n_resets_kick >= max_resets:
                    break

        # Registramos el historial al FINAL del ciclo (tras competencia y
        # abandono), no al inicio. Esto refleja la mejor solución REAL del
        # ciclo, no un snapshot obsoleto del estado anterior.
        if guardar_historial:
            historial_best.append(rep_final_it)

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
    # Fracción de iteraciones con violación media positiva (diagnóstico).
    fraccion_iter_con_viol = (
        iteraciones_con_violacion / ciclo_final if ciclo_final > 0 else 0.0
    )

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
            "mejora_absoluta": mejora_abs,
            "mejora_porcentaje": mejora_pct,
            # Las columnas "iteraciones", "num_nidos", "pasos_levy_base" y
            # "max_iter_sin_mejora" reportan los VALORES BASE pasados por el
            # usuario (cadena vacía si los dejó en None). Los valores REALMENTE
            # usados por el bucle van debajo en columnas "*_efectivo".
            "iteraciones": iteraciones_base_usuario if iteraciones_base_usuario is not None else "",
            "num_nidos": num_nidos_base_usuario if num_nidos_base_usuario is not None else "",
            "pasos_levy_base": pasos_levy_base_usuario if pasos_levy_base_usuario is not None else "",
            "max_iter_sin_mejora": (
                max_iter_sin_mejora_base_usuario if max_iter_sin_mejora_base_usuario is not None else ""
            ),
            # Factores de escala (cadena vacía si no se usaron).
            "factor_iter": factor_iter if factor_iter is not None else "",
            "factor_nidos": factor_nidos if factor_nidos is not None else "",
            "factor_pasos": factor_pasos if factor_pasos is not None else "",
            # Parámetros constantes (no instance-aware) de Cuckoo Search.
            "pa_abandono": pa_abandono,
            "beta_levy": beta_levy,
            # P(inter) base y piso bajo violación.
            "p_inter": p_inter,
            "p_inter_max_efectivo": p_inter_max_efectivo,
            # Penalización de capacidad.
            "usar_penalizacion_capacidad": usar_penalizacion_capacidad,
            "lambda_capacidad": lam_eff,
            # ---- VALORES EFECTIVOS REALES (lo que de verdad usó el bucle) ----
            "iteraciones_efectivas": iteraciones_eff,
            "num_nidos_efectivo": num_nidos_eff,
            "pasos_levy_efectivo": pasos_levy_eff,
            "max_iter_sin_mejora_efectivo": max_sin_mejora_eff,
            "n_tareas": n_tareas,
            # 36 columnas (4 categorías × 9 operadores).
            **contador.resumen_csv(),
            # Estadísticas de corrida.
            "iteraciones_totales": ciclo_final,
            "iteraciones_sin_mejora_final": iter_sin_mejora,
            "reemplazos_exitosos": reemplazos,
            "abandonos_totales": abandonos,
            "mejoras": mejoras,
            "aceptaciones_solucion_infactible": aceptaciones_sol_infactible,
            "iteraciones_con_violacion": iteraciones_con_violacion,
            "fraccion_iter_con_violacion": fraccion_iter_con_viol,
            "mejor_solucion_factible_final": mejor_factible_final,
            "mejor_solucion_tr_legible": solucion_legible_humana(sol_mejor),
            "reporte_detalle_deadheading": detalle_txt,
            "costo_total_desde_reporte": costo_total_reporte,
            # Columna del mecanismo de kick (strict_intra_inter_20260524).
            # 0 cuando la variante experimental no esta activa (default).
            "n_resets_kick": n_resets_kick,
        }
        # CORRECCIÓN DEL BUG: ``extra_csv`` antes se ignoraba. Ahora se vuelca a
        # la fila ANTES de guardar para que el script de experimentos pueda
        # añadir columnas personalizadas (p.ej. nombre del experimento).
        fila.update(extra_csv or {})
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
        iteraciones_totales=ciclo_final,
        nidos=num_nidos_eff,
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
        # Métricas instance-aware: reproducibilidad exacta desde el resultado.
        n_tareas=n_tareas,
        iteraciones_efectivas=iteraciones_eff,
        num_nidos_efectivo=num_nidos_eff,
        pasos_levy_efectivo=pasos_levy_eff,
        max_iter_sin_mejora_efectivo=max_sin_mejora_eff,
        p_inter_max_efectivo=p_inter_max_efectivo,
        factor_iter=factor_iter,
        factor_nidos=factor_nidos,
        factor_pasos=factor_pasos,
        iteraciones_sin_mejora_final=iter_sin_mejora,
        iteraciones_con_violacion=iteraciones_con_violacion,
        fraccion_iter_con_violacion=fraccion_iter_con_viol,
        # Kicks aplicados (variante experimental strict_intra_inter_20260524).
        n_resets_kick=n_resets_kick,
    )


def cuckoo_search_desde_instancia(
    nombre_instancia: str,
    *,
    root: str | None = None,
    # Parámetros instance-aware (None ⇒ calibración automática por n_tareas):
    iteraciones: int | None = None,           # None → max(200, 20·n)
    num_nidos: int | None = None,             # None → max(10, round(2·√n))
    pasos_levy_base: int | None = None,       # None → max(3, round(√n / 2))
    max_iter_sin_mejora: int | None = None,   # None → max(50, 3·n)
    pa_abandono: float = 0.25,
    beta_levy: float = 1.5,
    # Factores de escala alternativos (solo se aplican si el respectivo
    # parámetro absoluto es None). Pensados para el script de experimentos.
    factor_iter: float | None = None,
    factor_nidos: float | None = None,
    factor_pasos: float | None = None,
    semilla: int | None = None,
    operadores: Iterable[str] = OPERADORES_POPULARES,
    marcador_depot_etiqueta: str | None = None,
    usar_gpu: bool = False,
    backend_vecindario: Literal["labels", "ids"] = "labels",
    guardar_historial: bool = True,
    guardar_csv: bool = False,
    ruta_csv: str | None = None,
    repeticion: int | None = None,
    usar_penalizacion_capacidad: bool = True,
    lambda_capacidad: float | None = None,
    # ``alpha_inter`` fue ELIMINADO: el algoritmo eleva automáticamente P(inter)
    # a max(p_inter, 0.8) cuando hay violación de capacidad.
    p_inter: float = 0.6,
    metodo_seleccion: str = "canonico",  # método de combinación inter/intra: canonico|p_inter|binario|random
    # Kick reactivo (variante experimental strict_intra_inter_20260524).
    max_iter_sin_mejora_kick: int | None = None,
    max_resets: int | None = None,
    # Hook de intensificacion opcional (p.ej. Path Relinking limpio).
    intensificador: Callable | None = None,
    extra_csv: dict[str, object] | None = None,
    **_ignorado_kwargs: object,  # absorbe kwargs heredados (p.ej. id_corrida, config_id)
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
        max_iter_sin_mejora=max_iter_sin_mejora,
        beta_levy=beta_levy,
        # Propagamos los factores de escala al algoritmo principal.
        factor_iter=factor_iter,
        factor_nidos=factor_nidos,
        factor_pasos=factor_pasos,
        semilla=semilla,
        operadores=operadores,
        marcador_depot_etiqueta=marcador_depot_etiqueta,
        usar_gpu=usar_gpu,
        backend_vecindario=backend_vecindario,
        guardar_historial=guardar_historial,
        guardar_csv=guardar_csv,
        ruta_csv=ruta_csv,
        nombre_instancia=nombre_instancia,
        repeticion=repeticion,
        root=root,
        usar_penalizacion_capacidad=usar_penalizacion_capacidad,
        lambda_capacidad=lambda_capacidad,
        # ``alpha_inter`` eliminado: el piso 0.8 bajo violación se garantiza
        # internamente a partir de p_inter.
        p_inter=p_inter,
        metodo_seleccion=metodo_seleccion,
        # Propagamos el mecanismo de kick (variante experimental).
        max_iter_sin_mejora_kick=max_iter_sin_mejora_kick,
        max_resets=max_resets,
        # Propagamos el hook de intensificacion (p.ej. Path Relinking limpio).
        intensificador=intensificador,
        extra_csv=extra_csv,
    )
