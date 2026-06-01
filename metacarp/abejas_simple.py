"""
Artificial Bee Colony (ABC) SIMPLE — versión canónica de Karaboga (2005) para CARP.

Diferencias con ``abejas.py``
-----------------------------
Esta implementación está pensada como **referencia didáctica** y baseline limpio
para comparación contra la versión ``busqueda_abejas`` (que ya incluye varios
añadidos heurísticos, como sesgo inter/intra-ruta a lo largo de las tres fases,
backend_vecindario configurable y rastreo de mejor factible). Aquí seguimos la
estructura más fiel a Karaboga 2005 con UNA modificación intencional:

* **Scouts canónicos**: cuando una fuente alcanza ``limite_abandono`` intentos
  sin mejora se reemplaza por una **solución completamente ALEATORIA** generada
  desde cero (barajando todas las tareas requeridas y empacándolas greedy por
  capacidad). En la versión ``busqueda_abejas`` los scouts son simplemente
  vecinos de la mejor fuente; aquí volvemos al espíritu original del algoritmo
  donde el scout aterriza en un punto del espacio sin sesgo histórico.

* **Sesgo inter/intra-ruta en empleadas y observadoras**: usamos el mismo
  helper ``seleccionar_grupo_operadores_inter_intra`` que comparten SA, TS
  simple y RTS para que la metaheurística pueda reparar capacidad de forma
  consistente con el resto del proyecto. Es la única "modernización" que
  añadimos al ABC simple: garantiza que el algoritmo no se quede atascado en
  vecinos infactibles cuando la solución viola capacidad.

* **Bug de ``registrar_mejora`` corregido**: en ``busqueda_abejas`` la mejora
  global se detecta DESPUÉS del bucle de fase (comparando ``costo_para_reporte``
  antes/después) y se imputa al ``ultimo_movimiento_aceptado``, lo cual puede
  ser un movimiento posterior al que realmente produjo la mejora. Aquí
  registramos la mejora INMEDIATAMENTE al detectar que un vecino mejoró el
  mejor global, dentro del mismo if que lo actualiza. Como consecuencia,
  ``sum(operadores_mejoraron.values())`` coincide exactamente con ``mejoras``.

* **Criterio de parada por estancamiento**: añadimos ``max_iter_sin_mejora``
  (None = desactivado) para alinear la API con SA / TS simple / RTS.

Resumen de la estructura del bucle por iteración (ciclo ABC)
------------------------------------------------------------
1. Fase EMPLEADAS: cada una de las ``num_fuentes`` empleadas genera un vecino
   de SU fuente (siempre la misma, no se mueve), lo evalúa y, si mejora,
   reemplaza la fuente y resetea ``trials[i]``; si no mejora, ``trials[i] += 1``.

2. Fase OBSERVADORAS: se calcula un fitness inverso al objetivo penalizado de
   cada fuente, se normaliza a probabilidades y se eligen ``num_fuentes``
   índices (con repetición) por ruleta. Para cada índice se genera UN vecino
   de esa fuente y se aplica el mismo criterio greedy de las empleadas.

3. Fase SCOUTS: las fuentes con ``trials[i] >= limite_abandono`` se
   sustituyen por soluciones **aleatorias puras** generadas mediante
   ``_generar_solucion_aleatoria``.

4. Se actualiza el contador ``iter_sin_mejora`` y, si corresponde, se rompe
   por el criterio de estancamiento.

Por qué la GPU SOLO acelera la evaluación
-----------------------------------------
Los vecinos se generan en CPU porque los operadores combinatorios trabajan
sobre listas Python (no son operaciones de matriz). La GPU se aprovecha en la
**evaluación en lote** (``costo_lote_penalizado_ids``) de los ``num_fuentes``
vecinos de la fase observadoras, que sí es una reducción vectorial densa
trivialmente paralelizable. Por eso ``usar_gpu=True`` mueve tiempo en el
``construir_contexto`` y en las llamadas de lote, pero no en la generación.
"""
# Permite escribir tipos como ``float | None`` en Python < 3.10.
from __future__ import annotations

# Módulo estándar para generar números pseudoaleatorios reproducibles.
import random
# Para medir tiempo de ejecución con alta resolución.
import time
# math.sqrt: usado en el cálculo instance-aware de ``num_fuentes`` por defecto.
# La heurística es ``num_fuentes ~ 2·sqrt(n_tareas)``, inspirada en la práctica
# habitual de ABC donde el tamaño de la colonia crece con la raíz cuadrada de la
# dimensión del problema (compromiso entre exploración paralela y costo por ciclo).
import math
# Tipos abstractos para anotaciones (Iterable, Mapping).
from collections.abc import Iterable, Mapping
# @dataclass crea el boilerplate de clases de datos; field() configura defaults complejos.
from dataclasses import dataclass, field
# Any para tipados muy genéricos (resultado del unpickle de soluciones iniciales).
# Callable para anotar el hook de intensificacion opcional (p.ej. Path Relinking).
from typing import Any, Callable

# NetworkX: librería para manipular el grafo CARP.
import networkx as nx

# Importaciones internas del paquete metacarp.
from .busqueda_indices import (
    build_search_encoding,  # construye el encoding label<->id si el contexto no lo trae
    encode_solution,         # convierte solución por labels a listas de ids
)
from .cargar_grafos import cargar_objeto_gexf                # carga grafo desde GEXF
from .cargar_soluciones_iniciales import cargar_solucion_inicial  # solución inicial pickled
from .evaluador_costo import (
    costo_lote_penalizado_ids,           # evaluación vectorizada de lote (objetivo + costo + violación)
    costo_rapido_ids,                    # evaluación rápida de UNA solución por IDs
    exceso_capacidad_sol_ids,            # violación de capacidad por IDs (sin pasar por labels)
    lambda_penal_capacidad_por_defecto,  # λ por defecto basado en deadhead mediano
    objectivo_penalizado,                # combinador escalar costo + λ·violación
)
from .instances import load_instances    # lee el dict de datos de la instancia CARP
from .metaheuristicas_utils import (
    ContadorOperadores,                  # estadística de uso de operadores
    calcular_metricas_gap,               # gap absoluto / porcentual respecto a inicial
    construir_contexto_para_corrida,     # contexto de evaluación rápida (cache GEXF + Dijkstra)
    copiar_solucion_labels,              # copia profunda con normalización de strings
    generar_reporte_detallado,           # genera texto de reporte de la mejor solución
    guardar_resultado_csv,               # persiste una fila de resultados en CSV
    resumen_bks_csv,                     # columnas estándar de comparación con BKS
    seleccionar_grupo_operadores_inter_intra,  # helper compartido con SA / TS / RTS
    seleccionar_mejor_inicial_rapido,    # elige la mejor solución inicial entre las candidatas
    solucion_legible_humana,             # convierte solución a texto humano para el CSV
)
from .vecindarios import (
    MovimientoVecindario,
    OPERADORES_INTER,                    # subset oficial inter-ruta (canon del proyecto)
    OPERADORES_INTRA,                    # subset oficial intra-ruta (canon del proyecto)
    OPERADORES_POPULARES,                # lista canónica de 9 operadores
    generar_vecino_ids,                  # genera un vecino sobre la representación por IDs
)


# Importaciones del ContextoEvaluacion solo para anotaciones de tipo (TYPE_CHECKING evita el costo runtime).
# Se mantiene aquí en runtime para que el tipo sea utilizable en la firma de los helpers.
from .evaluador_costo import ContextoEvaluacion


# API pública del módulo: lo que se exporta vía ``from metacarp import ...``.
__all__ = [
    "AbejasSimpleResult",
    "busqueda_abejas_simple",
    "busqueda_abejas_simple_desde_instancia",
]


# --- CONCEPTO OOP: @dataclass(frozen=True, slots=True) ---
# frozen=True: el objeto resultado es inmutable, evita modificaciones accidentales tras la corrida.
# slots=True : reduce el footprint de memoria al no crear __dict__ por instancia.
@dataclass(frozen=True, slots=True)
class AbejasSimpleResult:
    """
    Resultado completo de la búsqueda ABC simple.

    Agrupa la mejor solución encontrada, métricas de calidad respecto al
    inicial, tiempo, contadores de operadores y diagnóstico de las fases
    de empleadas, observadoras y scouts.
    """

    # La mejor solución hallada durante la búsqueda, en formato de etiquetas legibles.
    mejor_solucion: list[list[str]]
    # Costo PURO (sin penalización) de la mejor solución.
    mejor_costo: float
    # Solución inicial elegida como referencia para medir mejora.
    solucion_inicial_referencia: list[list[str]]
    # Costo puro de la solución inicial de referencia.
    costo_solucion_inicial: float
    # Diferencia absoluta de costo: costo_inicial − costo_mejor (>0 ⇒ mejora).
    mejora_absoluta: float
    # Porcentaje de mejora del inicial al final.
    mejora_porcentaje_inicial_vs_final: float
    # Tiempo total de ejecución de la corrida en segundos.
    tiempo_segundos: float
    # Número total de iteraciones (ciclos completos) ejecutadas.
    iteraciones_totales: int
    # Iteraciones consecutivas sin mejora al cierre (criterio temprano).
    iteraciones_sin_mejora_final: int
    # Número de fuentes de alimento mantenidas en paralelo (= num_fuentes).
    fuentes_alimento: int
    # Veces que se reinició una fuente por agotamiento (fase scout).
    scouts_reinicios: int
    # Veces que el mejor global mejoró durante la corrida.
    mejoras: int
    # Semilla del RNG (None = aleatoria del sistema).
    semilla: int | None
    # Backend efectivo de evaluación: "cpu" o "gpu".
    backend_evaluacion: str = field(default="cpu")
    # Si True, se aplicó penalización por violación de capacidad en el objetivo.
    usar_penalizacion_capacidad: bool = field(default=True)
    # λ efectivo de penalización (0.0 cuando ``usar_penalizacion_capacidad`` está apagado).
    lambda_capacidad: float = field(default=0.0)
    # True si la mejor solución reportada respeta todas las restricciones de capacidad.
    mejor_solucion_factible_final: bool = field(default=True)
    # Cuántas aceptaciones del bucle aceptaron una solución que sigue siendo infactible.
    aceptaciones_solucion_infactible: int = field(default=0)
    # Iteraciones del bucle en las que la violación media de fuentes era > 0.
    iteraciones_con_violacion: int = field(default=0)
    # Fracción [0, 1] de iteraciones con violación media positiva.
    fraccion_iter_con_violacion: float = field(default=0.0)
    # Estadísticas por operador: propuestos / aceptados / mejoraron / trayectoria_mejor.
    operadores_propuestos: dict[str, int] = field(default_factory=dict)
    operadores_aceptados: dict[str, int] = field(default_factory=dict)
    operadores_mejoraron: dict[str, int] = field(default_factory=dict)
    operadores_trayectoria_mejor: dict[str, int] = field(default_factory=dict)
    # Trayectoria del mejor costo a lo largo de las iteraciones (vacía si guardar_historial=False).
    historial_mejor_costo: list[float] = field(default_factory=list)
    # Ruta del CSV generado o None si no se guardó.
    archivo_csv: str | None = field(default=None)
    # --- Métricas instance-aware (añadidas en la versión con calibración) ---
    # Número de tareas requeridas (= len(ctx.u_arr)). Equivale al "n" usado
    # como variable de escala en las fórmulas default.
    n_tareas: int = field(default=0)
    # Valores EFECTIVOS realmente utilizados en el bucle. Pueden venir del
    # valor absoluto que pasó el usuario, de un factor de escala, o de la
    # fórmula default si el usuario no pasó nada. Reportarlos permite
    # reproducir exactamente la corrida desde el CSV.
    iteraciones_efectivas: int = field(default=0)
    num_fuentes_efectivo: int = field(default=0)
    limite_abandono_efectivo: int = field(default=0)
    max_iter_sin_mejora_efectivo: int | None = field(default=None)
    # p_inter máximo efectivo (= max(p_inter, 0.8)) que se aplicó bajo
    # violación de capacidad. Si la corrida no tuvo violación, este valor
    # documenta lo que el algoritmo HABRÍA usado si la hubiera tenido.
    p_inter_max_efectivo: float = field(default=0.0)
    # Numero de kicks (perturbaciones inter-ruta) aplicados durante la corrida.
    # Solo es > 0 cuando se pasa max_iter_sin_mejora_kick al wrapper de la
    # variante experimental strict_intra_inter_20260524.
    n_resets_kick: int = field(default=0)


# ---------------------------------------------------------------------------
# Helper privado: _generar_solucion_aleatoria
# ---------------------------------------------------------------------------
def _generar_solucion_aleatoria(
    ctx: ContextoEvaluacion,
    rng: random.Random,
    marcador_depot: str,
) -> list[list[str]]:
    """
    Construye una solución completamente aleatoria para los scouts canónicos.

    Estrategia (sencilla y fiel al espíritu Karaboga 2005):

    1. Se toman TODOS los IDs de tareas requeridas conocidos por el encoding.
    2. Se barajan con ``rng`` para obtener un orden uniformemente aleatorio.
    3. Se recorren en ese orden y se empacan en rutas mediante un criterio
       greedy de capacidad: si la tarea cabe en la ruta actual la añadimos;
       si no, abrimos una nueva ruta.

    Esta heurística NO garantiza factibilidad estricta: si alguna tarea
    individual supera la capacidad de un vehículo, igualmente se inserta en una
    ruta nueva (porque no hay alternativa) y la solución resultante tendrá
    violación positiva que la penalización del objetivo se encargará de
    desincentivar. Esa decisión es coherente con que el algoritmo trabaje con
    objetivo penalizado: el scout es solo un PUNTO de aterrizaje aleatorio,
    no necesariamente bueno, y las fases siguientes pueden mejorarlo.

    Retorna la solución en formato de **etiquetas con depósito** al inicio y
    al final de cada ruta, listo para pasar por ``encode_solution`` o por
    cualquier evaluador label-based.
    """
    encoding = ctx.encoding
    # Lista de IDs de todas las tareas requeridas (0..n-1).
    todas_ids = list(range(len(encoding.id_to_label)))
    # Barajamos el orden en sitio. ``rng.shuffle`` es el método estándar de
    # ``random.Random`` para permutar una lista uniformemente.
    rng.shuffle(todas_ids)

    # Acumuladores de rutas y de demanda por ruta actual.
    rutas_labels: list[list[str]] = []
    ruta_actual_labels: list[str] = []
    demanda_actual = 0.0
    capacidad = float(ctx.capacidad_max)
    # demanda_arr está indexada por ID de tarea (acceso O(1)).
    demanda_arr = ctx.demanda_arr

    for tid in todas_ids:
        dem_t = float(demanda_arr[tid])
        # Si la tarea cabe en la ruta actual (o la capacidad es infinita / 0
        # interpretada como "sin restricción"), la añadimos a la ruta actual.
        # Si NO cabe y la ruta actual ya tiene al menos una tarea, cerramos
        # la ruta actual y abrimos una nueva con esta tarea (aunque sola
        # exceda la capacidad: a partir de ese punto el objetivo penalizado
        # decidirá su destino).
        cabe = (
            capacidad <= 0 or not (capacidad < float("inf"))
            or (demanda_actual + dem_t <= capacidad + 1e-12)
        )
        if not cabe and ruta_actual_labels:
            # Cerramos la ruta actual y arrancamos otra.
            rutas_labels.append(ruta_actual_labels)
            ruta_actual_labels = []
            demanda_actual = 0.0
        ruta_actual_labels.append(encoding.id_to_label[tid])
        demanda_actual += dem_t

    # Cerramos la última ruta si quedó con tareas.
    if ruta_actual_labels:
        rutas_labels.append(ruta_actual_labels)

    # Envolvemos cada ruta con el marcador de depósito al inicio y al final
    # (formato canónico que esperan los evaluadores label-based).
    return [[marcador_depot, *ruta, marcador_depot] for ruta in rutas_labels]


# ---------------------------------------------------------------------------
# Helper privado: _generar_vecinos_lote_simple
# ---------------------------------------------------------------------------
def _generar_vecinos_lote_simple(
    sources: list[list[list[int]]],
    *,
    rng: random.Random,
    ops_intra: list[str],
    ops_inter: list[str],
    ops_fallback: list[str],
    violacion_media: float,
    alpha_inter: float,
    p_inter: float,
    ctx: ContextoEvaluacion,
) -> tuple[list[list[list[int]]], list[MovimientoVecindario]]:
    """
    Genera un lote de vecinos en representación de IDs, uno por cada solución
    de ``sources``, aplicando el helper inter/intra-ruta compartido.

    Para cada solución fuente:
      1. Elegimos un GRUPO de operadores (inter o intra) mediante el helper
         compartido (un único ``rng.random()`` para preservar la secuencia).
      2. Llamamos a ``generar_vecino_ids`` con ese grupo, dejando que el
         operador concreto se elija uniformemente dentro del grupo.

    El criterio del grupo se decide por SOLUCIÓN, no por lote: cada fuente
    puede entrar con su propio sesgo, lo cual es coherente con el hecho de
    que en la fase observadoras la misma fuente puede ser muestreada varias
    veces y queremos diversidad real entre los vecinos generados.

    Devuelve un par paralelo (lista de vecinos IDs, lista de movimientos).
    """
    vecinos: list[list[list[int]]] = []
    movs: list[MovimientoVecindario] = []
    for sol_ids in sources:
        # Seleccionamos el grupo (un solo ``rng.random()`` por elección).
        grupo, _hubo_viol = seleccionar_grupo_operadores_inter_intra(
            rng,
            violacion_media,
            ops_intra,
            ops_inter,
            ops_fallback,
            alpha_inter=alpha_inter,
            p_inter=p_inter,
        )
        # Generamos UN vecino con el grupo elegido (uniforme dentro del grupo).
        # NOTA: generar_vecino_ids puede fallar si la solución es tan pequeña
        # que ningún operador es aplicable; en la práctica las soluciones
        # tienen tamaño suficiente y este caso no ocurre en las instancias
        # estándar. Lo dejamos sin manejo defensivo para que un fallo real
        # sí se propague durante depuración.
        vec_ids, mov = generar_vecino_ids(
            sol_ids,
            rng=rng,
            operadores=grupo,
            pesos_operadores=None,           # selección uniforme dentro del grupo
            usar_gpu=ctx.usar_gpu,
            encoding=ctx.encoding,
        )
        vecinos.append(vec_ids)
        movs.append(mov)
    return vecinos, movs


# ---------------------------------------------------------------------------
# Función principal: busqueda_abejas_simple
# ---------------------------------------------------------------------------
def busqueda_abejas_simple(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    # ---- Parámetros instance-aware (None ⇒ se calculan a partir de n_tareas) ----
    # Si el usuario pasa un valor concreto, manda el valor concreto.
    # Si no, se usa la fórmula de escala (ver bloque "calibración" del cuerpo).
    iteraciones: int | None = None,           # None → max(200, 20·n_tareas)
    num_fuentes: int | None = None,           # None → max(10, round(2·√n_tareas))
    limite_abandono: int | None = None,       # None → max(15, n_tareas // 2)
    max_iter_sin_mejora: int | None = None,   # None → max(50, 3·n_tareas)
    # ---- Factores de escala alternativos (sobrescriben la fórmula default) ----
    # Permiten que el script de experimentos barra POR FACTOR sin tener que
    # calcular él mismo el valor absoluto (que depende de cada instancia).
    # Precedencia: valor absoluto > factor > default calculado por fórmula.
    factor_fuentes: float | None = None,      # si se pasa, num_fuentes  = max(10, round(factor · √n))
    factor_abandono: float | None = None,     # si se pasa, limite_abandono = max(15, round(factor · n))
    factor_iter: int | float | None = None,   # si se pasa, iteraciones = max(200, round(factor · n))
    semilla: int | None = None,             # semilla del RNG; None = aleatoria del sistema
    operadores: Iterable[str] = OPERADORES_POPULARES,  # operadores de vecindario activos
    marcador_depot_etiqueta: str | None = None,  # token de depósito (None = el del contexto)
    usar_gpu: bool = False,                  # si True, evaluación en lote intenta usar GPU
    guardar_historial: bool = True,          # si True, registra el mejor costo en cada iteración
    guardar_csv: bool = False,               # si True, escribe la fila de resultados en CSV
    ruta_csv: str | None = None,             # ruta del CSV (None = nombre automático)
    nombre_instancia: str = "instancia",     # nombre de la instancia (para CSV)
    repeticion: int | None = None,           # número de repetición dentro de un experimento
    root: str | None = None,                 # directorio raíz para localizar los datos
    usar_penalizacion_capacidad: bool = True,  # si True, objetivo = costo + λ·violación
    lambda_capacidad: float | None = None,    # λ explícito (None = automático por contexto)
    # p_inter BASE: P(elegir inter) cuando la solución actual es factible.
    # Cuando hay violación de capacidad, el algoritmo eleva automáticamente
    # la P(inter) a max(p_inter, 0.8) — el parámetro ``alpha_inter`` fue
    # eliminado para que el usuario solo configure UN punto de la curva y el
    # piso 0.8 bajo violación esté garantizado por construcción.
    p_inter: float = 0.6,
    # --- Kick reactivo (variante experimental strict_intra_inter_20260524) ---
    # Cuando ``iter_sin_mejora`` alcanza este umbral se aplica una perturbacion
    # INTER-RUTA disruptiva a TODAS las fuentes y se reinicia el contador.
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
    extra_csv: dict[str, object] | None = None,  # columnas extra para el CSV (no se usan aquí)
    **_ignorado_kwargs: object,              # absorbe kwargs heredados (id_corrida, config_id, ...)
) -> AbejasSimpleResult:
    """
    Ejecuta ABC SIMPLE (canónico Karaboga 2005) con scouts aleatorios y sesgo
    inter/intra-ruta en empleadas y observadoras.

    Parámetros clave
    ----------------
    iteraciones : int | None
        Número de ciclos completos del algoritmo (cada ciclo = empleadas →
        observadoras → scouts). ``None`` ⇒ se calibra a ``max(200, 20·n_tareas)``.
    num_fuentes : int | None
        Número de fuentes mantenidas en paralelo. ``None`` ⇒ se calibra a
        ``max(10, round(2·√n_tareas))``.
    limite_abandono : int | None
        Una fuente que acumule este número de intentos fallidos consecutivos
        se reemplaza por una solución aleatoria. ``None`` ⇒ se calibra a
        ``max(15, n_tareas // 2)``.
    max_iter_sin_mejora : int | None
        Si se especifica, el bucle termina cuando lleve esa cantidad de
        iteraciones consecutivas sin mejorar el mejor global. ``None`` ⇒ se
        calibra a ``max(50, 3·n_tareas)`` (igual a la calibración instance-aware
        de RTS), de modo que el criterio de parada siempre esté activo y
        proporcional al tamaño de la instancia.
    factor_fuentes : float | None
        Factor de escala alternativo: si se pasa, ``num_fuentes`` se calcula
        como ``max(10, round(factor · √n_tareas))``. Solo se usa cuando
        ``num_fuentes`` es ``None``. Pensado para el grid del script de
        experimentos.
    factor_abandono : float | None
        Análogo: ``limite_abandono = max(15, round(factor · n_tareas))``.
    factor_iter : int | float | None
        Análogo: ``iteraciones = max(200, round(factor · n_tareas))``.
    p_inter : float (0..1)
        P(elegir el grupo INTER-RUTA) en una decisión cuando la solución es
        FACTIBLE. Cuando hay violación de capacidad, el algoritmo eleva
        automáticamente esa probabilidad a ``max(p_inter, 0.8)`` para
        favorecer operadores que redistribuyen carga entre rutas. El usuario
        configura un único valor base; el piso 0.8 bajo violación es una
        garantía interna del algoritmo (ya no se expone como parámetro).

    Devuelve un ``AbejasSimpleResult`` inmutable con la mejor solución y
    diagnóstico completo de la corrida.
    """
    # ----------------------------------------------------------
    # 1) Validaciones tempranas de parámetros.
    #    Solo validamos los valores que el USUARIO pasó explícitamente. Los
    #    parámetros instance-aware (None) se validarán implícitamente tras el
    #    cálculo (ver más abajo en el bloque "calibración instance-aware").
    # ----------------------------------------------------------
    if iteraciones is not None and iteraciones <= 0:
        raise ValueError("iteraciones debe ser > 0.")
    if num_fuentes is not None and num_fuentes <= 1:
        raise ValueError("num_fuentes debe ser >= 2.")
    if limite_abandono is not None and limite_abandono <= 0:
        raise ValueError("limite_abandono debe ser > 0.")
    # ``p_inter`` siempre llega como float (no admite None): se valida directo.
    if not (0.0 <= p_inter <= 1.0):
        raise ValueError("p_inter debe estar en [0, 1].")
    if max_iter_sin_mejora is not None and max_iter_sin_mejora <= 0:
        raise ValueError("max_iter_sin_mejora debe ser > 0 si se especifica.")
    # Validación de los factores de escala: solo si el usuario los pasó.
    # Los tres factores deben ser positivos para producir un valor sensato.
    if factor_fuentes is not None and factor_fuentes <= 0:
        raise ValueError("factor_fuentes debe ser > 0 si se especifica.")
    if factor_abandono is not None and factor_abandono <= 0:
        raise ValueError("factor_abandono debe ser > 0 si se especifica.")
    if factor_iter is not None and factor_iter <= 0:
        raise ValueError("factor_iter debe ser > 0 si se especifica.")

    # ----------------------------------------------------------
    # 2) RNG reproducible y marca de tiempo.
    # ----------------------------------------------------------
    # random.Random(None) produce una secuencia aleatoria distinta por proceso
    # (semilla del sistema). Esto es lo que queremos para los experimentos
    # paralelos del script: cada repetición observa una trayectoria distinta.
    rng = random.Random(semilla)
    t0 = time.perf_counter()

    # ----------------------------------------------------------
    # 3) Construcción del contexto de evaluación rápida.
    #    Precomputa la matriz Dijkstra y los arrays NumPy de tareas; cachea
    #    en disco si la instancia ya fue procesada antes.
    # ----------------------------------------------------------
    ctx = construir_contexto_para_corrida(
        data,
        G,
        nombre_instancia=nombre_instancia if nombre_instancia != "instancia" else None,
        usar_gpu=usar_gpu,
        root=root,
    )

    # λ efectiva para la penalización: si el usuario no la fijó, calculamos
    # el valor por defecto (~10× la mediana del deadhead) que ya usan los
    # demás metaheurísticos del proyecto.
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
    # con la misma convención que usa SA (recocido_simulado.py).
    #
    # Reglas de precedencia para cada parámetro escalable:
    #   1. Si el usuario pasó el valor ABSOLUTO (entero), se usa tal cual.
    #   2. Si pasó un FACTOR de escala, se calcula con la fórmula del factor.
    #   3. Si no pasó nada, se usa la fórmula DEFAULT en función de n_tareas.
    #
    # Las fórmulas default replican las heurísticas usadas en RTS:
    #   iteraciones      → max(200, 20 · n)
    #   num_fuentes      → max(10, round(2 · √n))
    #   limite_abandono  → max(15, n // 2)
    #   max_sin_mejora   → max(50, 3 · n)
    # Los pisos (200, 10, 15, 50) garantizan que instancias muy pequeñas no
    # ejecuten corridas degeneradas con tamaños insignificantes.
    # ----------------------------------------------------------
    n_tareas = int(len(ctx.u_arr))

    # iteraciones efectivas: absoluto > factor > default.
    if iteraciones is not None:
        iteraciones_eff = int(iteraciones)
    elif factor_iter is not None:
        iteraciones_eff = max(200, int(round(float(factor_iter) * n_tareas)))
    else:
        iteraciones_eff = max(200, 20 * n_tareas)

    # num_fuentes efectivo: absoluto > factor (escala con √n) > default.
    if num_fuentes is not None:
        num_fuentes_eff = int(num_fuentes)
    elif factor_fuentes is not None:
        num_fuentes_eff = max(10, int(round(float(factor_fuentes) * math.sqrt(n_tareas))))
    else:
        num_fuentes_eff = max(10, int(round(2.0 * math.sqrt(n_tareas))))

    # limite_abandono efectivo: absoluto > factor (escala con n) > default.
    if limite_abandono is not None:
        limite_abandono_eff = int(limite_abandono)
    elif factor_abandono is not None:
        limite_abandono_eff = max(15, int(round(float(factor_abandono) * n_tareas)))
    else:
        limite_abandono_eff = max(15, n_tareas // 2)

    # max_iter_sin_mejora efectivo: si el usuario lo dejó en None, lo
    # ACTIVAMOS automáticamente con un valor instance-aware (al contrario que
    # antes, donde None significaba "desactivado"). Justificación: en el
    # nuevo diseño instance-aware queremos que TODA corrida tenga criterio
    # de parada anticipada para evitar gastar tiempo en plateaus largos.
    if max_iter_sin_mejora is not None:
        max_sin_mejora_eff: int | None = int(max_iter_sin_mejora)
    else:
        max_sin_mejora_eff = max(50, 3 * n_tareas)

    # Guardamos los VALORES BASE que pasó el usuario (para reportarlos en el
    # CSV como cadenas vacías cuando son None). Estos valores muestran qué
    # configuró el usuario, no qué usó realmente el algoritmo.
    iteraciones_base_usuario = iteraciones
    num_fuentes_base_usuario = num_fuentes
    limite_abandono_base_usuario = limite_abandono
    max_iter_sin_mejora_base_usuario = max_iter_sin_mejora

    # p_inter máximo efectivo bajo violación: piso garantizado de 0.8.
    # Esto documenta lo que el algoritmo USARÁ en presencia de violación,
    # independientemente del valor base que pase el usuario.
    p_inter_max_efectivo = max(float(p_inter), 0.8)

    # Etiqueta del depósito que esperan los operadores cuando trabajan con
    # labels. Aquí trabajamos en IDs, pero la necesitamos para serializar
    # la solución inicial y la mejor final al formato legible.
    md_op = marcador_depot_etiqueta or ctx.marcador_depot

    # ----------------------------------------------------------
    # 4) Precomputación UNA VEZ de las particiones intra/inter/fallback.
    #    Se pasan al helper compartido en cada iteración del bucle sin
    #    reconstruirlas (microoptimización idéntica a la de RTS y SA).
    # ----------------------------------------------------------
    ops_list = list(operadores)
    ops_intra_list = [op for op in ops_list if op in OPERADORES_INTRA]
    ops_inter_list = [op for op in ops_list if op in OPERADORES_INTER]
    ops_fallback_list = list(ops_list)

    # ----------------------------------------------------------
    # 5) Selección de la mejor solución inicial entre las candidatas.
    # ----------------------------------------------------------
    sel_ini = seleccionar_mejor_inicial_rapido(
        inicial_obj,
        ctx,
        usar_penalizacion_capacidad=usar_penalizacion_capacidad,
        lambda_capacidad=lambda_capacidad,
    )
    sol_ref_labels = sel_ini.solucion         # solución de referencia en formato labels
    costo_ref = sel_ini.costo_puro            # costo PURO de la referencia
    viol_ref = float(sel_ini.violacion_capacidad)

    # Encoding usado para convertir entre labels y IDs en toda la corrida.
    encoding = ctx.encoding
    if encoding is None:
        # Fallback defensivo: construir_contexto_para_corrida normalmente sí
        # devuelve un encoding, pero si por alguna ruta de carga viniera vacío
        # lo reconstruimos a partir de los datos crudos.
        encoding = build_search_encoding(data)

    # ----------------------------------------------------------
    # 6) Inicialización de las FUENTES (representación por IDs).
    #
    #    Estrategia coherente con Karaboga 2005:
    #    - La fuente 0 se siembra con la solución inicial (mejor candidata).
    #    - Las fuentes 1..N-1 se siembran con un vecino de la fuente 0 para
    #      arrancar con DIVERSIDAD inmediata. Una alternativa sería sembrar
    #      las N con soluciones aleatorias (todavía más Karaboga puro), pero
    #      eso desaprovecharía la información de la solución inicial y haría
    #      la búsqueda mucho más lenta en instancias pequeñas. La variante
    #      adoptada es el compromiso típico de la literatura aplicada.
    # ----------------------------------------------------------
    # Codificamos la solución de referencia a IDs.
    fuente_inicial_ids = encode_solution(sol_ref_labels, encoding)
    fuentes_ids: list[list[list[int]]] = [fuente_inicial_ids]
    fuentes_pure: list[float] = [float(costo_ref)]
    fuentes_viol: list[float] = [viol_ref]
    # ``trials[i]`` = cuántos intentos consecutivos ha fallado la fuente i.
    trials: list[int] = [0]

    # Mientras no completemos las N fuentes, generamos vecinos de la fuente 0
    # con la lista completa de operadores (sin sesgo aún, porque todavía no
    # tenemos violación media calculada sobre el conjunto).
    # Usamos ``num_fuentes_eff`` (el valor instance-aware, posiblemente
    # derivado del factor o de la fórmula default).
    while len(fuentes_ids) < num_fuentes_eff:
        vec_ids, _mov = generar_vecino_ids(
            fuentes_ids[0],
            rng=rng,
            operadores=ops_list,
            pesos_operadores=None,
            usar_gpu=ctx.usar_gpu,
            encoding=encoding,
        )
        cp = costo_rapido_ids(vec_ids, ctx)
        vp = exceso_capacidad_sol_ids(vec_ids, ctx)
        fuentes_ids.append(vec_ids)
        fuentes_pure.append(float(cp))
        fuentes_viol.append(float(vp))
        trials.append(0)

    # ----------------------------------------------------------
    # 7) Inicialización del MEJOR GLOBAL y de contadores de diagnóstico.
    # ----------------------------------------------------------
    # Mejor global rastreado como el MÍNIMO objetivo penalizado entre todas
    # las fuentes (no solo la inicial), para reflejar cualquier mejora ya
    # introducida por la siembra de vecinos.
    def _obj_idx(i: int) -> float:
        """Objetivo penalizado de la fuente i (función local cerrando sobre listas)."""
        return float(
            objectivo_penalizado(
                fuentes_pure[i],
                fuentes_viol[i],
                usar_penal=usar_penalizacion_capacidad,
                lam=lam_eff,
            )
        )

    # Localizamos la mejor fuente inicial y la fijamos como mejor global de partida.
    idx_inicial_mejor = min(range(num_fuentes_eff), key=_obj_idx)
    mejor_costo: float = _obj_idx(idx_inicial_mejor)
    # Guardamos la mejor solución en formato IDs (más liviano) y la convertimos
    # a labels solo al final para reportarla.
    mejor_sol_ids: list[list[int]] = [list(r) for r in fuentes_ids[idx_inicial_mejor]]
    # Tracking del costo puro del mejor (independientemente de la penalización),
    # útil para el CSV donde reportamos el costo PURO como "mejor_costo".
    mejor_costo_puro: float = float(fuentes_pure[idx_inicial_mejor])
    mejor_viol: float = float(fuentes_viol[idx_inicial_mejor])

    # Contadores agregados de la corrida.
    mejoras = 0                 # veces que el mejor global mejoró
    scouts = 0                  # veces que se reinició una fuente (fase scout)
    aceptaciones_infactible = 0  # aceptaciones cuyo vecino sigue siendo infactible
    iteraciones_con_violacion = 0  # ciclos con violación media positiva al inicio
    iter_sin_mejora = 0
    # Numero de kicks (perturbaciones inter-ruta) aplicados durante la corrida.
    # Solo se incrementa si ``max_iter_sin_mejora_kick`` esta activo.
    n_resets_kick: int = 0
    contador = ContadorOperadores()
    historial_best: list[float] = []
    # Número de ciclos efectivamente ejecutados (puede ser < iteraciones si paramos antes).
    ciclo_final = 0

    # ============================================================
    # === BUCLE PRINCIPAL ABC SIMPLE =============================
    # ============================================================
    # Usamos ``iteraciones_eff`` (instance-aware) como tope duro.
    for ciclo in range(iteraciones_eff):
        ciclo_final = ciclo + 1

        # 8a) Criterio de parada anticipada por estancamiento.
        # ``max_sin_mejora_eff`` es el valor instance-aware: nunca es None en
        # la nueva versión (se calibra a max(50, 3·n) cuando el usuario no lo
        # especifica). Mantenemos el check de None por seguridad defensiva.
        if max_sin_mejora_eff is not None and iter_sin_mejora >= max_sin_mejora_eff:
            # Restamos 1 porque el ciclo_final ya se incrementó al entrar
            # pero esta iteración no llegó a ejecutarse. Esto mantiene
            # ``iteraciones_totales`` igual al número REAL de ciclos completos.
            ciclo_final -= 1
            break

        # Snapshot del mejor al inicio del ciclo (para detectar mejora del ciclo).
        mejor_inicio_ciclo = mejor_costo

        # Violación media de las fuentes ANTES de las fases (driver del sesgo).
        viol_media = sum(fuentes_viol) / num_fuentes_eff
        if viol_media > 0:
            iteraciones_con_violacion += 1

        # p_inter dinámico: cuando hay violación de capacidad, subimos a >=0.8
        # automáticamente para favorecer operadores inter-ruta que reequilibran
        # cargas. El usuario elige p_inter base; el algoritmo garantiza el piso
        # 0.8 bajo violación. Cuando NO hay violación, se respeta p_inter tal
        # cual lo pasó el usuario.
        p_efectivo = p_inter_max_efectivo if viol_media > 0 else float(p_inter)

        # =====================================================
        # FASE 1 — EMPLEADAS
        # Una empleada por fuente; genera un vecino y compara greedy.
        # =====================================================
        for i in range(num_fuentes_eff):
            # Elegimos el grupo de operadores (inter o intra) según la violación
            # MEDIA del conjunto, no la de la fuente i. Esta decisión global por
            # ciclo evita ráfagas erráticas y se alinea con el comportamiento de
            # los demás metaheurísticos del proyecto.
            # Pasamos ``p_efectivo`` tanto a ``alpha_inter`` como a ``p_inter``
            # del helper: el helper interpreta ``alpha_inter`` como la prob.
            # con violación y ``p_inter`` como la prob. sin violación; como
            # ``viol_media`` ya determina el régimen aquí, ambos parámetros
            # apuntan a ``p_efectivo`` para mantener la decisión coherente.
            grupo, _ = seleccionar_grupo_operadores_inter_intra(
                rng,
                viol_media,
                ops_intra_list,
                ops_inter_list,
                ops_fallback_list,
                alpha_inter=p_efectivo,
                p_inter=p_efectivo,
            )
            # Generamos UN vecino (operador uniforme dentro del grupo).
            vec_ids, mov = generar_vecino_ids(
                fuentes_ids[i],
                rng=rng,
                operadores=grupo,
                pesos_operadores=None,
                usar_gpu=ctx.usar_gpu,
                encoding=encoding,
            )
            contador.proponer(mov.operador)
            # Evaluamos: costo puro + violación, y objetivo combinado.
            cp = float(costo_rapido_ids(vec_ids, ctx))
            vp = float(exceso_capacidad_sol_ids(vec_ids, ctx))
            obj_nei = float(
                objectivo_penalizado(cp, vp, usar_penal=usar_penalizacion_capacidad, lam=lam_eff)
            )
            obj_old = _obj_idx(i)
            # Criterio greedy: SOLO se reemplaza la fuente si el vecino mejora
            # estrictamente el objetivo. El epsilon 1e-15 evita aceptaciones por
            # ruido numérico cuando los costos son matemáticamente idénticos.
            if obj_nei < obj_old - 1e-15:
                fuentes_ids[i] = vec_ids
                fuentes_pure[i] = cp
                fuentes_viol[i] = vp
                trials[i] = 0
                contador.aceptar(mov.operador)
                if vp > 1e-12:
                    aceptaciones_infactible += 1
                # CORRECCIÓN del bug presente en abejas.py: registramos la
                # mejora del mejor global EN EL MISMO IF que detecta el vecino
                # mejor. Esto asegura que el operador imputado a la mejora es
                # exactamente el que la produjo (no un movimiento posterior).
                if obj_nei < mejor_costo - 1e-15:
                    mejor_costo = obj_nei
                    mejor_costo_puro = cp
                    mejor_viol = vp
                    mejor_sol_ids = [list(r) for r in vec_ids]
                    contador.registrar_mejora(mov.operador)
                    mejoras += 1
            else:
                trials[i] += 1

        # =====================================================
        # FASE 2 — OBSERVADORAS (ruleta sobre fitness inverso)
        # =====================================================
        # Recalculamos la violación media tras la fase de empleadas: la nueva
        # composición de fuentes puede haber cambiado y queremos que el sesgo
        # de las observadoras refleje el estado actual.
        viol_media = sum(fuentes_viol) / num_fuentes_eff

        # Recalculamos también ``p_efectivo`` ANTES de la fase observadoras
        # con la violación media actualizada. Esto cierra el ciclo dinámico:
        # cualquier cambio en la composición de fuentes que haya alterado la
        # violación media se refleja inmediatamente en el sesgo de operadores.
        p_efectivo = p_inter_max_efectivo if viol_media > 0 else float(p_inter)

        # Construimos las probabilidades de la ruleta. ``max(obj, 0.0)`` evita
        # divisiones extrañas si por alguna razón el objetivo es negativo
        # (caso teórico inexistente en CARP pero defensivo).
        probs: list[float] = []
        total_inv = 0.0
        for i in range(num_fuentes_eff):
            obj_i = _obj_idx(i)
            inv = 1.0 / (1.0 + max(obj_i, 0.0))
            probs.append(inv)
            total_inv += inv
        if total_inv > 0.0:
            probs = [p / total_inv for p in probs]
        else:
            # Fallback degenerado: si todos los inversos son 0 (no debería
            # pasar nunca con costos reales), repartimos uniformemente.
            probs = [1.0 / num_fuentes_eff] * num_fuentes_eff

        # Muestreamos ``num_fuentes_eff`` fuentes CON REEMPLAZO ponderado.
        # Permitir repetición es clave para que las fuentes buenas reciban
        # más visitas (es la "danza del meneo" del modelo biológico).
        idxs = rng.choices(range(num_fuentes_eff), weights=probs, k=num_fuentes_eff)
        srcs = [fuentes_ids[i] for i in idxs]

        # Generamos el LOTE de vecinos en una sola llamada.
        # Pasamos ``p_efectivo`` a ambos parámetros del helper (ver
        # justificación en la fase EMPLEADAS).
        vecinos_lote, movs_lote = _generar_vecinos_lote_simple(
            srcs,
            rng=rng,
            ops_intra=ops_intra_list,
            ops_inter=ops_inter_list,
            ops_fallback=ops_fallback_list,
            violacion_media=viol_media,
            alpha_inter=p_efectivo,
            p_inter=p_efectivo,
            ctx=ctx,
        )
        # Evaluamos el lote completo en una pasada vectorizada (GPU si disponible).
        objs_np, bases_np, viols_np = costo_lote_penalizado_ids(
            vecinos_lote,
            ctx,
            lam_eff,
            usar_penal=usar_penalizacion_capacidad,
        )
        # Procesamos cada vecino contra su fuente original (no la observadora).
        for k in range(num_fuentes_eff):
            i_fuente = idxs[k]
            mov_k = movs_lote[k]
            contador.proponer(mov_k.operador)
            obj_nei = float(objs_np[k])
            obj_old = _obj_idx(i_fuente)
            if obj_nei < obj_old - 1e-15:
                fuentes_ids[i_fuente] = vecinos_lote[k]
                fuentes_pure[i_fuente] = float(bases_np[k])
                fuentes_viol[i_fuente] = float(viols_np[k])
                trials[i_fuente] = 0
                contador.aceptar(mov_k.operador)
                if float(viols_np[k]) > 1e-12:
                    aceptaciones_infactible += 1
                # Detección de mejora global EN EL MISMO if (igual que en empleadas).
                if obj_nei < mejor_costo - 1e-15:
                    mejor_costo = obj_nei
                    mejor_costo_puro = float(bases_np[k])
                    mejor_viol = float(viols_np[k])
                    mejor_sol_ids = [list(r) for r in vecinos_lote[k]]
                    contador.registrar_mejora(mov_k.operador)
                    mejoras += 1
            else:
                trials[i_fuente] += 1

        # =====================================================
        # FASE 3 — SCOUTS canónicos (soluciones ALEATORIAS PURAS)
        # =====================================================
        # En la versión canónica Karaboga 2005, una fuente agotada se sustituye
        # por una solución aleatoria del espacio, NO por un vecino de la mejor
        # actual. Esa diferencia con ``busqueda_abejas`` es deliberada y es lo
        # que da al ABC simple su diversificación "pura": el algoritmo puede
        # aterrizar en regiones del espacio nunca exploradas a partir de ahí.
        a_reiniciar = [i for i in range(num_fuentes_eff) if trials[i] >= limite_abandono_eff]
        if a_reiniciar:
            for i in a_reiniciar:
                nueva_labels = _generar_solucion_aleatoria(ctx, rng, md_op)
                nueva_ids = encode_solution(nueva_labels, encoding)
                fuentes_ids[i] = nueva_ids
                fuentes_pure[i] = float(costo_rapido_ids(nueva_ids, ctx))
                fuentes_viol[i] = float(exceso_capacidad_sol_ids(nueva_ids, ctx))
                trials[i] = 0
                scouts += 1
                # IMPORTANTE: los scouts NO llaman ``contador.proponer`` ni
                # ``contador.aceptar``: no se aplica ningún operador de
                # vecindad, se genera una solución desde cero. Mezclar los
                # scouts con los operadores corrompería las estadísticas.
                # Si el scout casualmente mejoró el global, lo detectamos y
                # lo registramos como mejora "sin operador" (no incrementa
                # ningún contador por operador, solo ``mejoras``).
                obj_scout = float(
                    objectivo_penalizado(
                        fuentes_pure[i],
                        fuentes_viol[i],
                        usar_penal=usar_penalizacion_capacidad,
                        lam=lam_eff,
                    )
                )
                if obj_scout < mejor_costo - 1e-15:
                    mejor_costo = obj_scout
                    mejor_costo_puro = float(fuentes_pure[i])
                    mejor_viol = float(fuentes_viol[i])
                    mejor_sol_ids = [list(r) for r in fuentes_ids[i]]
                    mejoras += 1
                    # No llamamos ``registrar_mejora`` porque no hay operador
                    # de vecindad asociado (sería None). Aceptamos que en este
                    # caso ``sum(operadores_mejoraron.values()) <= mejoras``,
                    # con la diferencia exacta atribuida a los scouts.

        # ----- Cierre del ciclo: estancamiento e historial -----
        if mejor_costo < mejor_inicio_ciclo - 1e-15:
            iter_sin_mejora = 0
        else:
            iter_sin_mejora += 1

            # --- Kick por estancamiento global (strict_intra_inter_20260524) ---
            # En ABC el kick es POBLACIONAL: aplicamos la rafaga inter-ruta a
            # TODAS las fuentes (no solo a una). Esto provoca una diversificacion
            # masiva del enjambre cuando lleva muchos ciclos sin progreso. Tambien
            # reseteamos los contadores de abandono (``trials``) para que las
            # fuentes recien perturbadas no sean inmediatamente reemplazadas por
            # scouts antes de poder explotar su nueva posicion.
            if (max_iter_sin_mejora_kick is not None
                    and iter_sin_mejora >= max_iter_sin_mejora_kick):
                # Import diferido del kick clasico: solo se carga si la corrida
                # activa el kick Y no se provee un intensificador externo.
                if intensificador is None:
                    from metacarp.strict_intra_inter_20260524 import aplicar_kick_ids
                for i in range(num_fuentes_eff):
                    if intensificador is not None:
                        # Respuesta de intensificacion (p.ej. Path Relinking limpio)
                        # hacia la mejor solucion global, en lugar del kick aleatorio.
                        fuentes_ids[i] = intensificador(
                            fuentes_ids[i], mejor_sol_ids, ctx, lam_eff, rng, encoding, None
                        )
                    else:
                        fuentes_ids[i] = aplicar_kick_ids(fuentes_ids[i], rng, encoding=encoding)
                    # Recalculamos costo y violacion para ambas ramas (comun).
                    fuentes_pure[i] = float(costo_rapido_ids(fuentes_ids[i], ctx))
                    fuentes_viol[i] = float(exceso_capacidad_sol_ids(fuentes_ids[i], ctx))
                    # Reseteamos el contador de abandono de esta fuente: la
                    # acabamos de perturbar, no tiene sentido contarla como
                    # "agotada" todavia.
                    trials[i] = 0
                iter_sin_mejora = 0
                n_resets_kick += 1
                if max_resets is not None and n_resets_kick >= max_resets:
                    # Restamos 1 a ciclo_final por la misma razon que el break
                    # del criterio max_iter_sin_mejora: este ciclo SI ejecuto
                    # todas sus fases pero el reporte queda mas coherente
                    # ajustando el contador antes de romper.
                    # NOTA: ciclo_final ya esta sincronizado con la iteracion
                    # actual (ciclo+1), asi que NO lo restamos: queremos contar
                    # este ciclo como ejecutado.
                    break

        if guardar_historial:
            historial_best.append(mejor_costo)

    # ============================================================
    # === FIN DEL BUCLE PRINCIPAL ================================
    # ============================================================
    elapsed = time.perf_counter() - t0

    # Decodificamos la mejor solución a labels para reportarla legible.
    # Reconstruimos las rutas con el marcador de depósito al inicio y al final.
    mejor_solucion_labels: list[list[str]] = []
    for ruta_ids in mejor_sol_ids:
        ruta_labels = [md_op] + [encoding.id_to_label[i] for i in ruta_ids] + [md_op]
        mejor_solucion_labels.append(ruta_labels)

    # Métricas de gap respecto al inicial (positivo si hubo mejora).
    _gap_descartado, mejora_abs, mejora_pct = calcular_metricas_gap(costo_ref, mejor_costo_puro)

    # Determinación de factibilidad del MEJOR final.
    es_factible_final = mejor_viol < 1e-12

    # ----------------------------------------------------------
    # 9) Guardado en CSV (opcional).
    #    El bloque genera el reporte detallado y construye la fila con TODAS
    #    las columnas que pide el plan. NO incluye id_corrida ni config_id
    #    (decisión del proyecto, ver MEMORY.md).
    # ----------------------------------------------------------
    archivo_csv: str | None = None
    if guardar_csv:
        ruta = ruta_csv or f"resultados_busqueda_abejas_simple_{nombre_instancia}.csv"
        detalle_txt, costo_total_reporte = generar_reporte_detallado(
            mejor_solucion_labels,
            data,
            G,
            nombre_instancia=nombre_instancia,
            marcador_depot_etiqueta=marcador_depot_etiqueta,
            usar_gpu=False,
        )
        _bks = resumen_bks_csv(data, mejor_costo_puro)
        fraccion_viol = (
            iteraciones_con_violacion / ciclo_final if ciclo_final > 0 else 0.0
        )
        fila = {
            "metaheuristica": "busqueda_abejas_simple",
            "instancia": nombre_instancia,
            "bks_referencia": _bks["bks_referencia"],
            "bks_origen": _bks["bks_origen"],
            "gap_bks_porcentaje": _bks["gap_bks_porcentaje"],
            "repeticion": repeticion if repeticion is not None else "",
            "semilla": semilla,
            "tiempo_segundos": elapsed,
            # Reportamos el costo PURO como ``mejor_costo`` (la penalización
            # solo es un mecanismo de guía interno, no debe ensuciar la
            # comparación con BKS).
            "mejor_costo": mejor_costo_puro,
            "costo_solucion_inicial": costo_ref,
            "mejora_absoluta": mejora_abs,
            "mejora_porcentaje": mejora_pct,
            # Las columnas "iteraciones", "num_fuentes", "limite_abandono" y
            # "max_iter_sin_mejora" reportan los VALORES BASE que pasó el
            # usuario por la firma (vacío "" si los dejó en None). Los
            # VALORES EFECTIVOS que realmente usó el algoritmo van debajo en
            # columnas "*_efectivo" (igual estilo que RTS).
            "iteraciones": iteraciones_base_usuario if iteraciones_base_usuario is not None else "",
            "num_fuentes": num_fuentes_base_usuario if num_fuentes_base_usuario is not None else "",
            "limite_abandono": limite_abandono_base_usuario if limite_abandono_base_usuario is not None else "",
            "max_iter_sin_mejora": (
                max_iter_sin_mejora_base_usuario if max_iter_sin_mejora_base_usuario is not None else ""
            ),
            # Factores de escala (si se pasaron). Cadena vacía cuando no se usaron.
            "factor_fuentes": factor_fuentes if factor_fuentes is not None else "",
            "factor_abandono": factor_abandono if factor_abandono is not None else "",
            "factor_iter": factor_iter if factor_iter is not None else "",
            # p_inter BASE (el que pasó el usuario; el algoritmo eleva el piso
            # a 0.8 cuando hay violación, ver columna ``p_inter_max_efectivo``).
            "p_inter": p_inter,
            # Valor REAL que se aplicó como P(inter) en presencia de violación
            # de capacidad: piso 0.8 garantizado por construcción.
            "p_inter_max_efectivo": p_inter_max_efectivo,
            "usar_penalizacion_capacidad": usar_penalizacion_capacidad,
            "lambda_capacidad": lam_eff,
            # ---- VALORES EFECTIVOS REALES (lo que de verdad usó el bucle) ----
            # Si el usuario pasó valor absoluto, "iteraciones_efectivas" =
            # ese valor. Si pasó factor, viene del factor. Si no pasó nada,
            # viene de la fórmula default. Reportarlos es indispensable para
            # que un análisis posterior pueda comparar corridas configuradas
            # con factores distintos sobre instancias de distinto tamaño.
            "iteraciones_efectivas": iteraciones_eff,
            "num_fuentes_efectivo": num_fuentes_eff,
            "limite_abandono_efectivo": limite_abandono_eff,
            "max_iter_sin_mejora_efectivo": max_sin_mejora_eff,
            "n_tareas": n_tareas,
            # 36 columnas (4 categorías × 9 operadores).
            **contador.resumen_csv(),
            "iteraciones_totales": ciclo_final,
            "iteraciones_sin_mejora_final": iter_sin_mejora,
            "scouts_reinicios": scouts,
            "mejoras": mejoras,
            "aceptaciones_solucion_infactible": aceptaciones_infactible,
            "iteraciones_con_violacion": iteraciones_con_violacion,
            "fraccion_iter_con_violacion": fraccion_viol,
            "mejor_solucion_factible_final": es_factible_final,
            "mejor_solucion_tr_legible": solucion_legible_humana(mejor_solucion_labels),
            "reporte_detalle_deadheading": detalle_txt,
            "costo_total_desde_reporte": costo_total_reporte,
            # Columna del mecanismo de kick (strict_intra_inter_20260524).
            # 0 cuando la variante experimental no esta activa (default).
            "n_resets_kick": n_resets_kick,
        }
        archivo_csv = guardar_resultado_csv(fila=fila, ruta_csv=ruta)

    # ----------------------------------------------------------
    # 10) Construcción del resultado inmutable.
    # ----------------------------------------------------------
    fraccion_iter_con_viol = (
        iteraciones_con_violacion / ciclo_final if ciclo_final > 0 else 0.0
    )
    return AbejasSimpleResult(
        mejor_solucion=mejor_solucion_labels,
        mejor_costo=mejor_costo_puro,
        solucion_inicial_referencia=copiar_solucion_labels(sol_ref_labels),
        costo_solucion_inicial=float(costo_ref),
        mejora_absoluta=mejora_abs,
        mejora_porcentaje_inicial_vs_final=mejora_pct,
        tiempo_segundos=elapsed,
        iteraciones_totales=ciclo_final,
        iteraciones_sin_mejora_final=iter_sin_mejora,
        # Reportamos el número EFECTIVO de fuentes (igual a num_fuentes si el
        # usuario lo pasó, o al valor calculado por factor/fórmula instance-aware).
        fuentes_alimento=num_fuentes_eff,
        scouts_reinicios=scouts,
        mejoras=mejoras,
        semilla=semilla,
        backend_evaluacion=ctx.backend_real,
        usar_penalizacion_capacidad=usar_penalizacion_capacidad,
        lambda_capacidad=lam_eff,
        mejor_solucion_factible_final=es_factible_final,
        aceptaciones_solucion_infactible=aceptaciones_infactible,
        iteraciones_con_violacion=iteraciones_con_violacion,
        fraccion_iter_con_violacion=fraccion_iter_con_viol,
        operadores_propuestos=contador.como_dict_ordenado(contador.propuestos),
        operadores_aceptados=contador.como_dict_ordenado(contador.aceptados),
        operadores_mejoraron=contador.como_dict_ordenado(contador.mejoraron),
        operadores_trayectoria_mejor=contador.como_dict_ordenado(contador.trayectoria_mejor),
        historial_mejor_costo=historial_best,
        archivo_csv=archivo_csv,
        # Métricas instance-aware: reproducibilidad exacta desde el resultado.
        n_tareas=n_tareas,
        iteraciones_efectivas=iteraciones_eff,
        num_fuentes_efectivo=num_fuentes_eff,
        limite_abandono_efectivo=limite_abandono_eff,
        max_iter_sin_mejora_efectivo=max_sin_mejora_eff,
        p_inter_max_efectivo=p_inter_max_efectivo,
        # Kicks aplicados (variante experimental strict_intra_inter_20260524).
        n_resets_kick=n_resets_kick,
    )


# ---------------------------------------------------------------------------
# Wrapper de conveniencia: busqueda_abejas_simple_desde_instancia
# ---------------------------------------------------------------------------
def busqueda_abejas_simple_desde_instancia(
    nombre_instancia: str,
    *,
    root: str | None = None,
    # Parámetros instance-aware: None ⇒ calibración automática en función
    # del número de tareas requeridas de la instancia (n_tareas). Si el
    # usuario pasa un valor concreto, ese valor manda sobre la fórmula.
    iteraciones: int | None = None,           # None → max(200, 20·n)
    num_fuentes: int | None = None,           # None → max(10, round(2·√n))
    limite_abandono: int | None = None,       # None → max(15, n//2)
    max_iter_sin_mejora: int | None = None,   # None → max(50, 3·n)
    # Factores de escala alternativos (solo se aplican si el respectivo
    # parámetro absoluto es None). Pensados para el script de experimentos.
    factor_fuentes: float | None = None,
    factor_abandono: float | None = None,
    factor_iter: int | float | None = None,
    semilla: int | None = None,
    operadores: Iterable[str] = OPERADORES_POPULARES,
    marcador_depot_etiqueta: str | None = None,
    usar_gpu: bool = False,
    guardar_historial: bool = True,
    guardar_csv: bool = False,
    ruta_csv: str | None = None,
    repeticion: int | None = None,
    usar_penalizacion_capacidad: bool = True,
    lambda_capacidad: float | None = None,
    # ``alpha_inter`` fue ELIMINADO: el algoritmo eleva automáticamente
    # P(inter) a max(p_inter, 0.8) cuando hay violación de capacidad.
    p_inter: float = 0.6,
    # Kick reactivo (variante experimental strict_intra_inter_20260524).
    max_iter_sin_mejora_kick: int | None = None,
    max_resets: int | None = None,
    # Hook de intensificacion opcional (p.ej. Path Relinking limpio).
    intensificador: Callable | None = None,
    extra_csv: dict[str, object] | None = None,
    **_ignorado_kwargs: object,
) -> AbejasSimpleResult:
    """
    Función de conveniencia que carga los recursos desde el NOMBRE de la
    instancia y ejecuta ``busqueda_abejas_simple`` con todos los parámetros.

    Equivale a invocar manualmente:
        data = load_instances(nombre_instancia)
        G = cargar_objeto_gexf(nombre_instancia)
        inicial_obj = cargar_solucion_inicial(nombre_instancia)
        busqueda_abejas_simple(inicial_obj, data, G, ...)
    """
    data = load_instances(nombre_instancia, root=root)
    G = cargar_objeto_gexf(nombre_instancia, root=root)
    inicial_obj = cargar_solucion_inicial(nombre_instancia, root=root)
    return busqueda_abejas_simple(
        inicial_obj,
        data,
        G,
        iteraciones=iteraciones,
        num_fuentes=num_fuentes,
        limite_abandono=limite_abandono,
        max_iter_sin_mejora=max_iter_sin_mejora,
        # Propagamos los factores de escala al algoritmo principal.
        factor_fuentes=factor_fuentes,
        factor_abandono=factor_abandono,
        factor_iter=factor_iter,
        semilla=semilla,
        operadores=operadores,
        marcador_depot_etiqueta=marcador_depot_etiqueta,
        usar_gpu=usar_gpu,
        guardar_historial=guardar_historial,
        guardar_csv=guardar_csv,
        ruta_csv=ruta_csv,
        nombre_instancia=nombre_instancia,
        repeticion=repeticion,
        root=root,
        usar_penalizacion_capacidad=usar_penalizacion_capacidad,
        lambda_capacidad=lambda_capacidad,
        # alpha_inter eliminado de la firma: el piso 0.8 bajo violación se
        # garantiza internamente a partir de p_inter.
        p_inter=p_inter,
        # Propagamos el mecanismo de kick (variante experimental).
        max_iter_sin_mejora_kick=max_iter_sin_mejora_kick,
        max_resets=max_resets,
        # Propagamos el hook de intensificacion (p.ej. Path Relinking limpio).
        intensificador=intensificador,
        extra_csv=extra_csv,
    )
