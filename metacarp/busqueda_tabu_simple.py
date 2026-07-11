"""
Búsqueda Tabú clásica en su versión MÁS SIMPLE para CARP.

Concepto algorítmico
--------------------
La Búsqueda Tabú (Tabu Search, TS), introducida por Fred Glover (1986), es una
metaheurística basada en búsqueda local que pretende escapar de mínimos locales
mediante una "memoria de corto plazo" llamada **lista tabú**:

- En cada iteración, el algoritmo se mueve al **mejor vecino disponible**
  (best-improvement) aunque ese vecino sea peor que la solución actual. Esto
  permite "subir colinas" para alcanzar otras zonas del espacio de búsqueda.
- Para evitar caer en ciclos (es decir, oscilar entre dos o tres soluciones
  cercanas), se prohíbe temporalmente revisitar ciertos movimientos: se
  guardan en la **lista tabú** durante un número fijo de iteraciones
  (``tabu_tenure``).

Versión MÁS SIMPLE implementada aquí
------------------------------------
Esta implementación es deliberadamente minimalista, pensada como referencia
didáctica (versión "de pizarra" del algoritmo):

1. **Memoria**: una sola lista tabú FIFO (cola circular). Se inserta el
   movimiento ejecutado y, si excede ``tabu_tenure``, se descarta el más
   antiguo. Sin memoria por frecuencia, sin memoria a largo plazo.

2. **Vecindario**: en cada iteración se genera un lote de ``tam_vecindario``
   vecinos aleatorios usando los operadores estándar del proyecto
   (``OPERADORES_POPULARES``: 9 movimientos intra e inter-ruta). Aunque el TS
   "puro" enumera todo el vecindario, aquí muestreamos porque enumerar
   exhaustivamente todos los operadores sobre todas las posiciones es
   prohibitivo en instancias grandes; esta simplificación es estándar en
   la literatura aplicada a CARP.

3. **Selección**: se elige el **MEJOR vecino no-tabú** del lote
   (best-improvement). Es decir, NO se acepta el primero que mejore
   (first-improvement), sino que se evalúan todos los del lote y se queda
   con el de menor costo entre los admisibles.

4. **Criterio de aspiración clásico**: si un movimiento está en la lista
   tabú pero produce una solución **estrictamente mejor que el mejor global
   conocido**, se "aspira" a él (se acepta de todas formas). Esto evita que
   la lista tabú nos cierre la puerta al óptimo global.

5. **Criterio de parada doble** (cualquiera de los dos detiene la búsqueda):

   - ``iteraciones_max``: número máximo de iteraciones totales.
   - ``max_iter_sin_mejora``: número máximo de iteraciones consecutivas sin
     mejorar el mejor global (estancamiento).

Lo que NO incluye esta versión simple (extensiones posibles)
------------------------------------------------------------
- Movimientos compuestos (chain moves, ejection chains).
- Lista tabú por atributos (en lugar de por movimiento completo).
- Memoria de frecuencias para diversificación.
- Lista de soluciones élite e intensificación por reinicio.
- Path relinking.
- Penalización dinámica de capacidad (aquí se usa solo costo puro como
  criterio de selección, con violación de capacidad apenas reportada).
- Sesgo inter/intra-ruta por violación de capacidad (como sí hace SA).

Estas extensiones se discuten al final del módulo en el docstring de
``busqueda_tabu_simple`` bajo el rótulo "Posibles extensiones".

Optimización de evaluación
--------------------------
- Construye un :class:`ContextoEvaluacion` una sola vez (matriz Dijkstra densa
  precomputada): cada vecino se evalúa con :func:`costo_rapido` (10×–50× más
  rápido que el evaluador clásico basado en NetworkX).
- El lote de vecinos por iteración se evalúa con :func:`costo_lote_ids` (una
  sola pasada vectorizada de NumPy sobre el lote).
"""
# Permite usar anotaciones de tipo como `float | None` en Python < 3.10.
from __future__ import annotations

# collections.deque es una cola doblemente enlazada con maxlen: ideal para una
# lista tabú FIFO de longitud fija (al insertar más allá de maxlen, el elemento
# más antiguo se elimina automáticamente; complejidad O(1) en ambos extremos).
# Counter es un diccionario especializado para conteo de ocurrencias: lo usamos
# para llevar un conteo O(1) de cuántas veces aparece cada clave en la deque
# tabú (necesario para mantener el ``set_tabu`` sincronizado al descartar el
# elemento más antiguo cuando puede haber duplicados).
from collections import Counter, deque
# Módulo estándar para generar números aleatorios de forma reproducible.
import random
# Módulo para medir el tiempo transcurrido con alta precisión.
import time
# Tipos abstractos para anotaciones de función.
from collections.abc import Iterable, Mapping
# @dataclass y field: para definir clases de datos sin escribir __init__ manualmente.
from dataclasses import dataclass, field
# Any = cualquier tipo; Callable = tipo para funciones/callables; Literal = un conjunto fijo de valores permitidos.
from typing import Any, Callable, Literal

# NetworkX: biblioteca para manipular grafos (nodos y aristas).
import networkx as nx

# Importaciones internas del paquete metacarp:
from .busqueda_indices import build_search_encoding, encode_solution  # codificación de soluciones
from .cargar_grafos import cargar_objeto_gexf                         # carga el grafo desde GEXF
from .cargar_soluciones_iniciales import cargar_solucion_inicial      # carga la solución inicial
from .evaluador_costo import (
    costo_lote_ids,                      # evalúa un lote de soluciones (NumPy vectorizado)
    costo_lote_penalizado_ids,           # ídem con objetivo penalizado: (obj, costo_puro, viol)
    costo_rapido,                        # evalúa UNA solución (label-based) — usado tras kick
    exceso_capacidad_rapido,             # calcula la violación de capacidad de una solución
    lambda_penal_capacidad_por_defecto,  # λ por defecto basado en la instancia
)
from .instances import load_instances  # carga los datos de la instancia CARP
from .metaheuristicas_utils import (
    ContadorOperadores,                  # cuenta propuestas/aceptaciones por operador
    calcular_metricas_gap,               # calcula el gap porcentual de mejora
    construir_contexto_para_corrida,     # construye el contexto de evaluación rápida
    copiar_solucion_labels,              # copia una solución a formato de strings
    generar_reporte_detallado,           # genera el texto de reporte final
    guardar_resultado_csv,               # persiste la fila de resultados en CSV
    resumen_bks_csv,                     # extrae columnas de comparación con BKS
    seleccionar_grupo_operadores_inter_intra,  # helper compartido con SA y RTS para sesgo inter/intra
    seleccionar_mejor_inicial_rapido,    # elige la mejor solución inicial
    solucion_legible_humana,             # convierte la solución a texto legible
)
from .vecindarios import (
    MovimientoVecindario,
    OPERADORES_INTRA,                    # subset intra-ruta (clasificación canónica del proyecto)
    OPERADORES_INTER,                    # subset inter-ruta (idem)
    OPERADORES_POPULARES,
    generar_vecino,
)

# Declara la API pública de este módulo.
__all__ = [
    "BusquedaTabuSimpleResult",
    "busqueda_tabu_simple",
    "busqueda_tabu_simple_desde_instancia",
]


# --- CONCEPTO OOP: @dataclass(frozen=True, slots=True) ---
# frozen=True: el objeto es inmutable una vez creado. Seguro porque el resultado
#              de la búsqueda no debe cambiar después de terminar la ejecución.
# slots=True:  Python reserva exactamente los atributos declarados, reduciendo el
#              uso de memoria por instancia.
@dataclass(frozen=True, slots=True)
class BusquedaTabuSimpleResult:
    """
    Resultado completo de la Búsqueda Tabú en su versión más simple.

    Agrupa la mejor solución encontrada, métricas de calidad, historial de
    costos, estadísticas de aceptaciones y conteo de operadores. Sigue el mismo
    patrón inmutable que los resultados de SA, abejas y cuckoo del proyecto.
    """

    # La mejor solución CARP encontrada durante toda la búsqueda.
    mejor_solucion: list[list[str]]
    # Costo de la mejor solución (objetivo a minimizar).
    mejor_costo: float
    # Solución inicial de referencia (para comparar la mejora final).
    solucion_inicial_referencia: list[list[str]]
    # Costo de la solución inicial.
    costo_solucion_inicial: float
    # Diferencia absoluta: costo_inicial - mejor_costo (positivo = hubo mejora).
    mejora_absoluta: float
    # Porcentaje de mejora respecto al costo inicial.
    mejora_porcentaje_inicial_vs_final: float
    # Tiempo total de ejecución en segundos.
    tiempo_segundos: float
    # Número total de iteraciones ejecutadas (≤ iteraciones_max).
    iteraciones_totales: int
    # Total de vecinos evaluados durante toda la corrida (= iteraciones * tam_vecindario).
    vecinos_evaluados: int
    # Iteración en la que se descubrió la mejor solución reportada.
    iteracion_mejor: int
    # Número de iteraciones consecutivas SIN mejora al terminar (criterio de parada temprana).
    iteraciones_sin_mejora_final: int
    # Veces que el criterio de aspiración rescató un movimiento tabú.
    aspiraciones: int
    # Veces que todos los vecinos del lote eran tabú y hubo que elegir el mejor tabú.
    iteraciones_todos_tabu: int
    # Número de veces que el mejor global mejoró durante la búsqueda.
    mejoras: int
    # Semilla del generador aleatorio (para reproducibilidad).
    semilla: int | None
    # Dispositivo de evaluación: 'cpu' o 'gpu'.
    backend_evaluacion: str = "cpu"
    # Historial del mejor costo registrado al inicio de cada iteración.
    historial_mejor_costo: list[float] = field(default_factory=list)
    # El último movimiento aceptado al final de la búsqueda.
    ultimo_movimiento_aceptado: MovimientoVecindario | None = None
    # Estadísticas de operadores de vecindario (4 categorías).
    operadores_propuestos: dict[str, int] = field(default_factory=dict)
    operadores_aceptados: dict[str, int] = field(default_factory=dict)
    operadores_mejoraron: dict[str, int] = field(default_factory=dict)
    operadores_trayectoria_mejor: dict[str, int] = field(default_factory=dict)
    # True si la mejor solución final respeta todas las restricciones de capacidad.
    mejor_solucion_factible_final: bool = True
    # Ruta del archivo CSV donde se guardaron los resultados, o None si no se guardó.
    archivo_csv: str | None = None
    # === Campos del mecanismo de sesgo inter/intra (compatibilidad con SA y RTS) ===
    # Estos campos reportan EXACTAMENTE qué valores del sesgo se aplicaron en la
    # corrida y cuántas iteraciones del bucle principal entraron al modo sesgado
    # (es decir, encontraron la solución actual violando capacidad y, por tanto,
    # usaron alpha_inter como probabilidad de elegir el grupo inter-ruta). Si la
    # corrida nunca tuvo violación, ``iteraciones_con_violacion`` valdrá 0 y todo
    # el sesgo aplicado correspondió a p_inter (estado factible).
    alpha_inter_aplicado: float = 0.0
    p_inter_aplicado: float = 0.0
    iteraciones_con_violacion: int = 0
    # Número de kicks (perturbaciones inter-ruta) aplicados durante la corrida.
    # Solo es > 0 cuando se pasa max_iter_sin_mejora_kick al wrapper de la
    # variante experimental strict_intra_inter_20260524.
    n_resets_kick: int = 0


def _clave_tabu(mov: MovimientoVecindario) -> tuple[Any, ...]:
    """
    Genera una clave hashable que identifica unívocamente un movimiento.

    La clave se usa para comparar movimientos contra la lista tabú: dos
    movimientos con la MISMA clave se consideran el mismo movimiento, aunque
    se produzcan en iteraciones distintas. Esta es la "memoria estructural"
    que evita revisitar el mismo cambio durante ``tabu_tenure`` iteraciones.

    Incluye:
    - El nombre del operador (relocate_intra, swap_inter, etc.).
    - Los índices de rutas afectadas (ruta_a, ruta_b).
    - Las posiciones internas (i, j, k, l).
    - Las etiquetas de las tareas movidas (tuple de strings).

    Decisión de diseño: en una versión más sofisticada se podría usar memoria
    POR ATRIBUTOS (p.ej., "prohibir mover la tarea TRk durante T iters", sin
    importar a qué posición). Esta versión simple usa el movimiento completo.
    """
    return (
        mov.operador,                  # nombre del operador (ej: 'swap_intra')
        mov.ruta_a, mov.ruta_b,        # índices de las rutas involucradas
        mov.i, mov.j, mov.k, mov.l,    # posiciones dentro de las rutas
        tuple(mov.labels_movidos),     # etiquetas de las tareas que se mueven
    )


def busqueda_tabu_simple(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    iteraciones_max: int = 400,        # límite duro de iteraciones totales (criterio 1 de parada)
    max_iter_sin_mejora: int = 100,    # iteraciones consecutivas sin mejorar best global (criterio 2 de parada)
    tam_vecindario: int = 25,          # tamaño del lote de vecinos por iteración
    tabu_tenure: int = 20,             # longitud fija de la lista tabú (FIFO)
    semilla: int | None = None,        # semilla para reproducibilidad
    operadores: Iterable[str] = OPERADORES_POPULARES,  # operadores de vecindario a usar
    marcador_depot_etiqueta: str | None = None,  # etiqueta del depósito (usa la del ctx si None)
    usar_gpu: bool = False,            # flag de GPU (placeholder; CPU es lo suficientemente rápido)
    backend_vecindario: Literal["labels", "ids"] = "labels",  # modo de generación de vecinos
    guardar_historial: bool = True,    # si True, guarda el mejor costo de cada iteración
    guardar_csv: bool = False,         # si True, escribe la fila de resultados en CSV
    ruta_csv: str | None = None,       # ruta del CSV (None = nombre automático)
    nombre_instancia: str = "instancia",  # nombre de la instancia para el CSV
    repeticion: int | None = None,     # número de repetición dentro de un experimento
    root: str | None = None,           # directorio raíz para localizar los datos
    extra_csv: dict[str, object] | None = None,  # columnas adicionales para el CSV (no usadas en versión simple)
    # --- Parámetros de sesgo inter/intra (compatibilidad EXACTA con el SA) ---
    # Defaults None = usar el mismo valor por defecto que SA (0.8 y 0.6). Se
    # mantiene la firma con None para que el script de experimentos pueda dejar
    # el parámetro en su valor canónico SA sin tener que conocer la constante.
    alpha_inter: float | None = None,  # P(elegir inter) cuando la sol. actual viola capacidad
    p_inter: float | None = None,      # P(elegir inter) cuando la sol. actual es factible
    metodo_seleccion: str = "canonico",  # método de combinación inter/intra: canonico|p_inter|binario|random
    # --- Kick reactivo (variante experimental strict_intra_inter_20260524) ---
    # Cuando ``iter_sin_mejora`` alcanza este umbral se aplica una perturbacion
    # INTER-RUTA disruptiva (ver ``metacarp.strict_intra_inter_20260524``) y se
    # reinicia el contador. None = mecanismo desactivado (comportamiento clasico).
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
    # Penalización por capacidad (lambda_grid_20260525). None = default instance-aware.
    lambda_capacidad: float | None = None,
    # --- Presupuesto de wall-clock (val_egl_20260710) ---
    # Corta la corrida cuando el tiempo transcurrido desde ``t0`` alcanza este
    # límite (en segundos). Se comprueba UNA vez al inicio de cada iteración
    # del bucle principal (best-non-tabu), lo que ya está en la granularidad
    # correcta (una iteración = un lote de tam_vecindario evaluaciones).
    # None = sin límite (comportamiento clásico).
    tiempo_limite_segundos: float | None = None,
    **_ignorado_kwargs: object,        # absorbe kwargs heredados (p.ej. id_corrida, config_id)
) -> BusquedaTabuSimpleResult:
    """
    Búsqueda Tabú clásica en su versión MÁS SIMPLE para minimizar el costo de
    soluciones CARP.

    Estructura del algoritmo (best-non-tabu / best-improvement)
    -----------------------------------------------------------
    1. Inicializa la solución actual y la mejor global con la mejor solución
       inicial disponible (la calidad de la inicial NO contamina el progreso
       reportado: se calcula la mejora absoluta y porcentual al final).
    2. En cada iteración:

       a. Genera un lote de ``tam_vecindario`` vecinos aleatorios usando los
          operadores estándar (uniformemente al azar, sin pesos ni sesgos).
       b. Evalúa todo el lote en una sola pasada vectorizada (``costo_lote_ids``).
       c. Selecciona el **mejor vecino no-tabú**. Si está tabú pero su costo
          es estrictamente mejor que el mejor global conocido, lo acepta por
          **aspiración clásica**.
       d. Si TODOS los vecinos del lote son tabú y ninguno cumple aspiración,
          como fallback elige el mejor tabú (forzamos avance). Esta situación
          se registra en ``iteraciones_todos_tabu`` para diagnóstico.
       e. Inserta el movimiento elegido en la lista tabú FIFO. Si excede
          ``tabu_tenure``, el más antiguo se descarta automáticamente.
       f. Actualiza el mejor global si el costo bajó.

    3. Repite hasta cumplir cualquiera de los criterios de parada:

       - ``iteraciones_totales >= iteraciones_max``, o
       - ``iter_sin_mejora >= max_iter_sin_mejora`` (estancamiento).

    Parámetros clave
    ----------------
    iteraciones_max:
        Cota dura de iteraciones. Valor típico: 300–800. Por defecto 400, igual
        al de la implementación anterior ``busqueda_tabu`` para comparabilidad.

    max_iter_sin_mejora:
        Cota de estancamiento. Si tras esta cantidad de iteraciones consecutivas
        el mejor global no baja, se detiene la búsqueda. Valor por defecto 100
        (≈ 25% de iteraciones_max), un compromiso entre dar tiempo a "salir"
        de mínimos locales y no malgastar cómputo cuando ya no se progresa.

    tam_vecindario:
        Tamaño del lote de vecinos generados por iteración. Más vecinos =
        mejor decisión local (mejor "best") pero más cómputo por iteración.
        Por defecto 25, valor pequeño pero razonable para best-improvement.

    tabu_tenure:
        Longitud fija de la lista tabú FIFO. Regla de pulgar clásica:
        ``tenure ≈ sqrt(n)`` donde n es el tamaño del problema. Valores
        típicos: 10–30. Por defecto 20.

    Parámetros de sesgo inter/intra-ruta (compartidos con SA y RTS)
    --------------------------------------------------------------
    alpha_inter:
        Probabilidad de elegir el GRUPO de operadores inter-ruta cuando la
        solución ACTUAL del bucle principal viola capacidad. La detección de
        violación se hace por iteración (``viol_actual > 1e-12``). Si la
        ``solución actual`` cambia de estado factible a infactible (o
        viceversa) durante la corrida, el umbral se recalcula en la siguiente
        iteración. Valor por defecto = el mismo que usa SA (``0.8``).
    p_inter:
        Probabilidad de elegir el GRUPO de operadores inter-ruta cuando la
        solución ACTUAL es factible (no viola capacidad). Mantiene una dosis
        constante de exploración inter-ruta para escapar de mínimos locales
        intra-ruta incluso cuando ya no hay nada que reparar. Valor por
        defecto = el mismo que usa SA (``0.6``).

    Mecanismo (idéntico al SA del proyecto):
        - En cada iteración del bucle principal se computa ``viol_actual``
          (exceso de capacidad de la solución actual).
        - Se invoca ``seleccionar_grupo_operadores_inter_intra`` que realiza
          UN único ``rng.random()`` con umbral ``alpha_inter`` (si hay
          violación) o ``p_inter`` (si es factible).
        - El grupo elegido se pasa a ``generar_vecino`` mediante el argumento
          ``operadores=`` (con ``pesos_operadores=None``, selección uniforme
          dentro del grupo).

    Posibles extensiones (NO implementadas en esta versión simple)
    --------------------------------------------------------------
    - Movimientos compuestos (chain moves, ejection chains).
    - Lista tabú por atributos en lugar de por movimiento completo
      (p.ej. "prohibir reinsertar TRk en la ruta r durante T iters").
    - Tenure dinámico (reactive tabu search): aumentar tenure cuando se
      detectan ciclos, reducirlo cuando hay mejoras frecuentes.
    - Memoria de frecuencias para penalizar regiones muy visitadas
      (intensificación / diversificación).
    - Lista de soluciones élite y reinicio desde una elite al estancarse.
    - Path relinking entre soluciones de alta calidad.
    - Penalización dinámica de capacidad (en esta versión simple se usa solo
      costo puro como criterio de selección; las soluciones infactibles
      pueden aceptarse y se reporta su violación).

    Returns
    -------
    BusquedaTabuSimpleResult
        Objeto inmutable con la mejor solución, métricas de calidad, tiempos,
        estadísticas de operadores e historial de costos.
    """
    # Validaciones de parámetros: fallan rápido si los valores son inválidos.
    if iteraciones_max <= 0:
        raise ValueError("iteraciones_max debe ser > 0.")
    if max_iter_sin_mejora <= 0:
        raise ValueError("max_iter_sin_mejora debe ser > 0.")
    if tam_vecindario <= 0:
        raise ValueError("tam_vecindario debe ser > 0.")
    if tabu_tenure <= 0:
        raise ValueError("tabu_tenure debe ser > 0.")
    # Resolución de defaults del sesgo inter/intra: si el usuario pasa None,
    # usamos EXACTAMENTE los mismos valores que SA del proyecto (0.8 y 0.6).
    # Centralizar los defaults aquí (en lugar de en la firma) permite ajustar
    # los valores canónicos del proyecto cambiando un único lugar si fuera
    # necesario en el futuro, manteniendo la trazabilidad en el CSV.
    alpha_inter_eff: float = 0.8 if alpha_inter is None else float(alpha_inter)
    p_inter_eff: float = 0.6 if p_inter is None else float(p_inter)
    # Validaciones de los parámetros del sesgo: deben quedar en [0, 1] para
    # tener una interpretación probabilística válida. Aceptamos 0 (desactiva
    # ese modo) y 1 (siempre inter en ese estado) como extremos legítimos.
    if not (0.0 <= alpha_inter_eff <= 1.0):
        raise ValueError("alpha_inter debe estar en [0, 1].")
    if not (0.0 <= p_inter_eff <= 1.0):
        raise ValueError("p_inter debe estar en [0, 1].")

    # Generador aleatorio reproducible (mismo patrón que SA).
    rng = random.Random(semilla)
    # Marca de tiempo de inicio (perf_counter es más preciso que time.time).
    t0 = time.perf_counter()

    # Construcción del contexto de evaluación rápida (una sola vez por corrida).
    # Esto precomputa la matriz Dijkstra densa y los arrays por id de tarea,
    # amortizando el costo en todas las llamadas a costo_rapido / costo_lote_ids.
    ctx = construir_contexto_para_corrida(
        data,
        G,
        nombre_instancia=nombre_instancia if nombre_instancia != "instancia" else None,
        usar_gpu=usar_gpu,
        root=root,
    )

    # λ efectivo para penalizar violaciones de capacidad en el objetivo del bucle.
    # Si el usuario pasa lambda_capacidad=None, usamos el default instance-aware.
    lam_eff: float = (
        float(lambda_capacidad)
        if lambda_capacidad is not None
        else lambda_penal_capacidad_por_defecto(ctx)
    )

    # Selección de la mejor solución inicial entre todas las candidatas del pickle.
    # Esta versión simple NO usa penalización por capacidad para elegir la inicial
    # (queremos que TS sea lo más "puro" posible), pero seleccionar_mejor_inicial_rapido
    # acepta el flag igualmente; lo dejamos en True para no descartar candidatas
    # infactibles que podrían tener mejor costo puro.
    sel_ini = seleccionar_mejor_inicial_rapido(
        inicial_obj,
        ctx,
        usar_penalizacion_capacidad=True,
        lambda_capacidad=lambda_capacidad,  # None → default instance-aware
    )
    sol_ref = sel_ini.solucion          # solución de referencia para medir mejora final
    costo_ref = sel_ini.costo_puro      # costo inicial de referencia (sin penalización)

    # La solución "actual" es la que TS modifica en cada iteración.
    sol_actual = copiar_solucion_labels(sol_ref)
    costo_actual = costo_ref
    viol_actual = sel_ini.violacion_capacidad  # exceso de demanda (0 = factible)

    # Mejor global registrado. ``mejor_obj_pen`` guía las decisiones internas
    # (selección, aspiración, estancamiento). ``mejor_costo`` es el costo puro
    # que se reporta en CSV y dataclass.
    mejor_costo = float(costo_ref)
    mejor_obj_pen = float(costo_ref) + lam_eff * float(viol_actual)
    mejor_sol = copiar_solucion_labels(sol_ref)
    mejor_factible = viol_actual < 1e-12  # True si la inicial es factible

    # Configuración del encoding para el backend de ids (evaluación en lote
    # vectorizada). El encoding se construye una sola vez.
    encoding = ctx.encoding if backend_vecindario == "ids" else None
    if backend_vecindario == "ids" and encoding is None:
        encoding = build_search_encoding(data)

    # --- ESTRUCTURA DE DATOS: Lista Tabú FIFO ---
    # collections.deque(maxlen=tabu_tenure) implementa una cola circular de
    # tamaño fijo: al hacer .append() con la cola llena, el elemento más antiguo
    # (el de la izquierda) se elimina automáticamente. Esto es exactamente la
    # semántica clásica de "tabu list FIFO" de la versión original de Glover.
    #
    # Para acelerar la consulta "¿está este movimiento en la lista?", mantenemos
    # también un set sincronizado con la deque: O(1) en lookup vs O(n) si solo
    # usáramos la deque. Cada vez que insertamos o desechamos por overflow,
    # actualizamos ambos contenedores.
    lista_tabu: deque[tuple[Any, ...]] = deque(maxlen=tabu_tenure)
    set_tabu: set[tuple[Any, ...]] = set()
    # --- MICROOPTIMIZACIÓN: Counter paralelo para conteo O(1) de duplicados ---
    # Antes, para decidir si un elemento descartado de la deque debía salir del
    # set, se hacía ``list(lista_tabu).count(clave_descartada)``: un escaneo
    # O(tabu_tenure) en cada iteración del bucle principal. Con tenures
    # pequeños es despreciable, pero crece con n y se repite cientos de veces.
    # Llevamos ahora un ``conteo_tabu`` Counter sincronizado con la deque:
    #   - al insertar una clave, ``conteo_tabu[clave] += 1``
    #   - al descartar, ``conteo_tabu[clave] -= 1`` y, si llega a 0, lo
    #     borramos del Counter y del set (regla de mantenimiento).
    # Así la consulta "¿queda al menos otra copia de la clave descartada en la
    # deque?" se responde en O(1) consultando ``conteo_tabu[clave_descartada]``.
    conteo_tabu: Counter[tuple[Any, ...]] = Counter()

    # Contadores para el reporte final.
    vecinos_evaluados = 0       # total de vecinos evaluados durante toda la corrida
    mejoras = 0                 # cuántas veces el mejor global se actualizó
    aspiraciones = 0            # cuántas veces se rescató un tabú por aspiración
    iteraciones_todos_tabu = 0  # cuántas iteraciones tuvieron todos los vecinos tabú
    iter_sin_mejora = 0         # contador de iteraciones consecutivas sin mejorar best global
    # Contador de kicks (perturbaciones inter-ruta) aplicados durante la corrida.
    # Solo se incrementa si ``max_iter_sin_mejora_kick`` es != None y el umbral
    # se alcanza. Se reporta en el dataclass y en el CSV.
    n_resets_kick: int = 0
    iter_mejor = 0              # iteración en la que se descubrió la mejor solución
    ultimo_mov_aceptado: MovimientoVecindario | None = None
    historial_best: list[float] = []
    contador = ContadorOperadores()

    # Etiqueta del depósito para los operadores de vecindario.
    md_op = marcador_depot_etiqueta or ctx.marcador_depot

    # Convertimos los operadores a lista (los operadores se eligen con
    # generar_vecino, que internamente hace rng.choice si no se pasan pesos).
    ops_list = list(operadores)
    # Precomputamos las particiones intra/inter a partir de la lista de
    # operadores activos. Estas particiones se pasan al helper compartido con
    # SA y RTS en cada iteración, evitando reconstruirlas dentro del bucle.
    # Mantenemos también el fallback (lista completa) por si el usuario pasara
    # una configuración degenerada en la que uno de los dos grupos queda vacío.
    ops_intra_list = [op for op in ops_list if op in OPERADORES_INTRA]
    ops_inter_list = [op for op in ops_list if op in OPERADORES_INTER]
    ops_fallback_list = list(ops_list)
    # Contador de iteraciones del bucle principal que entraron al modo sesgado
    # por violación de capacidad. Una iteración cuenta como "con violación" si
    # la solución actual al INICIO de esa iteración tenía ``viol_actual > 1e-12``;
    # el conteo se hace cuando el helper devuelve True en su segundo retorno.
    iteraciones_con_violacion = 0

    # === BUCLE PRINCIPAL DE BÚSQUEDA TABÚ SIMPLE ===
    # Iteramos hasta cumplir alguno de los dos criterios de parada.
    iteracion = 0
    while iteracion < iteraciones_max:
        # Criterio de parada por estancamiento: si llevamos demasiadas
        # iteraciones consecutivas sin mejorar el mejor global, terminamos.
        # Verificación al INICIO de la iteración para que la corrida pueda
        # detenerse en cualquier momento sin gastar trabajo extra.
        if iter_sin_mejora >= max_iter_sin_mejora:
            break

        # --- Presupuesto de wall-clock (val_egl_20260710) ---
        # Cortamos la corrida si se alcanza el tiempo máximo. La comprobación
        # es una sustracción trivial (nanosegundos) — el overhead es despreciable.
        if (
            tiempo_limite_segundos is not None
            and (time.perf_counter() - t0) >= tiempo_limite_segundos
        ):
            break

        if guardar_historial:
            # Registramos el mejor costo conocido al inicio de esta iteración.
            historial_best.append(mejor_costo)

        # --- Paso 1: Generación del lote de vecinos ---
        # Cada vecino se obtiene aplicando un operador sobre la solución
        # actual. La selección de operador se hace en dos pasos (mismo mecanismo
        # que SA y RTS):
        #   (a) seleccionar_grupo_operadores_inter_intra hace UN rng.random()
        #       y decide si esta iteración usa el grupo INTER-RUTA o el grupo
        #       INTRA-RUTA, con umbral alpha_inter (si hay violación) o p_inter
        #       (si la solución es factible). Esto sesga la búsqueda hacia
        #       reparación cuando ``viol_actual > 1e-12`` y permite
        #       diversificación inter-ruta cuando no la hay.
        #   (b) generar_vecino selecciona UN operador concreto dentro del grupo
        #       elegido (selección uniforme, ``pesos_operadores=None``).
        # Importante: la decisión del grupo se realiza UNA vez por iteración
        # del bucle principal — y NO una vez por cada vecino del lote —, así
        # todo el lote comparte el mismo "modo" de sesgo. Esto es coherente
        # con el espíritu de "best-improvement sobre un lote homogéneo".
        grupo_ops, _hubo_viol = seleccionar_grupo_operadores_inter_intra(
            rng,
            viol_actual,
            ops_intra_list,
            ops_inter_list,
            ops_fallback_list,
            alpha_inter=alpha_inter_eff,
            p_inter=p_inter_eff,
            metodo=metodo_seleccion,
        )
        # Si la solución actual viola capacidad en esta iteración, lo contamos
        # para el reporte (CSV y dataclass). El flag proviene directamente del
        # helper para garantizar consistencia con el criterio numérico usado en
        # SA (umbral 1e-12).
        if _hubo_viol:
            iteraciones_con_violacion += 1
        vecinos: list[list[list[str]]] = []
        movimientos: list[MovimientoVecindario] = []
        for _ in range(tam_vecindario):
            vecino, mov = generar_vecino(
                sol_actual,
                rng=rng,
                operadores=grupo_ops,          # grupo INTER o INTRA seleccionado arriba
                pesos_operadores=None,         # sin pesos: selección uniforme dentro del grupo
                marcador_depot=md_op,
                devolver_con_deposito=True,    # incluye "D" al inicio/fin de cada ruta
                usar_gpu=usar_gpu,
                backend=backend_vecindario,
                encoding=encoding,
            )
            vecinos.append(vecino)
            movimientos.append(mov)
            contador.proponer(mov.operador)     # registra el operador propuesto

        # --- Paso 2: Evaluación en lote ---
        # Codificamos los vecinos a IDs enteros y los evaluamos en una sola
        # pasada vectorizada. Usamos el objetivo penalizado para que TS guíe
        # la búsqueda equilibrando costo puro y violación de capacidad.
        sols_ids = [encode_solution(v, ctx.encoding) for v in vecinos]
        obj_np, costos_np, viols_np = costo_lote_penalizado_ids(sols_ids, ctx, lam_eff)
        vecinos_evaluados += len(vecinos)

        # --- Paso 3: Selección del mejor vecino no-tabú (best-improvement) ---
        # Selección y aspiración se hacen sobre el objetivo penalizado; los costos
        # puros solo se usan para actualizar ``mejor_costo`` al final del paso 6.
        mejor_admisible_idx = -1
        mejor_admisible_obj = float("inf")
        mejor_total_idx = 0
        mejor_total_obj = float("inf")
        aspiracion_en_iter = False

        for idx in range(len(obj_np)):
            o_v = float(obj_np[idx])    # objetivo penalizado del vecino idx

            if o_v < mejor_total_obj:
                mejor_total_obj = o_v
                mejor_total_idx = idx

            key = _clave_tabu(movimientos[idx])
            es_tabu = key in set_tabu

            # Aspiración: superar el mejor objetivo penalizado global conocido.
            aspiracion = o_v < mejor_obj_pen - 1e-15

            if es_tabu and not aspiracion:
                continue

            if o_v < mejor_admisible_obj:
                mejor_admisible_obj = o_v
                mejor_admisible_idx = idx
                if es_tabu and aspiracion:
                    aspiracion_en_iter = True

        # --- Paso 4: Elección del movimiento a ejecutar ---
        if mejor_admisible_idx == -1:
            elegido_idx = mejor_total_idx
            iteraciones_todos_tabu += 1
        else:
            elegido_idx = mejor_admisible_idx
            if aspiracion_en_iter:
                aspiraciones += 1

        # Actualizamos el estado actual al vecino elegido. costo_actual y
        # viol_actual se toman directamente del batch (sin recalcular).
        sol_actual = vecinos[elegido_idx]
        costo_actual = float(costos_np[elegido_idx])
        viol_actual = float(viols_np[elegido_idx])
        obj_actual = float(obj_np[elegido_idx])
        ultimo_mov_aceptado = movimientos[elegido_idx]
        contador.aceptar(ultimo_mov_aceptado.operador)

        # --- Paso 5: Actualización de la lista tabú FIFO ---
        # Insertamos la clave del movimiento ejecutado en la deque. Si la deque
        # estaba llena (len == maxlen), Python descarta automáticamente el
        # elemento más antiguo del lado izquierdo. Para mantener tanto el set
        # como el Counter sincronizados, detectamos manualmente el descarte y
        # actualizamos ambos contenedores antes de hacer el append.
        nueva_clave = _clave_tabu(ultimo_mov_aceptado)
        if len(lista_tabu) == tabu_tenure:
            # La deque va a descartar el elemento más antiguo al hacer append.
            # Capturamos cuál es para actualizar el Counter y, si corresponde,
            # removerlo también del set.
            clave_descartada = lista_tabu[0]
            # Decrementamos su conteo. ``Counter`` permite negativos pero NO
            # entramos en ese régimen porque cada inserción incrementa antes.
            conteo_tabu[clave_descartada] -= 1
            # Si tras el decremento el conteo llegó a 0, esa clave YA no
            # aparece en la deque tabú: la sacamos del set y del Counter
            # (descartar entradas con conteo 0 evita que el Counter crezca
            # indefinidamente con claves "fantasma"). Reemplaza el antiguo
            # ``list(lista_tabu).count(clave_descartada) == 1``, que era O(n).
            if conteo_tabu[clave_descartada] == 0:
                del conteo_tabu[clave_descartada]
                set_tabu.discard(clave_descartada)
        lista_tabu.append(nueva_clave)
        set_tabu.add(nueva_clave)
        # Incrementamos el conteo de la clave recién insertada. Si ya estaba
        # presente (duplicado), el conteo simplemente sube de 1 a 2, y el set
        # queda igual (un set ignora ``add`` de un elemento ya presente).
        conteo_tabu[nueva_clave] += 1

        # --- Paso 6: Actualización del mejor global ---
        # Comparamos por objetivo penalizado; guardamos el costo puro para el reporte.
        if obj_actual < mejor_obj_pen - 1e-15:
            mejor_obj_pen = obj_actual
            mejor_costo = costo_actual
            mejor_sol = copiar_solucion_labels(sol_actual)
            mejor_factible = viol_actual < 1e-12
            iter_mejor = iteracion
            mejoras += 1
            contador.registrar_mejora(ultimo_mov_aceptado.operador)
            iter_sin_mejora = 0
        else:
            # No mejoramos: incrementamos el contador de estancamiento. Cuando
            # alcance max_iter_sin_mejora, el bucle while se interrumpirá al
            # inicio de la siguiente iteración.
            iter_sin_mejora += 1

            # --- Kick por estancamiento global (strict_intra_inter_20260524) ---
            # Cuando la MH lleva demasiadas iteraciones sin mejorar el mejor
            # global, se aplica una perturbación inter-ruta disruptiva y se
            # reinicia el contador. Si se alcanza max_resets, se para. El kick
            # se aplica DESPUÉS de incrementar iter_sin_mejora para que la
            # decisión use el valor recién actualizado (consistencia con el
            # criterio de parada por estancamiento al inicio de la iteración).
            if (max_iter_sin_mejora_kick is not None
                    and iter_sin_mejora >= max_iter_sin_mejora_kick):
                if intensificador is not None:
                    # Respuesta de intensificacion (p.ej. Path Relinking limpio)
                    # hacia la mejor solucion global, en lugar del kick aleatorio.
                    sol_actual = intensificador(
                        sol_actual, mejor_sol, ctx, lam_eff, rng, encoding, md_op
                    )
                else:
                    # Import diferido: solo se carga si la corrida activa el kick.
                    from metacarp.strict_intra_inter_20260524 import aplicar_kick_labels
                    sol_actual = aplicar_kick_labels(
                        sol_actual, rng, md_op, encoding=encoding
                    )
                # Recalculamos costo, violación y objetivo penalizado tras el kick.
                viol_actual = float(exceso_capacidad_rapido(sol_actual, ctx))
                costo_actual = float(costo_rapido(sol_actual, ctx))
                obj_actual = costo_actual + lam_eff * viol_actual
                iter_sin_mejora = 0
                n_resets_kick += 1
                if max_resets is not None and n_resets_kick >= max_resets:
                    # Cota dura: incrementamos iteracion para reportar el total
                    # correcto y rompemos el bucle externo.
                    iteracion += 1
                    break

        # Avanzamos a la siguiente iteración.
        iteracion += 1

    # === FIN DEL BUCLE PRINCIPAL ===

    # Capturamos métricas finales.
    elapsed = time.perf_counter() - t0
    # _gap_descartado: el gap puro no se reporta directamente, solo abs y pct.
    _gap_descartado, mejora_abs, mejora_pct = calcular_metricas_gap(costo_ref, mejor_costo)

    # --- Guardado en CSV (opcional) ---
    archivo_csv: str | None = None
    if guardar_csv:
        ruta = ruta_csv or f"resultados_busqueda_tabu_simple_{nombre_instancia}.csv"
        # Generamos el reporte detallado al final (costoso pero solo una vez).
        detalle_txt, costo_total_reporte = generar_reporte_detallado(
            mejor_sol, data, G,
            nombre_instancia=nombre_instancia,
            marcador_depot_etiqueta=marcador_depot_etiqueta,
            usar_gpu=False,  # el reporte de texto usa NetworkX (más legible que GPU)
        )
        # Columnas BKS (referencia y gap) extraídas con la utilidad estándar.
        _bks = resumen_bks_csv(data, mejor_costo)
        # IMPORTANTE: la fila NO incluye id_corrida ni config_id (convención del
        # proyecto). Si alguien pasa esos kwargs por compatibilidad, se ignoran
        # vía **_ignorado_kwargs en la firma de la función.
        fila = {
            "metaheuristica": "busqueda_tabu_simple",
            "instancia": nombre_instancia,
            "bks_referencia": _bks["bks_referencia"],
            "bks_origen": _bks["bks_origen"],
            "gap_bks_porcentaje": _bks["gap_bks_porcentaje"],
            "repeticion": repeticion if repeticion is not None else "",
            "semilla": semilla,
            "tiempo_segundos": elapsed,
            "mejor_costo": mejor_costo,
            "costo_solucion_inicial": costo_ref,
            "mejora_absoluta": mejora_abs,
            "mejora_porcentaje": mejora_pct,
            # Parámetros del TS simple.
            "iteraciones_max": iteraciones_max,
            "max_iter_sin_mejora": max_iter_sin_mejora,
            "tam_vecindario": tam_vecindario,
            "tabu_tenure": tabu_tenure,
            # Contador de operadores (4 categorías × 9 operadores = 36 columnas).
            **contador.resumen_csv(),
            # Estadísticas de la corrida.
            "iteraciones_totales": iteracion,
            "vecinos_evaluados": vecinos_evaluados,
            "iteracion_mejor": iter_mejor,
            "iteraciones_sin_mejora_final": iter_sin_mejora,
            "aspiraciones": aspiraciones,
            "iteraciones_todos_tabu": iteraciones_todos_tabu,
            "aceptadas": sum(contador.aceptados.values()),
            "mejoras": mejoras,
            "mejor_solucion_factible_final": mejor_factible,
            "mejor_solucion_tr_legible": solucion_legible_humana(mejor_sol),
            "reporte_detalle_deadheading": detalle_txt,
            "costo_total_desde_reporte": costo_total_reporte,
            # ---- Columnas del sesgo inter/intra (añadidas al FINAL para no
            # alterar el orden histórico de las columnas del CSV; lectores
            # antiguos siguen funcionando porque las nuevas columnas solo
            # aparecen al final de cada fila). ----
            "alpha_inter": alpha_inter_eff,
            "p_inter": p_inter_eff,
            "iteraciones_con_violacion": iteraciones_con_violacion,
            # Fracción [0, 1] de iteraciones del bucle principal que entraron
            # al modo sesgado por violación. Se reporta 0.0 si no se ejecutó
            # ninguna iteración (caso patológico) para evitar división por cero.
            "fraccion_iter_con_violacion": (
                iteraciones_con_violacion / iteracion if iteracion > 0 else 0.0
            ),
            # Columna del mecanismo de kick (strict_intra_inter_20260524).
            # 0 cuando la variante experimental no esta activa (default).
            "n_resets_kick": n_resets_kick,
            # λ efectivo usado en el objetivo penalizado del bucle principal.
            "lambda_capacidad": lam_eff,
        }
        archivo_csv = guardar_resultado_csv(fila=fila, ruta_csv=ruta)

    # Construimos y retornamos el resultado inmutable.
    return BusquedaTabuSimpleResult(
        mejor_solucion=mejor_sol,
        mejor_costo=mejor_costo,
        solucion_inicial_referencia=sol_ref,
        costo_solucion_inicial=costo_ref,
        mejora_absoluta=mejora_abs,
        mejora_porcentaje_inicial_vs_final=mejora_pct,
        tiempo_segundos=elapsed,
        iteraciones_totales=iteracion,
        vecinos_evaluados=vecinos_evaluados,
        iteracion_mejor=iter_mejor,
        iteraciones_sin_mejora_final=iter_sin_mejora,
        aspiraciones=aspiraciones,
        iteraciones_todos_tabu=iteraciones_todos_tabu,
        mejoras=mejoras,
        semilla=semilla,
        backend_evaluacion=ctx.backend_real,
        historial_mejor_costo=historial_best,
        ultimo_movimiento_aceptado=ultimo_mov_aceptado,
        operadores_propuestos=contador.como_dict_ordenado(contador.propuestos),
        operadores_aceptados=contador.como_dict_ordenado(contador.aceptados),
        operadores_mejoraron=contador.como_dict_ordenado(contador.mejoraron),
        operadores_trayectoria_mejor=contador.como_dict_ordenado(contador.trayectoria_mejor),
        mejor_solucion_factible_final=mejor_factible,
        archivo_csv=archivo_csv,
        # Trazabilidad del sesgo inter/intra aplicado en esta corrida.
        alpha_inter_aplicado=alpha_inter_eff,
        p_inter_aplicado=p_inter_eff,
        iteraciones_con_violacion=iteraciones_con_violacion,
        # Kicks aplicados (variante experimental strict_intra_inter_20260524).
        n_resets_kick=n_resets_kick,
    )


def busqueda_tabu_simple_desde_instancia(
    nombre_instancia: str,
    *,
    root: str | None = None,
    iteraciones_max: int = 400,
    max_iter_sin_mejora: int = 100,
    tam_vecindario: int = 25,
    tabu_tenure: int = 20,
    semilla: int | None = None,
    operadores: Iterable[str] = OPERADORES_POPULARES,
    marcador_depot_etiqueta: str | None = None,
    usar_gpu: bool = False,
    backend_vecindario: Literal["labels", "ids"] = "labels",
    guardar_historial: bool = True,
    guardar_csv: bool = False,
    ruta_csv: str | None = None,
    repeticion: int | None = None,
    extra_csv: dict[str, object] | None = None,
    # Parámetros del sesgo inter/intra (mismos defaults que SA: None -> 0.8/0.6).
    alpha_inter: float | None = None,
    p_inter: float | None = None,
    metodo_seleccion: str = "canonico",  # método de combinación inter/intra: canonico|p_inter|binario|random
    # Kick reactivo (variante experimental strict_intra_inter_20260524).
    max_iter_sin_mejora_kick: int | None = None,
    max_resets: int | None = None,
    # Hook de intensificacion opcional (p.ej. Path Relinking limpio).
    intensificador: Callable | None = None,
    # Penalización por capacidad (lambda_grid_20260525). None = default instance-aware.
    lambda_capacidad: float | None = None,
    # Presupuesto de wall-clock (val_egl_20260710). None = sin límite.
    tiempo_limite_segundos: float | None = None,
    **_ignorado_kwargs: object,  # absorbe kwargs heredados (p.ej. id_corrida, config_id)
) -> BusquedaTabuSimpleResult:
    """
    Función de conveniencia: carga todos los recursos necesarios desde el nombre
    de la instancia y ejecuta la búsqueda tabú simple completa.

    Equivalente a llamar manualmente a load_instances + cargar_objeto_gexf +
    cargar_solucion_inicial + busqueda_tabu_simple.
    """
    # Cargamos los datos de la instancia (capacidad, demandas, BKS, etc.).
    data = load_instances(nombre_instancia, root=root)
    # Cargamos el grafo de la instancia desde el archivo GEXF.
    G = cargar_objeto_gexf(nombre_instancia, root=root)
    # Cargamos el objeto de solución inicial desde el archivo pickle.
    inicial_obj = cargar_solucion_inicial(nombre_instancia, root=root)
    return busqueda_tabu_simple(
        inicial_obj,
        data,
        G,
        iteraciones_max=iteraciones_max,
        max_iter_sin_mejora=max_iter_sin_mejora,
        tam_vecindario=tam_vecindario,
        tabu_tenure=tabu_tenure,
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
        extra_csv=extra_csv,
        # Propagamos los parámetros del sesgo inter/intra al núcleo del algoritmo.
        alpha_inter=alpha_inter,
        p_inter=p_inter,
        metodo_seleccion=metodo_seleccion,
        # Propagamos el mecanismo de kick (variante experimental).
        max_iter_sin_mejora_kick=max_iter_sin_mejora_kick,
        max_resets=max_resets,
        # Propagamos el hook de intensificacion (p.ej. Path Relinking limpio).
        intensificador=intensificador,
        # Propagamos la penalización de capacidad (lambda_grid_20260525).
        lambda_capacidad=lambda_capacidad,
        # Propagamos el presupuesto de wall-clock (val_egl_20260710).
        tiempo_limite_segundos=tiempo_limite_segundos,
    )
