"""
Reactive Tabu Search (RTS) de Battiti & Tecchiolli (1994) para CARP.

Concepto algorítmico
--------------------
La Búsqueda Tabú Reactiva (Reactive Tabu Search, RTS) extiende la Búsqueda
Tabú clásica de Glover (1986) con tres mecanismos de adaptación dinámica
introducidos por Roberto Battiti y Giampietro Tecchiolli (1994) en su artículo
seminal "The Reactive Tabu Search" (ORSA Journal on Computing, vol. 6, no. 2).
La idea fundamental es que la **lista tabú deja de ser estática**: el propio
algoritmo "siente" cuándo está atrapado en ciclos y ajusta los parámetros
de su memoria de corto plazo en consecuencia.

Mecanismos esenciales (los tres pilares de RTS)
------------------------------------------------
1. **Tenencia tabú dinámica (auto-ajustable)**

   En lugar de un ``tabu_tenure`` fijo (como en la versión simple),
   el tenure varía durante la ejecución dentro de un rango
   ``[tabu_tenure_min, tabu_tenure_max]``:

   - Cuando se detecta que la solución actual **ya fue visitada antes**
     (señal clara de un ciclo o de una "trampa" en una cuenca de atracción),
     el tenure **crece** multiplicándose por ``factor_aumento`` (>1).
     Tenure más grande = más prohibiciones = el algoritmo está obligado
     a explorar regiones nuevas.
   - Cuando han pasado ``iter_sin_repeticion_para_reducir`` iteraciones
     **sin detectar repeticiones**, asumimos que ya no estamos en una
     zona problemática y el tenure **decrece** multiplicándose por
     ``factor_reduccion`` (<1), permitiendo de nuevo movimientos más libres.
   - El tenure siempre se mantiene acotado en ``[min, max]``.

2. **Memoria de soluciones visitadas (detección de ciclos)**

   Para detectar repeticiones de forma eficiente se calcula un **hash**
   canónico de cada solución visitada y se almacena en un diccionario
   ``historial: dict[hash_solucion, info]``. La información asociada
   incluye:

   - La iteración en que se vio por última vez la solución.
   - El número total de veces que se ha visitado.

   En cada iteración se comprueba en O(1) si el hash de la solución
   actual ya estaba presente. Esto evita el costo prohibitivo de
   comparar la solución actual contra todas las anteriores.

3. **Mecanismo de escape (diversificación fuerte)**

   Si una solución se repite más de ``umbral_repeticiones_escape`` veces
   (señal de un ciclo "duro" que el aumento de tenure no logra romper),
   se dispara un **escape**:

   - Se aplican ``num_movimientos_escape`` movimientos aleatorios
     consecutivos sobre la solución actual (la lista tabú se ignora
     durante el escape; el objetivo es saltar lejos en el espacio).
   - **Se limpia completamente la lista tabú** tras el escape: las
     prohibiciones acumuladas en la zona "atrapada" ya no son útiles
     en la nueva zona.
   - **Se reinicia el historial de soluciones**: las soluciones de la
     zona vieja ya no nos interesan; nos importa detectar ciclos en
     la nueva región. Esta es una decisión de diseño documentada
     más abajo.

Lo que hereda de la Búsqueda Tabú simple
-----------------------------------------
- Lista tabú FIFO de movimientos (deque + set paralelo para lookup O(1)).
- Best-improvement sobre un lote muestreado de vecinos por iteración
  (``tam_vecindario`` vecinos, evaluación vectorizada con NumPy).
- Criterio de aspiración clásico: un movimiento tabú se acepta si
  produce una solución estrictamente mejor que el mejor global conocido.
- Doble criterio de parada: ``iteraciones_max`` o ``max_iter_sin_mejora``.
- Mismos 9 operadores de ``OPERADORES_POPULARES``.
- Misma rutina de selección de solución inicial
  (``seleccionar_mejor_inicial_rapido``).

Parámetros instance-aware
-------------------------
Los valores por defecto se calculan a partir del tamaño del problema
``n = número de tareas requeridas`` para que la metaheurística se
adapte automáticamente a la dificultad de cada instancia. Si el
usuario pasa un valor explícito, se respeta.

Referencias
-----------
- Battiti, R., & Tecchiolli, G. (1994). "The Reactive Tabu Search."
  ORSA Journal on Computing, 6(2), 126-140.
- Glover, F. (1986). "Future paths for integer programming and links
  to artificial intelligence." Computers & Operations Research, 13(5),
  533-549.
"""
# Permite usar anotaciones de tipo como `float | None` en Python < 3.10.
from __future__ import annotations

# collections.deque es una cola doblemente enlazada con maxlen: ideal para una
# lista tabú FIFO de longitud fija. A diferencia del TS simple, en RTS
# necesitamos AJUSTAR la maxlen dinámicamente cuando el tenure cambia
# (Python no permite mutar maxlen de una deque existente, así que reconstruimos
# la deque preservando los elementos más recientes; ver _ajustar_maxlen_deque).
# Counter es un diccionario especializado para conteo de ocurrencias: lo usamos
# como acompañante de la deque tabú para responder en O(1) la pregunta
# "¿queda al menos otra copia de esta clave?" cuando descartamos elementos.
from collections import Counter, deque
# math.sqrt para calcular las cotas de tenure a partir de n (regla clásica).
import math
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

# Importaciones internas del paquete metacarp:
from .busqueda_indices import build_search_encoding, encode_solution  # codificación de soluciones
from .cargar_grafos import cargar_objeto_gexf                         # carga el grafo desde GEXF
from .cargar_soluciones_iniciales import cargar_solucion_inicial      # carga la solución inicial
from .evaluador_costo import (
    costo_lote_ids,                      # evalúa un lote de soluciones (NumPy vectorizado)
    costo_rapido,                        # evalúa una sola solución (label-based)
    exceso_capacidad_rapido,             # calcula la violación de capacidad de una solución
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
    seleccionar_grupo_operadores_inter_intra,  # helper compartido con SA y TS simple para sesgo inter/intra
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
    "BusquedaTabuReactivaResult",
    "busqueda_tabu_reactiva",
    "busqueda_tabu_reactiva_desde_instancia",
]


# --- CONCEPTO OOP: @dataclass(frozen=True, slots=True) ---
# frozen=True: el objeto es inmutable una vez creado. Seguro porque el resultado
#              de la búsqueda no debe cambiar después de terminar la ejecución.
# slots=True:  Python reserva exactamente los atributos declarados, reduciendo
#              el uso de memoria por instancia.
@dataclass(frozen=True, slots=True)
class BusquedaTabuReactivaResult:
    """
    Resultado completo de la Búsqueda Tabú Reactiva (RTS).

    Hereda toda la información del TS simple y agrega métricas específicas
    de los mecanismos reactivos: trayectoria del tenure, número de
    repeticiones detectadas, número de escapes y de ajustes del tenure.
    """

    # === Campos comunes con el TS simple ===
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
    # Total de vecinos evaluados durante toda la corrida.
    vecinos_evaluados: int
    # Iteración en la que se descubrió la mejor solución reportada.
    iteracion_mejor: int
    # Iteraciones consecutivas SIN mejora al terminar (criterio de parada temprana).
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
    # Historial del tenure efectivo al inicio de cada iteración
    # (clave para visualizar el comportamiento reactivo).
    historial_tenure: list[int] = field(default_factory=list)
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

    # === Métricas específicas de RTS ===
    # Valor del tenure al terminar la corrida (importante para entender
    # en qué régimen se quedó el algoritmo: ¿con tenure alto = mucho ciclo? ¿bajo = libre?).
    tenure_final: int = 0
    # Promedio del tenure a lo largo de toda la ejecución
    # (cuantifica la "presión tabú media" que se aplicó).
    tenure_promedio: float = 0.0
    # Máximo y mínimo valor del tenure observados durante la corrida.
    # Permiten ver el rango efectivo recorrido por el mecanismo reactivo.
    tenure_max_alcanzado: int = 0
    tenure_min_alcanzado: int = 0
    # Número de iteraciones en las que el hash de la solución actual coincidió
    # con uno ya visto (señal de ciclo).
    num_repeticiones_detectadas: int = 0
    # Número de veces que se disparó el mecanismo de escape
    # (movimientos aleatorios + limpieza de memoria).
    num_escapes_realizados: int = 0
    # Cuántas veces se incrementó el tenure (porque hubo repetición).
    num_aumentos_tenure: int = 0
    # Cuántas veces se redujo el tenure (porque pasó un período sin repeticiones).
    num_reducciones_tenure: int = 0
    # Parámetros instance-aware finalmente aplicados (auditoría):
    tenure_inicial_aplicado: int = 0
    tenure_min_aplicado: int = 0
    tenure_max_aplicado: int = 0
    # === Campos del mecanismo de sesgo inter/intra (compatibilidad con SA y TS simple) ===
    # Estos campos reportan los valores efectivos del sesgo aplicado durante la
    # corrida y cuántas iteraciones del BUCLE PRINCIPAL entraron al modo
    # sesgado por violación. NO se cuentan aquí los movimientos aleatorios del
    # mecanismo de escape: el escape ignora la lista tabú y aplica una ráfaga
    # de movimientos uniformes con la lista completa de operadores (no
    # interviene el sesgo inter/intra), así que conceptualmente queda fuera
    # del modo "búsqueda guiada" que el sesgo regula.
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

    Idéntica a la del TS simple: incluye el nombre del operador, los índices
    de rutas, las posiciones internas y las etiquetas de las tareas movidas.
    Permite comparar movimientos contra la lista tabú en O(1) (vía set).

    Decisión de diseño: la "memoria estructural" es el movimiento completo
    (no un atributo aislado como "la tarea X"). Esto es lo más permisivo
    posible dentro del esquema de prohibición por movimiento, y coincide
    con la implementación del TS simple para que la comparación sea justa.
    """
    return (
        mov.operador,                  # nombre del operador (ej: 'swap_intra')
        mov.ruta_a, mov.ruta_b,        # índices de las rutas involucradas
        mov.i, mov.j, mov.k, mov.l,    # posiciones dentro de las rutas
        tuple(mov.labels_movidos),     # etiquetas de las tareas que se mueven
    )


def _hash_solucion(sol: list[list[str]]) -> int:
    """
    Calcula un hash canónico de una solución CARP, robusto al orden de las rutas.

    Razonamiento:
    - Dos soluciones que recorren EXACTAMENTE las mismas tareas en el mismo
      orden dentro de cada ruta, pero con las rutas listadas en distinto
      orden global, son **la misma solución** desde el punto de vista del
      costo objetivo (los vehículos son intercambiables).
    - Por eso, para detectar ciclos, ordenamos las rutas de forma canónica
      antes de calcular el hash. Como cada ruta es una secuencia de strings
      (incluyendo "D"), basta con convertirla a tupla y ordenar las tuplas
      lexicográficamente.

    Decisión de diseño:
    - No invertimos rutas (una ruta y su reverso pueden tener costos distintos
      en CARP dirigido; aquí asumimos no-direccionalidad solo en el orden
      global de las rutas, no dentro de cada ruta).
    - Si el problema fuera completamente no-dirigido podríamos canonicalizar
      también dentro de cada ruta (tomando el mínimo entre la ruta y su
      reverso), pero esto introduciría falsos positivos en variantes
      dirigidas. La versión actual es CONSERVADORA: solo identifica como
      "misma solución" lo que es indiscutiblemente lo mismo.

    Retorno:
        int — hash de Python, apto para usar como clave de diccionario.
    """
    # Convertimos cada ruta a una tupla inmutable (necesario porque las listas
    # no son hashables y queremos ordenar tuplas, no listas).
    rutas_canon = tuple(sorted(tuple(r) for r in sol))
    # hash() de Python sobre una tupla de tuplas de strings es O(L) donde
    # L es la longitud total de la solución; barato y suficiente.
    return hash(rutas_canon)


def _ajustar_maxlen_deque(
    lista_tabu: deque[tuple[Any, ...]],
    set_tabu: set[tuple[Any, ...]],
    nuevo_maxlen: int,
    conteo_tabu: Counter[tuple[Any, ...]] | None = None,
) -> deque[tuple[Any, ...]]:
    """
    Reconstruye la deque tabú con un nuevo ``maxlen`` preservando los elementos
    más RECIENTES (los del lado derecho) y descartando los más antiguos si
    hace falta.

    Por qué reconstruir y no mutar:
    - ``collections.deque.maxlen`` es de SOLO LECTURA en Python; no se puede
      asignar después de crear la deque. La única manera de cambiar la
      capacidad es construir una nueva deque.

    Estrategia ante un cambio de maxlen:
    - Si el nuevo maxlen es **mayor** que el actual: simplemente copiamos
      todos los elementos a una deque más grande. No descartamos nada.
    - Si el nuevo maxlen es **menor** que el actual: copiamos solo los
      ``nuevo_maxlen`` elementos más recientes (los del lado derecho).
      Los más antiguos se descartan, lo cual es coherente con la semántica
      FIFO de la lista tabú.

    Actualizamos también ``set_tabu`` para que siga sincronizado: si algunos
    elementos fueron descartados, deben salir del set.

    Sincronización del ``conteo_tabu`` (Counter paralelo, parámetro opcional):
    - Si se pasa un Counter, lo decrementamos por cada elemento descartado y
      lo limpiamos cuando alguna entrada llega a 0. Mantener este Counter
      coherente es CRÍTICO para que la microoptimización del paso 5 (lookup
      O(1) de duplicados al descartar la cabeza de la deque) siga siendo
      correcta tras un ajuste de ``maxlen``.
    - Si el parámetro es ``None`` (compatibilidad con llamadas antiguas), no
      se toca ningún Counter.

    Devuelve la nueva deque (el set y el Counter se actualizan in-place).
    """
    # Si la capacidad no cambia, no hacemos nada (optimización).
    if lista_tabu.maxlen == nuevo_maxlen:
        return lista_tabu

    # Convertimos la deque a lista para indexar por posiciones.
    elementos = list(lista_tabu)
    # Si tenemos más elementos que la nueva capacidad, conservamos solo
    # los más recientes (los últimos ``nuevo_maxlen`` del orden FIFO).
    if len(elementos) > nuevo_maxlen:
        descartados = elementos[: len(elementos) - nuevo_maxlen]
        elementos = elementos[len(elementos) - nuevo_maxlen :]
        # Reconstrucción del set: descartamos del set solo si el elemento
        # YA NO aparece en la deque truncada (un movimiento puede repetirse).
        # Construimos un set temporal de ``elementos`` para acelerar la
        # consulta de pertenencia (O(1) vs O(n) en lista) cuando hay muchos
        # descartados; conceptualmente equivalente al ``in elementos`` previo.
        elementos_set = set(elementos)
        for desc in descartados:
            if desc not in elementos_set:
                set_tabu.discard(desc)
            # Si llevamos Counter paralelo, decrementamos por cada
            # descarte y eliminamos la entrada del Counter al llegar a 0.
            # Hacemos la limpieza aquí (no solo en el bucle principal)
            # para que el invariante "conteo_tabu refleja la deque actual"
            # se conserve también tras un reescalado.
            if conteo_tabu is not None:
                conteo_tabu[desc] -= 1
                if conteo_tabu[desc] <= 0:
                    # Si por error llegara a negativo (no debería), igualmente
                    # eliminamos la entrada para evitar acumular basura.
                    del conteo_tabu[desc]

    # Construimos la nueva deque con la capacidad solicitada.
    return deque(elementos, maxlen=nuevo_maxlen)


def busqueda_tabu_reactiva(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    # ---------- Criterios de parada ----------
    iteraciones_max: int | None = None,        # cota dura de iteraciones (criterio 1); None -> 20·n
    max_iter_sin_mejora: int | None = None,    # estancamiento (criterio 2); None -> 5·n
    # ---------- Vecindario ----------
    tam_vecindario: int | None = None,         # tamaño del lote; None -> max(20, 2·n)
    # ---------- Parámetros reactivos del tenure ----------
    tabu_tenure_inicial: int | None = None,    # tenure de arranque; None -> max(3, round(sqrt(n)))
    tabu_tenure_min: int | None = None,        # cota inferior; None -> 3
    tabu_tenure_max: int | None = None,        # cota superior; None -> max(10, round(3·sqrt(n)))
    factor_aumento: float = 1.2,               # multiplicador del tenure al detectar repetición
    factor_reduccion: float = 0.9,             # multiplicador al pasar tiempo sin repeticiones
    iter_sin_repeticion_para_reducir: int | None = None,  # paciencia para reducir; None -> 2·round(sqrt(n))
    # ---------- Parámetros de escape (diversificación) ----------
    umbral_repeticiones_escape: int = 3,       # repeticiones de una misma sol. para disparar escape
    num_movimientos_escape: int | None = None, # cuántos movimientos aleatorios en el escape; None -> max(3, n//10)
    # ---------- Configuración general ----------
    semilla: int | None = None,                # semilla para reproducibilidad
    operadores: Iterable[str] = OPERADORES_POPULARES,  # operadores de vecindario
    marcador_depot_etiqueta: str | None = None,  # etiqueta del depósito
    usar_gpu: bool = False,                    # flag de GPU (placeholder)
    backend_vecindario: Literal["labels", "ids"] = "labels",
    guardar_historial: bool = True,            # si True, guarda el historial de costos y tenures
    guardar_csv: bool = False,                 # si True, escribe la fila de resultados en CSV
    ruta_csv: str | None = None,               # ruta del CSV (None = nombre automático)
    nombre_instancia: str = "instancia",       # nombre de la instancia para el CSV
    repeticion: int | None = None,             # número de repetición dentro de un experimento
    root: str | None = None,                   # directorio raíz para localizar los datos
    extra_csv: dict[str, object] | None = None,  # columnas adicionales para el CSV (no usadas)
    # --- Parámetros de sesgo inter/intra (compatibilidad EXACTA con SA y TS simple) ---
    # Defaults None = usar el mismo valor por defecto que SA (0.8 y 0.6). Se
    # mantiene la firma con None para que el script de experimentos pueda dejar
    # el parámetro en su valor canónico SA sin tener que conocer la constante.
    alpha_inter: float | None = None,          # P(elegir inter) cuando la sol. actual viola capacidad
    p_inter: float | None = None,              # P(elegir inter) cuando la sol. actual es factible
    # --- Kick reactivo (variante experimental strict_intra_inter_20260524) ---
    # Cuando ``iter_sin_mejora`` alcanza este umbral se aplica una perturbacion
    # INTER-RUTA disruptiva (ver ``metacarp.strict_intra_inter_20260524``) y se
    # reinicia el contador. None = mecanismo desactivado (comportamiento clasico).
    max_iter_sin_mejora_kick: int | None = None,
    # Cota dura del numero de kicks consecutivos. Cuando se alcanza, la corrida
    # termina. None = sin tope (los kicks se permiten indefinidamente).
    max_resets: int | None = None,
    **_ignorado_kwargs: object,                # absorbe kwargs heredados (id_corrida, config_id)
) -> BusquedaTabuReactivaResult:
    """
    Reactive Tabu Search (Battiti & Tecchiolli 1994) para CARP.

    Estructura del algoritmo (best-non-tabu + tenure dinámico + escape)
    -------------------------------------------------------------------
    1. **Inicialización**:
       - Calcula los parámetros instance-aware a partir de
         ``n = número de tareas requeridas`` si el usuario no los especifica.
       - Selecciona la mejor solución inicial entre las candidatas del pickle.
       - Crea la lista tabú FIFO con ``tabu_tenure_inicial`` slots.
       - Registra el hash de la solución inicial en ``historial``.

    2. **Bucle principal**, en cada iteración:

       a. Verifica criterios de parada (iteraciones, estancamiento).
       b. Genera ``tam_vecindario`` vecinos aleatorios.
       c. Evalúa el lote en una sola pasada vectorizada.
       d. Selecciona el mejor vecino no-tabú (con aspiración clásica).
       e. Se mueve al vecino elegido (siempre, aunque sea peor).
       f. Inserta el movimiento en la lista tabú FIFO.
       g. **Mecanismo reactivo del tenure**:

          - Calcula el hash de la solución actual.
          - Si ya estaba en ``historial``:

              * Incrementa el contador de visitas de ese hash.
              * Reinicia ``iter_sin_repeticion``.
              * Aumenta el tenure (acotado por ``tabu_tenure_max``)
                y reconstruye la deque con la nueva capacidad.
              * Si el contador de visitas alcanza
                ``umbral_repeticiones_escape``, dispara el ESCAPE:
                limpia lista tabú, limpia historial, aplica
                ``num_movimientos_escape`` movimientos aleatorios.

          - Si NO estaba en ``historial``:

              * Lo registra como nueva entrada.
              * Incrementa ``iter_sin_repeticion``.
              * Si ``iter_sin_repeticion >= iter_sin_repeticion_para_reducir``,
                reduce el tenure (acotado por ``tabu_tenure_min``) y
                reinicia el contador.

       h. Actualiza el mejor global si corresponde.

    3. **Terminación**: devuelve el resultado inmutable con la mejor
       solución encontrada y todas las estadísticas (incluyendo las
       métricas reactivas: historial de tenure, número de escapes, etc.).

    Decisiones de diseño documentadas
    ---------------------------------
    - **Hash de la solución**: tupla ordenada de tuplas de etiquetas
      (orden global de rutas canonicalizado, orden interno preservado).
      Ver docstring de ``_hash_solucion``.
    - **Limpieza tras escape**: se BORRAN tanto la lista tabú como el
      historial de soluciones. La justificación es que tras un escape
      grande (varios movimientos aleatorios) estamos en una región
      potencialmente lejana; las prohibiciones acumuladas no son
      pertinentes y queremos volver a detectar ciclos en la nueva zona
      desde cero.
    - **Reescalado de la deque**: como ``deque.maxlen`` es de solo lectura,
      cada cambio de tenure RECONSTRUYE la deque (función auxiliar
      ``_ajustar_maxlen_deque``) preservando los elementos más recientes.
    - **Sesgo inter/intra durante el ESCAPE**: el mecanismo de escape NO
      aplica el sesgo (usa selección uniforme sobre la lista completa de
      operadores). Justificación: el escape ya constituye en sí mismo una
      forma fuerte de diversificación; añadir el sesgo lo haría redundante
      y potencialmente sesgaría la elección de la "zona de aterrizaje".
      El sesgo sí actúa, en cambio, en TODAS las iteraciones del bucle
      principal de búsqueda guiada, exactamente como en SA y TS simple.

    Parámetros de sesgo inter/intra-ruta (compartidos con SA y TS simple)
    ---------------------------------------------------------------------
    alpha_inter:
        Probabilidad de elegir el GRUPO de operadores inter-ruta cuando la
        solución ACTUAL del bucle principal viola capacidad. La detección de
        violación se hace por iteración (``viol_actual > 1e-12``). Valor por
        defecto = el mismo que usa SA (``0.8``).
    p_inter:
        Probabilidad de elegir el GRUPO de operadores inter-ruta cuando la
        solución ACTUAL es factible. Mantiene una dosis constante de
        exploración inter-ruta para escapar de mínimos locales intra-ruta.
        Valor por defecto = el mismo que usa SA (``0.6``).

    Mecanismo (idéntico al SA del proyecto y al TS simple):
        - En cada iteración del bucle principal se computa ``viol_actual``
          (exceso de capacidad de la solución actual).
        - Se invoca ``seleccionar_grupo_operadores_inter_intra`` que realiza
          UN único ``rng.random()`` con umbral ``alpha_inter`` (si hay
          violación) o ``p_inter`` (si es factible).
        - El grupo elegido se pasa a ``generar_vecino`` mediante
          ``operadores=`` con ``pesos_operadores=None`` (selección uniforme
          dentro del grupo).

    Returns
    -------
    BusquedaTabuReactivaResult
        Objeto inmutable con la mejor solución, métricas de calidad,
        tiempos, estadísticas de operadores, historial de costos y tenures,
        y todas las métricas específicas del comportamiento reactivo.
    """
    # ----------------------------------------------------------
    # 1) Validación temprana de los parámetros que no dependen de n.
    # ----------------------------------------------------------
    if not (factor_aumento > 1.0):
        raise ValueError("factor_aumento debe ser estrictamente > 1.0.")
    if not (0.0 < factor_reduccion < 1.0):
        raise ValueError("factor_reduccion debe estar en (0, 1).")
    if umbral_repeticiones_escape < 2:
        raise ValueError("umbral_repeticiones_escape debe ser >= 2.")
    # Resolución de defaults del sesgo inter/intra: si el usuario pasa None,
    # usamos EXACTAMENTE los mismos valores que SA del proyecto (0.8 y 0.6).
    # Centralizar los defaults aquí (en lugar de en la firma) preserva la
    # comparabilidad de los tres algoritmos: cualquier ajuste futuro en el SA
    # debe replicarse aquí y en busqueda_tabu_simple para mantener la igualdad.
    alpha_inter_eff: float = 0.8 if alpha_inter is None else float(alpha_inter)
    p_inter_eff: float = 0.6 if p_inter is None else float(p_inter)
    # Validaciones de los parámetros del sesgo: deben quedar en [0, 1] para
    # tener interpretación probabilística válida. Aceptamos 0 (desactiva ese
    # modo) y 1 (siempre inter en ese estado) como extremos legítimos.
    if not (0.0 <= alpha_inter_eff <= 1.0):
        raise ValueError("alpha_inter debe estar en [0, 1].")
    if not (0.0 <= p_inter_eff <= 1.0):
        raise ValueError("p_inter debe estar en [0, 1].")

    # ----------------------------------------------------------
    # 2) Generador aleatorio reproducible y marca de tiempo.
    # ----------------------------------------------------------
    rng = random.Random(semilla)
    t0 = time.perf_counter()

    # ----------------------------------------------------------
    # 3) Construcción del contexto de evaluación rápida.
    #    Precomputa la matriz Dijkstra densa y los arrays por id de tarea,
    #    amortizando el costo en todas las llamadas a costo_rapido / costo_lote_ids.
    # ----------------------------------------------------------
    ctx = construir_contexto_para_corrida(
        data,
        G,
        nombre_instancia=nombre_instancia if nombre_instancia != "instancia" else None,
        usar_gpu=usar_gpu,
        root=root,
    )

    # ----------------------------------------------------------
    # 4) Cálculo de parámetros instance-aware.
    #    n = número de tareas requeridas (longitud del vector u_arr del ctx).
    # ----------------------------------------------------------
    n = max(1, int(len(ctx.encoding.id_to_label)))  # robusto frente a instancias muy pequeñas
    sqrt_n = math.sqrt(n)
    # Resolución de cada parámetro: si el usuario pasó valor explícito, lo respetamos;
    # si pasó None, calculamos el valor instance-aware.
    iter_max_eff = int(iteraciones_max) if iteraciones_max is not None else max(50, 20 * n)
    max_sin_mejora_eff = int(max_iter_sin_mejora) if max_iter_sin_mejora is not None else max(20, 5 * n)
    tam_vec_eff = int(tam_vecindario) if tam_vecindario is not None else max(20, 2 * n)
    # Tenure inicial: regla clásica sqrt(n) acotada inferior por 3.
    tenure_ini_eff = int(tabu_tenure_inicial) if tabu_tenure_inicial is not None else max(3, round(sqrt_n))
    # Cota inferior: tenure mínimo razonable (3 evita lista tabú efímera).
    tenure_min_eff = int(tabu_tenure_min) if tabu_tenure_min is not None else 3
    # Cota superior: 3·sqrt(n) acotado inferior por 15 para que el rango sea útil
    # incluso en instancias muy pequeñas.
    # ¿POR QUÉ subimos el piso de 10 a 15?
    # En instancias muy pequeñas como ``gdb1`` (n ≈ 11), la regla 3·sqrt(n) da
    # ≈ 10. Con el piso anterior en 10, el rango efectivo del mecanismo
    # reactivo era simplemente [3, 10] (7 unidades de margen). Ese rango es
    # demasiado estrecho: el tenure satura su cota superior tras muy pocos
    # ciclos detectados, y el algoritmo "se queda sin perilla" para
    # diferenciarse del TS clásico. Subir el piso a 15 garantiza que incluso
    # las instancias minúsculas dispongan de un rango [3, 15] (12 unidades),
    # suficiente para que el aumento/reducción dinámica produzca cambios
    # visibles. Para instancias medianas/grandes la fórmula 3·sqrt(n) ya
    # supera 15 con holgura, así que ahí el cambio es inocuo.
    tenure_max_eff = int(tabu_tenure_max) if tabu_tenure_max is not None else max(15, round(3.0 * sqrt_n))
    # Paciencia para reducir el tenure: 2·sqrt(n), mínimo 5 para que no se
    # reduzca demasiado agresivamente en instancias pequeñas.
    iter_paciencia_eff = (
        int(iter_sin_repeticion_para_reducir)
        if iter_sin_repeticion_para_reducir is not None
        else max(5, round(2.0 * sqrt_n))
    )
    # Tamaño del escape: máximo entre 3 y n/10 (regla de pulgar).
    num_mov_escape_eff = int(num_movimientos_escape) if num_movimientos_escape is not None else max(3, n // 10)

    # ----------------------------------------------------------
    # 5) Validaciones cruzadas (después de resolver defaults).
    # ----------------------------------------------------------
    if iter_max_eff <= 0:
        raise ValueError("iteraciones_max debe ser > 0.")
    if max_sin_mejora_eff <= 0:
        raise ValueError("max_iter_sin_mejora debe ser > 0.")
    if tam_vec_eff <= 0:
        raise ValueError("tam_vecindario debe ser > 0.")
    if tenure_min_eff < 1:
        raise ValueError("tabu_tenure_min debe ser >= 1.")
    if tenure_max_eff < tenure_min_eff:
        raise ValueError("tabu_tenure_max debe ser >= tabu_tenure_min.")
    if not (tenure_min_eff <= tenure_ini_eff <= tenure_max_eff):
        # Si el usuario configura un inicial fuera del rango, lo "aplastamos"
        # dentro del intervalo y avisamos (no fallar duramente: es un caso recuperable).
        tenure_ini_eff = max(tenure_min_eff, min(tenure_max_eff, tenure_ini_eff))
    if iter_paciencia_eff <= 0:
        raise ValueError("iter_sin_repeticion_para_reducir debe ser > 0.")
    if num_mov_escape_eff < 1:
        raise ValueError("num_movimientos_escape debe ser >= 1.")

    # ----------------------------------------------------------
    # 6) Selección de la mejor solución inicial (idéntico al TS simple).
    #    Misma rutina, mismo flag de penalización, mismos criterios.
    # ----------------------------------------------------------
    sel_ini = seleccionar_mejor_inicial_rapido(
        inicial_obj,
        ctx,
        usar_penalizacion_capacidad=True,
        lambda_capacidad=None,
    )
    sol_ref = sel_ini.solucion          # solución de referencia para medir mejora final
    costo_ref = sel_ini.costo_puro      # costo inicial de referencia (sin penalización)

    # La solución "actual" es la que RTS modifica en cada iteración.
    sol_actual = copiar_solucion_labels(sol_ref)
    costo_actual = costo_ref
    viol_actual = sel_ini.violacion_capacidad

    # Mejor global registrado: la mejor solución vista durante toda la búsqueda.
    mejor_costo = float(costo_ref)
    mejor_sol = copiar_solucion_labels(sol_ref)
    mejor_factible = viol_actual < 1e-12

    # Configuración del encoding para el backend de IDs (evaluación vectorizada).
    encoding = ctx.encoding if backend_vecindario == "ids" else None
    if backend_vecindario == "ids" and encoding is None:
        encoding = build_search_encoding(data)

    # ----------------------------------------------------------
    # 7) ESTRUCTURAS REACTIVAS.
    # ----------------------------------------------------------
    # 7a) Lista tabú FIFO de longitud DINÁMICA (arranca en tenure_ini_eff).
    # Se mantiene un set sincronizado para lookup O(1).
    tenure_actual = int(tenure_ini_eff)
    lista_tabu: deque[tuple[Any, ...]] = deque(maxlen=tenure_actual)
    set_tabu: set[tuple[Any, ...]] = set()
    # --- MICROOPTIMIZACIÓN: Counter paralelo para conteo O(1) de duplicados ---
    # Antes, al hacer descarte por capacidad llena, el código consultaba
    # ``list(lista_tabu).count(clave_descartada) == 1`` para decidir si la
    # clave también debía salir del ``set_tabu``. Esa cuenta lineal era
    # O(tenure_actual) por iteración y, aunque pequeña, se ejecuta dentro del
    # hot path. Mantenemos un Counter ``conteo_tabu`` sincronizado con la
    # deque para responder en O(1) cuántas copias de cada clave permanecen.
    # Reglas de mantenimiento (deben respetarse en TODOS los puntos que
    # modifican la deque: append normal, _ajustar_maxlen_deque, limpieza tras
    # escape):
    #   - al insertar una clave: ``conteo_tabu[clave] += 1``
    #   - al descartar una copia: ``conteo_tabu[clave] -= 1`` y, si la cuenta
    #     llega a 0, eliminamos la entrada del Counter y la clave del set.
    #   - al limpiar todo (escape): ``conteo_tabu.clear()`` además del
    #     ``lista_tabu.clear()`` y ``set_tabu.clear()`` existentes.
    conteo_tabu: Counter[tuple[Any, ...]] = Counter()

    # 7b) Historial de soluciones visitadas: dict[hash_solucion, {ultima_vista, veces_vista}].
    # Almacenamos un diccionario mutable como valor para poder actualizarlo in-place
    # sin copiar la entrada en cada repetición.
    historial: dict[int, dict[str, int]] = {}
    # Registramos la solución inicial como ya visitada (iteración -1 por convención:
    # "antes de empezar el bucle"). Esto permite que un retorno inmediato a la
    # solución inicial cuente como repetición.
    historial[_hash_solucion(sol_actual)] = {"ultima_vista": -1, "veces_vista": 1}

    # 7c) Contador local de iteraciones consecutivas SIN detectar repetición.
    # Cuando alcance iter_paciencia_eff, dispara la reducción de tenure.
    iter_sin_repeticion = 0

    # ----------------------------------------------------------
    # 8) Contadores y estructuras de reporte.
    # ----------------------------------------------------------
    vecinos_evaluados = 0
    mejoras = 0
    aspiraciones = 0
    iteraciones_todos_tabu = 0
    iter_sin_mejora = 0
    # Contador de kicks (perturbaciones inter-ruta) aplicados durante la corrida.
    # Solo se incrementa si ``max_iter_sin_mejora_kick`` es != None y el umbral
    # se alcanza. Se reporta en el dataclass y en el CSV.
    n_resets_kick: int = 0
    iter_mejor = 0
    ultimo_mov_aceptado: MovimientoVecindario | None = None
    historial_best: list[float] = []
    historial_tenure: list[int] = []
    contador = ContadorOperadores()

    # Métricas específicas de RTS.
    num_repeticiones = 0     # total de repeticiones detectadas en la corrida
    num_escapes = 0          # cuántas veces se disparó el escape
    num_aumentos = 0         # cuántas veces creció el tenure
    num_reducciones = 0      # cuántas veces decreció el tenure
    suma_tenure = 0          # acumulador para promedio
    tenure_max_obs = tenure_actual
    tenure_min_obs = tenure_actual

    # Etiqueta del depósito para los operadores de vecindario.
    md_op = marcador_depot_etiqueta or ctx.marcador_depot
    # Lista de operadores (necesaria para generar_vecino y para el escape aleatorio).
    ops_list = list(operadores)
    # Precomputamos las particiones intra/inter a partir de la lista de
    # operadores activos. Estas particiones se pasan al helper compartido con
    # SA y TS simple en cada iteración del bucle principal, evitando
    # reconstruirlas dentro del bucle. El escape sigue usando ``ops_list``
    # (lista completa, sin sesgo) por decisión de diseño explicada en el
    # docstring.
    ops_intra_list = [op for op in ops_list if op in OPERADORES_INTRA]
    ops_inter_list = [op for op in ops_list if op in OPERADORES_INTER]
    ops_fallback_list = list(ops_list)
    # Contador de iteraciones del bucle principal que entraron al modo sesgado
    # por violación de capacidad. Una iteración cuenta como "con violación" si
    # la solución actual al INICIO de esa iteración tenía ``viol_actual > 1e-12``
    # (criterio idéntico al del SA). NO contamos aquí las ráfagas de escape.
    iteraciones_con_violacion = 0

    # ============================================================
    # === BUCLE PRINCIPAL DE REACTIVE TABU SEARCH ================
    # ============================================================
    # Iteramos hasta cumplir alguno de los dos criterios de parada
    # (idénticos a los del TS simple para mantener la comparabilidad).
    iteracion = 0
    while iteracion < iter_max_eff:
        # Criterio de estancamiento: si llevamos demasiadas iteraciones
        # consecutivas sin mejorar el mejor global, terminamos.
        if iter_sin_mejora >= max_sin_mejora_eff:
            break

        # Registramos métricas al INICIO de la iteración.
        if guardar_historial:
            historial_best.append(mejor_costo)
            historial_tenure.append(tenure_actual)
        # Acumulador para el promedio de tenure (independiente de guardar_historial).
        suma_tenure += tenure_actual
        # Actualizamos el max/min observado (también fuera de guardar_historial).
        if tenure_actual > tenure_max_obs:
            tenure_max_obs = tenure_actual
        if tenure_actual < tenure_min_obs:
            tenure_min_obs = tenure_actual

        # --- Paso 1: Generación del lote de vecinos ---
        # Selección de operador en DOS pasos (mismo mecanismo que SA y TS simple):
        #   (a) seleccionar_grupo_operadores_inter_intra hace UN rng.random()
        #       y decide si esta iteración usa el grupo INTER-RUTA o el grupo
        #       INTRA-RUTA, con umbral alpha_inter (si hay violación) o p_inter
        #       (si la solución es factible).
        #   (b) generar_vecino selecciona UN operador dentro del grupo elegido
        #       (selección uniforme, ``pesos_operadores=None``).
        # La decisión se realiza UNA vez por iteración del bucle principal y
        # todo el lote de tam_vec_eff vecinos comparte el mismo modo de sesgo.
        # Es lo más coherente con el "best-improvement sobre un lote homogéneo"
        # del TS clásico: si el lote mezclara modos, el best-improvement
        # estaría comparando manzanas con peras.
        grupo_ops, _hubo_viol = seleccionar_grupo_operadores_inter_intra(
            rng,
            viol_actual,
            ops_intra_list,
            ops_inter_list,
            ops_fallback_list,
            alpha_inter=alpha_inter_eff,
            p_inter=p_inter_eff,
        )
        # Si la solución actual viola capacidad en esta iteración, lo contamos.
        if _hubo_viol:
            iteraciones_con_violacion += 1
        vecinos: list[list[list[str]]] = []
        movimientos: list[MovimientoVecindario] = []
        for _ in range(tam_vec_eff):
            vecino, mov = generar_vecino(
                sol_actual,
                rng=rng,
                operadores=grupo_ops,         # grupo INTER o INTRA elegido arriba
                pesos_operadores=None,
                marcador_depot=md_op,
                devolver_con_deposito=True,
                usar_gpu=usar_gpu,
                backend=backend_vecindario,
                encoding=encoding,
            )
            vecinos.append(vecino)
            movimientos.append(mov)
            contador.proponer(mov.operador)

        # --- Paso 2: Evaluación en lote ---
        sols_ids = [encode_solution(v, ctx.encoding) for v in vecinos]
        costos_np = costo_lote_ids(sols_ids, ctx)
        vecinos_evaluados += len(vecinos)

        # --- Paso 3: Selección del mejor vecino no-tabú con aspiración clásica ---
        mejor_admisible_idx = -1
        mejor_admisible_costo = float("inf")
        mejor_total_idx = 0
        mejor_total_costo = float("inf")
        aspiracion_en_iter = False

        for idx in range(len(costos_np)):
            c_v = float(costos_np[idx])
            # Mejor total (incluye tabú): fallback si todos resultan prohibidos.
            if c_v < mejor_total_costo:
                mejor_total_costo = c_v
                mejor_total_idx = idx
            # ¿Este movimiento está en la lista tabú?
            key = _clave_tabu(movimientos[idx])
            es_tabu = key in set_tabu
            # Aspiración: ¿mejora estrictamente al mejor global?
            aspiracion = c_v < mejor_costo - 1e-15
            if es_tabu and not aspiracion:
                continue
            if c_v < mejor_admisible_costo:
                mejor_admisible_costo = c_v
                mejor_admisible_idx = idx
                if es_tabu and aspiracion:
                    aspiracion_en_iter = True

        # --- Paso 4: Elección del movimiento a ejecutar ---
        if mejor_admisible_idx == -1:
            # Caso degenerado: todos los vecinos eran tabú y ninguno aspira.
            elegido_idx = mejor_total_idx
            iteraciones_todos_tabu += 1
        else:
            elegido_idx = mejor_admisible_idx
            if aspiracion_en_iter:
                aspiraciones += 1

        # Avanzamos al vecino elegido SIEMPRE (idea central de TS).
        sol_actual = vecinos[elegido_idx]
        costo_actual = float(costos_np[elegido_idx])
        viol_actual = exceso_capacidad_rapido(sol_actual, ctx)
        ultimo_mov_aceptado = movimientos[elegido_idx]
        contador.aceptar(ultimo_mov_aceptado.operador)

        # --- Paso 5: Actualización de la lista tabú FIFO ---
        # Insertamos la clave del movimiento ejecutado en la deque actual.
        # Si la deque estaba llena, el elemento más antiguo se descarta
        # automáticamente; mantenemos el set y el Counter sincronizados.
        nueva_clave = _clave_tabu(ultimo_mov_aceptado)
        if len(lista_tabu) == lista_tabu.maxlen:
            clave_descartada = lista_tabu[0]
            # Decrementamos el conteo de la clave descartada en O(1)
            # (reemplaza el antiguo ``list(lista_tabu).count(...)`` que era
            # O(tenure_actual)). Si tras decrementar el conteo llega a 0, esa
            # clave deja de existir en la deque y, por tanto, debe salir
            # también del set tabú; además limpiamos la entrada del Counter
            # para que no acumule claves "fantasma".
            conteo_tabu[clave_descartada] -= 1
            if conteo_tabu[clave_descartada] == 0:
                del conteo_tabu[clave_descartada]
                set_tabu.discard(clave_descartada)
        lista_tabu.append(nueva_clave)
        set_tabu.add(nueva_clave)
        # Incrementamos el conteo de la clave recién insertada. Si ya estaba
        # presente (duplicado), sube de 1 a 2 sin afectar el set.
        conteo_tabu[nueva_clave] += 1

        # --- Paso 6: Actualización del mejor global ---
        if costo_actual < mejor_costo - 1e-15:
            mejor_costo = costo_actual
            mejor_sol = copiar_solucion_labels(sol_actual)
            mejor_factible = viol_actual < 1e-12
            iter_mejor = iteracion
            mejoras += 1
            contador.registrar_mejora(ultimo_mov_aceptado.operador)
            iter_sin_mejora = 0
        else:
            iter_sin_mejora += 1

            # --- Kick por estancamiento global (strict_intra_inter_20260524) ---
            # Cuando la MH lleva demasiadas iteraciones sin mejorar el mejor
            # global, se aplica una perturbación inter-ruta disruptiva y se
            # reinicia el contador. Si se alcanza max_resets, se para. El kick
            # se aplica DESPUÉS de incrementar iter_sin_mejora para que la
            # decisión use el valor recién actualizado.
            if (max_iter_sin_mejora_kick is not None
                    and iter_sin_mejora >= max_iter_sin_mejora_kick):
                # Import diferido: solo se carga si la corrida activa el kick.
                from metacarp.strict_intra_inter_20260524 import aplicar_kick_labels
                sol_actual = aplicar_kick_labels(
                    sol_actual, rng, md_op, encoding=encoding
                )
                # Tras el kick recalculamos costo y violación para que el resto
                # de la iteración (paso 7 reactivo) tome decisiones coherentes.
                viol_actual = float(exceso_capacidad_rapido(sol_actual, ctx))
                costo_actual = float(costo_rapido(sol_actual, ctx))
                iter_sin_mejora = 0
                n_resets_kick += 1
                if max_resets is not None and n_resets_kick >= max_resets:
                    # Cota dura: incrementamos iteracion para reportar el total
                    # correcto y rompemos el bucle externo (el paso 7 reactivo
                    # se omite en esta iteración terminal).
                    iteracion += 1
                    break

        # =====================================================
        # === Paso 7: MECANISMO REACTIVO DEL TENURE ============
        # =====================================================
        # 7a) Calculamos el hash canónico de la solución actual.
        h = _hash_solucion(sol_actual)
        # 7b) ¿Es una repetición? Comparamos con el historial.
        if h in historial:
            # Sí: ya habíamos visitado esta solución antes.
            num_repeticiones += 1
            historial[h]["veces_vista"] += 1
            historial[h]["ultima_vista"] = iteracion
            # Resetear paciencia: detectamos ciclo, no es momento de relajar.
            iter_sin_repeticion = 0
            veces = historial[h]["veces_vista"]

            # AUMENTAR el tenure (mecanismo reactivo) — siempre que detectemos
            # repetición, aplicamos el factor de aumento. Si ya estamos en el
            # tope, el min() lo deja igual y no cuenta como aumento real.
            nuevo_tenure = min(tenure_max_eff, max(tenure_actual + 1, round(tenure_actual * factor_aumento)))
            # Asegurar al menos +1 cuando hay margen, porque
            # round(3 * 1.2) = 4 está bien pero round(3 * 1.05) podría
            # quedar igual a 3 y nunca subir; por eso forzamos +1 si hay sitio.
            if nuevo_tenure > tenure_actual:
                tenure_actual = nuevo_tenure
                num_aumentos += 1
                # Pasamos también el ``conteo_tabu`` para que el reescalado
                # mantenga su invariante (cuentas correctas tras posibles
                # descartes por truncamiento al reducir maxlen). En el caso de
                # AUMENTO no se descarta nada, pero el helper igual lo acepta
                # como argumento por simetría con la rama de reducción.
                lista_tabu = _ajustar_maxlen_deque(lista_tabu, set_tabu, tenure_actual, conteo_tabu)

            # ESCAPE si la solución se ha repetido lo suficiente: aplicamos
            # movimientos aleatorios consecutivos para saltar lejos.
            if veces >= umbral_repeticiones_escape:
                num_escapes += 1
                # Realizamos num_mov_escape_eff movimientos aleatorios encadenados.
                # Durante el escape NO miramos la lista tabú: el objetivo es
                # justamente romper con la zona actual.
                for _ in range(num_mov_escape_eff):
                    vecino_esc, mov_esc = generar_vecino(
                        sol_actual,
                        rng=rng,
                        operadores=ops_list,
                        pesos_operadores=None,
                        marcador_depot=md_op,
                        devolver_con_deposito=True,
                        usar_gpu=usar_gpu,
                        backend=backend_vecindario,
                        encoding=encoding,
                    )
                    sol_actual = vecino_esc
                    contador.proponer(mov_esc.operador)
                    contador.aceptar(mov_esc.operador)
                # Recalculamos costo y violación tras el escape para mantener
                # consistencia en las estadísticas que se consultan más abajo.
                costo_actual = float(costo_rapido(sol_actual, ctx))
                viol_actual = exceso_capacidad_rapido(sol_actual, ctx)
                vecinos_evaluados += num_mov_escape_eff
                # Limpieza tras escape (decisión documentada en el docstring):
                # vaciamos lista tabú e historial. Comenzamos desde "cero" en
                # la nueva región del espacio.
                lista_tabu.clear()
                set_tabu.clear()
                # También vaciamos el Counter ``conteo_tabu`` para mantener su
                # invariante "refleja la deque actual". Si lo olvidásemos, las
                # cuentas quedarían colgadas y los próximos descartes harían
                # ``[clave] -= 1`` desde valores irrealmente altos, dejando
                # claves "fantasma" en el set.
                conteo_tabu.clear()
                historial.clear()
                # Reseteamos también la paciencia (recién aterrizamos en una zona nueva).
                iter_sin_repeticion = 0
                # Registramos la nueva solución en historial para detectar ciclos
                # en la zona nueva (si volviera a aparecer rápido).
                historial[_hash_solucion(sol_actual)] = {
                    "ultima_vista": iteracion,
                    "veces_vista": 1,
                }
                # Si el escape produjo mejora del mejor global, la registramos.
                # Esto es importante porque el escape puede aterrizar en un
                # vecindario muy bueno aunque no lo hayamos buscado.
                if costo_actual < mejor_costo - 1e-15:
                    mejor_costo = costo_actual
                    mejor_sol = copiar_solucion_labels(sol_actual)
                    mejor_factible = viol_actual < 1e-12
                    iter_mejor = iteracion
                    mejoras += 1
                # --- REFINAMIENTO: reset INCONDICIONAL de iter_sin_mejora ---
                # ¿POR QUÉ resetear SIEMPRE tras un escape (no solo cuando
                # mejora el mejor global)?
                # Un escape es una diversificación FUERTE: aplicamos varios
                # movimientos aleatorios consecutivos y aterrizamos en una zona
                # potencialmente lejana del espacio de búsqueda. Tras aterrizar
                # necesitamos darle al algoritmo una "ventana de gracia" para
                # explotar esa nueva región mediante el bucle normal de TS:
                # localizar mínimos locales, construir lista tabú nueva,
                # etc. Si NO reseteamos ``iter_sin_mejora``, un escape tardío
                # (cuando el contador ya está cerca de ``max_iter_sin_mejora``)
                # apenas le daría unas pocas iteraciones a la nueva zona antes
                # de la parada por estancamiento, desperdiciando el coste del
                # propio escape. Con el reset incondicional garantizamos que
                # cada escape disfrute de un presupuesto completo de
                # iteraciones para explotar la nueva región; si tras ese
                # presupuesto sigue sin mejorar, el criterio de parada
                # eventualmente disparará como debe.
                iter_sin_mejora = 0
        else:
            # 7c) Solución NUEVA (no estaba en historial).
            historial[h] = {"ultima_vista": iteracion, "veces_vista": 1}
            iter_sin_repeticion += 1

            # REDUCIR tenure si llevamos suficientes iteraciones sin repeticiones:
            # el algoritmo está explorando libremente, no necesita tanta presión tabú.
            if iter_sin_repeticion >= iter_paciencia_eff:
                # max() con +/- 1 para asegurar que la reducción sea efectiva
                # incluso cuando round(tenure * 0.9) = tenure (instancias pequeñas).
                nuevo_tenure = max(tenure_min_eff, min(tenure_actual - 1, round(tenure_actual * factor_reduccion)))
                if nuevo_tenure < tenure_actual:
                    tenure_actual = nuevo_tenure
                    num_reducciones += 1
                    # En reducción el helper SÍ descarta elementos antiguos:
                    # pasarle el ``conteo_tabu`` es imprescindible para que las
                    # cuentas reflejen la deque truncada y los descartes
                    # posteriores no rompan el invariante.
                    lista_tabu = _ajustar_maxlen_deque(lista_tabu, set_tabu, tenure_actual, conteo_tabu)
                # Reiniciamos el contador (independientemente de si se redujo
                # efectivamente o ya estábamos en el mínimo). Así evitamos
                # acumular paciencia indefinidamente cuando ya no hay margen.
                iter_sin_repeticion = 0

        # Avanzamos a la siguiente iteración.
        iteracion += 1

    # ============================================================
    # === FIN DEL BUCLE PRINCIPAL ================================
    # ============================================================
    elapsed = time.perf_counter() - t0
    _gap_descartado, mejora_abs, mejora_pct = calcular_metricas_gap(costo_ref, mejor_costo)
    # Promedio de tenure: si no se ejecutó ninguna iteración (caso patológico),
    # usamos el tenure inicial para evitar división por cero.
    tenure_prom = (suma_tenure / iteracion) if iteracion > 0 else float(tenure_ini_eff)

    # ----------------------------------------------------------
    # Guardado en CSV (opcional). Misma convención del TS simple +
    # columnas adicionales con las métricas reactivas.
    # ----------------------------------------------------------
    archivo_csv: str | None = None
    if guardar_csv:
        ruta = ruta_csv or f"resultados_busqueda_tabu_reactiva_{nombre_instancia}.csv"
        detalle_txt, costo_total_reporte = generar_reporte_detallado(
            mejor_sol, data, G,
            nombre_instancia=nombre_instancia,
            marcador_depot_etiqueta=marcador_depot_etiqueta,
            usar_gpu=False,
        )
        _bks = resumen_bks_csv(data, mejor_costo)
        # IMPORTANTE: la fila NO incluye id_corrida ni config_id (convención del
        # proyecto). Si alguien pasa esos kwargs por compatibilidad, se ignoran
        # vía **_ignorado_kwargs en la firma de la función.
        fila = {
            "metaheuristica": "tabu_reactiva",  # etiqueta distintiva para análisis
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
            # Parámetros efectivos del RTS (los que realmente se usaron tras
            # resolver los defaults instance-aware).
            "iteraciones_max": iter_max_eff,
            "max_iter_sin_mejora": max_sin_mejora_eff,
            "tam_vecindario": tam_vec_eff,
            # ``tabu_tenure`` se conserva como columna para que el CSV sea
            # COMPATIBLE con el del TS simple. Reportamos aquí el tenure
            # INICIAL aplicado (los detalles reactivos van en columnas extra).
            "tabu_tenure": tenure_ini_eff,
            # Contador de operadores (4 categorías × 9 operadores = 36 columnas).
            **contador.resumen_csv(),
            # Estadísticas de la corrida (mismas que TS simple).
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
            # ---- Columnas ESPECÍFICAS de RTS ----
            "tenure_inicial": tenure_ini_eff,
            "tenure_min_aplicado": tenure_min_eff,
            "tenure_max_aplicado": tenure_max_eff,
            "tenure_final": tenure_actual,
            "tenure_promedio": tenure_prom,
            "tenure_max_alcanzado": tenure_max_obs,
            "tenure_min_alcanzado": tenure_min_obs,
            "factor_aumento": factor_aumento,
            "factor_reduccion": factor_reduccion,
            "iter_sin_repeticion_para_reducir": iter_paciencia_eff,
            "umbral_repeticiones_escape": umbral_repeticiones_escape,
            "num_movimientos_escape": num_mov_escape_eff,
            "num_repeticiones_detectadas": num_repeticiones,
            "num_escapes_realizados": num_escapes,
            "num_aumentos_tenure": num_aumentos,
            "num_reducciones_tenure": num_reducciones,
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
        }
        archivo_csv = guardar_resultado_csv(fila=fila, ruta_csv=ruta)

    # ----------------------------------------------------------
    # Construimos y retornamos el resultado inmutable.
    # ----------------------------------------------------------
    return BusquedaTabuReactivaResult(
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
        historial_tenure=historial_tenure,
        ultimo_movimiento_aceptado=ultimo_mov_aceptado,
        operadores_propuestos=contador.como_dict_ordenado(contador.propuestos),
        operadores_aceptados=contador.como_dict_ordenado(contador.aceptados),
        operadores_mejoraron=contador.como_dict_ordenado(contador.mejoraron),
        operadores_trayectoria_mejor=contador.como_dict_ordenado(contador.trayectoria_mejor),
        mejor_solucion_factible_final=mejor_factible,
        archivo_csv=archivo_csv,
        # Métricas específicas de RTS.
        tenure_final=tenure_actual,
        tenure_promedio=tenure_prom,
        tenure_max_alcanzado=tenure_max_obs,
        tenure_min_alcanzado=tenure_min_obs,
        num_repeticiones_detectadas=num_repeticiones,
        num_escapes_realizados=num_escapes,
        num_aumentos_tenure=num_aumentos,
        num_reducciones_tenure=num_reducciones,
        tenure_inicial_aplicado=tenure_ini_eff,
        tenure_min_aplicado=tenure_min_eff,
        tenure_max_aplicado=tenure_max_eff,
        # Trazabilidad del sesgo inter/intra aplicado en esta corrida.
        alpha_inter_aplicado=alpha_inter_eff,
        p_inter_aplicado=p_inter_eff,
        iteraciones_con_violacion=iteraciones_con_violacion,
        # Kicks aplicados (variante experimental strict_intra_inter_20260524).
        n_resets_kick=n_resets_kick,
    )


def busqueda_tabu_reactiva_desde_instancia(
    nombre_instancia: str,
    *,
    root: str | None = None,
    # ---------- Criterios de parada ----------
    iteraciones_max: int | None = None,
    max_iter_sin_mejora: int | None = None,
    # ---------- Vecindario ----------
    tam_vecindario: int | None = None,
    # ---------- Parámetros reactivos del tenure ----------
    tabu_tenure_inicial: int | None = None,
    tabu_tenure_min: int | None = None,
    tabu_tenure_max: int | None = None,
    factor_aumento: float = 1.2,
    factor_reduccion: float = 0.9,
    iter_sin_repeticion_para_reducir: int | None = None,
    # ---------- Parámetros de escape ----------
    umbral_repeticiones_escape: int = 3,
    num_movimientos_escape: int | None = None,
    # ---------- Configuración general ----------
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
    # Kick reactivo (variante experimental strict_intra_inter_20260524).
    max_iter_sin_mejora_kick: int | None = None,
    max_resets: int | None = None,
    **_ignorado_kwargs: object,  # absorbe kwargs heredados (id_corrida, config_id)
) -> BusquedaTabuReactivaResult:
    """
    Función de conveniencia: carga todos los recursos necesarios desde el nombre
    de la instancia y ejecuta Reactive Tabu Search completa.

    Equivalente a llamar manualmente a load_instances + cargar_objeto_gexf +
    cargar_solucion_inicial + busqueda_tabu_reactiva.
    """
    # Cargamos los datos de la instancia (capacidad, demandas, BKS, etc.).
    data = load_instances(nombre_instancia, root=root)
    # Cargamos el grafo de la instancia desde el archivo GEXF.
    G = cargar_objeto_gexf(nombre_instancia, root=root)
    # Cargamos el objeto de solución inicial desde el archivo pickle.
    inicial_obj = cargar_solucion_inicial(nombre_instancia, root=root)
    return busqueda_tabu_reactiva(
        inicial_obj,
        data,
        G,
        iteraciones_max=iteraciones_max,
        max_iter_sin_mejora=max_iter_sin_mejora,
        tam_vecindario=tam_vecindario,
        tabu_tenure_inicial=tabu_tenure_inicial,
        tabu_tenure_min=tabu_tenure_min,
        tabu_tenure_max=tabu_tenure_max,
        factor_aumento=factor_aumento,
        factor_reduccion=factor_reduccion,
        iter_sin_repeticion_para_reducir=iter_sin_repeticion_para_reducir,
        umbral_repeticiones_escape=umbral_repeticiones_escape,
        num_movimientos_escape=num_movimientos_escape,
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
        # Propagamos el mecanismo de kick (variante experimental).
        max_iter_sin_mejora_kick=max_iter_sin_mejora_kick,
        max_resets=max_resets,
    )
