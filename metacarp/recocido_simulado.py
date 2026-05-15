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

Recalentamiento (Reheat)
------------------------
Cuando T baja mucho, el SA se vuelve prácticamente voraz: la probabilidad de
aceptar empeoramientos es casi nula y el algoritmo queda atrapado en un mínimo
local del cual no puede escapar, incluso si el mínimo global está a "una
montaña de distancia". Esto se observa empíricamente en instancias donde el SA
encuentra soluciones de calidad cercanas al óptimo (BKS) pero no logra dar el
último salto: por ejemplo, en ``gdb19`` encuentra costo 63 mientras el BKS es
55 — la brecha es pequeña pero requiere aceptar empeoramientos significativos
cuando T ya está cerca de cero.

El mecanismo de **reheat** (recalentamiento) resuelve este problema reiniciando
parcialmente la temperatura cada vez que se detecta estancamiento. La
analogía metalúrgica es directa: cuando un metal queda atrapado en un mínimo
local de energía (con átomos en imperfecciones cristalinas), los metalúrgicos
lo vuelven a calentar para devolverles movilidad y permitir una nueva
reorganización; al enfriarlo de nuevo, los átomos tienen otra oportunidad de
encontrar la configuración perfecta. En el algoritmo, "calentar" significa
subir T para reaceptar empeoramientos y reexplorar el espacio de soluciones.

Parámetros del reheat:

- ``patience``: número de niveles de temperatura consecutivos sin mejora del
  mejor global antes de activar el reheat. Si ``patience = 0`` el reheat está
  desactivado (comportamiento SA clásico). Valores típicos: 20–100. Cuanto
  más pequeño ``patience``, más agresivo el reheat (más reinicios).

- ``reheat_factor``: fracción de ``T_init_eff`` a la que se sube la temperatura
  al activarse el reheat. Por ejemplo, si ``T_init_eff = 1500`` y
  ``reheat_factor = 0.5``, T salta a 750. Un valor cercano a 1 da un reheat
  muy fuerte (explora casi como al inicio); un valor pequeño da un reheat
  suave (solo unas pocas aceptaciones más). Valores típicos: 0.3–0.7.

Activación del reheat:

1. Al inicio de cada nivel externo se captura ``mejor_costo_antes_nivel``.
2. Al final del nivel, si el mejor costo reportable no mejoró respecto a ese
   valor, se incrementa el contador ``niveles_sin_mejora``.
3. Si ``niveles_sin_mejora >= patience`` (y ``patience > 0``), se sube T a
   ``T_init_eff * reheat_factor``, se reinicia el contador, y se contabiliza
   un reheat más en ``n_reheats``.

El reheat preserva la mejor solución encontrada (``mejor_any_s`` y
``mejor_fact_s`` no se reinician), pero permite escapar del mínimo local desde
la posición actual de búsqueda subiendo la tolerancia a empeoramientos.

Calibración adaptativa desde la instancia
-----------------------------------------
La longitud de la cadena de Markov L y las temperaturas por defecto se
calculan automáticamente desde la instancia (adaptado de Lourenço et al.
para CARP). Sea ``n`` el número de arcos requeridos y ``d_max`` la
distancia máxima en la matriz Dijkstra:

- L = n²              → longitud de la cadena de Markov (iteraciones por
  nivel de temperatura), reemplaza al antiguo ``iteraciones_por_temperatura``.
- T_init = 20 · d_max / n  → temperatura inicial por defecto (si el usuario
  no la especifica). Multiplicador 20 para exploración agresiva al inicio.
- T_end  = 20 · d_max / n² → temperatura mínima por defecto (si el usuario
  no la especifica).

El usuario puede pasar ``temperatura_inicial`` y ``temperatura_minima`` con
valores concretos para sobrescribir el cálculo automático.

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
# NumPy: necesario para la calibración adaptativa (cálculo de d_max sobre la matriz Dijkstra).
import numpy as np

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
    resumen_bks_csv,
    seleccionar_mejor_inicial_rapido,
    solucion_legible_humana,
)
from .vecindarios import (
    MovimientoVecindario,
    OPERADORES_INTRA,
    OPERADORES_INTER,
    OPERADORES_POPULARES,
    generar_vecino,
)

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
    # Número de veces que se activó el recalentamiento durante la búsqueda.
    # Un valor 0 indica que: o bien el reheat estaba desactivado (patience=0),
    # o el algoritmo encontró mejoras con suficiente frecuencia para no requerir
    # ningún reinicio de temperatura. Un valor alto sugiere estancamiento
    # frecuente y puede indicar que conviene revisar la calibración del SA.
    n_reheats: int = 0


def recocido_simulado(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    temperatura_inicial: float | None = None,  # None = calcular automáticamente como 5·d_max/n
    temperatura_minima: float | None = None,   # None = calcular automáticamente como 20·d_max/n²
    alpha: float = 0.95,                       # factor de enfriamiento geométrico (0 < alpha < 1)
    semilla: int | None = None,                # semilla para reproducibilidad
    operadores: Iterable[str] = OPERADORES_POPULARES,  # operadores de vecindario habilitados
    marcador_depot_etiqueta: str | None = None,  # etiqueta del nodo depósito
    usar_gpu: bool = False,                   # flag de GPU (solo para trazabilidad en SA)
    backend_vecindario: Literal["labels", "ids"] = "labels",  # modo de generación de vecinos
    guardar_historial: bool = True,           # si True, guarda historial por nivel de temperatura
    guardar_csv: bool = False,                # si True, escribe resultados en CSV
    ruta_csv: str | None = None,              # ruta del CSV
    nombre_instancia: str = "instancia",     # nombre para el CSV
    repeticion: int | None = None,
    root: str | None = None,
    usar_penalizacion_capacidad: bool = True,  # si True, penaliza violaciones de capacidad
    lambda_capacidad: float | None = None,     # peso λ (None = automático)
    extra_csv: dict[str, object] | None = None,  # columnas adicionales para el CSV
    alpha_inter: float = 0.8,  # fracción de prob. asignada a ops inter-ruta cuando hay violación
    p_inter: float = 0.6,  # fracción fija de probabilidad para operadores inter-ruta (activa incluso en soluciones factibles)
    patience: int = 50,    # niveles sin mejora antes de reheat (0 = desactivado)
    reheat_factor: float = 0.5,  # fracción de T_init_eff a la que se recalienta al activar el reheat
    **_ignorado_kwargs: object,  # absorbe kwargs heredados (p.ej. id_corrida, config_id)
) -> RecocidoSimuladoResult:
    """
    Recocido Simulado clásico para minimizar el costo de soluciones CARP.

    Estructura del algoritmo:
    - Bucle externo: niveles de temperatura (de T_inicial hasta T_minima).
    - Bucle interno: L evaluaciones por nivel, donde L = n² (n = número de
      arcos requeridos de la instancia).
    - En cada evaluación: genera un vecino, decide si aceptarlo (Metropolis).
    - Al finalizar cada nivel: T = alpha × T (enfriamiento geométrico).

    Criterio de parada: T < temperatura_minima.

    Calibración adaptativa: si ``temperatura_inicial`` o ``temperatura_minima``
    son ``None``, se calculan desde la instancia con las fórmulas:
        L       = n²
        T_init  = 5 · d_max / n
        T_end   = 20 · d_max / n²
    donde n = número de tareas (arcos requeridos) y d_max = distancia máxima
    en la matriz Dijkstra.

    Parámetros de sesgo de vecindario:
        alpha_inter: fracción de probabilidad asignada a operadores inter-ruta
            cuando la solución actual viola capacidad. Promueve la reparación
            redistribuyendo tareas entre rutas.
        p_inter: fracción fija de probabilidad asignada a operadores inter-ruta
            cuando la solución actual es factible. Permite seguir explorando
            movimientos inter-ruta para escapar de mínimos locales intra-ruta
            incluso cuando ya no hay violación de capacidad. Se recomienda
            ``alpha_inter >= p_inter``.

    Parámetros de recalentamiento (reheat):
        patience: número de niveles de temperatura consecutivos sin mejora del
            mejor global antes de activar el reheat. Un valor 0 desactiva por
            completo el mecanismo (se obtiene el SA clásico). Valores típicos:
            20–100. Cuanto menor sea, más agresivo es el reheat (más
            reinicios). El contador interno ``niveles_sin_mejora`` se compara
            contra este umbral al final de cada nivel externo.
        reheat_factor: fracción de ``T_init_eff`` (temperatura inicial efectiva
            tras la calibración adaptativa) a la que se sube la temperatura
            cuando se activa el reheat. Por ejemplo, ``reheat_factor=0.5`` con
            ``T_init_eff=1500`` reinicia T a 750. Debe estar en el intervalo
            ``(0, 1]``. Valores típicos: 0.3 a 0.7.
    """
    # Validaciones de parámetros.
    # Solo validamos los valores de temperatura si el usuario los pasó explícitamente.
    if temperatura_inicial is not None and temperatura_inicial <= 0:
        raise ValueError("temperatura_inicial debe ser > 0.")
    if temperatura_minima is not None and temperatura_minima <= 0:
        raise ValueError("temperatura_minima debe ser > 0.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha debe estar en (0, 1).")
    # Validaciones del mecanismo de reheat.
    # patience puede ser 0 (reheat desactivado) o un entero positivo.
    if patience < 0:
        raise ValueError("patience debe ser >= 0 (0 = reheat desactivado).")
    # reheat_factor debe ser una fracción válida de T_init_eff. Un valor de 0
    # equivaldría a apagar el SA (T = 0 nunca aceptaría empeoramientos);
    # valores > 1 subirían T por encima de la temperatura inicial, perdiendo
    # el sentido del "reheat parcial".
    if not (0.0 < reheat_factor <= 1.0):
        raise ValueError("reheat_factor debe estar en (0, 1].")

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

    # Calibración adaptativa desde la instancia (adaptada de Lourenço et al. para CARP).
    # n = número de arcos requeridos; d_max = distancia máxima en la matriz Dijkstra.
    n_tareas = int(len(ctx.u_arr))  # número de tareas (arcos requeridos)
    _dist_finita = ctx.dist[ctx.dist < np.inf]
    d_max = float(_dist_finita.max()) if len(_dist_finita) > 0 else 1.0
    # L = n² iteraciones por nivel (longitud de la cadena de Markov escalada a la instancia).
    L = n_tareas * n_tareas
    # Temperaturas calculadas si el usuario no las especificó.
    T_init_eff = float(temperatura_inicial) if temperatura_inicial is not None else 20.0 * d_max / n_tareas
    T_min_eff  = float(temperatura_minima)  if temperatura_minima  is not None else 20.0 * d_max / (n_tareas * n_tareas)

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
    T = T_init_eff
    iteraciones_totales = 0  # total de vecinos evaluados
    enfriamientos = 0        # número de reducciones de temperatura ejecutadas (solo conteo)
    aceptadas = 0            # vecinos aceptados (mejores + peores)
    mejoras = 0              # veces que el mejor global mejoró
    ultimo_mov_aceptado: MovimientoVecindario | None = None
    historial_best: list[float] = []  # historial del mejor costo por nivel de T
    historial_temp: list[float] = []  # historial de temperaturas por nivel
    contador = ContadorOperadores()
    # --- Estado del mecanismo de recalentamiento (reheat) ---
    # niveles_sin_mejora: contador de niveles consecutivos en los que el mejor
    # costo reportable no mejoró. Se incrementa al final de cada nivel externo
    # si no hubo mejora; se reinicia a 0 cuando sí la hay o cuando se dispara
    # un reheat. Es la variable clave para decidir cuándo recalentar.
    niveles_sin_mejora = 0
    # n_reheats: número total de veces que el reheat se activó durante la
    # corrida. Se reporta en el resultado y, si guardar_csv=True, en el CSV.
    # Sirve para estudiar empíricamente si el algoritmo necesitó muchos
    # reinicios (señal de estancamiento crónico) o casi ninguno (señal de
    # buena calibración inicial).
    n_reheats = 0

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
    # Se repite mientras T no baje del mínimo (criterio de parada puro del SA clásico).
    while T > T_min_eff:
        # Capturamos el mejor costo reportable ANTES del bucle interno.
        # Esta foto del estado se compara, al final del nivel, con el costo
        # reportable tras todas las iteraciones internas: si no mejoró, este
        # nivel cuenta como un "nivel sin mejora" hacia el umbral de patience.
        # Es crítico capturarlo aquí (antes del for interno) y NO después,
        # porque el bucle interno puede actualizar mejor_any_c / mejor_fact_c
        # muchas veces durante L iteraciones.
        mejor_costo_antes_nivel = costo_para_reporte()

        if guardar_historial:
            # Registramos la temperatura y el mejor costo al inicio de este nivel.
            historial_temp.append(T)
            historial_best.append(costo_para_reporte())

        # === BUCLE INTERNO: evaluaciones dentro del nivel de temperatura T ===
        # Selección de operador por lanzamiento de dado:
        # - Se genera un número aleatorio u ∈ [0, 1).
        # - Si p_efectiva > u → se elige al azar entre los operadores inter-ruta.
        # - Si no            → se elige al azar entre los operadores intra-ruta.
        # Cuando hay violación de capacidad se usa alpha_inter (más agresivo que
        # p_inter) como umbral, para favorecer la reparación redistribuyendo
        # demanda entre rutas. El umbral se recalcula cada iteración porque
        # viol_actual puede cambiar tras cada Metropolis.
        _ops_intra = [op for op in operadores if op in OPERADORES_INTRA]
        _ops_inter = [op for op in operadores if op in OPERADORES_INTER]
        for _ in range(L):
            iteraciones_totales += 1

            # Dado de decisión inter/intra: umbral depende de si hay violación.
            # El dado elige el GRUPO; generar_vecino selecciona el operador
            # concreto dentro del grupo y maneja los reintentos internamente.
            # Esto evita que un solo operador inaplicable agote los reintentos.
            p_efectiva = alpha_inter if viol_actual > 1e-12 else p_inter
            if rng.random() < p_efectiva and _ops_inter:
                # Lanzamiento exitoso: grupo inter-ruta completo.
                op_elegido = _ops_inter
            elif _ops_intra:
                # Lanzamiento fallido: grupo intra-ruta completo.
                op_elegido = _ops_intra
            else:
                # Fallback: cualquier operador disponible (caso degenerado).
                op_elegido = list(operadores)

            # Generamos un vecino aleatorio de la solución actual.
            vecino, mov = generar_vecino(
                sol_actual,
                rng=rng,
                operadores=op_elegido,
                pesos_operadores=None,
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

        # --- Mecanismo de recalentamiento (reheat) ---
        # Después del enfriamiento, comparamos el mejor costo reportable
        # actual con el que teníamos al inicio del nivel. Esta comparación
        # detecta si el SA está estancado: si tras L iteraciones internas el
        # mejor global no bajó, el algoritmo no está progresando y, si T ya
        # es pequeña, la probabilidad de escapar mediante Metropolis es casi
        # nula. El reheat sube T para devolverle al algoritmo la capacidad
        # de aceptar empeoramientos y reexplorar.
        if costo_para_reporte() < mejor_costo_antes_nivel - 1e-12:
            # Sí hubo mejora durante el nivel: reiniciamos el contador. El
            # SA está progresando, no necesita reheat por ahora.
            niveles_sin_mejora = 0
        else:
            # No hubo mejora durante el nivel: incrementamos el contador.
            # Cuanto más se acumula, más cerca está el algoritmo de disparar
            # un reheat (si patience > 0).
            niveles_sin_mejora += 1

        # Si se alcanzó el umbral de paciencia, recalentamos. La condición
        # ``patience > 0`` permite desactivar el mecanismo por completo
        # pasando patience=0 (comportamiento SA clásico sin reheat).
        if patience > 0 and niveles_sin_mejora >= patience:
            # Subimos T a una fracción de la temperatura inicial efectiva.
            # No subimos a T_init_eff completo para que el reheat sea
            # "parcial" — exploramos de nuevo, pero no tanto como al inicio.
            # Esto mantiene la fase explotativa más corta tras cada reheat.
            T = T_init_eff * reheat_factor
            # Reiniciamos el contador para que el siguiente reheat requiera
            # otros ``patience`` niveles consecutivos sin mejora.
            niveles_sin_mejora = 0
            # Registramos el reheat en el contador global de la corrida.
            n_reheats += 1

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
            "alpha_inter": alpha_inter,
            "p_inter": p_inter,
            # Parámetros y contador del mecanismo de reheat.
            "patience": patience,
            "reheat_factor": reheat_factor,
            "n_reheats": n_reheats,
            "n_tareas": n_tareas,
            "d_max": d_max,
            "L_cadena_markov": L,
            "temperatura_inicial_efectiva": T_init_eff,
            "temperatura_minima_efectiva": T_min_eff,
            **contador.resumen_csv(),
            "iteraciones_totales": iteraciones_totales,
            "enfriamientos_ejecutados": enfriamientos,
            "aceptadas": aceptadas,
            "mejoras": mejoras,
            "mejor_solucion_factible_final": mejor_factible_final,
            "mejor_solucion_tr_legible": solucion_legible_humana(sol_mejor),
            "reporte_detalle_deadheading": detalle_txt,
            "costo_total_desde_reporte": costo_total_reporte,
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
        # Conteo total de reheats activados durante esta corrida.
        n_reheats=n_reheats,
    )


def recocido_simulado_desde_instancia(
    nombre_instancia: str,
    *,
    root: str | None = None,
    temperatura_inicial: float | None = None,
    temperatura_minima: float | None = None,
    alpha: float = 0.95,
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
    extra_csv: dict[str, object] | None = None,
    alpha_inter: float = 0.8,
    p_inter: float = 0.6,  # fracción fija de probabilidad para operadores inter-ruta (activa incluso en soluciones factibles)
    patience: int = 50,    # niveles sin mejora antes de reheat (0 = desactivado)
    reheat_factor: float = 0.5,  # fracción de T_init_eff a la que se recalienta al activar el reheat
    **_ignorado_kwargs: object,  # absorbe kwargs heredados (p.ej. id_corrida, config_id)
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
        extra_csv=extra_csv,
        alpha_inter=alpha_inter,
        p_inter=p_inter,
        patience=patience,
        reheat_factor=reheat_factor,
    )
