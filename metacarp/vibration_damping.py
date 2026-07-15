"""
Vibration Damping Optimization (VDO) para CARP — versión instance-aware,
alineada con SA / TS simple / RTS / ABC simple / Cuckoo Search.

Concepto algorítmico
--------------------
Vibration Damping Optimization (Mehdizadeh y Tavakkoli-Moghaddam, 2008/2009) es
una metaheurística de trayectoria de una sola solución inspirada en el
**amortiguamiento de la amplitud vibratoria** en sistemas mecánicos. Su
estructura es prima hermana del Recocido Simulado (SA), pero el parámetro de
control ya no es la temperatura sino la **amplitud de vibración** ``A``:

- Un sistema mecánico oscilante (péndulo, resorte, membrana) que arranca con
  gran amplitud ``A0`` va perdiendo energía por fricción o resistencia del
  medio, y su amplitud decae según una ley exponencial hasta acercarse a cero:

      A(t) = A0 · exp(-γ · t / 2)

  donde ``t`` es el nivel discreto de la búsqueda (equivale al ciclo de
  temperatura del SA) y ``γ > 0`` es el **coeficiente de amortiguamiento**.
  Un γ pequeño corresponde a un sistema con poca fricción (amortiguamiento
  suave, decae despacio → mucha exploración); un γ grande corresponde a mucha
  fricción (decae rápido → convergencia veloz, poca exploración).

- Trasladado a optimización:

  * La **amplitud A** juega el rol que en SA jugaba la temperatura T: mide
    cuánto se permite "empeorar" al aceptar una solución vecina.
  * Al arrancar (A grande) la búsqueda salta libremente por todo el espacio.
  * A medida que A se amortigua, la búsqueda se vuelve más selectiva y
    termina comportándose como un descenso voraz cuando A ≈ 0.

Regla de aceptación por distribución de Rayleigh
------------------------------------------------
En cada iteración se genera un vecino con costo ``c_vecino``. Sea
``Δ = c_vecino − c_actual``:

- Si el vecino es **mejor o igual** (Δ ≤ 0): siempre se acepta.
- Si el vecino es **peor** (Δ > 0): se acepta con probabilidad

      p = 1 − exp( − A² / (2·σ²) )

  Esta expresión es exactamente la **función de distribución acumulada
  (CDF) de una variable Rayleigh** con parámetro de escala ``σ``, evaluada
  en el punto ``A``. Intuitivamente:

  * Cuando ``A`` es grande frente a ``σ`` (comienzo de la búsqueda), el
    exponente ``A²/(2σ²)`` es grande, ``exp(−·)`` es cercano a 0 y por
    tanto ``p ≈ 1`` (casi toda perturbación negativa se acepta →
    exploración agresiva).
  * Cuando ``A`` es pequeño frente a ``σ`` (final de la búsqueda), el
    exponente es cercano a 0, ``exp(−·)`` es cercano a 1 y por tanto
    ``p ≈ 0`` (casi ningún empeoramiento se acepta → explotación local).

Nótese que la regla NO depende de la magnitud de ``Δ`` (a diferencia del
Metropolis del SA). VDO decide "cuánto arriesgar" únicamente a partir de la
amplitud vibratoria actual: cualquier empeoramiento se trata con la misma
probabilidad de aceptación mientras el sistema esté vibrando con amplitud A.
Esto es una diferencia conceptual importante frente al SA y la razón por la
que ``σ`` aparece como un parámetro independiente que modula la respuesta
vibratoria.

Ley de amortiguamiento
----------------------
Al terminar cada nivel se aplica la ley:

      A_{t+1} = A0 · exp( − γ · t / 2 )

donde ``t`` es el índice del nivel (0, 1, 2, …). En t=0 se cumple A(0)=A0.
Esta forma sigue la letra del paper original, en el cual el decaimiento se
calcula SIEMPRE a partir de la amplitud inicial ``A0`` (no acumulando el
factor sobre la amplitud del nivel anterior como haría el SA con
``alpha × T_actual``). Es la ecuación del oscilador amortiguado clásico
evaluada en instantes discretos ``t``.

La búsqueda se detiene cuando ``A`` cae por debajo de un umbral
``umbral_amplitud_minima`` (analogía de ``T_min`` en SA), o cuando se agota
el presupuesto de niveles (parámetro opcional ``max_niveles``).

Calibración adaptativa desde la instancia
-----------------------------------------
Al igual que SA / ABC simple / Cuckoo, los parámetros escalables se calculan
por defecto en función de la instancia. Sea ``n`` el número de arcos
requeridos y ``d_max`` la distancia máxima en la matriz Dijkstra:

- ``L``                     = n²           (iteraciones por nivel de amplitud;
  equivalente a la cadena de Markov del SA).
- ``A0``                    = 20 · d_max / n
  (amplitud inicial: misma escala que ``T_init`` del SA, para que el
  régimen exploratorio inicial sea comparable entre las dos metaheurísticas).
- ``umbral_amplitud_minima`` = 20 · d_max / n²
  (frontera inferior de amortiguamiento, análoga a ``T_min``).
- ``sigma``                 = A0 / 2
  (parámetro de escala Rayleigh: garantiza que en el nivel inicial la
  probabilidad de aceptación de empeoramientos sea ``1 − exp(−2) ≈ 0.865``,
  suficientemente alta para explorar sin degenerar en un paseo aleatorio).
- ``gamma``                 = 0.05
  (coeficiente de amortiguamiento por defecto; no depende de la instancia
  porque ``γ`` es una constante adimensional del sistema oscilante).

El usuario puede sobrescribir cualquiera de estos valores pasándolos
explícitamente. La precedencia es siempre: **valor absoluto > factor > default**
(mismo contrato que Cuckoo / ABC simple).

Optimización de evaluación
--------------------------
Construye un :class:`ContextoEvaluacion` (matriz Dijkstra densa + arrays por id
de tarea) **una sola vez** al inicio. Cada vecino se evalúa con
:func:`costo_rapido` (NumPy fancy-indexing): 10×–50× más rápido que el
evaluador basado en NetworkX.

GPU
---
VDO evalúa una solución por iteración, por lo que el flag ``usar_gpu`` se
propaga al contexto solo para trazabilidad: el cuello de botella ya está
resuelto en CPU y mover datos a GPU no aporta speedup en este caso
(el mismo razonamiento aplicado al SA).
"""
# Permite usar `float | None` en Python < 3.10.
from __future__ import annotations

# math.exp calcula la función exponencial: necesaria para la ley de
# amortiguamiento y para la CDF de Rayleigh.
import math
# Generador de números aleatorios controlado por semilla.
import random
# Medición de tiempo de alta resolución.
import time
# Tipos abstractos para firmas de funciones.
from collections.abc import Callable, Iterable, Mapping
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
    seleccionar_grupo_operadores_inter_intra,
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
    "VibrationDampingResult",
    "vibration_damping",
    "vibration_damping_desde_instancia",
]


# --- CONCEPTO OOP: @dataclass(frozen=True, slots=True) ---
# frozen=True: objeto inmutable; slots=True: menor consumo de memoria.
# Apropiado para un objeto de resultado que solo se consulta, nunca se modifica.
@dataclass(frozen=True, slots=True)
class VibrationDampingResult:
    """
    Resultado completo de una corrida de Vibration Damping Optimization.

    Agrupa la mejor solución encontrada, métricas de calidad, historial de
    amplitudes y estadísticas de aceptaciones y operadores de vecindario.
    Sigue el mismo contrato de campos que RecocidoSimuladoResult para
    facilitar comparaciones directas entre SA y VDO.
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
    # Cuántos niveles de amortiguamiento se ejecutaron.
    niveles_ejecutados: int
    # Total de vecinos aceptados (mejores + peores aceptados por Rayleigh).
    aceptadas: int
    # Veces que el mejor global mejoró.
    mejoras: int
    # Semilla del generador aleatorio.
    semilla: int | None
    # Dispositivo de evaluación: 'cpu' o 'gpu'.
    backend_evaluacion: str = "cpu"
    # Historial del mejor costo al inicio de cada nivel de amplitud.
    historial_mejor_costo: list[float] = field(default_factory=list)
    # Historial del valor de amplitud por nivel.
    historial_amplitud: list[float] = field(default_factory=list)
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
    # ---- Campos instance-aware (calibración automática por n_tareas) ----
    # Número de arcos requeridos de la instancia (variable ``n`` en las fórmulas).
    n_tareas: int = 0
    # Valores REALES usados por el bucle (pueden venir del absoluto pasado por
    # el usuario o de la fórmula default). Reportarlos permite reproducir la
    # corrida sin ambigüedad.
    amplitud_inicial_efectiva: float = 0.0
    umbral_amplitud_minima_efectivo: float = 0.0
    sigma_efectivo: float = 0.0
    gamma_efectivo: float = 0.0
    iteraciones_por_nivel_L: int = 0


def vibration_damping(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    # ---- Parámetros de amortiguamiento (None ⇒ se calibran desde la instancia) ----
    # amplitud_inicial: análogo de temperatura_inicial en SA. Regula cuánto
    # empeoramiento se acepta al arrancar. None ⇒ 20 · d_max / n.
    amplitud_inicial: float | None = None,
    # umbral_amplitud_minima: análogo de temperatura_minima. Cuando A cae por
    # debajo de este umbral la búsqueda termina. None ⇒ 20 · d_max / n².
    umbral_amplitud_minima: float | None = None,
    # sigma: parámetro de escala de la distribución Rayleigh. Un σ grande
    # HACE la p de aceptación más pequeña para una A dada (curva más plana).
    # None ⇒ amplitud_inicial / 2 (garantiza p ≈ 0.865 al arranque).
    sigma: float | None = None,
    # gamma: coeficiente de amortiguamiento. Rango típico 0.01 – 0.1. Un γ
    # grande = decaimiento rápido (menos exploración). Adimensional, por lo
    # que su default NO depende de la instancia.
    gamma: float = 0.05,
    # L: iteraciones por nivel de amplitud (equivalente a la cadena de Markov
    # del SA). None ⇒ n² (misma calibración que SA).
    iteraciones_por_nivel: int | None = None,
    # Tope opcional al número de niveles. None ⇒ sin tope (parada solo por A).
    # Útil para presupuestos de tiempo/iteraciones acotados sin recalcular γ.
    max_niveles: int | None = None,
    semilla: int | None = None,                # semilla para reproducibilidad
    operadores: Iterable[str] = OPERADORES_POPULARES,  # operadores de vecindario habilitados
    marcador_depot_etiqueta: str | None = None,  # etiqueta del nodo depósito
    usar_gpu: bool = False,                   # flag de GPU (solo trazabilidad en VDO)
    backend_vecindario: Literal["labels", "ids"] = "labels",  # modo de generación de vecinos
    guardar_historial: bool = True,           # si True, guarda historial por nivel
    guardar_csv: bool = False,                # si True, escribe resultados en CSV
    ruta_csv: str | None = None,              # ruta del CSV
    nombre_instancia: str = "instancia",      # nombre para el CSV
    repeticion: int | None = None,
    root: str | None = None,
    usar_penalizacion_capacidad: bool = True,  # si True, penaliza violaciones de capacidad
    lambda_capacidad: float | None = None,     # peso λ (None = automático)
    extra_csv: dict[str, object] | None = None,  # columnas adicionales para el CSV
    alpha_inter: float = 0.8,   # fracción de prob. asignada a ops inter-ruta bajo violación
    p_inter: float = 0.6,       # fracción fija de prob. inter-ruta en régimen factible
    metodo_seleccion: str = "canonico",  # método de combinación inter/intra
    # --- Kick reactivo / intensificación (mismo contrato que SA) ---
    # Cuando ``niveles_sin_mejora_kick`` alcanza este umbral se aplica una
    # perturbación (o el intensificador) y se reinicia el contador.
    # None = mecanismo desactivado.
    max_iter_sin_mejora_kick: int | None = None,
    # Cota dura de kicks; al alcanzarla la corrida termina. None = sin tope.
    max_resets: int | None = None,
    # Hook de intensificación OPCIONAL (p.ej. Path Relinking limpio). Si se
    # provee, en el punto de estancamiento se ejecuta
    # ``intensificador(sol_actual, mejor_global, ctx, lam, rng, encoding,
    # md)`` HACIA la mejor solución global EN LUGAR del kick aleatorio.
    intensificador: Callable | None = None,
    # --- Presupuesto de wall-clock (val_egl_20260710) ---
    # Corta la corrida al alcanzar este límite en segundos. Se comprueba UNA
    # vez al inicio de cada nivel de amplitud (bucle externo), NO en cada
    # iteración interna. None = sin límite (comportamiento clásico).
    tiempo_limite_segundos: float | None = None,
    **_ignorado_kwargs: object,  # absorbe kwargs heredados (p.ej. id_corrida, config_id)
) -> VibrationDampingResult:
    """
    Vibration Damping Optimization (VDO) para minimizar el costo de soluciones CARP.

    Estructura del algoritmo:

    - Bucle externo: niveles de amplitud (de A0 hasta ``umbral_amplitud_minima``).
    - Bucle interno: L evaluaciones por nivel, donde L = n² por defecto.
    - En cada evaluación se genera un vecino y se decide si aceptarlo con la
      regla de aceptación basada en la CDF de Rayleigh:
          p = 1 − exp(−A² / (2·σ²))
    - Al terminar el nivel se aplica la ley de amortiguamiento:
          A = A0 · exp(−γ · t / 2), donde ``t`` es el índice del nivel.

    Criterios de parada:

        - Canónico: ``A < umbral_amplitud_minima``.
        - Opcional: ``max_niveles`` alcanzado (útil para presupuesto acotado).

    Calibración adaptativa (mismas fórmulas que SA / ABC simple / Cuckoo):

        L        = n²
        A0       = 20 · d_max / n
        A_min    = 20 · d_max / n²
        σ        = A0 / 2
        γ        = 0.05   (constante adimensional)

    donde ``n`` = número de arcos requeridos y ``d_max`` = distancia máxima en
    la matriz Dijkstra.

    Parámetros de sesgo de vecindario (idénticos a SA):

        alpha_inter : fracción de probabilidad para operadores inter-ruta bajo
            violación de capacidad. Favorece la reparación redistribuyendo
            demanda entre rutas.
        p_inter : fracción fija de probabilidad para operadores inter-ruta
            en régimen factible. Se recomienda ``alpha_inter >= p_inter``.
    """
    # --- Validaciones de parámetros ---
    # Solo validamos los valores que el usuario pasó explícitamente (los None
    # se validan implícitamente tras la calibración adaptativa).
    if amplitud_inicial is not None and amplitud_inicial <= 0:
        raise ValueError("amplitud_inicial debe ser > 0.")
    if umbral_amplitud_minima is not None and umbral_amplitud_minima <= 0:
        raise ValueError("umbral_amplitud_minima debe ser > 0.")
    if sigma is not None and sigma <= 0:
        raise ValueError("sigma debe ser > 0.")
    if gamma <= 0:
        raise ValueError("gamma debe ser > 0 (coeficiente de amortiguamiento).")
    if iteraciones_por_nivel is not None and iteraciones_por_nivel <= 0:
        raise ValueError("iteraciones_por_nivel debe ser > 0 si se especifica.")
    if max_niveles is not None and max_niveles <= 0:
        raise ValueError("max_niveles debe ser > 0 si se especifica.")
    if not (0.0 <= p_inter <= 1.0):
        raise ValueError("p_inter debe estar en [0, 1].")
    if not (0.0 <= alpha_inter <= 1.0):
        raise ValueError("alpha_inter debe estar en [0, 1].")

    # Generador aleatorio reproducible.
    rng = random.Random(semilla)
    # Marca de tiempo de inicio.
    t0 = time.perf_counter()

    # Construcción del contexto de evaluación rápida (una sola vez por corrida).
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

    # ----------------------------------------------------------
    # Calibración adaptativa desde la instancia.
    # n_tareas = número de arcos requeridos; d_max = distancia máxima en la
    # matriz Dijkstra. Las fórmulas replican las de SA para que ambos
    # métodos partan del mismo régimen exploratorio y sean comparables.
    # ----------------------------------------------------------
    n_tareas = int(len(ctx.u_arr))
    _dist_finita = ctx.dist[ctx.dist < np.inf]
    d_max = float(_dist_finita.max()) if len(_dist_finita) > 0 else 1.0

    # L: iteraciones por nivel de amplitud (cadena de Markov). Absoluto > default.
    L_eff = (
        int(iteraciones_por_nivel)
        if iteraciones_por_nivel is not None
        else max(1, n_tareas * n_tareas)
    )
    # A0: amplitud inicial. Absoluto > default (20·d_max/n).
    A0_eff = (
        float(amplitud_inicial)
        if amplitud_inicial is not None
        else 20.0 * d_max / max(1, n_tareas)
    )
    # A_min: umbral inferior de amplitud. Absoluto > default (20·d_max/n²).
    A_min_eff = (
        float(umbral_amplitud_minima)
        if umbral_amplitud_minima is not None
        else 20.0 * d_max / max(1, n_tareas * n_tareas)
    )
    # sigma: escala de Rayleigh. Absoluto > default (A0/2).
    sigma_eff = float(sigma) if sigma is not None else A0_eff / 2.0
    # gamma NO tiene calibración adaptativa: es una constante adimensional.
    gamma_eff = float(gamma)

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

    # La solución "actual" es la que se modifica en cada iteración.
    sol_actual = copiar_solucion_labels(sol_ref)
    costo_actual = costo_ref
    viol_actual = sel_ini.violacion_capacidad

    # Rastreo del mejor global y del mejor factible.
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

    # Etiqueta del depósito para los operadores de vecindario.
    md_op = marcador_depot_etiqueta or ctx.marcador_depot

    # --- Precómputo de particiones intra/inter (fuera del bucle externo) ---
    # Idéntica microoptimización a SA / RTS / Cuckoo: las listas de operadores
    # NO cambian dentro de una corrida, así que las construimos una sola vez
    # y luego solo se leen en el hot path.
    _ops_intra = [op for op in operadores if op in OPERADORES_INTRA]
    _ops_inter = [op for op in operadores if op in OPERADORES_INTER]
    _ops_fallback = list(operadores)

    # --- Estado del recorrido ---
    A = A0_eff                # amplitud actual (variable de control del algoritmo)
    nivel = 0                 # índice del nivel actual (0-based, entra en la fórmula)
    iteraciones_totales = 0   # total de vecinos evaluados
    aceptadas = 0             # vecinos aceptados (mejores + peores)
    mejoras = 0               # veces que el mejor global mejoró
    aceptaciones_sol_infactible = 0  # aceptaciones que violan capacidad
    ultimo_mov_aceptado: MovimientoVecindario | None = None
    historial_best: list[float] = []
    historial_amp: list[float] = []
    contador = ContadorOperadores()
    # Contadores del kick reactivo (solo activos si se pasó
    # max_iter_sin_mejora_kick; misma semántica que en SA).
    niveles_sin_mejora_kick = 0
    n_resets_kick = 0

    def costo_para_reporte() -> float:
        """Devuelve el mejor costo factible si existe; si no, el mejor global."""
        return float(mejor_fact_c) if mejor_fact_c is not None else mejor_any_c

    # === BUCLE EXTERNO: niveles de amplitud (amortiguamiento) ===
    # Se repite mientras la amplitud NO caiga bajo el umbral. Opcionalmente se
    # limita a ``max_niveles`` iteraciones si el usuario lo pasó.
    while A > A_min_eff:
        # Tope opcional por número de niveles ejecutados.
        if max_niveles is not None and nivel >= max_niveles:
            break

        # --- Presupuesto de wall-clock (val_egl_20260710) ---
        # Comprobado UNA sola vez por nivel para no meter overhead en el
        # bucle interno (un nivel dura L = n² evaluaciones).
        if (
            tiempo_limite_segundos is not None
            and (time.perf_counter() - t0) >= tiempo_limite_segundos
        ):
            break

        # Foto del mejor reportable ANTES del bucle interno: al final del
        # nivel se compara para decidir si este nivel cuenta como "sin
        # mejora" hacia el umbral del kick (mismo esquema que SA).
        _rep_antes_nivel = costo_para_reporte()

        if guardar_historial:
            # Registramos la amplitud y el mejor costo al inicio de este nivel.
            historial_amp.append(float(A))
            historial_best.append(costo_para_reporte())

        # Probabilidad de aceptación de empeoramientos EN ESTE NIVEL:
        # p = 1 − exp(−A² / (2·σ²))
        # Es constante durante el nivel entero: la decisión "cuánto arriesgar"
        # solo depende del estado vibratorio A, no del Δ del vecino. Por eso
        # la calculamos UNA sola vez fuera del bucle interno.
        p_aceptar_peor = 1.0 - math.exp(-(A * A) / (2.0 * sigma_eff * sigma_eff))

        # === BUCLE INTERNO: L evaluaciones dentro del nivel de amplitud A ===
        for _ in range(L_eff):
            iteraciones_totales += 1

            # Selección de grupo inter/intra (misma mecánica que en SA).
            # El dado elige el GRUPO; generar_vecino selecciona el operador
            # concreto dentro del grupo y maneja los reintentos internos.
            op_elegido, _hubo_viol = seleccionar_grupo_operadores_inter_intra(
                rng,
                viol_actual,
                _ops_intra,
                _ops_inter,
                _ops_fallback,
                alpha_inter=alpha_inter,
                p_inter=p_inter,
                metodo=metodo_seleccion,
            )

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

            # Objetivos penalizados para comparar candidatos en el mismo espacio.
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

            # delta > 0 significa que el vecino empeora la solución actual.
            delta = obj_vec - obj_actual

            # --- Regla de aceptación por Rayleigh ---
            if delta <= 0:
                # Vecino mejor o igual: siempre se acepta (misma regla que SA).
                aceptar = True
            else:
                # Vecino peor: se acepta con probabilidad p_aceptar_peor,
                # constante durante todo el nivel. Nótese que la magnitud de
                # delta NO influye — a diferencia del Metropolis del SA. VDO
                # decide "cuánto arriesgar" únicamente a partir de A.
                aceptar = rng.random() < p_aceptar_peor

            if aceptar:
                # Actualizamos la solución actual al vecino aceptado.
                sol_actual = vecino
                costo_actual = costo_vec
                viol_actual = viol_vec
                aceptadas += 1
                ultimo_mov_aceptado = mov
                contador.aceptar(mov.operador)
                if usar_penalizacion_capacidad and viol_vec > 1e-12:
                    aceptaciones_sol_infactible += 1

                # Verificamos si esta aceptación mejora el mejor global.
                antes_rep = costo_para_reporte()
                if costo_vec < mejor_any_c - 1e-15:
                    mejor_any_c = costo_vec
                    mejor_any_s = copiar_solucion_labels(sol_actual)
                if viol_vec < 1e-12:
                    lim_fact = mejor_fact_c if mejor_fact_c is not None else float("inf")
                    if costo_vec < lim_fact - 1e-15:
                        mejor_fact_c = float(costo_vec)
                        mejor_fact_s = copiar_solucion_labels(sol_actual)

                despues_rep = costo_para_reporte()
                if despues_rep < antes_rep - 1e-12:
                    mejoras += 1
                    contador.registrar_mejora(mov.operador)

        # --- Ley de amortiguamiento (aplicada al FINAL del nivel) ---
        # Avanzamos el índice del nivel y calculamos la amplitud del siguiente
        # nivel a partir de la fórmula canónica del oscilador amortiguado.
        # A diferencia del SA (T *= alpha), aquí NO acumulamos el decaimiento
        # sobre A_actual: siempre partimos de A0 y multiplicamos por el factor
        # e^(-γ·t/2). Esto es lo que dice la ecuación original del sistema.
        nivel += 1
        A = A0_eff * math.exp(-gamma_eff * nivel / 2.0)

        # --- Kick por estancamiento global (mismo contrato que SA) ---
        # El nivel cuenta como "sin mejora" si el mejor reportable no avanzó
        # respecto a la foto tomada antes del bucle interno.
        if costo_para_reporte() < _rep_antes_nivel - 1e-12:
            niveles_sin_mejora_kick = 0
        else:
            niveles_sin_mejora_kick += 1
        if (max_iter_sin_mejora_kick is not None
                and niveles_sin_mejora_kick >= max_iter_sin_mejora_kick):
            if intensificador is not None:
                # Intensificación (p.ej. Path Relinking limpio) hacia la
                # mejor solución global: la guía es la mejor factible si
                # existe; si no, la mejor any.
                guia = mejor_fact_s if mejor_fact_s is not None else mejor_any_s
                sol_actual = intensificador(
                    sol_actual, guia, ctx, lam_eff, rng, encoding, md_op
                )
            else:
                # Import diferido: solo se carga si la corrida activa el kick.
                from metacarp.strict_intra_inter_20260524 import aplicar_kick_labels
                sol_actual = aplicar_kick_labels(
                    sol_actual, rng, md_op, encoding=encoding
                )
            # Recalculamos costo y violación para que el siguiente nivel
            # arranque con datos coherentes.
            costo_actual = float(costo_rapido(sol_actual, ctx))
            viol_actual = float(exceso_capacidad_rapido(sol_actual, ctx))
            niveles_sin_mejora_kick = 0
            n_resets_kick += 1
            if max_resets is not None and n_resets_kick >= max_resets:
                break

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
    _gap_descartado, mejora_abs, mejora_pct = calcular_metricas_gap(costo_ref, costo_mejor)

    # --- Guardado en CSV (opcional) ---
    archivo_csv: str | None = None
    if guardar_csv:
        ruta = ruta_csv or f"resultados_vibration_damping_{nombre_instancia}.csv"
        detalle_txt, costo_total_reporte = generar_reporte_detallado(
            sol_mejor, data, G,
            nombre_instancia=nombre_instancia,
            marcador_depot_etiqueta=marcador_depot_etiqueta,
            usar_gpu=False,  # reporte usa NetworkX para texto detallado
        )
        _bks = resumen_bks_csv(data, costo_mejor)
        fila = {
            "metaheuristica": "vibration_damping",
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
            # Valores BASE pasados por el usuario (cadena vacía si eran None).
            "amplitud_inicial": amplitud_inicial if amplitud_inicial is not None else "",
            "umbral_amplitud_minima": (
                umbral_amplitud_minima if umbral_amplitud_minima is not None else ""
            ),
            "sigma": sigma if sigma is not None else "",
            "gamma": gamma,
            "iteraciones_por_nivel": (
                iteraciones_por_nivel if iteraciones_por_nivel is not None else ""
            ),
            "max_niveles": max_niveles if max_niveles is not None else "",
            # Parámetros del sesgo inter/intra.
            "alpha_inter": alpha_inter,
            "p_inter": p_inter,
            "metodo_seleccion": metodo_seleccion,
            # ---- VALORES EFECTIVOS (lo que realmente usó el bucle) ----
            "amplitud_inicial_efectiva": A0_eff,
            "umbral_amplitud_minima_efectivo": A_min_eff,
            "sigma_efectivo": sigma_eff,
            "gamma_efectivo": gamma_eff,
            "iteraciones_por_nivel_L": L_eff,
            "n_tareas": n_tareas,
            "d_max": d_max,
            # Penalización de capacidad.
            "usar_penalizacion_capacidad": usar_penalizacion_capacidad,
            "lambda_capacidad": lam_eff,
            # 36 columnas del contador de operadores.
            **contador.resumen_csv(),
            # Estadísticas de corrida.
            "iteraciones_totales": iteraciones_totales,
            "niveles_ejecutados": nivel,
            "aceptadas": aceptadas,
            "mejoras": mejoras,
            "aceptaciones_solucion_infactible": aceptaciones_sol_infactible,
            "mejor_solucion_factible_final": mejor_factible_final,
            "mejor_solucion_tr_legible": solucion_legible_humana(sol_mejor),
            "reporte_detalle_deadheading": detalle_txt,
            "costo_total_desde_reporte": costo_total_reporte,
            "n_resets_kick": n_resets_kick,
        }
        # Volcamos ``extra_csv`` a la fila (mismo contrato que Cuckoo).
        fila.update(extra_csv or {})
        archivo_csv = guardar_resultado_csv(fila=fila, ruta_csv=ruta)

    # Retornamos el objeto de resultado inmutable con todos los datos de la corrida.
    return VibrationDampingResult(
        mejor_solucion=sol_mejor,
        mejor_costo=costo_mejor,
        solucion_inicial_referencia=sol_ref,
        costo_solucion_inicial=costo_ref,
        mejora_absoluta=mejora_abs,
        mejora_porcentaje_inicial_vs_final=mejora_pct,
        tiempo_segundos=elapsed,
        iteraciones_totales=iteraciones_totales,
        niveles_ejecutados=nivel,
        aceptadas=aceptadas,
        mejoras=mejoras,
        semilla=semilla,
        backend_evaluacion=ctx.backend_real,
        historial_mejor_costo=historial_best,
        historial_amplitud=historial_amp,
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
        amplitud_inicial_efectiva=A0_eff,
        umbral_amplitud_minima_efectivo=A_min_eff,
        sigma_efectivo=sigma_eff,
        gamma_efectivo=gamma_eff,
        iteraciones_por_nivel_L=L_eff,
    )


def vibration_damping_desde_instancia(
    nombre_instancia: str,
    *,
    root: str | None = None,
    amplitud_inicial: float | None = None,
    umbral_amplitud_minima: float | None = None,
    sigma: float | None = None,
    gamma: float = 0.05,
    iteraciones_por_nivel: int | None = None,
    max_niveles: int | None = None,
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
    p_inter: float = 0.6,
    metodo_seleccion: str = "canonico",
    max_iter_sin_mejora_kick: int | None = None,
    max_resets: int | None = None,
    intensificador: Callable | None = None,
    tiempo_limite_segundos: float | None = None,
    **_ignorado_kwargs: object,  # absorbe kwargs heredados (p.ej. id_corrida, config_id)
) -> VibrationDampingResult:
    """
    Función de conveniencia: carga todos los recursos necesarios desde el nombre
    de la instancia y ejecuta Vibration Damping Optimization completo.

    Equivalente a llamar manualmente a ``load_instances`` + ``cargar_objeto_gexf``
    + ``cargar_solucion_inicial`` + :func:`vibration_damping`.
    """
    # Cargamos los datos de la instancia (capacidad, demandas, BKS, etc.).
    data = load_instances(nombre_instancia, root=root)
    # Cargamos el grafo de la instancia desde el archivo GEXF.
    G = cargar_objeto_gexf(nombre_instancia, root=root)
    # Cargamos la solución inicial desde el archivo pickle.
    inicial_obj = cargar_solucion_inicial(nombre_instancia, root=root)
    return vibration_damping(
        inicial_obj,
        data,
        G,
        amplitud_inicial=amplitud_inicial,
        umbral_amplitud_minima=umbral_amplitud_minima,
        sigma=sigma,
        gamma=gamma,
        iteraciones_por_nivel=iteraciones_por_nivel,
        max_niveles=max_niveles,
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
        metodo_seleccion=metodo_seleccion,
        max_iter_sin_mejora_kick=max_iter_sin_mejora_kick,
        max_resets=max_resets,
        intensificador=intensificador,
        tiempo_limite_segundos=tiempo_limite_segundos,
    )
