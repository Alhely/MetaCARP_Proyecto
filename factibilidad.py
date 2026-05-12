# Módulo: factibilidad.py
# Propósito: Verifica si una solución CARP cumple las cinco condiciones de factibilidad.
# Una solución "factible" es aquella que puede ser ejecutada realmente por los vehículos
# (todas las rutas son posibles, no se excede la capacidad, se cubren todas las tareas, etc.)
#
# Las cinco condiciones que se verifican (C1 a C5) son:
#
# C1 – Cobertura y unicidad de tareas requeridas:
#       Cada arista requerida debe aparecer EXACTAMENTE UNA vez en la solución.
#
# C2 – Conectividad entre tareas consecutivas:
#       Entre dos tareas consecutivas en la misma ruta debe existir un camino en el grafo.
#
# C3 – Capacidad por ruta:
#       La suma de demandas de las tareas de cada ruta no debe superar CAPACIDAD.
#
# C4 – Número de rutas vs. número de vehículos:
#       El número de rutas no vacías no puede superar VEHICULOS.
#
# C5 – Conectividad con el depósito:
#       Desde el depósito debe haber camino a la primera tarea de cada ruta,
#       y desde la última tarea debe haber camino de regreso al depósito.

from __future__ import annotations  # Permite anotaciones de tipo forward-reference

import math                                     # Para math.isnan, math.isinf
import os                                       # Para rutas del sistema de archivos
from dataclasses import dataclass, field        # Para clases de datos con defaults
from typing import Any, Hashable, Mapping, Sequence  # Tipos genéricos

import numpy as np  # Para manejar matrices NumPy de distancias

# Funciones del módulo de formato estándar de soluciones:
# - construir_mapa_tareas_por_etiqueta: crea el dict {etiqueta: datos_tarea}
# - etiquetas_tareas_requeridas: devuelve el conjunto de etiquetas que DEBEN aparecer
# - normalizar_rutas_etiquetas: limpia las rutas (quita depósito, valida etiquetas)
from .solucion_formato import (
    construir_mapa_tareas_por_etiqueta,
    etiquetas_tareas_requeridas,
    normalizar_rutas_etiquetas,
)

# API pública de este módulo.
__all__ = [
    "FeasibilityDetails",
    "FeasibilityResult",
    "verificar_factibilidad",
    "verificar_factibilidad_desde_instancia",
]


# ---------------------------------------------------------------------------
# Función interna: _aplicar_backend_gpu_placeholder
# ---------------------------------------------------------------------------
def _aplicar_backend_gpu_placeholder(usar_gpu: bool) -> tuple[str, str]:
    """
    Registra el backend solicitado y el real para trazabilidad futura.

    Cuando se implemente GPU para factibilidad, esta función se actualizará.
    Por ahora, siempre usa CPU como backend real aunque se pida GPU.

    Returns:
        Tupla (backend_solicitado, backend_real). Ejemplo: ("gpu", "cpu").
    """
    if not usar_gpu:
        return "cpu", "cpu"
    # Solicitud de GPU registrada, pero el backend real sigue siendo CPU.
    return "gpu", "cpu"


# ---------------------------------------------------------------------------
# Función interna: _dist
# ---------------------------------------------------------------------------
def _dist(matriz: Any, a: int, b: int) -> float:
    """
    Consulta la distancia entre los nodos ``a`` y ``b`` en la matriz de distancias.

    Maneja dos formatos de matriz:
    - ``dict`` anidado: {a: {b: distancia}} (formato Dijkstra en diccionarios).
    - ``np.ndarray`` 2D (formato denso cargado del archivo de matrices).

    Returns:
        Distancia como float, o ``float('inf')`` si no hay camino o el dato está ausente.
    """
    inf = float("inf")
    try:
        if isinstance(matriz, dict):
            # Buscamos la fila del nodo origen en el diccionario.
            row = matriz.get(a)
            if row is None:
                return inf  # Nodo 'a' no está en la matriz

            if isinstance(row, dict):
                if b not in row:
                    return inf  # No hay entrada para el destino 'b'
                val = row[b]
            else:
                return inf  # La fila no es un diccionario: formato inesperado
        else:
            # Formato NumPy: convertimos a array y accedemos por índice.
            arr = np.asarray(matriz)
            if arr.ndim != 2:
                return inf  # Solo soportamos matrices 2D

            n, m = arr.shape

            if 0 <= a < n and 0 <= b < m:
                # Indexación 0-based directa (el nodo 0 es la fila 0).
                val = arr[a, b]
            elif 1 <= a <= n and 1 <= b <= m and (a < n or a == n):
                # Indexación 1-based: si los nodos empiezan en 1, ajustamos.
                val = arr[a - 1, b - 1]
            else:
                return inf  # Índices fuera de rango

        # Convertimos el valor a float y verificamos que sea un número válido.
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return inf  # NaN o inf se interpretan como "sin camino"
        return v

    except (KeyError, TypeError, ValueError, IndexError):
        # Cualquier error al acceder a la matriz se interpreta como "sin camino".
        return inf


# ---------------------------------------------------------------------------
# Función interna: _hay_camino_entre_tareas
# ---------------------------------------------------------------------------
def _hay_camino_entre_tareas(
    matriz: Any,
    u_ant: int,   # Nodo inicio de la tarea anterior
    v_ant: int,   # Nodo fin de la tarea anterior
    u_act: int,   # Nodo inicio de la tarea actual
    v_act: int,   # Nodo fin de la tarea actual
) -> bool:
    """
    Verifica si existe al menos un camino entre algún extremo de la tarea anterior
    y algún extremo de la tarea actual.

    En CARP, el vehículo puede terminar en cualquier extremo de la tarea anterior
    (u o v, dependiendo del sentido de servicio), y puede entrar a la tarea actual
    por cualquier extremo. Por eso se comprueban las 4 combinaciones posibles.

    Returns:
        True si existe al menos uno de los cuatro caminos posibles.
    """
    # Comprobamos las 4 combinaciones: (u_ant→u_act), (u_ant→v_act),
    # (v_ant→u_act), (v_ant→v_act). La primera que no sea inf es suficiente.
    for p in (u_ant, v_ant):        # Nodo de salida de la tarea anterior
        for q in (u_act, v_act):    # Nodo de llegada de la tarea actual
            if _dist(matriz, p, q) != float("inf"):
                return True  # Hay camino: la conectividad C2/C5 se satisface
    return False  # Ninguna de las 4 combinaciones tiene camino


# ---------------------------------------------------------------------------
# Función interna: _hay_camino_a_deposito
# ---------------------------------------------------------------------------
def _hay_camino_a_deposito(matriz: Any, u: int, v: int, deposito: int) -> bool:
    """
    Verifica si existe camino desde algún extremo de la última tarea hasta el depósito.

    Al terminar la ruta, el vehículo puede estar en u o en v de la última tarea,
    por lo que se comprueban ambas posibilidades.

    Returns:
        True si al menos uno de los dos extremos tiene camino al depósito.
    """
    return (
        _dist(matriz, u, deposito) != float("inf")
        or _dist(matriz, v, deposito) != float("inf")
    )


# ---------------------------------------------------------------------------
# Función interna: _verificar_ruta
# ---------------------------------------------------------------------------
def _verificar_ruta(
    ruta: Sequence[Hashable],               # Lista de etiquetas de tareas de la ruta
    info_tareas: Mapping[Any, dict[str, Any]],  # Mapa {etiqueta: datos_tarea}
    data: Mapping[str, Any],                # Datos completos de la instancia
    matriz: Any,                            # Matriz de distancias (dict o ndarray)
    indice_ruta: int,                       # Índice de la ruta (para mensajes de error)
) -> tuple[list[str], list[str], list[str]]:
    """
    Verifica las condiciones C2, C3 y C5 para una ruta individual.

    C2 – Conectividad entre tareas consecutivas (solo entre pares de tareas, no depósito).
    C3 – Capacidad: la demanda acumulada no debe superar CAPACIDAD.
    C5 – Conectividad del depósito con la primera y última tarea de la ruta.

    Args:
        ruta: Lista de etiquetas de las tareas de la ruta (sin marcador de depósito).
        info_tareas: Mapa de etiqueta a datos de tarea.
        data: Datos completos de la instancia (para leer CAPACIDAD y DEPOSITO).
        matriz: Matriz de distancias mínimas para verificar conectividad.
        indice_ruta: Número de ruta (base 0) para los mensajes de error.

    Returns:
        Tupla (fallos_c2, fallos_c3, fallos_c5): listas de mensajes de error.
        Cada lista está vacía si su condición correspondiente se cumple.
    """
    # Listas de mensajes de fallo por condición.
    fallos_c2: list[str] = []
    fallos_c3: list[str] = []
    fallos_c5: list[str] = []

    # Ruta vacía: no hay nada que verificar en esta función.
    if not ruta:
        return fallos_c2, fallos_c3, fallos_c5

    # Leemos los parámetros relevantes de la instancia.
    capacidad_max = float(data.get("CAPACIDAD", 0) or 0)
    deposito = int(data.get("DEPOSITO", 1))

    # Variables de estado para el recorrido de la ruta.
    demanda_total = 0.0                         # Demanda acumulada en esta ruta
    nodos_anteriores: tuple[int, int] | None = None  # Nodos (u, v) de la tarea anterior
    capacidad_rota = False                      # Flag: ya se reportó la violación de C3

    # Recorremos cada tarea de la ruta.
    for paso, id_tarea in enumerate(ruta):
        # Buscamos los datos de la tarea en el mapa.
        tarea = info_tareas.get(id_tarea)
        if not tarea:
            # La tarea no existe: error fatal, no podemos continuar verificando esta ruta.
            fallos_c2.append(
                f"Ruta {indice_ruta}, posición {paso}: tarea {id_tarea!r} no existe en los datos de la instancia."
            )
            return fallos_c2, fallos_c3, fallos_c5

        # Extraemos el par de nodos (u, v) del arco de servicio.
        nodos = tarea.get("nodos")
        if not nodos or len(nodos) != 2:
            fallos_c2.append(
                f"Ruta {indice_ruta}, tarea {id_tarea!r}: falta par de nodos en los datos."
            )
            return fallos_c2, fallos_c3, fallos_c5

        u_act, v_act = int(nodos[0]), int(nodos[1])   # Nodos de la tarea actual
        dem = float(tarea.get("demanda", 0) or 0)     # Demanda de esta tarea
        demanda_total += dem                           # Acumulamos la demanda

        # --- Verificación C3: capacidad ---
        # Solo reportamos la primera violación por ruta (flag capacidad_rota).
        if demanda_total > capacidad_max and not capacidad_rota:
            capacidad_rota = True
            fallos_c3.append(
                f"Ruta {indice_ruta}: demanda acumulada {demanda_total:.4g} supera "
                f"CAPACIDAD {capacidad_max:.4g} (tras la tarea {id_tarea!r}, paso {paso})."
            )

        if paso == 0:
            # --- Verificación C5 (parte 1): depósito → primera tarea ---
            # El depósito tiene u=v=deposito (mismo nodo), por eso usamos (deposito, deposito).
            if not _hay_camino_entre_tareas(matriz, deposito, deposito, u_act, v_act):
                fallos_c5.append(
                    f"Ruta {indice_ruta}: desde el depósito {deposito} no hay camino hacia la "
                    f"primera tarea {id_tarea!r} (nodos {u_act},{v_act})."
                )
        else:
            # --- Verificación C2: conectividad entre tareas consecutivas ---
            # 'nodos_anteriores' es el par (u, v) de la tarea inmediatamente anterior.
            # type: ignore[assignment] suprime la advertencia de tipo: sabemos que
            # nodos_anteriores no es None en paso > 0 (se asignó en iteraciones previas).
            u_ant, v_ant = nodos_anteriores  # type: ignore[assignment]
            if not _hay_camino_entre_tareas(matriz, u_ant, v_ant, u_act, v_act):
                fallos_c2.append(
                    f"Ruta {indice_ruta}, entre la tarea previa ({u_ant},{v_ant}) y la tarea {id_tarea!r} "
                    f"({u_act},{v_act}): no hay camino en la matriz Dijkstra."
                )

        # Guardamos los nodos de esta tarea para usarlos en la siguiente iteración.
        nodos_anteriores = (u_act, v_act)

    # 'assert' verifica en tiempo de ejecución que nodos_anteriores no sea None.
    # Si ruta no estaba vacía (ya lo verificamos arriba), esta condición siempre se cumple.
    # Es una guardia defensiva para el análisis estático de tipos.
    assert nodos_anteriores is not None

    u_ant, v_ant = nodos_anteriores  # Nodos de la ÚLTIMA tarea de la ruta

    # --- Verificación C5 (parte 2): última tarea → depósito ---
    if not _hay_camino_a_deposito(matriz, u_ant, v_ant, deposito):
        fallos_c5.append(
            f"Ruta {indice_ruta}: desde el último servicio ({u_ant},{v_ant}) no hay camino al depósito {deposito}."
        )

    return fallos_c2, fallos_c3, fallos_c5


# ---------------------------------------------------------------------------
# Clase de datos: FeasibilityDetails
# ---------------------------------------------------------------------------
# @dataclass genera automáticamente __init__ y __repr__.
# No usamos frozen=True porque el objeto se construye incrementalmente
# (las listas se van rellenando condición por condición).
@dataclass
class FeasibilityDetails:
    """
    Detalle de los fallos de factibilidad por condición CARP.

    Cada campo es una lista de mensajes de error descriptivos.
    Una lista vacía significa que esa condición se cumple; una lista no vacía
    contiene los mensajes de cada violación detectada.

    - **c1**: cobertura de tareas requeridas y unicidad global de tareas.
    - **c2**: conectividad entre tareas consecutivas en cada ruta.
    - **c3**: capacidad por ruta (demanda no supera CAPACIDAD).
    - **c4**: número de rutas no vacías no supera VEHICULOS.
    - **c5**: conectividad del depósito con la primera y última tarea de cada ruta.
    """

    # field(default_factory=list): crea una lista nueva para CADA instancia.
    # Esto evita el bug clásico de compartir una lista mutable entre instancias.
    c1_tareas_requeridas: list[str] = field(default_factory=list)
    c2_consecutivas: list[str] = field(default_factory=list)
    c3_capacidad: list[str] = field(default_factory=list)
    c4_vehiculos: list[str] = field(default_factory=list)
    c5_deposito_extremos: list[str] = field(default_factory=list)

    def resumen(self) -> str:
        """
        Genera un string legible con todos los fallos detectados agrupados por condición.

        Returns:
            String multilínea con los fallos, o un mensaje de "Factible" si no hay ninguno.
        """
        bloques: list[str] = []  # Bloques de texto por condición

        # Por cada condición con fallos, añadimos un bloque con el encabezado y los mensajes.
        if self.c1_tareas_requeridas:
            bloques.append(
                "C1 — Tareas requeridas:\n" + "\n".join(f"  - {m}" for m in self.c1_tareas_requeridas)
            )
        if self.c2_consecutivas:
            bloques.append(
                "C2 — Conectividad entre tareas consecutivas:\n"
                + "\n".join(f"  - {m}" for m in self.c2_consecutivas)
            )
        if self.c3_capacidad:
            bloques.append(
                "C3 — Capacidad:\n" + "\n".join(f"  - {m}" for m in self.c3_capacidad)
            )
        if self.c4_vehiculos:
            bloques.append(
                "C4 — Vehículos disponibles:\n" + "\n".join(f"  - {m}" for m in self.c4_vehiculos)
            )
        if self.c5_deposito_extremos:
            bloques.append(
                "C5 — Depósito y extremos de ruta:\n"
                + "\n".join(f"  - {m}" for m in self.c5_deposito_extremos)
            )

        # Unimos los bloques con doble salto de línea; si no hay fallos, mensaje positivo.
        return "\n\n".join(bloques) if bloques else "Factible (sin incumplimientos registrados)."


# ---------------------------------------------------------------------------
# Clase de datos: FeasibilityResult
# ---------------------------------------------------------------------------
@dataclass
class FeasibilityResult:
    """
    Resultado de :func:`verificar_factibilidad`.

    Agrupa la bandera booleana de factibilidad y el detalle completo de fallos.

    El método especial ``__bool__`` permite usar el resultado directamente en
    condiciones: ``if result:`` es equivalente a ``if result.ok:``.
    Esto es un ejemplo del concepto OOP de "sobrecarga de operadores".
    """

    # True si la solución cumple TODAS las condiciones C1-C5; False en caso contrario.
    ok: bool

    # Detalle de todos los fallos detectados, organizado por condición.
    details: FeasibilityDetails

    def __bool__(self) -> bool:
        """
        Permite usar FeasibilityResult en condiciones booleanas.

        Ejemplo de uso:
            resultado = verificar_factibilidad(sol, data, matriz)
            if resultado:
                print("La solución es factible")
            else:
                print(resultado.details.resumen())
        """
        return self.ok


# ---------------------------------------------------------------------------
# Función principal: verificar_factibilidad
# ---------------------------------------------------------------------------
def verificar_factibilidad(
    solucion: Sequence[Sequence[Hashable]],     # Rutas de la solución con etiquetas
    data: Mapping[str, Any],                    # Datos de la instancia
    matriz_distancias: Any,                     # Matriz Dijkstra (dict o ndarray)
    *,
    marcador_depot_etiqueta: str | None = None, # Token de depósito (por defecto "D")
    usar_gpu: bool = False,                     # Reservado para futuro backend GPU
) -> FeasibilityResult:
    """
    Comprueba la factibilidad de una solución CARP según cinco condiciones (C1-C5).

    **Formato de solución por etiquetas:**
    Rutas como ``['D', 'TR1', 'TR5', ..., 'D']``, donde:
    - ``D`` (o el valor de ``marcador_depot_etiqueta``) marca el depósito.
    - ``TR*`` son etiquetas de tareas requeridas de ``LISTA_ARISTAS_REQ``.

    El marcador de depósito es solo ayuda visual y se elimina antes de las
    comprobaciones (C2-C5 ya asumen salida y regreso al depósito implícitamente).

    La matriz de distancias puede ser:
    - Un ``dict`` anidado: ``{nodo_u: {nodo_v: distancia}}``
    - Un ``numpy.ndarray`` 2D (1-indexed o 0-indexed, se detecta automáticamente).

    Args:
        solucion: Lista de rutas con etiquetas de texto.
        data: Diccionario de datos de la instancia.
        matriz_distancias: Matriz de distancias mínimas entre nodos.
        marcador_depot_etiqueta: Token de texto del depósito. Si None, usa el de data.
        usar_gpu: Reservado para futuro backend GPU; hoy siempre usa CPU.

    Returns:
        FeasibilityResult con ``ok=True`` si es factible, o con el detalle de fallos.
    """
    # Registramos el backend (por trazabilidad); los valores no se usan aún.
    _backend_solicitado, _backend_real = _aplicar_backend_gpu_placeholder(usar_gpu)

    # Creamos el objeto de detalle (inicialmente todas las listas vacías = sin fallos).
    det = FeasibilityDetails()

    # Construimos el mapa de tareas: {etiqueta: datos_tarea}.
    mapa_et = construir_mapa_tareas_por_etiqueta(data)

    # Si no hay tareas en la instancia, devolvemos infactible con un error en C1.
    if not mapa_et:
        return FeasibilityResult(
            False,
            FeasibilityDetails(
                c1_tareas_requeridas=["No hay LISTA_ARISTAS_REQ / LISTA_ARISTAS_NOREQ en data."]
            ),
        )

    # Normalizamos las rutas: eliminamos marcadores de depósito y validamos etiquetas.
    rutas_norm, err = normalizar_rutas_etiquetas(solucion, data, mapa_et, marcador_depot_etiqueta)
    if err:
        # Error de normalización (etiqueta desconocida, formato inválido, etc.).
        det.c1_tareas_requeridas.append(err)
        return FeasibilityResult(False, det)

    # --- Verificación C1: cobertura y unicidad de tareas requeridas ---

    # Conjunto de etiquetas de tareas REQUERIDAS (las que DEBEN aparecer en la solución).
    required_labels = etiquetas_tareas_requeridas(data)

    # Aplanamos todas las rutas en una sola lista de etiquetas.
    # Comprensión de lista anidada: recorre cada ruta y cada elemento dentro de ella.
    todas_e: list[str] = [x for r in rutas_norm for x in r]

    # Contamos cuántas veces aparece cada etiqueta en toda la solución.
    conteo_e: dict[str, int] = {}
    for lab in todas_e:
        conteo_e[lab] = conteo_e.get(lab, 0) + 1  # Incrementa o inicializa en 0

    # C1a: Unicidad — ninguna tarea debe aparecer más de una vez.
    for lab, k in conteo_e.items():
        if k > 1:
            det.c1_tareas_requeridas.append(
                f"La tarea {lab} aparece {k} veces (cada arista a lo sumo una vez)."
            )

    # C1b: Cobertura — todas las tareas requeridas deben aparecer.
    cub_e = set(conteo_e.keys())              # Conjunto de etiquetas que SÍ aparecen
    faltan_e = sorted(required_labels - cub_e)  # Tareas requeridas que NO aparecen

    if faltan_e:
        det.c1_tareas_requeridas.append(
            "Faltan tareas requeridas: "
            + ", ".join(faltan_e[:40])       # Mostramos hasta 40 para no saturar el log
            + (" ..." if len(faltan_e) > 40 else "")  # "..." si hay más de 40 faltantes
        )

    # --- Verificación C4: número de rutas vs. número de vehículos ---
    num_vehiculos = int(data.get("VEHICULOS", 0) or 0)

    # Solo contamos rutas activas (no vacías) para la comparación.
    rutas_activas = [r for r in rutas_norm if r]

    if len(rutas_activas) > num_vehiculos:
        det.c4_vehiculos.append(
            f"Se usan {len(rutas_activas)} rutas no vacías pero VEHICULOS={num_vehiculos}."
        )

    # --- Verificaciones C2, C3, C5: por ruta individual ---
    for idx, ruta in enumerate(rutas_norm):
        if not ruta:
            continue  # Las rutas vacías no tienen tareas que verificar

        # Verificamos las tres condiciones de esta ruta.
        f2, f3, f5 = _verificar_ruta(
            ruta, mapa_et, data, matriz_distancias, idx
        )

        # Acumulamos los fallos de esta ruta en los detalles globales.
        det.c2_consecutivas.extend(f2)
        det.c3_capacidad.extend(f3)
        det.c5_deposito_extremos.extend(f5)

    # La solución es factible si y solo si NINGUNA lista de fallos tiene contenido.
    # El operador 'not' sobre una lista devuelve True si está vacía, False si tiene elementos.
    ok = not (
        det.c1_tareas_requeridas
        or det.c2_consecutivas
        or det.c3_capacidad
        or det.c4_vehiculos
        or det.c5_deposito_extremos
    )

    return FeasibilityResult(ok, det)


# ---------------------------------------------------------------------------
# Función: verificar_factibilidad_desde_instancia
# ---------------------------------------------------------------------------
def verificar_factibilidad_desde_instancia(
    nombre_instancia: str,                              # Nombre de la instancia
    solucion: Sequence[Sequence[Hashable]],             # Rutas de la solución
    *,
    root: str | os.PathLike[str] | None = None,        # Directorio raíz alternativo
    marcador_depot_etiqueta: str | None = None,
    usar_gpu: bool = False,
) -> FeasibilityResult:
    """
    Versión de conveniencia: carga la instancia y la matriz por nombre y verifica factibilidad.

    Carga automáticamente los datos de la instancia (pickle) y la matriz Dijkstra
    precomputada del paquete, luego llama a :func:`verificar_factibilidad`.

    Args:
        nombre_instancia: Identificador de la instancia (ej. "EGL-E1-A").
        solucion: Lista de rutas con etiquetas de texto.
        root: Directorio raíz alternativo para buscar los archivos.
        marcador_depot_etiqueta: Token de texto del depósito.
        usar_gpu: Reservado para futuro backend GPU.

    Returns:
        FeasibilityResult con ``ok=True`` si es factible, o con el detalle de fallos.
    """
    # Importaciones diferidas: evitan ciclos de importación al cargar este módulo.
    from .cargar_matrices import cargar_matriz_dijkstra  # Carga la matriz Dijkstra del disco
    from .instances import load_instances                # Carga el pickle de la instancia

    # Cargamos los datos de la instancia.
    data = load_instances(nombre_instancia, root=root)

    # Cargamos la matriz Dijkstra precomputada.
    # A diferencia de costo_solucion, factibilidad siempre necesita la matriz
    # (no puede usar el grafo directamente, ya que trabaja con comparaciones de inf).
    matriz = cargar_matriz_dijkstra(nombre_instancia, root=root)

    # Delegamos al verificador principal con todos los parámetros.
    return verificar_factibilidad(
        solucion,
        data,
        matriz,
        marcador_depot_etiqueta=marcador_depot_etiqueta,
        usar_gpu=usar_gpu,
    )
