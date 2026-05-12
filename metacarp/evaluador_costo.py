"""
Evaluador rápido de costo para metaheurísticas CARP.

Diseño
======
Las metaheurísticas evalúan miles/millones de soluciones por corrida. Cada
``costo_solucion`` clásico recorre rutas y, por cada salto, llama a
``nx.shortest_path`` sobre el grafo. Eso convierte la evaluación en el cuello de
botella absoluto del experimento.

Este módulo construye **una sola vez** un *contexto vectorizado* a partir de la
matriz Dijkstra precomputada y de la codificación entera de tareas, y expone
funciones de evaluación O(longitud_de_ruta) por solución (por etiquetas o IDs).
La fórmula de costo se preserva: por cada tarea se suma el deadheading
(camino mínimo) + costo de servicio, y al final el regreso al depósito.

Backends
--------
- ``cpu``: NumPy puro, óptimo para una solución a la vez.
- ``gpu``: CuPy si está disponible; útil para evaluar **lotes** de soluciones
  (poblaciones de Tabu/Abejas/Cuckoo). Si CuPy no está disponible se hace
  fallback transparente a CPU.
"""
# Módulo: evaluador_costo.py
# Propósito: Evaluador de costo vectorizado para metaheurísticas CARP.
# Evita llamar a Dijkstra en cada evaluación usando una matriz de distancias
# precomputada (O(1) por consulta de distancia). Es el núcleo de rendimiento
# del proyecto cuando se ejecutan miles de iteraciones de búsqueda.

from __future__ import annotations  # Permite anotaciones de tipo forward-reference

import os                                       # Para rutas del sistema de archivos
from collections.abc import Mapping, Sequence  # Tipos abstractos de colecciones
from dataclasses import dataclass               # Para definir clases de datos inmutables
from typing import Any, Hashable               # Tipos genéricos para anotaciones

import networkx as nx   # Librería de grafos (usada para calcular APSP si no hay matriz)
import numpy as np      # NumPy: arrays numéricos de alto rendimiento

# Importamos la codificación entera de tareas (IDs en lugar de etiquetas de texto).
from .busqueda_indices import SearchEncoding, build_search_encoding

# Función para cargar la matriz Dijkstra precomputada desde disco.
from .cargar_matrices import cargar_matriz_dijkstra

# Constantes y funciones del módulo de formato estándar de soluciones.
from .solucion_formato import (
    CLAVE_MARCADOR_DEPOSITO_DEFAULT,        # Token de depósito por defecto ("D")
    construir_mapa_tareas_por_etiqueta,     # Crea {etiqueta: datos_tarea}
)

# Lista de nombres públicos exportados por este módulo.
__all__ = [
    "ContextoEvaluacion",
    "construir_contexto",
    "construir_contexto_desde_instancia",
    "costo_rapido",
    "costo_rapido_ids",
    "costo_lote_ids",
    "costo_lote_penalizado_ids",
    "exceso_capacidad_rapido",
    "objectivo_penalizado",
    "lambda_penal_capacidad_por_defecto",
    "gpu_disponible",
]


# Constante numérica: infinito en formato numpy float64.
# Se usa para indicar que no hay camino entre dos nodos en la matriz de distancias.
_INF = np.float64("inf")


# ---------------------------------------------------------------------------
# Función: gpu_disponible
# ---------------------------------------------------------------------------
def gpu_disponible() -> bool:
    """
    Detecta si CuPy está instalado y hay al menos un dispositivo CUDA accesible.

    CuPy es la versión GPU de NumPy; permite ejecutar operaciones de arrays
    en la tarjeta gráfica, lo que puede acelerar la evaluación de lotes grandes.

    Returns:
        True si CuPy importa correctamente y hay un dispositivo CUDA disponible.
        False en cualquier otro caso (sin GPU, sin CuPy, error de driver…).
    """
    try:
        import cupy as cp  # type: ignore  # CuPy puede no estar instalado
        try:
            # getDeviceCount() lanza excepción si no hay dispositivo CUDA.
            cp.cuda.runtime.getDeviceCount()
            return True
        except Exception:  # noqa: BLE001
            return False
    except Exception:  # noqa: BLE001
        # CuPy no está instalado en este entorno.
        return False


# ---------------------------------------------------------------------------
# Cache global: _CACHE_CONTEXTO
# ---------------------------------------------------------------------------
# Diccionario que almacena contextos de evaluación ya construidos para evitar
# recalcularlos en cada corrida del grid search. La clave es la tupla
# (nombre_instancia, backend) y el valor es el ContextoEvaluacion listo.
# Al vivir a nivel de módulo, persiste durante toda la sesión de Python.
_CACHE_CONTEXTO: dict[tuple[str, str | None], "ContextoEvaluacion"] = {}


# ---------------------------------------------------------------------------
# Clase de datos: ContextoEvaluacion
# ---------------------------------------------------------------------------
# @dataclass(frozen=True, slots=True):
#   - frozen=True: el objeto es inmutable; ningún atributo puede cambiarse tras crearlo.
#     Esto garantiza que el contexto compartido entre corridas de metaheurísticas
#     sea siempre coherente (no hay efectos secundarios accidentales).
#   - slots=True: usa __slots__ en lugar de __dict__ para los atributos → menor uso
#     de memoria y acceso más rápido, importante cuando se evalúan millones de soluciones.
@dataclass(frozen=True, slots=True)
class ContextoEvaluacion:
    """
    Contexto inmutable y compartido para evaluar costos de soluciones CARP.

    Agrupa todos los datos pre-procesados que necesitan los evaluadores rápidos:
    la matriz de distancias, los arrays de tareas y la codificación entera.
    Se construye una vez por instancia y se reutiliza en todas las iteraciones.

    Atributos clave:
    - ``dist``: matriz NumPy densa (N+1 × N+1) con distancias precomputadas.
      Indexada directamente por número de nodo (1-indexed), por eso tiene N+1 filas/cols.
      dist[i, j] = costo del camino mínimo del nodo i al nodo j.
    - ``u_arr`` / ``v_arr``: arrays NumPy con los nodos extremos de cada tarea,
      indexados por ID de tarea. u_arr[k] = nodo inicio de la tarea k.
    - ``costo_serv_arr``: array NumPy con el costo de servicio de cada tarea.
    - ``demanda_arr``: array NumPy con la demanda de cada tarea.
    - ``encoding``: SearchEncoding canónico (etiquetas ↔ IDs) de la instancia.
    - ``depot``: número entero del nodo depósito en el grafo.
    - ``backend_solicitado`` / ``backend_real``: para trazabilidad CPU vs GPU.
    - ``dist_gpu``: copia de la matriz en GPU (None si se usa CPU).
    - ``capacidad_max``: capacidad máxima por vehículo (inf si no aplica).
    """

    # Codificación entera de las tareas de la instancia.
    encoding: SearchEncoding

    # Matriz densa de distancias mínimas entre todos los pares de nodos.
    # Tipo np.ndarray con shape (N+1, N+1) donde N es el nodo con ID más alto.
    dist: np.ndarray

    # Arrays paralelos indexados por ID de tarea (np.ndarray de enteros).
    # u_arr[k] = nodo de inicio de la tarea k; v_arr[k] = nodo de fin.
    u_arr: np.ndarray
    v_arr: np.ndarray

    # Arrays paralelos indexados por ID de tarea (np.ndarray de floats).
    costo_serv_arr: np.ndarray  # Costo de servicio de cada tarea
    demanda_arr: np.ndarray     # Demanda de cada tarea

    # Número entero del nodo depósito en el grafo (ej. 1).
    depot: int

    # Token de texto del depósito en las rutas (ej. "D").
    marcador_depot: str

    # Strings que indican qué backend se pidió y cuál se usa realmente.
    # Útil para auditar logs: si el usuario pidió "gpu" pero no hay CUDA,
    # backend_real será "cpu" y se puede notificar al usuario.
    backend_solicitado: str
    backend_real: str

    # Copia de la matriz dist en memoria de la GPU (CuPy array).
    # Es None cuando el backend real es CPU.
    dist_gpu: Any | None = None

    # Capacidad máxima de cada vehículo. Si es inf, no se aplica restricción.
    capacidad_max: float = float("inf")

    # ---------------------------------------------------------------------------
    # Propiedad: usar_gpu
    # ---------------------------------------------------------------------------
    # @property convierte un método en un atributo de solo lectura.
    # En lugar de ctx.usar_gpu(), se accede como ctx.usar_gpu (sin paréntesis).
    # Esto es parte del concepto OOP de "encapsulamiento": el usuario no necesita
    # saber cómo se calcula; solo consulta el valor como si fuera un campo.
    @property
    def usar_gpu(self) -> bool:
        """True si el backend real es GPU y la matriz está cargada en GPU."""
        # Ambas condiciones deben cumplirse: backend configurado Y matriz disponible.
        return self.backend_real == "gpu" and self.dist_gpu is not None


# ---------------------------------------------------------------------------
# Función: lambda_penal_capacidad_por_defecto
# ---------------------------------------------------------------------------
def lambda_penal_capacidad_por_defecto(ctx: ContextoEvaluacion) -> float:
    """
    Calcula el factor lambda (λ) de penalización de capacidad para el objetivo penalizado.

    En metaheurísticas, cuando una solución viola la capacidad del vehículo,
    se penaliza su costo añadiendo λ × (exceso de demanda). Si λ es demasiado
    pequeño, el algoritmo ignora las violaciones; si es demasiado grande, rechaza
    soluciones buenas que podrían mejorar con ajustes menores.

    Esta función calibra λ automáticamente como ~10 veces la mediana del costo
    de los arcos de la instancia, lo que escala apropiadamente con el tamaño
    del grafo.

    Returns:
        Valor de λ como float. Mínimo garantizado: 10.0.
    """
    D = ctx.dist  # Matriz densa de distancias

    # Creamos una máscara booleana de valores finitos y positivos (excluye inf y 0).
    # np.isfinite(D): True donde D no es inf ni NaN.
    # D > 1e-12: True donde D es positivo (distancias reales, no la diagonal).
    fin = np.isfinite(D) & (D > 1e-12)

    # Si no hay ningún valor finito/positivo en la matriz, usamos un default conservador.
    if not np.any(fin):
        return 100.0

    # np.median calcula la mediana de los valores seleccionados por la máscara.
    # D[fin] aplica la máscara: devuelve un array 1D solo con los valores True de fin.
    mediana = float(np.median(D[fin]))

    # λ = max(mediana × 10, 10) garantiza un mínimo de 10 incluso en instancias pequeñas.
    return max(mediana * 10.0, 10.0)


# ---------------------------------------------------------------------------
# Función: exceso_capacidad_sol_ids
# ---------------------------------------------------------------------------
def exceso_capacidad_sol_ids(
    solucion_ids: Sequence[Sequence[int]],  # Solución como listas de IDs de tareas
    ctx: ContextoEvaluacion,
) -> float:
    """
    Calcula el exceso total de demanda sobre la capacidad máxima de la solución.

    Para cada ruta no vacía: exceso = max(0, demanda_total_ruta - capacidad_max).
    La suma de todos los excesos indica cuánto viola la solución la restricción C3.

    Si ``capacidad_max`` no es finita o es 0, devuelve 0 (sin restricción de capacidad).

    Args:
        solucion_ids: Lista de rutas, cada ruta es lista de IDs de tareas.
        ctx: Contexto de evaluación con demanda_arr y capacidad_max.

    Returns:
        Suma total de excesos de demanda (float ≥ 0).
    """
    cap = float(ctx.capacidad_max)  # Capacidad máxima del vehículo

    # Si la capacidad es no finita o 0, no hay restricción → exceso siempre cero.
    if cap <= 0 or not np.isfinite(cap):
        return 0.0

    dem = ctx.demanda_arr  # Array NumPy con demanda de cada tarea indexada por ID
    total_exc = 0.0         # Acumulador del exceso total

    for ruta in solucion_ids:
        if not ruta:
            continue  # Rutas vacías no aportan demanda

        # np.asarray convierte la lista Python a un array NumPy de enteros de 64 bits.
        # Esto permite usar fancy indexing en dem: dem[ids] extrae las demandas de
        # todas las tareas de esta ruta a la vez (vectorizado, sin bucle Python).
        ids = np.asarray(ruta, dtype=np.int64)

        # dem[ids] devuelve un array con las demandas de las tareas en esta ruta.
        # np.sum suma todos los valores del array en una sola operación NumPy.
        s = float(np.sum(dem[ids]))

        # Acumulamos el exceso: max(0, demanda_ruta - capacidad_max).
        total_exc += max(0.0, s - cap)

    return total_exc


# ---------------------------------------------------------------------------
# Función: exceso_capacidad_rapido
# ---------------------------------------------------------------------------
def exceso_capacidad_rapido(
    solucion_labels: Sequence[Sequence[Hashable]],  # Solución por etiquetas de texto
    ctx: ContextoEvaluacion,
) -> float:
    """
    Calcula el exceso de capacidad de una solución dada por etiquetas de texto.

    Versión de :func:`exceso_capacidad_sol_ids` que acepta rutas con etiquetas
    (como 'D', 'TR1', 'TR5') en lugar de IDs enteros. Convierte internamente y delega.

    Args:
        solucion_labels: Lista de rutas con etiquetas de texto.
        ctx: Contexto de evaluación.

    Returns:
        Suma total de excesos de demanda (float ≥ 0).
    """
    # Versión en mayúsculas del marcador de depósito para comparación case-insensitive.
    md = ctx.marcador_depot.upper()

    # Diccionario de mapeo etiqueta → ID del encoding del contexto.
    label_to_id = ctx.encoding.label_to_id

    # Convertimos todas las rutas de etiquetas a IDs antes de llamar al evaluador.
    rutas_ids: list[list[int]] = []
    for ruta in solucion_labels:
        rutas_ids.append(_ruta_labels_a_ids(ruta, label_to_id, md))

    return exceso_capacidad_sol_ids(rutas_ids, ctx)


# ---------------------------------------------------------------------------
# Función: objectivo_penalizado
# ---------------------------------------------------------------------------
def objectivo_penalizado(
    costo_puro: float,      # Costo real de la solución (sin penalización)
    violacion_cap: float,   # Exceso total de capacidad (≥ 0)
    *,
    usar_penal: bool,       # Si False, se devuelve solo el costo puro
    lam: float,             # Factor λ de penalización
) -> float:
    """
    Calcula el objetivo penalizado: ``costo_puro + λ × violación_capacidad``.

    En las metaheurísticas, penalizar las violaciones de capacidad permite al
    algoritmo explorar soluciones infactibles temporalmente, siempre que el
    costo adicional λ × violación las desincentive suficientemente.

    Args:
        costo_puro: Costo calculado sin penalización.
        violacion_cap: Exceso total de demanda sobre la capacidad.
        usar_penal: Si False, devuelve siempre el costo puro (sin penalizar).
        lam: Factor de escala de la penalización (λ).

    Returns:
        Valor escalar del objetivo penalizado.
    """
    # Si la penalización está desactivada, o si no hay violación, devolvemos el costo puro.
    if not usar_penal or violacion_cap <= 0:
        return float(costo_puro)

    # Objetivo penalizado = costo_puro + λ × violación
    return float(costo_puro) + lam * float(violacion_cap)


# ---------------------------------------------------------------------------
# Función interna: _matriz_dijkstra_densa
# ---------------------------------------------------------------------------
def _matriz_dijkstra_densa(
    dijkstra: Any,              # Matriz Dijkstra precomputada (dict, ndarray, o None)
    *,
    G: nx.Graph | None = None,  # Grafo alternativo si no hay matriz precomputada
) -> np.ndarray:
    """
    Convierte la matriz Dijkstra a un ``np.ndarray`` denso 1-indexed.

    Soporta tres formatos de entrada:
    1. ``None`` + grafo G: calcula APSP (All-Pairs Shortest Path) con NetworkX.
    2. ``np.ndarray``: devuelve el array tal cual (ya está en formato correcto).
    3. ``dict`` anidado {u: {v: dist}}: convierte al array denso.

    El resultado es siempre una matriz 2D donde ``D[i, j]`` es el costo del
    camino mínimo del nodo i al nodo j. Se usa indexación 1-based para que los
    nodos del grafo (que empiezan en 1) se mapeen directamente sin restar 1.

    Args:
        dijkstra: Matriz de distancias en cualquiera de los formatos soportados.
        G: Grafo NetworkX (solo necesario si dijkstra es None).

    Returns:
        np.ndarray 2D de float64 con distancias mínimas.

    Raises:
        ValueError: Si dijkstra es None y G tampoco se proporciona.
        TypeError: Si el formato de dijkstra no es soportado.
    """
    if dijkstra is None:
        # Caso 1: No hay matriz precomputada → calculamos APSP con NetworkX.
        if G is None:
            raise ValueError("Falta dijkstra y G para reconstruir distancias.")

        # Extraemos los nodos como enteros y encontramos el ID máximo.
        nodos = sorted(int(n) for n in G.nodes())
        idx_max = max(nodos)

        # Creamos una matriz (idx_max+1) × (idx_max+1) llena de infinito.
        # np.full crea un array con todas las celdas inicializadas al valor dado.
        D = np.full((idx_max + 1, idx_max + 1), _INF, dtype=np.float64)

        # Calculamos las distancias desde cada nodo con Dijkstra de una sola fuente.
        # Es más eficiente que llamar nx.shortest_path_length por cada par (u, v).
        for u_str in G.nodes():
            # single_source_dijkstra_path_length devuelve {v_str: distancia} para
            # todos los v alcanzables desde u_str.
            length = nx.single_source_dijkstra_path_length(G, u_str, weight="cost")
            u = int(u_str)  # Convertimos el nodo de string a entero para indexar
            for v_str, d in length.items():
                D[u, int(v_str)] = float(d)
        return D

    if isinstance(dijkstra, np.ndarray):
        # Caso 2: Ya es un array NumPy → solo garantizamos el tipo float64.
        # copy=False evita copiar si ya es float64 (ahorra memoria).
        return dijkstra.astype(np.float64, copy=False)

    if isinstance(dijkstra, Mapping):
        # Caso 3: Es un diccionario anidado {u: {v: dist}}.
        # Primero encontramos el ID máximo de nodo para dimensionar la matriz.
        keys = list(dijkstra.keys())
        try:
            idx_max = max(int(k) for k in keys)
            # También consideramos los nodos destino al calcular el máximo.
            for k, fila in dijkstra.items():
                if isinstance(fila, Mapping):
                    if fila:
                        idx_max = max(idx_max, max(int(j) for j in fila.keys()))
        except (TypeError, ValueError) as exc:
            raise ValueError("Las claves de la matriz Dijkstra deben ser enteros.") from exc

        # Creamos la matriz densa inicializada a infinito.
        D = np.full((idx_max + 1, idx_max + 1), _INF, dtype=np.float64)

        # Rellenamos la matriz con las distancias del diccionario.
        for k, fila in dijkstra.items():
            i = int(k)  # Nodo origen
            if isinstance(fila, Mapping):
                # Sub-diccionario {v: distancia}
                for j, d in fila.items():
                    D[i, int(j)] = float(d)
            else:
                # fila es un array o lista indexable por entero.
                for j, d in enumerate(fila):
                    D[i, j] = float(d)
        return D

    raise TypeError(f"Formato de matriz Dijkstra no soportado: {type(dijkstra).__name__}")


# ---------------------------------------------------------------------------
# Función: construir_contexto
# ---------------------------------------------------------------------------
def construir_contexto(
    data: Mapping[str, Any],        # Datos completos de la instancia
    *,
    dijkstra: Any | None = None,    # Matriz Dijkstra precomputada (opcional)
    G: nx.Graph | None = None,      # Grafo de la instancia (alternativa a dijkstra)
    usar_gpu: bool = False,         # Si True, intenta copiar la matriz a GPU
    encoding: SearchEncoding | None = None,  # Encoding preexistente (se crea si None)
) -> ContextoEvaluacion:
    """
    Construye un contexto reutilizable de evaluación para una instancia CARP.

    Prepara todos los datos en formato NumPy para evaluación vectorizada O(1)
    por consulta de distancia. Este contexto se debe construir una sola vez y
    reutilizar en todas las iteraciones de la metaheurística.

    Si ``dijkstra`` es None y se pasa ``G``, computa APSP con NetworkX una sola vez.
    Cuando ``usar_gpu=True`` y CuPy está disponible, copia la matriz dist a GPU
    para evaluación por lotes acelerada; si no, hace fallback transparente a CPU.

    Args:
        data: Diccionario de datos de la instancia.
        dijkstra: Matriz Dijkstra precomputada (dict o ndarray). Si None, se usa G.
        G: Grafo NetworkX de la instancia. Usado si dijkstra es None.
        usar_gpu: Si True, intenta usar CuPy/GPU para los arrays de distancias.
        encoding: SearchEncoding preexistente. Si None, se construye desde data.

    Returns:
        ContextoEvaluacion inmutable listo para evaluar soluciones.
    """
    # Construimos el encoding si no se proporcionó uno.
    # El operador 'or' evalúa el segundo operando solo si el primero es falsy (None).
    enc = encoding or build_search_encoding(data)

    # Verificamos que la instancia tenga tareas.
    mapa = construir_mapa_tareas_por_etiqueta(data)
    if not mapa:
        raise ValueError("La instancia no tiene tareas (LISTA_ARISTAS_REQ/NOREQ vacías).")

    # Convertimos la matriz Dijkstra a formato NumPy denso 1-indexed.
    D = _matriz_dijkstra_densa(dijkstra, G=G)

    # Número total de tareas en la instancia.
    n = len(enc.id_to_label)

    # Convertimos las listas del encoding a arrays NumPy para acceso vectorizado.
    # dtype=np.int64 garantiza enteros de 64 bits, necesario para indexar matrices grandes.
    u_arr = np.asarray(enc.u, dtype=np.int64)
    v_arr = np.asarray(enc.v, dtype=np.int64)

    # dtype=np.float64: precisión doble para los costos y demandas.
    costo_serv_arr = np.asarray(enc.costo_serv, dtype=np.float64)
    demanda_arr = np.asarray(enc.demanda, dtype=np.float64)

    # Verificación de coherencia: los arrays deben tener exactamente n elementos.
    if u_arr.shape[0] != n or v_arr.shape[0] != n:
        raise ValueError("Tamaño de arrays de tareas incoherente con el encoding.")

    # Nodo depósito como entero (por defecto 1 si no está en data).
    depot = int(data.get("DEPOSITO", 1))

    # Token de texto del depósito, normalizado a mayúsculas sin espacios.
    marcador_depot = str(
        data.get("MARCADOR_DEPOT_ETIQUETA") or CLAVE_MARCADOR_DEPOSITO_DEFAULT
    ).strip().upper() or CLAVE_MARCADOR_DEPOSITO_DEFAULT

    # Capacidad máxima del vehículo; si es 0 o negativa, se considera sin restricción.
    cap_raw = float(data.get("CAPACIDAD", 0) or 0)
    capacidad_max = cap_raw if cap_raw > 0 else float("inf")

    # Configuración del backend (CPU vs GPU).
    backend_solicitado = "gpu" if usar_gpu else "cpu"
    backend_real = "cpu"       # Por defecto siempre CPU
    dist_gpu: Any | None = None  # La copia GPU de la matriz (None si no aplica)

    if usar_gpu and gpu_disponible():
        try:
            import cupy as cp  # type: ignore

            # cp.asarray copia la matriz NumPy a la memoria de la GPU.
            dist_gpu = cp.asarray(D)
            backend_real = "gpu"
        except Exception:  # noqa: BLE001
            # Si falla la copia a GPU por cualquier razón, revertimos a CPU.
            backend_real = "cpu"
            dist_gpu = None

    # Construimos y devolvemos el contexto inmutable con todos los datos listos.
    return ContextoEvaluacion(
        encoding=enc,
        dist=D,
        u_arr=u_arr,
        v_arr=v_arr,
        costo_serv_arr=costo_serv_arr,
        demanda_arr=demanda_arr,
        depot=depot,
        marcador_depot=marcador_depot,
        backend_solicitado=backend_solicitado,
        backend_real=backend_real,
        dist_gpu=dist_gpu,
        capacidad_max=capacidad_max,
    )


# ---------------------------------------------------------------------------
# Función: construir_contexto_desde_instancia
# ---------------------------------------------------------------------------
def construir_contexto_desde_instancia(
    nombre_instancia: str,                           # Nombre de la instancia (sin extensión)
    *,
    root: str | os.PathLike[str] | None = None,     # Directorio raíz alternativo
    usar_gpu: bool = False,
) -> ContextoEvaluacion:
    """
    Construye el contexto de evaluación cargando la instancia por nombre.

    Carga automáticamente los datos de la instancia (pickle) y la matriz
    Dijkstra precomputada. Si la matriz no existe, la calcula con NetworkX
    desde el grafo GEXF. Usa caché global para evitar recomputar en el
    mismo proceso (útil en grid search con múltiples corridas).

    Args:
        nombre_instancia: Identificador de la instancia (ej. "EGL-E1-A").
        root: Directorio raíz donde buscar los archivos. Si None, usa el
            directorio por defecto del paquete.
        usar_gpu: Si True, intenta construir el contexto con backend GPU.

    Returns:
        ContextoEvaluacion construido (posiblemente desde caché).
    """
    # Importación diferida: evita importar todo el paquete al cargar este módulo.
    from .instances import load_instances

    # La clave de caché incluye el backend para no mezclar contextos CPU y GPU.
    backend = "gpu" if usar_gpu else "cpu"
    cache_key = (nombre_instancia, backend)

    # Si ya construimos este contexto antes, lo devolvemos directamente.
    cached = _CACHE_CONTEXTO.get(cache_key)
    if cached is not None:
        return cached

    # Cargamos los datos de la instancia desde el archivo pickle.
    data = load_instances(nombre_instancia, root=root)

    try:
        # Intentamos cargar la matriz Dijkstra precomputada desde disco.
        dijkstra = cargar_matriz_dijkstra(nombre_instancia, root=root)
    except FileNotFoundError:
        # Si no existe la matriz precomputada, la calculamos desde el grafo GEXF.
        # La importación diferida evita dependencias circulares.
        from .cargar_grafos import cargar_objeto_gexf

        G = cargar_objeto_gexf(nombre_instancia, root=root)
        ctx = construir_contexto(data, dijkstra=None, G=G, usar_gpu=usar_gpu)
    else:
        # Tenemos la matriz → construimos el contexto directamente.
        ctx = construir_contexto(data, dijkstra=dijkstra, usar_gpu=usar_gpu)

    # Guardamos en caché para futuras llamadas en este proceso.
    _CACHE_CONTEXTO[cache_key] = ctx
    return ctx


# ---------------------------------------------------------------------------
# Evaluadores rápidos
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Función interna: _ruta_labels_a_ids
# ---------------------------------------------------------------------------
def _ruta_labels_a_ids(
    ruta: Sequence[Hashable],            # Ruta con etiquetas de texto o depósito
    label_to_id: Mapping[str, int],      # Mapeo etiqueta → ID del encoding
    marcador_depot_upper: str,           # Token de depósito en mayúsculas (ej. "D")
) -> list[int]:
    """
    Convierte una ruta de etiquetas de texto a una lista de IDs enteros.

    Omite silenciosamente el marcador de depósito. Busca la etiqueta primero
    de forma exacta, luego en mayúsculas, y finalmente con una búsqueda
    lineal case-insensitive como último recurso.

    Args:
        ruta: Lista de tokens de la ruta (etiquetas de tareas y/o depósito).
        label_to_id: Diccionario de mapeo del encoding.
        marcador_depot_upper: Token de depósito normalizado a mayúsculas.

    Returns:
        Lista de IDs enteros de las tareas de la ruta (sin token de depósito).

    Raises:
        KeyError: Si una etiqueta no se encuentra en ningún formato.
    """
    ids: list[int] = []

    for tok in ruta:
        s = str(tok).strip()   # Normalizamos a string sin espacios

        if not s:
            continue  # Ignoramos tokens vacíos

        if s.upper() == marcador_depot_upper:
            continue  # Ignoramos el marcador de depósito

        # Intento 1: búsqueda exacta (más frecuente y más rápida).
        idx = label_to_id.get(s)

        if idx is None:
            # Intento 2: la etiqueta en mayúsculas (cubre "tr1" → "TR1").
            idx = label_to_id.get(s.upper())

        if idx is None:
            # Intento 3: búsqueda lineal case-insensitive como último recurso.
            # Este bucle es O(n) pero rara vez se ejecuta gracias a los intentos previos.
            for k, vid in label_to_id.items():
                if k.upper() == s.upper():
                    idx = vid
                    break

        if idx is None:
            raise KeyError(f"Etiqueta de tarea desconocida: {tok!r}")

        ids.append(idx)
    return ids


# ---------------------------------------------------------------------------
# Función: costo_rapido_ids
# ---------------------------------------------------------------------------
def costo_rapido_ids(
    solucion_ids: Sequence[Sequence[int]],  # Solución como listas de IDs de tareas
    ctx: ContextoEvaluacion,
) -> float:
    """
    Calcula el costo total de una solución dada por listas de IDs enteros.

    Es el evaluador principal de las metaheurísticas. Produce el mismo resultado
    que ``costo_solucion`` pero sin llamar a Dijkstra: usa la matriz dist pre-
    computada para acceder a distancias en O(1).

    Fórmula por ruta:
        costo = dist(depot, u_tarea_0) + costo_serv_tarea_0
              + sum_{k=1}^{n-1} [ dist(v_{k-1}, u_k) + costo_serv_k ]
              + dist(v_{n-1}, depot)

    Implementación vectorizada con NumPy: por cada ruta construye arrays de
    orígenes y destinos, luego extrae todas las distancias con fancy indexing
    (``dist[origen_prev, us]``) en una sola operación de array.

    Args:
        solucion_ids: Lista de rutas, cada una es una lista de IDs de tareas.
        ctx: Contexto de evaluación con la matriz dist y los arrays de tareas.

    Returns:
        Costo total de la solución como float.
    """
    # Acceso a los componentes del contexto (locales para evitar lookups repetidos).
    dist = ctx.dist          # Matriz densa de distancias mínimas
    u_arr = ctx.u_arr        # Nodos de inicio de cada tarea
    v_arr = ctx.v_arr        # Nodos de fin de cada tarea
    cs_arr = ctx.costo_serv_arr  # Costos de servicio de cada tarea
    depot = ctx.depot        # Nodo depósito

    total = 0.0  # Acumulador del costo total de la solución

    for ruta in solucion_ids:
        if not ruta:
            continue  # Las rutas vacías no tienen costo

        # np.asarray convierte la lista de IDs a un array NumPy int64.
        # Esto habilita el "fancy indexing": u_arr[ids] extrae un subarray de una vez.
        ids = np.asarray(ruta, dtype=np.int64)

        # us[k] = nodo de inicio de la k-ésima tarea en esta ruta.
        # vs[k] = nodo de fin de la k-ésima tarea en esta ruta.
        # Ambas operaciones son O(n_tareas) con NumPy (sin bucle Python).
        us = u_arr[ids]
        vs = v_arr[ids]

        # Construimos el array de "nodo previo" para calcular el deadheading:
        # - Para la primera tarea (k=0): el nodo previo es el depósito.
        # - Para las demás (k>0): el nodo previo es v de la tarea anterior (vs[k-1]).
        origen_prev = np.empty_like(us)  # Array vacío del mismo tamaño y tipo que us
        origen_prev[0] = depot           # Primera tarea siempre parte del depósito

        if us.shape[0] > 1:
            # vs[:-1] = todos los nodos v excepto el último (son los orígenes previos
            # para las tareas 1, 2, …, n-1).
            origen_prev[1:] = vs[:-1]

        # dist[origen_prev, us] es "fancy indexing" 2D de NumPy:
        # Extrae dist[origen_prev[0], us[0]], dist[origen_prev[1], us[1]], …
        # en un solo acceso vectorizado al array 2D (sin bucle Python).
        # dh[k] = costo de deadheading antes de servir la k-ésima tarea.
        dh = dist[origen_prev, us]

        # Sumamos DH de todas las tareas + costos de servicio de todas las tareas.
        # .sum() suma todos los elementos del array en una operación NumPy (O(n) C).
        total += float(dh.sum()) + float(cs_arr[ids].sum())

        # Añadimos el costo de regreso al depósito desde el nodo final de la última tarea.
        # vs[-1] = nodo v de la última tarea; dist[vs[-1], depot] = costo del retorno.
        total += float(dist[vs[-1], depot])

    return total


# ---------------------------------------------------------------------------
# Función: costo_rapido
# ---------------------------------------------------------------------------
def costo_rapido(
    solucion_labels: Sequence[Sequence[Hashable]],  # Solución con etiquetas de texto
    ctx: ContextoEvaluacion,
) -> float:
    """
    Calcula el costo total de una solución dada por etiquetas de texto.

    Acepta el formato estándar con marcador de depósito (ej. 'D').
    Convierte internamente las etiquetas a IDs y delega en :func:`costo_rapido_ids`.

    Args:
        solucion_labels: Lista de rutas con etiquetas de texto y marcador de depósito.
        ctx: Contexto de evaluación de la instancia.

    Returns:
        Costo total de la solución como float.
    """
    # Marcador de depósito normalizado para comparación case-insensitive.
    md = ctx.marcador_depot.upper()

    # Diccionario etiqueta → ID del encoding del contexto.
    label_to_id = ctx.encoding.label_to_id

    # Convertimos todas las rutas de etiquetas a IDs.
    rutas_ids: list[list[int]] = []
    for ruta in solucion_labels:
        rutas_ids.append(_ruta_labels_a_ids(ruta, label_to_id, md))

    # Delegamos al evaluador vectorizado por IDs.
    return costo_rapido_ids(rutas_ids, ctx)


# ---------------------------------------------------------------------------
# Evaluación por lotes (GPU opcional)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Función interna: _empaquetar_lote_ids
# ---------------------------------------------------------------------------
def _empaquetar_lote_ids(
    soluciones_ids: Sequence[Sequence[Sequence[int]]],  # Lote de soluciones
    ctx: ContextoEvaluacion,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Empaqueta un lote heterogéneo de soluciones en arrays planos para evaluación vectorizada.

    Las soluciones pueden tener diferente número de rutas y de tareas por ruta.
    Esta función las "aplana" en arrays 1D paralelos, añadiendo un índice de
    solución para poder sumar los costos por solución al final.

    Returns:
        Tupla de 4 arrays NumPy de igual longitud (n_pasos_totales,):
        - ``orig``: nodo origen de cada paso (depósito o v de la tarea anterior).
        - ``dest``: nodo destino de cada paso (u de la tarea actual o depósito).
        - ``cs``: costo de servicio de cada paso (0 para el regreso al depósito).
        - ``sol_idx``: índice de la solución a la que pertenece cada paso.
    """
    u_arr = ctx.u_arr
    v_arr = ctx.v_arr
    cs_arr = ctx.costo_serv_arr
    depot = ctx.depot

    # Listas Python para acumular antes de convertir a NumPy.
    origs: list[int] = []    # Nodos origen de cada paso
    dests: list[int] = []    # Nodos destino de cada paso
    cs_l: list[float] = []   # Costos de servicio de cada paso
    sol_idx: list[int] = []  # Índice de la solución dueña de cada paso

    for s_idx, sol in enumerate(soluciones_ids):
        # s_idx es el índice de esta solución en el lote (0, 1, 2, …).
        for ruta in sol:
            if not ruta:
                continue  # Rutas vacías no aportan pasos

            ids = np.asarray(ruta, dtype=np.int64)
            us = u_arr[ids]  # Nodos de inicio de las tareas de esta ruta
            vs = v_arr[ids]  # Nodos de fin de las tareas de esta ruta
            n = us.shape[0]  # Número de tareas en esta ruta

            # ---- Pasos de servicio (depot → u_0, v_0 → u_1, …, v_{n-2} → u_{n-1}) ----
            # Primer paso: del depósito al nodo inicio de la primera tarea.
            origs.append(depot)
            # Pasos intermedios: del nodo fin de la tarea anterior al nodo inicio de la siguiente.
            # vs[:-1].tolist() = [v_0, v_1, …, v_{n-2}] (todos los v excepto el último).
            origs.extend(vs[:-1].tolist() if n > 1 else [])
            # Los destinos son los nodos de inicio de todas las tareas: [u_0, u_1, …, u_{n-1}].
            dests.extend(us.tolist())
            # Los costos de servicio se asocian al destino (tarea que se va a servir).
            cs_l.extend(cs_arr[ids].tolist())
            # Todos estos pasos pertenecen a la solución s_idx.
            sol_idx.extend([s_idx] * n)

            # ---- Paso de regreso al depósito (v_{n-1} → depot) ----
            origs.append(int(vs[-1]))   # Nodo fin de la última tarea
            dests.append(depot)          # Destino: el depósito
            cs_l.append(0.0)             # El regreso no tiene costo de servicio
            sol_idx.append(s_idx)

    # Convertimos las listas a arrays NumPy para la evaluación vectorizada.
    return (
        np.asarray(origs, dtype=np.int64),
        np.asarray(dests, dtype=np.int64),
        np.asarray(cs_l, dtype=np.float64),
        np.asarray(sol_idx, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Función: costo_lote_ids
# ---------------------------------------------------------------------------
def costo_lote_ids(
    soluciones_ids: Sequence[Sequence[Sequence[int]]],  # Lote de soluciones por IDs
    ctx: ContextoEvaluacion,
) -> np.ndarray:
    """
    Evalúa un lote completo de soluciones y devuelve un array de costos.

    Es la función estrella para metaheurísticas con población (Abejas, Cuckoo):
    evalúa todas las soluciones del lote en pocas operaciones NumPy/CuPy, evitando
    bucles Python por solución.

    Si el contexto usa backend GPU real, toda la reducción se realiza en GPU;
    el resultado regresa a memoria del host como NumPy.

    Args:
        soluciones_ids: Lista de soluciones, cada una es lista de rutas de IDs.
        ctx: Contexto de evaluación de la instancia.

    Returns:
        np.ndarray de shape (n_soluciones,) con el costo total de cada solución.
    """
    n_sol = len(soluciones_ids)

    # Caso borde: lote vacío → devolvemos array vacío.
    if n_sol == 0:
        return np.zeros((0,), dtype=np.float64)

    # Empaquetamos todas las soluciones en arrays planos paralelos.
    orig, dest, cs, sol_idx = _empaquetar_lote_ids(soluciones_ids, ctx)

    # Caso borde: todas las rutas estaban vacías.
    if orig.size == 0:
        return np.zeros((n_sol,), dtype=np.float64)

    if ctx.usar_gpu:
        # ---- Evaluación en GPU con CuPy ----
        import cupy as cp  # type: ignore

        d_gpu = ctx.dist_gpu  # Matriz de distancias ya en memoria GPU

        # Transferimos los arrays de índices a la GPU con cp.asarray.
        orig_g = cp.asarray(orig)
        dest_g = cp.asarray(dest)
        cs_g = cp.asarray(cs)
        sol_g = cp.asarray(sol_idx)

        # Evaluamos dist[orig, dest] + cs para todos los pasos en paralelo en GPU.
        # d_gpu[orig_g, dest_g] usa fancy indexing en GPU: extrae dist[orig[i], dest[i]]
        # para cada i en una operación vectorizada.
        contrib = d_gpu[orig_g, dest_g] + cs_g

        # Array de salida: un costo por solución, inicializado a 0.
        out = cp.zeros((n_sol,), dtype=cp.float64)

        # scatter_add suma contrib[i] en out[sol_g[i]] para cada i.
        # Es la operación "reducción por grupo" en GPU.
        try:
            cupyx_scatter = cp.scatter_add
        except AttributeError:
            import cupyx
            cupyx_scatter = cupyx.scatter_add

        cupyx_scatter(out, sol_g, contrib)

        # Traemos el resultado de vuelta a memoria del host como NumPy.
        return cp.asnumpy(out)

    # ---- Evaluación en CPU con NumPy ----
    # dist[orig, dest]: fancy indexing 2D → array 1D con la distancia de cada paso.
    # + cs: suma element-wise del costo de servicio de cada paso.
    contrib = ctx.dist[orig, dest] + cs

    # Array de salida inicializado a 0: un acumulador por solución.
    out = np.zeros((n_sol,), dtype=np.float64)

    # np.add.at(out, sol_idx, contrib): para cada índice i, suma contrib[i] en out[sol_idx[i]].
    # Es el equivalente CPU de scatter_add: acumula las contribuciones de cada paso
    # en el costo de la solución correspondiente.
    np.add.at(out, sol_idx, contrib)

    return out


# ---------------------------------------------------------------------------
# Función: costo_lote_penalizado_ids
# ---------------------------------------------------------------------------
def costo_lote_penalizado_ids(
    soluciones_ids: Sequence[Sequence[Sequence[int]]],  # Lote de soluciones por IDs
    ctx: ContextoEvaluacion,
    lam: float,                  # Factor de penalización λ
    *,
    usar_penal: bool = True,     # Si False, devuelve solo el costo puro
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evalúa un lote con objetivo penalizado: ``costo_puro + λ × violación_capacidad``.

    Combina la evaluación de costo (:func:`costo_lote_ids`) con el cálculo de
    exceso de capacidad para producir el objetivo que guía la metaheurística
    cuando hay restricciones de capacidad activas.

    Args:
        soluciones_ids: Lote de soluciones, cada una como lista de rutas de IDs.
        ctx: Contexto de evaluación de la instancia.
        lam: Factor λ de penalización de capacidad.
        usar_penal: Si False, devuelve el costo puro sin penalización.

    Returns:
        Tupla de 3 arrays NumPy de shape (n_soluciones,):
        - ``objetivo``: costo_puro + λ × violación (el que minimiza la metaheurística).
        - ``costo_puro``: costo sin penalización.
        - ``violacion``: exceso total de capacidad de cada solución.
    """
    # Calculamos el costo puro del lote completo con evaluación vectorizada.
    base = costo_lote_ids(soluciones_ids, ctx)

    n = len(soluciones_ids)

    # Caso borde: lote vacío.
    if n == 0:
        z = np.zeros((0,), dtype=np.float64)
        return z, z, z

    # Si la penalización está desactivada o no hay restricción de capacidad,
    # devolvemos el costo puro como objetivo (violación = 0 para todos).
    if (
        not usar_penal
        or not np.isfinite(ctx.capacidad_max)
        or float(ctx.capacidad_max) <= 0
    ):
        z = np.zeros_like(base)  # Array de ceros del mismo shape y tipo que base
        return base.copy(), base, z

    # Calculamos el exceso de capacidad para cada solución individualmente.
    exc = np.zeros((n,), dtype=np.float64)
    for i, sid in enumerate(soluciones_ids):
        exc[i] = exceso_capacidad_sol_ids(sid, ctx)

    # Objetivo penalizado = costo_puro + λ × exceso (operación element-wise NumPy).
    obj = base + float(lam) * exc

    return obj, base, exc
