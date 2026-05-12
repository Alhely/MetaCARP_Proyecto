# =============================================================================
# __init__.py — Punto de entrada del paquete `metacarp`
#
# En Python, cada carpeta que contiene un archivo llamado `__init__.py` se
# convierte en un "paquete". Este archivo se ejecuta automáticamente la primera
# vez que alguien hace `import metacarp` o `from metacarp import ...`.
#
# Su propósito aquí es:
#   1. Importar funciones y clases de los submódulos internos (archivos .py de
#      esta misma carpeta, referenciados con el punto inicial ".").
#   2. Declararlos en `__all__`, que es la lista oficial de lo que este paquete
#      expone hacia afuera; es lo que ve un usuario al hacer `from metacarp import *`.
# =============================================================================

# --- Carga de grafos e imágenes ---
# Importa funciones para leer el grafo de la instancia CARP desde disco,
# ya sea como imagen PNG o como archivo de red GEXF (NetworkX).
from .cargar_grafos import (
    cargar_grafo,            # Función unificada: carga imagen O grafo según parámetro
    cargar_imagen_estatica,  # Carga el PNG estático de la instancia
    cargar_objeto_gexf,      # Carga el grafo NetworkX desde archivo .gexf
    ruta_gexf,               # Devuelve solo la ruta al .gexf (sin leerlo)
    ruta_imagen_estatica,    # Devuelve solo la ruta al PNG (sin leerlo)
)

# --- Matrices de distancias ---
# Importa funciones para leer la matriz de caminos mínimos (Dijkstra),
# precalculada y guardada como archivo pickle.
from .cargar_matrices import (
    cargar_matriz_dijkstra,          # Lee el pickle y devuelve la matriz numpy
    nombres_matrices_disponibles,    # Lista qué instancias ya tienen su matriz guardada
    ruta_matriz_dijkstra,            # Devuelve la ruta al pickle (sin leerlo)
)

# --- Soluciones iniciales ---
# Importa funciones para cargar soluciones de partida (construidas manualmente
# o con heurísticas) guardadas también como pickles.
from .cargar_soluciones_iniciales import (
    cargar_solucion_inicial,                    # Lee el pickle de la solución inicial
    nombres_soluciones_iniciales_disponibles,   # Lista qué instancias tienen solución inicial
    ruta_solucion_inicial,                      # Devuelve la ruta al pickle (sin leerlo)
)

# --- Codificación para búsqueda (índices numéricos) ---
# Los metaheurísticos trabajan con vectores de enteros (índices) en lugar de
# etiquetas de texto. Este submódulo convierte entre los dos formatos.
from .busqueda_indices import (
    DEPOT_ID,              # Constante: ID numérico especial que representa el depósito
    SearchEncoding,        # Clase que guarda la tabla de conversión etiqueta <-> índice
    build_search_encoding, # Construye un SearchEncoding a partir de una instancia
    decode_solution,       # Convierte lista de índices → lista de etiquetas de tareas
    decode_task_ids,       # Variante: convierte IDs de tarea a sus nombres originales
    encode_solution,       # Convierte lista de etiquetas → lista de índices numéricos
)

# --- Evaluación del costo de una solución ---
# Calcula el costo total (distancia recorrida) de una solución CARP.
from .costo_solucion import (
    CostoSolucionResult,            # Clase de resultado con el costo y detalles
    costo_solucion,                 # Calcula costo dada una solución y sus datos
    costo_solucion_desde_instancia, # Versión que toma el nombre de instancia directamente
)

# --- Evaluador rápido de costo (con soporte GPU) ---
# Versión optimizada del evaluador que puede usar GPU para evaluar muchas
# soluciones al mismo tiempo (evaluación en lote).
from .evaluador_costo import (
    ContextoEvaluacion,                  # Clase que precarga datos para evaluaciones rápidas
    construir_contexto,                  # Construye un ContextoEvaluacion desde datos
    construir_contexto_desde_instancia,  # Igual pero toma el nombre de la instancia
    costo_lote_ids,                      # Evalúa muchas soluciones (por IDs) de una vez
    costo_rapido,                        # Evaluación rápida de una sola solución
    costo_rapido_ids,                    # Evaluación rápida usando IDs numéricos
    gpu_disponible,                      # Función que verifica si hay GPU disponible (CUDA)
)

# --- Verificación de factibilidad ---
# Comprueba si una solución es válida: que cada vehículo no supere su capacidad
# y que todas las aristas requeridas estén cubiertas.
from .factibilidad import (
    FeasibilityDetails,                      # Clase con detalles del chequeo por ruta
    FeasibilityResult,                       # Clase con el resultado global (factible o no)
    verificar_factibilidad,                  # Función principal de verificación
    verificar_factibilidad_desde_instancia,  # Versión con nombre de instancia
)

# --- Metaheurísticas de optimización ---
# Cada submódulo implementa un algoritmo de búsqueda distinto.
# Todos siguen la misma interfaz: función principal + versión "desde instancia".

# Búsqueda Tabú: explora vecinos y prohíbe movimientos recientes para escapar
# de mínimos locales.
from .busqueda_tabu import BusquedaTabuResult, busqueda_tabu, busqueda_tabu_desde_instancia

# Algoritmo de Abejas (Bee Algorithm): inspirado en el comportamiento de búsqueda
# de alimento de colmenas de abejas.
from .abejas import AbejasResult, busqueda_abejas, busqueda_abejas_desde_instancia

# Búsqueda del Cucú (Cuckoo Search): algoritmo inspirado en el parasitismo de
# nidos del cucú, con vuelos de Lévy para exploración.
from .cuckoo_search import CuckooSearchResult, cuckoo_search, cuckoo_search_desde_instancia

# --- Utilidades de grafo y rutas ---
# Funciones de bajo nivel sobre el grafo NetworkX: caminos mínimos, costos de
# aristas individuales, etc.
from .grafo_ruta import (
    costo_camino_minimo,   # Distancia del camino más corto entre dos nodos
    edge_cost,             # Costo (peso) de una arista específica
    nodo_grafo,            # Devuelve el objeto nodo del grafo con sus atributos
    path_edges_and_cost,   # Devuelve lista de aristas de un camino y su costo total
    shortest_path_nodes,   # Lista de nodos en el camino más corto entre dos puntos
)

# --- Gestión de instancias CARP ---
# Carga y almacena las instancias del problema (archivos .pkl con el grafo,
# demandas, capacidad del vehículo, etc.).
from .instances import InstanceStore, dictionary_instances, load_instance, load_instances

# --- Reporte de solución ---
# Genera un reporte legible de una solución: rutas por vehículo, demanda
# atendida, costo por ruta, etc.
from .reporte_solucion import ReporteSolucionResult, reporte_solucion, reporte_solucion_desde_instancia

# --- Recocido Simulado ---
# Metaheurística inspirada en el proceso de enfriamiento del metal: acepta
# soluciones peores con cierta probabilidad para escapar de mínimos locales.
from .recocido_simulado import (
    RecocidoSimuladoResult,
    recocido_simulado,
    recocido_simulado_desde_instancia,
)

# --- Formato de solución por etiquetas ---
# Convierte entre el formato de etiquetas legibles (ej. "TR1", "D") y el
# formato interno de listas de tareas que usan los algoritmos.
from .solucion_formato import (
    construir_mapa_tareas_por_etiqueta,  # Crea un diccionario etiqueta -> datos de arista
    etiquetas_tareas_requeridas,         # Devuelve el conjunto de etiquetas requeridas
    normalizar_rutas_etiquetas,          # Limpia y valida rutas expresadas con etiquetas
)

# --- Vecindarios y operadores de movimiento ---
# Define los movimientos que puede hacer un metaheurístico para generar
# soluciones vecinas (ej. intercambiar tareas entre rutas, mover una tarea).
from .vecindarios import MovimientoVecindario, OPERADORES_POPULARES, generar_vecino, generar_vecino_ids


# =============================================================================
# __all__: lista pública del paquete
#
# Esta lista controla qué nombres se exportan cuando alguien hace:
#   from metacarp import *
#
# Es buena práctica declararla explícitamente para que los IDEs y herramientas
# de análisis sepan exactamente qué expone este paquete.
# =============================================================================
__all__ = [
    # Gestión de instancias
    "InstanceStore",
    "dictionary_instances",
    "load_instance",
    "load_instances",
    # Grafos e imágenes
    "cargar_grafo",
    "cargar_imagen_estatica",
    "cargar_objeto_gexf",
    "ruta_imagen_estatica",
    "ruta_gexf",
    # Matrices de distancias
    "cargar_matriz_dijkstra",
    "nombres_matrices_disponibles",
    "ruta_matriz_dijkstra",
    # Soluciones iniciales
    "cargar_solucion_inicial",
    "nombres_soluciones_iniciales_disponibles",
    "ruta_solucion_inicial",
    # Codificación para búsqueda por índices
    "DEPOT_ID",
    "SearchEncoding",
    "build_search_encoding",
    "encode_solution",
    "decode_solution",
    "decode_task_ids",
    # Factibilidad
    "FeasibilityDetails",
    "FeasibilityResult",
    "verificar_factibilidad",
    "verificar_factibilidad_desde_instancia",
    # Metaheurísticas
    "BusquedaTabuResult",
    "busqueda_tabu",
    "busqueda_tabu_desde_instancia",
    "AbejasResult",
    "busqueda_abejas",
    "busqueda_abejas_desde_instancia",
    "CuckooSearchResult",
    "cuckoo_search",
    "cuckoo_search_desde_instancia",
    # Utilidades de grafo
    "costo_camino_minimo",
    "edge_cost",
    "nodo_grafo",
    "path_edges_and_cost",
    "shortest_path_nodes",
    # Costo de solución
    "CostoSolucionResult",
    "costo_solucion",
    "costo_solucion_desde_instancia",
    # Evaluador rápido
    "ContextoEvaluacion",
    "construir_contexto",
    "construir_contexto_desde_instancia",
    "costo_lote_ids",
    "costo_rapido",
    "costo_rapido_ids",
    "gpu_disponible",
    # Reporte de solución
    "ReporteSolucionResult",
    "reporte_solucion",
    "reporte_solucion_desde_instancia",
    # Recocido Simulado
    "RecocidoSimuladoResult",
    "recocido_simulado",
    "recocido_simulado_desde_instancia",
    # Formato de etiquetas
    "construir_mapa_tareas_por_etiqueta",
    "etiquetas_tareas_requeridas",
    "normalizar_rutas_etiquetas",
    # Vecindarios
    "MovimientoVecindario",
    "OPERADORES_POPULARES",
    "generar_vecino",
    "generar_vecino_ids",
]
