"""
Script de prueba/documentacion para todos los modulos publicos de metacarp.

OBJETIVO:
- Mostrar llamadas reales (input/output) de la API.
- Servir como guia para debug visual rapido usando la instancia pequena: gdb19.
- Explicar cuando tiene sentido pedir GPU en vecindarios.

EJECUCION:
    python metacarp/scripts/testing.py
"""

# ============================================================
# testing.py
# ------------------------------------------------------------
# Script ejecutable que sirve como GUÍA INTERACTIVA de toda la
# API del paquete metacarp. No es un test automático (no usa
# pytest), sino una demostración documentada que imprime en
# consola los inputs y outputs de cada llamada.
#
# Estructura del script (bloques A a E):
#   A)  Carga de la instancia y recursos base
#   A.1) Catálogos disponibles
#   B)  Formato y normalización de la solución
#   B.1) Factibilidad, costo y reporte
#   C)  Utilidades de grafo y caminos
#   D)  Encoding indexado y operadores de vecindario
#   E)  Metaheurísticas (SA, Tabú, Abejas, Cuckoo)
# ============================================================

from __future__ import annotations

import random
from collections.abc import Callable   # Tipo para funciones que se pasan como argumento
from pprint import pprint              # Impresión "bonita" de estructuras de datos anidadas
from typing import Any

# Importa TODOS los símbolos públicos del paquete metacarp de una sola vez.
# Esto es posible porque metacarp/__init__.py re-exporta todo.
from metacarp import (
    busqueda_abejas_desde_instancia,        # Metaheurística: Colonia de Abejas Artificiales
    busqueda_tabu_desde_instancia,          # Metaheurística: Búsqueda Tabú
    cuckoo_search_desde_instancia,          # Metaheurística: Cuckoo Search
    OPERADORES_POPULARES,                   # Tupla con los 7 operadores de vecindario disponibles
    build_search_encoding,                  # Construye el encoding ID<->etiqueta de una instancia
    cargar_grafo,                           # Carga genérica de grafo por tipo (gexf, etc.)
    cargar_imagen_estatica,                 # Carga imagen PNG/JPG del grafo para visualización
    cargar_matriz_dijkstra,                 # Carga la matriz de distancias precalculada (Dijkstra)
    cargar_objeto_gexf,                     # Carga el grafo en formato GEXF como objeto NetworkX
    cargar_solucion_inicial,                # Carga la solución inicial precalculada (pickle)
    costo_camino_minimo,                    # Devuelve (costo, camino) entre dos nodos
    costo_solucion,                         # Calcula el costo total de una solución con detalle por ruta
    costo_solucion_desde_instancia,         # Versión que carga data+grafo internamente
    decode_solution,                        # Convierte solución de IDs enteros a etiquetas
    decode_task_ids,                        # Convierte lista de IDs a lista de etiquetas
    dictionary_instances,                   # Diccionario lazy con todas las instancias disponibles
    edge_cost,                              # Costo de un arco individual en el grafo
    encode_solution,                        # Convierte solución de etiquetas a IDs enteros
    etiquetas_tareas_requeridas,            # Conjunto de etiquetas de tareas obligatorias (TR...)
    generar_vecino,                         # Genera un vecino con backend labels o ids
    generar_vecino_ids,                     # Genera un vecino operando directamente sobre IDs
    load_instance,                          # Carga una instancia por nombre (singular)
    load_instances,                         # Carga una instancia por nombre (forma canónica)
    nodo_grafo,                             # Convierte ID entero al formato de nodo del grafo
    nombres_matrices_disponibles,           # Lista de nombres de matrices Dijkstra empaquetadas
    nombres_soluciones_iniciales_disponibles, # Lista de nombres de soluciones iniciales disponibles
    normalizar_rutas_etiquetas,             # Elimina "D" y valida etiquetas contra la instancia
    path_edges_and_cost,                    # Desglose de un camino en arcos con sus costos
    reporte_solucion,                       # Genera el reporte textual de una solución
    reporte_solucion_desde_instancia,       # Versión que carga data+grafo internamente
    recocido_simulado_desde_instancia,      # Metaheurística: Recocido Simulado
    ruta_gexf,                              # Devuelve la ruta al archivo GEXF de la instancia
    ruta_imagen_estatica,                   # Devuelve la ruta al PNG del grafo
    ruta_matriz_dijkstra,                   # Devuelve la ruta al archivo de matriz Dijkstra
    ruta_solucion_inicial,                  # Devuelve la ruta al pickle de solución inicial
    shortest_path_nodes,                    # Calcula el camino más corto entre dos nodos del grafo
    verificar_factibilidad,                 # Verifica C1..C5 con solución + data + matriz
    verificar_factibilidad_desde_instancia, # Versión que carga data+matriz internamente
)


# ------------------------------------------------------------
# CONSTANTES GLOBALES DEL SCRIPT
# ------------------------------------------------------------
INSTANCIA = "gdb19"          # Instancia pequeña (benchmark estándar CARP) para pruebas rápidas
SEED = 42                    # Semilla fija para reproducibilidad de resultados aleatorios
GUARDAR_CSV_DEMO = False     # Si True, guarda archivos CSV de historial en disco (para testing)


# ============================================================
# FUNCIONES DE PRESENTACIÓN
# ============================================================

def titulo(txt: str) -> None:
    """Imprime un separador visual de sección en la consola."""
    print("\n" + "=" * 90)
    print(txt)
    print("=" * 90)


def _resumen_salida(valor: Any, *, max_items: int = 5) -> str:
    """
    Resumen corto de retorno para imprimir en terminal.
    Limita la salida a max_items elementos para no saturar la consola.
    """
    if isinstance(valor, dict):
        # Muestra solo las primeras max_items claves del diccionario
        ks = list(valor.keys())[:max_items]
        return f"dict(len={len(valor)}, keys_sample={ks})"
    if isinstance(valor, list):
        # Muestra solo los primeros max_items elementos de la lista
        sample = valor[:max_items]
        return f"list(len={len(valor)}, sample={sample})"
    if isinstance(valor, tuple):
        sample = valor[:max_items]
        return f"tuple(len={len(valor)}, sample={sample})"
    if isinstance(valor, set):
        sample = list(valor)[:max_items]  # Los sets no tienen índice, se convierte a lista primero
        return f"set(len={len(valor)}, sample={sample})"
    # Para cualquier otro tipo, usa la representación por defecto de Python
    return repr(valor)


def mostrar_llamada(
    *,
    comentario: str,    # Descripción en lenguaje natural de qué hace la llamada
    modulo: str,        # Módulo de metacarp donde vive la función
    codigo: str,        # Código Python que se ejecutaría (string para mostrar)
    valor: Any | None = None,  # Valor devuelto por la llamada (para imprimirlo)
) -> None:
    """Imprime comentario + codigo + salida para documentacion explicita."""
    print("\n" + "-" * 90)
    print(f"Comentario : {comentario}")
    print(f"Modulo     : {modulo}")
    print("Codigo     :")
    print(f"  {codigo}")
    if valor is not None:
        print(f"Devuelve   : {_resumen_salida(valor)}")
    print("-" * 90)


def ejecutar_llamada(
    *,
    comentario: str,
    modulo: str,
    codigo: str,
    fn: Callable[[], Any],  # Función sin argumentos que encapsula la llamada real (lambda)
) -> Any:
    """
    Ejecuta y documenta una llamada individual.
    Recibe fn como una función "lambda" (función anónima sin argumentos) que
    encapsula la llamada real. Esto permite que el código solo se ejecute dentro
    de esta función (y no en el momento de definirlo).
    """
    valor = fn()   # Ejecuta la llamada real aquí
    mostrar_llamada(comentario=comentario, modulo=modulo, codigo=codigo, valor=valor)
    return valor   # Devuelve el resultado para que el llamador lo use


# ============================================================
# BLOQUE A: Carga de la instancia y recursos base
# ============================================================
def demo_cargas_basicas() -> tuple[dict, object, object, list[list[str]]]:
    """
    Carga la instancia gdb19 y todos sus recursos asociados:
    datos, matriz Dijkstra, grafo GEXF y solución inicial.

    Devuelve una tupla con los cuatro objetos principales para
    que los bloques posteriores los reutilicen sin recargar.
    """
    titulo("BLOQUE A) OBJETO INSTANCIA Y RECURSOS BASE")

    print(f"Instancia de prueba: {INSTANCIA}")

    # Rutas a los archivos en disco (solo las rutas, no los contenidos)
    ejecutar_llamada(
        comentario="Ruta al pickle de solucion inicial.",
        modulo="metacarp.cargar_soluciones_iniciales",
        codigo=f"ruta_solucion_inicial('{INSTANCIA}')",
        fn=lambda: ruta_solucion_inicial(INSTANCIA),
    )
    ejecutar_llamada(
        comentario="Ruta al archivo de matriz de distancias (Dijkstra).",
        modulo="metacarp.cargar_matrices",
        codigo=f"ruta_matriz_dijkstra('{INSTANCIA}')",
        fn=lambda: ruta_matriz_dijkstra(INSTANCIA),
    )
    ejecutar_llamada(
        comentario="Ruta al grafo GEXF de la instancia.",
        modulo="metacarp.cargar_grafos",
        codigo=f"ruta_gexf('{INSTANCIA}')",
        fn=lambda: ruta_gexf(INSTANCIA),
    )
    ejecutar_llamada(
        comentario="Ruta de imagen estatica para debug visual (si existe).",
        modulo="metacarp.cargar_grafos",
        codigo=f"ruta_imagen_estatica('{INSTANCIA}')",
        fn=lambda: ruta_imagen_estatica(INSTANCIA),
    )

    # Carga real de los objetos (no solo rutas)
    data = ejecutar_llamada(
        comentario="Carga completa de la instancia (dict).",
        modulo="metacarp.instances",
        codigo=f"load_instances('{INSTANCIA}')",
        fn=lambda: load_instances(INSTANCIA),
    )
    _data_single = ejecutar_llamada(
        comentario="Carga por nombre singular (equivalente practico).",
        modulo="metacarp.instances",
        codigo=f"load_instance('{INSTANCIA}')",
        fn=lambda: load_instance(INSTANCIA),
    )
    matriz = ejecutar_llamada(
        comentario="Carga de matriz Dijkstra para factibilidad/conectividad.",
        modulo="metacarp.cargar_matrices",
        codigo=f"cargar_matriz_dijkstra('{INSTANCIA}')",
        fn=lambda: cargar_matriz_dijkstra(INSTANCIA),
    )
    grafo = ejecutar_llamada(
        comentario="Carga objeto grafo NetworkX (Graph/MultiGraph).",
        modulo="metacarp.cargar_grafos",
        codigo=f"cargar_objeto_gexf('{INSTANCIA}')",
        fn=lambda: cargar_objeto_gexf(INSTANCIA),
    )
    _grafo_simple = ejecutar_llamada(
        comentario="API generica para cargar grafo por tipo='gexf'.",
        modulo="metacarp.cargar_grafos",
        codigo=f"cargar_grafo('{INSTANCIA}', 'gexf')",
        fn=lambda: cargar_grafo(INSTANCIA, "gexf"),
    )
    solucion = ejecutar_llamada(
        comentario="Carga de solucion inicial por etiquetas TR y marcador D.",
        modulo="metacarp.cargar_soluciones_iniciales",
        codigo=f"cargar_solucion_inicial('{INSTANCIA}')",
        fn=lambda: cargar_solucion_inicial(INSTANCIA),
    )

    print("\nOUTPUT esperado (tipos):")
    print("- data: dict con llaves DEPOSITO/CAPACIDAD/LISTA_ARISTAS_REQ...")
    print(f"- matriz: {type(matriz).__name__} | grafo: {type(grafo).__name__} | solucion: {type(solucion).__name__}")
    print(f"- ejemplo primera ruta: {solucion[0] if solucion else []}")

    # Intento opcional de carga de imagen para debug visual (puede no existir)
    try:
        img = cargar_imagen_estatica(INSTANCIA, show=False)
        print(f"- imagen cargada correctamente: {type(img).__name__}")
    except Exception as exc:  # noqa: BLE001 - demo de estado opcional
        print(f"- imagen no disponible/omitida: {exc}")

    return data, matriz, grafo, solucion


# ============================================================
# BLOQUE A.1: Catálogos de recursos disponibles
# ============================================================
def demo_catalogos() -> None:
    """
    Muestra los catálogos de instancias, matrices y soluciones
    iniciales disponibles en el paquete.
    """
    titulo("BLOQUE A.1) CATALOGOS DISPONIBLES (OBJETO INSTANCIA)")
    ejecutar_llamada(
        comentario="Catalogo de instancias disponibles en memoria lazy.",
        modulo="metacarp.instances",
        codigo="list(dictionary_instances.keys())[:10]",
        fn=lambda: list(dictionary_instances.keys())[:10],
    )
    ejecutar_llamada(
        comentario="Nombres de matrices dijkstra empaquetadas.",
        modulo="metacarp.cargar_matrices",
        codigo="nombres_matrices_disponibles()[:10]",
        fn=lambda: nombres_matrices_disponibles()[:10],
    )
    ejecutar_llamada(
        comentario="Nombres de soluciones iniciales empaquetadas.",
        modulo="metacarp.cargar_soluciones_iniciales",
        codigo="nombres_soluciones_iniciales_disponibles()[:10]",
        fn=lambda: nombres_soluciones_iniciales_disponibles()[:10],
    )


# ============================================================
# BLOQUE B: Formato y normalización de la solución
# ============================================================
def demo_formato_solucion(data: dict, solucion: list[list[str]]) -> list[list[str]]:
    """
    Demuestra las utilidades de formato: construcción del mapa de tareas,
    normalización de rutas y resolución de etiquetas canónicas.

    Devuelve las rutas normalizadas (sin "D") para que los bloques
    siguientes las usen directamente.
    """
    titulo("BLOQUE B) OBJETO SOLUCION - FORMATO Y NORMALIZACION")
    from metacarp.solucion_formato import resolver_etiqueta_canonica

    mapa = ejecutar_llamada(
        comentario="Construye mapa etiqueta -> dict de tarea.",
        modulo="metacarp.solucion_formato",
        codigo="construir_mapa_tareas_por_etiqueta(data)",
        fn=lambda: construir_mapa_tareas_por_etiqueta(data),
    )

    # normalizar_rutas_etiquetas devuelve una TUPLA (rutas, error):
    #   - Si todo está bien: (lista_de_rutas_sin_D, None)
    #   - Si hay error:      ([], "mensaje de error")
    rutas_norm, err = ejecutar_llamada(
        comentario="Normaliza rutas: elimina D y valida etiquetas conocidas.",
        modulo="metacarp.solucion_formato",
        codigo="normalizar_rutas_etiquetas(solucion, data, mapa)",
        fn=lambda: normalizar_rutas_etiquetas(solucion, data, mapa),
    )
    if err:
        raise ValueError(f"Error de formato inesperado: {err}")

    requeridas = ejecutar_llamada(
        comentario="Conjunto de tareas requeridas (TR) de la instancia.",
        modulo="metacarp.solucion_formato",
        codigo="etiquetas_tareas_requeridas(data)",
        fn=lambda: etiquetas_tareas_requeridas(data),
    )
    print(f"#tareas en mapa (REQ+NOREQ): {len(mapa)}")
    print(f"#tareas requeridas: {len(requeridas)}")
    print(f"Rutas normalizadas (sin D), primera ruta: {rutas_norm[0] if rutas_norm else []}")

    # Ejemplo de canonicalización de etiqueta (insensible a mayúsculas)
    if rutas_norm and rutas_norm[0]:
        et = rutas_norm[0][0]
        ejecutar_llamada(
            comentario="Canonicaliza una etiqueta (insensible a mayusculas).",
            modulo="metacarp.solucion_formato",
            codigo=f"resolver_etiqueta_canonica('{et.lower()}', mapa)",
            fn=lambda: resolver_etiqueta_canonica(et.lower(), mapa),
        )
    return rutas_norm


# ============================================================
# BLOQUE B.1: Factibilidad, costo y reporte de la solución
# ============================================================
def demo_factibilidad_y_costo(data: dict, matriz: object, grafo: object, solucion: list[list[str]]) -> None:
    """
    Verifica factibilidad (restricciones C1..C5), calcula el costo total
    y genera el reporte textual. Al final comprueba que el costo del
    módulo de costo coincide exactamente con el del módulo de reporte.
    """
    titulo("BLOQUE B.1) OBJETO SOLUCION - FACTIBILIDAD, COSTO Y REPORTE")

    # ---- Factibilidad ----
    feas = ejecutar_llamada(
        comentario="Valida C1..C5 con matriz de distancias.",
        modulo="metacarp.factibilidad",
        codigo="verificar_factibilidad(solucion, data, matriz)",
        fn=lambda: verificar_factibilidad(solucion, data, matriz),
    )
    print(f"Factible (verificar_factibilidad): {feas.ok}")
    if not feas.ok:
        print("Resumen de violaciones:")
        print(feas.details.resumen())

    feas2 = ejecutar_llamada(
        comentario="Helper que carga data+matriz y valida factibilidad.",
        modulo="metacarp.factibilidad",
        codigo=f"verificar_factibilidad_desde_instancia('{INSTANCIA}', solucion)",
        fn=lambda: verificar_factibilidad_desde_instancia(INSTANCIA, solucion),
    )
    print(f"Factible (desde_instancia): {feas2.ok}")

    # ---- Costo ----
    cost = ejecutar_llamada(
        comentario="Calcula costo total y por ruta.",
        modulo="metacarp.costo_solucion",
        codigo="costo_solucion(solucion, data, grafo, detalle=False)",
        fn=lambda: costo_solucion(solucion, data, grafo, detalle=False),
    )
    print(f"Costo total: {cost.costo_total}")
    print(f"Costos por ruta: {cost.costos_por_ruta}")
    print(f"Demandas por ruta: {cost.demandas_por_ruta}")

    cost2 = ejecutar_llamada(
        comentario="Helper de costo que carga data+grafo internamente.",
        modulo="metacarp.costo_solucion",
        codigo=f"costo_solucion_desde_instancia('{INSTANCIA}', solucion, detalle=False)",
        fn=lambda: costo_solucion_desde_instancia(INSTANCIA, solucion, detalle=False),
    )
    print(f"Costo total (desde_instancia): {cost2.costo_total}")

    # ---- Reporte textual ----
    rep = ejecutar_llamada(
        comentario="Genera reporte interpretable por vehiculo.",
        modulo="metacarp.reporte_solucion",
        codigo="reporte_solucion(solucion, data, grafo, nombre_instancia=INSTANCIA)",
        fn=lambda: reporte_solucion(solucion, data, grafo, nombre_instancia=INSTANCIA),
    )
    print(f"Costo total segun reporte: {rep.costo_total}")
    print("Primeras 8 lineas de reporte:")
    for line in rep.texto.splitlines()[:8]:
        print(line)

    rep2 = ejecutar_llamada(
        comentario="Helper de reporte que carga data+grafo.",
        modulo="metacarp.reporte_solucion",
        codigo=f"reporte_solucion_desde_instancia('{INSTANCIA}', solucion)",
        fn=lambda: reporte_solucion_desde_instancia(INSTANCIA, solucion),
    )
    print(f"Costo total (reporte_desde_instancia): {rep2.costo_total}")

    # Verificación de consistencia: costo_solucion y reporte_solucion deben coincidir.
    # abs() toma el valor absoluto; 1e-9 es la tolerancia numérica (casi cero).
    assert abs(cost.costo_total - rep.costo_total) < 1e-9, "Costo y reporte deben coincidir."


# ============================================================
# BLOQUE C: Utilidades de grafo y caminos mínimos
# ============================================================
def demo_grafo_utils(data: dict, grafo: object, rutas_norm: list[list[str]]) -> None:
    """
    Demuestra las funciones de navegación del grafo:
    camino mínimo, desglose en arcos y costo de un arco individual.
    """
    titulo("BLOQUE C) OBJETO GRAFO - UTILIDADES DE CAMINOS Y COSTOS")
    deposito = int(data["DEPOSITO"])   # Nodo depósito de la instancia
    mapa = construir_mapa_tareas_por_etiqueta(data)

    # Toma la primera tarea de la primera ruta como ejemplo de nodo a visitar
    et = rutas_norm[0][0]
    tarea = mapa[et]
    u, v = int(tarea["nodos"][0]), int(tarea["nodos"][1])

    ejecutar_llamada(
        comentario="Convierte id de nodo al formato del grafo GEXF (str).",
        modulo="metacarp.grafo_ruta",
        codigo=f"nodo_grafo({deposito})",
        fn=lambda: nodo_grafo(deposito),
    )
    path = ejecutar_llamada(
        comentario="Camino minimo ponderado por cost.",
        modulo="metacarp.grafo_ruta",
        codigo=f"shortest_path_nodes(grafo, {deposito}, {u})",
        fn=lambda: shortest_path_nodes(grafo, deposito, u),
    )
    edges, cpath = ejecutar_llamada(
        comentario="Desglose por arcos y costo acumulado del camino.",
        modulo="metacarp.grafo_ruta",
        codigo="path_edges_and_cost(grafo, path)",
        fn=lambda: path_edges_and_cost(grafo, path),
    )
    print(f"Camino minimo deposito->{u}: {path}")
    print(f"Costo camino (sumando arcos): {cpath}")
    if len(path) >= 2:
        a, b = path[0], path[1]
        ejecutar_llamada(
            comentario="Costo de un arco individual en el grafo.",
            modulo="metacarp.grafo_ruta",
            codigo=f"edge_cost(grafo, '{a}', '{b}')",
            fn=lambda: edge_cost(grafo, a, b),
        )
    ejecutar_llamada(
        comentario="Costo y path de origen a destino (helper completo).",
        modulo="metacarp.grafo_ruta",
        codigo=f"costo_camino_minimo(grafo, {deposito}, {u})",
        fn=lambda: costo_camino_minimo(grafo, deposito, u),
    )
    print(f"Ejemplo tarea servicio: {et} con nodos ({u},{v})")
    print(f"Primeros arcos del camino (si hay): {edges[:3]}")


# ============================================================
# BLOQUE D: Encoding indexado y operadores de vecindario
# ============================================================
def demo_encoding_y_vecindarios(data: dict, solucion: list[list[str]]) -> None:
    """
    Demuestra el flujo completo de búsqueda indexada:
      1. Construir el SearchEncoding (mapa label<->id).
      2. Encode/decode (roundtrip): verificar que la conversión es reversible.
      3. Generar vecinos en modo labels (strings) y en modo ids (enteros).
      4. Mostrar qué ocurre al pedir GPU (hoy: fallback controlado a CPU).
    """
    titulo("BLOQUE D) OBJETO VECINDARIO Y BUSQUEDA INDEXADA")

    # ---- Encoding ----
    encoding = ejecutar_llamada(
        comentario="Compila encoding estable label<->id y arrays densos.",
        modulo="metacarp.busqueda_indices",
        codigo="build_search_encoding(data)",
        fn=lambda: build_search_encoding(data),
    )
    print("SearchEncoding construido:")
    print(f"- #tareas: {len(encoding)}")
    print(f"- depot_marker: {encoding.depot_marker}")
    print(f"- primeros labels: {encoding.id_to_label[:5]}")

    # ---- Encode/decode roundtrip ----
    # encode_solution: ["D","TR1","TR5","D"] -> [-1, 0, 4, -1] (ejemplo)
    sol_ids = ejecutar_llamada(
        comentario="Convierte solucion de etiquetas a ids enteros.",
        modulo="metacarp.busqueda_indices",
        codigo="encode_solution(solucion, encoding)",
        fn=lambda: encode_solution(solucion, encoding),
    )
    # decode_solution: [-1, 0, 4, -1] -> ["D","TR1","TR5","D"] (operación inversa)
    sol_labels_roundtrip = ejecutar_llamada(
        comentario="Decodifica ids a etiquetas nuevamente.",
        modulo="metacarp.busqueda_indices",
        codigo="decode_solution(sol_ids, encoding, con_deposito=True)",
        fn=lambda: decode_solution(sol_ids, encoding, con_deposito=True),
    )
    # Si el encode/decode no es reversible, hay un bug en el encoding
    assert sol_labels_roundtrip == solucion, "Roundtrip labels->ids->labels debe conservar solucion."
    print("Roundtrip labels->ids->labels: OK")
    print(f"Primera ruta en ids: {sol_ids[0] if sol_ids else []}")

    ejecutar_llamada(
        comentario="Decodifica solo un subconjunto de IDs a etiquetas TR.",
        modulo="metacarp.busqueda_indices",
        codigo="decode_task_ids(sol_ids[0][:3], encoding)",
        fn=lambda: decode_task_ids((sol_ids[0][:3] if sol_ids else []), encoding),
    )

    # Dos generadores de números aleatorios con la misma semilla para comparar
    # los resultados de los dos backends (labels vs ids) de forma justa.
    rng_labels = random.Random(SEED)
    rng_ids = random.Random(SEED)

    # ---- Vecino en backend labels (default) ----
    # Internamente opera sobre listas de strings, sin encoding.
    vecino_labels, mov_labels = ejecutar_llamada(
        comentario="Genera un vecino operando sobre etiquetas (CPU).",
        modulo="metacarp.vecindarios",
        codigo=(
            "generar_vecino(solucion, rng=Random(SEED), "
            "operadores=OPERADORES_POPULARES, backend='labels', usar_gpu=False)"
        ),
        fn=lambda: generar_vecino(
            solucion,
            rng=rng_labels,
            operadores=OPERADORES_POPULARES,
            backend="labels",
            usar_gpu=False,
        ),
    )
    print("\nVecino (backend labels, CPU):")
    print(f"- movimiento: {mov_labels}")
    print(f"- primera ruta vecino: {vecino_labels[0] if vecino_labels else []}")

    # ---- Vecino en backend ids (ruta indexada, más rápida) ----
    # Internamente convierte a enteros, opera y decodifica al devolver.
    vecino_ids_labels, mov_ids = ejecutar_llamada(
        comentario="Genera un vecino usando backend indexado (ids, CPU).",
        modulo="metacarp.vecindarios",
        codigo="generar_vecino(solucion, rng=Random(SEED), backend='ids', encoding=encoding, usar_gpu=False)",
        fn=lambda: generar_vecino(
            solucion,
            rng=rng_ids,
            backend="ids",
            encoding=encoding,
            usar_gpu=False,
        ),
    )
    print("\nVecino (backend ids, CPU):")
    print(f"- movimiento: {mov_ids}")
    print(f"- tareas movidas (ids): {mov_ids.id_movidos}")
    print(f"- tareas movidas (labels): {mov_ids.labels_movidos}")
    print(f"- primera ruta vecino decodificado: {vecino_ids_labels[0] if vecino_ids_labels else []}")

    # ---- Llamada directa sobre ids (sin encode/decode implícito) ----
    # Útil cuando el algoritmo ya trabaja en representación entera
    # y quiere evitar el overhead de conversión en cada iteración.
    vecino_ids_directo, mov_ids_directo = ejecutar_llamada(
        comentario="Genera vecino directamente sobre solucion en ids.",
        modulo="metacarp.vecindarios",
        codigo="generar_vecino_ids(sol_ids, rng=Random(SEED+1), usar_gpu=False, encoding=encoding)",
        fn=lambda: generar_vecino_ids(
            sol_ids,
            rng=random.Random(SEED + 1),  # SEED+1 para obtener un vecino diferente
            usar_gpu=False,
            encoding=encoding,
        ),
    )
    print("\nVecino directo sobre IDs:")
    print(f"- movimiento: {mov_ids_directo}")
    print(f"- primera ruta ids vecino: {vecino_ids_directo[0] if vecino_ids_directo else []}")

    # ---- GPU: cuando sirve y cuándo no ----
    # Hoy usar_gpu=True registra que se solicitó GPU pero cae a CPU
    # (backend_real='cpu') porque no hay kernel implementado.
    # La API ya está preparada para cuando exista un kernel real.
    vecino_gpu, mov_gpu = ejecutar_llamada(
        comentario="Pide GPU para vecindario indexado (hoy: fallback controlado a CPU).",
        modulo="metacarp.vecindarios",
        codigo="generar_vecino(solucion, rng=Random(SEED), backend='ids', encoding=encoding, usar_gpu=True)",
        fn=lambda: generar_vecino(
            solucion,
            rng=random.Random(SEED),
            backend="ids",
            encoding=encoding,
            usar_gpu=True,
        ),
    )
    print("\nGPU flag en vecindarios:")
    print("- INPUT: usar_gpu=True + backend='ids'")
    print("- OUTPUT esperado hoy: backend_solicitado='gpu' y backend_real='cpu' (fallback)")
    print(f"- movimiento: {mov_gpu}")
    print(f"- primera ruta vecino gpu/fallback: {vecino_gpu[0] if vecino_gpu else []}")


# ============================================================
# BLOQUE E: Metaheurísticas
# ============================================================
def demo_metaheuristicas() -> None:
    """
    Ejecuta las cuatro metaheurísticas disponibles con parámetros
    reducidos (pocas iteraciones) para que la demo termine rápido.
    Muestra el costo final, la mejora porcentual y el tiempo.
    """
    titulo("BLOQUE E) METAHEURISTICAS - SA, TABU, ABEJAS, CUCKOO")
    print(f"Instancia: {INSTANCIA} | seed: {SEED} | guardar_csv_demo={GUARDAR_CSV_DEMO}")

    # ---- Recocido Simulado (Simulated Annealing) ----
    # Inspirado en el proceso de enfriamiento lento de metales.
    # Acepta soluciones peores con una probabilidad que decrece con la temperatura.
    # alpha: factor de enfriamiento geométrico (nueva_temp = alpha * temp_actual).
    sa = ejecutar_llamada(
        comentario="Ejecuta Recocido Simulado con parámetros compactos para demo.",
        modulo="metacarp.recocido_simulado",
        codigo=(
            "recocido_simulado_desde_instancia("
            f"'{INSTANCIA}', temperatura_inicial=150.0, temperatura_minima=1e-3, "
            "alpha=0.93, iteraciones_por_temperatura=40, max_enfriamientos=25, "
            f"semilla={SEED}, guardar_csv=GUARDAR_CSV_DEMO)"
        ),
        fn=lambda: recocido_simulado_desde_instancia(
            INSTANCIA,
            temperatura_inicial=150.0,   # Temperatura de inicio (alta = acepta muchos movimientos)
            temperatura_minima=1e-3,      # Temperatura final (baja = casi no acepta movimientos malos)
            alpha=0.93,                   # Factor de enfriamiento: cada ciclo multiplica la temp por 0.93
            iteraciones_por_temperatura=40,  # Cuántos vecinos se evalúan antes de enfriar
            max_enfriamientos=25,         # Cuántos ciclos de enfriamiento se ejecutan
            semilla=SEED,
            guardar_csv=GUARDAR_CSV_DEMO,
        ),
    )
    print(
        f"- SA: mejor_costo={sa.mejor_costo} "
        f"| mejora_inicial_vs_final={sa.mejora_porcentaje_inicial_vs_final:.4f}% "
        f"| tiempo={sa.tiempo_segundos:.4f}s | csv={sa.archivo_csv}"
    )

    # ---- Búsqueda Tabú ----
    # Explora el vecindario pero prohíbe volver a movimientos recientes
    # (lista tabú) para escapar de óptimos locales.
    # tenure_tabu: cuántas iteraciones permanece un movimiento en la lista tabú.
    tabu = ejecutar_llamada(
        comentario="Ejecuta Búsqueda Tabú clásica con aspiración.",
        modulo="metacarp.busqueda_tabu",
        codigo=(
            "busqueda_tabu_desde_instancia("
            f"'{INSTANCIA}', iteraciones=120, tam_vecindario=16, tenure_tabu=15, "
            f"semilla={SEED}, guardar_csv=GUARDAR_CSV_DEMO)"
        ),
        fn=lambda: busqueda_tabu_desde_instancia(
            INSTANCIA,
            iteraciones=120,        # Número total de iteraciones
            tam_vecindario=16,      # Cuántos vecinos se generan y evalúan en cada iteración
            tenure_tabu=15,         # Duración de la prohibición tabú
            semilla=SEED,
            guardar_csv=GUARDAR_CSV_DEMO,
        ),
    )
    print(
        f"- Tabu: mejor_costo={tabu.mejor_costo} "
        f"| mejora_inicial_vs_final={tabu.mejora_porcentaje_inicial_vs_final:.4f}% "
        f"| tiempo={tabu.tiempo_segundos:.4f}s | csv={tabu.archivo_csv}"
    )

    # ---- Colonia de Abejas Artificiales (ABC) ----
    # Inspirada en el comportamiento de exploración de las abejas.
    # Las "fuentes de alimento" son soluciones; las abejas las mejoran o las abandonan.
    abe = ejecutar_llamada(
        comentario="Ejecuta metaheurística de Abejas (empleadas/observadoras/scouts).",
        modulo="metacarp.abejas",
        codigo=(
            "busqueda_abejas_desde_instancia("
            f"'{INSTANCIA}', iteraciones=120, num_fuentes=12, limite_abandono=20, "
            f"semilla={SEED}, guardar_csv=GUARDAR_CSV_DEMO)"
        ),
        fn=lambda: busqueda_abejas_desde_instancia(
            INSTANCIA,
            iteraciones=120,
            num_fuentes=12,          # Número de soluciones en la población
            limite_abandono=20,      # Cuántos intentos fallidos antes de abandonar una fuente
            semilla=SEED,
            guardar_csv=GUARDAR_CSV_DEMO,
        ),
    )
    print(
        f"- Abejas: mejor_costo={abe.mejor_costo} "
        f"| mejora_inicial_vs_final={abe.mejora_porcentaje_inicial_vs_final:.4f}% "
        f"| tiempo={abe.tiempo_segundos:.4f}s | csv={abe.archivo_csv}"
    )

    # ---- Cuckoo Search ----
    # Inspirado en el parasitismo de nidificación del cucú.
    # Genera nuevas soluciones con vuelos de Lévy (saltos largos ocasionales)
    # y reemplaza nidos con probabilidad pa_abandono.
    cko = ejecutar_llamada(
        comentario="Ejecuta Cuckoo Search con vuelo tipo Levy discreto.",
        modulo="metacarp.cuckoo_search",
        codigo=(
            "cuckoo_search_desde_instancia("
            f"'{INSTANCIA}', iteraciones=120, num_nidos=14, pa_abandono=0.25, "
            "pasos_levy_base=3, beta_levy=1.5, "
            f"semilla={SEED}, guardar_csv=GUARDAR_CSV_DEMO)"
        ),
        fn=lambda: cuckoo_search_desde_instancia(
            INSTANCIA,
            iteraciones=120,
            num_nidos=14,            # Número de soluciones (nidos) en la población
            pa_abandono=0.25,        # Probabilidad de abandonar un nido por iteración
            pasos_levy_base=3,       # Número base de movimientos en el vuelo de Lévy
            beta_levy=1.5,           # Parámetro de la distribución de Lévy (controla longitud de saltos)
            semilla=SEED,
            guardar_csv=GUARDAR_CSV_DEMO,
        ),
    )
    print(
        f"- Cuckoo: mejor_costo={cko.mejor_costo} "
        f"| mejora_inicial_vs_final={cko.mejora_porcentaje_inicial_vs_final:.4f}% "
        f"| tiempo={cko.tiempo_segundos:.4f}s | csv={cko.archivo_csv}"
    )


# ============================================================
# FUNCIÓN LOCAL AUXILIAR: construir_mapa_tareas_por_etiqueta
# ============================================================
# Este wrapper local evita tener que importar desde
# metacarp.solucion_formato en cada función del script.
# Redirige a la función real del paquete.
def construir_mapa_tareas_por_etiqueta(data: dict) -> dict[str, dict]:
    """Wrapper local para evitar import extra en cada demo."""
    from metacarp import construir_mapa_tareas_por_etiqueta as _f

    return _f(data)


# ============================================================
# FUNCIÓN PRINCIPAL: main
# ============================================================
# Punto de entrada del script. Ejecuta todos los bloques en
# orden y al final imprime un resumen del input usado.
def main() -> None:
    titulo("GUIA EJECUTABLE AGRUPADA POR TIPO DE OBJETO")
    print("Orden de lectura recomendado:")
    print("A) Instancia  -> B) Solucion  -> C) Grafo  -> D) Vecindarios/Encoding -> E) Metaheuristicas")

    demo_catalogos()                                     # A.1
    data, matriz, grafo, solucion = demo_cargas_basicas()   # A
    rutas_norm = demo_formato_solucion(data, solucion)  # B
    demo_factibilidad_y_costo(data, matriz, grafo, solucion)  # B.1
    demo_grafo_utils(data, grafo, rutas_norm)           # C
    demo_encoding_y_vecindarios(data, solucion)         # D
    demo_metaheuristicas()                              # E

    titulo("FIN")
    print("Script completado correctamente.")
    print("Si algo falla, la traza te indica exactamente que modulo revisar.")
    print("\nResumen rapido de input principal:")
    pprint({"instancia": INSTANCIA, "seed": SEED})


# Punto de entrada estándar de Python:
# Este bloque solo se ejecuta cuando se corre el script directamente
# (python testing.py), no cuando se importa como módulo.
if __name__ == "__main__":
    main()
