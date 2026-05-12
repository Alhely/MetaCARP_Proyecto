"""
Script de prueba/documentacion para todos los modulos publicos de metacarp.

OBJETIVO:
- Mostrar llamadas reales (input/output) de la API.
- Servir como guia para debug visual rapido usando la instancia pequena: gdb19.
- Explicar cuando tiene sentido pedir GPU en vecindarios.

EJECUCION:
    python metacarp/scripts/testing.py
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pprint import pprint
from typing import Any

from metacarp import (
    busqueda_abejas_desde_instancia,
    busqueda_tabu_desde_instancia,
    cuckoo_search_desde_instancia,
    OPERADORES_POPULARES,
    build_search_encoding,
    cargar_grafo,
    cargar_imagen_estatica,
    cargar_matriz_dijkstra,
    cargar_objeto_gexf,
    cargar_solucion_inicial,
    costo_camino_minimo,
    costo_solucion,
    costo_solucion_desde_instancia,
    decode_solution,
    decode_task_ids,
    dictionary_instances,
    edge_cost,
    encode_solution,
    etiquetas_tareas_requeridas,
    generar_vecino,
    generar_vecino_ids,
    load_instance,
    load_instances,
    nodo_grafo,
    nombres_matrices_disponibles,
    nombres_soluciones_iniciales_disponibles,
    normalizar_rutas_etiquetas,
    path_edges_and_cost,
    reporte_solucion,
    reporte_solucion_desde_instancia,
    recocido_simulado_desde_instancia,
    ruta_gexf,
    ruta_imagen_estatica,
    ruta_matriz_dijkstra,
    ruta_solucion_inicial,
    shortest_path_nodes,
    verificar_factibilidad,
    verificar_factibilidad_desde_instancia,
)


INSTANCIA = "gdb19"
SEED = 42
GUARDAR_CSV_DEMO = False


def titulo(txt: str) -> None:
    print("\n" + "=" * 90)
    print(txt)
    print("=" * 90)


def _resumen_salida(valor: Any, *, max_items: int = 5) -> str:
    """Resumen corto de retorno para imprimir en terminal."""
    if isinstance(valor, dict):
        ks = list(valor.keys())[:max_items]
        return f"dict(len={len(valor)}, keys_sample={ks})"
    if isinstance(valor, list):
        sample = valor[:max_items]
        return f"list(len={len(valor)}, sample={sample})"
    if isinstance(valor, tuple):
        sample = valor[:max_items]
        return f"tuple(len={len(valor)}, sample={sample})"
    if isinstance(valor, set):
        sample = list(valor)[:max_items]
        return f"set(len={len(valor)}, sample={sample})"
    return repr(valor)


def mostrar_llamada(
    *,
    comentario: str,
    modulo: str,
    codigo: str,
    valor: Any | None = None,
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
    fn: Callable[[], Any],
) -> Any:
    """Ejecuta y documenta una llamada individual."""
    valor = fn()
    mostrar_llamada(comentario=comentario, modulo=modulo, codigo=codigo, valor=valor)
    return valor


def demo_cargas_basicas() -> tuple[dict, object, object, list[list[str]]]:
    titulo("BLOQUE A) OBJETO INSTANCIA Y RECURSOS BASE")

    print(f"Instancia de prueba: {INSTANCIA}")
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


def demo_catalogos() -> None:
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


def demo_formato_solucion(data: dict, solucion: list[list[str]]) -> list[list[str]]:
    titulo("BLOQUE B) OBJETO SOLUCION - FORMATO Y NORMALIZACION")
    from metacarp.solucion_formato import resolver_etiqueta_canonica

    mapa = ejecutar_llamada(
        comentario="Construye mapa etiqueta -> dict de tarea.",
        modulo="metacarp.solucion_formato",
        codigo="construir_mapa_tareas_por_etiqueta(data)",
        fn=lambda: construir_mapa_tareas_por_etiqueta(data),
    )
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

    # Ejemplo de canonicalizacion de etiqueta
    if rutas_norm and rutas_norm[0]:
        et = rutas_norm[0][0]
        ejecutar_llamada(
            comentario="Canonicaliza una etiqueta (insensible a mayusculas).",
            modulo="metacarp.solucion_formato",
            codigo=f"resolver_etiqueta_canonica('{et.lower()}', mapa)",
            fn=lambda: resolver_etiqueta_canonica(et.lower(), mapa),
        )
    return rutas_norm


def demo_factibilidad_y_costo(data: dict, matriz: object, grafo: object, solucion: list[list[str]]) -> None:
    titulo("BLOQUE B.1) OBJETO SOLUCION - FACTIBILIDAD, COSTO Y REPORTE")

    # Factibilidad directa
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

    # Factibilidad desde instancia (helper)
    feas2 = ejecutar_llamada(
        comentario="Helper que carga data+matriz y valida factibilidad.",
        modulo="metacarp.factibilidad",
        codigo=f"verificar_factibilidad_desde_instancia('{INSTANCIA}', solucion)",
        fn=lambda: verificar_factibilidad_desde_instancia(INSTANCIA, solucion),
    )
    print(f"Factible (desde_instancia): {feas2.ok}")

    # Costo directo
    cost = ejecutar_llamada(
        comentario="Calcula costo total y por ruta.",
        modulo="metacarp.costo_solucion",
        codigo="costo_solucion(solucion, data, grafo, detalle=False)",
        fn=lambda: costo_solucion(solucion, data, grafo, detalle=False),
    )
    print(f"Costo total: {cost.costo_total}")
    print(f"Costos por ruta: {cost.costos_por_ruta}")
    print(f"Demandas por ruta: {cost.demandas_por_ruta}")

    # Costo desde instancia
    cost2 = ejecutar_llamada(
        comentario="Helper de costo que carga data+grafo internamente.",
        modulo="metacarp.costo_solucion",
        codigo=f"costo_solucion_desde_instancia('{INSTANCIA}', solucion, detalle=False)",
        fn=lambda: costo_solucion_desde_instancia(INSTANCIA, solucion, detalle=False),
    )
    print(f"Costo total (desde_instancia): {cost2.costo_total}")

    # Reporte directo
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

    # Reporte desde instancia
    rep2 = ejecutar_llamada(
        comentario="Helper de reporte que carga data+grafo.",
        modulo="metacarp.reporte_solucion",
        codigo=f"reporte_solucion_desde_instancia('{INSTANCIA}', solucion)",
        fn=lambda: reporte_solucion_desde_instancia(INSTANCIA, solucion),
    )
    print(f"Costo total (reporte_desde_instancia): {rep2.costo_total}")

    assert abs(cost.costo_total - rep.costo_total) < 1e-9, "Costo y reporte deben coincidir."


def demo_grafo_utils(data: dict, grafo: object, rutas_norm: list[list[str]]) -> None:
    titulo("BLOQUE C) OBJETO GRAFO - UTILIDADES DE CAMINOS Y COSTOS")
    deposito = int(data["DEPOSITO"])
    mapa = construir_mapa_tareas_por_etiqueta(data)

    # Tomamos una tarea real para ejemplo.
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


def demo_encoding_y_vecindarios(data: dict, solucion: list[list[str]]) -> None:
    titulo("BLOQUE D) OBJETO VECINDARIO Y BUSQUEDA INDEXADA")

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

    # Encode/decode roundtrip
    sol_ids = ejecutar_llamada(
        comentario="Convierte solucion de etiquetas a ids enteros.",
        modulo="metacarp.busqueda_indices",
        codigo="encode_solution(solucion, encoding)",
        fn=lambda: encode_solution(solucion, encoding),
    )
    sol_labels_roundtrip = ejecutar_llamada(
        comentario="Decodifica ids a etiquetas nuevamente.",
        modulo="metacarp.busqueda_indices",
        codigo="decode_solution(sol_ids, encoding, con_deposito=True)",
        fn=lambda: decode_solution(sol_ids, encoding, con_deposito=True),
    )
    assert sol_labels_roundtrip == solucion, "Roundtrip labels->ids->labels debe conservar solucion."
    print("Roundtrip labels->ids->labels: OK")
    print(f"Primera ruta en ids: {sol_ids[0] if sol_ids else []}")
    ejecutar_llamada(
        comentario="Decodifica solo un subconjunto de IDs a etiquetas TR.",
        modulo="metacarp.busqueda_indices",
        codigo="decode_task_ids(sol_ids[0][:3], encoding)",
        fn=lambda: decode_task_ids((sol_ids[0][:3] if sol_ids else []), encoding),
    )

    rng_labels = random.Random(SEED)
    rng_ids = random.Random(SEED)

    # Vecino en backend labels (default)
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

    # Vecino en backend ids (ruta indexada)
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

    # Llamada directa sobre ids (sin encode/decode implícito)
    vecino_ids_directo, mov_ids_directo = ejecutar_llamada(
        comentario="Genera vecino directamente sobre solucion en ids.",
        modulo="metacarp.vecindarios",
        codigo="generar_vecino_ids(sol_ids, rng=Random(SEED+1), usar_gpu=False, encoding=encoding)",
        fn=lambda: generar_vecino_ids(
            sol_ids,
            rng=random.Random(SEED + 1),
            usar_gpu=False,
            encoding=encoding,
        ),
    )
    print("\nVecino directo sobre IDs:")
    print(f"- movimiento: {mov_ids_directo}")
    print(f"- primera ruta ids vecino: {vecino_ids_directo[0] if vecino_ids_directo else []}")

    # GPU: cuando sirve
    # - Sirve cuando tengas backend GPU real (kernel para arrays indexados).
    # - Hoy usar_gpu=True mantiene trazabilidad y cae a CPU (backend_real='cpu').
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


def demo_metaheuristicas() -> None:
    titulo("BLOQUE E) METAHEURISTICAS - SA, TABU, ABEJAS, CUCKOO")
    print(f"Instancia: {INSTANCIA} | seed: {SEED} | guardar_csv_demo={GUARDAR_CSV_DEMO}")

    # Recocido Simulado (implementación clásica con enfriamiento geométrico).
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
            temperatura_inicial=150.0,
            temperatura_minima=1e-3,
            alpha=0.93,
            iteraciones_por_temperatura=40,
            max_enfriamientos=25,
            semilla=SEED,
            guardar_csv=GUARDAR_CSV_DEMO,
        ),
    )
    print(
        f"- SA: mejor_costo={sa.mejor_costo} | gap={sa.gap_porcentaje:.4f}% | "
        f"tiempo={sa.tiempo_segundos:.4f}s | csv={sa.archivo_csv}"
    )

    # Búsqueda Tabú.
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
            iteraciones=120,
            tam_vecindario=16,
            tenure_tabu=15,
            semilla=SEED,
            guardar_csv=GUARDAR_CSV_DEMO,
        ),
    )
    print(
        f"- Tabu: mejor_costo={tabu.mejor_costo} | gap={tabu.gap_porcentaje:.4f}% | "
        f"tiempo={tabu.tiempo_segundos:.4f}s | csv={tabu.archivo_csv}"
    )

    # Artificial Bee Colony (versión simplificada discreta).
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
            num_fuentes=12,
            limite_abandono=20,
            semilla=SEED,
            guardar_csv=GUARDAR_CSV_DEMO,
        ),
    )
    print(
        f"- Abejas: mejor_costo={abe.mejor_costo} | gap={abe.gap_porcentaje:.4f}% | "
        f"tiempo={abe.tiempo_segundos:.4f}s | csv={abe.archivo_csv}"
    )

    # Cuckoo Search discreto.
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
            num_nidos=14,
            pa_abandono=0.25,
            pasos_levy_base=3,
            beta_levy=1.5,
            semilla=SEED,
            guardar_csv=GUARDAR_CSV_DEMO,
        ),
    )
    print(
        f"- Cuckoo: mejor_costo={cko.mejor_costo} | gap={cko.gap_porcentaje:.4f}% | "
        f"tiempo={cko.tiempo_segundos:.4f}s | csv={cko.archivo_csv}"
    )


def construir_mapa_tareas_por_etiqueta(data: dict) -> dict[str, dict]:
    """Wrapper local para evitar import extra en cada demo."""
    from metacarp import construir_mapa_tareas_por_etiqueta as _f

    return _f(data)


def main() -> None:
    titulo("GUIA EJECUTABLE AGRUPADA POR TIPO DE OBJETO")
    print("Orden de lectura recomendado:")
    print("A) Instancia  -> B) Solucion  -> C) Grafo  -> D) Vecindarios/Encoding -> E) Metaheuristicas")

    demo_catalogos()
    data, matriz, grafo, solucion = demo_cargas_basicas()
    rutas_norm = demo_formato_solucion(data, solucion)
    demo_factibilidad_y_costo(data, matriz, grafo, solucion)
    demo_grafo_utils(data, grafo, rutas_norm)
    demo_encoding_y_vecindarios(data, solucion)
    demo_metaheuristicas()

    titulo("FIN")
    print("Script completado correctamente.")
    print("Si algo falla, la traza te indica exactamente que modulo revisar.")
    print("\nResumen rapido de input principal:")
    pprint({"instancia": INSTANCIA, "seed": SEED})


if __name__ == "__main__":
    main()