"""
Constructor heuristico de Path-Scanning para CARP (Golden, DeArmon, Baker 1983).

Path-Scanning es la heuristica constructiva clasica para CARP. Inicia desde
el deposito y, en cada paso, selecciona la siguiente tarea como:

  1. La factible (cabe en lo que resta de capacidad)
  2. Mas cercana al nodo actual (minimiza dead-heading)
  3. En caso de empate, aplica una de 5 REGLAS de seleccion sobre la tarea
     o su orientacion:

       Regla 1: MAX dist(salida, deposito)  — alejarse del deposito
       Regla 2: MIN dist(salida, deposito)  — acercarse al deposito
       Regla 3: MAX demanda/costo_servicio  — priorizar densa/corta
       Regla 4: MIN demanda/costo_servicio  — priorizar ligera/larga
       Regla 5: si carga < Q/2 usa regla 1, si no regla 2 (adaptativa)

Cuando una ruta no admite mas tareas, regresa al deposito e inicia otra.
Se ejecutan las 5 variantes y se devuelve la mejor (menor costo total con
dead-heading + costo de servicio).

Formato de salida compatible con ``cargar_solucion_inicial``:
``list[list[str]]`` con etiquetas ``['D','TR3','TR5','D']`` (depot='D',
tareas='TRk').
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Path-Scanning (una variante = una regla)
# ---------------------------------------------------------------------------

def _path_scanning_una_regla(
    data: dict,
    M: dict[int, dict[int, float]],
    regla: int,
    marcador_depot: str = "D",
) -> tuple[list[list[str]], float]:
    """Construye UNA solucion CARP con la regla de Golden indicada.

    Parametros
    ----------
    data : dict
        Datos de la instancia segun ``load_instances``. Se usan:
        ``LISTA_ARISTAS_REQ`` (tareas), ``DEPOSITO``, ``CAPACIDAD``.
    M : dict[int, dict[int, num]]
        Matriz Dijkstra de distancias minimas entre cualquier par de nodos.
    regla : int in {1,2,3,4,5}
        Regla de Golden para resolver empates de distancia.
    marcador_depot : str
        Etiqueta del deposito en la solucion (por defecto ``"D"``).

    Retorna
    -------
    (rutas, costo_total) donde:
      - rutas : list[list[str]] en formato labels
      - costo_total : float, costo de servicio + dead-heading total
    """
    depot   = int(data["DEPOSITO"])
    Q       = int(data["CAPACIDAD"])
    tareas  = list(data["LISTA_ARISTAS_REQ"])  # cada uno: tarea,nodos,costo,demanda

    # Indices de tareas sin servir (los iremos eliminando).
    no_servidas: set[int] = set(range(len(tareas)))

    # Conjunto de rutas en construccion.
    rutas: list[list[str]] = []
    costo_total: float = 0.0

    while no_servidas:
        # Iniciamos una ruta nueva desde el deposito.
        ruta: list[str] = [marcador_depot]
        nodo_actual = depot
        carga = 0
        costo_ruta = 0.0

        while True:
            # ---- Generar opciones factibles (orientacion CANONICA u->v) ----
            # Las MH evaluan cada tarea entrando por ``u`` y saliendo por ``v``
            # tal como aparece en ``LISTA_ARISTAS_REQ`` (ver costo_rapido_ids).
            # NO elegimos orientacion porque las MH no la respetan al evaluar.
            # Cada opcion es (tarea_idx, u, v, dist_actual_a_u).
            opciones: list[tuple[int, int, int, float]] = []
            for i in no_servidas:
                t = tareas[i]
                if carga + int(t["demanda"]) > Q:
                    continue  # no cabe
                u, v = t["nodos"]
                opciones.append((i, int(u), int(v), float(M[nodo_actual][u])))

            if not opciones:
                # No hay tareas que quepan; cerramos la ruta.
                break

            # ---- Encontrar la(s) opcion(es) de menor distancia -------------
            dist_min = min(op[3] for op in opciones)
            candidatas = [op for op in opciones if op[3] == dist_min]

            # ---- Tie-break con la regla de Golden seleccionada -------------
            if len(candidatas) == 1:
                elegida = candidatas[0]
            else:
                # Aplicamos la regla solo cuando hay empate en la distancia
                # mas corta (es la formulacion clasica de Golden et al.).
                if regla == 1:
                    # max dist(salida, deposito)
                    elegida = max(candidatas, key=lambda op: float(M[op[2]][depot]))
                elif regla == 2:
                    # min dist(salida, deposito)
                    elegida = min(candidatas, key=lambda op: float(M[op[2]][depot]))
                elif regla == 3:
                    # max demanda/costo_servicio
                    elegida = max(
                        candidatas,
                        key=lambda op: tareas[op[0]]["demanda"] / max(tareas[op[0]]["costo"], 1e-9),
                    )
                elif regla == 4:
                    # min demanda/costo_servicio
                    elegida = min(
                        candidatas,
                        key=lambda op: tareas[op[0]]["demanda"] / max(tareas[op[0]]["costo"], 1e-9),
                    )
                else:  # regla 5: adaptativa segun carga
                    if carga < Q / 2:
                        elegida = max(candidatas, key=lambda op: float(M[op[2]][depot]))
                    else:
                        elegida = min(candidatas, key=lambda op: float(M[op[2]][depot]))

            # Aplicar la opcion elegida ---------------------------------------
            i_t, entrada, salida, d_entrada = elegida
            t_obj = tareas[i_t]

            # Sumar dead-heading hasta la entrada + costo de servicio.
            costo_ruta += d_entrada + float(t_obj["costo"])
            ruta.append(str(t_obj["tarea"]))
            carga += int(t_obj["demanda"])
            nodo_actual = salida
            no_servidas.discard(i_t)

        # Cerramos la ruta regresando al deposito.
        costo_ruta += float(M[nodo_actual][depot])
        ruta.append(marcador_depot)

        rutas.append(ruta)
        costo_total += costo_ruta

    return rutas, costo_total


# ---------------------------------------------------------------------------
# Path-Scanning mejor-de-5
# ---------------------------------------------------------------------------

def path_scanning(
    data: dict,
    M: dict[int, dict[int, float]],
    marcador_depot: str = "D",
) -> list[list[str]]:
    """Devuelve la MEJOR solucion entre las 5 reglas de Path-Scanning.

    Ejecuta el constructor con cada una de las 5 reglas de Golden et al.
    y devuelve la solucion con menor costo total (dead-heading + servicio).
    Cuando hay empate, prefiere la regla de menor indice.

    Parametros
    ----------
    data : dict
        Datos de la instancia (``load_instances``).
    M : dict[int, dict[int, num]]
        Matriz Dijkstra de distancias minimas.
    marcador_depot : str
        Etiqueta del deposito (por defecto ``"D"``).

    Retorna
    -------
    list[list[str]]
        Solucion en formato labels, compatible con el pickle original.
    """
    mejor_sol: list[list[str]] | None = None
    mejor_costo: float = float("inf")

    for regla in (1, 2, 3, 4, 5):
        sol, costo = _path_scanning_una_regla(data, M, regla=regla, marcador_depot=marcador_depot)
        if costo < mejor_costo:
            mejor_costo = costo
            mejor_sol = sol

    assert mejor_sol is not None  # siempre hay >= 1 solucion factible
    return mejor_sol


def path_scanning_desde_instancia(
    nombre_instancia: str,
    *,
    root: str | None = None,
    marcador_depot: str = "D",
) -> list[list[str]]:
    """Helper que carga los datos y la matriz, luego ejecuta path_scanning.

    Conveniencia para uso fuera del flujo de las MH (p.ej. al regenerar
    pickles o validar la calidad de la construccion).
    """
    # Importacion diferida para no inflar el modulo cuando solo se usa
    # ``path_scanning`` directamente con data y M ya cargados.
    from metacarp.instances import load_instances
    from metacarp.cargar_matrices import cargar_matriz_dijkstra

    data = load_instances(nombre_instancia, root=root)
    M = cargar_matriz_dijkstra(nombre_instancia, root=root)
    return path_scanning(data, M, marcador_depot=marcador_depot)
