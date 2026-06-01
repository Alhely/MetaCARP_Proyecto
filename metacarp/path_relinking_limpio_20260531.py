"""
Path Relinking LIMPIO e independiente (20260531).

Reescritura del Path Relinking del proyecto SIN las dependencias sucias del
módulo experimental previo ``path_relinking_20260528``:

  - SIN ``sys._getframe`` (no roba ``ctx``/``lambda``/mejor-solución del stack
    de la metaheurística).
  - SIN monkey-patches (no parchea ``copiar_solucion_labels`` ni
    ``ContadorOperadores.registrar_mejora``).
  - SIN acoplarse al kick: aquí PR es una función PURA + un *hook* explícito
    con una firma fija que las metaheurísticas invocan pasando, de forma
    EXPLÍCITA, la solución actual, la mejor solución global, ``ctx`` y
    ``lambda``.

La matemática del relinking es la misma validada en el primer ciclo
(``path_relinking_20260528``): se camina greedy desde ``sol_inicio`` hacia
``sol_guia`` reasignando, en cada paso, la tarea cuyo movimiento minimiza el
objetivo penalizado, y se devuelve el MEJOR intermedio observado (PR truncado).

Uso desde una metaheurística (vía el parámetro ``intensificador`` que se añadió
a las 5 MH del proyecto). La MH llama:

    sol = intensificador(sol_actual, mejor_global, ctx, lam, rng, encoding, md)

y este módulo provee dos *hooks* listos para usar con esa firma:
  - ``hook_pr_labels``: backend de etiquetas (SA, TS, RTS, Cuckoo).
  - ``hook_pr_ids``:    backend de IDs enteros (ABC simple).

Ambos devuelven una solución cuyo objetivo penalizado es ``<=`` al de
``sol_inicio`` (PR truncado guarda el mejor intermedio, que como mínimo es la
propia ``sol_inicio``), por lo que la MH puede asignarla directamente sin
arriesgar un retroceso.
"""
from __future__ import annotations

from typing import Any, Sequence

from metacarp.evaluador_costo import (
    costo_rapido,
    costo_rapido_ids,
    exceso_capacidad_rapido,
    exceso_capacidad_sol_ids,
)

# Tope absoluto de pasos de PR (cada paso reasigna UNA tarea hacia su posición
# guía). En la práctica el camino se trunca antes porque guardamos el MEJOR
# intermedio. Mismo valor que el módulo del primer ciclo, para comparabilidad.
MAX_PASOS_PR_FACTOR: int = 50


# ============================================================
# Mapas de posición (tarea -> (ruta, posición intra-ruta))
# ============================================================

def _construir_mapa_pos_labels(
    sol: Sequence[Sequence[str]], marcador_depot: str
) -> dict[str, tuple[int, int]]:
    """Mapea cada tarea (label) a su (ruta_idx, posición), ignorando depots.

    Una ruta ``["D","TR1","TR5","D"]`` da posición 0 a ``TR1`` y 1 a ``TR5``.
    """
    mapa: dict[str, tuple[int, int]] = {}
    for r_idx, ruta in enumerate(sol):
        pos = 0
        for tok in ruta:
            if tok == marcador_depot:
                continue
            mapa[str(tok)] = (r_idx, pos)
            pos += 1
    return mapa


def _construir_mapa_pos_ids(
    sol_ids: Sequence[Sequence[int]],
) -> dict[int, tuple[int, int]]:
    """Versión IDs: sin depots intercalados, ``len(ruta)`` = nº de tareas."""
    mapa: dict[int, tuple[int, int]] = {}
    for r_idx, ruta in enumerate(sol_ids):
        for pos, tid in enumerate(ruta):
            mapa[int(tid)] = (r_idx, pos)
    return mapa


# ============================================================
# Objetivo penalizado (costo + lambda * violación)
# ============================================================

def _objetivo_penalizado_labels(
    sol: Sequence[Sequence[str]], ctx: Any, lam: float
) -> float:
    """Objetivo penalizado de una solución en labels."""
    cp = float(costo_rapido(sol, ctx))
    viol = float(exceso_capacidad_rapido(sol, ctx))
    return cp + lam * viol


def _objetivo_penalizado_ids(
    sol_ids: Sequence[Sequence[int]], ctx: Any, lam: float
) -> float:
    """Objetivo penalizado de una solución en IDs."""
    cp = float(costo_rapido_ids(sol_ids, ctx))
    viol = float(exceso_capacidad_sol_ids(sol_ids, ctx))
    return cp + lam * viol


# ============================================================
# Movimiento de una tarea a (ruta_destino, pos_destino)
# ============================================================

def _mover_tarea_labels(
    sol: list[list[str]],
    tarea: str,
    ruta_destino: int,
    pos_destino: int,
    marcador_depot: str,
) -> list[list[str]] | None:
    """Nueva solución con ``tarea`` movida a (ruta_destino, pos_destino).

    Devuelve None si la tarea no existe. La posición se cuenta sin depots.
    """
    nueva = [list(r) for r in sol]
    encontrada = False
    for ruta in nueva:
        for i, tok in enumerate(ruta):
            if tok == tarea:
                ruta.pop(i)
                encontrada = True
                break
        if encontrada:
            break
    if not encontrada:
        return None
    # Ampliar el número de rutas si PR sugiere una ruta inexistente.
    while len(nueva) <= ruta_destino:
        nueva.append([marcador_depot, marcador_depot])
    ruta_d = nueva[ruta_destino]
    # Localizar el índice físico de inserción contando posiciones reales.
    pos_actual = 0
    idx_insertar = len(ruta_d)
    for i, tok in enumerate(ruta_d):
        if tok == marcador_depot:
            continue
        if pos_actual == pos_destino:
            idx_insertar = i
            break
        pos_actual += 1
    # Si la posición destino supera el nº de tareas, insertar antes del depot final.
    if pos_actual < pos_destino and ruta_d and ruta_d[-1] == marcador_depot:
        idx_insertar = len(ruta_d) - 1
    ruta_d.insert(idx_insertar, tarea)
    return nueva


def _mover_tarea_ids(
    sol_ids: list[list[int]],
    tarea: int,
    ruta_destino: int,
    pos_destino: int,
) -> list[list[int]] | None:
    """Versión IDs de ``_mover_tarea_labels`` (sin depots intercalados)."""
    nueva = [list(r) for r in sol_ids]
    encontrada = False
    for ruta in nueva:
        for i, tid in enumerate(ruta):
            if tid == tarea:
                ruta.pop(i)
                encontrada = True
                break
        if encontrada:
            break
    if not encontrada:
        return None
    while len(nueva) <= ruta_destino:
        nueva.append([])
    ruta_d = nueva[ruta_destino]
    idx_insertar = min(pos_destino, len(ruta_d))
    ruta_d.insert(idx_insertar, tarea)
    return nueva


# ============================================================
# Path Relinking truncado (funciones puras)
# ============================================================

def path_relinking_labels(
    sol_inicio: list[list[str]],
    sol_guia: list[list[str]],
    ctx: Any,
    lam: float,
    *,
    marcador_depot: str = "D",
) -> list[list[str]]:
    """PR truncado entre dos soluciones en formato labels.

    Camina greedy de ``sol_inicio`` hacia ``sol_guia`` reasignando en cada paso
    la tarea (de las que difieren) cuyo movimiento minimiza el objetivo
    penalizado, y devuelve el MEJOR intermedio observado. Si las dos soluciones
    ya coinciden, devuelve una copia de ``sol_inicio``.
    """
    mapa_ini = _construir_mapa_pos_labels(sol_inicio, marcador_depot)
    mapa_gui = _construir_mapa_pos_labels(sol_guia, marcador_depot)

    # delta: tareas con asignación (ruta, pos) distinta entre inicio y guía.
    delta: dict[str, tuple[int, int]] = {}
    for t, pos_g in mapa_gui.items():
        if mapa_ini.get(t) != pos_g:
            delta[t] = pos_g

    if not delta:
        return [list(r) for r in sol_inicio]

    max_pasos = min(len(delta), MAX_PASOS_PR_FACTOR)
    sol_actual = [list(r) for r in sol_inicio]
    sol_mejor = [list(r) for r in sol_actual]
    obj_mejor = _objetivo_penalizado_labels(sol_actual, ctx, lam)

    for _ in range(max_pasos):
        if not delta:
            break
        mejor_t: str | None = None
        mejor_sol_paso: list[list[str]] | None = None
        mejor_obj_paso = float("inf")
        for tarea, (r_dst, pos_dst) in list(delta.items()):
            candidata = _mover_tarea_labels(
                sol_actual, tarea, r_dst, pos_dst, marcador_depot
            )
            if candidata is None:
                delta.pop(tarea, None)
                continue
            obj_c = _objetivo_penalizado_labels(candidata, ctx, lam)
            if obj_c < mejor_obj_paso:
                mejor_obj_paso = obj_c
                mejor_sol_paso = candidata
                mejor_t = tarea
        if mejor_t is None or mejor_sol_paso is None:
            break
        # Avanzar greedy hacia la guía (aunque el paso empeore: el valle suele
        # estar entre las dos soluciones buenas).
        sol_actual = mejor_sol_paso
        delta.pop(mejor_t, None)
        if mejor_obj_paso < obj_mejor - 1e-15:
            obj_mejor = mejor_obj_paso
            sol_mejor = [list(r) for r in sol_actual]

    return sol_mejor


def path_relinking_ids(
    sol_inicio: list[list[int]],
    sol_guia: list[list[int]],
    ctx: Any,
    lam: float,
) -> list[list[int]]:
    """PR truncado para soluciones en formato IDs (ABC simple)."""
    mapa_ini = _construir_mapa_pos_ids(sol_inicio)
    mapa_gui = _construir_mapa_pos_ids(sol_guia)

    delta: dict[int, tuple[int, int]] = {}
    for t, pos_g in mapa_gui.items():
        if mapa_ini.get(t) != pos_g:
            delta[t] = pos_g

    if not delta:
        return [list(r) for r in sol_inicio]

    max_pasos = min(len(delta), MAX_PASOS_PR_FACTOR)
    sol_actual = [list(r) for r in sol_inicio]
    sol_mejor = [list(r) for r in sol_actual]
    obj_mejor = _objetivo_penalizado_ids(sol_actual, ctx, lam)

    for _ in range(max_pasos):
        if not delta:
            break
        mejor_t: int | None = None
        mejor_sol_paso: list[list[int]] | None = None
        mejor_obj_paso = float("inf")
        for tarea, (r_dst, pos_dst) in list(delta.items()):
            candidata = _mover_tarea_ids(sol_actual, tarea, r_dst, pos_dst)
            if candidata is None:
                delta.pop(tarea, None)
                continue
            obj_c = _objetivo_penalizado_ids(candidata, ctx, lam)
            if obj_c < mejor_obj_paso:
                mejor_obj_paso = obj_c
                mejor_sol_paso = candidata
                mejor_t = tarea
        if mejor_t is None or mejor_sol_paso is None:
            break
        sol_actual = mejor_sol_paso
        delta.pop(mejor_t, None)
        if mejor_obj_paso < obj_mejor - 1e-15:
            obj_mejor = mejor_obj_paso
            sol_mejor = [list(r) for r in sol_actual]

    return sol_mejor


# ============================================================
# Hooks de intensificación (firma fija que invocan las MH)
# ============================================================
# Firma del hook:
#   intensificador(sol_inicio, mejor_global, ctx, lam, rng, encoding, md) -> sol
# ``rng`` y ``encoding`` se aceptan por uniformidad de firma (PR no los usa:
# es determinista dado (sol_inicio, mejor_global, ctx, lam)).

def hook_pr_labels(
    sol_inicio: list[list[str]],
    mejor_global: list[list[str]] | None,
    ctx: Any,
    lam: float,
    rng: Any = None,
    encoding: Any = None,
    marcador_depot: str = "D",
) -> list[list[str]]:
    """Hook PR para metaheurísticas con backend de etiquetas.

    Si aún no hay mejor global (``None``) devuelve ``sol_inicio`` sin tocar.
    """
    if not mejor_global:
        return sol_inicio
    return path_relinking_labels(
        sol_inicio, mejor_global, ctx, lam, marcador_depot=marcador_depot
    )


def hook_pr_ids(
    sol_inicio: list[list[int]],
    mejor_global: list[list[int]] | None,
    ctx: Any,
    lam: float,
    rng: Any = None,
    encoding: Any = None,
    marcador_depot: Any = None,
) -> list[list[int]]:
    """Hook PR para metaheurísticas con backend de IDs (ABC simple)."""
    if not mejor_global:
        return sol_inicio
    return path_relinking_ids(sol_inicio, mejor_global, ctx, lam)
