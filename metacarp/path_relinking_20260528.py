"""
Variante experimental path_relinking_20260528 -- Path Relinking truncado
montado por encima del selector binario estricto + kick reactivo.

Diseno (Opcion 1: monkey-patch SIN tocar archivos de MH):
  - Las 5 MH (SA, TS Simple, RTS, ABC Simple, Cuckoo) hacen un import
    DIFERIDO dentro de su bucle principal:
        from metacarp.strict_intra_inter_20260524 import aplicar_kick_labels
        sol_actual = aplicar_kick_labels(sol_actual, rng, md_op, encoding=encoding)
    Al ser un import diferido, este modulo puede REASIGNAR el atributo
    ``aplicar_kick_labels`` (y ``aplicar_kick_ids`` para ABC Simple) en el
    namespace de ``metacarp.strict_intra_inter_20260524`` ANTES de que la MH
    lo invoque, y las MH automaticamente recibiran nuestra version aumentada
    con PR. CERO ediciones en las MH.

  - El wrapper aumentado mantiene la firma EXACTA de la funcion original:
      aplicar_kick_con_pr_labels(sol, rng, md_op, encoding=None)
      aplicar_kick_con_pr_ids(sol_ids, rng, encoding=None)
    para ser sustituible 1-a-1.

  - Con probabilidad ``p_pr`` (default 0.5) ejecutamos Path Relinking desde
    la solucion actual (despues del kick puro) hacia la MEJOR SOLUCION GLOBAL
    encontrada hasta ese momento. Con probabilidad ``1 - p_pr`` se ejecuta
    solo el kick puro (comportamiento canonico).

Captura de la mejor solucion global (``mejor_sol``):
  - Las 5 MH actualizan ``mejor_sol`` (o ``mejor_any_s``/``mejor_fact_s``/
    ``mejor_sol_ids``) cada vez que se detecta una mejora del mejor global.
  - Para LABELS: las MH SA, TS Simple, RTS, Cuckoo usan
    ``copiar_solucion_labels(sol)`` para hacer esa copia. Patcheamos esa
    funcion para registrar la ULTIMA solucion copiada como tentativa de
    "mejor_sol_labels" en ``_estado_pr``.
  - Para IDS: ABC Simple NO usa ningun helper analogo (``copiar_solucion_ids``
    no existe en el codebase). Para ABC, en su lugar, patcheamos
    ``ContadorOperadores.registrar_mejora`` y extraemos ``mejor_sol_ids``
    del frame del llamante via ``sys._getframe(1).f_locals['mejor_sol_ids']``.
    Esta tecnica es feo pero funcional y respeta la restriccion de no tocar MH.

Extraccion de ``ctx`` y ``lambda_capacidad`` sin tocar MH:
  - El kick wrapper necesita ``ctx`` (para evaluar costos) y ``lambda``
    (para el objetivo penalizado). Ni ``ctx`` ni ``lambda`` viajan en la
    firma de ``aplicar_kick_*``. Estrategia: usar ``sys._getframe(1).f_locals``
    para extraer ``ctx`` del scope del llamante. Las 5 MH tienen ``ctx``
    como variable local en el frame que invoca al kick. Si no se encuentra,
    se omite PR y se ejecuta solo el kick puro (fallback seguro).
  - ``lambda_capacidad`` se calcula como ``lambda_penal_capacidad_por_defecto(ctx)``
    si la MH no expone una variable con ese nombre. En la practica, las MH
    nombran su lambda como ``lam_eff`` o ``lambda_eff``. Lo intentamos leer
    del mismo frame; si falla, caemos al default canonico.

Falsos positivos del patch ``copiar_solucion_labels``:
  - Las MH SA y Cuckoo tambien copian la solucion inicial al arrancar
    (``mejor_any_s = copiar_solucion_labels(sol_ref)``) y la final
    (``sol_mejor = copiar_solucion_labels(mejor_fact_s if ...)``). Esto
    significa que el ULTIMO valor capturado puede NO ser la mejor global
    sino una copia rutinaria. En la practica:
      (a) al primer kick (que ocurre muchos pasos despues del arranque),
          ``mejor_sol_labels`` SI corresponde a la mejor global vigente
          porque las copias intermedias dentro del bucle solo ocurren cuando
          se detecta una mejora del mejor global.
      (b) las copias finales al cerrar la corrida ocurren DESPUES del bucle
          principal, asi que nunca se observan dentro de un kick.
    Conclusion: el patch es seguro EN LA PRACTICA, con el caveat de que el
    "primer kick" tras la inicializacion podria recibir la solucion inicial
    como "mejor" si la MH aun no ha mejorado. En ese caso PR es no-op porque
    sol_inicio == sol_guia => delta vacio.

Penalized objective:
  - PR SIEMPRE compara con ``costo + lambda*violacion``. NUNCA acepta una
    mejora solo por costo puro: una solucion mas barata pero infactible
    seria un retroceso desde el punto de vista de la MH.
"""
from __future__ import annotations

import importlib
import random
import sys
import threading
from typing import Any, Sequence

from metacarp.evaluador_costo import (
    costo_rapido,
    costo_rapido_ids,
    exceso_capacidad_rapido,
    exceso_capacidad_sol_ids,
    lambda_penal_capacidad_por_defecto,
)
from metacarp.strict_intra_inter_20260524 import (
    OPERADORES_INTER_STRICT,
    aplicar_kick_labels as _kick_labels_orig,
    aplicar_kick_ids as _kick_ids_orig,
)


# ============================================================
# Constantes
# ============================================================

# Probabilidad por defecto de disparar PR cuando se ejecuta un kick.
# El usuario eligio 0.5 para este experimento: balance 50/50 entre kick puro
# (diversificacion) y PR (intensificacion guiada hacia el mejor global).
P_PR_DEFAULT: float = 0.5

# Tope absoluto de pasos de PR para evitar degenerar en instancias grandes.
# Cada paso reasigna UNA tarea hacia su posicion guia. Con 50 tareas
# diferentes, 50 pasos significa reconstruir totalmente sol_inicio hacia
# sol_guia (que es lo mismo que sol_guia). En la practica, el camino se
# trunca antes porque guardamos el MEJOR intermedio.
MAX_PASOS_PR_FACTOR: int = 50


# ============================================================
# Estado global PR (thread-local + atributos de modulo)
# ============================================================

# Estado thread-local: cada thread/worker tiene su propia ULTIMA solucion
# copiada (la usamos como guia para PR). Aunque los workers son procesos
# independientes (ProcessPoolExecutor), seguimos usando threading.local()
# por hygiene en caso de que el usuario corra el experimento desde un
# notebook donde varios hilos compartan el modulo.
_estado_pr_local = threading.local()


def _get_mejor_sol_labels() -> list[list[str]] | None:
    """Devuelve la ultima solucion en labels capturada como tentativa de mejor."""
    return getattr(_estado_pr_local, "mejor_sol_labels", None)


def _set_mejor_sol_labels(sol: list[list[str]] | None) -> None:
    _estado_pr_local.mejor_sol_labels = sol


def _get_mejor_sol_ids() -> list[list[int]] | None:
    """Devuelve la ultima solucion en IDs capturada como tentativa de mejor."""
    return getattr(_estado_pr_local, "mejor_sol_ids", None)


def _set_mejor_sol_ids(sol: list[list[int]] | None) -> None:
    _estado_pr_local.mejor_sol_ids = sol


# Probabilidad efectiva de PR por proceso. Se actualiza en aplicar_patch_pr.
_p_pr_actual: float = P_PR_DEFAULT
# Guardas para evitar doble-wrapping en modo secuencial.
_copiar_labels_patched: bool = False
_registrar_mejora_patched: bool = False
_kicks_patched: bool = False


# ============================================================
# Helpers de extraccion de ctx / lambda desde el frame del llamante
# ============================================================

def _extraer_ctx_y_lambda(profundidad_inicial: int = 2) -> tuple[Any, float | None]:
    """Busca ``ctx`` y la lambda de penalizacion en frames ascendentes.

    Sube por la pila de llamadas hasta encontrar un frame con ``ctx`` en sus
    locales. Una vez encontrado ``ctx``, intenta leer ``lam_eff``,
    ``lambda_eff`` o ``lambda_capacidad`` del MISMO frame. Si ninguno de
    esos existe, retorna ``lambda_penal_capacidad_por_defecto(ctx)`` para
    no devolver ``None`` cuando si hay ctx valido.

    Devuelve ``(ctx, lambda)`` o ``(None, None)`` si no se encuentra ctx
    en ningun frame razonable.

    El parametro ``profundidad_inicial`` controla desde que nivel de la pila
    se empieza a buscar (default 2: saltarse este helper y el wrapper).
    """
    frame = sys._getframe(profundidad_inicial)
    # Limitamos la busqueda a una profundidad razonable: 25 frames hacia arriba
    # cubren cualquier MH de este proyecto sin recorrer todo el stack.
    for _ in range(25):
        if frame is None:
            break
        locales = frame.f_locals
        ctx = locales.get("ctx")
        if ctx is not None:
            # Intentamos leer lambda del mismo frame con varios nombres
            # conocidos. Si no esta, lo derivamos del ctx mismo.
            lam = (
                locales.get("lam_eff")
                or locales.get("lambda_eff")
                or locales.get("lambda_capacidad_eff")
                or locales.get("lambda_capacidad")
            )
            if lam is None:
                # Calculo defensivo: usa el default canonico del modulo
                # evaluador. Garantiza que PR siempre tenga una lambda valida.
                try:
                    lam = float(lambda_penal_capacidad_por_defecto(ctx))
                except Exception:  # noqa: BLE001
                    lam = None
            else:
                lam = float(lam)
            return ctx, lam
        frame = frame.f_back
    return None, None


def _extraer_mejor_sol_ids_del_caller() -> list[list[int]] | None:
    """Busca ``mejor_sol_ids`` en frames ascendentes (para ABC Simple).

    ABC Simple no usa ``copiar_solucion_*`` para mantener el mejor global,
    sino una asignacion explicita ``mejor_sol_ids = [list(r) for r in vec_ids]``
    inmediatamente antes de invocar ``contador.registrar_mejora(...)``. Cada
    vez que el patch de ``registrar_mejora`` se activa, leemos esa variable
    del frame del llamante para mantener actualizada la guia de PR.
    """
    frame = sys._getframe(2)
    for _ in range(15):
        if frame is None:
            break
        msi = frame.f_locals.get("mejor_sol_ids")
        if msi is not None:
            try:
                # Copia defensiva: no queremos compartir referencias con la MH.
                return [list(r) for r in msi]
            except Exception:  # noqa: BLE001
                return None
        frame = frame.f_back
    return None


# ============================================================
# Path Relinking propiamente dicho (representacion arc-set)
# ============================================================

def _construir_mapa_pos_labels(
    sol: Sequence[Sequence[str]], marcador_depot: str
) -> dict[str, tuple[int, int]]:
    """Mapea cada tarea (label) a su (ruta_idx, posicion_intra_ruta).

    La posicion se cuenta IGNORANDO los depots: una ruta ``["D","TR1","TR5","D"]``
    da posicion 0 para ``TR1`` y posicion 1 para ``TR5``. Esto facilita
    comparar dos soluciones en terminos de "tarea en posicion".
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
    """Mapea cada tarea (id entero) a su (ruta_idx, posicion_intra_ruta).

    En la representacion IDs no hay depots intercalados: ``len(ruta)`` ya
    es el numero de tareas reales.
    """
    mapa: dict[int, tuple[int, int]] = {}
    for r_idx, ruta in enumerate(sol_ids):
        for pos, tid in enumerate(ruta):
            mapa[int(tid)] = (r_idx, pos)
    return mapa


def _objetivo_penalizado_labels(
    sol: Sequence[Sequence[str]], ctx: Any, lam: float
) -> tuple[float, float, float]:
    """Devuelve (objetivo_penalizado, costo_puro, violacion) para labels."""
    cp = float(costo_rapido(sol, ctx))
    viol = float(exceso_capacidad_rapido(sol, ctx))
    return cp + lam * viol, cp, viol


def _objetivo_penalizado_ids(
    sol_ids: Sequence[Sequence[int]], ctx: Any, lam: float
) -> tuple[float, float, float]:
    """Devuelve (objetivo_penalizado, costo_puro, violacion) para IDs."""
    cp = float(costo_rapido_ids(sol_ids, ctx))
    viol = float(exceso_capacidad_sol_ids(sol_ids, ctx))
    return cp + lam * viol, cp, viol


def _mover_tarea_labels(
    sol: list[list[str]],
    tarea: str,
    ruta_destino: int,
    pos_destino: int,
    marcador_depot: str,
) -> list[list[str]] | None:
    """Construye una NUEVA solucion con ``tarea`` movida a (ruta_destino, pos_destino).

    Devuelve None si la tarea no existe en la solucion o la ruta destino
    no existe. La posicion se cuenta sin contar depots (igual que
    _construir_mapa_pos_labels), y se clampa al rango valido.
    """
    # Encontrar y eliminar la tarea de su ruta actual.
    nueva = [list(r) for r in sol]
    encontrada = False
    for r_idx, ruta in enumerate(nueva):
        for i, tok in enumerate(ruta):
            if tok == tarea:
                ruta.pop(i)
                encontrada = True
                break
        if encontrada:
            break
    if not encontrada:
        return None
    # Si la ruta destino no existe (PR podria sugerir una ruta inexistente
    # si las dos soluciones difieren en numero de rutas), creamos una ruta
    # vacia con depots a los extremos para ampliar el numero de vehiculos.
    while len(nueva) <= ruta_destino:
        nueva.append([marcador_depot, marcador_depot])
    # Insertar en la posicion destino. La posicion se cuenta ignorando depots.
    ruta_d = nueva[ruta_destino]
    # Recorremos la ruta destino contando posiciones de tareas reales y
    # localizamos el indice fisico donde insertar.
    pos_actual = 0
    idx_insertar = len(ruta_d)
    for i, tok in enumerate(ruta_d):
        if tok == marcador_depot:
            continue
        if pos_actual == pos_destino:
            idx_insertar = i
            break
        pos_actual += 1
    # Si no se encontro la posicion (pos_destino mayor que el numero de
    # tareas), insertamos antes del depot final. Buscamos el ultimo depot.
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
    """Version IDs de ``_mover_tarea_labels``. Sin depots intercalados."""
    nueva = [list(r) for r in sol_ids]
    encontrada = False
    for r_idx, ruta in enumerate(nueva):
        for i, tid in enumerate(ruta):
            if tid == tarea:
                ruta.pop(i)
                encontrada = True
                break
        if encontrada:
            break
    if not encontrada:
        return None
    # Ampliar el numero de rutas si hace falta (caso degenerado).
    while len(nueva) <= ruta_destino:
        nueva.append([])
    ruta_d = nueva[ruta_destino]
    idx_insertar = min(pos_destino, len(ruta_d))
    ruta_d.insert(idx_insertar, tarea)
    return nueva


def path_relinking_labels(
    sol_inicio: list[list[str]],
    sol_guia: list[list[str]],
    ctx: Any,
    lam: float,
    marcador_depot: str = "D",
) -> list[list[str]]:
    """Path Relinking truncado entre dos soluciones en formato labels.

    Algoritmo:
      1. Calcular ``delta`` = tareas con asignacion distinta entre inicio y guia.
      2. En cada paso, considerar TODAS las tareas en delta y aplicar la
         que produce el menor objetivo_penalizado en el siguiente estado.
         Si ninguna mejora estricta, igual se aplica la mejor (PR avanza
         hacia la guia aunque empeore: el espacio entre dos soluciones
         buenas suele contener un valle).
      3. Truncar tras ``min(len(delta_inicial), MAX_PASOS_PR_FACTOR)`` pasos.
      4. Devolver la MEJOR solucion intermedia observada (no la final).

    Si el delta inicial es vacio (sol_inicio == sol_guia) devuelve sol_inicio.
    """
    mapa_ini = _construir_mapa_pos_labels(sol_inicio, marcador_depot)
    mapa_gui = _construir_mapa_pos_labels(sol_guia, marcador_depot)

    # Tareas con asignacion distinta. Comparamos (ruta, pos) en ambas
    # soluciones; si difieren en cualquiera de las dos componentes, la
    # tarea pertenece al delta de PR.
    delta: dict[str, tuple[int, int]] = {}
    for t, pos_g in mapa_gui.items():
        pos_i = mapa_ini.get(t)
        if pos_i != pos_g:
            delta[t] = pos_g

    if not delta:
        # PR no tiene nada que hacer: las dos soluciones ya coinciden.
        return [list(r) for r in sol_inicio]

    max_pasos = min(len(delta), MAX_PASOS_PR_FACTOR)
    sol_actual = [list(r) for r in sol_inicio]
    obj_actual, _, _ = _objetivo_penalizado_labels(sol_actual, ctx, lam)
    sol_mejor = [list(r) for r in sol_actual]
    obj_mejor = obj_actual

    for _ in range(max_pasos):
        if not delta:
            break
        # Buscar el movimiento que minimiza el objetivo penalizado.
        mejor_t: str | None = None
        mejor_sol_paso: list[list[str]] | None = None
        mejor_obj_paso: float = float("inf")
        for tarea, (r_dst, pos_dst) in list(delta.items()):
            candidata = _mover_tarea_labels(
                sol_actual, tarea, r_dst, pos_dst, marcador_depot
            )
            if candidata is None:
                # La tarea ya no esta en sol_actual (no deberia pasar, pero
                # somos defensivos). La quitamos del delta para no atorarnos.
                delta.pop(tarea, None)
                continue
            obj_c, _, _ = _objetivo_penalizado_labels(candidata, ctx, lam)
            if obj_c < mejor_obj_paso:
                mejor_obj_paso = obj_c
                mejor_sol_paso = candidata
                mejor_t = tarea
        if mejor_t is None or mejor_sol_paso is None:
            break
        # Avanzar al siguiente estado (greedy hacia la guia).
        sol_actual = mejor_sol_paso
        obj_actual = mejor_obj_paso
        delta.pop(mejor_t, None)
        # Actualizar el mejor intermedio si corresponde.
        if obj_actual < obj_mejor - 1e-15:
            obj_mejor = obj_actual
            sol_mejor = [list(r) for r in sol_actual]

    # Truncated PR: devolvemos el MEJOR intermedio, no el final.
    return sol_mejor


def path_relinking_ids(
    sol_inicio: list[list[int]],
    sol_guia: list[list[int]],
    ctx: Any,
    lam: float,
) -> list[list[int]]:
    """Path Relinking truncado para soluciones en formato IDs (ABC Simple)."""
    mapa_ini = _construir_mapa_pos_ids(sol_inicio)
    mapa_gui = _construir_mapa_pos_ids(sol_guia)

    delta: dict[int, tuple[int, int]] = {}
    for t, pos_g in mapa_gui.items():
        pos_i = mapa_ini.get(t)
        if pos_i != pos_g:
            delta[t] = pos_g

    if not delta:
        return [list(r) for r in sol_inicio]

    max_pasos = min(len(delta), MAX_PASOS_PR_FACTOR)
    sol_actual = [list(r) for r in sol_inicio]
    obj_actual, _, _ = _objetivo_penalizado_ids(sol_actual, ctx, lam)
    sol_mejor = [list(r) for r in sol_actual]
    obj_mejor = obj_actual

    for _ in range(max_pasos):
        if not delta:
            break
        mejor_t: int | None = None
        mejor_sol_paso: list[list[int]] | None = None
        mejor_obj_paso: float = float("inf")
        for tarea, (r_dst, pos_dst) in list(delta.items()):
            candidata = _mover_tarea_ids(sol_actual, tarea, r_dst, pos_dst)
            if candidata is None:
                delta.pop(tarea, None)
                continue
            obj_c, _, _ = _objetivo_penalizado_ids(candidata, ctx, lam)
            if obj_c < mejor_obj_paso:
                mejor_obj_paso = obj_c
                mejor_sol_paso = candidata
                mejor_t = tarea
        if mejor_t is None or mejor_sol_paso is None:
            break
        sol_actual = mejor_sol_paso
        obj_actual = mejor_obj_paso
        delta.pop(mejor_t, None)
        if obj_actual < obj_mejor - 1e-15:
            obj_mejor = obj_actual
            sol_mejor = [list(r) for r in sol_actual]

    return sol_mejor


# ============================================================
# Wrappers de kick aumentados con PR
# ============================================================

def aplicar_kick_con_pr_labels(
    sol: list[list[str]],
    rng: random.Random,
    md_op: str,
    encoding: Any = None,
) -> list[list[str]]:
    """Kick aumentado: con prob ``p_pr`` lanza PR; con prob ``1-p_pr`` kick puro.

    Firma identica a ``aplicar_kick_labels`` original para sustitucion 1-a-1.
    """
    # Paso 1: ejecutar el kick canonico (perturbacion inter-ruta).
    sol_kicked = _kick_labels_orig(sol, rng, md_op, encoding=encoding)
    # Paso 2: decidir si lanzamos PR encima.
    if rng.random() >= _p_pr_actual:
        # Camino kick puro: comportamiento canonico de strict_intra_inter.
        return sol_kicked
    guia = _get_mejor_sol_labels()
    if guia is None:
        # Aun no hay mejor global capturado: no podemos guiar PR. Fallback.
        return sol_kicked
    # Paso 3: extraer ctx y lambda del frame del llamante (la MH).
    # profundidad=2 porque estamos a 2 frames por debajo del frame de la MH
    # (this function -> _extraer_ctx_y_lambda).
    ctx, lam = _extraer_ctx_y_lambda(profundidad_inicial=2)
    if ctx is None or lam is None:
        # No pudimos extraer ctx: PR no es seguro. Devolvemos solo el kick.
        return sol_kicked
    # Paso 4: ejecutar PR truncado de sol_kicked -> guia.
    try:
        return path_relinking_labels(
            sol_kicked, guia, ctx, lam, marcador_depot=md_op
        )
    except Exception:  # noqa: BLE001
        # PR es OPCIONAL: cualquier error degrada al kick puro sin
        # romper la corrida de la MH.
        return sol_kicked


def aplicar_kick_con_pr_ids(
    sol_ids: list[list[int]],
    rng: random.Random,
    encoding: Any = None,
) -> list[list[int]]:
    """Version IDs del kick aumentado (para ABC Simple).

    Firma identica a ``aplicar_kick_ids`` original.
    """
    sol_kicked = _kick_ids_orig(sol_ids, rng, encoding=encoding)
    if rng.random() >= _p_pr_actual:
        return sol_kicked
    # Para ABC: actualizamos la guia desde el frame del llamante (la unica
    # forma de capturar mejor_sol_ids en ABC sin tocar el archivo).
    guia_actual = _extraer_mejor_sol_ids_del_caller()
    if guia_actual is not None:
        _set_mejor_sol_ids(guia_actual)
    guia = _get_mejor_sol_ids()
    if guia is None:
        return sol_kicked
    ctx, lam = _extraer_ctx_y_lambda(profundidad_inicial=2)
    if ctx is None or lam is None:
        return sol_kicked
    try:
        return path_relinking_ids(sol_kicked, guia, ctx, lam)
    except Exception:  # noqa: BLE001
        return sol_kicked


# ============================================================
# Patches sobre helpers de copia (captura de mejor_sol)
# ============================================================

def _make_copiar_labels_patched(orig):
    """Envuelve ``copiar_solucion_labels`` para registrar la ultima copia.

    Cada vez que una MH copia una solucion en labels (tipicamente para
    asignarla al mejor global tras detectar una mejora), guardamos esa
    copia en ``_estado_pr_local.mejor_sol_labels``. PR la usara como guia.
    """
    def _patched(sol):
        salida = orig(sol)
        # Solo guardamos copias NO triviales (rutas no vacias). Las copias
        # finales de cierre tambien pasan por aqui pero ocurren DESPUES del
        # bucle principal, asi que nunca se observan dentro de un kick.
        try:
            if salida and isinstance(salida, list):
                _set_mejor_sol_labels([list(r) for r in salida])
        except Exception:  # noqa: BLE001
            # Nunca dejamos que el patch rompa el flujo de la MH.
            pass
        return salida
    return _patched


def _make_registrar_mejora_pr_patched(orig_method):
    """Patch para ``ContadorOperadores.registrar_mejora`` (para ABC Simple).

    Cuando se invoca, leemos ``mejor_sol_ids`` del frame del llamante para
    mantener actualizada la guia de PR para ABC. Para las demas MH (que
    operan en labels), este patch no hace dano: simplemente registra una
    posible guia en IDs que no se usara en esa MH.
    """
    def _patched(self, op):
        # Primero ejecutamos la logica original (sin interferir).
        orig_method(self, op)
        # Luego intentamos capturar la guia desde el llamante.
        try:
            msi = _extraer_mejor_sol_ids_del_caller()
            if msi is not None:
                _set_mejor_sol_ids(msi)
        except Exception:  # noqa: BLE001
            pass
    return _patched


# ============================================================
# Funcion de patch principal
# ============================================================

def aplicar_patch_pr(
    nombre_modulo_mh: str,
    p_pr: float = P_PR_DEFAULT,
) -> None:
    """Activa Path Relinking sobre el experimento strict_intra_inter.

    Pasos:
      1. Setear la probabilidad efectiva de PR (``_p_pr_actual``) y
         reiniciar la guia (``_estado_pr_local.mejor_sol_*``).
      2. Patchear ``copiar_solucion_labels`` en ``metacarp.metaheuristicas_utils``
         para capturar la mejor solucion en labels (SA, TS, RTS, Cuckoo).
         Re-bindear tambien el atributo en el modulo de la MH si ya lo
         importo (es lo que hacen las 4 MH que usan labels).
      3. Patchear ``ContadorOperadores.registrar_mejora`` para capturar la
         mejor solucion en IDs (ABC Simple).
      4. Patchear ``aplicar_kick_labels`` y ``aplicar_kick_ids`` en
         ``metacarp.strict_intra_inter_20260524`` por sus versiones
         aumentadas con PR. Como las MH hacen ``from metacarp.strict_intra_inter_20260524
         import aplicar_kick_*`` DIFERIDO dentro del bucle, basta con
         reasignar el atributo en strict_intra_inter para que las MH
         reciban automaticamente la version aumentada.

    Parametros
    ----------
    nombre_modulo_mh : str
        Modulo de la MH (necesario solo para re-bindear ``copiar_solucion_labels``
        si ya estaba importado en su namespace).
    p_pr : float
        Probabilidad de disparar PR cuando se ejecuta un kick. El usuario
        eligio 0.5 para este experimento.
    """
    global _p_pr_actual, _copiar_labels_patched, _registrar_mejora_patched
    global _kicks_patched

    # 1) Reset de estado y configuracion de p_pr.
    _p_pr_actual = float(p_pr)
    _set_mejor_sol_labels(None)
    _set_mejor_sol_ids(None)

    # 2) Patch de copiar_solucion_labels (captura mejor_sol en labels).
    if not _copiar_labels_patched:
        import metacarp.metaheuristicas_utils as mhu
        orig_copiar = mhu.copiar_solucion_labels
        mhu.copiar_solucion_labels = _make_copiar_labels_patched(orig_copiar)
        _copiar_labels_patched = True
    # Re-bindear en el modulo de la MH si ya tenia el simbolo importado.
    # Las MH SA/TS/RTS/Cuckoo importan ``copiar_solucion_labels`` en su
    # top-level (``from .metaheuristicas_utils import copiar_solucion_labels``),
    # asi que el atributo del modulo MH apunta al original. Lo actualizamos
    # para que tambien apunte al wrapper.
    try:
        mh = importlib.import_module(nombre_modulo_mh)
        import metacarp.metaheuristicas_utils as mhu_ref
        if hasattr(mh, "copiar_solucion_labels"):
            mh.copiar_solucion_labels = mhu_ref.copiar_solucion_labels
    except Exception:  # noqa: BLE001
        # No abortamos: el patch principal sobre el modulo helper igual
        # captura cualquier copia hecha via la importacion canonica.
        pass

    # 3) Patch de registrar_mejora (captura mejor_sol en IDs para ABC Simple).
    if not _registrar_mejora_patched:
        import metacarp.metaheuristicas_utils as mhu
        orig_rm = mhu.ContadorOperadores.registrar_mejora
        mhu.ContadorOperadores.registrar_mejora = (
            _make_registrar_mejora_pr_patched(orig_rm)
        )
        _registrar_mejora_patched = True

    # 4) Patch de aplicar_kick_labels / aplicar_kick_ids en strict_intra_inter.
    # Como las MH hacen el import DIFERIDO dentro del bucle principal, basta
    # con reasignar el atributo del modulo: las proximas invocaciones del
    # ``from metacarp.strict_intra_inter_20260524 import aplicar_kick_*``
    # recogeran nuestra version aumentada automaticamente.
    if not _kicks_patched:
        import metacarp.strict_intra_inter_20260524 as strict
        strict.aplicar_kick_labels = aplicar_kick_con_pr_labels
        strict.aplicar_kick_ids = aplicar_kick_con_pr_ids
        _kicks_patched = True


__all__ = [
    "P_PR_DEFAULT",
    "MAX_PASOS_PR_FACTOR",
    "OPERADORES_INTER_STRICT",
    "aplicar_kick_con_pr_labels",
    "aplicar_kick_con_pr_ids",
    "aplicar_patch_pr",
    "path_relinking_labels",
    "path_relinking_ids",
]
