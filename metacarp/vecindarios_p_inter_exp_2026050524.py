"""
Variante experimental ``p_inter_exp_2026050524`` de los generadores de vecinos.

Hipotesis a probar: anclar la **primera tarea servida** de cada ruta
(la inmediatamente posterior al deposito) acelera la convergencia porque
mantiene estable la "boca" de la ruta y solo deja que la metaheuristica
reordene/migre las tareas restantes.

Este modulo expone dos funciones simetricas a las originales en
``metacarp.vecindarios``:

* ``generar_vecino_exp``     (backend labels) → reemplazo de ``generar_vecino``.
* ``generar_vecino_ids_exp`` (backend ids)    → reemplazo de ``generar_vecino_ids``.

Ambas mantienen la misma firma, los mismos nombres de operadores y los mismos
contratos que las originales. La unica diferencia operativa: al seleccionar
posiciones para aplicar los 9 operadores, NO se permiten los indices que
afectarian la posicion 0 (la primera tarea servida tras el deposito, despues
de normalizar la solucion eliminando los marcadores "D").

Reusamos las funciones-operador originales (``op_relocate_intra``, etc.)
porque la "restriccion posicional" se aplica enteramente en el dispatcher
(seleccion de indices), no en el operador en si.

No se introducen tuplas nuevas de operadores: se reusa
``OPERADORES_POPULARES`` con los mismos 9 nombres.
"""
from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from typing import Hashable, Literal

# ------------------------------------------------------------
# Reusamos toda la infraestructura del modulo original.
# Las 9 funciones-operador se invocan tal cual desde el dispatcher:
# la restriccion posicional vive en como se ELIGEN los indices,
# no en las funciones que realizan la transformacion.
# ------------------------------------------------------------
from .vecindarios import (
    MovimientoVecindario,
    OPERADORES_POPULARES,
    _aplicar_backend_gpu_placeholder,
    _moved_ids,
    _rutas_con_indices,
    decode_solution,
    decode_task_ids,
    desnormalizar_con_deposito,
    encode_solution,
    normalizar_para_vecindario,
    op_2opt_intra,
    op_cross_exchange,
    op_or_opt_2,
    op_or_opt_3,
    op_relocate_inter,
    op_relocate_intra,
    op_swap_inter,
    op_swap_intra,
    op_two_opt_star,
)
from .busqueda_indices import SearchEncoding

__all__ = [
    "generar_vecino_exp",
    "generar_vecino_ids_exp",
    # Constantes de configuracion utiles si alguien quiere parametrizar el
    # indice minimo (por ahora hardcoded a 1, pero documentado).
    "POS_MIN_PROTEGIDA",
]


# ============================================================
# Constante de configuracion
# ============================================================
# Indice MINIMO desde el cual los operadores pueden actuar dentro de cada
# ruta (post-normalizacion sin marcador de deposito). Con POS_MIN_PROTEGIDA=1
# la posicion 0 (1a tarea tras el deposito) queda intacta durante toda la
# busqueda. Si en el futuro se quisiera anclar mas tareas (p.ej. las 2
# primeras), basta con subir esta constante.
POS_MIN_PROTEGIDA = 1


# ============================================================
# generar_vecino_ids_exp
# ============================================================
# Replica fiel del dispatcher ``generar_vecino_ids`` de ``vecindarios.py``
# (lineas 740-947) salvo por las llamadas a ``rng.randrange(...)`` que
# ahora arrancan en POS_MIN_PROTEGIDA en lugar de 0. Cada operador necesita
# rutas "suficientemente largas" para que la restriccion deje al menos
# UNA posicion valida (por ejemplo, relocate_intra necesita len(ruta) >= 2
# en el original; con restriccion necesita len(ruta) >= 3 para que existan
# dos indices distintos en [1, n)).
def generar_vecino_ids_exp(
    solucion_ids: Sequence[Sequence[int]],
    *,
    rng: random.Random | None = None,
    operadores: Iterable[str] = OPERADORES_POPULARES,
    pesos_operadores: Sequence[float] | None = None,
    usar_gpu: bool = False,
    encoding: SearchEncoding | None = None,
) -> tuple[list[list[int]], MovimientoVecindario]:
    """
    Variante de ``generar_vecino_ids`` que ancla la posicion 0 de cada ruta.

    Misma firma y mismo contrato de retorno que el original; lo unico que
    cambia es la seleccion aleatoria de indices, que respeta el invariante:

        indice_aplicado >= POS_MIN_PROTEGIDA (== 1)

    para todas las posiciones que afectan la 1a tarea de cualquier ruta.
    """
    rng = rng or random.Random()
    p = POS_MIN_PROTEGIDA  # alias corto para legibilidad dentro del dispatcher

    rutas = [[int(x) for x in r] for r in solucion_ids]
    ops = list(operadores)
    if not ops:
        raise ValueError("operadores está vacío.")

    backend_solicitado, backend_real = _aplicar_backend_gpu_placeholder(usar_gpu)

    intentos = 0
    while True:
        intentos += 1
        if intentos > 500:
            raise RuntimeError(
                "No se pudo generar un vecino (variante exp): solución demasiado "
                "pequeña para los operadores con la restricción posicional activa."
            )

        op = (
            rng.choices(ops, weights=pesos_operadores, k=1)[0]
            if pesos_operadores is not None
            else rng.choice(ops)
        )
        activos = _rutas_con_indices(rutas)
        if not activos:
            continue

        # ---- OPERADORES INTRA ----

        if op == "relocate_intra":
            # Original: i, j ∈ [0, n) con i != j  (necesita n >= 2)
            # Exp:      i, j ∈ [p, n) con i != j  (necesita n >= p+2 = 3)
            cand = [x for x in activos if len(rutas[x]) >= p + 2]
            if not cand:
                continue
            r = rng.choice(cand)
            n = len(rutas[r])
            i = rng.randrange(p, n)
            j = rng.randrange(p, n)
            if i == j:
                continue
            vec = op_relocate_intra(rutas, r, i, j)
            mov = MovimientoVecindario(op, ruta_a=r, i=i, j=j)

        elif op == "swap_intra":
            # Original: i != j ambos en [0, n)  (necesita n >= 2)
            # Exp:      i != j ambos en [p, n)  (necesita n >= p+2 = 3)
            cand = [x for x in activos if len(rutas[x]) >= p + 2]
            if not cand:
                continue
            r = rng.choice(cand)
            n = len(rutas[r])
            # sample(range(p, n), 2) garantiza i != j directamente
            i, j = rng.sample(range(p, n), 2)
            vec = op_swap_intra(rutas, r, i, j)
            mov = MovimientoVecindario(op, ruta_a=r, i=i, j=j)

        elif op == "2opt_intra":
            # Original: i ∈ [0, n-1), j ∈ (i, n)   (necesita n >= 3)
            # Exp:      i ∈ [p, n-1), j ∈ (i, n)   (necesita n >= p+2 = 3,
            # pero ademas i debe poder valer al menos p, lo que requiere n-1 > p
            # → n >= p+2 = 3; ya esta cubierto)
            cand = [x for x in activos if len(rutas[x]) >= p + 2]
            if not cand:
                continue
            r = rng.choice(cand)
            n = len(rutas[r])
            i = rng.randrange(p, n - 1)
            j = rng.randrange(i + 1, n)
            vec = op_2opt_intra(rutas, r, i, j)
            mov = MovimientoVecindario(op, ruta_a=r, i=i, j=j)

        # ---- OPERADORES INTER ----

        elif op == "relocate_inter":
            # Original: i ∈ [0, len(rutas[ra])); j ∈ [0, len(rutas[rb])+1]
            # Exp:      i ∈ [p, len(rutas[ra])); j ∈ [p, len(rutas[rb])+1]
            # La ruta origen necesita len >= p+1 = 2 para que i pueda valer p.
            # La ruta destino necesita len >= p (puede ser p si insertamos al final).
            if len(activos) < 1 or len(rutas) < 2:
                continue
            cand_origen = [x for x in activos if len(rutas[x]) >= p + 1]
            if not cand_origen:
                continue
            ra = rng.choice(cand_origen)
            rb = rng.randrange(len(rutas))
            if ra == rb:
                continue
            # j puede ser cualquier posicion de insercion >= p, hasta len(rutas[rb])
            # (incluido el "uno mas alla" para insertar al final).
            n_rb = len(rutas[rb])
            if n_rb < p:
                # Caso degenerado: la ruta destino tiene menos de p tareas;
                # no podemos insertar respetando el ancla.
                continue
            i = rng.randrange(p, len(rutas[ra]))
            j = rng.randrange(p, n_rb + 1)
            vec = op_relocate_inter(rutas, ra, i, rb, j)
            mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=i, j=j)

        elif op == "swap_inter":
            # Original: i, j ∈ [0, len_ruta_respectiva)
            # Exp:      i, j ∈ [p, ...)
            # Ambas rutas necesitan len >= p+1 = 2.
            if len(rutas) < 2:
                continue
            non_empty = [x for x in range(len(rutas)) if len(rutas[x]) >= p + 1]
            if len(non_empty) < 2:
                continue
            ra, rb = rng.sample(non_empty, 2)
            i = rng.randrange(p, len(rutas[ra]))
            j = rng.randrange(p, len(rutas[rb]))
            vec = op_swap_inter(rutas, ra, i, rb, j)
            mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=i, j=j)

        elif op == "2opt_star":
            # Original: cut_a ∈ [0, len(ra)), cut_b ∈ [0, len(rb))
            # Exp:      cut_a ∈ [p, len(ra)), cut_b ∈ [p, len(rb))
            # Cortar en la posicion 0 transferiria la 1a tarea, asi que
            # el corte debe quedar despues de la 1a tarea (idx >= p).
            if len(rutas) < 2:
                continue
            non_empty = [x for x in range(len(rutas)) if len(rutas[x]) >= p + 1]
            if len(non_empty) < 2:
                continue
            ra, rb = rng.sample(non_empty, 2)
            cut_a = rng.randrange(p, len(rutas[ra]))
            cut_b = rng.randrange(p, len(rutas[rb]))
            vec = op_two_opt_star(rutas, ra, cut_a, rb, cut_b)
            mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=cut_a, j=cut_b)

        elif op == "cross_exchange":
            # Original: i ∈ [0, na-1), j ∈ (i, na); idem para k, l en B.
            # Exp:      i ∈ [p, na-1), j ∈ (i, na); idem para k, l (≥ p) en B.
            # Cada ruta necesita len >= p+2 = 3 para que exista un segmento
            # [i, j] de longitud ≥ 2 con i ≥ p.
            if len(rutas) < 2:
                continue
            non_empty = [x for x in range(len(rutas)) if len(rutas[x]) >= p + 2]
            if len(non_empty) < 2:
                continue
            ra, rb = rng.sample(non_empty, 2)
            na, nb = len(rutas[ra]), len(rutas[rb])
            i = rng.randrange(p, na - 1)
            j = rng.randrange(i + 1, na)
            k = rng.randrange(p, nb - 1)
            l = rng.randrange(k + 1, nb)
            vec = op_cross_exchange(rutas, ra, i, j, rb, k, l)
            mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=i, j=j, k=k, l=l)

        elif op in ("or_opt_2", "or_opt_3"):
            # Original: i ∈ [0, na - k_blk + 1); j ∈ [0, len(rb)+1] o
            #           [0, tam_destino_post+1] si rb == ra.
            # Exp:      i ∈ [p, na - k_blk + 1); j ∈ [p, ...].
            # La ruta origen necesita len >= p + k_blk para que pueda existir
            # un bloque de k_blk tareas con inicio >= p.
            k_blk = 2 if op == "or_opt_2" else 3
            cand_origen = [x for x in activos if len(rutas[x]) >= p + k_blk]
            if not cand_origen:
                continue
            ra = rng.choice(cand_origen)
            destinos = [x for x in range(len(rutas)) if x != ra]
            if not destinos:
                destinos = [ra]
            rb = rng.choice(destinos)
            na = len(rutas[ra])
            # i: inicio del bloque dentro de la ruta origen (debe respetar el ancla)
            i = rng.randrange(p, na - k_blk + 1)
            if rb == ra:
                # Tras eliminar el bloque, en la misma ruta quedaran (na - k_blk)
                # tareas. La 1a tarea original sigue en la posicion 0 (porque
                # i >= p = 1), asi que la insercion tambien debe ser >= p.
                tam_destino_post = na - k_blk
                if tam_destino_post <= p:
                    # No hay sitio para reinsertar respetando el ancla.
                    continue
                j = rng.randrange(p, tam_destino_post + 1)
                if j == i:
                    continue
            else:
                n_rb = len(rutas[rb])
                if n_rb < p:
                    continue
                j = rng.randrange(p, n_rb + 1)
            if k_blk == 2:
                vec = op_or_opt_2(rutas, ra, i, rb, j)
            else:
                vec = op_or_opt_3(rutas, ra, i, rb, j)
            mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=i, j=j, k=k_blk)

        else:
            raise ValueError(f"Operador desconocido: {op!r}")

        # ---- Enriquecer el MovimientoVecindario con IDs y labels ----
        # Identica al original: aprovechamos ``_moved_ids`` para extraer los
        # IDs de las tareas que se desplazaron y, si hay encoding disponible,
        # decodificamos a etiquetas humanas (TR1, TR2, ...).
        ids_m = _moved_ids(op, rutas, mov)
        labels_m = (
            tuple(decode_task_ids(ids_m, encoding))
            if encoding is not None
            else ()
        )

        mov_out = MovimientoVecindario(
            operador=mov.operador,
            ruta_a=mov.ruta_a,
            ruta_b=mov.ruta_b,
            i=mov.i,
            j=mov.j,
            k=mov.k,
            l=mov.l,
            id_movidos=ids_m,
            labels_movidos=labels_m,
            backend_solicitado=backend_solicitado,
            backend_real=backend_real,
        )
        return [[int(x) for x in r] for r in vec], mov_out


# ============================================================
# generar_vecino_exp
# ============================================================
# Replica fiel del dispatcher ``generar_vecino`` (labels) de
# ``vecindarios.py`` (lineas 956-1175) con la misma restriccion posicional.
# Recordamos que en este modo de operacion la solucion llega CON el marcador
# de deposito "D" en las posiciones 0 y -1 de cada ruta; el dispatcher la
# normaliza eliminando "D" antes de operar, asi que tras la normalizacion la
# "1a tarea servida tras el deposito" queda en la posicion 0 del array
# (equivalente al caso de los ids).
def generar_vecino_exp(
    solucion: Sequence[Sequence[Hashable]],
    *,
    rng: random.Random | None = None,
    operadores: Iterable[str] = OPERADORES_POPULARES,
    pesos_operadores: Sequence[float] | None = None,
    marcador_depot: str = "D",
    devolver_con_deposito: bool = True,
    usar_gpu: bool = False,
    backend: Literal["labels", "ids"] = "labels",
    encoding: SearchEncoding | None = None,
) -> tuple[list[list[str]], MovimientoVecindario]:
    """
    Variante de ``generar_vecino`` que ancla la posicion 0 (post-deposito).

    Comportamiento equivalente al original; misma firma, mismos contratos.
    Cuando ``backend='ids'`` se delega a ``generar_vecino_ids_exp``.
    """
    # ---- Backend ids: delegacion directa con codificacion ----
    if backend == "ids":
        if encoding is None:
            raise ValueError(
                "backend='ids' requiere un SearchEncoding en el parámetro 'encoding'."
            )
        rutas_ids = encode_solution(solucion, encoding)
        vecino_ids, mov = generar_vecino_ids_exp(
            rutas_ids,
            rng=rng,
            operadores=operadores,
            pesos_operadores=pesos_operadores,
            usar_gpu=usar_gpu,
            encoding=encoding,
        )
        if devolver_con_deposito:
            return decode_solution(vecino_ids, encoding, con_deposito=True), mov
        return decode_solution(vecino_ids, encoding, con_deposito=False), mov

    # ---- Backend labels: operamos directamente con strings ----
    backend_solicitado, backend_real = _aplicar_backend_gpu_placeholder(usar_gpu)
    rng = rng or random.Random()
    p = POS_MIN_PROTEGIDA  # alias corto

    # Quita "D" para razonar con posiciones reales de tareas
    rutas = normalizar_para_vecindario(solucion, marcador_depot=marcador_depot)
    ops = list(operadores)
    if not ops:
        raise ValueError("operadores está vacío.")

    intentos = 0
    while True:
        intentos += 1
        if intentos > 500:
            raise RuntimeError(
                "No se pudo generar un vecino (variante exp): solución demasiado "
                "pequeña para los operadores con la restricción posicional activa."
            )

        op = (
            rng.choices(ops, weights=pesos_operadores, k=1)[0]
            if pesos_operadores is not None
            else rng.choice(ops)
        )
        activos = _rutas_con_indices(rutas)
        if not activos:
            continue

        # ---- Operadores intra ----
        if op == "relocate_intra":
            cand = [x for x in activos if len(rutas[x]) >= p + 2]
            if not cand:
                continue
            r = rng.choice(cand)
            n = len(rutas[r])
            i = rng.randrange(p, n)
            j = rng.randrange(p, n)
            if i == j:
                continue
            vec = op_relocate_intra(rutas, r, i, j)
            mov = MovimientoVecindario(op, ruta_a=r, i=i, j=j)

        elif op == "swap_intra":
            cand = [x for x in activos if len(rutas[x]) >= p + 2]
            if not cand:
                continue
            r = rng.choice(cand)
            n = len(rutas[r])
            i, j = rng.sample(range(p, n), 2)
            vec = op_swap_intra(rutas, r, i, j)
            mov = MovimientoVecindario(op, ruta_a=r, i=i, j=j)

        elif op == "2opt_intra":
            cand = [x for x in activos if len(rutas[x]) >= p + 2]
            if not cand:
                continue
            r = rng.choice(cand)
            n = len(rutas[r])
            i = rng.randrange(p, n - 1)
            j = rng.randrange(i + 1, n)
            if j - i < 1:
                continue
            vec = op_2opt_intra(rutas, r, i, j)
            mov = MovimientoVecindario(op, ruta_a=r, i=i, j=j)

        # ---- Operadores inter ----
        elif op == "relocate_inter":
            if len(activos) < 1 or len(rutas) < 2:
                continue
            cand_origen = [x for x in activos if len(rutas[x]) >= p + 1]
            if not cand_origen:
                continue
            ra = rng.choice(cand_origen)
            rb = rng.randrange(len(rutas))
            if ra == rb and len(rutas) < 2:
                continue
            n_rb = len(rutas[rb])
            if n_rb < p:
                continue
            i = rng.randrange(p, len(rutas[ra]))
            j = rng.randrange(p, n_rb + 1)
            if ra == rb:
                continue  # garantiza que sea inter y no intra
            vec = op_relocate_inter(rutas, ra, i, rb, j)
            mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=i, j=j)

        elif op == "swap_inter":
            if len(rutas) < 2:
                continue
            non_empty = [x for x in range(len(rutas)) if len(rutas[x]) >= p + 1]
            if len(non_empty) < 2:
                continue
            ra, rb = rng.sample(non_empty, 2)
            i = rng.randrange(p, len(rutas[ra]))
            j = rng.randrange(p, len(rutas[rb]))
            vec = op_swap_inter(rutas, ra, i, rb, j)
            mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=i, j=j)

        elif op == "2opt_star":
            if len(rutas) < 2:
                continue
            non_empty = [x for x in range(len(rutas)) if len(rutas[x]) >= p + 1]
            if len(non_empty) < 2:
                continue
            ra, rb = rng.sample(non_empty, 2)
            cut_a = rng.randrange(p, len(rutas[ra]))
            cut_b = rng.randrange(p, len(rutas[rb]))
            vec = op_two_opt_star(rutas, ra, cut_a, rb, cut_b)
            mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=cut_a, j=cut_b)

        elif op == "cross_exchange":
            if len(rutas) < 2:
                continue
            non_empty = [x for x in range(len(rutas)) if len(rutas[x]) >= p + 2]
            if len(non_empty) < 2:
                continue
            ra, rb = rng.sample(non_empty, 2)
            na, nb = len(rutas[ra]), len(rutas[rb])
            i = rng.randrange(p, na - 1)
            j = rng.randrange(i + 1, na)
            k = rng.randrange(p, nb - 1)
            l = rng.randrange(k + 1, nb)
            vec = op_cross_exchange(rutas, ra, i, j, rb, k, l)
            mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=i, j=j, k=k, l=l)

        elif op in ("or_opt_2", "or_opt_3"):
            k_blk = 2 if op == "or_opt_2" else 3
            cand_origen = [x for x in activos if len(rutas[x]) >= p + k_blk]
            if not cand_origen:
                continue
            ra = rng.choice(cand_origen)
            destinos = [x for x in range(len(rutas)) if x != ra]
            if not destinos:
                destinos = [ra]
            rb = rng.choice(destinos)
            na = len(rutas[ra])
            i = rng.randrange(p, na - k_blk + 1)
            if rb == ra:
                tam_destino_post = na - k_blk
                if tam_destino_post <= p:
                    continue
                j = rng.randrange(p, tam_destino_post + 1)
                if j == i:
                    continue
            else:
                n_rb = len(rutas[rb])
                if n_rb < p:
                    continue
                j = rng.randrange(p, n_rb + 1)
            if k_blk == 2:
                vec = op_or_opt_2(rutas, ra, i, rb, j)
            else:
                vec = op_or_opt_3(rutas, ra, i, rb, j)
            mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=i, j=j, k=k_blk)

        else:
            raise ValueError(f"Operador desconocido: {op!r}")

        # Reconstruye el movimiento con la informacion del backend
        # (en modo labels no rellenamos ids_movidos/labels_movidos: la MH
        # los recibe vacios igual que con el dispatcher original).
        mov = MovimientoVecindario(
            operador=mov.operador,
            ruta_a=mov.ruta_a,
            ruta_b=mov.ruta_b,
            i=mov.i,
            j=mov.j,
            k=mov.k,
            l=mov.l,
            backend_solicitado=backend_solicitado,
            backend_real=backend_real,
        )

        if devolver_con_deposito:
            return desnormalizar_con_deposito(vec, marcador_depot=marcador_depot), mov
        return vec, mov
