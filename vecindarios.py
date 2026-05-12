from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Hashable, Iterable, Literal, Sequence

from .busqueda_indices import SearchEncoding, decode_solution, decode_task_ids, encode_solution

__all__ = [
    "MovimientoVecindario",
    "normalizar_para_vecindario",
    "desnormalizar_con_deposito",
    "op_relocate_intra",
    "op_swap_intra",
    "op_2opt_intra",
    "op_relocate_inter",
    "op_swap_inter",
    "op_two_opt_star",
    "op_cross_exchange",
    "OPERADORES_POPULARES",
    "generar_vecino_ids",
    "generar_vecino",
]


@dataclass(frozen=True, slots=True)
class MovimientoVecindario:
    """Describe el operador aplicado y los cortes/índices usados (si aplica)."""

    operador: str
    ruta_a: int | None = None
    ruta_b: int | None = None
    i: int | None = None
    j: int | None = None
    k: int | None = None
    l: int | None = None
    id_movidos: tuple[int, ...] = ()
    labels_movidos: tuple[str, ...] = ()
    backend_solicitado: str = "labels"
    backend_real: str = "cpu"


def _is_depot_token(x: Hashable, marcador_depot: str) -> bool:
    return str(x).strip().upper() == str(marcador_depot).strip().upper()


def normalizar_para_vecindario(
    solucion: Sequence[Sequence[Hashable]],
    *,
    marcador_depot: str = "D",
) -> list[list[str]]:
    """
    Devuelve rutas sólo con etiquetas (sin el marcador de depósito).

    Esto NO valida contra una instancia; únicamente elimina tokens iguales a ``marcador_depot``.
    """
    out: list[list[str]] = []
    for ruta in solucion:
        fila = [str(x).strip() for x in ruta if str(x).strip() and not _is_depot_token(x, marcador_depot)]
        out.append(fila)
    return out


def desnormalizar_con_deposito(
    rutas: Sequence[Sequence[Hashable]],
    *,
    marcador_depot: str = "D",
) -> list[list[str]]:
    """Agrega ``[D, ..., D]`` a cada ruta (incluye también rutas vacías)."""
    md = str(marcador_depot).strip().upper() or "D"
    return [[md, *[str(x).strip() for x in r], md] for r in rutas]


def _copy_solution(sol: Sequence[Sequence[Hashable]]) -> list[list[str]]:
    return [[x for x in r] for r in sol]  # type: ignore[list-item]


def op_relocate_intra(sol: Sequence[Sequence[Hashable]], r: int, i: int, j: int) -> list[list[str]]:
    """
    Relocate dentro de una ruta: mueve la tarea en posición i a la posición j.
    Requiere len(ruta) >= 2.
    """
    s = _copy_solution(sol)
    ruta = s[r]
    x = ruta.pop(i)
    ruta.insert(j, x)
    return s  # type: ignore[return-value]


def op_swap_intra(sol: Sequence[Sequence[Hashable]], r: int, i: int, j: int) -> list[list[str]]:
    """Swap dentro de una ruta: intercambia posiciones i y j."""
    s = _copy_solution(sol)
    ruta = s[r]
    ruta[i], ruta[j] = ruta[j], ruta[i]
    return s  # type: ignore[return-value]


def op_2opt_intra(sol: Sequence[Sequence[Hashable]], r: int, i: int, j: int) -> list[list[str]]:
    """
    2-opt (intra): revierte el segmento [i:j] (i < j) en la ruta r.
    """
    s = _copy_solution(sol)
    ruta = s[r]
    ruta[i : j + 1] = reversed(ruta[i : j + 1])
    return s  # type: ignore[return-value]


def op_relocate_inter(
    sol: Sequence[Sequence[Hashable]],
    ra: int,
    i: int,
    rb: int,
    j: int,
) -> list[list[str]]:
    """Relocate (inter): mueve una tarea de ruta ra posición i hacia ruta rb posición j."""
    s = _copy_solution(sol)
    x = s[ra].pop(i)
    s[rb].insert(j, x)
    return s  # type: ignore[return-value]


def op_swap_inter(
    sol: Sequence[Sequence[Hashable]],
    ra: int,
    i: int,
    rb: int,
    j: int,
) -> list[list[str]]:
    """Swap (inter): intercambia una tarea entre rutas ra y rb."""
    s = _copy_solution(sol)
    s[ra][i], s[rb][j] = s[rb][j], s[ra][i]
    return s  # type: ignore[return-value]


def op_two_opt_star(
    sol: Sequence[Sequence[Hashable]],
    ra: int,
    cut_a: int,
    rb: int,
    cut_b: int,
) -> list[list[str]]:
    """
    2-opt* (inter): intercambia las colas después de los cortes.

    - A = [a0..a_cut] + tailA
    - B = [b0..b_cut] + tailB
    -> A' = [a0..a_cut] + tailB
    -> B' = [b0..b_cut] + tailA
    """
    s = _copy_solution(sol)
    a = s[ra]
    b = s[rb]
    head_a, tail_a = a[: cut_a + 1], a[cut_a + 1 :]
    head_b, tail_b = b[: cut_b + 1], b[cut_b + 1 :]
    s[ra] = head_a + tail_b
    s[rb] = head_b + tail_a
    return s  # type: ignore[return-value]


def op_cross_exchange(
    sol: Sequence[Sequence[Hashable]],
    ra: int,
    i: int,
    j: int,
    rb: int,
    k: int,
    l: int,
) -> list[list[str]]:
    """
    Cross-exchange: intercambia segmentos [i:j] de A con [k:l] de B.
    Índices inclusivos.
    """
    s = _copy_solution(sol)
    a = s[ra]
    b = s[rb]
    seg_a = a[i : j + 1]
    seg_b = b[k : l + 1]
    s[ra] = a[:i] + seg_b + a[j + 1 :]
    s[rb] = b[:k] + seg_a + b[l + 1 :]
    return s  # type: ignore[return-value]


OPERADORES_POPULARES = (
    "relocate_intra",
    "swap_intra",
    "2opt_intra",
    "relocate_inter",
    "swap_inter",
    "2opt_star",
    "cross_exchange",
)


def _rutas_con_indices(rutas: Sequence[Sequence[Hashable]]) -> list[int]:
    return [idx for idx, r in enumerate(rutas) if len(r) > 0]


def _moved_ids(op: str, rutas: Sequence[Sequence[int]], mov: MovimientoVecindario) -> tuple[int, ...]:
    if mov.ruta_a is None:
        return ()
    if op in {"relocate_intra", "swap_intra", "2opt_intra"}:
        r = mov.ruta_a
        i = mov.i if mov.i is not None else 0
        j = mov.j if mov.j is not None else i
        if op == "relocate_intra":
            return (rutas[r][i],)
        return tuple(rutas[r][min(i, j) : max(i, j) + 1])
    if op in {"relocate_inter", "swap_inter"}:
        ids: list[int] = []
        if mov.ruta_a is not None and mov.i is not None:
            ids.append(rutas[mov.ruta_a][mov.i])
        if op == "swap_inter" and mov.ruta_b is not None and mov.j is not None:
            ids.append(rutas[mov.ruta_b][mov.j])
        return tuple(ids)
    if op == "2opt_star":
        ids2: list[int] = []
        if mov.ruta_a is not None and mov.i is not None:
            ids2.extend(rutas[mov.ruta_a][mov.i + 1 :])
        if mov.ruta_b is not None and mov.j is not None:
            ids2.extend(rutas[mov.ruta_b][mov.j + 1 :])
        return tuple(ids2)
    if op == "cross_exchange":
        ids3: list[int] = []
        if mov.ruta_a is not None and mov.i is not None and mov.j is not None:
            ids3.extend(rutas[mov.ruta_a][mov.i : mov.j + 1])
        if mov.ruta_b is not None and mov.k is not None and mov.l is not None:
            ids3.extend(rutas[mov.ruta_b][mov.k : mov.l + 1])
        return tuple(ids3)
    return ()


def _aplicar_backend_gpu_placeholder(usar_gpu: bool) -> tuple[str, str]:
    if not usar_gpu:
        return "cpu", "cpu"
    # Placeholder de backend GPU: aún no hay kernel indexado implementado.
    return "gpu", "cpu"


def generar_vecino_ids(
    solucion_ids: Sequence[Sequence[int]],
    *,
    rng: random.Random | None = None,
    operadores: Iterable[str] = OPERADORES_POPULARES,
    usar_gpu: bool = False,
    encoding: SearchEncoding | None = None,
) -> tuple[list[list[int]], MovimientoVecindario]:
    """
    Genera un vecino sobre representación indexada (IDs enteros).
    """
    rng = rng or random.Random()
    rutas = [[int(x) for x in r] for r in solucion_ids]
    ops = list(operadores)
    if not ops:
        raise ValueError("operadores está vacío.")

    backend_solicitado, backend_real = _aplicar_backend_gpu_placeholder(usar_gpu)

    intentos = 0
    while True:
        intentos += 1
        if intentos > 500:
            raise RuntimeError("No se pudo generar un vecino: solución demasiado pequeña para los operadores.")

        op = rng.choice(ops)
        activos = _rutas_con_indices(rutas)
        if not activos:
            continue

        if op == "relocate_intra":
            cand = [x for x in activos if len(rutas[x]) >= 2]
            if not cand:
                continue
            r = rng.choice(cand)
            n = len(rutas[r])
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            vec = op_relocate_intra(rutas, r, i, j)
            mov = MovimientoVecindario(op, ruta_a=r, i=i, j=j)

        elif op == "swap_intra":
            cand = [x for x in activos if len(rutas[x]) >= 2]
            if not cand:
                continue
            r = rng.choice(cand)
            n = len(rutas[r])
            i, j = rng.sample(range(n), 2)
            vec = op_swap_intra(rutas, r, i, j)
            mov = MovimientoVecindario(op, ruta_a=r, i=i, j=j)

        elif op == "2opt_intra":
            cand = [x for x in activos if len(rutas[x]) >= 3]
            if not cand:
                continue
            r = rng.choice(cand)
            n = len(rutas[r])
            i = rng.randrange(0, n - 1)
            j = rng.randrange(i + 1, n)
            vec = op_2opt_intra(rutas, r, i, j)
            mov = MovimientoVecindario(op, ruta_a=r, i=i, j=j)

        elif op == "relocate_inter":
            if len(activos) < 1 or len(rutas) < 2:
                continue
            ra = rng.choice(activos)
            if not rutas[ra]:
                continue
            rb = rng.randrange(len(rutas))
            if ra == rb:
                continue
            i = rng.randrange(len(rutas[ra]))
            j = rng.randrange(len(rutas[rb]) + 1)
            vec = op_relocate_inter(rutas, ra, i, rb, j)
            mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=i, j=j)

        elif op == "swap_inter":
            if len(rutas) < 2:
                continue
            non_empty = [x for x in range(len(rutas)) if len(rutas[x]) > 0]
            if len(non_empty) < 2:
                continue
            ra, rb = rng.sample(non_empty, 2)
            i = rng.randrange(len(rutas[ra]))
            j = rng.randrange(len(rutas[rb]))
            vec = op_swap_inter(rutas, ra, i, rb, j)
            mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=i, j=j)

        elif op == "2opt_star":
            if len(rutas) < 2:
                continue
            non_empty = [x for x in range(len(rutas)) if len(rutas[x]) > 0]
            if len(non_empty) < 2:
                continue
            ra, rb = rng.sample(non_empty, 2)
            cut_a = rng.randrange(len(rutas[ra]))
            cut_b = rng.randrange(len(rutas[rb]))
            vec = op_two_opt_star(rutas, ra, cut_a, rb, cut_b)
            mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=cut_a, j=cut_b)

        elif op == "cross_exchange":
            if len(rutas) < 2:
                continue
            non_empty = [x for x in range(len(rutas)) if len(rutas[x]) >= 2]
            if len(non_empty) < 2:
                continue
            ra, rb = rng.sample(non_empty, 2)
            na, nb = len(rutas[ra]), len(rutas[rb])
            i = rng.randrange(0, na - 1)
            j = rng.randrange(i + 1, na)
            k = rng.randrange(0, nb - 1)
            l = rng.randrange(k + 1, nb)
            vec = op_cross_exchange(rutas, ra, i, j, rb, k, l)
            mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=i, j=j, k=k, l=l)

        else:
            raise ValueError(f"Operador desconocido: {op!r}")

        ids_m = _moved_ids(op, rutas, mov)
        labels_m = tuple(decode_task_ids(ids_m, encoding)) if encoding is not None else ()
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


def generar_vecino(
    solucion: Sequence[Sequence[Hashable]],
    *,
    rng: random.Random | None = None,
    operadores: Iterable[str] = OPERADORES_POPULARES,
    marcador_depot: str = "D",
    devolver_con_deposito: bool = True,
    usar_gpu: bool = False,
    backend: Literal["labels", "ids"] = "labels",
    encoding: SearchEncoding | None = None,
) -> tuple[list[list[str]], MovimientoVecindario]:
    """
    Genera un vecino aleatorio aplicando un operador de vecindario popular.

    - La entrada puede incluir o no ``D``; internamente se normaliza removiendo ``D``.
    - El vecino preserva el multiconjunto de tareas (solo reordenamientos/transferencias).

    backend:
    - ``labels``: opera directamente con etiquetas (compatibilidad retro).
    - ``ids``: codifica a enteros, aplica movimientos sobre IDs y decodifica.

    Nota sobre GPU: por ahora el backend GPU es placeholder y cae a CPU
    (``backend_real='cpu'``), pero la API queda lista para un kernel futuro.
    """

    if backend == "ids":
        if encoding is None:
            raise ValueError("backend='ids' requiere un SearchEncoding en el parámetro 'encoding'.")
        rutas_ids = encode_solution(solucion, encoding)
        vecino_ids, mov = generar_vecino_ids(
            rutas_ids,
            rng=rng,
            operadores=operadores,
            usar_gpu=usar_gpu,
            encoding=encoding,
        )
        if devolver_con_deposito:
            return decode_solution(vecino_ids, encoding, con_deposito=True), mov
        return decode_solution(vecino_ids, encoding, con_deposito=False), mov

    backend_solicitado, backend_real = _aplicar_backend_gpu_placeholder(usar_gpu)
    rng = rng or random.Random()
    rutas = normalizar_para_vecindario(solucion, marcador_depot=marcador_depot)
    ops = list(operadores)
    if not ops:
        raise ValueError("operadores está vacío.")

    intentos = 0
    while True:
        intentos += 1
        if intentos > 500:
            raise RuntimeError("No se pudo generar un vecino: solución demasiado pequeña para los operadores.")

        op = rng.choice(ops)
        activos = _rutas_con_indices(rutas)
        if not activos:
            continue

        # Intra
        if op == "relocate_intra":
            cand = [x for x in activos if len(rutas[x]) >= 2]
            if not cand:
                continue
            r = rng.choice(cand)
            n = len(rutas[r])
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            vec = op_relocate_intra(rutas, r, i, j)
            mov = MovimientoVecindario(op, ruta_a=r, i=i, j=j)

        elif op == "swap_intra":
            cand = [x for x in activos if len(rutas[x]) >= 2]
            if not cand:
                continue
            r = rng.choice(cand)
            n = len(rutas[r])
            i, j = rng.sample(range(n), 2)
            vec = op_swap_intra(rutas, r, i, j)
            mov = MovimientoVecindario(op, ruta_a=r, i=i, j=j)

        elif op == "2opt_intra":
            cand = [x for x in activos if len(rutas[x]) >= 3]
            if not cand:
                continue
            r = rng.choice(cand)
            n = len(rutas[r])
            i = rng.randrange(0, n - 1)
            j = rng.randrange(i + 1, n)
            if j - i < 1:
                continue
            vec = op_2opt_intra(rutas, r, i, j)
            mov = MovimientoVecindario(op, ruta_a=r, i=i, j=j)

        # Inter
        elif op == "relocate_inter":
            if len(activos) < 1 or len(rutas) < 2:
                continue
            ra = rng.choice(activos)
            if not rutas[ra]:
                continue
            rb = rng.randrange(len(rutas))
            if ra == rb and len(rutas) < 2:
                continue
            i = rng.randrange(len(rutas[ra]))
            j = rng.randrange(len(rutas[rb]) + 1)
            if ra == rb:
                continue
            vec = op_relocate_inter(rutas, ra, i, rb, j)
            mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=i, j=j)

        elif op == "swap_inter":
            if len(rutas) < 2:
                continue
            non_empty = [x for x in range(len(rutas)) if len(rutas[x]) > 0]
            if len(non_empty) < 2:
                continue
            ra, rb = rng.sample(non_empty, 2)
            i = rng.randrange(len(rutas[ra]))
            j = rng.randrange(len(rutas[rb]))
            vec = op_swap_inter(rutas, ra, i, rb, j)
            mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=i, j=j)

        elif op == "2opt_star":
            if len(rutas) < 2:
                continue
            non_empty = [x for x in range(len(rutas)) if len(rutas[x]) > 0]
            if len(non_empty) < 2:
                continue
            ra, rb = rng.sample(non_empty, 2)
            cut_a = rng.randrange(len(rutas[ra]))
            cut_b = rng.randrange(len(rutas[rb]))
            vec = op_two_opt_star(rutas, ra, cut_a, rb, cut_b)
            mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=cut_a, j=cut_b)

        elif op == "cross_exchange":
            if len(rutas) < 2:
                continue
            non_empty = [x for x in range(len(rutas)) if len(rutas[x]) >= 2]
            if len(non_empty) < 2:
                continue
            ra, rb = rng.sample(non_empty, 2)
            na, nb = len(rutas[ra]), len(rutas[rb])
            i = rng.randrange(0, na - 1)
            j = rng.randrange(i + 1, na)
            k = rng.randrange(0, nb - 1)
            l = rng.randrange(k + 1, nb)
            vec = op_cross_exchange(rutas, ra, i, j, rb, k, l)
            mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=i, j=j, k=k, l=l)

        else:
            raise ValueError(f"Operador desconocido: {op!r}")

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

