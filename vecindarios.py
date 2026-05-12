# ============================================================
# vecindarios.py
# ------------------------------------------------------------
# Módulo de OPERADORES DE VECINDARIO para el problema CARP.
#
# Un "vecindario" en optimización combinatoria es el conjunto
# de soluciones que se pueden alcanzar desde la solución actual
# aplicando un pequeño cambio (movimiento). Este módulo define
# los movimientos permitidos y la función que elige y aplica
# uno de ellos al azar para generar un "vecino".
#
# Los operadores se dividen en dos familias:
#   - INTRA: el movimiento ocurre DENTRO de una sola ruta.
#   - INTER: el movimiento mueve tareas ENTRE dos rutas distintas.
# ============================================================

from __future__ import annotations

import random
from dataclasses import dataclass  # Herramienta para crear clases de datos de forma declarativa
from typing import Hashable, Iterable, Literal, Sequence

# SearchEncoding y helpers de codificación entero<->etiqueta
# (definidos en busqueda_indices.py)
from .busqueda_indices import SearchEncoding, decode_solution, decode_task_ids, encode_solution

# __all__ lista los nombres públicos que se exportan cuando alguien hace
# "from vecindarios import *". Sirve como contrato explícito de la API.
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


# ------------------------------------------------------------
# CLASE: MovimientoVecindario
# ------------------------------------------------------------
# OOP utilizado: DATACLASS (clase de datos).
# Un @dataclass genera automáticamente __init__, __repr__ y
# __eq__ a partir de los campos declarados, sin tener que
# escribirlos a mano.
#
# frozen=True  → los objetos son INMUTABLES: una vez creados,
#               sus campos no se pueden cambiar. Esto evita
#               errores accidentales al pasar movimientos entre
#               funciones y permite usarlos como claves de dict.
# slots=True   → Python reserva espacio fijo para los atributos
#               en memoria (más eficiente que un dict interno).
#
# Esta clase describe QUÉ movimiento se aplicó, sobre QUÉ rutas
# y en QUÉ posiciones. Se devuelve junto con la solución vecina
# para que el algoritmo que llama (Tabú, SA, etc.) pueda
# registrar el historial de movimientos.
@dataclass(frozen=True, slots=True)
class MovimientoVecindario:
    """Describe el operador aplicado y los cortes/índices usados (si aplica)."""

    operador: str           # Nombre del operador, p.ej. "relocate_intra"
    ruta_a: int | None = None   # Índice de la primera ruta involucrada
    ruta_b: int | None = None   # Índice de la segunda ruta (solo operadores inter)
    i: int | None = None        # Primer índice de posición dentro de la ruta
    j: int | None = None        # Segundo índice (posición destino o fin de segmento)
    k: int | None = None        # Tercer índice (inicio de segmento en ruta_b, cross-exchange)
    l: int | None = None        # Cuarto índice (fin de segmento en ruta_b, cross-exchange)
    id_movidos: tuple[int, ...] = ()       # IDs enteros de las tareas desplazadas
    labels_movidos: tuple[str, ...] = ()  # Etiquetas (TR...) de las tareas desplazadas
    backend_solicitado: str = "labels"    # Backend pedido por el llamador ("cpu" o "gpu")
    backend_real: str = "cpu"             # Backend que realmente ejecutó el movimiento


# ------------------------------------------------------------
# FUNCIÓN AUXILIAR PRIVADA: _is_depot_token
# ------------------------------------------------------------
# El prefijo _ indica que esta función es PRIVADA: solo se usa
# dentro de este módulo, no forma parte de la API pública.
#
# Compara un valor con el marcador de depósito (p.ej. "D")
# de forma insensible a mayúsculas y sin espacios sobrantes.
def _is_depot_token(x: Hashable, marcador_depot: str) -> bool:
    return str(x).strip().upper() == str(marcador_depot).strip().upper()


# ------------------------------------------------------------
# FUNCIÓN: normalizar_para_vecindario
# ------------------------------------------------------------
# Antes de aplicar cualquier operador, las rutas deben estar
# en formato "limpio": solo etiquetas de tareas (TR1, TR2...),
# SIN el marcador de depósito "D" al inicio y al final.
#
# Ejemplo de entrada:  [["D","TR1","TR5","D"], ["D","TR2","D"]]
# Ejemplo de salida:   [["TR1","TR5"], ["TR2"]]
#
# Parámetros con * en la firma: todo lo que viene después del *
# DEBE pasarse como argumento con nombre (keyword-only), lo que
# hace las llamadas más claras y evita confusión de posiciones.
def normalizar_para_vecindario(
    solucion: Sequence[Sequence[Hashable]],
    *,
    marcador_depot: str = "D",  # Marcador de depósito a eliminar (default "D")
) -> list[list[str]]:
    """
    Devuelve rutas sólo con etiquetas (sin el marcador de depósito).

    Esto NO valida contra una instancia; únicamente elimina tokens iguales a ``marcador_depot``.
    """
    out: list[list[str]] = []  # Lista de rutas normalizadas que se irá llenando
    for ruta in solucion:
        # Filtra: convierte cada elemento a str, elimina espacios, y descarta
        # los que sean vacíos o sean el marcador de depósito.
        fila = [str(x).strip() for x in ruta if str(x).strip() and not _is_depot_token(x, marcador_depot)]
        out.append(fila)
    return out


# ------------------------------------------------------------
# FUNCIÓN: desnormalizar_con_deposito
# ------------------------------------------------------------
# Operación inversa a normalizar_para_vecindario: agrega el
# marcador "D" al inicio y al final de cada ruta para restaurar
# el formato completo que usan los módulos de costo y reporte.
#
# Ejemplo de entrada:  [["TR1","TR5"], ["TR2"]]
# Ejemplo de salida:   [["D","TR1","TR5","D"], ["D","TR2","D"]]
def desnormalizar_con_deposito(
    rutas: Sequence[Sequence[Hashable]],
    *,
    marcador_depot: str = "D",
) -> list[list[str]]:
    """Agrega ``[D, ..., D]`` a cada ruta (incluye también rutas vacías)."""
    md = str(marcador_depot).strip().upper() or "D"  # Asegura que el marcador no sea cadena vacía
    # Comprensión de lista: para cada ruta r, construye ["D", tarea1, tarea2, ..., "D"]
    return [[md, *[str(x).strip() for x in r], md] for r in rutas]


# ------------------------------------------------------------
# FUNCIÓN AUXILIAR PRIVADA: _copy_solution
# ------------------------------------------------------------
# Crea una COPIA PROFUNDA de la solución (lista de listas).
# Es esencial copiar antes de modificar, ya que Python pasa
# listas por referencia: sin copiar, el operador estaría
# modificando la solución original, lo cual causaría bugs
# muy difíciles de detectar.
def _copy_solution(sol: Sequence[Sequence[Hashable]]) -> list[list[str]]:
    return [[x for x in r] for r in sol]  # type: ignore[list-item]
    # Equivalente a: [list(r) for r in sol]
    # Genera una nueva lista por cada ruta (copia independiente).


# ============================================================
# OPERADORES INTRA-RUTA
# Los siguientes tres operadores modifican UNA sola ruta.
# ============================================================

# ------------------------------------------------------------
# OPERADOR: op_relocate_intra  (Reubicación dentro de la ruta)
# ------------------------------------------------------------
# MOVIMIENTO: Saca la tarea que está en la posición i y la
# vuelve a insertar en la posición j de LA MISMA ruta.
#
# Ejemplo visual (ruta de 4 tareas):
#   Antes:  [A, B, C, D]   (i=0 → A, j=2)
#   pop(0):        [B, C, D]   (A sale)
#   insert(2,A):   [B, C, A, D]  (A entra en posición 2)
#
# Por qué es útil: permite probar si servir una tarea en un
# orden diferente reduce el costo de traslado (deadheading).
def op_relocate_intra(sol: Sequence[Sequence[Hashable]], r: int, i: int, j: int) -> list[list[str]]:
    """
    Relocate dentro de una ruta: mueve la tarea en posición i a la posición j.
    Requiere len(ruta) >= 2.
    """
    s = _copy_solution(sol)   # Copia para no alterar la solución original
    ruta = s[r]               # Referencia a la ruta r dentro de la copia
    x = ruta.pop(i)           # Extrae la tarea de la posición i (la lista se acorta)
    ruta.insert(j, x)         # Inserta la tarea extraída en la posición j
    return s  # type: ignore[return-value]


# ------------------------------------------------------------
# OPERADOR: op_swap_intra  (Intercambio dentro de la ruta)
# ------------------------------------------------------------
# MOVIMIENTO: Intercambia las posiciones i y j dentro de la
# misma ruta. Las dos tareas "cambian de lugar" entre sí.
#
# Ejemplo visual:
#   Antes:  [A, B, C, D]   (i=0, j=3)
#   Después:[D, B, C, A]
#
# Por qué es útil: explora si un orden diferente de dos tareas
# reduce el costo de deadheading entre ellas.
def op_swap_intra(sol: Sequence[Sequence[Hashable]], r: int, i: int, j: int) -> list[list[str]]:
    """Swap dentro de una ruta: intercambia posiciones i y j."""
    s = _copy_solution(sol)
    ruta = s[r]
    # Intercambio en una sola línea usando desempaquetado de tuplas de Python:
    # Python primero evalúa el lado derecho completo, luego asigna.
    ruta[i], ruta[j] = ruta[j], ruta[i]
    return s  # type: ignore[return-value]


# ------------------------------------------------------------
# OPERADOR: op_2opt_intra  (2-opt dentro de la ruta)
# ------------------------------------------------------------
# MOVIMIENTO: Invierte (voltea) el segmento de tareas entre
# las posiciones i y j (inclusive) dentro de la misma ruta.
#
# Ejemplo visual:
#   Antes:  [A, B, C, D, E]   (i=1, j=3)
#   Segmento [B, C, D] → invertido → [D, C, B]
#   Después: [A, D, C, B, E]
#
# Por qué es útil: en rutas sobre grafos, a veces visitar un
# sub-segmento en sentido contrario acorta el recorrido total.
# Es el operador clásico del problema del viajero (TSP).
#
# La notación ruta[i:j+1] es "slicing":
#   - ruta[i:j+1] selecciona los elementos desde índice i
#     hasta j INCLUIDO (j+1 porque el límite superior es exclusivo).
#   - reversed(...) devuelve el segmento en orden inverso.
#   - La asignación ruta[i:j+1] = ... reemplaza ese segmento.
def op_2opt_intra(sol: Sequence[Sequence[Hashable]], r: int, i: int, j: int) -> list[list[str]]:
    """
    2-opt (intra): revierte el segmento [i:j] (i < j) en la ruta r.
    """
    s = _copy_solution(sol)
    ruta = s[r]
    ruta[i : j + 1] = reversed(ruta[i : j + 1])  # Invierte el segmento entre i y j (inclusive)
    return s  # type: ignore[return-value]


# ============================================================
# OPERADORES INTER-RUTA
# Los siguientes operadores mueven tareas ENTRE dos rutas
# distintas (ra = ruta A, rb = ruta B).
# ============================================================

# ------------------------------------------------------------
# OPERADOR: op_relocate_inter  (Reubicación entre rutas)
# ------------------------------------------------------------
# MOVIMIENTO: Saca la tarea en posición i de la ruta ra y la
# inserta en la posición j de la ruta rb (diferente a ra).
#
# Ejemplo visual:
#   Antes:  ruta_A=[A, B, C]   ruta_B=[X, Y]   (i=2, j=1)
#   Saca C de ruta_A:  ruta_A=[A, B]
#   Inserta C en pos 1 de ruta_B: ruta_B=[X, C, Y]
#
# Por qué es útil: equilibra la carga entre rutas y puede
# acercar una tarea al depósito o a otras tareas vecinas.
def op_relocate_inter(
    sol: Sequence[Sequence[Hashable]],
    ra: int,   # Índice de la ruta origen
    i: int,    # Posición de la tarea a mover dentro de la ruta origen
    rb: int,   # Índice de la ruta destino
    j: int,    # Posición de inserción dentro de la ruta destino
) -> list[list[str]]:
    """Relocate (inter): mueve una tarea de ruta ra posición i hacia ruta rb posición j."""
    s = _copy_solution(sol)
    x = s[ra].pop(i)      # Extrae la tarea de la ruta origen
    s[rb].insert(j, x)   # La inserta en la ruta destino
    return s  # type: ignore[return-value]


# ------------------------------------------------------------
# OPERADOR: op_swap_inter  (Intercambio entre rutas)
# ------------------------------------------------------------
# MOVIMIENTO: Intercambia la tarea en posición i de la ruta ra
# con la tarea en posición j de la ruta rb.
#
# Ejemplo visual:
#   Antes:  ruta_A=[A, B]   ruta_B=[X, Y]   (i=0, j=1)
#   Después: ruta_A=[X, B]   ruta_B=[A, Y]
#
# Por qué es útil: redistribuye tareas entre rutas sin cambiar
# la cantidad total de tareas en cada una.
def op_swap_inter(
    sol: Sequence[Sequence[Hashable]],
    ra: int,
    i: int,
    rb: int,
    j: int,
) -> list[list[str]]:
    """Swap (inter): intercambia una tarea entre rutas ra y rb."""
    s = _copy_solution(sol)
    # Intercambio simultáneo usando desempaquetado de tuplas (ver op_swap_intra)
    s[ra][i], s[rb][j] = s[rb][j], s[ra][i]
    return s  # type: ignore[return-value]


# ------------------------------------------------------------
# OPERADOR: op_two_opt_star  (2-opt* entre rutas)
# ------------------------------------------------------------
# MOVIMIENTO: "Intercambio de colas". Divide cada ruta en dos
# partes (cabeza y cola) por un punto de corte, y luego
# intercambia las colas entre las dos rutas.
#
# Notación:
#   ruta_A = [a0..a_cut] + tailA   (cabeza_A + cola_A)
#   ruta_B = [b0..b_cut] + tailB   (cabeza_B + cola_B)
#
# Después del movimiento:
#   ruta_A' = cabeza_A + cola_B    (A conserva su inicio, hereda el final de B)
#   ruta_B' = cabeza_B + cola_A   (B conserva su inicio, hereda el final de A)
#
# Ejemplo visual (cut_a=1, cut_b=1):
#   Antes:  A=[P, Q, R, S]   B=[X, Y, Z]
#           head_A=[P, Q]  tail_A=[R, S]
#           head_B=[X, Y]  tail_B=[Z]
#   Después: A'=[P, Q, Z]    B'=[X, Y, R, S]
#
# El slicing a[:cut_a+1] toma desde el inicio hasta cut_a inclusive.
# El slicing a[cut_a+1:] toma desde cut_a+1 hasta el final.
# La suma de listas en Python las concatena: [1,2]+[3,4] = [1,2,3,4].
#
# Por qué es útil: puede "corregir" rutas que se cruzan
# geográficamente, similar al 2-opt clásico pero entre rutas.
def op_two_opt_star(
    sol: Sequence[Sequence[Hashable]],
    ra: int,      # Índice de la ruta A
    cut_a: int,   # Posición del corte en la ruta A (inclusive en la cabeza)
    rb: int,      # Índice de la ruta B
    cut_b: int,   # Posición del corte en la ruta B (inclusive en la cabeza)
) -> list[list[str]]:
    """
    2-opt* (inter): intercambia las colas después de los cortes.

    - A = [a0..a_cut] + tailA
    - B = [b0..b_cut] + tailB
    -> A' = [a0..a_cut] + tailB
    -> B' = [b0..b_cut] + tailA
    """
    s = _copy_solution(sol)
    a = s[ra]   # Referencia a la ruta A dentro de la copia
    b = s[rb]   # Referencia a la ruta B dentro de la copia

    # Dividir cada ruta en cabeza (hasta cut inclusive) y cola (desde cut+1)
    head_a, tail_a = a[: cut_a + 1], a[cut_a + 1 :]
    head_b, tail_b = b[: cut_b + 1], b[cut_b + 1 :]

    # Reconstruir rutas intercambiando colas
    s[ra] = head_a + tail_b   # La ruta A conserva su cabeza y adopta la cola de B
    s[rb] = head_b + tail_a   # La ruta B conserva su cabeza y adopta la cola de A
    return s  # type: ignore[return-value]


# ------------------------------------------------------------
# OPERADOR: op_cross_exchange  (Intercambio de segmentos)
# ------------------------------------------------------------
# MOVIMIENTO: Extrae un SEGMENTO COMPLETO de tareas de cada
# ruta y los intercambia entre ellas.
#
# Notación (índices inclusivos):
#   seg_A = ruta_A[i..j]    (tareas en posiciones i hasta j)
#   seg_B = ruta_B[k..l]    (tareas en posiciones k hasta l)
#
# Después del movimiento:
#   ruta_A' = [inicio_A] + seg_B + [fin_A]
#   ruta_B' = [inicio_B] + seg_A + [fin_B]
#
# Ejemplo visual (i=1, j=2, k=0, l=1):
#   Antes:  A=[P, Q, R, S]   B=[X, Y, Z]
#           seg_A=[Q, R]   seg_B=[X, Y]
#           a[:1]=[P]   a[3:]=[S]   b[:0]=[]   b[2:]=[Z]
#   Después: A'=[P, X, Y, S]   B'=[Q, R, Z]
#
# El slicing a[j+1:] toma desde j+1 hasta el final (el "resto" de la ruta).
#
# Por qué es útil: permite transferir bloques de tareas adyacentes
# entre rutas, generando vecinos más "disruptivos" que un swap simple.
def op_cross_exchange(
    sol: Sequence[Sequence[Hashable]],
    ra: int,   # Índice de la ruta A
    i: int,    # Inicio del segmento en la ruta A (inclusive)
    j: int,    # Fin del segmento en la ruta A (inclusive)
    rb: int,   # Índice de la ruta B
    k: int,    # Inicio del segmento en la ruta B (inclusive)
    l: int,    # Fin del segmento en la ruta B (inclusive)
) -> list[list[str]]:
    """
    Cross-exchange: intercambia segmentos [i:j] de A con [k:l] de B.
    Índices inclusivos.
    """
    s = _copy_solution(sol)
    a = s[ra]   # Referencia a la ruta A dentro de la copia
    b = s[rb]   # Referencia a la ruta B dentro de la copia

    seg_a = a[i : j + 1]   # Extrae el segmento de A (de i hasta j inclusive)
    seg_b = b[k : l + 1]   # Extrae el segmento de B (de k hasta l inclusive)

    # Reconstruye ruta A: parte antes de i + segmento de B + parte después de j
    s[ra] = a[:i] + seg_b + a[j + 1 :]
    # Reconstruye ruta B: parte antes de k + segmento de A + parte después de l
    s[rb] = b[:k] + seg_a + b[l + 1 :]
    return s  # type: ignore[return-value]


# ------------------------------------------------------------
# CONSTANTE: OPERADORES_POPULARES
# ------------------------------------------------------------
# Tupla con los nombres de todos los operadores disponibles.
# Se usa como valor por defecto en generar_vecino y
# generar_vecino_ids para que el algoritmo elija entre todos.
OPERADORES_POPULARES = (
    "relocate_intra",
    "swap_intra",
    "2opt_intra",
    "relocate_inter",
    "swap_inter",
    "2opt_star",
    "cross_exchange",
)


# ------------------------------------------------------------
# FUNCIÓN AUXILIAR PRIVADA: _rutas_con_indices
# ------------------------------------------------------------
# Devuelve los índices de las rutas que tienen al menos una
# tarea (rutas no vacías). Se usa para filtrar rutas vacías
# antes de seleccionar dónde aplicar un operador.
def _rutas_con_indices(rutas: Sequence[Sequence[Hashable]]) -> list[int]:
    # enumerate(rutas) genera pares (índice, ruta)
    # Solo se incluye el índice si la ruta tiene longitud > 0
    return [idx for idx, r in enumerate(rutas) if len(r) > 0]


# ------------------------------------------------------------
# FUNCIÓN AUXILIAR PRIVADA: _moved_ids
# ------------------------------------------------------------
# Dado el operador aplicado y el movimiento registrado,
# extrae los IDs enteros de las tareas que fueron desplazadas.
# Esto permite registrar con precisión qué tareas cambiaron
# de posición (útil para la lista tabú y el historial).
def _moved_ids(op: str, rutas: Sequence[Sequence[int]], mov: MovimientoVecindario) -> tuple[int, ...]:
    # Si no hay ruta_a definida, no hay información de movimiento
    if mov.ruta_a is None:
        return ()

    # Operadores intra: solo involucran una ruta (ruta_a)
    if op in {"relocate_intra", "swap_intra", "2opt_intra"}:
        r = mov.ruta_a
        i = mov.i if mov.i is not None else 0
        j = mov.j if mov.j is not None else i
        if op == "relocate_intra":
            # Solo se mueve la tarea en posición i
            return (rutas[r][i],)
        # Para swap y 2opt: se devuelven todas las tareas del rango [min(i,j)..max(i,j)]
        return tuple(rutas[r][min(i, j) : max(i, j) + 1])

    # Operadores inter simples: involucran una tarea de cada ruta
    if op in {"relocate_inter", "swap_inter"}:
        ids: list[int] = []
        if mov.ruta_a is not None and mov.i is not None:
            ids.append(rutas[mov.ruta_a][mov.i])   # Tarea que sale de la ruta A
        if op == "swap_inter" and mov.ruta_b is not None and mov.j is not None:
            ids.append(rutas[mov.ruta_b][mov.j])   # Tarea que sale de la ruta B (solo en swap)
        return tuple(ids)

    # 2opt_star: las colas completas son las que se mueven
    if op == "2opt_star":
        ids2: list[int] = []
        if mov.ruta_a is not None and mov.i is not None:
            ids2.extend(rutas[mov.ruta_a][mov.i + 1 :])  # Cola de la ruta A (desde cut_a+1)
        if mov.ruta_b is not None and mov.j is not None:
            ids2.extend(rutas[mov.ruta_b][mov.j + 1 :])  # Cola de la ruta B (desde cut_b+1)
        return tuple(ids2)

    # cross_exchange: los segmentos completos de ambas rutas
    if op == "cross_exchange":
        ids3: list[int] = []
        if mov.ruta_a is not None and mov.i is not None and mov.j is not None:
            ids3.extend(rutas[mov.ruta_a][mov.i : mov.j + 1])  # Segmento de la ruta A
        if mov.ruta_b is not None and mov.k is not None and mov.l is not None:
            ids3.extend(rutas[mov.ruta_b][mov.k : mov.l + 1])  # Segmento de la ruta B
        return tuple(ids3)

    return ()


# ------------------------------------------------------------
# FUNCIÓN AUXILIAR PRIVADA: _aplicar_backend_gpu_placeholder
# ------------------------------------------------------------
# Registra qué backend se pidió y cuál realmente se usó.
# El backend GPU es una API preparada para el futuro: hoy no
# hay un kernel GPU implementado, por lo que siempre se usa
# CPU aunque se haya pedido GPU. Esto se registra en el
# MovimientoVecindario para trazabilidad.
def _aplicar_backend_gpu_placeholder(usar_gpu: bool) -> tuple[str, str]:
    if not usar_gpu:
        return "cpu", "cpu"   # Pedido: cpu, real: cpu
    # Placeholder de backend GPU: aún no hay kernel indexado implementado.
    return "gpu", "cpu"       # Pedido: gpu, real: cpu (fallback)


# ============================================================
# FUNCIÓN PRINCIPAL: generar_vecino_ids
# ============================================================
# Genera un vecino operando sobre la representación INDEXADA
# (listas de enteros en lugar de etiquetas de texto).
# Esta representación es más rápida para comparar y copiar,
# especialmente útil cuando el algoritmo evalúa miles de
# vecinos por segundo.
def generar_vecino_ids(
    solucion_ids: Sequence[Sequence[int]],  # Solución como listas de enteros
    *,
    rng: random.Random | None = None,             # Generador de números aleatorios (para reproducibilidad)
    operadores: Iterable[str] = OPERADORES_POPULARES,  # Qué operadores pueden aplicarse
    usar_gpu: bool = False,                       # Flag de backend GPU (hoy: fallback a CPU)
    encoding: SearchEncoding | None = None,       # Mapa ID<->etiqueta para decodificar labels_movidos
) -> tuple[list[list[int]], MovimientoVecindario]:
    """
    Genera un vecino sobre representación indexada (IDs enteros).
    """
    rng = rng or random.Random()   # Si no se pasa un RNG, crea uno nuevo (no reproducible)

    # Convierte cada elemento a int (garantiza tipo correcto aunque venga como numpy.int64, etc.)
    rutas = [[int(x) for x in r] for r in solucion_ids]

    ops = list(operadores)   # Convierte el iterable a lista para poder elegir al azar
    if not ops:
        raise ValueError("operadores está vacío.")

    # Registra el backend real que se usará
    backend_solicitado, backend_real = _aplicar_backend_gpu_placeholder(usar_gpu)

    # Bucle de intentos: si el operador elegido no es aplicable a la solución
    # actual (p.ej. todas las rutas tienen solo 1 tarea y se elige swap_intra),
    # se vuelve a intentar con otro operador. Límite de 500 intentos para evitar
    # bucles infinitos en soluciones degeneradas.
    intentos = 0
    while True:
        intentos += 1
        if intentos > 500:
            raise RuntimeError("No se pudo generar un vecino: solución demasiado pequeña para los operadores.")

        op = rng.choice(ops)           # Elige un operador al azar
        activos = _rutas_con_indices(rutas)  # Índices de rutas no vacías
        if not activos:
            continue  # Si todas las rutas están vacías, reintenta

        # ---- OPERADORES INTRA ----

        if op == "relocate_intra":
            # Necesita al menos 2 tareas en la ruta para que el movimiento tenga efecto
            cand = [x for x in activos if len(rutas[x]) >= 2]
            if not cand:
                continue
            r = rng.choice(cand)   # Elige una ruta candidata al azar
            n = len(rutas[r])      # Longitud de la ruta elegida
            i = rng.randrange(n)   # Posición origen (aleatoria)
            j = rng.randrange(n)   # Posición destino (aleatoria)
            if i == j:
                continue  # Mover al mismo lugar no genera vecino nuevo
            vec = op_relocate_intra(rutas, r, i, j)
            mov = MovimientoVecindario(op, ruta_a=r, i=i, j=j)

        elif op == "swap_intra":
            cand = [x for x in activos if len(rutas[x]) >= 2]
            if not cand:
                continue
            r = rng.choice(cand)
            n = len(rutas[r])
            # rng.sample(range(n), 2) devuelve 2 índices distintos sin repetición
            i, j = rng.sample(range(n), 2)
            vec = op_swap_intra(rutas, r, i, j)
            mov = MovimientoVecindario(op, ruta_a=r, i=i, j=j)

        elif op == "2opt_intra":
            # Necesita al menos 3 tareas para que exista un segmento de longitud 2
            cand = [x for x in activos if len(rutas[x]) >= 3]
            if not cand:
                continue
            r = rng.choice(cand)
            n = len(rutas[r])
            i = rng.randrange(0, n - 1)      # i puede ir de 0 a n-2
            j = rng.randrange(i + 1, n)      # j siempre > i (garantiza segmento de al menos 2)
            vec = op_2opt_intra(rutas, r, i, j)
            mov = MovimientoVecindario(op, ruta_a=r, i=i, j=j)

        # ---- OPERADORES INTER ----

        elif op == "relocate_inter":
            # Necesita al menos 2 rutas y que la ruta origen tenga tareas
            if len(activos) < 1 or len(rutas) < 2:
                continue
            ra = rng.choice(activos)   # Ruta origen (con al menos 1 tarea)
            if not rutas[ra]:
                continue
            rb = rng.randrange(len(rutas))  # Ruta destino (puede estar vacía)
            if ra == rb:
                continue  # No puede ser la misma ruta (sería intra)
            i = rng.randrange(len(rutas[ra]))        # Posición en la ruta origen
            j = rng.randrange(len(rutas[rb]) + 1)   # Posición en la ruta destino (+1 para insertar al final)
            vec = op_relocate_inter(rutas, ra, i, rb, j)
            mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=i, j=j)

        elif op == "swap_inter":
            if len(rutas) < 2:
                continue
            # Ambas rutas deben tener al menos 1 tarea para intercambiar
            non_empty = [x for x in range(len(rutas)) if len(rutas[x]) > 0]
            if len(non_empty) < 2:
                continue
            ra, rb = rng.sample(non_empty, 2)   # Dos rutas distintas no vacías
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
            # El corte puede ser cualquier posición válida dentro de la ruta
            cut_a = rng.randrange(len(rutas[ra]))
            cut_b = rng.randrange(len(rutas[rb]))
            vec = op_two_opt_star(rutas, ra, cut_a, rb, cut_b)
            mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=cut_a, j=cut_b)

        elif op == "cross_exchange":
            # Necesita al menos 2 tareas en cada ruta para extraer un segmento
            if len(rutas) < 2:
                continue
            non_empty = [x for x in range(len(rutas)) if len(rutas[x]) >= 2]
            if len(non_empty) < 2:
                continue
            ra, rb = rng.sample(non_empty, 2)
            na, nb = len(rutas[ra]), len(rutas[rb])
            i = rng.randrange(0, na - 1)     # Inicio del segmento en A
            j = rng.randrange(i + 1, na)     # Fin del segmento en A (j > i)
            k = rng.randrange(0, nb - 1)     # Inicio del segmento en B
            l = rng.randrange(k + 1, nb)     # Fin del segmento en B (l > k)
            vec = op_cross_exchange(rutas, ra, i, j, rb, k, l)
            mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=i, j=j, k=k, l=l)

        else:
            raise ValueError(f"Operador desconocido: {op!r}")

        # ---- Enriquecer el MovimientoVecindario con IDs y labels de tareas movidas ----

        # Obtiene los IDs enteros de las tareas que se desplazaron
        ids_m = _moved_ids(op, rutas, mov)

        # Si hay un encoding disponible, decodifica los IDs a etiquetas legibles (TR1, TR2...)
        labels_m = tuple(decode_task_ids(ids_m, encoding)) if encoding is not None else ()

        # Construye el MovimientoVecindario definitivo con toda la información
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
        # Devuelve la solución vecina (como listas de enteros) y el movimiento aplicado
        return [[int(x) for x in r] for r in vec], mov_out


# ============================================================
# FUNCIÓN PRINCIPAL: generar_vecino
# ============================================================
# Versión de alto nivel que acepta la solución en formato de
# etiquetas (strings como "TR1", "D", etc.) y ofrece dos modos
# de operación según el parámetro `backend`.
def generar_vecino(
    solucion: Sequence[Sequence[Hashable]],   # Solución en formato etiquetas (puede incluir "D")
    *,
    rng: random.Random | None = None,
    operadores: Iterable[str] = OPERADORES_POPULARES,
    marcador_depot: str = "D",                  # Marcador de depósito a eliminar antes de operar
    devolver_con_deposito: bool = True,          # Si True, agrega "D" al inicio y fin de cada ruta al devolver
    usar_gpu: bool = False,
    backend: Literal["labels", "ids"] = "labels",  # Modo de operación interno
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

    # ---- Modo IDs: usa la ruta rápida con enteros ----
    if backend == "ids":
        if encoding is None:
            raise ValueError("backend='ids' requiere un SearchEncoding en el parámetro 'encoding'.")
        # Codifica la solución de etiquetas a IDs enteros usando el encoding de la instancia
        rutas_ids = encode_solution(solucion, encoding)
        # Aplica el operador sobre la representación indexada
        vecino_ids, mov = generar_vecino_ids(
            rutas_ids,
            rng=rng,
            operadores=operadores,
            usar_gpu=usar_gpu,
            encoding=encoding,
        )
        # Decodifica de vuelta a etiquetas, con o sin marcador "D"
        if devolver_con_deposito:
            return decode_solution(vecino_ids, encoding, con_deposito=True), mov
        return decode_solution(vecino_ids, encoding, con_deposito=False), mov

    # ---- Modo labels: opera directamente con strings ----

    backend_solicitado, backend_real = _aplicar_backend_gpu_placeholder(usar_gpu)
    rng = rng or random.Random()

    # Elimina los marcadores "D" de la solución de entrada para operar solo con tareas
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

        # ---- Operadores intra ----
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
                # Segmento de longitud 1 no genera vecino útil
                continue
            vec = op_2opt_intra(rutas, r, i, j)
            mov = MovimientoVecindario(op, ruta_a=r, i=i, j=j)

        # ---- Operadores inter ----
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
                continue  # Garantiza que sea inter y no intra
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

        # Reconstruye el movimiento con la información de backend
        # (en el modo labels no hay ids_movidos ni labels_movidos)
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

        # Devuelve el vecino con o sin marcador de depósito según lo pedido
        if devolver_con_deposito:
            return desnormalizar_con_deposito(vec, marcador_depot=marcador_depot), mov
        return vec, mov
