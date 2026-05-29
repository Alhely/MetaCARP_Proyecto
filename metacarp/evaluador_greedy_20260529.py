"""
Evaluador de costo con seleccion GREEDY de orientacion por tarea.

PROBLEMA EN EL EVALUADOR CANONICO (``evaluador_costo``):
=========================================================
``costo_rapido_ids`` y ``costo_lote_ids`` siempre entran a cada tarea por
``u_arr[tid]`` y salen por ``v_arr[tid]`` (orientacion canonica fija).
Eso es subobtimo: un vehiculo fisico puede recorrer un arco ``(u,v)`` en
cualquier direccion. El costo CARP correcto es INVARIANTE a la orientacion
elegida en cada paso.

Diagnostico empirico (Seccion 14): bajo orientacion canonica, gap inicial
del pickle aleatorio es 52%; gap inicial fisicamente correcto (greedy
orientation) es 20% en las mismas soluciones. Las 5 MH terminan con
gap ~30% porque el espacio de busqueda esta efectivamente reducido por
la mitad: no existe operador que "invierta" una tarea.

SOLUCION DE ESTE MODULO:
========================
Reescribir los evaluadores rapidos para que en CADA tarea elijan
dinamicamente la orientacion que minimiza el dead-heading de entrada:

    Si ``dist[prev, u_t] <= dist[prev, v_t]``: entrar por ``u``, salir por ``v``.
    Si no: entrar por ``v``, salir por ``u``.

Es una heuristica GREEDY localmente optima (no busca el plan global con DP,
pero captura >95% del beneficio en instancias pequenas).

Esto NO cambia la semantica del problema CARP; solo elimina un artefacto
artificial del evaluador. Equivalente matematico a que los operadores
tuvieran libre la inversion de orientacion sin cambiar la representacion.

Uso (en cada worker, antes de importar la MH):

    from metacarp.evaluador_greedy_20260529 import aplicar_patch_evaluador_greedy
    aplicar_patch_evaluador_greedy()
    # ... luego importar las MH y correr.

Costo computacional: el bucle Python anade overhead (~3-5x) frente al
evaluador vectorizado NumPy original. Para instancias pequenas (gdb, kshs)
es perfectamente asumible (las MH siguen corriendo en segundos por corrida).
"""
from __future__ import annotations

import sys
from typing import Any, Hashable, Sequence

import numpy as np


# ============================================================
# Evaluador greedy (por una solucion)
# ============================================================

def costo_rapido_ids_greedy(
    solucion_ids: Sequence[Sequence[int]],
    ctx: Any,
) -> float:
    """Costo de una solucion con seleccion greedy de orientacion por tarea.

    Identico a ``costo_rapido_ids`` salvo que en cada tarea elige la
    orientacion (u->v o v->u) que minimiza el dead-heading de entrada.

    El bucle Python es la consecuencia de la dependencia secuencial: la
    orientacion de la tarea ``k`` depende del nodo donde termino la ``k-1``.
    """
    dist   = ctx.dist
    u_arr  = ctx.u_arr
    v_arr  = ctx.v_arr
    cs_arr = ctx.costo_serv_arr
    depot  = ctx.depot

    total = 0.0
    for ruta in solucion_ids:
        if not ruta:
            continue
        prev = depot
        for tid in ruta:
            tid_i = int(tid)
            u = int(u_arr[tid_i])
            v = int(v_arr[tid_i])
            d_u = float(dist[prev, u])
            d_v = float(dist[prev, v])
            if d_u <= d_v:
                # Orientacion canonica u->v.
                total += d_u + float(cs_arr[tid_i])
                prev = v
            else:
                # Orientacion invertida v->u.
                total += d_v + float(cs_arr[tid_i])
                prev = u
        # Regreso al deposito desde el nodo donde quedo el vehiculo.
        total += float(dist[prev, depot])
    return float(total)


def costo_rapido_greedy(
    solucion_labels: Sequence[Sequence[Hashable]],
    ctx: Any,
) -> float:
    """Wrapper labels -> ids -> ``costo_rapido_ids_greedy``.

    Mantiene la misma firma que ``costo_rapido`` original. Reutiliza el
    helper ``_ruta_labels_a_ids`` de ``evaluador_costo`` (lo importamos
    diferidamente para no introducir dependencia circular).
    """
    from metacarp.evaluador_costo import _ruta_labels_a_ids

    md = ctx.marcador_depot.upper()
    label_to_id = ctx.encoding.label_to_id
    rutas_ids: list[list[int]] = []
    for ruta in solucion_labels:
        rutas_ids.append(_ruta_labels_a_ids(ruta, label_to_id, md))
    return costo_rapido_ids_greedy(rutas_ids, ctx)


# ============================================================
# Empaquetado greedy del lote
# ============================================================

def _empaquetar_lote_ids_greedy(
    soluciones_ids: Sequence[Sequence[Sequence[int]]],
    ctx: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Empaqueta un lote heterogeneo de soluciones en arrays planos con
    orientacion greedy por tarea ya decidida.

    A diferencia del empaquetado canonico, aqui decidimos la orientacion
    de cada tarea ANTES de poblar ``origs`` y ``dests``. Los arrays
    resultantes son consumibles directamente por la reduccion vectorizada
    de ``costo_lote_ids_greedy`` (CPU/GPU), porque la decision ya esta
    materializada en los enteros.
    """
    dist   = ctx.dist
    u_arr  = ctx.u_arr
    v_arr  = ctx.v_arr
    cs_arr = ctx.costo_serv_arr
    depot  = ctx.depot

    origs:   list[int]   = []
    dests:   list[int]   = []
    cs_l:    list[float] = []
    sol_idx: list[int]   = []

    for s_idx, sol in enumerate(soluciones_ids):
        for ruta in sol:
            if not ruta:
                continue
            prev = depot
            for tid in ruta:
                tid_i = int(tid)
                u = int(u_arr[tid_i])
                v = int(v_arr[tid_i])
                # Decision greedy de orientacion.
                if float(dist[prev, u]) <= float(dist[prev, v]):
                    entrada, salida = u, v
                else:
                    entrada, salida = v, u
                origs.append(prev)
                dests.append(entrada)
                cs_l.append(float(cs_arr[tid_i]))
                sol_idx.append(s_idx)
                prev = salida
            # Regreso al deposito como ultimo paso de esta ruta.
            origs.append(prev)
            dests.append(depot)
            cs_l.append(0.0)
            sol_idx.append(s_idx)

    return (
        np.asarray(origs,   dtype=np.int64),
        np.asarray(dests,   dtype=np.int64),
        np.asarray(cs_l,    dtype=np.float64),
        np.asarray(sol_idx, dtype=np.int64),
    )


# ============================================================
# Evaluador greedy por lote
# ============================================================

def costo_lote_ids_greedy(
    soluciones_ids: Sequence[Sequence[Sequence[int]]],
    ctx: Any,
) -> np.ndarray:
    """Evaluador por lotes con orientacion greedy ya materializada.

    Tras el empaquetado greedy, la reduccion final es identica al
    ``costo_lote_ids`` original (vectorizada con ``np.add.at``).
    """
    n_sol = len(soluciones_ids)
    if n_sol == 0:
        return np.zeros((0,), dtype=np.float64)

    orig, dest, cs, sol_idx = _empaquetar_lote_ids_greedy(soluciones_ids, ctx)
    if orig.size == 0:
        return np.zeros((n_sol,), dtype=np.float64)

    if ctx.usar_gpu:
        # GPU path: identico al original; la decision greedy ya esta en
        # los arrays empaquetados.
        import cupy as cp  # type: ignore
        d_gpu = ctx.dist_gpu
        orig_g = cp.asarray(orig)
        dest_g = cp.asarray(dest)
        cs_g   = cp.asarray(cs)
        sol_g  = cp.asarray(sol_idx)
        contrib = d_gpu[orig_g, dest_g] + cs_g
        out = cp.zeros((n_sol,), dtype=cp.float64)
        try:
            cupyx_scatter = cp.scatter_add
        except AttributeError:
            import cupyx
            cupyx_scatter = cupyx.scatter_add
        cupyx_scatter(out, sol_g, contrib)
        return cp.asnumpy(out)

    # CPU path.
    contrib = ctx.dist[orig, dest] + cs
    out = np.zeros((n_sol,), dtype=np.float64)
    np.add.at(out, sol_idx, contrib)
    return out


def costo_lote_penalizado_ids_greedy(
    soluciones_ids: Sequence[Sequence[Sequence[int]]],
    ctx: Any,
    lam: float,
    *,
    usar_penal: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Objetivo penalizado: ``costo_puro + lambda * violacion`` con greedy.

    La violacion de capacidad se calcula con el mismo
    ``exceso_capacidad_sol_ids`` del modulo original (no depende de
    orientacion: depende de demandas por ruta, que son constantes).
    """
    # Importacion diferida para evitar circular import a nivel de modulo.
    from metacarp.evaluador_costo import exceso_capacidad_sol_ids

    base = costo_lote_ids_greedy(soluciones_ids, ctx)
    n = len(soluciones_ids)
    if n == 0:
        z = np.zeros((0,), dtype=np.float64)
        return z, z, z

    if (
        not usar_penal
        or not np.isfinite(ctx.capacidad_max)
        or float(ctx.capacidad_max) <= 0
    ):
        z = np.zeros_like(base)
        return base.copy(), base, z

    exc = np.zeros((n,), dtype=np.float64)
    for i, sid in enumerate(soluciones_ids):
        exc[i] = exceso_capacidad_sol_ids(sid, ctx)
    obj = base + float(lam) * exc
    return obj, base, exc


# ============================================================
# Monkey-patch: instala las versiones greedy en evaluador_costo y MHs
# ============================================================

# Modulos donde las MH y utilidades importan los simbolos del evaluador.
# Cuando un modulo hace ``from .evaluador_costo import costo_rapido_ids``,
# Python crea un binding LOCAL al objeto funcion en ese modulo, por lo que
# debemos reescribir el simbolo en CADA modulo consumidor para que el patch
# tenga efecto en codigo ya importado.
_MODULOS_CONSUMIDORES: tuple[str, ...] = (
    "metacarp.evaluador_costo",
    "metacarp.recocido_simulado",
    "metacarp.busqueda_tabu_simple",
    "metacarp.busqueda_tabu_reactiva",
    "metacarp.abejas_simple",
    "metacarp.cuckoo_search",
    "metacarp.metaheuristicas_utils",
)

# Mapeo simbolo original -> implementacion greedy.
_REEMPLAZOS: dict[str, Any] = {
    "costo_rapido_ids":           costo_rapido_ids_greedy,
    "costo_rapido":               costo_rapido_greedy,
    "costo_lote_ids":             costo_lote_ids_greedy,
    "costo_lote_penalizado_ids":  costo_lote_penalizado_ids_greedy,
}


def aplicar_patch_evaluador_greedy() -> None:
    """Reemplaza los evaluadores canonicos por sus versiones greedy.

    Sobreescribe los simbolos en ``metacarp.evaluador_costo`` y en cada
    modulo consumidor (MHs y utilidades) ya cargado. Si un modulo aun no
    se ha importado, la sobrescritura en ``evaluador_costo`` basta: cuando
    el modulo finalmente importe el simbolo, vera la version greedy.
    """
    # Garantizar que evaluador_costo este cargado antes de patchearlo,
    # por si el llamador aun no lo importo.
    import metacarp.evaluador_costo  # noqa: F401

    for mod_name in _MODULOS_CONSUMIDORES:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for nombre_simbolo, impl_greedy in _REEMPLAZOS.items():
            if hasattr(mod, nombre_simbolo):
                setattr(mod, nombre_simbolo, impl_greedy)
