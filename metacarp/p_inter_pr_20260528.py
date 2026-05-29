"""
Selector probabilistico ``p_inter`` para el experimento p_inter_pr_20260528.

Motivacion (extraida de la Seccion 12 del notebook de analisis):
  - Los operadores INTER-ruta tienen tasa de mejora 4-7x superior a los
    INTRA-ruta por propuesta, incluso cuando la solucion ya es factible.
  - El selector binario estricto (``strict_intra_inter_20260524``) NUNCA
    propone inter cuando la solucion es factible, dejando esa ganancia
    sin capturar.
  - El experimento anterior ``p_inter_exp_2026050524`` fallo por dos
    razones simultaneas (anclaje de 1a tarea + p_inter agresivo > 0.5).
    Este modulo retoma la idea de p_inter MODERADO sin esos defectos.

Diseno:
  - Se reutiliza el selector original ``seleccionar_grupo_operadores_inter_intra``
    de ``metacarp.metaheuristicas_utils`` (no se reescribe la logica).
  - Se envuelve en un wrapper que IGNORA los kwargs ``alpha_inter`` y
    ``p_inter`` que la MH le pasaria, sustituyendolos por los valores
    experimentales fijos (``ALPHA_INTER=0.80``, ``P_INTER=0.20``).
  - El monkey-patch ``aplicar_patch_p_inter`` instala el wrapper en el
    modulo de la MH; debe llamarse DESPUES de ``aplicar_patch_pr`` (de
    ``metacarp.path_relinking_20260528``), que internamente tambien
    parchea el selector con la version binaria estricta. La segunda
    escritura gana, por lo que el orden importa.

Combinacion con Path Relinking:
  - El experimento p_inter_pr_20260528 se monta sobre el experimento PR
    validado en la Seccion 12: selector p_inter (NUEVO) + kick reactivo
    (igual que strict) + Path Relinking truncado tras el kick (igual que
    PR). Asi cualquier mejora vs PR se atribuye al cambio de selector.
"""
from __future__ import annotations

import importlib

# Importamos la implementacion original del selector para reutilizar su
# logica (un solo rng.random + reparto inter/intra por probabilidad).
# Nuestro wrapper solo sobreescribe los kwargs alpha_inter y p_inter
# con los valores experimentales fijos.
from metacarp.metaheuristicas_utils import (
    seleccionar_grupo_operadores_inter_intra as _selector_orig,
)

# -----------------------------------------------------------------------------
# Constantes experimentales
# -----------------------------------------------------------------------------
# Probabilidad de proponer el grupo INTER cuando la solucion es FACTIBLE.
# Valor moderado (0.20) elegido tras el analisis cualitativo de la Seccion 12:
#   - Suficiente para capturar la alta tasa de mejora de los inter (4-7x
#     superior a los intra por propuesta).
#   - Lo bastante bajo como para no degradar la factibilidad cuando ya
#     estamos en region factible (el experimento p_inter_exp uso > 0.5
#     y empeoro 4 de 5 MH).
P_INTER: float = 0.20

# Probabilidad de proponer el grupo INTER cuando la solucion VIOLA capacidad.
# 0.80 es el default del selector original; mantenerlo asegura reparacion
# agresiva igual que en el baseline.
ALPHA_INTER: float = 0.80


# -----------------------------------------------------------------------------
# Wrapper del selector
# -----------------------------------------------------------------------------
def seleccionar_grupo_p_inter_fijo(
    rng,
    violacion: float,
    ops_intra,
    ops_inter,
    operadores_fallback,
    *,
    alpha_inter: float = ALPHA_INTER,
    p_inter: float = P_INTER,
    **_ignorado,
):
    """Wrapper que delega en el selector original con valores fijos.

    Las MH (TS, RTS, SA, ABC, Cuckoo) pasan su propio ``p_inter`` (que
    suele venir de ``p_inter=0.0`` o del default ``0.6`` del wrapper)
    como kwarg. Este wrapper IGNORA esos valores y fuerza ``ALPHA_INTER``
    y ``P_INTER`` definidos en este modulo. El resto de la mecanica
    (UN solo ``rng.random()``, fallback a operadores_fallback, etc.)
    queda intacta porque delegamos a la implementacion original.

    Parametros
    ----------
    rng : Random
        Generador aleatorio de la MH.
    violacion : float
        Exceso de capacidad agregado de la solucion actual (>= 0).
    ops_intra, ops_inter : Sequence[str]
        Listas de operadores intra y inter disponibles.
    operadores_fallback : Sequence[str]
        Lista a usar si tanto ops_intra como ops_inter estuvieran vacias.
    alpha_inter, p_inter : float
        IGNORADOS: el wrapper los sobreescribe con ``ALPHA_INTER`` y
        ``P_INTER`` de este modulo. Aceptamos los kwargs por compatibilidad
        de firma con la funcion original.
    **_ignorado :
        Cualquier kwarg adicional se absorbe sin error.

    Retorno
    -------
    tuple[list[str], bool]
        ``(grupo_operadores, hubo_violacion)`` — misma semantica que el
        selector original.
    """
    # Forzamos los valores experimentales; los argumentos recibidos los
    # descartamos intencionalmente.
    return _selector_orig(
        rng,
        violacion,
        ops_intra,
        ops_inter,
        operadores_fallback,
        alpha_inter=ALPHA_INTER,
        p_inter=P_INTER,
    )


# -----------------------------------------------------------------------------
# Monkey-patch
# -----------------------------------------------------------------------------
def aplicar_patch_p_inter(nombre_modulo_mh: str) -> None:
    """Reemplaza ``seleccionar_grupo_operadores_inter_intra`` en el modulo MH.

    El nombre de la funcion en el modulo de la MH (``recocido_simulado``,
    ``busqueda_tabu_simple``, etc.) se reasigna a nuestro wrapper. Como
    las MH llaman el selector usando el nombre local (despues del
    ``from metacarp.metaheuristicas_utils import seleccionar_grupo_*``),
    es necesario sobrescribir el atributo en el modulo de la MH y no
    en ``metaheuristicas_utils``.

    Importante: este patch debe instalarse DESPUES de
    ``aplicar_patch_pr`` (del modulo ``path_relinking_20260528``), que
    a su vez parchea el selector con la version binaria estricta. La
    segunda asignacion gana; por eso el orden es:
        1. aplicar_patch_pr(modulo, p_pr=P_PR)   -> instala strict + PR
        2. aplicar_patch_p_inter(modulo)          -> sobreescribe selector

    Parametros
    ----------
    nombre_modulo_mh : str
        Nombre completamente cualificado del modulo de la MH a parchear
        (p.ej. ``"metacarp.recocido_simulado"``).
    """
    mh = importlib.import_module(nombre_modulo_mh)
    mh.seleccionar_grupo_operadores_inter_intra = seleccionar_grupo_p_inter_fijo
