"""
Modulo de soporte para la variante experimental strict_intra_inter_20260524.

Provee:
  - Constantes de los conjuntos REDUCIDOS de operadores (5 = 2 intra + 3 inter).
  - Selector BINARIO ESTRICTO (reemplazo de
    ``seleccionar_grupo_operadores_inter_intra``).
  - Funciones de KICK inter-ruta (``aplicar_kick_labels`` y
    ``aplicar_kick_ids``) usadas por las metaheuristicas cuando se acumulan
    demasiadas iteraciones sin mejora del mejor global.

Diferencia clave con la variante anterior (``p_inter_exp_2026050524``):
  - No se monkey-patchea el generador de vecinos. La unica intervencion en
    el dispatcher de vecindario es el SELECTOR BINARIO (capa 1).
  - El KICK (capa 2) se ejecuta dentro de cada MH cuando se cumple el
    criterio de estancamiento (``max_iter_sin_mejora_kick``). Aplica una
    rafaga de movimientos inter-ruta para forzar diversificacion.

Convenciones:
  - Los conjuntos de operadores se exponen como ``tuple[str, ...]`` para que
    sean inmutables y puedan pasarse directamente al argumento ``operadores=``
    de los wrappers ``*_desde_instancia``.
  - El selector ignora ``alpha_inter`` y ``p_inter`` (absorbidos en
    ``**_ignorado``): la decision es DETERMINISTA en funcion de la violacion.
  - Los kicks usan el ``generar_vecino`` ORIGINAL (alias ``_gv_orig``), nunca
    una version monkey-patched. Esto garantiza que el kick siempre aplique
    movimientos inter-ruta sin restricciones posicionales adicionales.
"""
from __future__ import annotations

from metacarp.vecindarios import (
    generar_vecino as _gv_orig,
    generar_vecino_ids as _gvi_orig,
)

# ============================================================
# Conjuntos REDUCIDOS de operadores
# ============================================================

# Operadores intra-ruta admitidos por la variante experimental.
# Excluye ``2opt_intra`` por considerarlo redundante con ``swap_intra`` en
# instancias pequenas (decision de diseno del experimento).
OPERADORES_INTRA_STRICT: tuple[str, ...] = ("relocate_intra", "swap_intra")

# Operadores inter-ruta admitidos por la variante experimental.
# Excluye ``relocate_inter`` para favorecer reorganizaciones SIMETRICAS
# entre rutas (intercambios y partidos) en vez de transferencias asimetricas.
OPERADORES_INTER_STRICT: tuple[str, ...] = ("swap_inter", "2opt_star", "cross_exchange")

# Conjunto completo (5 operadores) que se pasa como ``operadores=`` a las
# metaheuristicas. El orden importa para reproducibilidad de la seleccion
# uniforme dentro del grupo.
OPERADORES_STRICT_5: tuple[str, ...] = OPERADORES_INTRA_STRICT + OPERADORES_INTER_STRICT


# ============================================================
# Selector BINARIO ESTRICTO (capa 1 del experimento)
# ============================================================

def seleccionar_grupo_strict(
    rng, violacion, ops_intra, ops_inter, operadores_fallback, **_ignorado
):
    """Selector BINARIO DETERMINISTA.

    Reemplaza a ``seleccionar_grupo_operadores_inter_intra`` cuando se
    aplica el monkey-patch del modulo de la MH. Misma firma posicional
    (``rng``, ``violacion``, ``ops_intra``, ``ops_inter``,
    ``operadores_fallback``); los kwargs ``alpha_inter`` y ``p_inter`` se
    absorben en ``**_ignorado`` (no consumen ningun ``rng.random()``).

    Logica:
      - Si ``violacion > 1e-12``: devuelve el GRUPO INTER (reparacion).
      - Si ``violacion <= 1e-12``: devuelve el GRUPO INTRA (refinamiento).
      - Si el grupo elegido esta vacio, cae al otro o al fallback.

    Devuelve la tupla ``(grupo_operadores, hubo_violacion)`` con la misma
    semantica que el helper canonico, para preservar compatibilidad con los
    contadores de iteraciones-con-violacion de TS y RTS.

    NOTA: este selector NO consume ningun valor del ``rng``. Es DETERMINISTA
    una vez fijada la violacion de la solucion actual. Si la MH compara la
    secuencia de ``rng`` con otra corrida, la diferencia respecto al helper
    canonico sera EXACTAMENTE 1 ``rng.random()`` menos por iteracion.
    """
    # Comparacion con tolerancia numerica identica al helper canonico (1e-12).
    # Una violacion "real" siempre supera 1; valores por debajo del umbral
    # son ruido numerico que tratamos como factible.
    if violacion > 1e-12 and ops_inter:
        # Hay violacion + tenemos operadores inter: forzar reparacion.
        return list(ops_inter), True
    if ops_intra:
        # Solucion factible (o sin operadores inter): refinamiento intra.
        return list(ops_intra), violacion > 1e-12
    # Casos degenerados (uno de los dos grupos vacio): caemos al otro o
    # al fallback completo. Mantenemos el flag de violacion para que los
    # consumidores cuenten correctamente las iteraciones-con-violacion.
    if ops_inter:
        return list(ops_inter), violacion > 1e-12
    return list(operadores_fallback), violacion > 1e-12


# ============================================================
# KICK inter-ruta (capa 2 del experimento)
# ============================================================

def aplicar_kick_labels(sol, rng, md_op, encoding=None):
    """Aplica una rafaga de movimientos INTER-RUTA en formato labels.

    Numero de pasos: ``max(1, n_tareas // 20)``. Asi una instancia con 20
    tareas recibe 1 movimiento, una con 40 tareas recibe 2, etc. Es una
    perturbacion proporcional al tamano del problema.

    Usa el ``generar_vecino`` ORIGINAL (alias ``_gv_orig``), nunca una
    version monkey-patched. La solucion resultante se acepta INCONDICIONALMENTE
    por la MH llamadora: el kick es un mecanismo de DIVERSIFICACION, no de
    busqueda local.

    Parametros
    ----------
    sol : list[list[str]]
        Solucion en formato labels (con depot al inicio/fin de cada ruta).
    rng : random.Random
        Generador aleatorio que comparte la MH llamadora (preserva
        reproducibilidad de la corrida).
    md_op : str
        Etiqueta del deposito (normalmente ``"D"``).
    encoding : SearchEncoding | None
        Encoding opcional para ``generar_vecino`` cuando opera en backend
        ``"ids"``. En la mayoria de MH (SA, TS, RTS, Cuckoo) basta con
        ``None`` porque el dispatcher default usa el backend ``"labels"``.

    Devuelve
    --------
    list[list[str]]
        Solucion perturbada (mismo formato de entrada).
    """
    # Contamos las tareas activas (excluimos los marcadores de deposito de los
    # extremos de cada ruta). Una ruta vacia (solo "D"-"D") aporta 0 tareas.
    n_tareas = sum(max(len(r) - 2, 0) for r in sol)
    # Numero de movimientos inter encadenados: proporcional al tamano del
    # problema pero con un piso de 1 para evitar kicks degenerados.
    pasos = max(1, n_tareas // 20)
    for _ in range(pasos):
        # IMPORTANTE: usamos ``_gv_orig`` (referencia importada al cargar este
        # modulo). Si la MH monkey-patchea ``generar_vecino`` dentro de su
        # propio modulo, esta llamada NO se ve afectada.
        sol, _ = _gv_orig(
            sol,
            rng=rng,
            operadores=list(OPERADORES_INTER_STRICT),
            marcador_depot=md_op,
            devolver_con_deposito=True,
            encoding=encoding,
        )
    return sol


def aplicar_kick_ids(sol_ids, rng, encoding=None):
    """Aplica una rafaga de movimientos INTER-RUTA en formato IDs.

    Variante para metaheuristicas que mantienen la solucion en
    representacion entera (actualmente solo ABC Simple). El numero de pasos
    sigue la misma formula que ``aplicar_kick_labels``: ``max(1, n_tareas // 20)``.

    Usa el ``generar_vecino_ids`` ORIGINAL (alias ``_gvi_orig``), nunca una
    version monkey-patched.

    Parametros
    ----------
    sol_ids : list[list[int]]
        Solucion en formato IDs (sin depot; depot implicito en los extremos).
    rng : random.Random
        Generador aleatorio compartido con la MH llamadora.
    encoding : SearchEncoding | None
        Encoding necesario para el dispatcher de operadores ``_ids``.

    Devuelve
    --------
    list[list[int]]
        Solucion perturbada en formato IDs.
    """
    # En formato IDs no hay marcadores de deposito en las rutas: todas las
    # entradas son tareas reales. ``len(r)`` ya da el numero de tareas.
    n_tareas = sum(len(r) for r in sol_ids)
    pasos = max(1, n_tareas // 20)
    for _ in range(pasos):
        # Misma justificacion que en aplicar_kick_labels: usamos el dispatcher
        # ORIGINAL (no la version monkey-patched de la MH).
        sol_ids, _ = _gvi_orig(
            sol_ids,
            rng=rng,
            operadores=list(OPERADORES_INTER_STRICT),
            encoding=encoding,
        )
    return sol_ids
