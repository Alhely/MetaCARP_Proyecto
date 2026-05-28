"""
AOS con Probability Matching para la variante experimental aos_pm_20260527.

Provee:
  - Constantes de los conjuntos de operadores (mismas 5 que strict_intra_inter_20260524).
  - ``_AOSState``: pesos adaptativos por operador (Probability Matching).
  - ``seleccionar_grupo_aos``: selector BINARIO ESTRICTO para el grupo +
    PM ponderado para el operador especifico dentro del grupo.
    Devuelve ``([op_elegido], hubo_viol)`` -- lista de UN elemento -- para
    forzar que ``generar_vecino`` use exactamente ese operador.
  - ``aplicar_patch_aos``: monkey-patch en DOS puntos del modulo de la MH:
    (1) ``seleccionar_grupo_operadores_inter_intra`` -> selector AOS,
    (2) ``ContadorOperadores.registrar_mejora`` -> actualiza pesos AOS
        cuando la MH registra una mejora del mejor global.

Diferencia clave respecto a ``strict_intra_inter_20260524``:
  - El selector ya NO es determinista dentro del grupo: usa pesos adaptativos
    (Probability Matching) para favorecer los operadores que han generado
    mas mejoras globales en la corrida actual.
  - ``alpha_aos`` controla la velocidad de adaptacion (default 0.2).
  - ``w_min`` es el peso minimo para evitar que un operador se descarte
    completamente al principio (default 0.05).
"""
from __future__ import annotations

import importlib

# ============================================================
# Conjuntos de operadores (iguales que strict_intra_inter)
# ============================================================

# Operadores intra-ruta admitidos por la variante AOS PM.
# Excluye ``2opt_intra`` por considerarlo redundante con ``swap_intra`` en
# instancias pequenas (decision de diseno heredada del experimento strict).
OPERADORES_INTRA_AOS: tuple[str, ...] = ("relocate_intra", "swap_intra")

# Operadores inter-ruta admitidos por la variante AOS PM.
# Excluye ``relocate_inter`` para favorecer reorganizaciones SIMETRICAS
# entre rutas (intercambios y partidos) en vez de transferencias asimetricas.
OPERADORES_INTER_AOS: tuple[str, ...] = ("swap_inter", "2opt_star", "cross_exchange")

# Conjunto completo (5 operadores) que se pasa como ``operadores=`` a las
# metaheuristicas. El orden importa para reproducibilidad de la inicializacion
# uniforme de pesos del Probability Matching.
OPERADORES_AOS_5: tuple[str, ...] = OPERADORES_INTRA_AOS + OPERADORES_INTER_AOS

# Tasa de aprendizaje por defecto del Probability Matching: valor moderado
# que permite adaptarse en pocas docenas de mejoras sin volverse miope.
_ALPHA_DEFAULT: float = 0.2
# Peso minimo por operador: evita que un operador caiga a 0 y quede
# permanentemente descartado tras varias derrotas consecutivas tempranas.
_W_MIN_DEFAULT: float = 0.05


# ============================================================
# Estado AOS (uno por proceso/corrida)
# ============================================================

class _AOSState:
    """Pesos adaptativos por operador -- Probability Matching.

    Cada vez que un operador genera una mejora del mejor global (via
    ``actualizar(op)``), su peso sube; los demas bajan. La seleccion
    usa los pesos como probabilidades.

    Los pesos se inicializan uniformes (1/N por operador) y se mantienen
    en [w_min, 1.0] para evitar la extincion de operadores prometedores
    en etapas tempranas de la corrida.
    """

    def __init__(
        self,
        operadores: tuple[str, ...] | list[str],
        alpha: float = _ALPHA_DEFAULT,
        w_min: float = _W_MIN_DEFAULT,
    ) -> None:
        self.alpha = alpha
        self.w_min = w_min
        n = len(operadores)
        # Inicializar con pesos uniformes para que ningun operador tenga
        # ventaja inicial: la primera ronda de seleccion equivale a uniform.
        self.pesos: dict[str, float] = {op: 1.0 / n for op in operadores}

    def actualizar(self, op: str | None) -> None:
        """Regla PM: reward=1 para op ganador, reward=0 para el resto.

        Actualizacion: w[o] = max(w_min, (1 - alpha)*w[o] + alpha*reward[o])
        """
        if op is None or op not in self.pesos:
            return
        for o in self.pesos:
            reward = 1.0 if o == op else 0.0
            # Promedio movil exponencial con piso w_min para evitar extincion.
            self.pesos[o] = max(
                self.w_min,
                (1.0 - self.alpha) * self.pesos[o] + self.alpha * reward,
            )

    def seleccionar(self, grupo: list[str], rng) -> str:
        """Elige UN operador del grupo con probabilidad proporcional a pesos.

        Operadores del grupo que no estan en ``self.pesos`` reciben ``w_min``.
        """
        # Construimos la lista de pesos respetando el orden del grupo.
        # max(..., w_min) garantiza pesos estrictamente positivos.
        ws = [max(self.pesos.get(op, self.w_min), self.w_min) for op in grupo]
        total = sum(ws)
        r = rng.random()
        acc = 0.0
        for op, w in zip(grupo, ws):
            acc += w / total
            if r <= acc:
                return op
        # Fallback numerico: si el redondeo deja r ligeramente por encima
        # del ultimo acumulado, devolvemos el ultimo operador.
        return grupo[-1]


# ============================================================
# Estado global del proceso (uno por worker)
# ============================================================

# Estado AOS activo en el proceso actual. Se reinicia con cada llamada a
# ``aplicar_patch_aos`` (modo paralelo: una vez por worker; modo secuencial:
# una vez por corrida).
_estado_aos: _AOSState | None = None
# Guards para evitar doble-wrapping en modo secuencial (varias corridas en el
# mismo proceso). Cada flag indica si el patch correspondiente ya esta activo.
_registrar_patched: bool = False
_gv_patched: bool = False


# ============================================================
# Selector AOS (capa 1 del experimento)
# ============================================================

def seleccionar_grupo_aos(
    rng,
    violacion,
    ops_intra,
    ops_inter,
    operadores_fallback,
    **_ignorado,
):
    """Selector BINARIO para el grupo (identico a seleccionar_grupo_strict).

    Misma firma que ``seleccionar_grupo_operadores_inter_intra``; kwargs
    extra (``alpha_inter``, ``p_inter``) se absorben en ``**_ignorado``.

    Devuelve el GRUPO COMPLETO (no un operador unico) para que
    ``generar_vecino`` pueda reintentar con otro operador del grupo si el
    elegido no es aplicable a la solucion actual (solution too small).
    El efecto AOS opera en el nivel de ``generar_vecino`` via
    ``pesos_operadores`` inyectados por ``_make_patched_gv``.

    Logica de grupo: binaria estricta.
      - ``violacion > 1e-12`` -> grupo INTER (reparacion).
      - ``violacion <= 1e-12`` -> grupo INTRA (refinamiento).
    """
    if violacion > 1e-12 and ops_inter:
        return list(ops_inter), True
    if ops_intra:
        return list(ops_intra), False
    if ops_inter:
        # Fallback: sin ops_intra pero con ops_inter (caso degenerado).
        return list(ops_inter), violacion > 1e-12
    # Ambos grupos vacios: fallback completo.
    return list(operadores_fallback), violacion > 1e-12


# ============================================================
# Funcion de patch (capa 2 del experimento)
# ============================================================

def _make_patched_gv(orig_gv):
    """Envuelve ``generar_vecino`` para inyectar pesos AOS como
    ``pesos_operadores``.

    ``generar_vecino`` elige el operador con ``rng.choices(ops,
    weights=pesos_operadores)``. Al inyectar los pesos AOS, operadores con
    mas mejoras historicas se eligen mas frecuentemente. Si el elegido no
    es aplicable (solucion demasiado pequena), el bucle interno de
    ``generar_vecino`` reintenta automaticamente: en el siguiente intento
    puede caer en otro operador del grupo segun los mismos pesos, lo que
    eventualmente resuelve el caso sin lanzar RuntimeError.
    """
    def _patched(sol, *args, operadores=None, pesos_operadores=None, **kwargs):
        import metacarp.aos_pm_20260527 as _mod
        if (pesos_operadores is None
                and _mod._estado_aos is not None
                and operadores is not None):
            pesos_operadores = [
                _mod._estado_aos.pesos.get(op, _mod._W_MIN_DEFAULT)
                for op in operadores
            ]
        return orig_gv(
            sol, *args,
            operadores=operadores,
            pesos_operadores=pesos_operadores,
            **kwargs,
        )
    return _patched


def _make_patched_gvi(orig_gvi):
    """Idem para ``generar_vecino_ids`` (ABC Simple usa representacion IDs)."""
    def _patched(sol_ids, *args, operadores=None, pesos_operadores=None, **kwargs):
        import metacarp.aos_pm_20260527 as _mod
        if (pesos_operadores is None
                and _mod._estado_aos is not None
                and operadores is not None):
            pesos_operadores = [
                _mod._estado_aos.pesos.get(op, _mod._W_MIN_DEFAULT)
                for op in operadores
            ]
        return orig_gvi(
            sol_ids, *args,
            operadores=operadores,
            pesos_operadores=pesos_operadores,
            **kwargs,
        )
    return _patched


def _make_registrar_patched(orig_registrar):
    """Fabrica el metodo patched sin capturar ``_estado_aos`` en la clausura.

    El metodo resultante siempre lee el ``_estado_aos`` DEL MODULO (global),
    no el valor que tenia cuando se creo el closure. Esto garantiza que
    multiples llamadas a ``aplicar_patch_aos`` (modo secuencial) reutilicen
    el estado fresco asignado en cada llamada.
    """
    def _patched(self, op):
        # Primero ejecutamos la logica original (incrementa contador, snapshot).
        orig_registrar(self, op)
        # Luego leemos ``_estado_aos`` del modulo en tiempo de ejecucion (no
        # en tiempo de definicion del closure). Tras una nueva llamada a
        # ``aplicar_patch_aos`` el closure ve el estado fresco recien creado.
        import metacarp.aos_pm_20260527 as _self_mod
        if _self_mod._estado_aos is not None:
            _self_mod._estado_aos.actualizar(op)
    return _patched


def aplicar_patch_aos(
    nombre_modulo_mh: str,
    operadores: tuple[str, ...] | list[str] | None = None,
    alpha: float = _ALPHA_DEFAULT,
) -> None:
    """Aplica dos patches al modulo de la MH para activar AOS.

    Debe llamarse al INICIO de cada worker (o UNA vez por corrida en modo
    secuencial) para:
    (1) Reemplazar ``seleccionar_grupo_operadores_inter_intra`` en el modulo
        de la MH por el selector AOS (``seleccionar_grupo_aos``).
    (2) Reemplazar ``ContadorOperadores.registrar_mejora`` en
        ``metacarp.metaheuristicas_utils`` por una version que tambien
        actualiza los pesos AOS.

    El estado ``_estado_aos`` se REINICIA con cada llamada a esta funcion,
    garantizando trayectorias independientes entre corridas en modo secuencial.

    El patch de ``registrar_mejora`` se aplica UNA SOLA VEZ (guardado por
    ``_registrar_patched``) para evitar doble-wrapping en modo secuencial.

    Parametros
    ----------
    nombre_modulo_mh : str
        Modulo de la MH donde reasignar el selector.
        Ej: ``"metacarp.busqueda_tabu_simple"``.
    operadores : tuple o list, opcional
        Conjunto de operadores activos. Default: ``OPERADORES_AOS_5``.
    alpha : float
        Tasa de aprendizaje del Probability Matching. Default 0.2.
    """
    global _estado_aos, _registrar_patched, _gv_patched

    ops = tuple(operadores) if operadores is not None else OPERADORES_AOS_5
    # Reiniciar estado AOS para esta corrida (pesos uniformes de nuevo).
    _estado_aos = _AOSState(ops, alpha=alpha)

    # --- Patch 1: selector de grupo dentro del modulo de la MH ---
    # Las 5 MH del proyecto importan el selector en su top-level, por lo que
    # basta con reasignar el atributo en el namespace del modulo de la MH.
    mh = importlib.import_module(nombre_modulo_mh)
    mh.seleccionar_grupo_operadores_inter_intra = seleccionar_grupo_aos

    # --- Patch 2: generar_vecino / generar_vecino_ids (solo una vez) ---
    # Inyecta pesos AOS como ``pesos_operadores`` en cada llamada a
    # generar_vecino. El grupo COMPLETO se pasa igual que antes, de modo que
    # si el operador elegido por los pesos no es aplicable, el bucle interno
    # de generar_vecino reintenta con otro operador del grupo (fallback seguro).
    if not _gv_patched:
        if hasattr(mh, "generar_vecino_ids"):
            # ABC Simple usa representacion IDs.
            mh.generar_vecino_ids = _make_patched_gvi(mh.generar_vecino_ids)
        if hasattr(mh, "generar_vecino"):
            mh.generar_vecino = _make_patched_gv(mh.generar_vecino)
        _gv_patched = True

    # --- Patch 3: registrar_mejora (solo una vez por proceso) ---
    # Envolvemos el metodo original para que tambien notifique al AOS cada
    # vez que la MH registre una mejora del mejor global.
    if not _registrar_patched:
        import metacarp.metaheuristicas_utils as mhu
        orig = mhu.ContadorOperadores.registrar_mejora
        mhu.ContadorOperadores.registrar_mejora = _make_registrar_patched(orig)
        _registrar_patched = True
