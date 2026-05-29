"""
Monkey-patch para reemplazar la solucion inicial cargada del pickle por una
construccion Path-Scanning (Golden, DeArmon, Baker 1983).

Diagnostico (Seccion 14 del notebook): la solucion inicial pickle es la
causa raiz del gap residual ~30% en las 5 MH bajo budget x25. El pickle
contiene una construccion aleatoria con gap inicial medio ~20% (medido
con orientacion greedy) y hasta ~87% (medido segun orientacion del
pickle). Las MH cierran solo el ~40% del gap inicial.

Solucion: sustituir ``cargar_solucion_inicial`` por un wrapper que
construye dinamicamente la inicial con Path-Scanning mejor-de-5.

Resultado esperado: gap inicial de ~9% (medido con orientacion greedy),
del que las MH deberian cerrar mas de la mitad → gap final <5-6%.

Uso (en cada worker, antes de importar la MH):

    from metacarp.warmstart_20260529 import aplicar_patch_warmstart_ps
    aplicar_patch_warmstart_ps(root="/ruta/al/proyecto")
    from metacarp import busqueda_tabu_simple_desde_instancia
    ...

El patch modifica ``metacarp.cargar_soluciones_iniciales.cargar_solucion_inicial``.
Como las MH importan esta funcion desde ese modulo, basta con reemplazarla
ahi para que TODAS las MH (SA, TS, RTS, ABC, Cuckoo) usen Path-Scanning.
"""
from __future__ import annotations

import os
from typing import Any


def aplicar_patch_warmstart_ps(root: str | os.PathLike[str] | None = None) -> None:
    """Sustituye cargar_solucion_inicial por la construccion Path-Scanning.

    El nuevo wrapper:
      1. Recibe ``nombre_instancia`` y ``root`` (la firma original).
      2. Carga los datos de la instancia y la matriz Dijkstra.
      3. Devuelve la solucion construida por Path-Scanning mejor-de-5.

    Parametros
    ----------
    root : str | PathLike | None
        Ruta raiz por defecto para cargar instancia y matriz cuando el
        llamador no la especifica. Si es ``None``, se usa el comportamiento
        por defecto de ``_resolve_root`` (envvar o paquete).
    """
    # Importacion diferida: solo necesitamos modificar el simbolo en el
    # modulo de carga; las MH no se ven afectadas estructuralmente.
    import metacarp.cargar_soluciones_iniciales as _csi
    from metacarp.path_scanning import path_scanning
    from metacarp.instances import load_instances
    from metacarp.cargar_matrices import cargar_matriz_dijkstra

    # Capturamos la referencia original por si alguna parte del codigo
    # necesita volver a leer el pickle (no es el caso, pero lo dejamos
    # accesible para diagnostico).
    _csi._cargar_solucion_inicial_orig = _csi.cargar_solucion_inicial  # type: ignore[attr-defined]

    def _cargar_path_scanning(
        nombre_instancia: str,
        *,
        root: Any = root,  # type: ignore[assignment]
    ) -> list[list[str]]:
        """Construye dinamicamente la solucion inicial con Path-Scanning."""
        data = load_instances(nombre_instancia, root=root)
        M    = cargar_matriz_dijkstra(nombre_instancia, root=root)
        return path_scanning(data, M)

    # Reasignamos en el modulo: las MH llaman a ``cargar_solucion_inicial``
    # como ``from metacarp.cargar_soluciones_iniciales import cargar_solucion_inicial``,
    # pero cada wrapper ``_desde_instancia`` la importa por nombre cualificado
    # ``from metacarp.X import cargar_solucion_inicial as ...`` o accede
    # via el modulo, por lo que es CRITICO parchear el simbolo ANTES de la
    # primera importacion del wrapper MH. El patron en los scripts del
    # experimento es: (1) llamar a este patch, (2) recien entonces hacer el
    # ``from metacarp import busqueda_X_desde_instancia``.
    _csi.cargar_solucion_inicial = _cargar_path_scanning

    # Algunas MH importan el simbolo directamente al modulo MH; debemos
    # reescribirlo tambien en cada MH ya cargada para que la siguiente
    # llamada lo use. Si la MH aun no esta cargada, el reemplazo del
    # paso anterior basta (cuando importe el simbolo, lo tomara ya parcheado).
    import sys
    for mod_name in [
        "metacarp.recocido_simulado",
        "metacarp.busqueda_tabu_simple",
        "metacarp.busqueda_tabu_reactiva",
        "metacarp.abejas_simple",
        "metacarp.cuckoo_search",
    ]:
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "cargar_solucion_inicial"):
            mod.cargar_solucion_inicial = _cargar_path_scanning  # type: ignore[attr-defined]
