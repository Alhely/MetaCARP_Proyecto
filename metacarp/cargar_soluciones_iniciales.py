# =============================================================================
# cargar_soluciones_iniciales.py — Carga de soluciones iniciales para CARP
#
# Los metaheurísticos (Búsqueda Tabú, Abejas, Cucú, Recocido Simulado)
# necesitan una solución de partida ("solución inicial") desde la cual empezar
# a buscar mejoras. En lugar de generarla aleatoriamente en cada ejecución,
# aquí se guardan en disco soluciones preconstruidas (por ejemplo, generadas
# con una heurística voraz) para asegurar reproducibilidad.
#
# Formato esperado en disco:
#   <root>/InitialSolution/<instancia>_init_sol.pkl
#
# El contenido del pickle puede variar: lista de rutas, diccionario, etc.
# =============================================================================

# Permite anotaciones de tipo como `Path | None` en Python < 3.10
from __future__ import annotations

import os       # Para leer la variable de entorno CARPTHESIS_ROOT
import pickle   # Para deserializar archivos .pkl
from pathlib import Path  # Rutas de archivo orientadas a objetos
from typing import Any    # El tipo exacto varía según el pickle guardado

# Reutiliza la función que devuelve la carpeta raíz del paquete
from .instances import _package_dir

# Nombres públicos que este módulo expone
__all__ = [
    "ruta_solucion_inicial",
    "cargar_solucion_inicial",
    "nombres_soluciones_iniciales_disponibles",
]

# Sufijo fijo de todos los archivos de soluciones iniciales.
# Al definirlo como constante se evita repetir el literal en varios lugares
# y cualquier cambio futuro se hace en un solo punto.
_SUFFIX = "_init_sol.pkl"


# -----------------------------------------------------------------------------
# Funciones internas (prefijo _ = privadas al módulo)
# -----------------------------------------------------------------------------

def _resolve_root(root: str | os.PathLike[str] | None) -> Path:
    """Determina la carpeta raíz de datos según prioridad:

    1. Si se pasó ``root`` explícitamente, úsalo.
    2. Si existe la variable de entorno ``CARPTHESIS_ROOT``, úsala.
    3. Si la carpeta del paquete contiene ``InitialSolution``, úsala.
    4. Si un ancestro del paquete contiene ``InitialSolution`` (modo
       desarrollo con datos en el root del repo), úsalo.
    5. Como último recurso, usa la carpeta del paquete.

    Misma lógica que en cargar_grafos.py y cargar_matrices.py para coherencia.
    """
    if root is not None:
        return Path(root).expanduser().resolve()
    env = os.environ.get("CARPTHESIS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    pkg_dir = _package_dir()
    # Si la carpeta del paquete ya tiene la carpeta de soluciones iniciales,
    # usarla directamente sin escanear ancestros.
    for name in ("InitialSolution", "initialsolution", "initial_solution"):
        if (pkg_dir / name).is_dir():
            return pkg_dir
    # Modo desarrollo: buscar un ancestro que contenga la carpeta de datos.
    for ancestor in pkg_dir.parents:
        for name in ("InitialSolution", "initialsolution", "initial_solution"):
            if (ancestor / name).is_dir():
                return ancestor
    return pkg_dir


def _initial_solution_dir(data_root: Path) -> Path:
    """Localiza la subcarpeta de soluciones iniciales dentro de ``data_root``.

    Acepta tres variantes de nombre para mayor compatibilidad con distintos
    sistemas operativos o convenciones de nomenclatura usadas al crear el proyecto:
    - ``InitialSolution``
    - ``initialsolution``
    - ``initial_solution``

    Lanza ``FileNotFoundError`` si ninguna de las variantes existe en disco.
    """
    # Prueba cada nombre alternativo hasta encontrar el que existe
    for name in ("InitialSolution", "initialsolution", "initial_solution"):
        p = data_root / name
        if p.is_dir():
            return p
    raise FileNotFoundError(
        f"No existe la carpeta InitialSolution bajo {data_root}. "
        f"Añade ahí los archivos <instancia>{_SUFFIX}."
    )


# -----------------------------------------------------------------------------
# Funciones públicas
# -----------------------------------------------------------------------------

def ruta_solucion_inicial(
    nombre_instancia: str,
    *,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Devuelve la ruta absoluta al pickle de la solución inicial sin leerlo.

    Útil para verificar existencia o pasar la ruta a otra herramienta.

    El ``*`` obliga a pasar ``root`` siempre como argumento nombrado.
    """
    # Construye: <initial_solution_dir>/<nombre>_init_sol.pkl
    return _initial_solution_dir(_resolve_root(root)) / f"{nombre_instancia}{_SUFFIX}"


def cargar_solucion_inicial(
    nombre_instancia: str,
    *,
    root: str | os.PathLike[str] | None = None,
) -> Any:
    """
    Lee el pickle de la solución inicial y devuelve el objeto deserializado.

    El tipo exacto del objeto depende de cómo se guardó la solución:
    puede ser una lista de rutas, un diccionario, una estructura propia, etc.

    Parámetros
    ----------
    nombre_instancia : str
        Nombre de la instancia, por ejemplo ``"egl-e1-A"``.
    root : ruta opcional
        Carpeta raíz alternativa donde buscar ``InitialSolution/``.
    """
    # Construye la ruta completa al archivo pickle
    path = ruta_solucion_inicial(nombre_instancia, root=root)

    # Verifica que el archivo exista antes de intentar abrirlo
    if not path.is_file():
        raise FileNotFoundError(f"No existe la solución inicial: {path}")

    # Abre en modo binario ("rb") y deserializa el objeto Python con pickle
    with path.open("rb") as f:
        return pickle.load(f)


def nombres_soluciones_iniciales_disponibles(
    *,
    root: str | os.PathLike[str] | None = None,
) -> list[str]:
    """
    Lista los nombres de instancias que ya tienen solución inicial guardada.

    Escanea la carpeta ``InitialSolution/`` en busca de archivos con el sufijo
    ``_init_sol.pkl`` y devuelve solo el nombre base sin sufijo.

    Por ejemplo, si existe ``egl-e1-A_init_sol.pkl``, devuelve ``"egl-e1-A"``.

    Parámetros
    ----------
    root : ruta opcional
        Carpeta raíz alternativa donde buscar ``InitialSolution/``.
    """
    # Localiza la carpeta de soluciones iniciales
    sdir = _initial_solution_dir(_resolve_root(root))

    out: list[str] = []
    # Itera en orden alfabético sobre los archivos que coinciden con el sufijo
    for p in sorted(sdir.glob(f"*{_SUFFIX}")):
        # Recorta el sufijo para obtener el nombre de instancia limpio
        # Ejemplo: "egl-e1-A_init_sol.pkl" → "egl-e1-A"
        stem = p.name[: -len(_SUFFIX)]
        out.append(stem)
    return out
