# =============================================================================
# cargar_matrices.py — Carga de matrices de distancias mínimas (Dijkstra)
#
# En CARP necesitamos conocer la distancia más corta entre cualquier par de
# nodos del grafo para calcular los desplazamientos ("deadheading") de los
# vehículos. Calcular Dijkstra en tiempo real es costoso, por lo que se
# precalcula la matriz completa y se guarda en disco como archivo pickle.
#
# Formato esperado en disco:
#   <root>/Matrices/<instancia>_dijkstra_matrix.pkl
#
# El contenido del pickle suele ser un numpy.ndarray de forma (N, N) donde
# matrix[i][j] es la distancia mínima del nodo i al nodo j.
# =============================================================================

# Permite anotaciones de tipo como `Path | None` en Python < 3.10
from __future__ import annotations

import os       # Para leer la variable de entorno CARPTHESIS_ROOT
import pickle   # Para deserializar archivos .pkl
from pathlib import Path  # Rutas de archivo orientadas a objetos
from typing import Any    # El tipo de retorno exacto varía según el pickle

# Reutiliza la función que determina la carpeta raíz del paquete
from .instances import _package_dir

# Lista de nombres que este módulo expone al exterior
__all__ = [
    "ruta_matriz_dijkstra",
    "cargar_matriz_dijkstra",
    "nombres_matrices_disponibles",
]

# Sufijo fijo que tienen todos los archivos de matrices de Dijkstra.
# Definirlo como constante evita repetir el string literal en múltiples lugares.
_SUFFIX = "_dijkstra_matrix.pkl"


# -----------------------------------------------------------------------------
# Funciones internas (prefijo _ = uso privado al módulo)
# -----------------------------------------------------------------------------

def _resolve_root(root: str | os.PathLike[str] | None) -> Path:
    """Determina la carpeta raíz de datos según prioridad:

    1. Si se pasó ``root`` explícitamente, úsalo.
    2. Si existe la variable de entorno ``CARPTHESIS_ROOT``, úsala.
    3. Si la carpeta del paquete contiene ``Matrices``, úsala.
    4. Si un ancestro del paquete contiene ``Matrices`` (modo desarrollo),
       úsalo.
    5. Como último recurso, usa la carpeta del paquete.

    Lógica idéntica a la de cargar_grafos.py para consistencia entre módulos.
    """
    if root is not None:
        return Path(root).expanduser().resolve()
    env = os.environ.get("CARPTHESIS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    pkg_dir = _package_dir()
    # Si la carpeta del paquete ya tiene Matrices/matrices, usarla directamente.
    if (pkg_dir / "Matrices").is_dir() or (pkg_dir / "matrices").is_dir():
        return pkg_dir
    # Modo desarrollo: buscar un ancestro que contenga la carpeta de datos.
    for ancestor in pkg_dir.parents:
        if (ancestor / "Matrices").is_dir() or (ancestor / "matrices").is_dir():
            return ancestor
    return pkg_dir


def _matrices_dir(data_root: Path) -> Path:
    """Localiza la subcarpeta de matrices dentro de ``data_root``.

    Acepta tanto ``Matrices`` (mayúscula) como ``matrices`` (minúscula)
    para compatibilidad con distintos sistemas operativos.
    Lanza ``FileNotFoundError`` si ninguna existe.
    """
    # Prueba cada variante de nombre hasta encontrar la que existe en disco
    for name in ("Matrices", "matrices"):
        p = data_root / name
        if p.is_dir():
            return p
    raise FileNotFoundError(
        f"No existe la carpeta Matrices ni matrices bajo {data_root}. "
        f"Añade ahí los archivos <instancia>{_SUFFIX}."
    )


# -----------------------------------------------------------------------------
# Funciones públicas
# -----------------------------------------------------------------------------

def ruta_matriz_dijkstra(
    nombre_instancia: str,
    *,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Devuelve la ruta absoluta al pickle de la matriz Dijkstra sin leerlo.

    Útil para verificar si el archivo existe o para pasárselo a otra herramienta.

    El ``*`` obliga a que ``root`` se use siempre como argumento nombrado.
    """
    # Construye: <matrices_dir>/<nombre>_dijkstra_matrix.pkl
    return _matrices_dir(_resolve_root(root)) / f"{nombre_instancia}{_SUFFIX}"


def cargar_matriz_dijkstra(
    nombre_instancia: str,
    *,
    root: str | os.PathLike[str] | None = None,
) -> Any:
    """
    Lee el pickle de la matriz Dijkstra y devuelve el objeto deserializado.

    El objeto suele ser un ``numpy.ndarray`` de forma (N, N) con las distancias
    mínimas entre pares de nodos; el tipo exacto depende de cómo se guardó.

    Parámetros
    ----------
    nombre_instancia : str
        Nombre de la instancia, por ejemplo ``"egl-e1-A"``.
    root : ruta opcional
        Carpeta raíz alternativa donde buscar ``Matrices/``.
    """
    # Obtiene la ruta completa al archivo pickle
    path = ruta_matriz_dijkstra(nombre_instancia, root=root)

    # Verifica que el archivo exista antes de intentar abrirlo
    if not path.is_file():
        raise FileNotFoundError(f"No existe la matriz: {path}")

    # Abre el archivo en modo binario ("rb") y deserializa el objeto Python
    with path.open("rb") as f:
        return pickle.load(f)


def nombres_matrices_disponibles(
    *,
    root: str | os.PathLike[str] | None = None,
) -> list[str]:
    """
    Lista los nombres de instancias que ya tienen su matriz Dijkstra guardada.

    Escanea la carpeta ``Matrices/`` buscando archivos con el sufijo esperado
    y devuelve solo el nombre base (sin el sufijo ``_dijkstra_matrix``).

    Por ejemplo, si existe ``egl-e1-A_dijkstra_matrix.pkl``, devuelve ``"egl-e1-A"``.

    Útil para saber qué instancias están listas para ejecutar los metaheurísticos
    sin tener que recalcular la matriz desde cero.
    """
    # Localiza la carpeta de matrices
    mdir = _matrices_dir(_resolve_root(root))

    out: list[str] = []
    # Itera en orden alfabético sobre los archivos que coinciden con el sufijo
    for p in sorted(mdir.glob(f"*{_SUFFIX}")):
        # Recorta el sufijo del nombre del archivo para obtener solo el nombre de instancia
        # Ejemplo: "egl-e1-A_dijkstra_matrix.pkl" → "egl-e1-A"
        stem = p.name[: -len(_SUFFIX)]
        out.append(stem)
    return out
