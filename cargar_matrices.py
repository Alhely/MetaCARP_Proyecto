from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

from .instances import _package_dir

__all__ = [
    "ruta_matriz_dijkstra",
    "cargar_matriz_dijkstra",
    "nombres_matrices_disponibles",
]

_SUFFIX = "_dijkstra_matrix.pkl"


def _resolve_root(root: str | os.PathLike[str] | None) -> Path:
    if root is not None:
        return Path(root).expanduser().resolve()
    env = os.environ.get("CARPTHESIS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return _package_dir()


def _matrices_dir(data_root: Path) -> Path:
    for name in ("Matrices", "matrices"):
        p = data_root / name
        if p.is_dir():
            return p
    raise FileNotFoundError(
        f"No existe la carpeta Matrices ni matrices bajo {data_root}. "
        f"Añade ahí los archivos <instancia>{_SUFFIX}."
    )


def ruta_matriz_dijkstra(
    nombre_instancia: str,
    *,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Ruta absoluta al pickle de la matriz Dijkstra (sin abrir el archivo)."""
    return _matrices_dir(_resolve_root(root)) / f"{nombre_instancia}{_SUFFIX}"


def cargar_matriz_dijkstra(
    nombre_instancia: str,
    *,
    root: str | os.PathLike[str] | None = None,
) -> Any:
    """
    Lee ``<nombre_instancia>_dijkstra_matrix.pkl`` y devuelve el objeto deserializado.

    Suele ser un ``numpy.ndarray`` de distancias; el tipo exacto depende de cómo
    guardaste el pickle.
    """
    path = ruta_matriz_dijkstra(nombre_instancia, root=root)
    if not path.is_file():
        raise FileNotFoundError(f"No existe la matriz: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def nombres_matrices_disponibles(
    *,
    root: str | os.PathLike[str] | None = None,
) -> list[str]:
    """
    Lista los nombres de instancia que tienen matriz (stems sin ``_dijkstra_matrix``).
    """
    mdir = _matrices_dir(_resolve_root(root))
    out: list[str] = []
    for p in sorted(mdir.glob(f"*{_SUFFIX}")):
        stem = p.name[: -len(_SUFFIX)]
        out.append(stem)
    return out
