from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

from .instances import _package_dir

__all__ = [
    "ruta_solucion_inicial",
    "cargar_solucion_inicial",
    "nombres_soluciones_iniciales_disponibles",
]

_SUFFIX = "_init_sol.pkl"


def _resolve_root(root: str | os.PathLike[str] | None) -> Path:
    if root is not None:
        return Path(root).expanduser().resolve()
    env = os.environ.get("CARPTHESIS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return _package_dir()


def _initial_solution_dir(data_root: Path) -> Path:
    for name in ("InitialSolution", "initialsolution", "initial_solution"):
        p = data_root / name
        if p.is_dir():
            return p
    raise FileNotFoundError(
        f"No existe la carpeta InitialSolution bajo {data_root}. "
        f"Añade ahí los archivos <instancia>{_SUFFIX}."
    )


def ruta_solucion_inicial(
    nombre_instancia: str,
    *,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Ruta absoluta al pickle de la solución inicial (sin abrir el archivo)."""
    return _initial_solution_dir(_resolve_root(root)) / f"{nombre_instancia}{_SUFFIX}"


def cargar_solucion_inicial(
    nombre_instancia: str,
    *,
    root: str | os.PathLike[str] | None = None,
) -> Any:
    """
    Lee ``<nombre_instancia>_init_sol.pkl`` y devuelve el objeto deserializado.

    El tipo exacto depende de cómo guardaste la solución manual (lista, dict,
    estructura propia, etc.).
    """
    path = ruta_solucion_inicial(nombre_instancia, root=root)
    if not path.is_file():
        raise FileNotFoundError(f"No existe la solución inicial: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def nombres_soluciones_iniciales_disponibles(
    *,
    root: str | os.PathLike[str] | None = None,
) -> list[str]:
    """
    Lista los nombres de instancia que tienen solución inicial (stem sin ``_init_sol``).
    """
    sdir = _initial_solution_dir(_resolve_root(root))
    out: list[str] = []
    for p in sorted(sdir.glob(f"*{_SUFFIX}")):
        stem = p.name[: -len(_SUFFIX)]
        out.append(stem)
    return out
