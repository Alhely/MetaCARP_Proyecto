from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import networkx as nx
from PIL import Image

from .instances import _package_dir

# Imágenes estáticas del proyecto (origen confiable): desactiva el tope anti
# "decompression bomb" de Pillow en PNG muy grandes.
Image.MAX_IMAGE_PIXELS = None

__all__ = [
    "cargar_imagen_estatica",
    "cargar_objeto_gexf",
    "cargar_grafo",
    "ruta_imagen_estatica",
    "ruta_gexf",
]


def _resolve_root(root: str | os.PathLike[str] | None) -> Path:
    if root is not None:
        return Path(root).expanduser().resolve()
    env = os.environ.get("CARPTHESIS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return _package_dir()


def _grafos_dir(data_root: Path) -> Path:
    for name in ("Grafos", "grafos"):
        p = data_root / name
        if p.is_dir():
            return p
    raise FileNotFoundError(
        f"No existe la carpeta Grafos ni grafos bajo {data_root}. "
        "Añade ahí los archivos <instancia>_estatico.png y <instancia>_gobject.gexf."
    )


def ruta_imagen_estatica(
    nombre_instancia: str,
    *,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Ruta absoluta al PNG estático de la instancia (sin abrir el archivo)."""
    return _grafos_dir(_resolve_root(root)) / f"{nombre_instancia}_estatico.png"


def ruta_gexf(
    nombre_instancia: str,
    *,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Ruta absoluta al .gexf de la instancia (sin abrir el archivo)."""
    return _grafos_dir(_resolve_root(root)) / f"{nombre_instancia}_gobject.gexf"


def cargar_imagen_estatica(
    nombre_instancia: str,
    *,
    root: str | os.PathLike[str] | None = None,
    show: bool = False,
) -> Image.Image:
    """
    Carga ``<nombre_instancia>_estatico.png`` como imagen PIL.

    Usa la misma raíz de datos que el resto del paquete (carpeta del paquete
    o ``CARPTHESIS_ROOT``).

    Si ``show=True``, abre la imagen en el visor predeterminado del sistema
    (usa :meth:`PIL.Image.Image.show`).
    """
    path = ruta_imagen_estatica(nombre_instancia, root=root)
    if not path.is_file():
        raise FileNotFoundError(f"No existe la imagen: {path}")
    img = Image.open(path)
    if show:
        img.show()
    return img


def cargar_objeto_gexf(
    nombre_instancia: str,
    *,
    root: str | os.PathLike[str] | None = None,
) -> nx.Graph:
    """
    Carga ``<nombre_instancia>_gobject.gexf`` con NetworkX.

    El tipo concreto depende del GEXF (p. ej. ``Graph`` o ``MultiGraph``).
    """
    path = ruta_gexf(nombre_instancia, root=root)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el GEXF: {path}")
    return nx.read_gexf(path)


def cargar_grafo(
    nombre_instancia: str,
    tipo: Literal["imagen", "gexf"],
    *,
    root: str | os.PathLike[str] | None = None,
    show: bool = False,
) -> Image.Image | nx.Graph:
    """
    Carga la imagen estática o el grafo según ``tipo``.

    - ``"imagen"``: mismo resultado que :func:`cargar_imagen_estatica`.
    - ``"gexf"``: mismo resultado que :func:`cargar_objeto_gexf`.

    ``show`` solo aplica cuando ``tipo=="imagen"`` (véase :func:`cargar_imagen_estatica`).
    """
    if tipo == "imagen":
        return cargar_imagen_estatica(nombre_instancia, root=root, show=show)
    if tipo == "gexf":
        return cargar_objeto_gexf(nombre_instancia, root=root)
    raise ValueError(f"tipo debe ser 'imagen' o 'gexf', recibido: {tipo!r}")
