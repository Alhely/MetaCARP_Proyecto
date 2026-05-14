# =============================================================================
# cargar_grafos.py — Carga de grafos e imágenes de instancias CARP
#
# Cada instancia CARP tiene dos representaciones visuales/computacionales
# guardadas en la carpeta Grafos/:
#
#   1. <instancia>_estatico.png  — imagen PNG del grafo para visualización.
#   2. <instancia>_gobject.gexf  — el grafo como estructura de red en formato
#      GEXF, que NetworkX puede leer y procesar (nodos, aristas, pesos, etc.).
#
# Este módulo centraliza la lógica de resolución de rutas y apertura de estos
# archivos, de modo que el resto del código solo llame a funciones de alto nivel.
# =============================================================================

# Permite anotaciones de tipo como `Path | None` en Python < 3.10
from __future__ import annotations

import os                    # Para leer la variable de entorno CARPTHESIS_ROOT
from pathlib import Path     # Rutas de archivo orientadas a objetos
from typing import Literal   # Para restringir un parámetro a valores concretos

import networkx as nx        # Librería de grafos: permite analizar redes complejas
from PIL import Image        # Pillow: librería para abrir y manipular imágenes

# Importa la función que devuelve la carpeta raíz del paquete
from .instances import _package_dir

# Pillow tiene un límite de píxeles para protegerse de imágenes maliciosas
# ("decompression bomb"). Aquí lo desactivamos porque las imágenes de grafos
# CARP son grandes pero son fuente confiable del propio proyecto.
Image.MAX_IMAGE_PIXELS = None

# Lista de nombres que este módulo exporta públicamente
__all__ = [
    "cargar_imagen_estatica",
    "cargar_objeto_gexf",
    "cargar_grafo",
    "ruta_imagen_estatica",
    "ruta_gexf",
]


# -----------------------------------------------------------------------------
# Funciones internas de resolución de rutas (prefijo _ = uso privado)
# -----------------------------------------------------------------------------

def _resolve_root(root: str | os.PathLike[str] | None) -> Path:
    """Determina la carpeta raíz de datos según prioridad:

    1. Si se pasó ``root`` explícitamente, úsalo.
    2. Si existe la variable de entorno ``CARPTHESIS_ROOT``, úsala.
    3. Si la carpeta del paquete contiene ``Grafos``, úsala.
    4. Si un ancestro del paquete contiene ``Grafos`` (modo desarrollo
       con datos en el root del repo), úsalo.
    5. Como último recurso, usa la carpeta del paquete.
    """
    if root is not None:
        return Path(root).expanduser().resolve()
    env = os.environ.get("CARPTHESIS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    pkg_dir = _package_dir()
    # Si la carpeta del paquete ya tiene Grafos/grafos, usarla directamente.
    if (pkg_dir / "Grafos").is_dir() or (pkg_dir / "grafos").is_dir():
        return pkg_dir
    # Modo desarrollo: buscar un ancestro que contenga la carpeta de datos.
    for ancestor in pkg_dir.parents:
        if (ancestor / "Grafos").is_dir() or (ancestor / "grafos").is_dir():
            return ancestor
    return pkg_dir


def _grafos_dir(data_root: Path) -> Path:
    """Localiza la subcarpeta de grafos dentro de ``data_root``.

    Acepta tanto ``Grafos`` (mayúscula inicial) como ``grafos`` (minúscula)
    para mayor compatibilidad entre sistemas de archivos.
    Lanza ``FileNotFoundError`` si ninguna de las dos existe.
    """
    # Prueba cada variante de nombre hasta encontrar la que existe
    for name in ("Grafos", "grafos"):
        p = data_root / name
        if p.is_dir():
            return p
    raise FileNotFoundError(
        f"No existe la carpeta Grafos ni grafos bajo {data_root}. "
        "Añade ahí los archivos <instancia>_estatico.png y <instancia>_gobject.gexf."
    )


# -----------------------------------------------------------------------------
# Funciones públicas de rutas (solo construyen la ruta, no abren el archivo)
# -----------------------------------------------------------------------------

def ruta_imagen_estatica(
    nombre_instancia: str,
    *,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Devuelve la ruta absoluta al PNG estático de la instancia sin abrirlo.

    Útil cuando solo se necesita saber dónde está el archivo, por ejemplo
    para pasarlo a otra herramienta o verificar que existe.

    El ``*`` en la firma obliga a que ``root`` se pase siempre como argumento
    nombrado (``root=...``), no posicional. Esto hace la interfaz más clara.
    """
    # Construye la ruta: <grafos_dir>/<nombre>_estatico.png
    return _grafos_dir(_resolve_root(root)) / f"{nombre_instancia}_estatico.png"


def ruta_gexf(
    nombre_instancia: str,
    *,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Devuelve la ruta absoluta al archivo .gexf de la instancia sin abrirlo.

    Análogo a :func:`ruta_imagen_estatica` pero para el archivo de grafo.
    """
    # Construye la ruta: <grafos_dir>/<nombre>_gobject.gexf
    return _grafos_dir(_resolve_root(root)) / f"{nombre_instancia}_gobject.gexf"


# -----------------------------------------------------------------------------
# Funciones públicas de carga (abren y devuelven el contenido del archivo)
# -----------------------------------------------------------------------------

def cargar_imagen_estatica(
    nombre_instancia: str,
    *,
    root: str | os.PathLike[str] | None = None,
    show: bool = False,
) -> Image.Image:
    """
    Carga ``<nombre_instancia>_estatico.png`` como objeto imagen de Pillow.

    Usa la misma raíz de datos que el resto del paquete (carpeta del paquete
    o ``CARPTHESIS_ROOT``).

    Parámetros
    ----------
    nombre_instancia : str
        Nombre de la instancia, por ejemplo ``"egl-e1-A"``.
    root : ruta opcional
        Carpeta raíz alternativa donde buscar ``Grafos/``.
    show : bool
        Si es ``True``, abre la imagen en el visor predeterminado del sistema
        operativo (útil para exploración interactiva en notebooks o scripts).
    """
    # Obtiene la ruta completa al archivo PNG
    path = ruta_imagen_estatica(nombre_instancia, root=root)

    # Verifica que el archivo realmente exista antes de intentar abrirlo
    if not path.is_file():
        raise FileNotFoundError(f"No existe la imagen: {path}")

    # Abre la imagen con Pillow (no la decodifica completamente hasta que se use)
    img = Image.open(path)

    # Muestra la imagen en el visor del sistema si se solicitó
    if show:
        img.show()

    return img


def cargar_objeto_gexf(
    nombre_instancia: str,
    *,
    root: str | os.PathLike[str] | None = None,
) -> nx.Graph:
    """
    Carga ``<nombre_instancia>_gobject.gexf`` y lo devuelve como grafo NetworkX.

    GEXF (Graph Exchange XML Format) es un estándar para guardar grafos con
    atributos. NetworkX puede leerlo directamente y convertirlo a su
    estructura interna de grafo, que permite calcular caminos, centralidades, etc.

    El tipo concreto del grafo devuelto depende del GEXF:
    puede ser ``nx.Graph`` (no dirigido) o ``nx.MultiGraph`` (con aristas múltiples).
    """
    # Obtiene la ruta completa al archivo GEXF
    path = ruta_gexf(nombre_instancia, root=root)

    # Verifica que el archivo exista
    if not path.is_file():
        raise FileNotFoundError(f"No existe el GEXF: {path}")

    # NetworkX lee y parsea el XML del GEXF y construye el objeto grafo
    return nx.read_gexf(path)


def cargar_grafo(
    nombre_instancia: str,
    tipo: Literal["imagen", "gexf"],
    *,
    root: str | os.PathLike[str] | None = None,
    show: bool = False,
) -> Image.Image | nx.Graph:
    """
    Función unificada: carga la imagen estática o el grafo NetworkX según ``tipo``.

    ``Literal["imagen", "gexf"]`` significa que ``tipo`` solo puede tomar
    exactamente esos dos valores de texto; cualquier otro valor lanza un error.
    Esto es una forma de documentar y validar la interfaz al mismo tiempo.

    Parámetros
    ----------
    nombre_instancia : str
        Nombre de la instancia CARP, por ejemplo ``"egl-e1-A"``.
    tipo : "imagen" | "gexf"
        Qué representación cargar:
        - ``"imagen"``: equivale a llamar :func:`cargar_imagen_estatica`.
        - ``"gexf"``: equivale a llamar :func:`cargar_objeto_gexf`.
    root : ruta opcional
        Carpeta raíz alternativa.
    show : bool
        Solo aplica cuando ``tipo=="imagen"``; muestra la imagen en pantalla.
    """
    if tipo == "imagen":
        return cargar_imagen_estatica(nombre_instancia, root=root, show=show)
    if tipo == "gexf":
        return cargar_objeto_gexf(nombre_instancia, root=root)
    # Si tipo no es ninguno de los dos valores válidos, lanza un error descriptivo
    raise ValueError(f"tipo debe ser 'imagen' o 'gexf', recibido: {tipo!r}")
