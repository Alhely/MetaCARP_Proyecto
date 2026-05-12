# =============================================================================
# instances.py — Gestión de instancias CARP (carga perezosa desde pickle)
#
# Una "instancia" en CARP es un problema concreto: un grafo con nodos, aristas,
# demandas, capacidad del vehículo, etc. Aquí se gestionan como archivos .pkl
# (pickle: formato binario de Python para serializar objetos).
#
# Conceptos clave usados:
#   - @dataclass: decorador que genera automáticamente __init__, __repr__, etc.
#   - Mapping: tipo abstracto de Python que se comporta como un diccionario
#     (se puede acceder con corchetes []). Al heredar de él, InstanceStore
#     se convierte en un "diccionario personalizado".
#   - Carga perezosa (lazy loading): los pickles no se leen del disco hasta que
#     se necesiten; esto ahorra memoria y tiempo de arranque.
# =============================================================================

# `from __future__ import annotations` permite escribir tipos como `Path | None`
# en versiones de Python < 3.10 (sintaxis futura).
from __future__ import annotations

import os       # Para leer variables de entorno (os.environ.get)
import pickle   # Para deserializar archivos .pkl (formato binario de Python)
from collections.abc import Iterable, Iterator, Mapping  # Tipos abstractos de colecciones
from dataclasses import dataclass, field  # Herramientas para crear clases de datos
from pathlib import Path  # Manejo de rutas de archivo de forma orientada a objetos
from typing import Any    # `Any` indica que el tipo puede ser cualquier cosa


# -----------------------------------------------------------------------------
# Funciones auxiliares de rutas
# -----------------------------------------------------------------------------

def _package_dir() -> Path:
    """Directorio raíz del paquete (carpeta que contiene este módulo).

    `__file__` es una variable especial de Python que contiene la ruta de
    este mismo archivo. `.resolve().parent` sube un nivel para obtener la
    carpeta que lo contiene.
    """
    return Path(__file__).resolve().parent


def _default_root() -> Path:
    """
    Raíz de datos: por defecto la carpeta del paquete (portable entre usuarios).

    Si existe la variable de entorno ``CARPTHESIS_ROOT``, se usa esa ruta en
    su lugar (útil para tests o para datos guardados fuera del paquete).
    Los pickles de instancias se buscan en ``<root>/PickleInstances/``.
    """
    # Intenta leer la variable de entorno CARPTHESIS_ROOT
    env = os.environ.get("CARPTHESIS_ROOT")
    if env:
        # expanduser() resuelve "~" (home del usuario), resolve() convierte a ruta absoluta
        return Path(env).expanduser().resolve()
    # Si no hay variable de entorno, usa la carpeta donde está este archivo
    return _package_dir()


# -----------------------------------------------------------------------------
# Clase principal: InstanceStore
#
# CONCEPTO OOP: Esta clase hereda de `Mapping[str, Any]`, que es una clase
# abstracta de Python que define el contrato de un "diccionario de solo lectura".
# Al heredar de Mapping, InstanceStore puede usarse exactamente igual que un
# diccionario normal: store["egl-e1-A"] carga esa instancia.
#
# CONCEPTO @dataclass: el decorador @dataclass analiza los atributos declarados
# con anotaciones de tipo (root, cache, _index) y genera automáticamente el
# método __init__ para inicializarlos. Sin @dataclass habría que escribir
# def __init__(self, root=...): self.root = root etc. manualmente.
# -----------------------------------------------------------------------------

@dataclass
class InstanceStore(Mapping[str, Any]):
    """
    Mapa perezoso: nombre de instancia -> objeto Python cargado desde pickle.

    Descubre instancias buscando archivos ``<root>/PickleInstances/*.pkl``
    y carga cada pickle solo la primera vez que se accede a él.

    Hereda de ``Mapping`` para comportarse como un diccionario de solo lectura:
    se puede usar ``store["nombre"]``, ``len(store)``, ``for k in store``, etc.
    """

    # Atributo `root`: carpeta raíz donde se buscan los datos.
    # `field(default_factory=_default_root)` significa: si no se proporciona
    # root al crear el objeto, llama a _default_root() para obtener el valor.
    root: Path = field(default_factory=_default_root)

    # Atributo `cache`: diccionario en memoria que guarda instancias ya cargadas.
    # `init=False` significa que NO aparece en el __init__; Python lo inicializa
    # automáticamente con un dict vacío. Evita repetir la lectura del disco.
    cache: dict[str, Any] = field(default_factory=dict, init=False)

    # Atributo `_index`: mapea nombre de instancia -> ruta del archivo .pkl.
    # `repr=False` lo oculta al imprimir el objeto (por legibilidad).
    # El guion bajo inicial indica que es "privado" (convención de Python).
    _index: dict[str, Path] = field(default_factory=dict, init=False, repr=False)

    # -------------------------------------------------------------------------
    # Propiedad: pickle_dir
    #
    # CONCEPTO @property: convierte un método en un "atributo calculado".
    # En lugar de llamarlo como store.pickle_dir(), se accede como store.pickle_dir
    # (sin paréntesis). Aquí simplemente construye la ruta a la subcarpeta
    # PickleInstances dentro de root.
    # -------------------------------------------------------------------------
    @property
    def pickle_dir(self) -> Path:
        """Ruta a la carpeta que contiene los archivos .pkl de instancias."""
        return self.root / "PickleInstances"

    def reindex(self) -> None:
        """Escanea ``pickle_dir`` y reconstruye el índice nombre -> ruta.

        Se llama automáticamente la primera vez que se necesita el índice, o
        manualmente si se añaden nuevos archivos a la carpeta en tiempo de ejecución.
        """
        pdir = self.pickle_dir
        # Si la carpeta no existe, el índice queda vacío (no es un error)
        if not pdir.exists():
            self._index = {}
            return

        # Recorre todos los .pkl en orden alfabético y los indexa por su nombre (stem)
        index: dict[str, Path] = {}
        for pkl in sorted(pdir.glob("*.pkl")):
            name = pkl.stem  # stem: nombre del archivo sin extensión (ej. "egl-e1-A")
            index[name] = pkl
        self._index = index

    def set_root(self, root: str | os.PathLike[str]) -> None:
        """Cambia la carpeta raíz de datos y limpia el caché y el índice.

        Útil cuando se quiere apuntar a un directorio de datos diferente sin
        crear un nuevo InstanceStore.
        """
        self.root = Path(root).expanduser().resolve()
        self.cache.clear()   # Descarta instancias ya cargadas del viejo root
        self.reindex()       # Reconstruye el índice con el nuevo root

    def _ensure_index(self) -> None:
        """Construye el índice si aún no se ha hecho (inicialización perezosa).

        Se llama antes de cualquier operación que necesite saber qué instancias
        existen, evitando escanear el disco más de una vez.
        """
        if not self._index:
            self.reindex()

    # -------------------------------------------------------------------------
    # Métodos especiales requeridos por Mapping
    #
    # Para que InstanceStore funcione como un diccionario, Mapping exige
    # implementar tres métodos: __getitem__, __iter__ y __len__.
    # -------------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        """Devuelve la instancia con nombre ``key``, cargándola del disco si es necesario.

        Implementa el acceso por corchetes: ``store["egl-e1-A"]``.
        Primero busca en el caché en memoria; si no está, lee el .pkl del disco.
        """
        # Si ya fue cargada antes, devuélvela directamente del caché
        if key in self.cache:
            return self.cache[key]

        # Asegura que el índice esté construido
        self._ensure_index()

        # Busca la ruta del .pkl; lanza KeyError si el nombre no existe
        path = self._index.get(key)
        if path is None:
            raise KeyError(
                f"Unknown instance '{key}'. Available: {', '.join(list(self.keys())[:20])}"
                + (" ..." if len(self) > 20 else "")
            )

        # Lee el archivo binario y deserializa el objeto Python
        with path.open("rb") as f:
            obj = pickle.load(f)

        # Guarda en caché para futuras consultas (evita leer el disco de nuevo)
        self.cache[key] = obj
        return obj

    def __iter__(self) -> Iterator[str]:
        """Permite iterar sobre los nombres de instancias: ``for name in store``.

        Requerido por Mapping para implementar la interfaz de diccionario.
        """
        self._ensure_index()
        return iter(self._index.keys())

    def __len__(self) -> int:
        """Devuelve cuántas instancias hay disponibles: ``len(store)``.

        Requerido por Mapping.
        """
        self._ensure_index()
        return len(self._index)

    def keys(self) -> Iterable[str]:  # type: ignore[override]
        """Devuelve los nombres de todas las instancias disponibles.

        ``# type: ignore[override]`` silencia una advertencia de mypy porque el
        tipo de retorno difiere levemente de la firma base de Mapping.
        """
        self._ensure_index()
        return self._index.keys()

    def paths(self) -> Mapping[str, Path]:
        """Devuelve un diccionario nombre -> ruta al .pkl (sin cargar nada).

        Útil para inspeccionar qué archivos existen sin deserializarlos.
        """
        self._ensure_index()
        return dict(self._index)


# -----------------------------------------------------------------------------
# Instancia global compartida
#
# Se crea UN SOLO InstanceStore que usa la raíz por defecto. Todo el código
# del paquete que no especifica un root distinto comparte este objeto,
# aprovechando así el caché en memoria.
# -----------------------------------------------------------------------------
dictionary_instances = InstanceStore()


# -----------------------------------------------------------------------------
# Funciones de conveniencia
# -----------------------------------------------------------------------------

def load_instance(name: str, *, root: str | os.PathLike[str] | None = None) -> Any:
    """
    Carga una instancia CARP por nombre desde su archivo pickle.

    Si ``root`` es ``None``, usa la instancia global ``dictionary_instances``
    (y su caché). Si se proporciona ``root``, crea un ``InstanceStore``
    temporal apuntando a ese directorio.

    Parámetros
    ----------
    name : str
        Nombre de la instancia, por ejemplo ``"egl-e1-A"``.
    root : ruta opcional
        Carpeta raíz alternativa donde buscar ``PickleInstances/``.
    """
    if root is None:
        # Usa el store global para aprovechar el caché compartido
        return dictionary_instances[name]
    # Store temporal para un root distinto (sin contaminar el caché global)
    tmp = InstanceStore(Path(root).expanduser().resolve())
    return tmp[name]


def _store_for_root(root: str | os.PathLike[str] | None) -> InstanceStore:
    """Función interna: devuelve el store global o uno nuevo según ``root``.

    Centraliza la lógica de selección de store para evitar repetirla en
    cada función pública.
    """
    if root is None:
        return dictionary_instances
    return InstanceStore(Path(root).expanduser().resolve())


def load_instances(
    name_or_all: str,
    *,
    root: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """
    Carga una instancia desde ``<root>/PickleInstances/<nombre>.pkl`` o todas.

    Parámetros
    ----------
    name_or_all : str
        Nombre del archivo sin ``.pkl`` (ej. ``"egl-e1-A"``), o la cadena
        ``"all"`` (sin distinguir mayúsculas) para cargar todas las instancias.
    root : ruta opcional
        Carpeta raíz alternativa; por defecto la carpeta del paquete o la
        variable de entorno ``CARPTHESIS_ROOT``.

    Retorna
    -------
    dict
        Cuando se pide una instancia concreta: el diccionario de esa instancia.
    list[dict]
        Cuando se pide ``"all"``: lista de diccionarios en orden alfabético.
        Cada pickle debe deserializar a un ``dict``.
    """
    # Obtiene el store adecuado (global o temporal según root)
    store = _store_for_root(root)

    # Limpia espacios accidentales al inicio/fin del nombre
    key = name_or_all.strip()

    if key.lower() == "all":
        # Modo "cargar todo": itera sobre todos los nombres en orden alfabético
        out: list[dict[str, Any]] = []
        for name in sorted(store.keys()):
            obj = store[name]
            # Valida que cada pickle sea un dict (el formato esperado de instancia CARP)
            if not isinstance(obj, dict):
                raise TypeError(
                    f"Instancia {name!r}: se esperaba dict, obtuvo {type(obj).__name__}"
                )
            out.append(obj)
        return out

    # Modo "instancia única": carga y valida solo la instancia pedida
    obj = store[key]
    if not isinstance(obj, dict):
        raise TypeError(
            f"Instancia {key!r}: se esperaba dict, obtuvo {type(obj).__name__}"
        )
    return obj
