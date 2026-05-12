from __future__ import annotations

import os
import pickle
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _package_dir() -> Path:
    """Directorio raíz del paquete (carpeta que contiene este módulo)."""
    return Path(__file__).resolve().parent


def _default_root() -> Path:
    """
    Raíz de datos: por defecto la carpeta del paquete (portable entre usuarios).

    Si existe ``CARPTHESIS_ROOT``, se usa esa ruta (p. ej. tests o datos fuera del paquete).
    Los pickles se buscan en ``<root>/PickleInstances/``.
    """
    env = os.environ.get("CARPTHESIS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return _package_dir()


@dataclass
class InstanceStore(Mapping[str, Any]):
    """
    Lazy mapping from instance name -> loaded python object (from pickle).

    It discovers instances from: <root>/PickleInstances/*.pkl
    and loads each pickle on first access.
    """

    root: Path = field(default_factory=_default_root)
    cache: dict[str, Any] = field(default_factory=dict, init=False)
    _index: dict[str, Path] = field(default_factory=dict, init=False, repr=False)

    @property
    def pickle_dir(self) -> Path:
        return self.root / "PickleInstances"

    def reindex(self) -> None:
        pdir = self.pickle_dir
        if not pdir.exists():
            self._index = {}
            return

        index: dict[str, Path] = {}
        for pkl in sorted(pdir.glob("*.pkl")):
            name = pkl.stem
            index[name] = pkl
        self._index = index

    def set_root(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()
        self.cache.clear()
        self.reindex()

    def _ensure_index(self) -> None:
        if not self._index:
            self.reindex()

    def __getitem__(self, key: str) -> Any:
        if key in self.cache:
            return self.cache[key]

        self._ensure_index()
        path = self._index.get(key)
        if path is None:
            raise KeyError(
                f"Unknown instance '{key}'. Available: {', '.join(list(self.keys())[:20])}"
                + (" ..." if len(self) > 20 else "")
            )

        with path.open("rb") as f:
            obj = pickle.load(f)
        self.cache[key] = obj
        return obj

    def __iter__(self) -> Iterator[str]:
        self._ensure_index()
        return iter(self._index.keys())

    def __len__(self) -> int:
        self._ensure_index()
        return len(self._index)

    def keys(self) -> Iterable[str]:  # type: ignore[override]
        self._ensure_index()
        return self._index.keys()

    def paths(self) -> Mapping[str, Path]:
        """Return instance name -> pickle path (no loading)."""
        self._ensure_index()
        return dict(self._index)


dictionary_instances = InstanceStore()


def load_instance(name: str, *, root: str | os.PathLike[str] | None = None) -> Any:
    """
    Convenience loader.

    If root is provided, it loads from that CARP root instead of the default.
    """
    if root is None:
        return dictionary_instances[name]
    tmp = InstanceStore(Path(root).expanduser().resolve())
    return tmp[name]


def _store_for_root(root: str | os.PathLike[str] | None) -> InstanceStore:
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

    - ``name_or_all``: nombre del archivo sin ``.pkl`` (ej. ``egl-e1-A``), o
      ``"all"`` (sin distinguir mayúsculas) para cargar todas en una lista.
    - ``root``: raíz donde está ``PickleInstances/``; por defecto la carpeta del
      paquete, o la variable de entorno ``CARPTHESIS_ROOT`` si está definida.

    Con ``"all"`` devuelve una lista de diccionarios en orden de nombre
    ordenado. Cada pickle debe deserializar a un ``dict``.
    """
    store = _store_for_root(root)
    key = name_or_all.strip()
    if key.lower() == "all":
        out: list[dict[str, Any]] = []
        for name in sorted(store.keys()):
            obj = store[name]
            if not isinstance(obj, dict):
                raise TypeError(
                    f"Instancia {name!r}: se esperaba dict, obtuvo {type(obj).__name__}"
                )
            out.append(obj)
        return out

    obj = store[key]
    if not isinstance(obj, dict):
        raise TypeError(
            f"Instancia {key!r}: se esperaba dict, obtuvo {type(obj).__name__}"
        )
    return obj
