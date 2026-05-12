from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Hashable, Mapping, Sequence

import numpy as np

from .solucion_formato import (
    construir_mapa_tareas_por_etiqueta,
    etiquetas_tareas_requeridas,
    normalizar_rutas_etiquetas,
)

__all__ = [
    "FeasibilityDetails",
    "FeasibilityResult",
    "verificar_factibilidad",
    "verificar_factibilidad_desde_instancia",
]


def _aplicar_backend_gpu_placeholder(usar_gpu: bool) -> tuple[str, str]:
    """Devuelve (backend_solicitado, backend_real) para trazabilidad futura."""
    if not usar_gpu:
        return "cpu", "cpu"
    # Placeholder: aún no existe backend GPU real para factibilidad.
    return "gpu", "cpu"


def _dist(matriz: Any, a: int, b: int) -> float:
    """Distancia entre nodos ``a`` y ``b``; ``inf`` si no hay camino o dato ausente."""
    inf = float("inf")
    try:
        if isinstance(matriz, dict):
            row = matriz.get(a)
            if row is None:
                return inf
            if isinstance(row, dict):
                if b not in row:
                    return inf
                val = row[b]
            else:
                return inf
        else:
            arr = np.asarray(matriz)
            if arr.ndim != 2:
                return inf
            n, m = arr.shape
            if 0 <= a < n and 0 <= b < m:
                val = arr[a, b]
            elif 1 <= a <= n and 1 <= b <= m and (a < n or a == n):
                val = arr[a - 1, b - 1]
            else:
                return inf
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return inf
        return v
    except (KeyError, TypeError, ValueError, IndexError):
        return inf


def _hay_camino_entre_tareas(
    matriz: Any,
    u_ant: int,
    v_ant: int,
    u_act: int,
    v_act: int,
) -> bool:
    """¿Existe camino entre algún extremo de la tarea anterior y alguno de la actual?"""
    for p in (u_ant, v_ant):
        for q in (u_act, v_act):
            if _dist(matriz, p, q) != float("inf"):
                return True
    return False


def _hay_camino_a_deposito(matriz: Any, u: int, v: int, deposito: int) -> bool:
    return _dist(matriz, u, deposito) != float("inf") or _dist(matriz, v, deposito) != float(
        "inf"
    )


def _verificar_ruta(
    ruta: Sequence[Hashable],
    info_tareas: Mapping[Any, dict[str, Any]],
    data: Mapping[str, Any],
    matriz: Any,
    indice_ruta: int,
) -> tuple[list[str], list[str], list[str]]:
    """
    Devuelve (fallos_c2, fallos_c3, fallos_c5).

    C2: solo entre **dos tareas consecutivas** (no depósito).
    C5: depósito ↔ primera tarea y última tarea ↔ depósito.
    """
    fallos_c2: list[str] = []
    fallos_c3: list[str] = []
    fallos_c5: list[str] = []

    if not ruta:
        return fallos_c2, fallos_c3, fallos_c5

    capacidad_max = float(data.get("CAPACIDAD", 0) or 0)
    deposito = int(data.get("DEPOSITO", 1))

    demanda_total = 0.0
    nodos_anteriores: tuple[int, int] | None = None
    capacidad_rota = False

    for paso, id_tarea in enumerate(ruta):
        tarea = info_tareas.get(id_tarea)
        if not tarea:
            fallos_c2.append(
                f"Ruta {indice_ruta}, posición {paso}: tarea {id_tarea!r} no existe en los datos de la instancia."
            )
            return fallos_c2, fallos_c3, fallos_c5

        nodos = tarea.get("nodos")
        if not nodos or len(nodos) != 2:
            fallos_c2.append(
                f"Ruta {indice_ruta}, tarea {id_tarea!r}: falta par de nodos en los datos."
            )
            return fallos_c2, fallos_c3, fallos_c5

        u_act, v_act = int(nodos[0]), int(nodos[1])
        dem = float(tarea.get("demanda", 0) or 0)
        demanda_total += dem

        if demanda_total > capacidad_max and not capacidad_rota:
            capacidad_rota = True
            fallos_c3.append(
                f"Ruta {indice_ruta}: demanda acumulada {demanda_total:.4g} supera "
                f"CAPACIDAD {capacidad_max:.4g} (tras la tarea {id_tarea!r}, paso {paso})."
            )

        if paso == 0:
            if not _hay_camino_entre_tareas(matriz, deposito, deposito, u_act, v_act):
                fallos_c5.append(
                    f"Ruta {indice_ruta}: desde el depósito {deposito} no hay camino hacia la "
                    f"primera tarea {id_tarea!r} (nodos {u_act},{v_act})."
                )
        else:
            u_ant, v_ant = nodos_anteriores  # type: ignore[assignment]
            if not _hay_camino_entre_tareas(matriz, u_ant, v_ant, u_act, v_act):
                fallos_c2.append(
                    f"Ruta {indice_ruta}, entre la tarea previa ({u_ant},{v_ant}) y la tarea {id_tarea!r} "
                    f"({u_act},{v_act}): no hay camino en la matriz Dijkstra."
                )

        nodos_anteriores = (u_act, v_act)

    assert nodos_anteriores is not None
    u_ant, v_ant = nodos_anteriores
    if not _hay_camino_a_deposito(matriz, u_ant, v_ant, deposito):
        fallos_c5.append(
            f"Ruta {indice_ruta}: desde el último servicio ({u_ant},{v_ant}) no hay camino al depósito {deposito}."
        )

    return fallos_c2, fallos_c3, fallos_c5


@dataclass
class FeasibilityDetails:
    """
    Detalle por condición. ``None`` o lista vacía indica que esa condición se cumple.

    - **c1**: cobertura de tareas requeridas y unicidad global de tareas.
    - **c2**: conectividad entre tareas consecutivas en cada ruta.
    - **c3**: capacidad por ruta.
    - **c4**: número de rutas no vacías vs ``VEHICULOS``.
    - **c5**: conectividad depósito ↔ primera y última tarea de cada ruta no vacía.
    """

    c1_tareas_requeridas: list[str] = field(default_factory=list)
    c2_consecutivas: list[str] = field(default_factory=list)
    c3_capacidad: list[str] = field(default_factory=list)
    c4_vehiculos: list[str] = field(default_factory=list)
    c5_deposito_extremos: list[str] = field(default_factory=list)

    def resumen(self) -> str:
        bloques: list[str] = []
        if self.c1_tareas_requeridas:
            bloques.append(
                "C1 — Tareas requeridas:\n" + "\n".join(f"  - {m}" for m in self.c1_tareas_requeridas)
            )
        if self.c2_consecutivas:
            bloques.append(
                "C2 — Conectividad entre tareas consecutivas:\n"
                + "\n".join(f"  - {m}" for m in self.c2_consecutivas)
            )
        if self.c3_capacidad:
            bloques.append(
                "C3 — Capacidad:\n" + "\n".join(f"  - {m}" for m in self.c3_capacidad)
            )
        if self.c4_vehiculos:
            bloques.append(
                "C4 — Vehículos disponibles:\n" + "\n".join(f"  - {m}" for m in self.c4_vehiculos)
            )
        if self.c5_deposito_extremos:
            bloques.append(
                "C5 — Depósito y extremos de ruta:\n"
                + "\n".join(f"  - {m}" for m in self.c5_deposito_extremos)
            )
        return "\n\n".join(bloques) if bloques else "Factible (sin incumplimientos registrados)."


@dataclass
class FeasibilityResult:
    """Resultado de :func:`verificar_factibilidad`. Usa ``bool(result)`` como alias de ``ok``."""

    ok: bool
    details: FeasibilityDetails

    def __bool__(self) -> bool:
        return self.ok


def verificar_factibilidad(
    solucion: Sequence[Sequence[Hashable]],
    data: Mapping[str, Any],
    matriz_distancias: Any,
    *,
    marcador_depot_etiqueta: str | None = None,
    usar_gpu: bool = False,
) -> FeasibilityResult:
    """
    Comprueba factibilidad CARP según cinco condiciones.

    **Formato por etiquetas**: rutas como
    ``['D', 'TR1', 'TR5', ..., 'D']``, donde ``D`` marca depósito (nodo real en
    ``data['DEPOSITO']``) y ``TR*`` son los identificadores de ``tarea`` en
    ``LISTA_ARISTAS_REQ``. El marcador de texto configurable con ``marcador_depot_etiqueta``
    o ``data['MARCADOR_DEPOT_ETIQUETA']`` (por defecto ``"D"``).

    **Depósito lógico:** C2–C5 ya asumen salida y regreso al depósito; ``D`` en el
    vector solo es ayuda visual y se elimina antes de las comprobaciones.

    La matriz puede ser el ``dict`` anidado de Dijkstra o un ``numpy.ndarray`` 2D.
    ``usar_gpu`` deja la API preparada para backend acelerado; hoy usa fallback CPU.
    """
    _backend_solicitado, _backend_real = _aplicar_backend_gpu_placeholder(usar_gpu)
    det = FeasibilityDetails()
    mapa_et = construir_mapa_tareas_por_etiqueta(data)
    if not mapa_et:
        return FeasibilityResult(
            False,
            FeasibilityDetails(
                c1_tareas_requeridas=["No hay LISTA_ARISTAS_REQ / LISTA_ARISTAS_NOREQ en data."]
            ),
        )

    rutas_norm, err = normalizar_rutas_etiquetas(solucion, data, mapa_et, marcador_depot_etiqueta)
    if err:
        det.c1_tareas_requeridas.append(err)
        return FeasibilityResult(False, det)

    required_labels = etiquetas_tareas_requeridas(data)
    todas_e: list[str] = [x for r in rutas_norm for x in r]
    conteo_e: dict[str, int] = {}
    for lab in todas_e:
        conteo_e[lab] = conteo_e.get(lab, 0) + 1
    for lab, k in conteo_e.items():
        if k > 1:
            det.c1_tareas_requeridas.append(
                f"La tarea {lab} aparece {k} veces (cada arista a lo sumo una vez)."
            )
    cub_e = set(conteo_e.keys())
    faltan_e = sorted(required_labels - cub_e)
    if faltan_e:
        det.c1_tareas_requeridas.append(
            "Faltan tareas requeridas: "
            + ", ".join(faltan_e[:40])
            + (" ..." if len(faltan_e) > 40 else "")
        )

    num_vehiculos = int(data.get("VEHICULOS", 0) or 0)

    rutas_activas = [r for r in rutas_norm if r]
    if len(rutas_activas) > num_vehiculos:
        det.c4_vehiculos.append(
            f"Se usan {len(rutas_activas)} rutas no vacías pero VEHICULOS={num_vehiculos}."
        )

    for idx, ruta in enumerate(rutas_norm):
        if not ruta:
            continue
        f2, f3, f5 = _verificar_ruta(
            ruta, mapa_et, data, matriz_distancias, idx
        )
        det.c2_consecutivas.extend(f2)
        det.c3_capacidad.extend(f3)
        det.c5_deposito_extremos.extend(f5)

    ok = not (
        det.c1_tareas_requeridas
        or det.c2_consecutivas
        or det.c3_capacidad
        or det.c4_vehiculos
        or det.c5_deposito_extremos
    )
    return FeasibilityResult(ok, det)


def verificar_factibilidad_desde_instancia(
    nombre_instancia: str,
    solucion: Sequence[Sequence[Hashable]],
    *,
    root: str | os.PathLike[str] | None = None,
    marcador_depot_etiqueta: str | None = None,
    usar_gpu: bool = False,
) -> FeasibilityResult:
    """Carga instancia y matriz Dijkstra del paquete y llama a :func:`verificar_factibilidad`."""
    from .cargar_matrices import cargar_matriz_dijkstra
    from .instances import load_instances

    data = load_instances(nombre_instancia, root=root)
    matriz = cargar_matriz_dijkstra(nombre_instancia, root=root)
    return verificar_factibilidad(
        solucion,
        data,
        matriz,
        marcador_depot_etiqueta=marcador_depot_etiqueta,
        usar_gpu=usar_gpu,
    )
