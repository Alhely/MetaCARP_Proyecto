"""
Utilidades comunes a todas las metaheurísticas (selección inicial, reporte y
exportación a CSV).

La evaluación de costo dentro de los bucles internos NO debe usar las funciones
de este módulo: para eso existe ``evaluador_costo.costo_rapido`` (10×–50× más
rápido que el evaluador clásico basado en NetworkX).
"""
from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Hashable

import networkx as nx

from .costo_solucion import costo_solucion
from .evaluador_costo import (
    ContextoEvaluacion,
    construir_contexto,
    costo_rapido,
    exceso_capacidad_rapido,
    lambda_penal_capacidad_por_defecto,
    objectivo_penalizado,
)
from .reporte_solucion import reporte_solucion

__all__ = [
    "SeleccionInicialResult",
    "ContadorOperadores",
    "copiar_solucion_labels",
    "extraer_candidatas_desde_objeto",
    "evaluar_costo_solucion",
    "seleccionar_mejor_inicial",
    "seleccionar_mejor_inicial_rapido",
    "calcular_metricas_gap",
    "extraer_referencia_bks",
    "calcular_gap_bks",
    "resumen_bks_csv",
    "solucion_legible_humana",
    "generar_reporte_detallado",
    "guardar_resultado_csv",
    "construir_contexto_para_corrida",
    "pesos_inter_bias",
]


def _es_nan(x: Any) -> bool:
    """True si ``x`` es float NaN. Robusto frente a None/strings/etc."""
    try:
        return isinstance(x, float) and x != x
    except Exception:  # noqa: BLE001
        return False


def extraer_referencia_bks(data: Mapping[str, Any]) -> tuple[float | None, str]:
    """
    Extrae la referencia de óptimo/cota desde el dict de instancia original.

    Regla:
    - Si ``BKS`` es un número válido → ``(float(BKS), 'BKS')``.
    - Si ``BKS`` es ``NaN`` o ausente y ``GAP_Value`` es número → ``(float(GAP_Value), 'GAP_Value')``.
    - En cualquier otro caso → ``(None, 'no_disponible')``.
    """
    bks = data.get("BKS")
    if bks is not None and not _es_nan(bks):
        try:
            return float(bks), "BKS"
        except (TypeError, ValueError):
            pass
    gv = data.get("GAP_Value")
    if gv is not None and not _es_nan(gv):
        try:
            return float(gv), "GAP_Value"
        except (TypeError, ValueError):
            pass
    return None, "no_disponible"


def calcular_gap_bks(costo_mejor: float, bks: float | None) -> float | None:
    """
    Gap relativo del mejor costo encontrado contra la referencia BKS:

    .. math:: gap_{bks} = \\frac{(costo\\_mejor - bks)}{bks} \\times 100

    Positivo = peor que la referencia; negativo = mejora la referencia.
    ``None`` si no hay referencia válida o BKS == 0.
    """
    if bks is None or bks == 0:
        return None
    return ((float(costo_mejor) - float(bks)) / float(bks)) * 100.0


def resumen_bks_csv(
    data: Mapping[str, Any],
    costo_mejor: float,
) -> dict[str, Any]:
    """
    Devuelve las tres columnas que documentan la referencia BKS en el CSV:

    - ``bks_referencia``: valor numérico usado como referencia (BKS o GAP_Value).
    - ``bks_origen``: ``'BKS'`` / ``'GAP_Value'`` / ``'no_disponible'``.
    - ``gap_bks_porcentaje``: gap relativo del mejor costo vs la referencia.
    """
    bks, origen = extraer_referencia_bks(data)
    return {
        "bks_referencia": bks if bks is not None else "",
        "bks_origen": origen,
        "gap_bks_porcentaje": calcular_gap_bks(costo_mejor, bks) if bks is not None else "",
    }


@dataclass(frozen=True, slots=True)
class SeleccionInicialResult:
    """
    Resultado de elegir la mejor solución inicial entre candidatos del pickle.

    La elección usa costo puro + penalización por exceso de demanda por ruta,
    para no quedar atrapado sólo en soluciones greedy con costo artificialmente
    bajo pero infactibles por capacidad.
    """

    solucion: list[list[str]]
    costo_puro: float
    violacion_capacidad: float
    n_candidatos_evaluados: int
    n_candidatos_infactibles: int


@dataclass
class ContadorOperadores:
    """
    Lleva la cuenta de uso de operadores de vecindario durante una corrida.

    - ``propuestos``: cada vez que un operador se invoca para generar un vecino.
    - ``aceptados``: cada vez que el movimiento del operador se incorpora al estado actual.
    - ``mejoraron``: subconjunto de ``aceptados`` que además bajó el mejor global histórico.
    - ``trayectoria_mejor``: snapshot de ``aceptados`` capturado cuando se descubrió
      la mejor solución reportada al final.
    """

    propuestos: Counter = field(default_factory=Counter)
    aceptados: Counter = field(default_factory=Counter)
    mejoraron: Counter = field(default_factory=Counter)
    trayectoria_mejor: Counter = field(default_factory=Counter)

    def proponer(self, op: str | None) -> None:
        """Registra que el operador 'op' fue propuesto (generó un vecino)."""
        if op:
            self.propuestos[op] += 1

    def aceptar(self, op: str | None) -> None:
        """Registra que el operador 'op' fue aceptado (la solución cambió)."""
        if op:
            self.aceptados[op] += 1

    def registrar_mejora(self, op: str | None) -> None:
        """Marca una mejora del mejor global y congela el snapshot de aceptados."""
        if op:
            self.mejoraron[op] += 1
        self.trayectoria_mejor = Counter(self.aceptados)

    def como_dict_ordenado(self, contador: Counter) -> dict[str, int]:
        """Convierte un Counter a dict ordenado por valor descendente."""
        return dict(sorted(contador.items(), key=lambda kv: (-kv[1], kv[0])))

    def resumen_csv(self) -> dict[str, int]:
        """
        Devuelve 4 categorías × 7 operadores = 28 columnas planas.

        Formato: ``<categoria>_<operador>`` con conteo entero (0 si no apareció).
        Categorías: ``propuesto``, ``aceptado``, ``mejoraron``, ``trayectoria_mejor``.
        """
        # Importación local para evitar circularidad (vecindarios también importa de aquí).
        from .vecindarios import OPERADORES_POPULARES

        categorias: tuple[tuple[str, Counter], ...] = (
            ("propuesto", self.propuestos),
            ("aceptado", self.aceptados),
            ("mejoraron", self.mejoraron),
            ("trayectoria_mejor", self.trayectoria_mejor),
        )
        salida: dict[str, int] = {}
        for prefijo, contador in categorias:
            for op in OPERADORES_POPULARES:
                salida[f"{prefijo}_{op}"] = int(contador.get(op, 0))
        return salida


def copiar_solucion_labels(sol: Sequence[Sequence[Hashable]]) -> list[list[str]]:
    """Copia ligera a formato de etiquetas string."""
    return [[str(x).strip() for x in ruta] for ruta in sol]


def _es_solucion_lista_de_rutas(obj: Any) -> bool:
    """
    Heurística para detectar si un objeto tiene forma de solución CARP.

    Una solución CARP válida es una lista de rutas, donde cada ruta es
    una lista de etiquetas (strings o enteros) que representan tareas.
    """
    if not isinstance(obj, (list, tuple)) or isinstance(obj, (str, bytes)):
        return False
    for ruta in obj:
        if not isinstance(ruta, (list, tuple)) or isinstance(ruta, (str, bytes)):
            return False
        for token in ruta:
            if isinstance(token, Mapping):
                return False
    return True


def extraer_candidatas_desde_objeto(obj: Any, *, max_nodos: int = 20000) -> list[list[list[str]]]:
    """
    Recorre recursivamente un objeto (dict, lista, etc.) y extrae todas las
    estructuras que tienen forma de solución CARP.

    ``max_nodos`` limita el número de sub-objetos inspeccionados para evitar
    recorridos excesivos en estructuras muy grandes.
    """
    candidatas: list[list[list[str]]] = []
    visitados = 0

    def _walk(x: Any) -> None:
        nonlocal visitados
        visitados += 1
        if visitados > max_nodos:
            return
        if _es_solucion_lista_de_rutas(x):
            candidatas.append(copiar_solucion_labels(x))
            return
        if isinstance(x, Mapping):
            for v in x.values():
                _walk(v)
            return
        if isinstance(x, (list, tuple, set)):
            for v in x:
                _walk(v)

    _walk(obj)
    return candidatas


def evaluar_costo_solucion(
    solucion: Sequence[Sequence[Hashable]],
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    marcador_depot_etiqueta: str | None,
    usar_gpu: bool,
) -> float:
    """
    Evaluación lenta (NetworkX). Conservada solo para usos puntuales fuera de
    bucles internos. Para metaheurísticas, prefiere ``costo_rapido(sol, ctx)``.
    """
    return costo_solucion(
        solucion,
        data,
        G,
        detalle=False,
        marcador_depot_etiqueta=marcador_depot_etiqueta,
        usar_gpu=usar_gpu,
    ).costo_total


def construir_contexto_para_corrida(
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    nombre_instancia: str | None,
    usar_gpu: bool,
    root: str | None = None,
) -> ContextoEvaluacion:
    """
    Construye un contexto de evaluación rápida para una corrida.

    Precomputa la matriz de distancias más cortas (Dijkstra APSP) y la almacena
    en arrays NumPy para que cada evaluación posterior sea 10×–50× más rápida.
    Si ``nombre_instancia`` se proporciona, intenta cargar la matriz desde disco.
    """
    if nombre_instancia:
        # Importación local para evitar circularidad.
        from .evaluador_costo import construir_contexto_desde_instancia

        try:
            return construir_contexto_desde_instancia(
                nombre_instancia, root=root, usar_gpu=usar_gpu
            )
        except FileNotFoundError:
            pass
    return construir_contexto(data, G=G, usar_gpu=usar_gpu)


def seleccionar_mejor_inicial(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    marcador_depot_etiqueta: str | None,
    usar_gpu: bool,
) -> tuple[list[list[str]], float]:
    """
    Versión lenta (NetworkX). Mantenida para retrocompatibilidad.
    En código nuevo usa ``seleccionar_mejor_inicial_rapido`` con el contexto.
    """
    candidatas = extraer_candidatas_desde_objeto(inicial_obj)
    if not candidatas:
        raise ValueError(
            "No se encontraron soluciones candidatas en el objeto inicial. "
            "Se esperaba lista de rutas o estructura anidada que la contenga."
        )

    mejor_sol: list[list[str]] | None = None
    mejor_cost = float("inf")
    errores = 0
    for cand in candidatas:
        try:
            c = evaluar_costo_solucion(
                cand,
                data,
                G,
                marcador_depot_etiqueta=marcador_depot_etiqueta,
                usar_gpu=usar_gpu,
            )
        except Exception:  # noqa: BLE001
            errores += 1
            continue
        if c < mejor_cost:
            mejor_cost = c
            mejor_sol = cand

    if mejor_sol is None:
        raise ValueError(
            "Ninguna candidata inicial pudo evaluarse con costo_solucion. "
            f"Candidatas detectadas: {len(candidatas)} | inválidas: {errores}."
        )
    return mejor_sol, mejor_cost


def seleccionar_mejor_inicial_rapido(
    inicial_obj: Any,
    ctx: ContextoEvaluacion,
    *,
    usar_penalizacion_capacidad: bool = True,
    lambda_capacidad: float | None = None,
) -> SeleccionInicialResult:
    """
    Selecciona la mejor candidata inicial usando el evaluador rápido (NumPy).

    Cuando ``usar_penalizacion_capacidad`` está activo, minimiza
    ``costo_puro + λ·violación``. Si no, minimiza sólo ``costo_puro``.
    """
    candidatas = extraer_candidatas_desde_objeto(inicial_obj)
    if not candidatas:
        raise ValueError(
            "No se encontraron soluciones candidatas en el objeto inicial."
        )

    lam = (
        float(lambda_capacidad)
        if lambda_capacidad is not None
        else lambda_penal_capacidad_por_defecto(ctx)
    )

    mejor_sol: list[list[str]] | None = None
    mejor_obj = float("inf")
    mejor_puro = float("inf")
    mejor_viol = 0.0
    errores = 0
    n_ev = 0
    n_inf = 0

    for cand in candidatas:
        try:
            c = costo_rapido(cand, ctx)
            v = exceso_capacidad_rapido(cand, ctx)
        except Exception:  # noqa: BLE001
            errores += 1
            continue
        n_ev += 1
        if v > 1e-12:
            n_inf += 1
        obj = objectivo_penalizado(
            c, v, usar_penal=usar_penalizacion_capacidad, lam=lam
        )
        if obj < mejor_obj - 1e-12 or (
            abs(obj - mejor_obj) < 1e-12 and c < mejor_puro
        ):
            mejor_obj = obj
            mejor_puro = c
            mejor_viol = v
            mejor_sol = cand

    if mejor_sol is None:
        raise ValueError(
            "Ninguna candidata pudo evaluarse con el evaluador rápido. "
            f"Candidatas detectadas: {len(candidatas)} | inválidas: {errores}."
        )
    return SeleccionInicialResult(
        solucion=mejor_sol,
        costo_puro=mejor_puro,
        violacion_capacidad=float(mejor_viol),
        n_candidatos_evaluados=n_ev,
        n_candidatos_infactibles=n_inf,
    )


def calcular_metricas_gap(costo_inicial: float, costo_mejor: float) -> tuple[float, float, float]:
    """
    Devuelve (gap_pct, mejora_abs, mejora_pct).

    - gap_pct: porcentaje de cambio del costo respecto al inicial
               (negativo = mejora; positivo = empeoramiento).
    - mejora_abs: diferencia absoluta costo_inicial - costo_mejor.
    - mejora_pct: mejora_abs expresada como porcentaje del costo inicial.
    """
    if costo_inicial == 0:
        gap = 0.0 if costo_mejor == 0 else float("inf")
        return gap, costo_inicial - costo_mejor, 0.0
    gap = ((costo_mejor - costo_inicial) / costo_inicial) * 100.0
    mejora_abs = costo_inicial - costo_mejor
    mejora_pct = (mejora_abs / costo_inicial) * 100.0
    return gap, mejora_abs, mejora_pct


def solucion_legible_humana(solucion: Sequence[Sequence[Hashable]]) -> str:
    """
    Convierte una solución a texto legible para el CSV:
    ``R1: D -> TR1 -> ... -> D || R2: ...``
    """
    rutas_txt: list[str] = []
    for i, ruta in enumerate(solucion, start=1):
        seq = [str(x).strip() for x in ruta]
        rutas_txt.append(f"R{i}: " + " -> ".join(seq))
    return " || ".join(rutas_txt)


def generar_reporte_detallado(
    solucion: Sequence[Sequence[Hashable]],
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    nombre_instancia: str = "instancia",
    marcador_depot_etiqueta: str | None,
    usar_gpu: bool,
) -> tuple[str, float]:
    """
    Genera un detalle textual con deadheading (DH) y costo total.

    Se ejecuta solo una vez (al final de la corrida) por lo que se mantiene
    en el evaluador clásico. Retorna ``(texto_reporte, costo_total)``.
    """
    rep = reporte_solucion(
        solucion,
        data,
        G,
        nombre_instancia=nombre_instancia,
        marcador_depot_etiqueta=marcador_depot_etiqueta,
        usar_gpu=usar_gpu,
        guardar=False,
    )
    return rep.texto, rep.costo_total


def guardar_resultado_csv(
    *,
    fila: Mapping[str, Any],
    ruta_csv: str | Path,
) -> str:
    """
    Guarda una ejecución en CSV (una fila por corrida, una columna por item).
    Si el archivo no existe, escribe el encabezado automáticamente.

    Retorna la ruta absoluta del archivo CSV guardado.
    """
    path = Path(ruta_csv).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    normalizada: dict[str, Any] = {}
    for k, v in fila.items():
        normalizada[k] = str(v) if isinstance(v, (list, tuple, set, dict)) else v

    existe = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(normalizada.keys()))
        if not existe:
            writer.writeheader()
        writer.writerow(normalizada)
    return str(path)


# Operadores que mueven tareas DENTRO de una misma ruta. Estos no pueden reducir
# una violación de capacidad por ruta porque no redistribuyen tareas entre rutas.
_INTRA_OPS: frozenset[str] = frozenset({"relocate_intra", "swap_intra", "2opt_intra"})


def pesos_inter_bias(
    violacion: float,
    operadores: Sequence[str],
    *,
    alpha_inter: float = 0.8,
) -> list[float] | None:
    """Calcula pesos para ``rng.choices()`` sesgados hacia operadores inter-ruta.

    Cuando ``violacion > 0``, reparar la solución requiere redistribuir tareas
    entre rutas: solo los operadores inter-ruta pueden hacerlo. Los operadores
    intra-ruta reordenan tareas dentro de una ruta sin afectar la demanda por
    vehículo, por lo que no reducen el exceso de capacidad.

    Devuelve ``None`` si la solución es factible, si la lista de operadores es
    homogénea (solo intra o solo inter), o si no hay violación — indicando
    selección uniforme.

    Args:
        violacion: exceso total de demanda sobre la capacidad (>= 0).
        operadores: lista de nombres de operadores activos en el metaheurístico.
        alpha_inter: fracción de probabilidad total asignada a operadores
            inter-ruta cuando hay violación (0–1). Por defecto 0.8.
    """
    if violacion <= 1e-12:
        return None
    n_intra = sum(1 for op in operadores if op in _INTRA_OPS)
    n_inter = len(operadores) - n_intra
    if n_intra == 0 or n_inter == 0:
        return None
    w_inter = alpha_inter / n_inter
    w_intra = (1.0 - alpha_inter) / n_intra
    return [w_intra if op in _INTRA_OPS else w_inter for op in operadores]
