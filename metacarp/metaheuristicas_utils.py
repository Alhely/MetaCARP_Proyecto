"""
Utilidades comunes a todas las metaheurísticas (selección inicial, reporte y
exportación a CSV).

La evaluación de costo dentro de los bucles internos NO debe usar las funciones
de este módulo: para eso existe ``evaluador_costo.costo_rapido`` (10×–50× más
rápido que el evaluador clásico basado en NetworkX).
"""
# El import `from __future__ import annotations` permite escribir tipos como
# `float | None` incluso en versiones de Python anteriores a 3.10, sin error.
from __future__ import annotations

# Módulo estándar para escribir archivos CSV (valores separados por comas).
import csv
# Counter es un diccionario especializado que cuenta ocurrencias de elementos.
from collections import Counter
# Tipos abstractos para firmas de funciones: Mapping = cualquier dict-like,
# Sequence = cualquier lista/tupla de solo lectura.
from collections.abc import Mapping, Sequence
# @dataclass genera automáticamente __init__, __repr__, etc.
# field() permite configurar atributos de un dataclass con valores por defecto complejos.
from dataclasses import dataclass, field
# Path es la forma moderna de manejar rutas de archivos de forma portable.
from pathlib import Path
# Any = cualquier tipo; Hashable = tipos que pueden usarse como claves de diccionario.
from typing import Any, Hashable

# NetworkX: biblioteca para trabajar con grafos (nodos y aristas).
import networkx as nx

# Importaciones internas del paquete metacarp:
from .costo_solucion import costo_solucion  # evaluador clásico (más lento, usa NetworkX)
from .evaluador_costo import (
    ContextoEvaluacion,            # contenedor de matrices precomputadas para evaluación rápida
    construir_contexto,            # función que construye el contexto desde datos y grafo
    costo_rapido,                  # evaluador rápido con NumPy (para bucles internos)
    exceso_capacidad_rapido,       # calcula cuánto excede la demanda la capacidad del vehículo
    lambda_penal_capacidad_por_defecto,  # valor λ para penalizar soluciones infactibles
    objectivo_penalizado,          # f(costo) + λ·violación, función objetivo combinada
)
from .reporte_solucion import reporte_solucion  # genera el reporte textual final

# __all__ define qué nombres son exportados cuando alguien hace `from módulo import *`.
# Es buena práctica declararla explícitamente para indicar la API pública del módulo.
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
    "pesos_intra_bias",
]


# Función auxiliar privada (el guion bajo inicial indica que es de uso interno).
def _es_nan(x: Any) -> bool:
    """True si ``x`` es float NaN. Robusto frente a None/strings/etc."""
    try:
        # NaN es el único valor en Python donde x != x es verdadero (por la norma IEEE 754).
        return isinstance(x, float) and x != x
    except Exception:  # noqa: BLE001
        return False


def extraer_referencia_bks(data: Mapping[str, Any]) -> tuple[float | None, str]:
    """
    Extrae la referencia de óptimo/cota desde el dict de instancia original.

    Regla solicitada:
    - Si ``BKS`` es un número válido → ``(float(BKS), 'BKS')``.
    - Si ``BKS`` es ``NaN`` o ausente y ``GAP_Value`` es número → ``(float(GAP_Value), 'GAP_Value')``.
    - En cualquier otro caso → ``(None, 'no_disponible')``.

    Estos valores se guardan en el CSV junto con el gap calculado, para
    documentar exactamente qué referencia se usó por instancia.

    BKS = Best Known Solution (Mejor Solución Conocida de la literatura científica).
    GAP_Value es una cota alternativa cuando no hay BKS publicado.
    """
    # Intentamos obtener el valor BKS del diccionario de datos de la instancia.
    bks = data.get("BKS")
    # Si existe y no es NaN, lo convertimos a float y lo retornamos con su origen.
    if bks is not None and not _es_nan(bks):
        try:
            return float(bks), "BKS"
        except (TypeError, ValueError):
            pass
    # Si no hay BKS válido, intentamos con GAP_Value como alternativa.
    gv = data.get("GAP_Value")
    if gv is not None and not _es_nan(gv):
        try:
            return float(gv), "GAP_Value"
        except (TypeError, ValueError):
            pass
    # Si ninguna cota está disponible, retornamos None para indicar ausencia.
    return None, "no_disponible"


def calcular_gap_bks(costo_mejor: float, bks: float | None) -> float | None:
    """
    Gap relativo del mejor costo encontrado contra la referencia BKS:

    .. math:: gap_{bks} = \\frac{(costo\\_mejor - bks)}{bks} \\times 100

    Interpretación del resultado:
    - **Positivo** → la solución hallada es **peor** que la referencia (% por encima del BKS).
    - **Cero**     → coincide con la referencia.
    - **Negativo** → la solución hallada **mejora** la referencia (poco común si BKS es óptimo demostrado).
    - ``None`` si no hay referencia válida o BKS == 0 (división indefinida).
    """
    # Si no hay referencia o el BKS es cero (evitamos división por cero), no podemos calcular gap.
    if bks is None or bks == 0:
        return None
    # Fórmula: ((costo_obtenido - costo_referencia) / costo_referencia) × 100
    # Un gap del 5% significa que nuestra solución está 5% por encima del óptimo conocido.
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

    Se usa para comparar automáticamente cada corrida contra la literatura.
    """
    # Extraemos el valor de referencia y su origen (BKS, GAP_Value o no_disponible).
    bks, origen = extraer_referencia_bks(data)
    return {
        # Si no hay BKS, se escribe cadena vacía en el CSV para no romper el formato.
        "bks_referencia": bks if bks is not None else "",
        "bks_origen": origen,
        "gap_bks_porcentaje": calcular_gap_bks(costo_mejor, bks)
        if bks is not None
        else "",
    }


# --- CONCEPTO OOP: @dataclass ---
# Un @dataclass es una "clase de datos": Python genera automáticamente el constructor
# (__init__), la representación (__repr__) y la comparación (__eq__) a partir de los
# atributos declarados. Evita escribir código repetitivo de inicialización.
#
# frozen=True: hace que los atributos sean de solo lectura después de crear el objeto
#              (similar a una tupla nombrada). Garantiza que el resultado no se modifique.
# slots=True:  usa __slots__ internamente, lo que reduce el uso de memoria por instancia.
@dataclass(frozen=True, slots=True)
class SeleccionInicialResult:
    """
    Resultado de elegir la mejor solución inicial entre candidatos del pickle.

    La elección usa costo puro + penalización por exceso de demanda por ruta,
    para no quedar atrapado sólo en soluciones greedy con costo artificialmente
    bajo pero infactibles por capacidad.

    Este objeto se devuelve como resultado de la selección inicial y agrupa
    todos los datos relevantes en un solo lugar (patrón de resultado inmutable).
    """

    # La solución elegida: lista de rutas, donde cada ruta es lista de etiquetas string.
    solucion: list[list[str]]
    # Costo de la solución elegida calculado sin penalizaciones.
    costo_puro: float
    # Suma del exceso de demanda sobre la capacidad máxima (0.0 = solución factible).
    violacion_capacidad: float
    # Cantidad de candidatas que se evaluaron antes de elegir la mejor.
    n_candidatos_evaluados: int
    # Cuántas de esas candidatas violaban las restricciones de capacidad.
    n_candidatos_infactibles: int


# --- CONCEPTO OOP: @dataclass mutable (sin frozen=True) ---
# A diferencia de SeleccionInicialResult, este dataclass SÍ permite modificar
# sus atributos después de crear el objeto, porque necesitamos ir actualizando
# los conteos durante la ejecución de la metaheurística.
@dataclass
class ContadorOperadores:
    """
    Lleva la cuenta de uso de operadores de vecindario durante una corrida.

    Un "operador de vecindario" es una función que modifica una solución para
    generar una solución vecina (ej: intercambiar dos tareas entre rutas).

    - ``propuestos``: cada vez que un operador se invoca para generar un vecino.
    - ``aceptados``: cada vez que el movimiento del operador se incorpora al
      estado actual (cambia la solución actual).
    - ``mejoraron``: subconjunto de ``aceptados`` que además bajó el mejor
      global histórico.
    - ``trayectoria_mejor``: snapshot de ``aceptados`` capturado en el momento
      en que se descubrió la mejor solución reportada al final. Responde
      directamente "qué operadores se usaron para construir la mejor".

    Sirve para el análisis posterior: ¿qué operadores fueron más efectivos?
    """

    # Counter es un dict que cuenta ocurrencias; default_factory=Counter crea un
    # Counter vacío por defecto, evitando el error de compartir el mismo objeto
    # entre múltiples instancias (problema clásico con mutables como valor por defecto).
    propuestos: Counter = field(default_factory=Counter)
    aceptados: Counter = field(default_factory=Counter)
    mejoraron: Counter = field(default_factory=Counter)
    # Snapshot del estado de aceptados en el momento de la mejor solución encontrada.
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
        # Guardamos una copia del estado actual de aceptados como referencia histórica.
        # Counter(self.aceptados) crea una copia independiente (no una referencia).
        # El snapshot incluye TODOS los operadores aceptados hasta ahora,
        # incluyendo el actual (asumimos que aceptar(op) ya fue llamado).
        self.trayectoria_mejor = Counter(self.aceptados)

    def como_dict_ordenado(self, contador: Counter) -> dict[str, int]:
        """Convierte un Counter a dict ordenado por valor descendente."""
        # key=lambda kv: (-kv[1], kv[0]) ordena primero por conteo descendente,
        # y desempata alfabéticamente por nombre de operador.
        return dict(sorted(contador.items(), key=lambda kv: (-kv[1], kv[0])))

    def resumen_csv(self) -> dict[str, int]:
        """
        Devuelve 4 categorías × 7 operadores = 28 columnas planas.

        Formato: ``<categoria>_<operador>`` con conteo entero (0 si no apareció).
        Categorías: ``propuesto``, ``aceptado``, ``mejoraron``, ``trayectoria_mejor``.

        Esta forma plana es preferible a un diccionario serializado para
        análisis posterior (filtros, agregaciones, gráficas en pandas).
        """
        # Importación local: se hace aquí para evitar importaciones circulares
        # (vecindarios también importa de este módulo en algunos contextos).
        from .vecindarios import OPERADORES_POPULARES

        # Tupla de pares (nombre_categoria, contador_correspondiente).
        # Una tupla de tuplas es inmutable y clara de leer.
        categorias: tuple[tuple[str, Counter], ...] = (
            ("propuesto", self.propuestos),
            ("aceptado", self.aceptados),
            ("mejoraron", self.mejoraron),
            ("trayectoria_mejor", self.trayectoria_mejor),
        )
        # Construimos el diccionario de salida con una columna por cada
        # combinación de categoría × operador.
        salida: dict[str, int] = {}
        for prefijo, contador in categorias:
            for op in OPERADORES_POPULARES:
                # Si el operador nunca apareció en este contador, se registra 0.
                salida[f"{prefijo}_{op}"] = int(contador.get(op, 0))
        return salida


def copiar_solucion_labels(sol: Sequence[Sequence[Hashable]]) -> list[list[str]]:
    """
    Copia ligera a formato de etiquetas string.

    Convierte cualquier solución (con etiquetas de cualquier tipo) a una
    lista de listas de strings limpios. Esto garantiza que todas las
    metaheurísticas trabajen con el mismo formato uniforme.
    """
    # str(x).strip() convierte a string y elimina espacios al inicio/fin.
    return [[str(x).strip() for x in ruta] for ruta in sol]


def _es_solucion_lista_de_rutas(obj: Any) -> bool:
    """
    Heurística para detectar si un objeto tiene forma de solución CARP.

    Una solución CARP válida es una lista de rutas, donde cada ruta es
    una lista de etiquetas (strings o enteros) que representan tareas.
    """
    # Verificamos que sea una lista o tupla pero no un string (que también es iterable).
    if not isinstance(obj, (list, tuple)) or isinstance(obj, (str, bytes)):
        return False
    for ruta in obj:
        # Cada elemento de la solución debe ser a su vez una lista/tupla (una ruta).
        if not isinstance(ruta, (list, tuple)) or isinstance(ruta, (str, bytes)):
            return False
        for token in ruta:
            # Si un token es un diccionario, no es una etiqueta de tarea válida.
            if isinstance(token, Mapping):
                return False
    return True


def extraer_candidatas_desde_objeto(obj: Any, *, max_nodos: int = 20000) -> list[list[list[str]]]:
    """
    Recorre recursivamente un objeto (dict, lista, etc.) y extrae todas las
    estructuras que tienen forma de solución CARP.

    Útil para extraer candidatas de archivos pickle que pueden contener
    diccionarios anidados con múltiples soluciones guardadas.

    ``max_nodos`` limita el número de sub-objetos inspeccionados para evitar
    recorridos infinitos en estructuras muy grandes.
    """
    candidatas: list[list[list[str]]] = []
    visitados = 0  # contador de sub-objetos inspeccionados

    # Función interna recursiva: se define dentro de extraer_candidatas_desde_objeto
    # para tener acceso a 'candidatas' y 'visitados' mediante el mecanismo de clausura.
    def _walk(x: Any) -> None:
        nonlocal visitados  # indica que 'visitados' es la variable del ámbito externo
        visitados += 1
        if visitados > max_nodos:
            return  # límite de seguridad para estructuras muy profundas
        if _es_solucion_lista_de_rutas(x):
            # Encontramos una solución; la copiamos en formato string y la guardamos.
            candidatas.append(copiar_solucion_labels(x))
            return
        if isinstance(x, Mapping):
            # Si es un diccionario, recorremos sus valores (no las claves).
            for v in x.values():
                _walk(v)
            return
        if isinstance(x, (list, tuple, set)):
            # Si es una colección, recorremos cada elemento.
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

    Esta función llama a costo_solucion() que recalcula rutas en el grafo usando
    el algoritmo Dijkstra en cada llamada: útil para precisión pero muy lento
    para miles de evaluaciones por segundo.
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

    El contexto precomputa la matriz de distancias más cortas entre todos los
    pares de nodos (Dijkstra APSP) y la almacena en arrays NumPy. Esto permite
    que cada evaluación posterior sea 10×–50× más rápida que recalcular con NetworkX.

    Si se proporciona ``nombre_instancia``, intenta cargar la matriz Dijkstra
    ya guardada en disco. Si no existe, la computa desde ``G`` y la guarda.
    """
    if nombre_instancia:
        # Importación local para evitar circularidad en imports.
        from .evaluador_costo import construir_contexto_desde_instancia

        try:
            # Intenta cargar la matriz precomputada desde disco (mucho más rápido).
            return construir_contexto_desde_instancia(
                nombre_instancia, root=root, usar_gpu=usar_gpu
            )
        except FileNotFoundError:
            # Si no existe el archivo de caché, continúa y computa desde el grafo.
            pass
    # Fallback: computa la matriz Dijkstra directamente desde el grafo G.
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
    Versión lenta (NetworkX). Mantenida para retrocompatibilidad. En código
    nuevo usa ``seleccionar_mejor_inicial_rapido`` con el contexto.

    Evalúa todas las candidatas con el evaluador clásico y retorna la mejor
    junto con su costo. Se usa solo cuando no se dispone de un contexto rápido.
    """
    candidatas = extraer_candidatas_desde_objeto(inicial_obj)
    if not candidatas:
        raise ValueError(
            "No se encontraron soluciones candidatas en el objeto inicial. "
            "Se esperaba lista de rutas o estructura anidada que la contenga."
        )

    mejor_sol: list[list[str]] | None = None
    mejor_cost = float("inf")  # infinito como punto de partida (cualquier costo real será menor)
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
    ``costo_puro + λ·violación``, donde la violación es la suma de excesos de
    demanda por ruta sobre ``CAPACIDAD``. Si no, minimiza sólo ``costo_puro``.

    La penalización permite comparar soluciones factibles e infactibles en la
    misma escala, priorizando las factibles sin descartar completamente las
    infactibles que tienen buen costo.
    """
    candidatas = extraer_candidatas_desde_objeto(inicial_obj)
    if not candidatas:
        raise ValueError(
            "No se encontraron soluciones candidatas en el objeto inicial."
        )

    # λ (lambda) es el peso de penalización por violación de capacidad.
    # Si el usuario no lo especifica, usamos el valor por defecto del contexto.
    lam = (
        float(lambda_capacidad)
        if lambda_capacidad is not None
        else lambda_penal_capacidad_por_defecto(ctx)
    )

    # Inicializamos los "mejores" con valores extremos para que cualquier candidata real sea mejor.
    mejor_sol: list[list[str]] | None = None
    mejor_obj = float("inf")   # objetivo penalizado del mejor encontrado
    mejor_puro = float("inf")  # costo sin penalización del mejor encontrado
    mejor_viol = 0.0           # violación de capacidad del mejor encontrado
    errores = 0
    n_ev = 0   # número de candidatas evaluadas exitosamente
    n_inf = 0  # número de candidatas que violan alguna restricción

    for cand in candidatas:
        try:
            c = costo_rapido(cand, ctx)          # costo sin penalización
            v = exceso_capacidad_rapido(cand, ctx)  # suma de excesos de demanda
        except Exception:  # noqa: BLE001
            errores += 1
            continue
        n_ev += 1
        if v > 1e-12:  # umbral numérico: consideramos infactible si el exceso es mayor que ε
            n_inf += 1
        # Calculamos el objetivo combinado: costo_puro + λ × violación
        obj = objectivo_penalizado(
            c, v, usar_penal=usar_penalizacion_capacidad, lam=lam
        )
        # Seleccionamos esta candidata si es estrictamente mejor en el objetivo penalizado,
        # o si empatan en objetivo y tiene menor costo puro (desempate por costo real).
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
    # assert confirma la postcondición: si llegamos aquí, mejor_sol no puede ser None.
    assert mejor_sol is not None
    # Retornamos un objeto de resultado inmutable con todos los datos relevantes.
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

    Gap negativo indica mejora contra la referencia inicial.
    """
    if costo_inicial == 0:
        # Evitamos división por cero: si el costo inicial es 0, el gap no está definido.
        gap = 0.0 if costo_mejor == 0 else float("inf")
        return gap, costo_inicial - costo_mejor, 0.0
    # gap = ((costo_final - costo_inicial) / costo_inicial) × 100
    gap = ((costo_mejor - costo_inicial) / costo_inicial) * 100.0
    mejora_abs = costo_inicial - costo_mejor
    mejora_pct = (mejora_abs / costo_inicial) * 100.0
    return gap, mejora_abs, mejora_pct


def solucion_legible_humana(solucion: Sequence[Sequence[Hashable]]) -> str:
    """
    Convierte una solución a texto legible para el CSV:
    ``R1: D -> TR1 -> ... -> D || R2: ...``

    El formato muestra cada ruta numerada, con las tareas y el depósito (D)
    separados por flechas, y las rutas separadas por ||.
    """
    rutas_txt: list[str] = []
    # enumerate(solucion, start=1) numera las rutas desde 1 (más legible que desde 0).
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

    "Deadheading" es el recorrido en vacío que hace un vehículo entre tareas
    (sin recoger residuos). Es importante reportarlo porque forma parte del
    costo total de la solución.

    Se ejecuta solo una vez (al final de la corrida) por lo que se mantiene
    en el evaluador clásico (no necesita ser ultra-rápido).

    Retorna una tupla: (texto_reporte, costo_total).
    """
    rep = reporte_solucion(
        solucion,
        data,
        G,
        nombre_instancia=nombre_instancia,
        marcador_depot_etiqueta=marcador_depot_etiqueta,
        usar_gpu=usar_gpu,
        guardar=False,  # no guarda a disco, solo retorna el texto
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

    El archivo CSV acumula resultados: cada llamada agrega una fila nueva al final.
    Esto permite ejecutar múltiples corridas y comparar resultados en Excel/pandas.

    Retorna la ruta absoluta del archivo CSV guardado.
    """
    # Path.expanduser() expande ~ a la carpeta home del usuario.
    # Path.resolve() convierte la ruta a absoluta.
    path = Path(ruta_csv).expanduser().resolve()
    # Crea todos los directorios intermedios necesarios (como `mkdir -p`).
    path.parent.mkdir(parents=True, exist_ok=True)

    # Normalizamos los valores: listas y dicts se convierten a string para el CSV.
    normalizada: dict[str, Any] = {}
    for k, v in fila.items():
        normalizada[k] = str(v) if isinstance(v, (list, tuple, set, dict)) else v

    # Verificamos si el archivo ya existe para saber si necesitamos escribir el encabezado.
    existe = path.is_file()
    # Abrimos en modo "a" (append = agregar al final), con codificación UTF-8.
    with path.open("a", newline="", encoding="utf-8") as f:
        # DictWriter escribe filas como diccionarios, usando fieldnames como orden de columnas.
        writer = csv.DictWriter(f, fieldnames=list(normalizada.keys()))
        if not existe:
            writer.writeheader()  # escribe la primera fila con los nombres de columnas
        writer.writerow(normalizada)
    return str(path)


# ---------------------------------------------------------------------------
# Utilidad: pesos_intra_bias
# ---------------------------------------------------------------------------
_INTRA_OPS: frozenset[str] = frozenset({"relocate_intra", "swap_intra", "2opt_intra"})


def pesos_intra_bias(
    violacion: float,
    operadores: Sequence[str],
    *,
    alpha_intra: float = 0.8,
) -> list[float] | None:
    """Calcula pesos para ``rng.choices()`` sesgados hacia operadores intra-ruta.

    Cuando la solución actual viola capacidad (``violacion > 0``), devuelve una
    lista de pesos donde los operadores intra-ruta reciben en conjunto una
    probabilidad de ``alpha_intra`` y los inter-ruta el resto.

    Si la solución es factible o la lista de operadores es homogénea (solo intra
    o solo inter), devuelve ``None`` para indicar selección uniforme.

    Args:
        violacion: exceso total de demanda sobre la capacidad (>= 0).
        operadores: lista de nombres de operadores activos en el metaheurístico.
        alpha_intra: fracción de probabilidad total asignada a intra-ruta (0–1).
    """
    if violacion <= 1e-12:
        return None
    n_intra = sum(1 for op in operadores if op in _INTRA_OPS)
    n_inter = len(operadores) - n_intra
    if n_intra == 0 or n_inter == 0:
        return None
    w_intra = alpha_intra / n_intra
    w_inter = (1.0 - alpha_intra) / n_inter
    return [w_intra if op in _INTRA_OPS else w_inter for op in operadores]
