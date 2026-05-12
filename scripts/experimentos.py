"""
Script de experimentación para metaheurísticas CARP.

Objetivo:
- Ejecutar SA / Tabu / Abejas / Cuckoo por instancia.
- Guardar CSV por metaheurística e instancia.
- Permitir corridas reproducibles con espacio de búsqueda tipo literatura.

Ejemplos:
    python metacarp/scripts/experimentos.py
    python metacarp/scripts/experimentos.py --instancias gdb19 --metaheuristicas sa tabu
    python metacarp/scripts/experimentos.py --seed 7 --salida-dir experimentos
    python metacarp/scripts/experimentos.py --lambda-capacidad 500 --instancias gdb19
    python metacarp/scripts/experimentos.py --sin-penal-capacidad --metaheuristicas sa
"""

# ============================================================
# experimentos.py
# ------------------------------------------------------------
# Script de CAMPAÑA DE EXPERIMENTACIÓN para el proyecto CARP.
#
# Permite lanzar desde línea de comandos corridas sistemáticas
# de las cuatro metaheurísticas sobre múltiples instancias,
# variando parámetros en una grilla (grid search), con
# repeticiones y semillas derivadas para reproducibilidad.
#
# Flujo general:
#   1. Parsear argumentos de línea de comandos.
#   2. Construir el espacio de configuraciones (grid search).
#   3. Para cada instancia x metaheurística x configuración x repetición:
#      a. Derivar una semilla única.
#      b. Ejecutar la metaheurística.
#      c. Guardar resultados en CSV.
#   4. Imprimir resumen de éxitos y fallos.
# ============================================================

from __future__ import annotations

import argparse   # Módulo estándar para parsear argumentos de línea de comandos
import math       # Para math.isnan (validar que lambda no sea NaN)
from dataclasses import dataclass   # Para crear clases de datos simples
from datetime import datetime       # Para generar timestamps en nombres de archivos
from itertools import product       # Para el producto cartesiano en el grid search
from pathlib import Path            # Para manejar rutas de archivos de forma portable
from typing import Callable, Sequence

# Importa las cuatro metaheurísticas y el catálogo de instancias del paquete
from metacarp import (
    busqueda_abejas_desde_instancia,
    busqueda_tabu_desde_instancia,
    cuckoo_search_desde_instancia,
    nombres_soluciones_iniciales_disponibles,
    recocido_simulado_desde_instancia,
)

# Alias canónicos de metaheurísticas disponibles en este script.
# Solo estas cuatro son válidas; cualquier otra generará un error.
METAHEURISTICAS_VALIDAS = ("sa", "tabu", "abejas", "cuckoo")


# ------------------------------------------------------------
# CLASE: MetaRunner
# ------------------------------------------------------------
# OOP utilizado: DATACLASS con frozen=True y slots=True.
#
# Agrupa en un solo objeto toda la información necesaria para
# ejecutar una metaheurística: su nombre, la función que la
# corre, la configuración por defecto y el espacio de búsqueda.
#
# frozen=True → inmutable después de crearse, evita accidentes.
# slots=True  → más eficiente en memoria (sin dict interno).
#
# Callable[..., object] es un tipo que representa cualquier
# función que acepte cualquier número de argumentos y devuelva
# un objeto. Aquí se usa para almacenar funciones como
# recocido_simulado_desde_instancia sin especificar su firma exacta.
@dataclass(frozen=True, slots=True)
class MetaRunner:
    nombre: str                              # Clave canónica: "sa", "tabu", "abejas", "cuckoo"
    run: Callable[..., object]               # Función que ejecuta la metaheurística
    parametros_default: dict[str, object]    # Configuración base si no hay espacio definido
    espacio_parametros: list[dict[str, object]]  # Lista de todas las configuraciones del grid


# ------------------------------------------------------------
# FUNCIÓN AUXILIAR: _grid
# ------------------------------------------------------------
# Genera todas las combinaciones posibles de un espacio de
# parámetros (producto cartesiano). Esto se llama "grid search".
#
# Ejemplo:
#   _grid({"alpha": [0.9, 0.95], "iteraciones": [100, 200]})
#   → [{"alpha": 0.9, "iteraciones": 100},
#      {"alpha": 0.9, "iteraciones": 200},
#      {"alpha": 0.95, "iteraciones": 100},
#      {"alpha": 0.95, "iteraciones": 200}]
#
# itertools.product genera el producto cartesiano de los valores:
#   product([0.9, 0.95], [100, 200]) → (0.9,100), (0.9,200), (0.95,100), (0.95,200)
# zip(keys, vals) empareja cada clave con su valor correspondiente.
def _grid(param_space: dict[str, list[object]]) -> list[dict[str, object]]:
    """
    Construye combinaciones cartesianas de un espacio de búsqueda.
    """
    keys = list(param_space.keys())   # Nombres de los parámetros, en orden fijo
    combos: list[dict[str, object]] = []
    # Para cada combinación de valores (una de cada lista de parámetros):
    for vals in product(*(param_space[k] for k in keys)):
        # zip empareja cada clave con su valor en esta combinación
        combos.append({k: v for k, v in zip(keys, vals)})
    return combos


# ------------------------------------------------------------
# FUNCIÓN: _construir_runners
# ------------------------------------------------------------
# Crea el diccionario de MetaRunner para cada metaheurística,
# con su espacio de búsqueda basado en rangos de la literatura
# de CARP/VRP.
#
# El espacio de parámetros es amplio (modo "literatura") para
# encontrar la mejor configuración. En un experimento rápido,
# puedes reducir los rangos para acelerar la ejecución.
def _construir_runners() -> dict[str, MetaRunner]:
    """
    Define un espacio de búsqueda ampliado (modo literatura) para cada metaheurística.
    """
    # ---- Espacio de parámetros para Recocido Simulado (SA) ----
    sa_space = _grid(
        {
            "temperatura_inicial": [150.0, 300.0, 500.0, 800.0],  # Alta temp → acepta más movimientos malos
            "temperatura_minima": [1e-3],                           # Temperatura de parada
            "alpha": [0.90, 0.93, 0.95, 0.97],                    # Factor de enfriamiento geométrico
            "iteraciones_por_temperatura": [40, 80, 120],           # Iteraciones antes de enfriar
            "max_enfriamientos": [60, 100],                         # Total de ciclos de enfriamiento
        }
    )

    # ---- Espacio de parámetros para Búsqueda Tabú ----
    tabu_space = _grid(
        {
            "iteraciones": [300, 600, 900],           # Total de iteraciones del algoritmo
            "tam_vecindario": [20, 30, 40, 60],       # Vecinos evaluados por iteración
            "tenure_tabu": [10, 15, 20, 30],          # Duración de la prohibición (en iteraciones)
        }
    )

    # ---- Espacio de parámetros para Colonia de Abejas (ABC) ----
    abejas_space = _grid(
        {
            "iteraciones": [300, 600, 900],
            "num_fuentes": [10, 20, 30, 40],          # Tamaño de la población (fuentes de alimento)
            "limite_abandono": [15, 30, 45, 60],      # Intentos fallidos antes de abandonar una fuente
        }
    )

    # ---- Espacio de parámetros para Cuckoo Search ----
    cuckoo_space = _grid(
        {
            "iteraciones": [300, 600, 900],
            "num_nidos": [15, 25, 35],                # Tamaño de la población (nidos)
            "pa_abandono": [0.15, 0.25, 0.35],        # Probabilidad de reemplazar un nido por iteración
            "pasos_levy_base": [2, 3, 5],             # Número base de movimientos en el vuelo de Lévy
            "beta_levy": [1.2, 1.5],                  # Parámetro β de la distribución de Lévy
        }
    )

    # Construye un MetaRunner por metaheurística.
    # parametros_default = primera configuración del espacio (índice 0).
    return {
        "sa": MetaRunner(
            nombre="sa",
            run=recocido_simulado_desde_instancia,
            parametros_default=sa_space[0],
            espacio_parametros=sa_space,
        ),
        "tabu": MetaRunner(
            nombre="tabu",
            run=busqueda_tabu_desde_instancia,
            parametros_default=tabu_space[0],
            espacio_parametros=tabu_space,
        ),
        "abejas": MetaRunner(
            nombre="abejas",
            run=busqueda_abejas_desde_instancia,
            parametros_default=abejas_space[0],
            espacio_parametros=abejas_space,
        ),
        "cuckoo": MetaRunner(
            nombre="cuckoo",
            run=cuckoo_search_desde_instancia,
            parametros_default=cuckoo_space[0],
            espacio_parametros=cuckoo_space,
        ),
    }


# ------------------------------------------------------------
# FUNCIÓN: _parse_args
# ------------------------------------------------------------
# Define y parsea los argumentos de línea de comandos.
# argparse genera automáticamente el mensaje de ayuda (--help)
# y convierte los valores al tipo correcto (int, float, etc.).
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta campañas de experimentación para metaheurísticas CARP."
    )

    # --instancias: lista de nombres de instancias a ejecutar
    # nargs="*" significa "cero o más valores" (lista opcional)
    parser.add_argument(
        "--instancias",
        nargs="*",
        default=["all"],
        help="Lista de instancias (por defecto: all = todas con solución inicial).",
    )

    # --metaheuristicas: subconjunto de algoritmos a ejecutar
    parser.add_argument(
        "--metaheuristicas",
        nargs="*",
        default=["all"],
        help="Subconjunto a ejecutar: sa tabu abejas cuckoo (por defecto: all).",
    )

    # --seed: semilla base para reproducibilidad
    # type=int: convierte el string del argumento al tipo int automáticamente
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Semilla base para reproducibilidad. Si se omite, se genera una "
            "aleatoria (se imprime en consola y queda registrada en cada fila CSV)."
        ),
    )

    # --salida-dir: carpeta raíz donde se guardan los resultados CSV
    parser.add_argument(
        "--salida-dir",
        type=str,
        default="experimentos",
        help="Carpeta raíz donde se guardan los CSV (default: experimentos).",
    )

    # --root: directorio raíz de datos del paquete (avanzado)
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root de datos opcional (si no, usa configuración por defecto del paquete).",
    )

    # --repeticiones: cuántas veces se repite cada configuración
    parser.add_argument(
        "--repeticiones",
        type=int,
        default=2,
        help="Número de repeticiones por configuración (por defecto: 2).",
    )

    # --experimento: etiqueta para nombrar los archivos CSV
    parser.add_argument(
        "--experimento",
        type=str,
        default="tesis",
        help="Etiqueta del experimento para el nombre de archivo CSV.",
    )

    # --usar-gpu: flag booleano (action="store_true" → si se pasa, vale True)
    parser.add_argument(
        "--usar-gpu",
        action="store_true",
        help=(
            "Activa backend GPU (CuPy) para evaluación por lotes (Tabu/Abejas/Cuckoo). "
            "Si CuPy no está disponible se hace fallback automático a CPU rápido."
        ),
    )

    # --sin-penal-capacidad: minimiza solo costo puro, ignora violaciones de capacidad
    parser.add_argument(
        "--sin-penal-capacidad",
        action="store_true",
        help=(
            "Minimiza sólo el costo puro (sin término λ·violación de capacidad). "
            "Equivalente a usar_penalizacion_capacidad=False en todas las metaheurísticas."
        ),
    )

    # --lambda-capacidad: peso λ para penalizar excesos de capacidad
    # metavar="LAMBDA" es el nombre que aparece en el mensaje de ayuda
    parser.add_argument(
        "--lambda-capacidad",
        type=float,
        default=None,
        metavar="LAMBDA",
        help=(
            "Peso λ para la penalización por exceso de capacidad (costo + λ·violación). "
            "Si se omite, cada metaheurística usa la heurística por defecto del evaluador."
        ),
    )
    return parser.parse_args()


# ------------------------------------------------------------
# FUNCIÓN AUXILIAR: _resolver_instancias
# ------------------------------------------------------------
# Convierte el argumento --instancias en una lista concreta.
# Si se pidió "all", consulta las instancias disponibles en
# el paquete y las ordena alfabéticamente.
def _resolver_instancias(raw: Sequence[str], *, root: str | None) -> list[str]:
    if not raw or (len(raw) == 1 and raw[0].lower() == "all"):
        # sorted() ordena alfabéticamente para corridas reproducibles
        return sorted(nombres_soluciones_iniciales_disponibles(root=root))
    return list(raw)


# ------------------------------------------------------------
# FUNCIÓN AUXILIAR: _resolver_metaheuristicas
# ------------------------------------------------------------
# Convierte el argumento --metaheuristicas en una lista concreta.
# Valida que todos los nombres pedidos sean válidos.
def _resolver_metaheuristicas(raw: Sequence[str]) -> list[str]:
    if not raw or (len(raw) == 1 and raw[0].lower() == "all"):
        return list(METAHEURISTICAS_VALIDAS)

    seleccion = [x.lower() for x in raw]   # Normaliza a minúsculas
    # Detecta cualquier nombre que no esté en la lista de válidos
    invalidas = [x for x in seleccion if x not in METAHEURISTICAS_VALIDAS]
    if invalidas:
        raise ValueError(
            f"Metaheurísticas inválidas: {invalidas}. "
            f"Válidas: {METAHEURISTICAS_VALIDAS}."
        )
    return seleccion


# ------------------------------------------------------------
# FUNCIÓN AUXILIAR: _ruta_csv
# ------------------------------------------------------------
# Construye la ruta completa del archivo CSV de resultados.
# El nombre incluye metaheurística, instancia, experimento y
# timestamp para evitar sobreescrituras entre corridas.
def _ruta_csv(
    salida_dir: Path,  # Directorio raíz de la corrida actual
    meta: str,         # Nombre de la metaheurística ("sa", "tabu", etc.)
    instancia: str,    # Nombre de la instancia ("gdb19", etc.)
    *,
    experimento: str,  # Etiqueta del experimento
    ydmh: str,         # Timestamp en formato YYYYDDMMHHMM
) -> Path:
    """
    Un CSV por metaheurística e instancia, con nombre:
    <metaheuristica>_<instancia>_<experimento>_YDMhm.csv
    """
    nombre = f"{meta}_{instancia}_{experimento}_{ydmh}.csv"
    # La carpeta se organiza por metaheurística para no mezclar resultados
    return salida_dir / meta / nombre


# ------------------------------------------------------------
# FUNCIÓN AUXILIAR: _espacio_parametros
# ------------------------------------------------------------
# Devuelve la lista de configuraciones a probar para un runner.
# Si el runner tiene un espacio definido, lo usa; si no, usa
# solo la configuración por defecto como única opción.
# dict(cfg) crea una copia del diccionario para evitar mutaciones.
def _espacio_parametros(runner: MetaRunner) -> list[dict[str, object]]:
    """
    Devuelve el espacio de búsqueda de parámetros para un runner.
    Si no hay espacio definido, usa la configuración default como única opción.
    """
    if runner.espacio_parametros:
        return [dict(cfg) for cfg in runner.espacio_parametros]   # Copia cada config
    return [dict(runner.parametros_default)]


# ============================================================
# FUNCIÓN PRINCIPAL: main
# ============================================================
def main() -> None:
    """
    Punto de entrada del script de experimentación.
    Parsea argumentos, construye el plan de experimentos,
    ejecuta todas las corridas y guarda los resultados en CSV.
    """
    args = _parse_args()

    # expanduser() expande "~" a la ruta del home del usuario
    # resolve() convierte a ruta absoluta
    salida_base_dir = Path(args.salida_dir).expanduser().resolve()
    salida_base_dir.mkdir(parents=True, exist_ok=True)

    # Construye los runners (uno por metaheurística)
    runners = _construir_runners()

    # Resuelve qué instancias y metaheurísticas se van a ejecutar
    instancias = _resolver_instancias(args.instancias, root=args.root)
    metas = _resolver_metaheuristicas(args.metaheuristicas)

    # Validaciones de argumentos
    if not instancias:
        raise RuntimeError("No se encontraron instancias para ejecutar.")
    if args.repeticiones <= 0:
        raise ValueError("--repeticiones debe ser > 0.")
    if not str(args.experimento).strip():
        raise ValueError("--experimento no puede estar vacío.")

    # Determina si se usa penalización por capacidad
    # getattr con default evita AttributeError si el argumento no existe
    usar_penal_cap = not bool(getattr(args, "sin_penal_capacidad", False))
    lambda_cap = getattr(args, "lambda_capacidad", None)
    if lambda_cap is not None and math.isnan(lambda_cap):
        raise ValueError("--lambda-capacidad debe ser un número finito.")

    # Si no se proveyó semilla, genera una aleatoria y la registra
    import random as _rng_mod
    if args.seed is None:
        args.seed = _rng_mod.randint(0, 2**31 - 1)

    # Timestamp fijo para toda la corrida.
    # ydmh se mantiene para el nombre de CSV (compatibilidad histórica).
    now = datetime.now()
    ydmh = now.strftime("%Y%d%m%H%M")     # Para nombres de archivos CSV

    # run_stamp es más preciso (segundos) para el nombre del directorio de corrida.
    # Así dos corridas en el mismo minuto no se sobreescriben.
    run_stamp = now.strftime("%Y%m%d_%H%M%S")
    experimento = str(args.experimento).strip()

    # Directorio único para esta corrida: evita mezclar resultados de campañas distintas
    salida_dir = salida_base_dir / f"{experimento}_{run_stamp}"
    salida_dir.mkdir(parents=True, exist_ok=False)  # exist_ok=False: falla si ya existe (imposible por timestamp)

    # ---- Guardar metadatos de la corrida para reproducibilidad ----
    # Este archivo permite recrear exactamente la misma corrida en el futuro.
    meta_file = salida_dir / "run_info.txt"
    meta_file.write_text(
        f"seed_base={args.seed}\n"
        f"experimento={experimento}\n"
        f"timestamp={run_stamp}\n"
        f"instancias={instancias}\n"
        f"metaheuristicas={metas}\n"
        f"repeticiones={args.repeticiones}\n"
        f"usar_gpu={args.usar_gpu}\n"
        f"usar_penalizacion_capacidad={usar_penal_cap}\n"
        f"lambda_capacidad={lambda_cap}\n",
        encoding="utf-8",
    )

    # ---- Calcular el total de corridas planeadas ----
    # Total = sum(configs_por_meta * repeticiones) * len(instancias)
    total_planeadas = 0
    for meta in metas:
        total_planeadas += len(_espacio_parametros(runners[meta])) * args.repeticiones
    total_planeadas *= len(instancias)

    # ---- Imprimir cabecera del experimento ----
    print("=" * 96)
    print("EXPERIMENTOS METAHEURÍSTICAS")
    print("=" * 96)
    print(f"Instancias       : {instancias}")
    print(f"Metaheurísticas  : {metas}")
    print(f"Seed base        : {args.seed}")
    print(f"Experimento      : {experimento}")
    print(f"Repeticiones     : {args.repeticiones}")
    print(f"Backend          : {'gpu (CuPy)' if args.usar_gpu else 'cpu (NumPy)'}")
    print(
        f"Penal. capacidad : {'sí (costo + λ·violación)' if usar_penal_cap else 'no (sólo costo puro)'}"
    )
    print(
        f"λ capacidad      : {lambda_cap if lambda_cap is not None else '(defecto evaluador)'}"
    )
    for meta in metas:
        # Muestra cuántas configuraciones tiene cada metaheurística
        print(f"Configs {meta:7s} : {len(_espacio_parametros(runners[meta]))}")
    print(f"Corridas planeadas: {total_planeadas}")
    print(f"Salida base      : {salida_base_dir}")
    print(f"Salida corrida   : {salida_dir}")
    print("-" * 96)

    # Contadores de éxitos y fallos
    total_ok = 0
    total_fail = 0

    # ---- Bucle principal de experimentación ----
    # Para cada instancia, para cada metaheurística, para cada configuración, para cada repetición:
    for idx_inst, instancia in enumerate(instancias):
        print(f"\n[INSTANCIA] {instancia}")

        for idx_meta, meta in enumerate(metas):
            runner = runners[meta]  # Obtiene el MetaRunner de esta metaheurística

            # Ruta del CSV donde se guardarán los resultados de esta combinación
            ruta_csv = _ruta_csv(
                salida_dir,
                meta,
                instancia,
                experimento=experimento,
                ydmh=ydmh,
            )
            configs = _espacio_parametros(runner)  # Lista de configuraciones a probar

            print(f"  -> {meta:7s} | configs={len(configs)} | csv={ruta_csv}")

            # cfg_idx empieza en 1 (start=1) para que el índice sea legible en el CSV
            for cfg_idx, cfg in enumerate(configs, start=1):
                for rep in range(1, args.repeticiones + 1):
                    # ---- Derivación de semilla por corrida ----
                    # Cada corrida tiene una semilla ÚNICA derivada de la semilla base
                    # y de su posición en el espacio instancia x meta x config x rep.
                    # Esto garantiza que:
                    #   - La misma corrida siempre produce el mismo resultado (reproducible).
                    #   - Dos corridas distintas tienen semillas distintas (diversidad).
                    semilla = args.seed + (idx_inst * 100000) + (idx_meta * 10000) + (cfg_idx * 100) + rep

                    # kwargs es un diccionario con los parámetros de la corrida.
                    # Se construye copiando la configuración base y agregando
                    # los parámetros comunes a todas las corridas.
                    kwargs = dict(cfg)  # Copia los parámetros de esta configuración

                    # Identificadores para rastrear la corrida en el CSV
                    config_id = f"{meta}-cfg{cfg_idx}"
                    id_corrida = f"{instancia}-{meta}-cfg{cfg_idx}-rep{rep}-seed{semilla}"

                    # extra_csv: columnas adicionales en el CSV con los valores de cada parámetro
                    # Ejemplo: {"param_iteraciones": 300, "param_alpha": 0.93, ...}
                    extra_csv = {f"param_{k}": v for k, v in cfg.items()}

                    # Agrega parámetros comunes al diccionario de kwargs
                    kwargs.update(
                        {
                            "root": args.root,
                            "semilla": semilla,
                            "id_corrida": id_corrida,
                            "config_id": config_id,
                            "repeticion": rep,
                            "guardar_csv": True,            # Siempre guarda en experimentos
                            "ruta_csv": str(ruta_csv),
                            "guardar_historial": False,     # No guarda historial para ahorrar espacio
                            "usar_gpu": args.usar_gpu,
                            "usar_penalizacion_capacidad": usar_penal_cap,
                            "lambda_capacidad": lambda_cap,
                            "extra_csv": extra_csv,
                        }
                    )

                    print(
                        f"     cfg={cfg_idx}/{len(configs)} "
                        f"| rep={rep}/{args.repeticiones} "
                        f"| semilla={semilla}"
                    )

                    # ---- Ejecutar la metaheurística ----
                    # try/except captura cualquier error para que la campaña continúe
                    # aunque una corrida falle (p.ej. instancia corrupta, memoria insuficiente).
                    try:
                        res = runner.run(instancia, **kwargs)  # ** desempaqueta el dict como argumentos
                        print(
                            "       OK "
                            f"| inicial={res.costo_solucion_inicial:.6f} "
                            f"| mejor={res.mejor_costo:.6f} "
                            f"| mejora_inicial_vs_final={res.mejora_porcentaje_inicial_vs_final:.4f}% "
                            f"| tiempo={res.tiempo_segundos:.4f}s"
                        )
                        total_ok += 1
                    except Exception as exc:  # noqa: BLE001 - queremos continuar campaña.
                        # type(exc).__name__ devuelve el nombre de la clase del error
                        print(f"       FAIL | {type(exc).__name__}: {exc}")
                        total_fail += 1

    # ---- Resumen final ----
    print("\n" + "-" * 96)
    print(f"Ejecuciones OK   : {total_ok}")
    print(f"Ejecuciones FAIL : {total_fail}")
    print(f"CSV raíz         : {salida_dir}")
    print("-" * 96)


# Punto de entrada estándar de Python:
# Este bloque solo se ejecuta cuando el script se corre directamente,
# no cuando se importa como módulo desde otro script.
if __name__ == "__main__":
    main()
