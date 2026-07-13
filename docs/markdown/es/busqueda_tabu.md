# Búsqueda Tabú — Documentación Técnica

## Introducción conceptual

La **Búsqueda Tabú** (_Tabu Search_) es una metaheurística de optimización combinatoria diseñada para escapar de mínimos locales mediante el uso de **memoria explícita de movimientos recientes**. A diferencia de las búsquedas locales clásicas, que quedan atrapadas en el primer óptimo local que encuentran, la Búsqueda Tabú se mueve siempre al mejor vecino disponible, incluso si ese vecino es peor que la solución actual. Para evitar que el algoritmo regrese indefinidamente a los mismos estados (ciclos), mantiene una **lista tabú**: un registro de los movimientos recientemente ejecutados que quedan temporalmente prohibidos.

### Inspiración del algoritmo

El mecanismo central es la **memoria de corto plazo**: el algoritmo "recuerda" qué perturbaciones acaba de aplicar y se prohíbe a sí mismo deshacerlas durante un número configurable de iteraciones (`tenure_tabu`). Esto fuerza la exploración de zonas del espacio de soluciones que no se habrían visitado con una búsqueda codiciosa pura.

### Por qué es adecuada para CARP

El Problema de Rutas sobre Arcos con Capacidad (CARP, _Capacitated Arc Routing Problem_) es NP-difícil y tiene un espacio de soluciones altamente no convexo con numerosos mínimos locales. La Búsqueda Tabú es especialmente adecuada porque:

- Navega eficientemente el espacio de rutas evitando ciclos sin necesidad de reinicializaciones costosas.
- Su estructura de vecindario es flexible: admite múltiples operadores (intra-ruta e inter-ruta) que pueden explorarse en un único lote por iteración.
- El criterio de aspiración permite capturar soluciones óptimas aun cuando el movimiento que las genera esté en la lista tabú.
- La penalización de capacidad (`usar_penalizacion_capacidad`) permite explorar soluciones infactibles temporalmente, lo que enriquece la diversificación sin abandonar la factibilidad como objetivo final.

---

## Cómo funciona paso a paso

El siguiente diagrama muestra el flujo completo de la Búsqueda Tabú implementada en `busqueda_tabu.py`:

```
┌──────────────────────────────────────────────────────┐
│                   INICIALIZACIÓN                     │
│  · Cargar instancia: data, G, inicial_obj            │
│  · Construir ContextoEvaluacion (Dijkstra + NumPy)   │
│  · Seleccionar mejor solución inicial (rápida)       │
│  · sol_actual ← mejor candidata inicial              │
│  · Inicializar: tabu_hasta = {}, ContadorOperadores  │
│  · mejor_any_c, mejor_fact_c ← costos iniciales     │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  it = 0, 1, …, iter-1  │◄──────────────────┐
          └──────────┬─────────────┘                    │
                     │                                  │
                     ▼                                  │
   ┌─────────────────────────────────────┐             │
   │  PASO 1: Generación del vecindario  │             │
   │  Generar tam_vecindario vecinos     │             │
   │  usando generar_vecino() con        │             │
   │  operadores seleccionados al azar   │             │
   │  → lista vecinos[], movimientos[]   │             │
   └──────────────────┬──────────────────┘             │
                      │                                 │
                      ▼                                 │
   ┌─────────────────────────────────────┐             │
   │  PASO 2: Evaluación en lote         │             │
   │  costo_lote_penalizado_ids()        │             │
   │  → objs_np[], bases_np[], viols_np[]│             │
   └──────────────────┬──────────────────┘             │
                      │                                 │
                      ▼                                 │
   ┌─────────────────────────────────────┐             │
   │  PASO 3: Selección admisible        │             │
   │  Para cada vecino idx:              │             │
   │    key = _clave_tabu(movimientos[idx])│           │
   │    es_tabu = tabu_hasta[key] > it   │             │
   │    aspiracion = costo_puro < rep_best│            │
   │    Si es_tabu AND NOT aspiracion:   │             │
   │      → saltar (no admisible)        │             │
   │    Si no: candidato admisible       │             │
   └──────────────────┬──────────────────┘             │
                      │                                 │
                      ▼                                 │
   ┌─────────────────────────────────────┐             │
   │  PASO 4: Elección del movimiento    │             │
   │  ¿Hay algún admisible?              │             │
   │    SÍ → elegido = mejor admisible   │             │
   │    NO  → elegido = mejor global     │             │
   │          (bloqueados += N)          │             │
   └──────────────────┬──────────────────┘             │
                      │                                 │
                      ▼                                 │
   ┌─────────────────────────────────────┐             │
   │  PASO 5: Actualización de memoria   │             │
   │  sol_actual ← vecinos[elegido_idx]  │             │
   │  tabu_hasta[clave] = it + tenure    │             │
   │  Limpiar tabu_hasta vencidos (c/25) │             │
   └──────────────────┬──────────────────┘             │
                      │                                 │
                      ▼                                 │
   ┌─────────────────────────────────────┐             │
   │  PASO 6: Actualización de mejores   │             │
   │  Si costo_actual < mejor_any_c:     │             │
   │    mejor_any_c ← costo_actual       │             │
   │  Si sol factible Y costo < mejor_f: │             │
   │    mejor_fact_c ← costo_actual      │             │
   │  Si mejora reportable: mejoras += 1 │             │
   └──────────────────┬──────────────────┘             │
                      │                                 │
                      ▼                                 │
          ┌───────────────────────┐                    │
          │  ¿it < iteraciones-1? ├────sSÍ─────────────┘
          └───────────────────────┘
                      │ NO
                      ▼
   ┌─────────────────────────────────────┐
   │  RESULTADO FINAL                    │
   │  sol_mejor = mejor_fact_s           │
   │             OR mejor_any_s          │
   │  Calcular métricas (gap, mejora)    │
   │  Guardar CSV (opcional)             │
   │  Retornar BusquedaTabuResult        │
   └─────────────────────────────────────┘
```

---

## Componentes clave

### La lista tabú (`tabu_hasta`)

La lista tabú se implementa como un **diccionario** de Python:

```python
tabu_hasta: dict[tuple[Any, ...], int] = {}
```

- **Clave**: una tupla hashable generada por `_clave_tabu(mov)` que identifica unívocamente el movimiento. La clave incluye: nombre del operador, índices de rutas (`ruta_a`, `ruta_b`), posiciones dentro de las rutas (`i`, `j`, `k`, `l`) y las etiquetas de las tareas desplazadas (`labels_movidos`).
- **Valor**: el número de iteración en que el movimiento deja de estar prohibido.
- **Verificación de estado**: un movimiento está tabú si `tabu_hasta.get(clave, -1) > it`, donde `it` es la iteración actual.
- **Complejidad**: la búsqueda es O(1) gracias al uso de un diccionario, más eficiente que una lista circular.
- **Registro**: al aceptar un movimiento elegido, se añade: `tabu_hasta[clave] = it + tenure_tabu`.

Dos movimientos se consideran iguales si producen exactamente la misma perturbación estructural, lo que evita aplicar variantes simétricas del mismo movimiento durante la tenencia.

### El criterio de aspiración

El criterio de aspiración es la única condición bajo la cual el algoritmo ignora una prohibición tabú:

```python
aspiracion = c_p < rep_best - 1e-15
```

donde `c_p` es el **costo puro** (sin penalización) del vecino tabú y `rep_best` es el mejor costo reportable actual (que prioriza la mejor solución factible cuando existe). Si el vecino tabú tiene un costo estrictamente mejor que el mejor histórico, se acepta de todas formas. Esto evita que la lista tabú bloquee la convergencia al óptimo global cuando ese salto pasa a través de un movimiento prohibido.

### El tamaño del vecindario (`tam_vecindario`)

En cada iteración se generan exactamente `tam_vecindario` vecinos evaluados en un único lote. Aumentar este valor amplía la exploración por iteración a costa de mayor tiempo de cómputo. En instancias grandes con `usar_gpu=True`, el overhead de transferencia PCIe se amortiza mejor con tamaños grandes (`tam_vecindario >= 30`).

### La tenencia tabú (`tenure_tabu`)

La tenencia define cuántas iteraciones permanece prohibido un movimiento después de haber sido ejecutado:

```python
tabu_hasta[_clave_tabu(ultimo_mov_aceptado)] = it + tenure_tabu
```

- Una tenencia **pequeña** diversifica menos: el algoritmo puede volver a estados anteriores rápidamente.
- Una tenencia **grande** fuerza mayor exploración pero puede prohibir movimientos beneficiosos durante demasiado tiempo.
- La implementación realiza una **limpieza periódica** del diccionario cada 25 iteraciones para eliminar entradas vencidas (`vence <= it`), manteniendo el uso de memoria acotado.

### El doble seguimiento de mejor solución

El algoritmo mantiene simultáneamente dos tipos de "mejor solución":

| Variable | Significado | Condición de actualización |
|---|---|---|
| `mejor_any_c` / `mejor_any_s` | Mejor costo sin restricción de factibilidad | `costo_actual < mejor_any_c - 1e-15` |
| `mejor_fact_c` / `mejor_fact_s` | Mejor costo entre soluciones factibles | `viol_actual < 1e-12` AND `costo_actual < mejor_fact_c - 1e-15` |

La función de reporte interno `costo_para_reporte()` devuelve `mejor_fact_c` si existe; en caso contrario, devuelve `mejor_any_c`. La solución final retornada es `mejor_fact_s` cuando existe, o `mejor_any_s` si ninguna solución factible fue encontrada. El campo `mejor_solucion_factible_final` del resultado indica cuál de los dos casos aplica.

---

## La función `busqueda_tabu`

### Firma exacta

```python
def busqueda_tabu(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    iteraciones: int = 400,
    tam_vecindario: int = 25,
    tenure_tabu: int = 20,
    semilla: int | None = None,
    operadores: Iterable[str] = OPERADORES_POPULARES,
    marcador_depot_etiqueta: str | None = None,
    usar_gpu: bool = False,
    backend_vecindario: Literal["labels", "ids"] = "labels",
    guardar_historial: bool = True,
    guardar_csv: bool = False,
    ruta_csv: str | None = None,
    nombre_instancia: str = "instancia",
    id_corrida: str | None = None,
    config_id: str | None = None,
    repeticion: int | None = None,
    root: str | None = None,
    usar_penalizacion_capacidad: bool = True,
    lambda_capacidad: float | None = None,
    extra_csv: dict[str, object] | None = None,
) -> BusquedaTabuResult
```

### Tabla de parámetros

| Parámetro | Tipo | Valor por defecto | Descripción |
|---|---|---|---|
| `inicial_obj` | `Any` | — | Objeto con la(s) solución(es) inicial(es). Puede ser una lista de rutas, un diccionario anidado o cualquier estructura que contenga soluciones CARP. |
| `data` | `Mapping[str, Any]` | — | Diccionario de datos de la instancia (demandas, capacidad, BKS, listas de aristas, etc.), tal como lo devuelve `load_instances()`. |
| `G` | `nx.Graph` | — | Grafo de la instancia cargado desde GEXF. Se usa para construir la matriz Dijkstra si no existe en disco. |
| `iteraciones` | `int` | `400` | Número total de iteraciones del bucle principal. Condiciona la parada del algoritmo. |
| `tam_vecindario` | `int` | `25` | Vecinos generados y evaluados por iteración. Mayor valor amplía la exploración local. |
| `tenure_tabu` | `int` | `20` | Número de iteraciones que un movimiento permanece prohibido en la lista tabú. |
| `semilla` | `int \| None` | `None` | Semilla del generador de números aleatorios. `None` produce corridas no reproducibles. |
| `operadores` | `Iterable[str]` | `OPERADORES_POPULARES` | Conjunto de operadores de vecindario habilitados. Ver `OPERADORES_POPULARES` en `vecindarios.py`. |
| `marcador_depot_etiqueta` | `str \| None` | `None` | Etiqueta del nodo depósito en las rutas (p. ej. `"D"`). Si `None`, se toma del contexto. |
| `usar_gpu` | `bool` | `False` | Si `True`, intenta evaluar el lote de vecinos en GPU con CuPy. Hace fallback a CPU si CuPy no está disponible. |
| `backend_vecindario` | `Literal["labels", "ids"]` | `"labels"` | Representación interna para generar vecinos: `"labels"` usa etiquetas de texto; `"ids"` usa enteros. |
| `guardar_historial` | `bool` | `True` | Si `True`, registra el mejor costo al inicio de cada iteración en `historial_mejor_costo`. |
| `guardar_csv` | `bool` | `False` | Si `True`, escribe una fila de resultados en el archivo CSV al finalizar la corrida. |
| `ruta_csv` | `str \| None` | `None` | Ruta del archivo CSV de salida. Si `None`, se genera automáticamente como `resultados_busqueda_tabu_{nombre_instancia}.csv`. |
| `nombre_instancia` | `str` | `"instancia"` | Identificador de la instancia usado en el CSV y en la carga del contexto desde caché. |
| `id_corrida` | `str \| None` | `None` | Identificador de corrida para el CSV (útil en experimentos repetidos). |
| `config_id` | `str \| None` | `None` | Identificador de configuración de hiperparámetros para el CSV. |
| `repeticion` | `int \| None` | `None` | Número de repetición dentro de un experimento. Se escribe en el CSV. |
| `root` | `str \| None` | `None` | Directorio raíz donde buscar los archivos de la instancia. `None` usa el directorio por defecto del paquete. |
| `usar_penalizacion_capacidad` | `bool` | `True` | Si `True`, el objetivo de comparación entre vecinos es `costo_puro + λ × violación`. |
| `lambda_capacidad` | `float \| None` | `None` | Peso λ de penalización por violación de capacidad. Si `None`, se calcula automáticamente como ~10 × mediana de arcos. |
| `extra_csv` | `dict[str, object] \| None` | `None` | Columnas adicionales que se añaden a la fila del CSV. Útil para registrar metadatos de experimentos. |

### Qué retorna

La función retorna un objeto `BusquedaTabuResult` inmutable con todos los datos de la corrida.

---

## `BusquedaTabuResult`

Dataclass inmutable (`frozen=True, slots=True`) que agrupa la solución encontrada, métricas de calidad, información de tiempo y estadísticas de operadores.

```python
@dataclass(frozen=True, slots=True)
class BusquedaTabuResult:
    ...
```

### Tabla de campos

| Campo | Tipo | Descripción |
|---|---|---|
| `mejor_solucion` | `list[list[str]]` | Mejor solución encontrada: lista de rutas, cada ruta es una lista de etiquetas de tarea. |
| `mejor_costo` | `float` | Costo total (puro, sin penalización) de la mejor solución reportada. |
| `solucion_inicial_referencia` | `list[list[str]]` | Solución inicial seleccionada, usada como referencia para calcular la mejora. |
| `costo_solucion_inicial` | `float` | Costo de la solución inicial (punto de partida de la búsqueda). |
| `mejora_absoluta` | `float` | `costo_inicial - mejor_costo`. Positivo indica mejora real. |
| `mejora_porcentaje_inicial_vs_final` | `float` | `mejora_absoluta / costo_inicial × 100`. |
| `tiempo_segundos` | `float` | Tiempo total de ejecución en segundos (medido con `time.perf_counter`). |
| `iteraciones_totales` | `int` | Número total de iteraciones ejecutadas (igual al parámetro `iteraciones`). |
| `vecinos_evaluados` | `int` | Total de soluciones vecinas evaluadas durante toda la búsqueda. |
| `movimientos_tabu_bloqueados` | `int` | Total de veces que un movimiento fue ignorado por estar en la lista tabú. |
| `mejoras` | `int` | Veces que el mejor costo reportable mejoró durante la búsqueda. |
| `semilla` | `int \| None` | Semilla usada para el generador de números aleatorios. |
| `backend_evaluacion` | `str` | Backend real usado para la evaluación: `"cpu"` o `"gpu"`. |
| `historial_mejor_costo` | `list[float]` | Historial del mejor costo al inicio de cada iteración (vacío si `guardar_historial=False`). |
| `ultimo_movimiento_aceptado` | `MovimientoVecindario \| None` | Último movimiento que fue aceptado al finalizar la búsqueda. |
| `operadores_propuestos` | `dict[str, int]` | Conteo de veces que cada operador fue propuesto para generar un vecino, ordenado por frecuencia descendente. |
| `operadores_aceptados` | `dict[str, int]` | Conteo de veces que cada operador generó un vecino aceptado, ordenado por frecuencia descendente. |
| `operadores_mejoraron` | `dict[str, int]` | Conteo de veces que cada operador produjo una mejora del mejor global. |
| `operadores_trayectoria_mejor` | `dict[str, int]` | Snapshot de `operadores_aceptados` en el momento exacto en que se encontró la mejor solución. |
| `usar_penalizacion_capacidad` | `bool` | Indica si la penalización de capacidad estuvo activa durante la búsqueda. |
| `lambda_capacidad` | `float` | Valor efectivo de λ usado para penalizar violaciones de capacidad. |
| `n_iniciales_evaluados` | `int` | Número de candidatas iniciales que fueron evaluadas durante la selección. |
| `iniciales_infactibles_aceptadas` | `int` | Candidatas iniciales que violaban restricciones de capacidad. |
| `aceptaciones_solucion_infactible` | `int` | Veces que se aceptó una solución infactible como sol_actual durante el bucle. |
| `mejor_solucion_factible_final` | `bool` | `True` si la mejor solución retornada respeta la restricción de capacidad. |
| `archivo_csv` | `str \| None` | Ruta absoluta del CSV donde se guardaron los resultados, o `None` si `guardar_csv=False`. |

---

## `ContadorOperadores`

`ContadorOperadores` es un dataclass mutable definido en `metaheuristicas_utils.py` que lleva la cuenta del uso de cada operador de vecindario durante una corrida.

```python
@dataclass
class ContadorOperadores:
    propuestos: Counter
    aceptados: Counter
    mejoraron: Counter
    trayectoria_mejor: Counter
```

### Qué registra cada contador

| Contador | Método de registro | Qué mide |
|---|---|---|
| `propuestos` | `contador.proponer(op)` | Cuántas veces el operador `op` fue invocado para generar un vecino. Se llama una vez por cada vecino del lote, en cada iteración. |
| `aceptados` | `contador.aceptar(op)` | Cuántas veces el movimiento del operador `op` fue seleccionado y aplicado como nueva solución actual. |
| `mejoraron` | `contador.registrar_mejora(op)` | Subconjunto de `aceptados`: cuántas veces el operador `op` fue responsable de mejorar el mejor costo reportable histórico. |
| `trayectoria_mejor` | Snapshot automático en `registrar_mejora()` | Fotografía del estado de `aceptados` en el instante exacto en que se descubrió la mejor solución. Responde directamente a "¿qué combinación de operadores construyó la mejor solución?". |

### Cómo interpretar los contadores

- **Alta tasa de propuesta y baja de aceptación** en un operador indica que sus movimientos son frecuentemente tabú o dominados por otros.
- **`operadores_mejoraron`** identifica qué operadores son los más productivos para la instancia: los que tienen mayor conteo en este campo son los que deben priorizarse en un ajuste fino.
- **`operadores_trayectoria_mejor`** es especialmente útil para análisis _post hoc_: revela qué secuencia de operadores fue acumulada hasta el momento de la mejor solución, sin importar lo que ocurrió después.
- El método `resumen_csv()` expande los cuatro contadores en 28 columnas planas (`<categoria>_<operador>`) para facilitar el análisis con pandas.

---

## Ejemplo completo de uso

### Uso con `busqueda_tabu_desde_instancia` (forma más simple)

```python
from metacarp.busqueda_tabu import busqueda_tabu_desde_instancia

resultado = busqueda_tabu_desde_instancia(
    "gdb1",
    iteraciones=400,
    tam_vecindario=25,
    tenure_tabu=20,
    semilla=42,
    guardar_historial=True,
    guardar_csv=True,
    ruta_csv="resultados/corrida_gdb1.csv",
    id_corrida="exp_01",
    repeticion=1,
    usar_penalizacion_capacidad=True,
)

print(f"Mejor costo: {resultado.mejor_costo:.2f}")
print(f"Costo inicial: {resultado.costo_solucion_inicial:.2f}")
print(f"Mejora absoluta: {resultado.mejora_absoluta:.2f}")
print(f"Mejora porcentual: {resultado.mejora_porcentaje_inicial_vs_final:.2f}%")
print(f"Tiempo de ejecución: {resultado.tiempo_segundos:.3f} s")
print(f"Iteraciones ejecutadas: {resultado.iteraciones_totales}")
print(f"Vecinos evaluados: {resultado.vecinos_evaluados}")
print(f"Movimientos tabú bloqueados: {resultado.movimientos_tabu_bloqueados}")
print(f"Mejoras del mejor global: {resultado.mejoras}")
print(f"Solución factible final: {resultado.mejor_solucion_factible_final}")
print(f"Backend de evaluación: {resultado.backend_evaluacion}")
print(f"Lambda capacidad (λ): {resultado.lambda_capacidad:.4f}")

print("\nOperadores más efectivos (mejoraron el mejor global):")
for op, conteo in resultado.operadores_mejoraron.items():
    print(f"  {op}: {conteo}")

print("\nMejor solución encontrada:")
for i, ruta in enumerate(resultado.mejor_solucion, start=1):
    print(f"  Ruta {i}: {' -> '.join(ruta)}")
```

### Uso con `busqueda_tabu` (control completo)

```python
from metacarp.busqueda_tabu import busqueda_tabu
from metacarp.cargar_grafos import cargar_objeto_gexf
from metacarp.cargar_soluciones_iniciales import cargar_solucion_inicial
from metacarp.instances import load_instances
from metacarp.vecindarios import OPERADORES_POPULARES

nombre = "gdb1"

# Carga de recursos
data = load_instances(nombre)
G = cargar_objeto_gexf(nombre)
inicial_obj = cargar_solucion_inicial(nombre)

# Ejecutar la búsqueda tabú
resultado = busqueda_tabu(
    inicial_obj,
    data,
    G,
    iteraciones=600,
    tam_vecindario=30,
    tenure_tabu=15,
    semilla=7,
    operadores=OPERADORES_POPULARES,
    usar_gpu=False,
    backend_vecindario="labels",
    guardar_historial=True,
    guardar_csv=False,
    nombre_instancia=nombre,
    usar_penalizacion_capacidad=True,
    lambda_capacidad=None,  # Calculado automáticamente
)

# Acceso a resultados
print(f"Mejor costo: {resultado.mejor_costo:.2f}")
print(f"Resultados en CSV: {resultado.archivo_csv}")

# Análisis del historial de convergencia
if resultado.historial_mejor_costo:
    primer_costo = resultado.historial_mejor_costo[0]
    ultimo_costo = resultado.historial_mejor_costo[-1]
    print(f"Costo en iter 0: {primer_costo:.2f} → iter final: {ultimo_costo:.2f}")

# Trayectoria de operadores hasta la mejor solución
print("\nTrayectoria de operadores hasta la mejor solución:")
for op, conteo in resultado.operadores_trayectoria_mejor.items():
    print(f"  {op}: {conteo}")
```

---

## Guía de ajuste de parámetros

| Parámetro | Efecto en exploración vs explotación | Instancia pequeña (<50 arcos) | Instancia mediana (50–200 arcos) | Instancia grande (>200 arcos) |
|---|---|---|---|---|
| `iteraciones` | Más iteraciones → mayor exploración global, pero más tiempo | 100–300 | 300–600 | 600–1 500 |
| `tam_vecindario` | Mayor → más candidatos evaluados por iteración (exploración local más densa) | 10–20 | 20–40 | 30–60 |
| `tenure_tabu` | Mayor → fuerza exploración; menor → permite retomar movimientos útiles antes | 5–10 | 10–25 | 20–40 |
| `semilla` | No afecta exploración; controla reproducibilidad | `None` o entero fijo | `None` o entero fijo | `None` o entero fijo |
| `usar_gpu` | Sin efecto en calidad; acelera la evaluación en lote en instancias grandes | `False` | `False` o `True` | `True` si CuPy disponible |
| `lambda_capacidad` | Mayor λ → penaliza más fuertemente las infactibilidades; menor → acepta infactibles con más facilidad | `None` (auto) | `None` (auto) | `None` (auto) o calibrado manualmente |
| `usar_penalizacion_capacidad` | `True` → permite explorar infactibles temporalmente; `False` → restricción estricta en cada paso | `True` | `True` | `True` |
| `operadores` | Subconjunto de operadores → exploración más dirigida; conjunto completo → más diversa | `OPERADORES_POPULARES` | `OPERADORES_POPULARES` | `OPERADORES_POPULARES` o subconjunto |

**Regla general de equilibrio exploración / explotación:**

- Para priorizar **explotación** (buscar a fondo cerca de soluciones buenas): usar `tenure_tabu` bajo y `tam_vecindario` moderado.
- Para priorizar **exploración** (salir de regiones ya visitadas): usar `tenure_tabu` alto e `iteraciones` alto.
- En experimentos repetidos con `repeticion`, fijar `semilla` a un valor diferente por repetición para muestrear el espacio de forma más uniforme.

---

## Notas de implementación

### Limpieza periódica de la lista tabú

Para mantener acotado el uso de memoria, el diccionario `tabu_hasta` se limpia cada 25 iteraciones eliminando todas las entradas cuya iteración de expiración ya ha pasado:

```python
if it % 25 == 0 and tabu_hasta:
    for k in [k for k, vence in tabu_hasta.items() if vence <= it]:
        del tabu_hasta[k]
```

Este comportamiento es transparente para el algoritmo: las entradas eliminadas son movimientos que ya no están prohibidos, por lo que su eliminación no altera la lógica de la lista tabú. En corridas largas con muchos operadores y alto `tam_vecindario`, esta limpieza evita que el diccionario crezca ilimitadamente.

### Evaluación en lote con GPU

Cuando `usar_gpu=True` y CuPy está disponible, `costo_lote_penalizado_ids()` evalúa **el lote completo de vecinos de una iteración en un solo kernel de GPU**. El contexto de evaluación (`ContextoEvaluacion`) copia la matriz Dijkstra a la memoria de la GPU una única vez durante la inicialización:

```python
dist_gpu = cp.asarray(D)  # copia única al construir el contexto
```

Las evaluaciones posteriores operan directamente sobre `dist_gpu` sin transferencias adicionales, lo que hace el overhead PCIe despreciable. En instancias pequeñas (pocos nodos) el overhead PCIe puede superar el beneficio de la GPU; en instancias grandes con `tam_vecindario >= 30` el speedup es significativo. Si CuPy no está instalado o no hay dispositivo CUDA, el código realiza **fallback transparente a CPU** sin lanzar excepciones: `backend_real` se registra como `"cpu"` aunque se haya solicitado `"gpu"`.

### Diferencia entre mejor solución global y mejor solución factible

El algoritmo rastrea dos mejores simultáneamente durante toda la búsqueda:

- **`mejor_any`**: la solución con menor costo puro que se ha visitado, **sin importar si viola la capacidad**. Esta variable permite que el algoritmo explore infactibles temporalmente, lo cual a menudo conduce a descubrir regiones del espacio de soluciones que luego se pueden "reparar" hacia la factibilidad.
- **`mejor_fact`**: la solución factible (violación de capacidad < 1e-12) con menor costo puro. Esta es la solución con validez real para el problema CARP.

La función `costo_para_reporte()` y el criterio de aspiración siempre trabajan sobre el mejor de los dos que esté disponible, dando prioridad a `mejor_fact` cuando existe. Al finalizar la corrida, la solución retornada es `mejor_fact_s` si se encontró al menos una solución factible durante la búsqueda; en caso contrario, se retorna `mejor_any_s` y el campo `mejor_solucion_factible_final` queda en `False`. Esto garantiza que siempre se retorne la mejor solución posible encontrada, comunicando de forma explícita su factibilidad.
