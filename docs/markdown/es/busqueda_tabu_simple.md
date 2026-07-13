# Búsqueda Tabú Simple para CARP

La Búsqueda Tabú Simple (*Tabu Search*, TS simple) implementada en `metacarp.busqueda_tabu_simple` es la versión clásica y didáctica del algoritmo de Glover (1986): lista FIFO de longitud fija, selección best-improvement sobre un lote muestreado, y aspiración clásica. Sirve como referencia de tesis y como punto de partida limpio antes de estudiar extensiones reactivas.

---

## Introducción conceptual

### La memoria de corto plazo

Una búsqueda local voraz queda atrapada en el primer mínimo local que encuentra: ningún vecino mejora la solución actual, el algoritmo se detiene. La Búsqueda Tabú escapa de esa trampa moviéndose **siempre al mejor vecino disponible**, aunque sea peor que la posición actual. El precio de esa libertad es que podría oscilar entre las mismas pocas soluciones indefinidamente: la **lista tabú** lo impide. Registra los movimientos ejecutados recientemente y los prohíbe temporalmente durante `tabu_tenure` iteraciones, forzando al algoritmo a explorar regiones aún no visitadas.

### Por qué es adecuada para CARP

El espacio de soluciones del CARP (*Capacitated Arc Routing Problem*) es altamente no-convexo con numerosos mínimos locales. La Búsqueda Tabú es adecuada porque:

- Navega eficientemente el espacio de rutas sin reinicializaciones costosas.
- El vecindario multi-operador (intra e inter-ruta) permite explorar perturbaciones de diferente escala en un solo lote.
- El criterio de aspiración impide que la lista tabú bloquee el camino al óptimo global.
- La penalización de capacidad (implícita a través del sesgo `alpha_inter` / `p_inter`) guía la búsqueda hacia regiones factibles sin descartar soluciones infactibles temporalmente prometedoras.

### Diferencia con `busqueda_tabu.py`

El módulo `busqueda_tabu.py` ya existente en el proyecto implementa una versión más compleja con limpieza periódica del diccionario y tenencia fija gestionada con un dict. `busqueda_tabu_simple.py` es la versión didáctica: lista tabú FIFO explícita implementada con `collections.deque(maxlen=tabu_tenure)` + set paralelo para lookup O(1), parámetros instance-aware y doble criterio de parada.

---

## Cómo funciona paso a paso

### Diagrama de flujo ASCII

```
INICIO
  |
  v
Construir ContextoEvaluacion (una sola vez: matriz Dijkstra + arrays NumPy)
  |
  v
Calibración instance-aware:
  n          = número de tareas requeridas (len(ctx.encoding.id_to_label))
  iter_max   = max(50, 20·n)           si iteraciones_max no se pasa
  sin_mejora = max(20, 5·n)            si max_iter_sin_mejora no se pasa
  tam_vec    = max(20, 2·n)            si tam_vecindario no se pasa
  tenure     = max(3, round(sqrt(n)))  regla Glover-Laguna  ← FIJO en TS simple
  |
  v
Seleccionar mejor solución inicial (seleccionar_mejor_inicial_rapido)
  |
  v
Inicializar:
  sol_actual    = sol_inicial
  mejor_costo   = costo(sol_inicial)
  lista_tabu    = deque(maxlen=tabu_tenure)  +  set_tabu = set()  +  counter_tabu = Counter()
  iter_sin_mejora = 0
  |
  v
+----------------------------------------------------------------+
|  BUCLE PRINCIPAL: mientras iteracion < iteraciones_max        |
|  Y iter_sin_mejora < max_iter_sin_mejora                      |
|                                                               |
|  -- Selección de grupo de operadores (sesgo p_inter) --       |
|  viol_actual = exceso_capacidad(sol_actual)                   |
|  p_ef = alpha_inter si viol_actual > 1e-12                    |
|         p_inter     si factible                               |
|  dado = rng.random()                                          |
|  si dado < p_ef  → grupo = ops_inter (inter-ruta)             |
|  si no           → grupo = ops_intra (intra-ruta)             |
|  (fallback al conjunto completo si algún grupo está vacío)    |
|                                                               |
|  -- Generación del lote (tam_vecindario vecinos) --           |
|  para i en range(tam_vecindario):                             |
|      vecino_i, mov_i = generar_vecino(sol_actual,             |
|                                       operadores=grupo)       |
|      contador.proponer(mov_i.operador)                        |
|                                                               |
|  -- Evaluación en lote (una sola pasada NumPy) --             |
|  costos = costo_lote_ids(vecinos, ctx)                        |
|                                                               |
|  -- Selección best-improvement no-tabú --                     |
|  para cada vecino idx:                                        |
|      key = _clave_tabu(mov_idx)                               |
|      es_tabu = key in set_tabu                                |
|      aspiracion = costos[idx] < mejor_costo - 1e-15           |
|      si es_tabu AND NOT aspiracion:  continuar                |
|      si costos[idx] < mejor_admisible:  elegido = idx         |
|                                                               |
|  si ningún admisible:                                         |
|      elegido = idx con menor costo (fallback tabú)            |
|      iteraciones_todos_tabu += 1                              |
|                                                               |
|  -- Movimiento --                                             |
|  sol_actual = vecinos[elegido]                                |
|  nueva_clave = _clave_tabu(mov_elegido)                       |
|  si lista_tabu llena:                                         |
|      descartada = lista_tabu[0]                               |
|      counter_tabu[descartada] -= 1                            |
|      si counter_tabu[descartada] == 0:                        |
|          set_tabu.discard(descartada)  (O(1), sin escaneo)    |
|  lista_tabu.append(nueva_clave)                               |
|  set_tabu.add(nueva_clave)                                    |
|  counter_tabu[nueva_clave] += 1                               |
|                                                               |
|  -- Actualización mejor global --                             |
|  si costos[elegido] < mejor_costo - 1e-15:                    |
|      mejor_costo = costos[elegido]                            |
|      mejor_sol  = copia(sol_actual)                           |
|      iter_sin_mejora = 0                                      |
|  si no:                                                       |
|      iter_sin_mejora += 1                                     |
|                                                               |
|  iteracion += 1                                               |
+----------------------------------------------------------------+
  |
  v
Calcular métricas (mejora_absoluta, mejora_porcentaje)
  |
  v
Opcional: guardar CSV
  |
  v
Retornar BusquedaTabuSimpleResult
  |
  v
FIN
```

---

## Componentes clave

### Lista tabú FIFO (`deque` + `set` + `Counter`)

```python
from collections import Counter, deque

lista_tabu:   deque[tuple[Any, ...]]           = deque(maxlen=tabu_tenure)
set_tabu:     set[tuple[Any, ...]]             = set()
counter_tabu: Counter[tuple[Any, ...]]         = Counter()
```

La `deque` con `maxlen=tabu_tenure` implementa el comportamiento FIFO automáticamente: al insertar más allá de la capacidad, Python descarta el elemento más antiguo sin intervención explícita. El `set_tabu` paralelo permite comprobar si un movimiento está prohibido en O(1), frente al O(n) que implicaría recorrer la `deque`. El `Counter` paralelo resuelve el problema de los **duplicados en la lista tabú**: cuando la `deque` descarta una clave por desbordamiento, el `Counter` dice en O(1) si esa clave sigue presente en la `deque` (frecuencia > 0 tras decrementar) o si se puede eliminar del `set_tabu` con seguridad. Sin el `Counter`, detectar duplicados requeriría un `list(lista_tabu).count(clave)` en O(n), lo que afectaría al rendimiento en tenencias largas.

### Clave tabú (`_clave_tabu`)

Identifica un movimiento de forma unívoca mediante una tupla que incluye: nombre del operador, índices de rutas (`ruta_a`, `ruta_b`), posiciones internas (`i`, `j`, `k`, `l`) y etiquetas de las tareas desplazadas. Dos movimientos con la misma clave se consideran el mismo cambio estructural, independientemente de la iteración en que ocurran.

### Criterio de aspiración clásico

```python
aspiracion = c_v < mejor_costo - 1e-15
```

Si un movimiento está en la lista tabú pero su costo es estrictamente mejor que el mejor global conocido, se acepta de todas formas. Esto evita que la lista tabú cierre el camino al óptimo global.

### Doble criterio de parada

El bucle termina cuando se cumple cualquiera de las dos condiciones:

1. `iteracion >= iteraciones_max` — límite duro de iteraciones.
2. `iter_sin_mejora >= max_iter_sin_mejora` — estancamiento: demasiadas iteraciones consecutivas sin mejorar el mejor global.

---

## La función `busqueda_tabu_simple`

### Firma exacta

```python
def busqueda_tabu_simple(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    iteraciones_max: int = 400,
    max_iter_sin_mejora: int = 100,
    tam_vecindario: int = 25,
    tabu_tenure: int = 20,
    semilla: int | None = None,
    operadores: Iterable[str] = OPERADORES_POPULARES,
    marcador_depot_etiqueta: str | None = None,
    usar_gpu: bool = False,
    backend_vecindario: Literal["labels", "ids"] = "labels",
    guardar_historial: bool = True,
    guardar_csv: bool = False,
    ruta_csv: str | None = None,
    nombre_instancia: str = "instancia",
    repeticion: int | None = None,
    root: str | None = None,
    extra_csv: dict[str, object] | None = None,
    alpha_inter: float | None = None,
    p_inter: float | None = None,
    intensificador: Callable | None = None,
    **_ignorado_kwargs: object,
) -> BusquedaTabuSimpleResult
```

> **Nota:** Los parámetros `id_corrida` y `config_id` son absorbidos por `**_ignorado_kwargs` y **no se escriben en el CSV**. Esta es la convención del proyecto.

### Tabla de parámetros

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `inicial_obj` | `Any` | — | Objeto pickle con la(s) solución(es) inicial(es). Se extraen recursivamente todas las estructuras de tipo solución CARP. |
| `data` | `Mapping[str, Any]` | — | Datos de la instancia CARP (capacidad, demandas, BKS, etc.), obtenidos con `load_instances`. |
| `G` | `nx.Graph` | — | Grafo de la instancia cargado desde GEXF. Se usa para construir el contexto si no existe la matriz precalculada. |
| `iteraciones_max` | `int` | `400` | Cota dura de iteraciones del bucle principal (criterio de parada 1). |
| `max_iter_sin_mejora` | `int` | `100` | Iteraciones consecutivas sin mejorar el mejor global antes de detener la búsqueda (criterio de parada 2). |
| `tam_vecindario` | `int` | `25` | Vecinos generados y evaluados por iteración (best-improvement sobre el lote). |
| `tabu_tenure` | `int` | `20` | Longitud de la lista tabú FIFO. Regla clásica de Glover-Laguna: `tenure ≈ √n`. |
| `semilla` | `int \| None` | `None` | Semilla para `random.Random`. `None` produce corridas no reproducibles. |
| `operadores` | `Iterable[str]` | `OPERADORES_POPULARES` | Conjunto de operadores de vecindario habilitados (9 en total). |
| `marcador_depot_etiqueta` | `str \| None` | `None` | Etiqueta del nodo depósito (ej. `"D"`). `None` la lee del contexto. |
| `usar_gpu` | `bool` | `False` | Flag de GPU. Solo se pasa al contexto para trazabilidad; el bucle principal ya es eficiente en CPU. |
| `backend_vecindario` | `Literal["labels", "ids"]` | `"labels"` | Representación interna para generar vecinos. `"ids"` usa enteros (más rápido en lotes grandes). |
| `guardar_historial` | `bool` | `True` | Si `True`, registra el mejor costo al inicio de cada iteración en `historial_mejor_costo`. |
| `guardar_csv` | `bool` | `False` | Si `True`, escribe una fila de resultados en el CSV al finalizar. |
| `ruta_csv` | `str \| None` | `None` | Ruta del CSV. Si `None` y `guardar_csv=True`, se genera como `resultados_busqueda_tabu_simple_{nombre_instancia}.csv`. |
| `nombre_instancia` | `str` | `"instancia"` | Identificador de la instancia para el CSV y la carga del contexto desde caché. |
| `repeticion` | `int \| None` | `None` | Número de repetición dentro de un experimento. Se escribe en el CSV. |
| `root` | `str \| None` | `None` | Directorio raíz alternativo para localizar los archivos de la instancia. |
| `extra_csv` | `dict[str, object] \| None` | `None` | Columnas adicionales escritas en el CSV. |
| `alpha_inter` | `float \| None` | `None` → `0.8` | Probabilidad de elegir el grupo inter-ruta cuando la solución actual **viola** capacidad. `None` usa el mismo default que SA (`0.8`). |
| `p_inter` | `float \| None` | `None` → `0.6` | Probabilidad de elegir el grupo inter-ruta cuando la solución es **factible**. `None` usa el mismo default que SA (`0.6`). |
| `intensificador` | `Callable \| None` | `None` | Hook opcional de intensificación. Si se provee, se invoca en el punto de estancamiento (`max_iter_sin_mejora`) en lugar del kick aleatorio con la firma `intensificador(sol, mejor_global, ctx, lam, rng, encoding, md)`. Con `None` (default) el comportamiento es idéntico al anterior. |

### Parámetros instance-aware

Cuando `iteraciones_max`, `max_iter_sin_mejora` y `tam_vecindario` están en sus valores por defecto fijos de la firma (400, 100, 25), el script de experimentos `run_tabu_simple_automatico.py` los pasa explícitamente. La función también tiene sentido con los valores por defecto para uso interactivo. La **regla `tabu_tenure ≈ √n`** (Glover & Laguna 1997) se aplica manualmente al configurar el experimento, no dentro de la función (ver script de experimentos).

### Qué retorna

La función retorna un objeto `BusquedaTabuSimpleResult` inmutable descrito a continuación.

### Función de conveniencia: `busqueda_tabu_simple_desde_instancia`

```python
def busqueda_tabu_simple_desde_instancia(
    nombre_instancia: str,
    *,
    root: str | None = None,
    # mismos parámetros que busqueda_tabu_simple, sin inicial_obj, data, G
) -> BusquedaTabuSimpleResult
```

Carga automáticamente `data`, `G` e `inicial_obj` desde el nombre de la instancia y llama a `busqueda_tabu_simple`. Equivalente a:

```python
data        = load_instances(nombre_instancia, root=root)
G           = cargar_objeto_gexf(nombre_instancia, root=root)
inicial_obj = cargar_solucion_inicial(nombre_instancia, root=root)
resultado   = busqueda_tabu_simple(inicial_obj, data, G, ...)
```

---

## `BusquedaTabuSimpleResult`

Dataclass inmutable (`frozen=True, slots=True`) con todos los resultados de una corrida.

### Tabla de campos

| Campo | Tipo | Descripción |
|---|---|---|
| `mejor_solucion` | `list[list[str]]` | Mejor solución encontrada. Lista de rutas; cada ruta incluye el marcador depósito `"D"`. |
| `mejor_costo` | `float` | Costo de `mejor_solucion` (costo puro, sin penalización). |
| `solucion_inicial_referencia` | `list[list[str]]` | Solución inicial usada como referencia para medir la mejora. |
| `costo_solucion_inicial` | `float` | Costo de la solución inicial. |
| `mejora_absoluta` | `float` | `costo_inicial - mejor_costo`. Positivo indica mejora real. |
| `mejora_porcentaje_inicial_vs_final` | `float` | `mejora_absoluta / costo_inicial × 100`. |
| `tiempo_segundos` | `float` | Tiempo total de ejecución (medido con `time.perf_counter`). |
| `iteraciones_totales` | `int` | Iteraciones ejecutadas (puede ser menor que `iteraciones_max` si paró por estancamiento). |
| `vecinos_evaluados` | `int` | Total de soluciones vecinas evaluadas (`iteraciones_totales × tam_vecindario`). |
| `iteracion_mejor` | `int` | Iteración en la que se descubrió la mejor solución. |
| `iteraciones_sin_mejora_final` | `int` | Iteraciones consecutivas sin mejora al terminar la corrida. |
| `aspiraciones` | `int` | Veces que el criterio de aspiración rescató un movimiento tabú. |
| `iteraciones_todos_tabu` | `int` | Veces que todos los vecinos del lote eran tabú (se eligió el mejor tabú como fallback). |
| `mejoras` | `int` | Veces que el mejor global se actualizó. |
| `semilla` | `int \| None` | Semilla usada en esta corrida. |
| `backend_evaluacion` | `str` | Backend real de evaluación: `"cpu"` o `"gpu"`. |
| `historial_mejor_costo` | `list[float]` | Mejor costo al inicio de cada iteración (vacío si `guardar_historial=False`). |
| `ultimo_movimiento_aceptado` | `MovimientoVecindario \| None` | Último movimiento aceptado al terminar. |
| `operadores_propuestos` | `dict[str, int]` | Veces que cada operador fue propuesto para generar un vecino. |
| `operadores_aceptados` | `dict[str, int]` | Veces que cada operador fue aceptado. |
| `operadores_mejoraron` | `dict[str, int]` | Veces que cada operador produjo una mejora del mejor global. |
| `operadores_trayectoria_mejor` | `dict[str, int]` | Snapshot de `operadores_aceptados` en el momento de la mejor solución. |
| `mejor_solucion_factible_final` | `bool` | `True` si la mejor solución respeta la restricción de capacidad. |
| `archivo_csv` | `str \| None` | Ruta absoluta del CSV guardado, o `None` si `guardar_csv=False`. |
| `alpha_inter_aplicado` | `float` | Valor efectivo de `alpha_inter` usado en esta corrida. |
| `p_inter_aplicado` | `float` | Valor efectivo de `p_inter` usado en esta corrida. |
| `iteraciones_con_violacion` | `int` | Iteraciones del bucle principal en las que la solución actual violaba capacidad. |

---

## Columnas del CSV de salida

El CSV se guarda en `experimentos/tabu_simple_small_20260517/` cuando se usa `run_tabu_simple_automatico.py`. La etiqueta de metaheurística en el CSV es `"busqueda_tabu_simple"`.

> **Nota:** Los CSV del proyecto **no incluyen** las columnas `id_corrida` ni `config_id`. Esos parámetros son absorbidos silenciosamente por `**_ignorado_kwargs` pero no se escriben en `fila`.

### Columnas de identificación y parámetros

| Columna | Descripción |
|---|---|
| `metaheuristica` | Siempre `"busqueda_tabu_simple"` |
| `instancia` | Nombre de la instancia (ej. `gdb1`) |
| `bks_referencia` | Valor BKS de la literatura |
| `bks_origen` | Fuente del BKS (`"BKS"`, `"GAP_Value"`, `"no_disponible"`) |
| `gap_bks_porcentaje` | `(mejor_costo - BKS) / BKS × 100` |
| `repeticion` | Número de repetición dentro del experimento |
| `semilla` | Semilla usada |
| `tiempo_segundos` | Duración real de la corrida |
| `mejor_costo` | Mejor costo encontrado |
| `costo_solucion_inicial` | Costo de la solución inicial |
| `mejora_absoluta` | `costo_inicial - mejor_costo` |
| `mejora_porcentaje` | `mejora_absoluta / costo_inicial × 100` |
| `iteraciones_max` | Cota dura de iteraciones usada |
| `max_iter_sin_mejora` | Umbral de estancamiento usado |
| `tam_vecindario` | Tamaño del lote de vecinos |
| `tabu_tenure` | Longitud de la lista tabú usada |

### Columnas de operadores (36 columnas: 4 categorías × 9 operadores)

Formato `<categoria>_<operador>`. Categorías: `propuesto`, `aceptado`, `mejoraron`, `trayectoria_mejor`. Operadores: los 9 de `OPERADORES_POPULARES`.

### Columnas de estadísticas de corrida

| Columna | Descripción |
|---|---|
| `iteraciones_totales` | Iteraciones ejecutadas |
| `vecinos_evaluados` | Total de vecinos evaluados |
| `iteracion_mejor` | Iteración en que se encontró la mejor solución |
| `iteraciones_sin_mejora_final` | Iteraciones sin mejora al terminar |
| `aspiraciones` | Activaciones del criterio de aspiración |
| `iteraciones_todos_tabu` | Iteraciones con todos los vecinos tabú |
| `aceptadas` | Total de movimientos aceptados |
| `mejoras` | Total de mejoras del mejor global |
| `mejor_solucion_factible_final` | Si la mejor solución es factible |
| `mejor_solucion_tr_legible` | Representación textual de la solución |
| `reporte_detalle_deadheading` | Desglose de costos de arrastre por ruta |
| `costo_total_desde_reporte` | Verificación cruzada del costo |

### Columnas del sesgo inter/intra (nuevas)

| Columna | Descripción |
|---|---|
| `alpha_inter` | Valor efectivo de `alpha_inter` |
| `p_inter` | Valor efectivo de `p_inter` |
| `iteraciones_con_violacion` | Iteraciones donde la solución actual violaba capacidad |
| `fraccion_iter_con_violacion` | `iteraciones_con_violacion / iteraciones_totales` |

---

## Ejemplos de uso

### Opción 1: función de conveniencia (recomendada)

```python
from metacarp.busqueda_tabu_simple import busqueda_tabu_simple_desde_instancia

resultado = busqueda_tabu_simple_desde_instancia(
    "gdb1",
    iteraciones_max=400,
    max_iter_sin_mejora=100,
    tam_vecindario=25,
    tabu_tenure=20,
    semilla=42,
    guardar_historial=True,
    guardar_csv=True,
    ruta_csv="resultados/tabu_simple_gdb1.csv",
    repeticion=1,
)

print(f"Mejor costo           : {resultado.mejor_costo:.2f}")
print(f"Costo inicial         : {resultado.costo_solucion_inicial:.2f}")
print(f"Mejora absoluta       : {resultado.mejora_absoluta:.2f}")
print(f"Mejora porcentual     : {resultado.mejora_porcentaje_inicial_vs_final:.2f} %")
print(f"Tiempo de ejecución   : {resultado.tiempo_segundos:.2f} s")
print(f"Iteraciones ejecutadas: {resultado.iteraciones_totales}")
print(f"Aspiraciones          : {resultado.aspiraciones}")
print(f"Solución factible     : {resultado.mejor_solucion_factible_final}")
```

### Opción 2: carga manual de recursos

```python
from metacarp.busqueda_tabu_simple import busqueda_tabu_simple
from metacarp.instances import load_instances
from metacarp.cargar_grafos import cargar_objeto_gexf
from metacarp.cargar_soluciones_iniciales import cargar_solucion_inicial
from metacarp.vecindarios import OPERADORES_POPULARES

nombre = "gdb1"
data        = load_instances(nombre)
G           = cargar_objeto_gexf(nombre)
inicial_obj = cargar_solucion_inicial(nombre)

resultado = busqueda_tabu_simple(
    inicial_obj,
    data,
    G,
    iteraciones_max=400,
    max_iter_sin_mejora=100,
    tam_vecindario=25,
    tabu_tenure=20,
    semilla=42,
    operadores=OPERADORES_POPULARES,
    guardar_historial=True,
    guardar_csv=False,
    nombre_instancia=nombre,
    # alpha_inter=None  ->  usa 0.8 (mismo default que SA)
    # p_inter=None      ->  usa 0.6 (mismo default que SA)
)

# Ver la mejor solución encontrada
for i, ruta in enumerate(resultado.mejor_solucion, start=1):
    print(f"Ruta {i}: {' -> '.join(ruta)}")
```

### Script de experimentos automático

```bash
# Corrida básica (6 tenures × 23 instancias × 2 repeticiones = 276 corridas)
python scripts/run_tabu_simple_automatico.py

# Con directorio de salida personalizado
python scripts/run_tabu_simple_automatico.py --salida-dir mis_experimentos

# Con más repeticiones y tenure personalizado
python scripts/run_tabu_simple_automatico.py \
    --repeticiones 5 \
    --iteraciones-max 600 \
    --tam-vecindario 30
```

Los CSV se guardan en `<salida-dir>/tabu_simple_small_20260517/`.

---

## Guía de ajuste de parámetros

| Parámetro | Efecto con valor bajo | Efecto con valor alto | Recomendación |
|---|---|---|---|
| `tabu_tenure` | Lista corta; riesgo de ciclos cortos; más libertad de movimiento | Lista larga; mayor diversificación pero prohibiciones excesivas | Usar regla `round(√n)`. Para `gdb1` (`n ≈ 12`): tenure ≈ 3-4. Para `EGL-E1-A` (`n ≈ 190`): tenure ≈ 14. |
| `iteraciones_max` | Búsqueda corta; menos calidad | Más iteraciones; mayor costo computacional | 300–600 para instancias pequeñas. |
| `max_iter_sin_mejora` | Para rápido si se estanca | Persiste aunque no haya progreso | `20–30 %` de `iteraciones_max`. |
| `tam_vecindario` | Menos candidatos por iter; decisión local pobre | Más candidatos; mejor decisión local pero más cómputo por iter | 20–40 para instancias pequeñas. |
| `alpha_inter` | Menos énfasis en reparación de capacidad | Más agresivo al reparar infactibilidades | `0.8` (default, igual que SA). |
| `p_inter` | Selección casi uniforme cuando es factible | Sesga fuertemente hacia inter-ruta en todo momento | `0.6` (default, igual que SA). |

---

## Pseudocódigo

```
TS-Simple(sol_inicial, tabu_tenure, iteraciones_max, max_iter_sin_mejora,
          tam_vecindario, alpha_inter, p_inter):

  s     ← sol_inicial
  s*    ← s
  c*    ← costo(s)
  T     ← deque(maxlen = tabu_tenure)   // lista tabú FIFO
  T_set ← {}                            // set para lookup O(1)
  iter  ← 0
  sin_mejora ← 0

  mientras iter < iteraciones_max Y sin_mejora < max_iter_sin_mejora:

    // Seleccionar grupo de operadores (sesgo p_inter / alpha_inter)
    viol ← exceso_capacidad(s)
    p_ef ← alpha_inter si viol > ε, p_inter si no
    si random() < p_ef: grupo ← ops_inter     // random() ~ Uniforme[0,1)
    si no:              grupo ← ops_intra
    // El operador concreto se elige luego UNIFORMEMENTE dentro del grupo

    // Generar lote de vecinos
    N ← {generar_vecino(s, grupo) para _ en range(tam_vecindario)}
    evaluar_lote(N)

    // Seleccionar el mejor no-tabú con aspiración
    s_mejor ← argmin { costo(v) : v ∈ N,
                        (_clave(v) ∉ T_set)
                        OR (costo(v) < c* - ε) }   // aspiración

    si N completo tabú:
        s_mejor ← argmin { costo(v) : v ∈ N }      // fallback

    // Actualizar lista tabú
    si T llena: T_set.discard( T[0] )
    T.append( _clave(s_mejor) )
    T_set.add( _clave(s_mejor) )

    s ← s_mejor

    // Actualizar mejor global
    si costo(s) < c* - ε:
        s* ← s;  c* ← costo(s);  sin_mejora ← 0
    si no:
        sin_mejora ← sin_mejora + 1

    iter ← iter + 1

  retornar s*, c*
```

---

## Notas de implementación

### Evaluación en lote con NumPy

En cada iteración, los `tam_vecindario` vecinos se codifican a IDs enteros con `encode_solution` y se evalúan con `costo_lote_ids(sols_ids, ctx)`, que vectoriza el cálculo con fancy indexing de NumPy sobre la matriz Dijkstra precalculada. Esto es 10×–50× más rápido que evaluar uno por uno con el evaluador clásico basado en NetworkX.

### Sesgo inter/intra compartido con SA y RTS

El mecanismo de sesgo está centralizado en `seleccionar_grupo_operadores_inter_intra` (módulo `metaheuristicas_utils`). La función realiza exactamente un `rng.random()` —un sorteo **uniforme** en `[0,1)`— por iteración del bucle principal y devuelve el grupo de operadores elegido; el operador concreto se sortea después **uniformemente** dentro de ese grupo (`rng.choice`). Esto preserva la reproducibilidad bit-a-bit: la secuencia de números aleatorios del TS simple con `semilla=42` es idéntica en términos de estructura a la del SA o RTS con la misma semilla.

---

## Referencias

- Glover, F. (1986). "Future paths for integer programming and links to artificial intelligence." *Computers & Operations Research*, 13(5), 533–549.
- Glover, F., & Laguna, M. (1997). *Tabu Search*. Kluwer Academic Publishers.
