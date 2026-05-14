# Recocido Simulado para CARP

El Recocido Simulado (*Simulated Annealing*, SA) es una metaheurística de búsqueda local que resuelve problemas de optimización combinatoria difícil, como el CARP (*Capacitated Arc Routing Problem*). Esta documentación describe la implementación concreta del módulo `metacarp.recocido_simulado`.

---

## Introducción conceptual

### La analogía metalúrgica

En metalurgia, el **recocido** (*annealing*) es un proceso de tratamiento térmico: se calienta un metal a temperatura muy alta y luego se enfría de forma controlada y lenta. A alta temperatura, los átomos del metal tienen suficiente energía cinética para moverse libremente y reorganizarse. A medida que la temperatura baja, esa movilidad disminuye. Si el enfriamiento es suficientemente lento, los átomos tienen tiempo de acomodarse en las configuraciones de **mínima energía** — estructuras cristalinas perfectas, el estado termodinámicamente estable. Si, en cambio, el enfriamiento es brusco (*temple*), los átomos quedan "atrapados" en configuraciones subóptimas con muchos defectos — los llamados *mínimos locales*.

La analogía con la optimización combinatoria es directa:

| Metalurgia | Optimización |
|---|---|
| Átomo en una configuración | Solución candidata |
| Energía del sistema | Costo de la solución (a minimizar) |
| Temperatura alta | Alta tolerancia a soluciones peores |
| Temperatura baja | Solo se aceptan mejoras |
| Cristal perfecto | Mínimo global (o muy buena solución) |
| Defecto en el cristal | Mínimo local |

### Por qué el RS escapa de mínimos locales

Una búsqueda local voraz (solo acepta mejoras) queda atrapada en el primer mínimo local que encuentra: ningún movimiento del vecindario mejora la solución actual, así que el algoritmo se detiene. El RS lo evita mediante la **regla de Metropolis**: con temperatura alta, incluso soluciones peores que la actual pueden aceptarse con probabilidad positiva. Esto permite que el algoritmo "suba colinas" en el paisaje del espacio de búsqueda, escapando del mínimo local y explorando regiones que una búsqueda voraz nunca visitaría. A medida que la temperatura disminuye, esa tolerancia se reduce progresivamente hasta que el algoritmo converge de forma casi voraz hacia el mejor mínimo encontrado.

---

## El criterio de Metropolis

### Formulación

En cada iteración se genera una solución vecina y se compara su costo con el de la solución actual:

- Si el vecino es **mejor o igual** (`delta <= 0`): se acepta siempre.
- Si el vecino es **peor** (`delta > 0`): se acepta con probabilidad

```
P = exp(-delta / T)
```

donde:

- **`delta`** = `costo_vecino - costo_actual`: la diferencia de costo entre el vecino y la solución actual. Un valor positivo indica que el vecino es peor.
- **`T`** = temperatura actual del sistema: controla cuánta tolerancia hay hacia empeoramientos. Al inicio `T` es alta; va disminuyendo en cada ciclo de enfriamiento.
- **`exp`** es la función exponencial natural (`math.exp` en Python).

Cuando la temperatura `T` es **grande**, `exp(-delta / T)` tiende a 1 incluso para valores positivos de `delta`, por lo que el vecino peor se acepta casi siempre. Cuando `T` tiende a 0, `exp(-delta / T)` tiende a 0, y la regla se convierte en una búsqueda voraz: solo se aceptan mejoras.

En código (tomado directamente de `recocido_simulado.py`):

```python
if delta <= 0:
    aceptar = True
else:
    aceptar = rng.random() < math.exp(-delta / T)
```

`rng.random()` genera un número uniforme en `[0, 1)`. Si ese número es menor que la probabilidad de Metropolis, se acepta el vecino peor.

### Tabla de ejemplos numéricos

La siguiente tabla muestra cómo cambia la probabilidad de aceptar un vecino peor en función de `delta` y `T`. Los valores ilustran por qué la temperatura alta es clave para la exploración.

| `delta` (empeoramiento) | `T = 1000` | `T = 100` | `T = 10` | `T = 1` | `T = 0.01` |
|---|---|---|---|---|---|
| 1 | 0.999 | 0.990 | 0.905 | 0.368 | ~0.000 |
| 5 | 0.995 | 0.951 | 0.607 | 0.007 | ~0.000 |
| 10 | 0.990 | 0.905 | 0.368 | ~0.000 | ~0.000 |
| 50 | 0.951 | 0.607 | 0.007 | ~0.000 | ~0.000 |
| 100 | 0.905 | 0.368 | ~0.000 | ~0.000 | ~0.000 |
| 500 | 0.607 | 0.007 | ~0.000 | ~0.000 | ~0.000 |

**Lectura de la tabla**: con `T = 1000` y un empeoramiento de 100 unidades de costo, la probabilidad de aceptar el vecino peor es del 90.5 %. Con `T = 1`, esa misma diferencia ya tiene una probabilidad prácticamente nula (~0.000). La transición entre exploración y explotación ocurre de forma gradual conforme T decrece.

---

## El enfriamiento geométrico

### Fórmula

Al finalizar cada nivel de temperatura (después de ejecutar `L = n²` evaluaciones, donde `n` es el número de tareas de la instancia), la temperatura se reduce multiplicándola por el factor `alpha`:

```
T_nueva = T_actual * alpha
```

Como `0 < alpha < 1`, la temperatura decrece de forma **geométrica** (exponencialmente en el tiempo). El módulo documenta valores típicos de `alpha` entre 0.90 y 0.99. En código:

```python
T *= alpha
enfriamientos += 1
```

### Interacción entre parámetros

Los parámetros que definen la "trayectoria de enfriamiento" son:

| Parámetro | Rol |
|---|---|
| `n_tareas` | Número de arcos requeridos de la instancia. Se calcula automáticamente. |
| `L = n²` | Longitud de la cadena de Markov (evaluaciones por nivel de T). Se calcula automáticamente desde n. |
| `temperatura_inicial` | Temperatura de partida. Si es `None`, se calcula como `5 · d_max / n`. |
| `temperatura_minima` | Umbral de parada. Si es `None`, se calcula como `20 · d_max / n²`. |
| `alpha` | Factor de reducción por ciclo. Más alto = enfriamiento más lento = más calidad, más tiempo. |

El número total de evaluaciones que realiza el algoritmo es:

```
iteraciones_totales = L × n_niveles_temperatura = n² × ceil(log(T_init/T_min) / log(1/alpha))
```

El algoritmo termina cuando `T` cae por debajo de `temperatura_minima`.

---

## Recalentamiento (Reheat)

### Por qué el SA se estanca

El enfriamiento geométrico tiene un efecto colateral importante: cuando `T` se acerca a `temperatura_minima`, la probabilidad de aceptar empeoramientos (`exp(-delta / T)`) se vuelve casi nula. En esa fase final el SA se comporta como una búsqueda voraz y, si la solución actual está atrapada en un mínimo local cuya "pared" requiere aceptar empeoramientos no triviales, **el algoritmo se queda atascado sin remedio**. La temperatura ya no le da margen para escapar.

Este fenómeno se observa empíricamente en instancias donde el SA llega a soluciones de calidad muy próximas al BKS (*Best Known Solution*) pero no logra dar el salto final. Por ejemplo, en `gdb19` el SA encuentra costo 63 mientras el BKS es 55. La brecha es pequeña en términos absolutos (8 unidades), pero requiere aceptar movimientos que empeoran la solución en ese mismo orden de magnitud cuando `T` ya está en la fase fría: la probabilidad de Metropolis para `delta = 4` con `T = 0.5` es `exp(-8) ≈ 0.00034`, prácticamente imposible.

### La solución: recalentar

El mecanismo de **reheat** (recalentamiento) resuelve este problema reiniciando parcialmente la temperatura cada vez que se detecta estancamiento.

La **analogía metalúrgica** sigue siendo válida y directa: cuando un metal queda atrapado en un mínimo local de energía (con átomos detenidos en imperfecciones cristalinas), los metalúrgicos lo vuelven a calentar para devolverles movilidad. Al subir la temperatura, los átomos recuperan energía cinética y pueden reorganizarse; al enfriarse de nuevo, tienen otra oportunidad de encontrar la configuración perfecta. En el algoritmo, "calentar" significa subir `T` para reaceptar empeoramientos y permitir que la búsqueda escape del mínimo local.

A diferencia de un reinicio completo (que perdería la información acumulada), el reheat **preserva la mejor solución encontrada** (`mejor_any_s` y `mejor_fact_s` no se reinician), pero permite explorar desde la posición actual de búsqueda con tolerancia renovada a empeoramientos.

### Comportamiento de la temperatura con reheat

Sin reheat, la curva de `T` es una exponencial decreciente monótona. Con reheat, presenta saltos discretos hacia arriba cuando se detecta estancamiento:

```
T  ^
   |  T_init
   |  *
   |   \
   |    \                    reheat
   |     \                    /\
   |      \                  /  \
   |       \    plateau     /    \      reheat
   |        \..............*      \     /\
   |         \              \      \   /  \
   |          \              \      \ /    \
   |           \              \      *      \..........
   |            \              \              \
   |  T_min      *--------------*--------------*-----------
   +--------------------------------------------------> niveles
```

En cada activación del reheat, `T` salta a `T_init_eff * reheat_factor` y luego vuelve a enfriarse geométricamente. Cada reheat da al algoritmo una nueva "ventana de exploración" durante la cual puede escapar de mínimos locales mediante Metropolis.

### Parámetros del reheat

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `patience` | `int` | `50` | Número de niveles de temperatura consecutivos sin mejora del mejor global antes de activar el reheat. Si `patience = 0` el reheat está **desactivado** y se obtiene el SA clásico. Valores típicos: 20–100. |
| `reheat_factor` | `float` | `0.5` | Fracción de `T_init_eff` a la que se sube la temperatura cuando se activa el reheat. Debe estar en `(0, 1]`. Valores típicos: 0.3–0.7. |

### Activación del reheat (lógica interna)

1. Al inicio de cada nivel externo se captura `mejor_costo_antes_nivel = costo_para_reporte()`.
2. Se ejecuta el bucle interno (L iteraciones de Metropolis).
3. Tras el enfriamiento geométrico (`T = T * alpha`), se compara el costo reportable actual contra `mejor_costo_antes_nivel`:
   - Si hubo mejora: `niveles_sin_mejora = 0`.
   - Si no hubo mejora: `niveles_sin_mejora += 1`.
4. Si `patience > 0` y `niveles_sin_mejora >= patience`:
   - `T = T_init_eff * reheat_factor` (reinicio parcial de la temperatura).
   - `niveles_sin_mejora = 0`.
   - `n_reheats += 1`.

### Ejemplos numéricos

Supongamos que `T_init_eff = 1500` (calculado adaptativamente como `20 · d_max / n`).

| Configuración | `reheat_factor` | `T` tras reheat | Efecto |
|---|---|---|---|
| Reheat suave | `0.3` | 450 | Solo unas pocas aceptaciones más; explotación predominante. |
| Reheat estándar | `0.5` | 750 | Equilibrio razonable entre exploración y explotación. |
| Reheat fuerte | `0.7` | 1050 | Casi como reiniciar; explora mucho de nuevo. |
| Reheat máximo | `1.0` | 1500 | Equivale a reiniciar la temperatura desde cero. |

### Cuándo usar cada valor de `patience`

| Valor | Cuándo usar |
|---|---|
| `0` | Reheat desactivado. Útil para comparar contra el SA clásico o cuando la instancia se resuelve sin estancamiento (BKS alcanzado fácilmente). |
| `20–40` | Reheat agresivo. Útil en instancias pequeñas donde los niveles tienen pocas iteraciones internas y se quiere diversificar con frecuencia. |
| `50` | Valor por defecto. Equilibrio típico para instancias medianas. |
| `60–100` | Reheat conservador. Útil cuando se quiere dar tiempo a la fase de explotación antes de reiniciar; recomendado para instancias grandes. |

### Ejemplo de uso

```python
from metacarp.recocido_simulado import recocido_simulado_desde_instancia

# SA con reheat activado (default).
resultado = recocido_simulado_desde_instancia(
    "gdb19",
    alpha=0.97,
    patience=50,        # reheat tras 50 niveles sin mejora
    reheat_factor=0.5,  # subir T a la mitad de T_init_eff
    semilla=42,
)
print(f"Mejor costo : {resultado.mejor_costo}")
print(f"Reheats     : {resultado.n_reheats}")

# SA clásico sin reheat (para comparar).
resultado_clasico = recocido_simulado_desde_instancia(
    "gdb19",
    alpha=0.97,
    patience=0,         # reheat desactivado
    semilla=42,
)
print(f"Costo clasico : {resultado_clasico.mejor_costo}")
print(f"Reheats       : {resultado_clasico.n_reheats}  # siempre 0")
```

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
Calibracion adaptativa desde la instancia:
  n      = numero de arcos requeridos
  d_max  = max(matriz Dijkstra)
  L      = n^2                          (longitud de la cadena de Markov)
  T_init = 5 * d_max / n                (si no la pasa el usuario)
  T_min  = 20 * d_max / n^2             (si no la pasa el usuario)
  |
  v
Seleccionar la mejor solución inicial entre las candidatas del pickle
  |
  v
T = T_init
sol_actual = sol_inicial
costo_actual = costo(sol_inicial)
mejor_solucion = sol_inicial
mejor_costo = costo_actual
  |
  v
+-----------------------------------------------------------+
|  BUCLE EXTERNO: mientras T > T_min                        |
|                                                           |
|  mejor_costo_antes_nivel = costo_para_reporte()           |
|  Registrar historial (T, mejor_costo)                     |
|  |                                                        |
|  v                                                        |
|  +-----------------------------------------------------+  |
|  |  BUCLE INTERNO: para cada iter en range(L)           |  |
|  |                       (L = n^2 = adaptativo)         |  |
|  |                                                      |  |
|  |  vecino, mov = generar_vecino(sol_actual)            |  |
|  |  costo_vec  = costo_rapido(vecino, ctx)              |  |
|  |  viol_vec   = exceso_capacidad_rapido(vecino, ctx)   |  |
|  |  obj_actual = costo_actual + lambda * viol_actual    |  |
|  |  obj_vec    = costo_vec    + lambda * viol_vec       |  |
|  |  delta      = obj_vec - obj_actual                   |  |
|  |                                                      |  |
|  |  si delta <= 0:                                      |  |
|  |      aceptar = True          (vecino mejor o igual)  |  |
|  |  si delta > 0:                                       |  |
|  |      P = exp(-delta / T)                             |  |
|  |      aceptar = (rng.random() < P)   (Metropolis)    |  |
|  |                                                      |  |
|  |  si aceptar:                                         |  |
|  |      sol_actual  = vecino                            |  |
|  |      costo_actual = costo_vec                        |  |
|  |      si costo_vec < mejor_costo:                     |  |
|  |          mejor_solucion = vecino                     |  |
|  |          mejor_costo    = costo_vec                  |  |
|  +-----------------------------------------------------+  |
|                                                           |
|  T = T * alpha          (enfriamiento geometrico)         |
|                                                           |
|  --- Chequeo de reheat ---                                |
|  si costo_para_reporte() < mejor_costo_antes_nivel:       |
|      niveles_sin_mejora = 0                               |
|  si no:                                                   |
|      niveles_sin_mejora += 1                              |
|                                                           |
|  si patience > 0 y niveles_sin_mejora >= patience:        |
|      T = T_init_eff * reheat_factor   (recalentamiento)   |
|      niveles_sin_mejora = 0                               |
|      n_reheats += 1                                       |
+-----------------------------------------------------------+
  |
  v
Calcular metricas (mejora_absoluta, mejora_porcentaje)
  |
  v
Opcional: guardar CSV
  |
  v
Retornar RecocidoSimuladoResult
  |
  v
FIN
```

### Notas sobre el flujo

- El algoritmo rastreaa por separado el **mejor global sin restriccion** (`mejor_any_c`) y el **mejor factible** (sin violacion de capacidad, `mejor_fact_c`). El resultado reportado prioriza el factible.
- El `ContextoEvaluacion` se construye **una sola vez** al inicio mediante `construir_contexto_para_corrida`. Contiene la matriz Dijkstra densa precomputada y arrays NumPy con los datos de cada tarea. Esto hace que `costo_rapido` sea 10 a 50 veces mas rapido que el evaluador clasico basado en NetworkX.
- El generador aleatorio `rng = random.Random(semilla)` es local a la corrida, lo que garantiza reproducibilidad completa cuando se pasa una semilla.

---

## La funcion `recocido_simulado`

### Firma exacta

```python
def recocido_simulado(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    temperatura_inicial: float | None = None,
    temperatura_minima: float | None = None,
    alpha: float = 0.95,
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
) -> RecocidoSimuladoResult:
```

### Tabla de parametros

| Parametro | Tipo | Default | Descripcion |
|---|---|---|---|
| `inicial_obj` | `Any` | — | Objeto pickle con la solucion inicial (puede ser dict, lista u otra estructura anidada). Se extraen todas las soluciones candidatas recursivamente. |
| `data` | `Mapping[str, Any]` | — | Datos de la instancia CARP: capacidad, demandas, BKS, deposito, etc. Se obtiene con `load_instances`. |
| `G` | `nx.Graph` | — | Grafo de la instancia cargado desde GEXF. Se usa para construir el contexto si no hay matriz Dijkstra en disco. |
| `temperatura_inicial` | `float \| None` | `None` | Temperatura de inicio del recocido. Si es `None`, se calcula adaptativamente como `5 · d_max / n`. Si se pasa un valor, debe ser `> 0`. |
| `temperatura_minima` | `float \| None` | `None` | Temperatura de parada. El bucle externo termina cuando `T < temperatura_minima`. Si es `None`, se calcula adaptativamente como `20 · d_max / n²`. Si se pasa un valor, debe ser `> 0`. |
| `alpha` | `float` | `0.95` | Factor de enfriamiento geometrico. Debe estar en `(0, 1)`. Un valor cercano a 1 enfria mas lento. |
| `n_tareas` | `int` (calculado) | — | Numero de arcos requeridos de la instancia. Se obtiene como `len(ctx.u_arr)`. |
| `d_max` | `float` (calculado) | — | Distancia maxima en la matriz Dijkstra (`ctx.dist`). Se usa para calibrar las temperaturas. |
| `L` | `int` (calculado) | `n²` | Longitud de la cadena de Markov: numero de evaluaciones de vecinos por nivel de temperatura. Se calcula automaticamente desde `n_tareas`. |
| `semilla` | `int \| None` | `None` | Semilla para `random.Random`. Si es `None`, la corrida no es reproducible. |
| `operadores` | `Iterable[str]` | `OPERADORES_POPULARES` | Conjunto de operadores de vecindario habilitados. Ver tabla de operadores en la seccion de vecindarios. |
| `marcador_depot_etiqueta` | `str \| None` | `None` | Etiqueta del nodo deposito en las rutas (ej. `"D"`). Si es `None`, se lee del contexto. |
| `usar_gpu` | `bool` | `False` | Flag de GPU. En SA solo se pasa al contexto para trazabilidad; no produce speedup (ver seccion GPU). |
| `backend_vecindario` | `Literal["labels", "ids"]` | `"labels"` | Modo de generacion de vecinos. `"ids"` usa codificacion entera (mas rapido); `"labels"` usa etiquetas de texto. |
| `guardar_historial` | `bool` | `True` | Si es `True`, registra el mejor costo y la temperatura al inicio de cada nivel en `historial_mejor_costo` e `historial_temperatura`. |
| `guardar_csv` | `bool` | `False` | Si es `True`, escribe los resultados de la corrida en un archivo CSV. |
| `ruta_csv` | `str \| None` | `None` | Ruta del archivo CSV. Si es `None` y `guardar_csv=True`, se genera automaticamente como `resultados_recocido_simulado_{nombre_instancia}.csv`. |
| `nombre_instancia` | `str` | `"instancia"` | Nombre de la instancia para identificarla en el CSV y en la carga del contexto. |
| `id_corrida` | `str \| None` | `None` | Identificador de la corrida (util en experimentos con multiples corridas). Se escribe en el CSV. |
| `config_id` | `str \| None` | `None` | Identificador de la configuracion de hiperparametros. Se escribe en el CSV. |
| `repeticion` | `int \| None` | `None` | Numero de repeticion dentro de un experimento. Se escribe en el CSV. |
| `root` | `str \| None` | `None` | Directorio raiz alternativo para buscar archivos de la instancia. |
| `usar_penalizacion_capacidad` | `bool` | `True` | Si es `True`, el objetivo incluye una penalizacion por violacion de capacidad: `costo + lambda * violacion`. |
| `lambda_capacidad` | `float \| None` | `None` | Valor de lambda para la penalizacion de capacidad. Si es `None`, se calcula automaticamente como ~10 veces la mediana de la matriz de distancias. |
| `extra_csv` | `dict[str, object] \| None` | `None` | Columnas adicionales que se escribiran en el CSV. Util para registrar hiperparametros del experimento. |
| `patience` | `int` | `50` | Numero de niveles de temperatura consecutivos sin mejora del mejor global antes de activar el reheat (recalentamiento). Si `patience = 0` el reheat esta desactivado y se obtiene el SA clasico. Valores tipicos: 20–100. Debe ser `>= 0`. |
| `reheat_factor` | `float` | `0.5` | Fraccion de `T_init_eff` a la que se sube la temperatura cuando se activa el reheat. Ej.: con `T_init_eff = 1500` y `reheat_factor = 0.5`, `T` salta a 750. Debe estar en `(0, 1]`. Valores tipicos: 0.3–0.7. |

### Que retorna

La funcion retorna un objeto `RecocidoSimuladoResult` (descrito en la siguiente seccion).

### Funcion de conveniencia: `recocido_simulado_desde_instancia`

```python
def recocido_simulado_desde_instancia(
    nombre_instancia: str,
    *,
    root: str | None = None,
    # ... mismos parametros que recocido_simulado, sin inicial_obj, data, G
) -> RecocidoSimuladoResult:
```

Carga automaticamente `data`, `G` e `inicial_obj` desde el nombre de la instancia y llama a `recocido_simulado`. Es equivalente a:

```python
data = load_instances(nombre_instancia, root=root)
G = cargar_objeto_gexf(nombre_instancia, root=root)
inicial_obj = cargar_solucion_inicial(nombre_instancia, root=root)
resultado = recocido_simulado(inicial_obj, data, G, ...)
```

---

## `RecocidoSimuladoResult`

Dataclass inmutable (`frozen=True, slots=True`) que agrupa todos los resultados de una corrida. Sus atributos son de solo lectura tras la creacion.

### Tabla de campos

| Campo | Tipo | Descripcion |
|---|---|---|
| `mejor_solucion` | `list[list[str]]` | La mejor solucion CARP encontrada durante toda la busqueda. Lista de rutas; cada ruta es lista de etiquetas incluyendo el deposito. |
| `mejor_costo` | `float` | Costo de `mejor_solucion`. Si existe solucion factible, es el costo factible; si no, el mejor costo general. |
| `solucion_inicial_referencia` | `list[list[str]]` | Solucion inicial usada como punto de partida y referencia para calcular la mejora. |
| `costo_solucion_inicial` | `float` | Costo de `solucion_inicial_referencia` (sin penalizacion). |
| `mejora_absoluta` | `float` | Diferencia `costo_solucion_inicial - mejor_costo`. Positivo indica mejora. |
| `mejora_porcentaje_inicial_vs_final` | `float` | `mejora_absoluta / costo_solucion_inicial * 100`. |
| `tiempo_segundos` | `float` | Tiempo total de ejecucion de la corrida en segundos (medido con `time.perf_counter`). |
| `iteraciones_totales` | `int` | Total de evaluaciones individuales de soluciones vecinas realizadas. |
| `enfriamientos_ejecutados` | `int` | Numero de ciclos de enfriamiento completados (reducciones de `T`). |
| `aceptadas` | `int` | Total de vecinos aceptados (mejores + peores aceptados por Metropolis). |
| `mejoras` | `int` | Numero de veces que la mejor solucion global reportada mejoro durante la corrida. |
| `semilla` | `int \| None` | Semilla del generador aleatorio usada en esta corrida. |
| `backend_evaluacion` | `str` | Backend real de evaluacion usado: `"cpu"` o `"gpu"`. |
| `historial_mejor_costo` | `list[float]` | Mejor costo reportado al inicio de cada nivel de temperatura. Solo se llena si `guardar_historial=True`. |
| `historial_temperatura` | `list[float]` | Valor de `T` al inicio de cada nivel. Solo se llena si `guardar_historial=True`. |
| `ultimo_movimiento_aceptado` | `MovimientoVecindario \| None` | Ultimo movimiento de vecindario que fue aceptado. Incluye operador, rutas e indices. |
| `operadores_propuestos` | `dict[str, int]` | Numero de veces que cada operador propuso un vecino. Ordenado por frecuencia descendente. |
| `operadores_aceptados` | `dict[str, int]` | Numero de veces que cada operador fue aceptado. |
| `operadores_mejoraron` | `dict[str, int]` | Numero de veces que cada operador produjo una mejora del mejor global. |
| `operadores_trayectoria_mejor` | `dict[str, int]` | Snapshot de `operadores_aceptados` en el momento en que se encontro la mejor solucion. |
| `usar_penalizacion_capacidad` | `bool` | Si `True`, se uso penalizacion de capacidad en el objetivo. |
| `lambda_capacidad` | `float` | Valor efectivo de lambda usado para penalizar violaciones de capacidad. |
| `n_iniciales_evaluados` | `int` | Numero de soluciones candidatas iniciales evaluadas antes de empezar la busqueda. |
| `iniciales_infactibles_aceptadas` | `int` | Cuantas candidatas iniciales violaban la restriccion de capacidad. |
| `aceptaciones_solucion_infactible` | `int` | Numero de veces que se acepto un vecino que viola capacidad durante la busqueda. |
| `mejor_solucion_factible_final` | `bool` | `True` si la mejor solucion encontrada respeta todas las restricciones de capacidad. |
| `archivo_csv` | `str \| None` | Ruta absoluta del CSV guardado, o `None` si `guardar_csv=False`. |
| `n_reheats` | `int` | Numero de veces que se activo el recalentamiento durante la busqueda. `0` indica que el reheat estaba desactivado (`patience=0`) o que el algoritmo no se estanco lo suficiente para dispararlo. |

---

## Ejemplo completo de uso

### Opcion 1: usando `recocido_simulado_desde_instancia` (recomendada)

```python
from metacarp.recocido_simulado import recocido_simulado_desde_instancia

resultado = recocido_simulado_desde_instancia(
    "EGL-E1-A",
    # temperatura_inicial y temperatura_minima se calculan automaticamente:
    #   T_init = 5 * d_max / n
    #   T_end  = 20 * d_max / n^2
    # L = n^2 tambien se calcula automaticamente desde el numero de tareas.
    alpha=0.97,
    semilla=42,
    # Mecanismo de recalentamiento (reheat) — sube T cuando el SA se estanca.
    patience=50,        # niveles sin mejora antes de recalentar
    reheat_factor=0.5,  # T se reinicia a la mitad de T_init_eff
    usar_penalizacion_capacidad=True,
    guardar_historial=True,
    guardar_csv=True,
    nombre_instancia="EGL-E1-A",
)

print(f"Mejor costo encontrado : {resultado.mejor_costo:.2f}")
print(f"Costo inicial          : {resultado.costo_solucion_inicial:.2f}")
print(f"Mejora absoluta        : {resultado.mejora_absoluta:.2f}")
print(f"Mejora porcentual      : {resultado.mejora_porcentaje_inicial_vs_final:.2f} %")
print(f"Tiempo de ejecucion    : {resultado.tiempo_segundos:.2f} s")
print(f"Iteraciones totales    : {resultado.iteraciones_totales}")
print(f"Enfriamientos          : {resultado.enfriamientos_ejecutados}")
print(f"Vecinos aceptados      : {resultado.aceptadas}")
print(f"Reheats activados      : {resultado.n_reheats}")
print(f"Solucion factible      : {resultado.mejor_solucion_factible_final}")
print()
print("Operadores que mas mejoraron:")
for op, n in resultado.operadores_mejoraron.items():
    print(f"  {op}: {n}")
```

### Opcion 2: cargando recursos manualmente y llamando a `recocido_simulado`

```python
from metacarp.recocido_simulado import recocido_simulado
from metacarp.instances import load_instances
from metacarp.cargar_grafos import cargar_objeto_gexf
from metacarp.cargar_soluciones_iniciales import cargar_solucion_inicial
from metacarp.vecindarios import OPERADORES_POPULARES

nombre = "EGL-E1-A"
data = load_instances(nombre)
G = cargar_objeto_gexf(nombre)
inicial_obj = cargar_solucion_inicial(nombre)

resultado = recocido_simulado(
    inicial_obj,
    data,
    G,
    # Pasar None (o simplemente omitir) -> calculo adaptativo desde la instancia.
    # temperatura_inicial = 5 * d_max / n
    # temperatura_minima  = 20 * d_max / n^2
    # L (longitud de la cadena de Markov) = n^2, tambien adaptativo.
    temperatura_inicial=None,
    temperatura_minima=None,
    alpha=0.95,
    semilla=7,
    operadores=OPERADORES_POPULARES,
    usar_penalizacion_capacidad=True,
    lambda_capacidad=None,          # calculo automatico
    backend_vecindario="labels",
    guardar_historial=True,
    guardar_csv=False,
    # Mecanismo de recalentamiento (reheat). Para SA clasico sin reheat
    # usar patience=0; aqui usamos los valores por defecto.
    patience=50,
    reheat_factor=0.5,
    nombre_instancia=nombre,
)

# Acceder al historial de temperatura para graficar la convergencia
for t, c in zip(resultado.historial_temperatura, resultado.historial_mejor_costo):
    print(f"T={t:.4f}  mejor_costo={c:.2f}")
```

### Acceder a la mejor solucion

```python
# La mejor solucion es una lista de rutas con el deposito incluido
for i, ruta in enumerate(resultado.mejor_solucion, start=1):
    print(f"Ruta {i}: {' -> '.join(ruta)}")
```

---

## Guia de ajuste de parametros

### Tabla de efectos

| Parametro | Valor bajo | Valor alto | Efecto principal |
|---|---|---|---|
| `temperatura_inicial` | Poca exploracion inicial; puede quedar en minimo local cercano al punto de partida. | Mucha exploracion; acepta casi cualquier vecino al inicio. | Si es `None` se calcula adaptativamente: ver subseccion de calibracion automatica. |
| `temperatura_minima` | El algoritmo corre hasta que `T` es extremadamente pequena; mas tiempo. | El algoritmo para antes; menos refinamiento final. | Si es `None` se calcula adaptativamente como `20 · d_max / n²`. |
| `alpha` | Enfriamiento rapido (ej. 0.80). Menos tiempo, menor calidad. | Enfriamiento lento (ej. 0.99). Mas tiempo, mayor calidad. | El valor mas influyente en la calidad de la solucion. Valores tipicos: 0.90 a 0.99. |

### Calibracion automatica (adaptativa)

Cuando `temperatura_inicial` o `temperatura_minima` son `None`, el algoritmo
calcula sus valores desde la propia instancia. Sean:

- `n` = numero de arcos requeridos (tareas) de la instancia (`len(ctx.u_arr)`).
- `d_max` = distancia maxima en la matriz Dijkstra (`ctx.dist`).

Las formulas (adaptadas de Lourenço et al. al CARP) son:

```
L       = n^2                  # longitud de la cadena de Markov por nivel de T
T_init  = 5 · d_max / n        # si temperatura_inicial es None
T_end   = 20 · d_max / n^2     # si temperatura_minima es None
```

Estas formulas garantizan que la temperatura inicial sea suficientemente
alta para permitir mucha exploracion, y que la temperatura minima sea lo
bastante pequena para que la fase final actue como busqueda voraz. La
longitud de la cadena de Markov crece cuadraticamente con el numero de
tareas, lo que da al algoritmo mas iteraciones internas en instancias
grandes y mantiene corridas rapidas en instancias pequenas.

### Regla practica para estimar `temperatura_inicial`

Una heuristica ampliamente usada es que `temperatura_inicial` deberia permitir aceptar un empeoramiento "tipico" con probabilidad de aproximadamente 0.80 en la primera iteracion. Si se denota ese empeoramiento tipico como `delta_0`:

```
exp(-delta_0 / T_inicial) ≈ 0.80
=> T_inicial ≈ -delta_0 / ln(0.80)
=> T_inicial ≈ delta_0 * 4.48
```

Para estimar `delta_0` en la practica: ejecutar la funcion `costo_rapido` sobre unos 50 vecinos aleatorios de la solucion inicial, calcular la diferencia promedio con el costo inicial en los vecinos que empeoran, y multiplicar por 4 a 5. Un punto de partida conservador es usar un porcentaje del costo inicial:

```python
# Estimacion rapida: T_inicial ≈ 10% del costo inicial
costo_ini = resultado_previo.costo_solucion_inicial
temperatura_inicial_estimada = 0.10 * costo_ini
```

Esta estimacion garantiza que al inicio se acepten mejoras del 10 % con probabilidad `exp(-0.10 * costo / T) ≈ exp(-1) ≈ 0.37`. Para mayor exploracion inicial, usar un porcentaje mayor (20 %–50 %).

### Configuraciones de referencia

| Escenario | `temperatura_inicial` | `temperatura_minima` | `alpha` |
|---|---|---|---|
| Prueba rapida | adaptativa (`None`) | adaptativa (`None`) | `0.90` |
| Produccion estandar | adaptativa (`None`) | adaptativa (`None`) | `0.95` |
| Alta calidad (lento) | adaptativa (`None`) | adaptativa (`None`) | `0.99` |

Notas:
- La longitud de la cadena de Markov `L = n²` siempre se calcula
  automaticamente desde la instancia.
- Para forzar valores especificos de temperatura, pasarlos directamente
  en lugar de `None`.

---

## Por que el RS no usa GPU

### Evaluacion de una solucion a la vez

El RS es una metaheuristica de **trayectoria unica**: en cada iteracion mantiene exactamente una solucion activa (`sol_actual`) y genera exactamente un vecino. La decision de aceptar o rechazar ese vecino debe tomarse *antes* de generar el siguiente, porque el vecino aceptado se convierte en el nuevo punto de partida.

Esta dependencia secuencial hace imposible paralelizar el bucle interno de forma significativa:

```
vecino_1 → acepto? → si → vecino_2 parte desde vecino_1
                    → no → vecino_2 parte desde sol_actual
```

En contraste, las metaheuristicas basadas en **poblaciones** (Abejas, Cuckoo Search) mantienen un conjunto de soluciones independientes. Cada solucion de la poblacion puede evaluarse en paralelo, formando un *lote* (*batch*). La funcion `costo_lote_ids` del modulo `evaluador_costo` esta disenada exactamente para ese caso: evalua todas las soluciones del lote en pocas operaciones NumPy/CuPy, con un solo bloque de fancy indexing sobre la matriz de distancias.

### Por que la GPU no ayuda en SA

El cuello de botella del RS es la evaluacion de `costo_rapido` en cada iteracion. Esa funcion ya es muy rapida en CPU gracias a NumPy: `u_arr[ids]`, `v_arr[ids]` y `dist[origen_prev, us]` son operaciones de fancy indexing sobre arrays pequenos (el numero de tareas de una ruta es tipicamente decenas, no millones). El overhead de transferir datos entre CPU y GPU (memcpy al dispositivo + lanzamiento de kernel + copia del resultado de regreso) superaria ampliamente el tiempo de calculo en GPU para un array de esa talla.

Por esta razon, el parametro `usar_gpu=True` en `recocido_simulado` se pasa al contexto solo para **trazabilidad** (queda registrado en `backend_evaluacion_solicitado` del CSV), pero no produce ninguna aceleracion real. El modulo lo documenta explicitamente:

> SA evalua una solucion por iteracion, por lo que el flag `usar_gpu` se pasa al contexto solo para trazabilidad: el cuello de botella ya esta resuelto en CPU y mover datos a GPU no aporta speedup en este caso.

### Resumen

| Metaheuristica | Tipo | Evaluaciones paralelas | GPU util |
|---|---|---|---|
| Recocido Simulado | Trayectoria unica | No (dependencia secuencial) | No |
| Busqueda Tabu | Trayectoria unica | No | No |
| Abejas (Bee Colony) | Poblacional | Si (lote de abejas) | Si |
| Cuckoo Search | Poblacional | Si (lote de cucos) | Si |

---

## Operadores de vecindario disponibles

La constante `OPERADORES_POPULARES` del modulo `vecindarios` define el conjunto de operadores habilitados por defecto:

```python
OPERADORES_POPULARES = (
    "relocate_intra",   # Mueve una tarea a otra posicion dentro de la misma ruta
    "swap_intra",       # Intercambia dos tareas dentro de la misma ruta
    "2opt_intra",       # Invierte un segmento de tareas dentro de la misma ruta
    "relocate_inter",   # Mueve una tarea de una ruta a otra
    "swap_inter",       # Intercambia una tarea entre dos rutas distintas
    "2opt_star",        # Intercambia las colas de dos rutas a partir de un punto de corte
    "cross_exchange",   # Intercambia segmentos completos entre dos rutas
)
```

En cada iteracion, `generar_vecino` elige uno de estos operadores al azar y lo aplica a la solucion actual, produciendo un vecino y un objeto `MovimientoVecindario` que describe exactamente que cambio se realizo (operador, rutas involucradas, indices, etiquetas de tareas movidas).

Se puede restringir el conjunto de operadores pasando una lista personalizada al parametro `operadores`:

```python
from metacarp.recocido_simulado import recocido_simulado_desde_instancia

resultado = recocido_simulado_desde_instancia(
    "EGL-E1-A",
    operadores=["relocate_inter", "swap_inter"],  # solo operadores inter-ruta
    semilla=1,
)
```
