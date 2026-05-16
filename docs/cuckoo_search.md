# Cuckoo Search para CARP

Documentación técnica del módulo `metacarp.cuckoo_search`.

---

## Introducción conceptual

### La analogía biológica

El **Cuckoo Search** es una metaheurística propuesta por Yang y Deb (2009) inspirada en el parasitismo de nidificación del cucú (*cuckoo*): esta ave pone sus huevos en los nidos de otras especies. Si el ave huésped detecta el huevo intruso lo abandona y construye un nido nuevo. Si no lo detecta, el polluelo de cucú eclosa y compite —a menudo exitosamente— con las crías del huésped.

Trasladado a optimización combinatoria:

- Los **nidos** son soluciones candidatas al problema.
- Los **huevos de cucú** son nuevas soluciones generadas mediante *vuelo de Lévy*.
- La **competencia**: si la nueva solución es mejor que la del nido que elige al azar, la reemplaza (el huésped "acepta" el huevo).
- El **abandono**: una fracción de los peores nidos es descartada y reconstruida desde el mejor nido conocido (el huésped "detecta" el intruso y abandona el nido).

### Por qué el vuelo de Lévy es útil en optimización

Un paseo aleatorio ordinario produce pasos de longitud aproximadamente uniforme, lo que conduce a una exploración lenta y localizada. La distribución de Lévy tiene **colas pesadas** (*heavy-tail*): la mayoría de los pasos son cortos, pero ocasionalmente ocurren saltos muy largos. Esta propiedad permite al algoritmo escapar de mínimos locales y explorar zonas alejadas del espacio de búsqueda con una frecuencia mucho mayor que un paseo aleatorio clásico.

En el contexto de CARP (Capacitated Arc Routing Problem), el espacio de búsqueda es discreto y combinatorio. El vuelo de Lévy se adapta controlando cuántas perturbaciones locales consecutivas se aplican a una solución, usando el número de perturbaciones como proxy del "tamaño del salto".

---

## Vuelo de Lévy

### Distribuciones de cola pesada

Una distribución de cola pesada asigna probabilidad no despreciable a valores muy grandes. Mientras que en una distribución normal la probabilidad de observar un valor 10 veces mayor que la desviación estándar es prácticamente nula, en una distribución de Lévy esa probabilidad es significativa. Esto es lo que produce los saltos ocasionalmente largos.

### La fórmula usada en la implementación

En espacio continuo, el paso de Lévy sigue una distribución exacta. La implementación discreta usada en este proyecto la aproxima con la siguiente fórmula (ver `_vuelo_levy_discreto` en `cuckoo_search.py`):

```
n_pasos = min(12, 1 + floor( |N(0,1)|^(1/beta) * pasos_base ))
```

Donde:

- `N(0,1)` es una muestra de la distribución normal estándar.
- `|·|` toma el valor absoluto.
- `beta` controla la forma de la distribución (valor clásico: `1.5`).
- `pasos_base` escala la magnitud de los saltos.
- El tope de **12 pasos** evita que saltos extremadamente grandes saturen el tiempo de cómputo sin beneficio proporcional.

### El rol del parámetro `beta`

| Valor de `beta` | Efecto |
|---|---|
| `beta` pequeño (p.ej. 0.8) | Exponente `1/beta` grande → pasos más largos más frecuentes → mayor exploración global |
| `beta = 1.5` (clásico) | Balance estándar entre exploración y explotación |
| `beta` grande (p.ej. 3.0) | Exponente `1/beta` pequeño → tiende a pasos cortos → mayor explotación local |

### Adaptación a espacio discreto

Cada "paso" del vuelo de Lévy es una perturbación local: se aplica un operador de vecindario (`generar_vecino`) a la solución actual, produciendo una solución vecina. El número de perturbaciones consecutivas `n_pasos` sigue la distribución discreta descrita arriba. El resultado es una solución que ha sido modificada `n_pasos` veces desde el punto de partida.

---

## Cómo funciona paso a paso

### Diagrama de flujo

```
  Inicialización
  ┌─────────────────────────────────────────────────────────┐
  │  Cargar solución de referencia (mejor inicial)          │
  │  Crear num_nidos nidos: 1 desde referencia, resto       │
  │  generados como vecinos aleatorios de la referencia     │
  └────────────────────────┬────────────────────────────────┘
                           │
                           ▼
  ┌──────────────────── ITERACIÓN ──────────────────────────┐
  │                                                         │
  │  PASO 1: GENERACIÓN DE CUCKOOS (Vuelo de Lévy)          │
  │  ┌───────────────────────────────────────────────────┐  │
  │  │  Para cada nido i (i = 0..num_nidos-1):           │  │
  │  │    Calcular n_pasos con fórmula Lévy discreta     │  │
  │  │    Aplicar n_pasos perturbaciones consecutivas    │  │
  │  │    → cuckoo_i (nueva solución candidata)          │  │
  │  │  Evaluar todos los cuckoos en lote                │  │
  │  └───────────────────────────────────────────────────┘  │
  │                        │                                 │
  │                        ▼                                 │
  │  PASO 2: COMPETENCIA CON NIDO ALEATORIO                 │
  │  ┌───────────────────────────────────────────────────┐  │
  │  │  Para cada cuckoo_i:                              │  │
  │  │    Elegir nido j al azar (j puede ser == i)       │  │
  │  │    Si obj(cuckoo_i) < obj(nido_j):                │  │
  │  │      nido_j ← cuckoo_i   (reemplazo exitoso)     │  │
  │  └───────────────────────────────────────────────────┘  │
  │                        │                                 │
  │                        ▼                                 │
  │  PASO 3: ABANDONO DE PEORES NIDOS                       │
  │  ┌───────────────────────────────────────────────────┐  │
  │  │  n_abandonar = floor(pa_abandono × num_nidos)     │  │
  │  │  Identificar los n_abandonar peores nidos         │  │
  │  │  Para cada peor nido:                             │  │
  │  │    Aplicar vuelo Lévy desde el mejor nido actual  │  │
  │  │    → nuevo nido (reemplazo incondicional)         │  │
  │  └───────────────────────────────────────────────────┘  │
  │                        │                                 │
  │                        ▼                                 │
  │  Actualizar mejor global y mejor factible               │
  │  Registrar historial de costo                           │
  │                        │                                 │
  └────────────────────────┘
                           │ (fin de iteraciones)
                           ▼
  Retornar CuckooSearchResult
```

---

## Componentes clave

### Los nidos (`num_nidos`)

Los nidos representan la **población activa** de soluciones candidatas. A diferencia de las metaheurísticas de trayectoria única (como Búsqueda Tabú o Recocido Simulado), Cuckoo Search mantiene `num_nidos` soluciones en paralelo durante toda la ejecución.

**Inicialización:**

1. El primer nido es la mejor solución inicial disponible (seleccionada entre las candidatas del archivo pickle mediante `seleccionar_mejor_inicial_rapido`).
2. Los nidos restantes se generan aplicando una única perturbación aleatoria (`generar_vecino`) a la solución de referencia.

Cada nido almacena tres valores: la solución (`nidos_sol`), su costo puro (`nidos_pure`) y su violación de capacidad (`nidos_viol`). El objetivo penalizado de un nido se computa como `costo_puro + λ × violación`.

### El vuelo de Lévy discreto

La función `_vuelo_levy_discreto` adapta el vuelo continuo al espacio combinatorio de CARP:

1. Se muestrea `x = |N(0,1)|` (valor absoluto de una normal estándar).
2. Se calcula `n_pasos = min(12, 1 + int(x^(1/beta) * max(1, pasos_base)))`.
3. Se aplican `n_pasos` operadores de vecindario consecutivos a la solución base, comenzando desde una copia de la misma.
4. El resultado es una solución que ha sido perturbada `n_pasos` veces.

El mínimo garantizado es **1 paso** (siempre se produce al menos una perturbación). El máximo es **12 pasos** (cota dura en el código).

Los operadores de vecindario disponibles son los definidos en `OPERADORES_POPULARES`: `relocate_intra`, `swap_intra`, `2opt_intra`, `relocate_inter`, `swap_inter`, `2opt_star` y `cross_exchange`.

### La competencia de nidos

En el Paso 2, cada cuckoo `i` compite con un nido `j` elegido **uniformemente al azar** entre los `num_nidos` nidos (incluyendo posiblemente el propio nido `i`). El reemplazo ocurre si:

```
obj(cuckoo_i) < obj(nido_j) - 1e-15
```

Es decir, la nueva solución debe ser estrictamente mejor (con tolerancia numérica). No hay probabilidad de aceptación de soluciones peores: el criterio es puramente de mejora. Cuando el reemplazo ocurre, los tres valores del nido `j` se actualizan: solución, costo puro y violación.

### El abandono (`pa_abandono`)

En el Paso 3, una fracción `pa_abandono` de los **peores nidos** (ordenados por objetivo penalizado de mayor a menor) es descartada y reconstruida. El número exacto de nidos a abandonar es:

```
n_abandonar = max(1, floor(pa_abandono × num_nidos))
```

El mínimo garantizado es siempre 1, incluso si `pa_abandono × num_nidos < 1`.

Los nuevos nidos se generan aplicando un **vuelo de Lévy desde el mejor nido actual** (el nido con menor objetivo penalizado en ese instante). Esto garantiza diversidad alrededor de la región prometedora, en lugar de reiniciar aleatoriamente desde cero. El reemplazo de los peores nidos es incondicional: no hay comparación de calidad.

---

## La función `cuckoo_search`

### Firma

```python
def cuckoo_search(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    iteraciones: int = 260,
    num_nidos: int = 20,
    pa_abandono: float = 0.25,
    pasos_levy_base: int = 3,
    beta_levy: float = 1.5,
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
) -> CuckooSearchResult:
```

### Tabla de parámetros

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `inicial_obj` | `Any` | — | Objeto con soluciones iniciales candidatas (dict, pickle, lista de rutas o estructura anidada). Se explora recursivamente para extraer soluciones. |
| `data` | `Mapping[str, Any]` | — | Datos de la instancia CARP cargados con `load_instances`. |
| `G` | `nx.Graph` | — | Grafo de la instancia cargado con `cargar_objeto_gexf`. |
| `iteraciones` | `int` | `260` | Número de ciclos del bucle principal. Cada ciclo ejecuta los tres pasos (generación, competencia, abandono). |
| `num_nidos` | `int` | `20` | Número de soluciones candidatas mantenidas en paralelo. Debe ser >= 2. |
| `pa_abandono` | `float` | `0.25` | Fracción de peores nidos a abandonar por iteración. Debe estar en el intervalo abierto (0, 1). |
| `pasos_levy_base` | `int` | `3` | Escala base del número de perturbaciones en el vuelo de Lévy. Valores más altos producen saltos más largos en promedio. |
| `beta_levy` | `float` | `1.5` | Parámetro de forma de la distribución de Lévy discreta. Valor clásico de la literatura. Si se pasa un valor <= 0, se usa 1.5 automáticamente. |
| `semilla` | `int \| None` | `None` | Semilla del generador aleatorio. Si `None`, la ejecución no es reproducible. |
| `operadores` | `Iterable[str]` | `OPERADORES_POPULARES` | Nombres de los operadores de vecindario habilitados. Por defecto, los 9 operadores definidos en `vecindarios.py`. |
| `marcador_depot_etiqueta` | `str \| None` | `None` | Etiqueta del nodo depósito en las rutas (p.ej. `"D"`). Si `None`, se toma del contexto de evaluación. |
| `usar_gpu` | `bool` | `False` | Si `True` y CuPy está disponible, la evaluación en lote se realiza en GPU. |
| `backend_vecindario` | `Literal["labels", "ids"]` | `"labels"` | Modo de generación de vecinos. `"ids"` opera sobre representación entera (más rápido para instancias grandes). |
| `guardar_historial` | `bool` | `True` | Si `True`, registra el mejor costo al inicio de cada iteración en `historial_mejor_costo`. |
| `guardar_csv` | `bool` | `False` | Si `True`, escribe una fila de resultados en un archivo CSV al finalizar. |
| `ruta_csv` | `str \| None` | `None` | Ruta del CSV de resultados. Si `None`, se genera automáticamente como `resultados_cuckoo_search_{nombre_instancia}.csv`. |
| `nombre_instancia` | `str` | `"instancia"` | Identificador de la instancia usado en el CSV y en la carga del contexto desde caché. |
| `id_corrida` | `str \| None` | `None` | Identificador de la corrida para el CSV (útil en experimentos con múltiples repeticiones). |
| `config_id` | `str \| None` | `None` | Identificador de configuración de hiperparámetros para el CSV. |
| `repeticion` | `int \| None` | `None` | Número de repetición dentro de un experimento para el CSV. |
| `root` | `str \| None` | `None` | Directorio raíz alternativo para buscar archivos de instancia. |
| `usar_penalizacion_capacidad` | `bool` | `True` | Si `True`, el objetivo penaliza las violaciones de capacidad con `costo + λ × exceso`. |
| `lambda_capacidad` | `float \| None` | `None` | Factor λ de penalización. Si `None`, se calcula automáticamente como ~10 veces la mediana de distancias en la instancia. |
| `extra_csv` | `dict[str, object] \| None` | `None` | Columnas adicionales a incluir en la fila CSV. |

### Qué retorna

La función retorna un objeto `CuckooSearchResult` (ver sección siguiente).

---

## `CuckooSearchResult`

Dataclass inmutable (`frozen=True, slots=True`) que agrupa todos los resultados de una ejecución de Cuckoo Search.

### Tabla de campos

| Campo | Tipo | Descripción |
|---|---|---|
| `mejor_solucion` | `list[list[str]]` | La mejor solución CARP encontrada. Lista de rutas; cada ruta es lista de etiquetas de tareas y marcador de depósito. |
| `mejor_costo` | `float` | Costo de la mejor solución. Si existe solución factible, es el costo factible; si no, el mejor costo general. |
| `solucion_inicial_referencia` | `list[list[str]]` | La solución inicial usada como referencia (mejor candidata inicial seleccionada). |
| `costo_solucion_inicial` | `float` | Costo puro de la solución inicial de referencia. |
| `mejora_absoluta` | `float` | Diferencia `costo_inicial - mejor_costo`. Positivo indica mejora. |
| `mejora_porcentaje_inicial_vs_final` | `float` | `mejora_absoluta / costo_inicial × 100`. Porcentaje de reducción del costo. |
| `tiempo_segundos` | `float` | Tiempo total de ejecución medido con `time.perf_counter`. |
| `iteraciones_totales` | `int` | Número de iteraciones ejecutadas (igual al parámetro `iteraciones`). |
| `nidos` | `int` | Número de nidos usados (igual al parámetro `num_nidos`). |
| `abandonos_totales` | `int` | Total de nidos descartados y reconstruidos durante toda la búsqueda (Paso 3 acumulado). |
| `reemplazos_exitosos` | `int` | Veces que un cuckoo reemplazó exitosamente un nido (Paso 2 acumulado). |
| `mejoras` | `int` | Veces que el mejor global mejoró durante la búsqueda. |
| `semilla` | `int \| None` | Semilla usada (o `None` si no se especificó). |
| `backend_evaluacion` | `str` | Backend real de evaluación usado: `"cpu"` o `"gpu"`. |
| `historial_mejor_costo` | `list[float]` | Mejor costo al inicio de cada iteración (vacío si `guardar_historial=False`). |
| `ultimo_movimiento_aceptado` | `MovimientoVecindario \| None` | El último movimiento que fue aceptado (reemplazo de nido exitoso o abandono). |
| `operadores_propuestos` | `dict[str, int]` | Conteo de cuántas veces cada operador generó un vecino (en vuelo de Lévy). |
| `operadores_aceptados` | `dict[str, int]` | Conteo de cuántas veces cada operador fue aceptado (el vecino reemplazó un nido). |
| `operadores_mejoraron` | `dict[str, int]` | Conteo de cuántas veces cada operador contribuyó a mejorar el mejor global. |
| `operadores_trayectoria_mejor` | `dict[str, int]` | Snapshot de `operadores_aceptados` en el momento en que se descubrió la mejor solución. |
| `usar_penalizacion_capacidad` | `bool` | Si se usó penalización de capacidad durante la búsqueda. |
| `lambda_capacidad` | `float` | Valor efectivo de λ usado para la penalización. |
| `n_iniciales_evaluados` | `int` | Número de candidatas iniciales evaluadas para elegir la de referencia. |
| `iniciales_infactibles_aceptadas` | `int` | Candidatas iniciales que violaban restricciones de capacidad. |
| `aceptaciones_solucion_infactible` | `int` | Veces que se aceptó en un nido una solución que viola restricciones de capacidad. |
| `mejor_solucion_factible_final` | `bool` | `True` si la mejor solución final respeta todas las restricciones de capacidad. |
| `archivo_csv` | `str \| None` | Ruta absoluta del CSV guardado, o `None` si no se guardó. |

---

## Ejemplo completo de uso

### Uso con recursos precargados

```python
from metacarp.cuckoo_search import cuckoo_search
from metacarp.instances import load_instances
from metacarp.cargar_grafos import cargar_objeto_gexf
from metacarp.cargar_soluciones_iniciales import cargar_solucion_inicial

# Cargar los recursos de la instancia
nombre = "EGL-E1-A"
data = load_instances(nombre)
G = cargar_objeto_gexf(nombre)
inicial_obj = cargar_solucion_inicial(nombre)

# Ejecutar Cuckoo Search con configuración explícita
resultado = cuckoo_search(
    inicial_obj,
    data,
    G,
    iteraciones=260,
    num_nidos=20,
    pa_abandono=0.25,
    pasos_levy_base=3,
    beta_levy=1.5,
    semilla=42,
    guardar_historial=True,
    nombre_instancia=nombre,
)

# Consultar resultados
print(f"Costo inicial:   {resultado.costo_solucion_inicial:.2f}")
print(f"Mejor costo:     {resultado.mejor_costo:.2f}")
print(f"Mejora:          {resultado.mejora_porcentaje_inicial_vs_final:.2f}%")
print(f"Tiempo:          {resultado.tiempo_segundos:.2f}s")
print(f"Reemplazos:      {resultado.reemplazos_exitosos}")
print(f"Abandonos:       {resultado.abandonos_totales}")
print(f"Factible final:  {resultado.mejor_solucion_factible_final}")

# Acceder a la mejor solución
for i, ruta in enumerate(resultado.mejor_solucion, start=1):
    print(f"  Ruta {i}: {' -> '.join(ruta)}")
```

### Uso con la función de conveniencia

```python
from metacarp.cuckoo_search import cuckoo_search_desde_instancia

# Equivalente al ejemplo anterior: carga todo automáticamente
resultado = cuckoo_search_desde_instancia(
    "EGL-E1-A",
    iteraciones=260,
    num_nidos=20,
    pa_abandono=0.25,
    pasos_levy_base=3,
    beta_levy=1.5,
    semilla=42,
    guardar_csv=True,
    ruta_csv="resultados/cuckoo_egl_e1_a.csv",
)
```

### Guardar resultados en CSV con metadatos de experimento

```python
from metacarp.cuckoo_search import cuckoo_search_desde_instancia

resultado = cuckoo_search_desde_instancia(
    "EGL-E1-A",
    iteraciones=500,
    num_nidos=30,
    pa_abandono=0.20,
    semilla=7,
    guardar_csv=True,
    id_corrida="exp_grid_001",
    config_id="nidos30_pa020",
    repeticion=1,
    extra_csv={"experimento": "grid_search_fase2"},
)
```

---

## Guía de ajuste de parámetros

| Parámetro | Default | Efecto al aumentar | Efecto al disminuir | Recomendación |
|---|---|---|---|---|
| `num_nidos` | `20` | Mayor diversidad, mejor cobertura del espacio, mayor costo por iteración | Menor diversidad, convergencia más rápida pero riesgo de mínimos locales | Aumentar para instancias grandes o con muchos mínimos locales; reducir si el tiempo de cómputo es crítico |
| `pa_abandono` | `0.25` | Más diversificación: más nidos se reconstruyen cada iteración | Menos diversificación: la población converge más rápido | Valores entre 0.15 y 0.35 son típicos; aumentar si el algoritmo se estanca |
| `pasos_levy_base` | `3` | Saltos más largos en promedio → mayor exploración global | Saltos más cortos → explotación más local | Aumentar para escapar de mínimos locales profundos; reducir para explotar buenas soluciones |
| `beta_levy` | `1.5` | (disminuir) Más saltos largos, mayor exploración | (aumentar) Saltos más cortos, más explotación local | Mantener en 1.5 (valor clásico) salvo necesidad demostrada de ajuste |
| `num_iteraciones` | `260` | Más tiempo de búsqueda, potencialmente mejor solución | Menos tiempo, soluciones de menor calidad | Escalar con el tamaño de la instancia; monitorear `historial_mejor_costo` para detectar convergencia prematura |

**Notas adicionales:**

- El número total de evaluaciones de soluciones por iteración es aproximadamente `num_nidos × (1 + pa_abandono) × pasos_levy_base`, más la varianza introducida por la distribución de Lévy.
- Si `mejor_solucion_factible_final` es `False`, considerar aumentar `lambda_capacidad` para que la penalización desincentive más las violaciones.
- El parámetro `usar_penalizacion_capacidad=True` es el modo recomendado: permite que el algoritmo explore soluciones temporalmente infactibles sin quedar atrapado en ellas.

---

## Diferencia con las otras metaheurísticas poblacionales del proyecto

El proyecto MetaCARP implementa dos metaheurísticas que mantienen una **población de soluciones** en paralelo: Cuckoo Search (`cuckoo_search.py`) y Artificial Bee Colony (`abejas.py`). Ambas comparten la infraestructura de evaluación vectorizada (`evaluador_costo.py`) y los mismos operadores de vecindario (`vecindarios.py`), pero difieren en sus mecanismos centrales.

### Mecanismo de exploración

**Cuckoo Search** genera nuevas soluciones exclusivamente mediante **vuelo de Lévy discreto**: cada nido produce un cuckoo aplicando entre 1 y 12 perturbaciones consecutivas, con el número de perturbaciones determinado por la distribución de Lévy. El tamaño del salto varía significativamente entre iteraciones y entre nidos. El reemplazo de un nido es **directo por mejora**: si el cuckoo es mejor que el nido aleatorio elegido, lo reemplaza sin más condición.

**Artificial Bee Colony** distingue tres roles con comportamientos diferentes:

- Las *abejas empleadas* generan exactamente **un vecino** por fuente (un solo operador aleatorio, sin acumulación de pasos).
- Las *abejas observadoras* eligen fuentes con **probabilidad proporcional a su calidad** (las mejores fuentes atraen más abejas) y también generan un vecino por fuente seleccionada.
- Las *abejas scout* reemplazan fuentes que llevan más de `limite_abandono` intentos sin mejorar.

### Diferencias clave

| Aspecto | Cuckoo Search | Artificial Bee Colony |
|---|---|---|
| Fuente del movimiento | Vuelo de Lévy (1–12 pasos) | Un solo operador de vecindario |
| Selección de fuente competidora | Nido aleatorio uniforme | Proporcional a calidad (sesgada hacia las mejores) |
| Criterio de reemplazo | Mejora directa (sin umbral) | Mejora directa (sin umbral) |
| Mecanismo de abandono | Los `pa_abandono × num_nidos` peores, reconstruidos desde el mejor | Fuentes que exceden `limite_abandono` intentos sin mejorar, reconstruidas desde el mejor |
| Tipo de exploración global | Saltos de longitud variable (Lévy) | Reinicio de fuentes agotadas con un salto unitario |
| Parámetro de diversificación | `pa_abandono` (fracción fija por iteración) | `limite_abandono` (paciencia antes del abandono) |

La principal ventaja del Cuckoo Search es el vuelo de Lévy: la variabilidad en el tamaño del salto produce una exploración más agresiva e irregular del espacio de búsqueda, lo que puede ser beneficioso cuando el paisaje de costo tiene muchos mínimos locales de profundidad similar. ABC, en cambio, concentra progresivamente más esfuerzo en las regiones prometedoras gracias a la selección sesgada de las observadoras, lo que puede ser más eficiente cuando la mejor región ya ha sido identificada.
