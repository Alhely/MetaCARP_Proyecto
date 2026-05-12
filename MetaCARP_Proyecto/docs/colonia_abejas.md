# Colonia de Abejas Artificiales (ABC) para CARP

## Introducción conceptual

El algoritmo **Artificial Bee Colony (ABC)** es una metaheurística poblacional propuesta por Karaboga (2005), inspirada en el comportamiento colectivo de búsqueda de alimento de las abejas melíferas. A diferencia de los algoritmos de trayectoria única (como Búsqueda Tabú o Recocido Simulado), ABC mantiene en paralelo un conjunto de soluciones candidatas activas, lo que le otorga una capacidad natural de diversificación.

### Analogía biológica completa

En una colmena real, las abejas especializadas realizan roles distintos para localizar y explotar fuentes de alimento eficientemente. El algoritmo ABC traslada esa dinámica al dominio de la optimización combinatoria:

```
COLMENA REAL                         ALGORITMO ABC (CARP)
============================================================
Fuente de alimento             <-->  Solución candidata (lista de rutas)
Calidad del néctar             <-->  Inverso del costo total (1 / (1 + obj))
Abeja empleada                 <-->  Agente asignado a una fuente, la mejora localmente
Danza del meneo (waggle dance) <-->  Señal de calidad que influye en las observadoras
Abeja observadora              <-->  Agente que elige fuentes según su calidad
Flor agotada                   <-->  Fuente sin mejora tras `limite_abandono` intentos
Abeja exploradora (scout)      <-->  Agente que reemplaza fuentes agotadas
```

La clave del mecanismo biológico es la **danza del meneo**: una abeja que ha encontrado una buena fuente de alimento ejecuta una danza cuya duración e intensidad comunica a las observadoras tanto la dirección como la calidad de esa fuente. Las observadoras eligen adónde ir con una probabilidad proporcional a la calidad percibida; las mejores fuentes atraen más abejas, lo que concentra el esfuerzo de búsqueda donde más rinde.

En la implementación de MetaCARP, la "calidad" de una fuente se cuantifica con la función de fitness:

```
fitness(i) = 1 / (1 + objetivo_penalizado(i))
```

Cuanto menor el objetivo penalizado (costo + penalización de capacidad), mayor el fitness y mayor la probabilidad de ser seleccionada por las observadoras.

### Los tres tipos de abejas y sus roles

| Tipo | Cantidad | Asignación | Rol en ABC |
|------|----------|------------|------------|
| Empleadas | `num_fuentes` | Una por fuente | Generan un vecino de su fuente; actualizan si mejora |
| Observadoras | `num_fuentes` | Elegidas por ruleta | Generan vecinos en fuentes seleccionadas por probabilidad |
| Scouts (exploradoras) | Variable | Fuentes agotadas | Reemplazan fuentes sin mejora con nuevas soluciones |

---

## Cómo funciona paso a paso

### Diagrama de flujo del ciclo ABC

```
┌─────────────────────────────────────────────────────────┐
│                  INICIALIZACIÓN                         │
│  • Cargar solución inicial de referencia                │
│  • Construir contexto de evaluación rápida (NumPy)      │
│  • Poblar `num_fuentes` fuentes con vecinos aleatorios  │
│  • Inicializar `trials[i] = 0` para cada fuente         │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  iteración = 1..N   │◄─────────────────────┐
              └──────────┬──────────┘                      │
                         │                                  │
        ╔════════════════▼════════════════╗                 │
        ║     FASE 1: ABEJAS EMPLEADAS    ║                 │
        ║  Para cada fuente i (en lote):  ║                 │
        ║  • generar_vecino(fuente_i)     ║                 │
        ║  • evaluar vecino (penalizado)  ║                 │
        ║  • si vecino < fuente_i:        ║                 │
        ║      fuente_i = vecino          ║                 │
        ║      trials[i] = 0              ║                 │
        ║  • si no:                       ║                 │
        ║      trials[i] += 1             ║                 │
        ╚════════════════╤════════════════╝                 │
                         │                                  │
        ╔════════════════▼════════════════╗                 │
        ║   FASE 2: ABEJAS OBSERVADORAS   ║                 │
        ║  • calcular fitness de cada     ║                 │
        ║    fuente: 1/(1+obj_i)          ║                 │
        ║  • normalizar a probabilidades  ║                 │
        ║  • seleccionar `num_fuentes`    ║                 │
        ║    fuentes por ruleta           ║                 │
        ║  • generar vecinos (en lote)    ║                 │
        ║  • comparar con fuente origen   ║                 │
        ║  • actualizar si mejora         ║                 │
        ╚════════════════╤════════════════╝                 │
                         │                                  │
        ╔════════════════▼════════════════╗                 │
        ║    FASE 3: ABEJAS SCOUT         ║                 │
        ║  Para cada fuente i:            ║                 │
        ║  • si trials[i] >= limite:      ║                 │
        ║      nueva = vecino(mejor_global)║                │
        ║      fuente_i = nueva           ║                 │
        ║      trials[i] = 0              ║                 │
        ║      scouts += 1                ║                 │
        ╚════════════════╤════════════════╝                 │
                         │                                  │
              ┌──────────▼──────────┐                      │
              │  actualizar mejor   │                      │
              │  global (factible   │                      │
              │  y no factible)     │                      │
              └──────────┬──────────┘                      │
                         │                                  │
              ┌──────────▼──────────┐     ¿más       ┌─────┘
              │  guardar historial  ├──iteraciones?───┘
              └──────────┬──────────┘
                         │ no
              ┌──────────▼──────────┐
              │  retornar AbejasResult│
              └────────────────────-─┘
```

### Fase 1: Abejas empleadas

Cada abeja empleada tiene asignada una fuente de alimento (una solución CARP). En cada iteración genera exactamente un vecino de esa fuente usando un operador de vecindario elegido al azar. Si el vecino tiene menor objetivo penalizado que la fuente actual, la reemplaza y resetea el contador de intentos fallidos `trials[i] = 0`. Si no mejora, incrementa `trials[i] += 1`. Todas las evaluaciones de esta fase se realizan en lote con `costo_lote_penalizado_ids`.

### Fase 2: Abejas observadoras

Las observadoras no están asignadas a fuentes fijas: eligen adónde ir en cada iteración con una probabilidad proporcional a la calidad de las fuentes. La selección se realiza por **ruleta ponderada por fitness**:

```
fitness(i) = 1 / (1 + max(obj_i, 0))

P(elegir fuente i) = fitness(i) / sum_j(fitness(j))
```

Se seleccionan `num_fuentes` fuentes con reemplazo (la misma fuente puede ser elegida más de una vez, intensificando la búsqueda en torno a las mejores). Para cada fuente seleccionada se genera un vecino y se compara contra la fuente original. Si mejora, se actualiza. Si no, se incrementa `trials` de esa fuente.

Si la suma de todos los fitness es cero o negativa (caso excepcional), se usan probabilidades uniformes como mecanismo de seguridad.

### Fase 3: Abejas scout (exploradoras)

Al final de cada iteración, el algoritmo identifica las fuentes cuyo contador de intentos fallidos superó el `limite_abandono`. Esas fuentes se consideran "agotadas" y son abandonadas. Para reemplazarlas, se generan vecinos de la mejor fuente actual (la de menor objetivo penalizado). La fuente agotada se sustituye por este nuevo punto de búsqueda y su contador se resetea a cero. Esto garantiza diversificación continua: ninguna fuente puede monopolizar el esfuerzo de búsqueda indefinidamente.

---

## Componentes clave

### Las fuentes de alimento (`num_fuentes`)

Las fuentes de alimento son el conjunto de soluciones candidatas que ABC mantiene activo en paralelo durante toda la ejecución. Cada fuente es una solución CARP completa: una lista de rutas donde cada ruta es una lista de etiquetas de tareas (`["D", "TR3", "TR7", "D"]`).

**Inicialización:** La primera fuente siempre es la mejor solución inicial cargada desde el archivo pickle de la instancia (seleccionada por `seleccionar_mejor_inicial_rapido` entre todas las candidatas disponibles). Las fuentes restantes se generan aplicando operadores de vecindario aleatorios a esa solución de referencia:

```python
# Pseudocódigo de la inicialización
fuentes_sol[0] = mejor_solucion_inicial
for i in 1..num_fuentes-1:
    fuentes_sol[i] = generar_vecino(fuentes_sol[0], ...)
```

Cada fuente mantiene tres valores paralelos: `fuentes_sol[i]` (la solución), `fuentes_pure[i]` (su costo sin penalización) y `fuentes_viol[i]` (su violación de capacidad). El objetivo penalizado se calcula bajo demanda con `objectivo_penalizado`.

### Las abejas empleadas

Hay exactamente una abeja empleada por fuente de alimento. La relación es uno a uno: la abeja empleada i trabaja exclusivamente sobre la fuente i. Su comportamiento es **greedy y miope**: acepta cualquier vecino que mejore el objetivo penalizado, sin memoria de movimientos anteriores. Es la fase de **explotación** del algoritmo.

### Las abejas observadoras: selección por ruleta

La selección por ruleta proporcional al fitness es el mecanismo central que distingue al ABC de una simple búsqueda de vecindario con múltiples reinicios. La fórmula exacta implementada es:

```
fitness_i = 1 / (1 + max(obj_i, 0.0))

P_i = fitness_i / sum_{j=0}^{num_fuentes-1} fitness_j
```

donde `obj_i` es el objetivo penalizado de la fuente i: `costo_puro_i + lambda * violacion_i`.

La selección se realiza con `rng.choices(rango_fuentes, weights=probs, k=num_fuentes)`, que permite repeticiones. Una fuente con objetivo bajo (solución de alta calidad) tiene un fitness alto y una probabilidad alta de ser elegida múltiples veces, concentrando el esfuerzo de búsqueda en las zonas prometedoras del espacio de soluciones.

### Los scouts: abandono y reinicio

El mecanismo scout controla la **diversificación**. Cada fuente lleva un contador `trials[i]` que se incrementa cada vez que un vecino generado para esa fuente (ya sea por una empleada o por una observadora) no mejora el objetivo. Cuando `trials[i] >= limite_abandono`, la fuente se declara agotada.

El scout no genera una solución completamente aleatoria: genera un vecino de la **mejor fuente activa** en ese momento. Esto equilibra exploración y explotación: se explora una zona nueva del espacio, pero cercana a la región de mejor calidad conocida.

```
# Pseudocódigo de la fase scout
best_idx = argmin_i(obj_fuente(i))
a_reiniciar = [i for i in fuentes if trials[i] >= limite_abandono]
for i in a_reiniciar:
    fuentes_sol[i] = generar_vecino(fuentes_sol[best_idx])
    trials[i] = 0
    scouts += 1
```

### El contador de intentos fallidos (`trial`)

`trials` es una lista de enteros de longitud `num_fuentes`. `trials[i]` cuenta cuántas veces consecutivas la fuente i no ha mejorado. Un intento "fallido" ocurre en dos situaciones:

1. En la fase empleada: el vecino generado no mejora el objetivo penalizado de la fuente i.
2. En la fase observadora: la fuente i fue seleccionada por una observadora pero el vecino tampoco mejora.

Cuando el vecino sí mejora, `trials[i]` se resetea a cero. Esto significa que una fuente activa puede acumular fallos de forma no consecutiva: si mejora ocasionalmente, nunca alcanza el límite. El scout solo interviene cuando la fuente lleva `limite_abandono` intentos sin ninguna mejora.

---

## La función `busqueda_abejas`

### Firma exacta

```python
def busqueda_abejas(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    iteraciones: int = 250,
    num_fuentes: int = 16,
    limite_abandono: int = 35,
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
) -> AbejasResult
```

### Tabla de parámetros

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `inicial_obj` | `Any` | — | Objeto con soluciones candidatas iniciales (pickle cargado). Puede ser una solución directa, un dict o una estructura anidada; se extrae con `extraer_candidatas_desde_objeto`. |
| `data` | `Mapping[str, Any]` | — | Datos completos de la instancia CARP (cargados con `load_instances`). |
| `G` | `nx.Graph` | — | Grafo de la instancia (cargado con `cargar_objeto_gexf`). |
| `iteraciones` | `int` | `250` | Número de ciclos completos (empleadas + observadoras + scouts). Debe ser > 0. |
| `num_fuentes` | `int` | `16` | Número de fuentes de alimento mantenidas en paralelo. Debe ser >= 2. |
| `limite_abandono` | `int` | `35` | Intentos fallidos consecutivos antes de que una fuente sea abandonada por un scout. Debe ser > 0. |
| `semilla` | `int \| None` | `None` | Semilla para el generador aleatorio (`random.Random`). `None` = no reproducible. |
| `operadores` | `Iterable[str]` | `OPERADORES_POPULARES` | Conjunto de operadores de vecindario habilitados. |
| `marcador_depot_etiqueta` | `str \| None` | `None` | Token del depósito en las rutas. Si `None`, se infiere del contexto (`ctx.marcador_depot`). |
| `usar_gpu` | `bool` | `False` | Si `True`, intenta evaluar lotes en GPU con CuPy. Hace fallback a CPU si CuPy no está disponible. |
| `backend_vecindario` | `Literal["labels", "ids"]` | `"labels"` | Representación interna para la generación de vecinos: `"labels"` usa strings, `"ids"` usa enteros. |
| `guardar_historial` | `bool` | `True` | Si `True`, registra el mejor costo al inicio de cada iteración en `historial_mejor_costo`. |
| `guardar_csv` | `bool` | `False` | Si `True`, escribe los resultados de la corrida en un archivo CSV al finalizar. |
| `ruta_csv` | `str \| None` | `None` | Ruta del archivo CSV. Si `None`, se genera automáticamente como `resultados_busqueda_abejas_<instancia>.csv`. |
| `nombre_instancia` | `str` | `"instancia"` | Nombre identificador de la instancia. Aparece en el CSV y en el nombre del archivo de salida. |
| `id_corrida` | `str \| None` | `None` | Identificador de corrida para el CSV (útil en experimentos repetidos). |
| `config_id` | `str \| None` | `None` | Identificador de configuración de parámetros para el CSV. |
| `repeticion` | `int \| None` | `None` | Número de repetición del experimento. |
| `root` | `str \| None` | `None` | Directorio raíz donde buscar archivos de datos. `None` usa el directorio por defecto del paquete. |
| `usar_penalizacion_capacidad` | `bool` | `True` | Si `True`, el objetivo incluye `lambda * violacion_capacidad`. Permite explorar temporalmente soluciones infactibles. |
| `lambda_capacidad` | `float \| None` | `None` | Peso λ de la penalización. Si `None`, se calcula automáticamente con `lambda_penal_capacidad_por_defecto` (~10 × mediana de la matriz de distancias). |
| `extra_csv` | `dict[str, object] \| None` | `None` | Columnas adicionales que se añaden al registro CSV de esta corrida. |

### Qué retorna

La función retorna un objeto `AbejasResult` inmutable (`frozen=True`) con todos los datos de la corrida.

### Validaciones de parámetros

La función lanza `ValueError` en los siguientes casos:
- `iteraciones <= 0`
- `num_fuentes <= 1`
- `limite_abandono <= 0`

---

## `AbejasResult`

`AbejasResult` es un `@dataclass(frozen=True, slots=True)`: inmutable, con bajo uso de memoria por instancia. Todos sus campos son de solo lectura.

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `mejor_solucion` | `list[list[str]]` | La mejor solución CARP encontrada. Lista de rutas; cada ruta es lista de etiquetas (`["D", "TR3", "TR7", "D"]`). Se prefiere la mejor solución factible sobre la mejor infactible. |
| `mejor_costo` | `float` | Costo total de `mejor_solucion` (calculado sin penalización). |
| `solucion_inicial_referencia` | `list[list[str]]` | Solución inicial usada como referencia para medir la mejora. |
| `costo_solucion_inicial` | `float` | Costo puro de la solución inicial de referencia. |
| `mejora_absoluta` | `float` | `costo_solucion_inicial - mejor_costo`. Positivo indica que se mejoró. |
| `mejora_porcentaje_inicial_vs_final` | `float` | `mejora_absoluta / costo_solucion_inicial × 100`. |
| `tiempo_segundos` | `float` | Tiempo total de la corrida en segundos (medido con `time.perf_counter`). |
| `iteraciones_totales` | `int` | Número de iteraciones ejecutadas (igual al parámetro `iteraciones`). |
| `fuentes_alimento` | `int` | Número de fuentes de alimento mantenidas (igual a `num_fuentes`). |
| `scouts_reinicios` | `int` | Número de veces que se ejecutó la fase scout (reinicio de una fuente agotada). |
| `mejoras` | `int` | Número de iteraciones en las que el mejor global mejoró. |
| `semilla` | `int \| None` | Semilla usada. `None` si no se especificó. |
| `backend_evaluacion` | `str` | Backend de evaluación realmente usado: `"cpu"` o `"gpu"`. |
| `historial_mejor_costo` | `list[float]` | Costo del mejor global al inicio de cada iteración. Lista vacía si `guardar_historial=False`. |
| `ultimo_movimiento_aceptado` | `MovimientoVecindario \| None` | Último operador de vecindario que produjo una mejora aceptada. |
| `operadores_propuestos` | `dict[str, int]` | Conteo de veces que cada operador generó un vecino (propuesto). Ordenado por frecuencia descendente. |
| `operadores_aceptados` | `dict[str, int]` | Conteo de veces que cada operador produjo un vecino aceptado (reemplazó la fuente). |
| `operadores_mejoraron` | `dict[str, int]` | Conteo de veces que cada operador produjo una mejora del mejor global. |
| `operadores_trayectoria_mejor` | `dict[str, int]` | Snapshot de `operadores_aceptados` en el momento en que se encontró la mejor solución. |
| `usar_penalizacion_capacidad` | `bool` | Indica si la penalización de capacidad estuvo activa. |
| `lambda_capacidad` | `float` | Valor efectivo de λ usado durante la corrida. |
| `n_iniciales_evaluados` | `int` | Número de candidatas iniciales evaluadas para seleccionar la mejor. |
| `iniciales_infactibles_aceptadas` | `int` | Candidatas iniciales que violaban restricciones de capacidad. |
| `aceptaciones_solucion_infactible` | `int` | Número de veces que se aceptó una solución que viola la capacidad (en cualquier fase). |
| `mejor_solucion_factible_final` | `bool` | `True` si `mejor_solucion` respeta todas las restricciones de capacidad. |
| `archivo_csv` | `str \| None` | Ruta absoluta del CSV guardado. `None` si `guardar_csv=False`. |

---

## Ejemplo completo de uso

### Uso directo con recursos ya cargados

```python
from metacarp.abejas import busqueda_abejas
from metacarp.cargar_grafos import cargar_objeto_gexf
from metacarp.cargar_soluciones_iniciales import cargar_solucion_inicial
from metacarp.instances import load_instances

# 1. Cargar los recursos de la instancia
nombre = "EGL-E1-A"
data = load_instances(nombre)
G = cargar_objeto_gexf(nombre)
inicial_obj = cargar_solucion_inicial(nombre)

# 2. Ejecutar el algoritmo ABC
resultado = busqueda_abejas(
    inicial_obj,
    data,
    G,
    iteraciones=300,
    num_fuentes=20,
    limite_abandono=40,
    semilla=42,
    guardar_historial=True,
    usar_penalizacion_capacidad=True,
)

# 3. Inspeccionar resultados
print(f"Mejor costo encontrado : {resultado.mejor_costo:.2f}")
print(f"Costo inicial          : {resultado.costo_solucion_inicial:.2f}")
print(f"Mejora absoluta        : {resultado.mejora_absoluta:.2f}")
print(f"Mejora porcentual      : {resultado.mejora_porcentaje_inicial_vs_final:.2f}%")
print(f"Tiempo de ejecución    : {resultado.tiempo_segundos:.2f}s")
print(f"Reinicios scout        : {resultado.scouts_reinicios}")
print(f"Iteraciones con mejora : {resultado.mejoras}")
print(f"Solución factible      : {resultado.mejor_solucion_factible_final}")
print(f"Backend de evaluación  : {resultado.backend_evaluacion}")

# 4. Ver la mejor solución ruta por ruta
for i, ruta in enumerate(resultado.mejor_solucion, start=1):
    print(f"  Ruta {i}: {' -> '.join(ruta)}")

# 5. Ver los operadores más efectivos
print("\nOperadores que más mejoraron el global:")
for op, n in resultado.operadores_mejoraron.items():
    print(f"  {op}: {n}")
```

### Uso con la función de conveniencia

```python
from metacarp.abejas import busqueda_abejas_desde_instancia

# Equivalente al bloque anterior, en una sola llamada
resultado = busqueda_abejas_desde_instancia(
    "EGL-E1-A",
    iteraciones=300,
    num_fuentes=20,
    limite_abandono=40,
    semilla=42,
    guardar_csv=True,
    ruta_csv="resultados/abejas_EGL-E1-A.csv",
    nombre_instancia="EGL-E1-A",
    id_corrida="experimento_01",
    repeticion=1,
)
```

### Análisis del historial de convergencia

```python
import matplotlib.pyplot as plt

resultado = busqueda_abejas_desde_instancia("EGL-E1-A", semilla=42)

plt.plot(resultado.historial_mejor_costo)
plt.xlabel("Iteración")
plt.ylabel("Mejor costo")
plt.title("Convergencia de ABC en EGL-E1-A")
plt.grid(True)
plt.show()
```

---

## Guía de ajuste de parámetros

La calidad de los resultados de ABC depende del equilibrio entre exploración (diversidad de la búsqueda) y explotación (intensidad sobre las mejores zonas). Los tres parámetros principales que controlan ese equilibrio son:

| Parámetro | Default | Efecto al aumentar | Efecto al reducir | Rango recomendado |
|-----------|---------|-------------------|-------------------|-------------------|
| `num_fuentes` | `16` | Mayor diversificación: se exploran más zonas del espacio. Más costoso computacionalmente por iteración. | Mayor intensificación en pocas zonas. Riesgo de convergencia prematura. | `8`–`32` para instancias medianas; `16`–`64` para instancias grandes |
| `limite_abandono` | `35` | Las fuentes se mantienen más tiempo antes de ser reemplazadas: mayor intensificación local. | Los scouts actúan más frecuentemente: mayor diversificación. | `10`–`100`; se recomienda ~2–3 × `num_fuentes` |
| `iteraciones` | `250` | Más tiempo de búsqueda; mejores resultados en instancias difíciles. Tiempo de ejecución lineal. | Corridas más rápidas; útil para calibración rápida. | `100`–`500` para uso normal; `>500` para instancias grandes |

### Notas de ajuste

**`num_fuentes` y `limite_abandono` interactúan:** con muchas fuentes y un límite bajo, los scouts actúan frecuentemente sobre un espacio grande, lo que favorece la exploración. Con pocas fuentes y un límite alto, el algoritmo se comporta más como una búsqueda de vecindario con pocos reinicios.

**Regla práctica para `limite_abandono`:** en el artículo original de Karaboga, se recomienda `limite_abandono = num_fuentes × dimension_problema / 2`. Para CARP, la "dimensión" puede aproximarse al número de tareas requeridas de la instancia. En la práctica, valores de `30`–`50` funcionan bien para la mayoría de instancias del benchmark EGL.

**`lambda_capacidad` automático:** cuando se deja en `None`, el sistema calibra λ como ~10 × mediana de la matriz de distancias de la instancia. Este valor escala apropiadamente con el tamaño del grafo y es un buen punto de partida. Si la mejor solución final es infactible (`mejor_solucion_factible_final=False`), considerar aumentar λ manualmente.

**`usar_gpu`:** solo tiene efecto si CuPy está instalado y hay un dispositivo CUDA disponible. Para instancias pequeñas o medianas, el overhead de transferencia de datos puede superar la ganancia; el modo GPU es más beneficioso con `num_fuentes >= 32` y muchas iteraciones.

---

## Comparación con otras metaheurísticas del proyecto

MetaCARP incluye tres metaheurísticas para resolver CARP: Búsqueda Tabú (tabú), Recocido Simulado (SA) y Colonia de Abejas (ABC). Las tres usan los mismos operadores de vecindario y el mismo evaluador rápido vectorizado, pero difieren en su estrategia de búsqueda.

| Característica | Búsqueda Tabú | Recocido Simulado | Colonia de Abejas |
|----------------|--------------|-------------------|-------------------|
| Tipo | Trayectoria única | Trayectoria única | Poblacional |
| Soluciones activas | 1 | 1 | `num_fuentes` (16 por defecto) |
| Mecanismo de aceptación | Mejora + lista tabú | Mejora + criterio probabilístico | Greedy por fuente + ruleta |
| Diversificación | Lista tabú prohíbe movimientos recientes | Temperatura controla aceptación de peores soluciones | Scouts reemplazan fuentes agotadas |
| Intensificación | Aspiration criterion | Enfriamiento gradual | Observadoras concentran búsqueda en mejores fuentes |
| Memoria | Lista tabú explícita | Temperatura como memoria implícita | `trials` por fuente |
| Paralelismo natural | No | No | Sí (evaluación de fuentes en lote) |
| Parámetros clave | Tamaño de lista tabú, iteraciones | Temperatura inicial/final, factor de enfriamiento | `num_fuentes`, `limite_abandono`, `iteraciones` |

### Cuándo preferir ABC

**ABC es preferible sobre Tabú o SA cuando:**

- Se dispone de tiempo de cómputo suficiente para múltiples evaluaciones por iteración: ABC evalúa `2 × num_fuentes` vecinos por iteración (empleadas + observadoras), mientras que Tabú y SA evalúan uno o pocos.
- La instancia es difícil y el espacio de soluciones tiene muchos óptimos locales: el mantenimiento de múltiples fuentes reduce el riesgo de quedar atrapado en un único mínimo local.
- Se quiere diversificación automática sin ajustar manualmente una temperatura de enfriamiento (SA) ni el tamaño de la lista tabú (Tabú).
- Se ejecuta un grid search o un experimento con múltiples corridas: el contexto de evaluación (`ContextoEvaluacion`) se construye una vez y puede reutilizarse entre corridas.

**Tabú o SA pueden preferirse sobre ABC cuando:**

- El presupuesto de evaluaciones es muy limitado: con pocas iteraciones, una trayectoria única bien dirigida (Tabú) puede superar a ABC, que "reparte" el esfuerzo entre `num_fuentes` soluciones.
- Se necesita control fino de la intensidad de la búsqueda mediante la temperatura (SA) o el horizonte de memoria (Tabú), y se dispone de conocimiento previo sobre la instancia.
- La instancia es pequeña y el espacio de soluciones es manejable: la diversificación de ABC es menos necesaria cuando una trayectoria única puede recorrer una fracción significativa del espacio.

En experimentos comparativos, la ventaja de ABC sobre los algoritmos de trayectoria única se vuelve más pronunciada al aumentar el número de iteraciones totales y el tamaño de la instancia, ya que el mantenimiento de múltiples fuentes amortiza su costo con mejores cotas de solución.
