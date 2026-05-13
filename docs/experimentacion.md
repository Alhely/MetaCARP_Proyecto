# Sistema de experimentación — MetaCARP

`scripts/experimentos.py` es el punto de entrada principal para ejecutar campañas de experimentación controladas sobre las cuatro metaheurísticas del proyecto. Recibe parámetros por línea de comandos, construye el espacio de búsqueda de hiperparámetros de cada algoritmo mediante producto cartesiano, ejecuta cada combinación (instancia × configuración × repetición) con una semilla derivada única, y persiste los resultados en archivos CSV. Su propósito en el pipeline de tesis es generar el conjunto de datos empírico sobre el que se realiza el análisis comparativo de calidad de solución, comportamiento de operadores y sensibilidad a hiperparámetros.

---

## Conceptos clave

**Instancia.** Un problema CARP concreto: un grafo con aristas que tienen demanda y una capacidad de vehículo. Las instancias se identifican por nombre (ej. `gdb1`, `kshs3`) y se cargan desde `PickleInstances/` mediante `InstanceStore`.

**Configuración.** Una combinación específica de hiperparámetros para una metaheurística. Por ejemplo, SA con `temperatura_inicial=300.0`, `alpha=0.95`, `iteraciones_por_temperatura=80`, `max_enfriamientos=100`. El script construye todas las configuraciones posibles mediante un grid search (producto cartesiano de los valores candidatos de cada parámetro).

**Corrida.** La ejecución de una metaheurística sobre una instancia con una configuración y semilla determinadas. Cada corrida produce exactamente una fila en el CSV de salida.

**Repetición.** El script ejecuta cada configuración `--repeticiones` veces (por defecto 2) con semillas diferentes, para estimar la variabilidad del resultado frente a la aleatoriedad interna del algoritmo.

**Grid search.** Exploración exhaustiva del espacio de hiperparámetros: se evalúan todas las combinaciones posibles definidas en `_construir_runners()`. La función `_grid()` implementa el producto cartesiano con `itertools.product`.

---

## Espacio de hiperparámetros

### Recocido Simulado (`sa`)

| Parámetro | Valores candidatos | Cardinalidad |
|---|---|---|
| `temperatura_inicial` | 150.0, 300.0, 500.0, 800.0 | 4 |
| `temperatura_minima` | 1e-3 | 1 |
| `alpha` | 0.90, 0.93, 0.95, 0.97 | 4 |
| `iteraciones_por_temperatura` | 40, 80, 120 | 3 |
| `max_enfriamientos` | 60, 100 | 2 |

**Total de configuraciones:** 4 × 1 × 4 × 3 × 2 = **96**

### Búsqueda Tabú (`tabu`)

| Parámetro | Valores candidatos | Cardinalidad |
|---|---|---|
| `iteraciones` | 500, 700, 1000 | 3 |
| `tam_vecindario` | 20, 30, 40, 60 | 4 |
| `tenure_tabu` | 10, 15, 20, 30 | 4 |

**Total de configuraciones:** 3 × 4 × 4 = **48**

### Colonia de Abejas (`abejas`)

| Parámetro | Valores candidatos | Cardinalidad |
|---|---|---|
| `iteraciones` | 500, 700, 1000 | 3 |
| `num_fuentes` | 10, 20, 30, 40 | 4 |
| `limite_abandono` | 15, 30, 45, 60 | 4 |

**Total de configuraciones:** 3 × 4 × 4 = **48**

### Cuckoo Search (`cuckoo`)

| Parámetro | Valores candidatos | Cardinalidad |
|---|---|---|
| `iteraciones` | 500, 750, 1000 | 3 |
| `num_nidos` | 15, 25, 35 | 3 |
| `pa_abandono` | 0.15, 0.25, 0.35 | 3 |
| `pasos_levy_base` | 2, 3, 5 | 3 |
| `beta_levy` | 1.2, 1.5 | 2 |

**Total de configuraciones:** 3 × 3 × 3 × 3 × 2 = **162**

### Resumen de configuraciones totales

| Metaheurística | Alias | Configuraciones |
|---|---|---|
| Recocido Simulado | `sa` | 96 |
| Búsqueda Tabú | `tabu` | 48 |
| Colonia de Abejas | `abejas` | 48 |
| Cuckoo Search | `cuckoo` | 162 |
| **Total** | | **354** |

---

## Sistema de semillas

Cada corrida individual recibe una semilla única derivada de una semilla base (`--seed`, por defecto 42) mediante la fórmula:

```
semilla = seed_base
        + (idx_instancia   × 100_000)
        + (idx_metaheurística × 10_000)
        + (idx_configuración  ×    100)
        + repeticion
```

Donde los índices empiezan en 0 (instancias, metaheurísticas, configuraciones) y `repeticion` empieza en 1.

**Ejemplo.** Con `seed_base=42`, instancia `gdb1` (idx=0), metaheurística `sa` (idx=0), configuración 1 (idx_cfg=1), repetición 1:

```
semilla = 42 + (0 × 100_000) + (0 × 10_000) + (1 × 100) + 1 = 143
```

**Propiedades del esquema:**

1. **Reproducibilidad total.** Dado el mismo `seed_base` y los mismos argumentos de línea de comandos, todas las corridas producen exactamente los mismos resultados.
2. **Diversidad entre corridas.** Cada combinación `(instancia, meta, config, rep)` recibe una semilla aritméticamente distinta, por lo que los números aleatorios generados en cada corrida son independientes entre sí.
3. **Sin colisiones.** Los multiplicadores (100 000, 10 000, 100, 1) garantizan que ninguna combinación de índices razonables produzca la misma semilla: el espacio disponible soporta hasta 999 instancias, 9 metaheurísticas, 99 configuraciones y 99 repeticiones sin solapamiento.

---

## Estructura del CSV de salida

El script guarda los resultados en la ruta:

```
<salida_dir>/<metaheuristica>/<meta>_<instancia>_<experimento>_<ydmh>.csv
```

Donde `ydmh` es un timestamp fijo al inicio de la campaña con formato `%Y%d%m%H%M`. Cada fila del CSV corresponde a una corrida individual. Las columnas se agrupan en cinco categorías.

### Identificación de la corrida

| Columna | Descripción |
|---|---|
| `metaheuristica` | Nombre canónico (`recocido_simulado`, `busqueda_tabu`, `busqueda_abejas`, `cuckoo_search`) |
| `instancia` | Nombre de la instancia (ej. `gdb1`, `kshs3`) |
| `bks_referencia` | Valor BKS de la literatura para esta instancia |
| `bks_origen` | Fuente del BKS (`BKS`, `lower_bound`, etc.) |
| `repeticion` | Número de repetición (1, 2, …) |
| `semilla` | Semilla derivada usada en esta corrida |
| `tiempo_segundos` | Duración real de la corrida |

### Parámetros específicos de cada metaheurística

Cada metaheurística escribe las columnas de sus propios hiperparámetros. Las columnas presentes dependen del algoritmo ejecutado.

**SA:** `temperatura_inicial`, `temperatura_minima`, `alpha`, `iteraciones_por_temperatura`, `max_enfriamientos`

**Tabu:** `iteraciones`, `tam_vecindario`, `tenure_tabu`

**Abejas:** `iteraciones`, `num_fuentes`, `limite_abandono`

**Cuckoo:** `iteraciones`, `num_nidos`, `pa_abandono`, `pasos_levy_base`, `beta_levy`

### Métricas de rendimiento y calidad

| Columna | Descripción |
|---|---|
| `mejor_costo` | Costo de la mejor solución encontrada |
| `costo_solucion_inicial` | Costo de la solución con la que arrancó la búsqueda (referencia) |
| `aceptadas` | Total de movimientos aceptados durante la búsqueda |
| `mejoras` | Total de veces que se encontró una solución mejor |
| `gap_bks_porcentaje` | **Métrica principal de tesis:** distancia relativa al BKS en porcentaje |

La métrica `gap_bks_porcentaje` se calcula como:

```
gap_bks_porcentaje = (mejor_costo − bks_referencia) / bks_referencia × 100
```

Un valor de 0 indica que el algoritmo alcanzó el óptimo conocido. Un valor de 5 indica que la solución encontrada es un 5 % peor que el mejor resultado reportado en la literatura. Esta métrica es la base de las comparaciones estadísticas del capítulo de resultados.

### Columnas de operadores de vecindario

El sistema registra las estadísticas de los 7 operadores de vecindario en 4 categorías, produciendo 28 columnas. Los operadores son:

| Operador | Tipo | Descripción |
|---|---|---|
| `relocate_intra` | intra-ruta | Mueve una tarea a otra posición dentro de su misma ruta |
| `swap_intra` | intra-ruta | Intercambia dos tareas dentro de la misma ruta |
| `2opt_intra` | intra-ruta | Invierte un segmento dentro de una ruta |
| `relocate_inter` | inter-ruta | Mueve una tarea de una ruta a otra |
| `swap_inter` | inter-ruta | Intercambia una tarea entre dos rutas distintas |
| `2opt_star` | inter-ruta | Reencadena segmentos finales de dos rutas |
| `cross_exchange` | inter-ruta | Intercambia segmentos completos entre dos rutas |

Las cuatro categorías de estadísticas registradas para cada operador:

| Prefijo de columna | Qué cuenta |
|---|---|
| `propuesto_<op>` | Cuántas veces fue seleccionado para generar un vecino |
| `aceptado_<op>` | Cuántas veces el vecino generado fue aceptado como nueva solución actual |
| `mejoraron_<op>` | Cuántas veces el vecino generado mejoró el mejor costo global |
| `trayectoria_mejor_<op>` | Cuántas veces aparece en la secuencia de movimientos que llevó al mejor resultado |

Ejemplo de nombre de columna: `aceptado_2opt_intra`, `trayectoria_mejor_cross_exchange`.

### Columnas complementarias

| Columna | Descripción |
|---|---|
| `mejor_solucion_factible_final` | `True` si la mejor solución encontrada es factible (respeta capacidades) |
| `mejor_solucion_tr_legible` | Representación textual de la solución: `R1: D -> TR3 -> TR7 -> D \|\| ...` |
| `reporte_detalle_deadheading` | Desglose de costos de arrastre (*deadheading*) por ruta |
| `costo_total_desde_reporte` | Verificación cruzada del costo calculado desde el reporte textual |

---

## Sesgo dinámico de operadores inter-ruta

Cuando la solución actual viola restricciones de capacidad, el sistema activa un mecanismo de sesgo implementado en `pesos_inter_bias()` (módulo `metaheuristicas_utils`). La justificación es directa: una ruta que viola capacidad tiene **demasiada demanda asignada**, por lo que la única forma de corregirlo es mover tareas de esa ruta a otras. Eso requiere operadores **inter-ruta** (`relocate_inter`, `swap_inter`, `2opt_star`, `cross_exchange`). Los operadores intra-ruta solo reordenan tareas dentro de la misma ruta y no pueden reducir la demanda de ninguna ruta.

El parámetro que controla el sesgo es `alpha_inter=0.8`: la fracción de probabilidad total asignada en conjunto a los cuatro operadores inter-ruta cuando hay violación.

| Estado de la solución | Selección | P(cada op. inter) | P(cada op. intra) |
|---|---|---|---|
| Con violación de capacidad | sesgada (`alpha_inter=0.8`) | 80% / 4 = 20.0% | 20% / 3 ≈ 6.7% |
| Sin violación (factible) | uniforme | 1/7 ≈ 14.3% | 1/7 ≈ 14.3% |

El sesgo se desactiva automáticamente en cuanto la solución vuelve a ser factible (`violacion ≤ 1e-12`), y se reactiva en cualquier iteración en que la solución actual viole capacidad.

**Cómo leerlo en el CSV.** Las columnas `propuesto_*` reflejan directamente la distribución de selección a lo largo de la corrida. Si hubo muchas iteraciones con soluciones infactibles, se espera observar una proporción elevada de `propuesto_relocate_inter + propuesto_swap_inter + propuesto_2opt_star + propuesto_cross_exchange` respecto al total.

---

## Cálculo del total de corridas

La fórmula general es:

```
corridas_totales = Σ(meta ∈ metas) (configs_meta × repeticiones) × num_instancias
```

**Ejemplo de campaña completa** con 23 instancias, 2 repeticiones, las cuatro metaheurísticas:

```
SA:     96 configs × 2 reps × 23 instancias =  4 416 corridas
Tabu:   48 configs × 2 reps × 23 instancias =  2 208 corridas
Abejas: 48 configs × 2 reps × 23 instancias =  2 208 corridas
Cuckoo: 162 configs × 2 reps × 23 instancias = 7 452 corridas
─────────────────────────────────────────────────────────────
Total:                                         16 284 corridas
```

El script imprime `Corridas planeadas: <N>` al inicio de la ejecución para que sea posible estimar el tiempo total antes de lanzar la campaña completa.

---

## Comandos de uso

> **Nota:** la variable de entorno `CARPTHESIS_ROOT` indica al paquete dónde encontrar los datos (pickles, matrices, grafos). El intérprete del entorno conda es necesario para disponer de CuPy si se usa `--usar-gpu`.

### Corrida mínima de prueba

```bash
CARPTHESIS_ROOT=/home/alhely/Desktop/MetaCARP_Proyecto \
/home/alhely/miniconda3/envs/carp_gpu2/bin/python scripts/experimentos.py \
  --metaheuristicas sa \
  --instancias gdb1 \
  --repeticiones 1 \
  --salida-dir scripts/testing_20260512 \
  --experimento prueba
```

Ejecuta SA sobre `gdb1` con todas sus 96 configuraciones (1 repetición cada una). Produce un único CSV en `scripts/testing_20260512/sa/`.

### Campaña SA sobre 23 instancias con GPU

```bash
CARPTHESIS_ROOT=/home/alhely/Desktop/MetaCARP_Proyecto \
/home/alhely/miniconda3/envs/carp_gpu2/bin/python scripts/experimentos.py \
  --metaheuristicas sa \
  --instancias gdb19 kshs1 kshs2 kshs3 kshs4 kshs5 kshs6 \
              gdb4 gdb14 gdb15 gdb1 gdb20 gdb3 gdb6 gdb7 \
              gdb12 gdb10 gdb2 gdb5 gdb13 gdb16 gdb17 gdb21 \
  --repeticiones 2 \
  --usar-gpu \
  --salida-dir scripts/experimentos \
  --experimento tesis
```

### Campaña completa (todas las metaheurísticas, todas las instancias)

```bash
CARPTHESIS_ROOT=/home/alhely/Desktop/MetaCARP_Proyecto \
/home/alhely/miniconda3/envs/carp_gpu2/bin/python scripts/experimentos.py \
  --repeticiones 2 \
  --usar-gpu \
  --salida-dir scripts/experimentos \
  --experimento tesis
```

Cuando no se especifica `--instancias` ni `--metaheuristicas`, el script resuelve ambos a `all` usando `nombres_soluciones_iniciales_disponibles()`.

### Inspección rápida del CSV generado

```bash
column -t -s, scripts/testing_20260512/sa/sa_gdb1_prueba_*.csv | less -S
```

### Referencia de argumentos

| Argumento | Tipo | Default | Descripción |
|---|---|---|---|
| `--instancias` | `str...` | `all` | Lista de instancias a ejecutar; `all` toma todas las disponibles |
| `--metaheuristicas` | `str...` | `all` | Subconjunto: `sa tabu abejas cuckoo`; `all` ejecuta las cuatro |
| `--seed` | `int` | `42` | Semilla base para derivar todas las semillas de corrida |
| `--repeticiones` | `int` | `2` | Repeticiones por configuración |
| `--experimento` | `str` | `tesis` | Etiqueta incluida en el nombre del archivo CSV |
| `--salida-dir` | `str` | `experimentos` | Carpeta raíz donde se crean las subcarpetas por metaheurística |
| `--usar-gpu` | flag | desactivado | Activa evaluación por lotes con CuPy; hace fallback a CPU si no está disponible |
| `--root` | `str` | `None` | Raíz de datos alternativa; si no se pasa, usa `CARPTHESIS_ROOT` o la ruta del paquete |

---

## Estructura de salida en disco

```
<salida_dir>/
├── sa/
│   └── sa_<instancia>_<experimento>_<ydmh>.csv   # un archivo por instancia
├── tabu/
│   └── tabu_<instancia>_<experimento>_<ydmh>.csv
├── abejas/
│   └── abejas_<instancia>_<experimento>_<ydmh>.csv
└── cuckoo/
    └── cuckoo_<instancia>_<experimento>_<ydmh>.csv
```

Dentro de cada CSV, cada fila es una corrida. Para una instancia ejecutada con SA (96 configuraciones × 2 repeticiones), el CSV tendrá 192 filas.

---

## Documentación relacionada

- `docs/recocido_simulado.md` — descripción detallada de SA y sus parámetros
- `docs/busqueda_tabu.md` — descripción detallada de Búsqueda Tabú
- `docs/colonia_abejas.md` — descripción detallada de ABC
- `docs/cuckoo_search.md` — descripción detallada de Cuckoo Search
- `docs/generacion_vecinos.md` — catálogo de los 7 operadores de vecindario
