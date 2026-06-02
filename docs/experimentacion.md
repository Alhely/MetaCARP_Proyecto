# Sistema de experimentación — MetaCARP

`scripts/experimentos.py` es el punto de entrada principal para ejecutar campañas de experimentación controladas sobre las cuatro metaheurísticas del proyecto. Recibe parámetros por línea de comandos, construye el espacio de búsqueda de hiperparámetros de cada algoritmo mediante producto cartesiano, ejecuta cada combinación (instancia × configuración × repetición) con una semilla derivada única, y persiste los resultados en archivos CSV. Su propósito en el pipeline de tesis es generar el conjunto de datos empírico sobre el que se realiza el análisis comparativo de calidad de solución, comportamiento de operadores y sensibilidad a hiperparámetros.

---

## Conceptos clave

**Instancia.** Un problema CARP concreto: un grafo con aristas que tienen demanda y una capacidad de vehículo. Las instancias se identifican por nombre (ej. `gdb1`, `kshs3`) y se cargan desde `PickleInstances/` mediante `InstanceStore`.

**Configuración.** Una combinación específica de hiperparámetros para una metaheurística. Por ejemplo, SA con `temperatura_inicial=None` (calibración automática desde la instancia) y `alpha=0.92`. El script construye todas las configuraciones posibles mediante un grid search (producto cartesiano de los valores candidatos de cada parámetro).

**Corrida.** La ejecución de una metaheurística sobre una instancia con una configuración y semilla determinadas. Cada corrida produce exactamente una fila en el CSV de salida.

**Repetición.** El script ejecuta cada configuración `--repeticiones` veces (por defecto 2) con semillas diferentes, para estimar la variabilidad del resultado frente a la aleatoriedad interna del algoritmo.

**Grid search.** Exploración exhaustiva del espacio de hiperparámetros: se evalúan todas las combinaciones posibles definidas en `_construir_runners()`. La función `_grid()` implementa el producto cartesiano con `itertools.product`.

---

## Espacio de hiperparámetros

### Recocido Simulado (`sa`)

| Parámetro | Valores candidatos | Cardinalidad |
|---|---|---|
| `temperatura_inicial` | None, 300.0, 500.0, 800.0 | 4 |
| `temperatura_minima` | 1e-3 | 1 |
| `alpha` | 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99 | 11 |

`temperatura_inicial=None` activa la calibración automática: el valor se calcula como `20 · d_max / n`, donde `n` es el número de arcos requeridos de la instancia y `d_max` es la distancia máxima en la matriz Dijkstra. `temperatura_minima=1e-3` es siempre fijo.

**Total de configuraciones:** 4 × 1 × 11 = **44**

### Búsqueda Tabú (`tabu`)

| Parámetro | Valores candidatos | Cardinalidad |
|---|---|---|
| `iteraciones` | 400, 700 | 2 |
| `tam_vecindario` | 25, 40, 60 | 3 |
| `tenure_tabu` | 7, 15, 25 | 3 |

**Total de configuraciones:** 2 × 3 × 3 = **18**

### Colonia de Abejas (`abejas`)

| Parámetro | Valores candidatos | Cardinalidad |
|---|---|---|
| `iteraciones` | 300, 600 | 2 |
| `num_fuentes` | 10, 20, 30 | 3 |
| `limite_abandono` | 20, 40, 60 | 3 |

**Total de configuraciones:** 2 × 3 × 3 = **18**

### Cuckoo Search (`cuckoo`)

| Parámetro | Valores candidatos | Cardinalidad |
|---|---|---|
| `iteraciones` | 400, 700 | 2 |
| `num_nidos` | 15, 25, 35 | 3 |
| `pa_abandono` | 0.20, 0.25, 0.30 | 3 |
| `pasos_levy_base` | 3 | 1 |
| `beta_levy` | 1.5 | 1 |

**Total de configuraciones:** 2 × 3 × 3 × 1 × 1 = **18**

### Resumen de configuraciones totales

| Metaheurística | Alias | Configuraciones |
|---|---|---|
| Recocido Simulado | `sa` | 44 |
| Búsqueda Tabú | `tabu` | 18 |
| Colonia de Abejas | `abejas` | 18 |
| Cuckoo Search | `cuckoo` | 18 |
| **Total** | | **98** |

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

**SA:** `temperatura_inicial`, `temperatura_minima`, `alpha`, `L` (longitud de cadena de Markov, calculado como `n²`; no se escribe en el CSV como hiperparámetro configurable). `temperatura_inicial` y `temperatura_minima` pueden aparecer como `None` cuando se usó calibración automática.

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

El sistema registra las estadísticas de los 9 operadores de vecindario en 4 categorías, produciendo 36 columnas. Los operadores son:

| Operador | Tipo | Descripción |
|---|---|---|
| `relocate_intra` | intra-ruta | Mueve una tarea a otra posición dentro de su misma ruta |
| `swap_intra` | intra-ruta | Intercambia dos tareas dentro de la misma ruta |
| `2opt_intra` | intra-ruta | Invierte un segmento dentro de una ruta |
| `relocate_inter` | inter-ruta | Mueve una tarea de una ruta a otra |
| `swap_inter` | inter-ruta | Intercambia una tarea entre dos rutas distintas |
| `2opt_star` | inter-ruta | Reencadena segmentos finales de dos rutas |
| `cross_exchange` | inter-ruta | Intercambia segmentos completos entre dos rutas |
| `or_opt_2` | inter-ruta | Mueve un bloque de 2 tareas consecutivas a otra ruta |
| `or_opt_3` | inter-ruta | Mueve un bloque de 3 tareas consecutivas a otra ruta |

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
SA:     44 configs × 2 reps × 23 instancias = 2 024 corridas
Tabu:   18 configs × 2 reps × 23 instancias =   828 corridas
Abejas: 18 configs × 2 reps × 23 instancias =   828 corridas
Cuckoo: 18 configs × 2 reps × 23 instancias =   828 corridas
─────────────────────────────────────────────────────────────
Total:                                          4 508 corridas
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

Ejecuta SA sobre `gdb1` con todas sus 44 configuraciones (1 repetición cada una). Produce un único CSV en `scripts/testing_20260512/sa/`.

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

Dentro de cada CSV, cada fila es una corrida. Para una instancia ejecutada con SA (44 configuraciones × 2 repeticiones), el CSV tendrá 88 filas.

---

## Programa experimental con evaluador de costo corregido (mayo–jun 2026)

Después de integrar el evaluador de **orientación greedy** como lógica nativa de `evaluador_costo.py` (commit mayo-31), se construyó un programa experimental paralelo en `scripts/` que ya no usa `experimentos.py`. Sus componentes son:

### Módulos `_common` (configuración canónica fija)

| Archivo | Approach |
|---|---|
| `scripts/_solo_p_inter_20260531_common.py` | Barrido de `p_inter` con la config canónica de cada MH |
| `scripts/_binario_capacidad_20260531_common.py` | `p_inter` fijo + penalización binaria de capacidad |
| `scripts/_pr_aislado_20260531_common.py` | `p_inter` fijo + Path Relinking como intensificador aislado |
| `scripts/_calibracion_2knob_20260601.py` | Calibración del 2.º parámetro más influyente por MH |
| `scripts/_calibracion_restantes_20260601.py` | Calibración de los tres parámetros restantes (lambda, alpha_inter, umbral PR) |
| `scripts/_canonico_puro_20260601.py` | Experimento 0: línea de base inferior con mecanismos de escape desactivados |

### Runners por MH y approach

Cada approach tiene un runner por metaheurística con el prefijo `run_<mh>_<approach>_20260531.py` (p. ej., `run_sa_solo_p_inter_20260531.py`, `run_cuckoo_pr_aislado_20260531.py`). Los scripts de shell `run_all_<approach>_20260531.sh` lanzan las cinco MH en secuencia.

### Diferencias clave respecto a `experimentos.py`

- La config canónica está definida en el módulo `_common` y versionada en `config_fija.json` / `mejor_2knob.json` en la raíz del proyecto.
- El evaluador de costo es greedy (no canónico), por lo que los gaps BKS de estas corridas son directamente comparables con los resultados finales.
- Path Relinking usa el módulo limpio `metacarp/path_relinking_limpio_20260531.py` (sin frame-hacks) en lugar de `path_relinking_20260528.py`.

---

### Calibración de parámetros restantes (`_calibracion_restantes_20260601.py`)

Sobre la base del approach `solo_p_inter` con la config canónica fija (p_inter + knob 1 + knob 2 ya calibrados), este script calibra los tres parámetros que no habían sido tocados en el programa experimental con costo corregido. La calibración se ejecuta con 23 instancias × 3 repeticiones.

**Parámetro 1 — `lambda_factor` (transversal a todas las MH)**

Multiplica `lambda_penal_capacidad_por_defecto(ctx)` por un factor escalar. El factor `1.0` reproduce la config canónica; `0.3` la aligera (penalización más suave) y `3.0` la endurece. La calibración usa SA como MH representativa.

| `lambda_factor` | Descripción |
|---|---|
| `0.3` | Penalización de capacidad más suave; mayor tolerancia a infactibilidades |
| `1.0` | Reproduce el valor canónico actual |
| `3.0` | Penalización de capacidad más estricta |

**Parámetro 2 — `alpha_inter` (solo SA, TS simple y RTS)**

Fracción de probabilidad total asignada a los operadores inter-ruta cuando la solución es **infactible**. Afecta únicamente a las MH que exponen este parámetro. El valor canónico previo era `0.8`.

| `alpha_inter` | SA | TS simple | RTS |
|---|---|---|---|
| `0.5` | | | ganador |
| `0.7` | ganador | | |
| `0.9` | | ganador | |

**Parámetro 3 — `umbral_pr` / `max_iter_sin_mejora_kick` (solo approach `pr_aislado`)**

Número de iteraciones de estancamiento sin mejora antes de disparar el hook de Path Relinking. El valor canónico actual es `30`. La calibración cubre todas las MH con el selector `p_inter`.

| `umbral_pr` | Descripción |
|---|---|
| `15` | Dispara PR rápido; mayor frecuencia de intensificación |
| `30` | Valor canónico actual |
| `60` | Dispara PR tardío; mayor libertad de exploración antes de intensificar |

> **Nota:** Los resultados del umbral PR no estaban disponibles al cierre de esta documentación.

**Resultados de calibración obtenidos**

| Parámetro | MH de referencia | Valor ganador | Gap medio |
|---|---|---|---|
| `lambda_factor` | SA | `3.0` | 2.38 % |
| `alpha_inter` | SA | `0.7` | 2.79 % |
| `alpha_inter` | TS simple | `0.9` | 5.50 % |
| `alpha_inter` | RTS | `0.5` | 4.56 % |
| `umbral_pr` | todas | pendiente | — |

**Salida en disco**

```
experimentos_costo_fixed/_calibracion_restantes/
├── mejor_lambda.json        # factor lambda ganador + gap medio
├── mejor_alpha_inter.json   # alpha_inter ganador por MH + gap medio
├── mejor_umbral_pr.json     # umbral PR ganador por MH + gap medio (cuando esté disponible)
└── _partials/               # CSVs parciales de las corridas individuales
```

**Uso**

```bash
# Calibrar los tres parámetros en secuencia (por defecto)
python scripts/_calibracion_restantes_20260601.py

# Calibrar solo uno
python scripts/_calibracion_restantes_20260601.py --objetivo lambda
python scripts/_calibracion_restantes_20260601.py --objetivo alpha
python scripts/_calibracion_restantes_20260601.py --objetivo umbral
```

**Referencia de argumentos**

| Argumento | Tipo | Default | Descripción |
|---|---|---|---|
| `--objetivo` | `str` | `todos` | Parámetro a calibrar: `lambda`, `alpha`, `umbral` o `todos` |
| `--reps` | `int` | `3` | Repeticiones por combinación instancia × valor |
| `--workers` | `int` | `cpu_count()` | Procesos paralelos (ProcessPoolExecutor) |
| `--instancias` | `str...` | las 23 canónicas | Subconjunto de instancias a usar |
| `--salida` | `str` | `experimentos_costo_fixed/_calibracion_restantes` | Directorio de salida |
| `--root` | `str` | `None` | Raíz de datos alternativa; si no se pasa, usa `CARPTHESIS_ROOT` |

---

### Experimento 0: canónico puro (`_canonico_puro_20260601.py`)

Mide la calidad de cada metaheurística en su forma más básica —la descrita en el artículo original— sin ningún mecanismo de diversificación o escape. Sirve como **línea de base inferior** del programa experimental: al compararlo con el Experimento 8 (`solo_p_inter`), se cuantifica el aporte real de cada mecanismo de escape.

**Principio de diseño.** Para que la comparación sea controlada, todo lo demás permanece igual al Exp. 8: misma config canónica calibrada (p_inter, knob 1, knob 2), mismos operadores (`OPERADORES_POPULARES`, 9 operadores), lambda por defecto (`None`). La única diferencia es el parámetro que desactiva el mecanismo de escape de cada MH.

**Configuración canónica pura por metaheurística**

| MH | Referencia | Mecanismo desactivado | Parámetro modificado |
|---|---|---|---|
| SA | Kirkpatrick, Gelatt & Vecchi (1983) | Reheat | `patience=0` |
| TS simple | Glover (1986) | Parada anticipada por estancamiento | `max_iter_sin_mejora=10_000` |
| RTS | Battiti & Tecchiolli (1994) | Reactividad del tenure | `factor_aumento=1.0`, `factor_reduccion=1.0` |
| ABC | Karaboga (2005) | Fase scout | `limite_abandono=10_000` |
| Cuckoo | Yang & Deb (2009) | Abandono de nidos | `pa_abandono=0.0` |

**Volumen de corridas:** 5 MH × 23 instancias × 5 repeticiones = **575 corridas**.

**Salida en disco**

```
experimentos_costo_fixed/canonico_puro_<YYYYMMDD-HHmm>/
├── sa_canonico_<instancia>.csv
├── tabu_simple_canonico_<instancia>.csv
├── tabu_reactiva_canonico_<instancia>.csv
├── abc_simple_canonico_<instancia>.csv
├── cuckoo_canonico_<instancia>.csv
└── _partials/    # CSVs parciales antes de consolidar
```

**Uso**

```bash
# Todas las MH, 23 instancias, 5 reps (ejecución completa)
python scripts/_canonico_puro_20260601.py

# Solo algunas MH
python scripts/_canonico_puro_20260601.py --mhs sa tabu_simple

# Prueba rápida (2 instancias, 1 rep)
python scripts/_canonico_puro_20260601.py --smoke
```

**Referencia de argumentos**

| Argumento | Tipo | Default | Descripción |
|---|---|---|---|
| `--mhs` | `str...` | las 5 MH | Subconjunto de metaheurísticas: `sa tabu_simple tabu_reactiva abc_simple cuckoo` |
| `--reps` | `int` | `5` | Repeticiones por instancia |
| `--workers` | `int` | `cpu_count()` | Procesos paralelos (ProcessPoolExecutor) |
| `--instancias` | `str...` | las 23 canónicas | Subconjunto de instancias a usar |
| `--salida-base` | `str` | `experimentos_costo_fixed` | Directorio raíz; el script crea un subdirectorio `canonico_puro_<ts>` dentro |
| `--root` | `str` | `None` | Raíz de datos alternativa; si no se pasa, usa `CARPTHESIS_ROOT` |
| `--smoke` | flag | desactivado | Modo prueba rápida: 2 instancias (`gdb19`, `kshs1`), 1 repetición |

---

## Documentación relacionada

- `docs/recocido_simulado.md` — descripción detallada de SA y sus parámetros
- `docs/busqueda_tabu.md` — descripción detallada de Búsqueda Tabú
- `docs/colonia_abejas.md` — descripción detallada de ABC
- `docs/cuckoo_search.md` — descripción detallada de Cuckoo Search
- `docs/generacion_vecinos.md` — catálogo de los 9 operadores de vecindario
