# Comparativa: TS simple vs Reactive Tabu Search

Este documento compara lado a lado las dos implementaciones de Búsqueda Tabú del proyecto: la versión simple (`busqueda_tabu_simple.py`) y la versión reactiva (`busqueda_tabu_reactiva.py`). Ambas heredan el mismo esqueleto de búsqueda y comparten el helper de sesgo inter/intra-ruta, pero difieren en cómo gestionan la memoria de corto plazo y cómo responden al estancamiento.

---

## Comparativa de mecanismos

| Mecanismo | TS simple | Reactive Tabu Search (RTS) |
|---|---|---|
| **Lista tabú** | FIFO de longitud **fija** (`tabu_tenure`) | FIFO de longitud **dinámica** en `[tabu_tenure_min, tabu_tenure_max]` |
| **Estructura de datos** | `deque(maxlen=tenure)` + `set` | `deque(maxlen=tenure_actual)` + `set`; se reconstruye cuando cambia el tenure |
| **Lookup tabú** | O(1) via `set_tabu` | O(1) via `set_tabu` |
| **Detección de ciclos** | Implícita (lista tabú evita revisitar movimientos recientes) | Explícita: hash canónico de la solución en `historial: dict` |
| **Respuesta a ciclo** | Ninguna (solo la tenencia lo limita) | Aumentar tenure × `factor_aumento` (acotado por `tabu_tenure_max`) |
| **Reducción de tenure** | No | Cuando pasan `iter_sin_repeticion_para_reducir` iters sin repetición → × `factor_reduccion` |
| **Mecanismo de escape** | No | Sí: `num_movimientos_escape` movimientos aleatorios + limpieza de lista tabú + limpieza de historial |
| **Disparador del escape** | — | Solución repetida ≥ `umbral_repeticiones_escape` veces |
| **Criterio de aspiración** | Sí (igual en ambos) | Sí (igual en ambos) |
| **Selección** | Best-improvement sobre lote de `tam_vecindario` vecinos | Best-improvement sobre lote de `tam_vecindario` vecinos |
| **Sesgo inter/intra** | Sí, via `seleccionar_grupo_operadores_inter_intra` | Sí, mismo helper; el escape **no** aplica el sesgo |
| **Criterios de parada** | `iteraciones_max` + `max_iter_sin_mejora` | `iteraciones_max` + `max_iter_sin_mejora` (iguales) |
| **Parámetros instance-aware** | No (defaults fijos en la firma) | Sí: todos los params reactivos dependen de `n = nº tareas` |

---

## Comparativa de parámetros

| Parámetro | TS simple | RTS | Fórmula instance-aware (RTS) |
|---|---|---|---|
| `iteraciones_max` | `400` (default fijo) | `None` → calculado | `max(50, 20·n)` |
| `max_iter_sin_mejora` | `100` (default fijo) | `None` → calculado | `max(20, 5·n)` |
| `tam_vecindario` | `25` (default fijo) | `None` → calculado | `max(20, 2·n)` |
| `tabu_tenure` | `20` (fijo durante la corrida) | — | — |
| `tabu_tenure_inicial` | — | `None` → calculado | `max(3, round(√n))` |
| `tabu_tenure_min` | — | `None` → `3` | `3` |
| `tabu_tenure_max` | — | `None` → calculado | `max(15, round(3·√n))` |
| `factor_aumento` | — | `1.2` | — |
| `factor_reduccion` | — | `0.9` | — |
| `iter_sin_repeticion_para_reducir` | — | `None` → calculado | `max(5, round(2·√n))` |
| `umbral_repeticiones_escape` | — | `3` | — |
| `num_movimientos_escape` | — | `None` → calculado | `max(3, n // 10)` |
| `alpha_inter` | `None` → `0.8` | `None` → `0.8` | Compartido |
| `p_inter` | `None` → `0.6` | `None` → `0.6` | Compartido |

---

## Comparativa de columnas CSV

### Columnas comunes a ambos algoritmos (64 columnas)

| Grupo | Columnas |
|---|---|
| Identificación | `metaheuristica`, `instancia`, `bks_referencia`, `bks_origen`, `gap_bks_porcentaje`, `repeticion`, `semilla`, `tiempo_segundos` |
| Resultados | `mejor_costo`, `costo_solucion_inicial`, `mejora_absoluta`, `mejora_porcentaje` |
| Parámetros TS | `iteraciones_max`, `max_iter_sin_mejora`, `tam_vecindario`, `tabu_tenure` |
| Operadores (36) | `propuesto_<op>`, `aceptado_<op>`, `mejoraron_<op>`, `trayectoria_mejor_<op>` para los 9 operadores |
| Estadísticas | `iteraciones_totales`, `vecinos_evaluados`, `iteracion_mejor`, `iteraciones_sin_mejora_final`, `aspiraciones`, `iteraciones_todos_tabu`, `aceptadas`, `mejoras`, `mejor_solucion_factible_final`, `mejor_solucion_tr_legible`, `reporte_detalle_deadheading`, `costo_total_desde_reporte` |
| Sesgo inter/intra | `alpha_inter`, `p_inter`, `iteraciones_con_violacion`, `fraccion_iter_con_violacion` |

> **Nota sobre `tabu_tenure`:** en el TS simple es el valor fijo de la tenencia. En el CSV de RTS es el **valor inicial** del tenure, para mantener compatibilidad con los lectores del CSV del TS simple.

### Columnas exclusivas de RTS (16 columnas adicionales)

| Columna | Descripción |
|---|---|
| `tenure_inicial` | Tenure inicial efectivo (igual que `tabu_tenure` en este CSV) |
| `tenure_min_aplicado` | Cota inferior del tenure |
| `tenure_max_aplicado` | Cota superior del tenure |
| `tenure_final` | Tenure al terminar la corrida |
| `tenure_promedio` | Promedio del tenure a lo largo de la corrida |
| `tenure_max_alcanzado` | Máximo tenure observado |
| `tenure_min_alcanzado` | Mínimo tenure observado |
| `factor_aumento` | Factor multiplicativo al detectar ciclo |
| `factor_reduccion` | Factor multiplicativo al reducir tenure |
| `iter_sin_repeticion_para_reducir` | Paciencia para reducir el tenure |
| `umbral_repeticiones_escape` | Umbral de repeticiones para disparar escape |
| `num_movimientos_escape` | Movimientos aleatorios por escape |
| `num_repeticiones_detectadas` | Total de repeticiones detectadas |
| `num_escapes_realizados` | Total de escapes realizados |
| `num_aumentos_tenure` | Total de incrementos del tenure |
| `num_reducciones_tenure` | Total de reducciones del tenure |

---

## Comparativa de scripts de experimentos

| Aspecto | TS simple | RTS |
|---|---|---|
| Script | `scripts/run_tabu_simple_automatico.py` | `scripts/run_tabu_reactiva_automatico.py` |
| Parámetro en el grid | `tabu_tenure` ∈ {5, 10, 15, 20, 25, 30} | `factor_aumento` ∈ {1.05, 1.1, 1.2, 1.3, 1.4} × `umbral_escape` ∈ {2, 3, 5, 8} × `p_inter` ∈ {0.4, 0.5, 0.6, 0.7, 0.8} × `factor_reduccion` ∈ {0.85, 0.9, 0.95} |
| Parámetros fijos | `iteraciones_max=400`, `max_iter_sin_mejora=100`, `tam_vecindario=25` | instance-aware (todos `None`) |
| Corridas totales | 6 × 23 × 2 = **276** | 5 × 4 × 5 × 3 × 23 × 5 = **34,500** |
| Semillas | Configurables con `--semilla-base` | Aleatorias del sistema (corridas independientes) |
| Carpeta de salida | `<salida-dir>/` (directa) | `<salida-dir>/` (directa) |
| Etiqueta en CSV | `"busqueda_tabu_simple"` | `"tabu_reactiva"` |

---

## Comparativa de estructuras de datos internas

```
TS simple                              RTS
─────────────────────────────────────  ──────────────────────────────────────────
deque(maxlen=tenure_fijo)              deque(maxlen=tenure_actual)
  + set_tabu                             + set_tabu
                                       historial: dict[hash_sol → {veces, ultima}]
                                       iter_sin_repeticion: int
```

La complejidad de memoria del RTS es potencialmente mayor que la del TS simple: el historial puede crecer indefinidamente si no se producen escapes. En la práctica, cada escape borra el historial completo, por lo que el tamaño real depende de la frecuencia de escapes.

---

## Cuándo usar cada variante

| Situación | Recomendación |
|---|---|
| Primera exploración de una instancia nueva | TS simple: menos parámetros, comportamiento más predecible. |
| Instancias donde el TS simple se estanca consistentemente | RTS: la tenencia dinámica y el escape pueden romper el estancamiento. |
| Análisis de sensibilidad a `tabu_tenure` | TS simple: barrido sobre `tabu_tenure` con parámetros fijos. |
| Análisis del impacto de los mecanismos reactivos | RTS: barrido sobre `factor_aumento` y `umbral_repeticiones_escape`. |
| Comparación justa entre ambos algoritmos | Usar los mismos defaults de `alpha_inter` y `p_inter` (ambos `0.8` / `0.6`) y la misma semilla. |

---

## Interpretación de los campos reactivos del CSV de RTS

### `num_repeticiones_detectadas` vs `num_escapes_realizados`

- **Muchas repeticiones, pocos escapes:** el tenure dinámico está funcionando bien; las repeticiones disparan el aumento de tenure y eso es suficiente para salir del ciclo sin necesidad de escape.
- **Muchas repeticiones, muchos escapes:** la instancia tiene cuencas de atracción muy fuertes; el aumento de tenure no basta y el algoritmo necesita la perturbación fuerte del escape.
- **Pocas repeticiones:** el algoritmo está explorando libremente; el tenure probablemente se fue reduciendo durante la corrida.

### Rango efectivo del tenure (`tenure_min_alcanzado`, `tenure_max_alcanzado`)

Un rango amplio indica que el mecanismo reactivo se activó frecuentemente en ambas direcciones. Un rango estrecho puede indicar que: (a) la instancia es fácil y hay pocas repeticiones (rango bajo), o (b) hay ciclos continuos y el tenure sube hasta el máximo y se queda allí (rango alto).

### `tenure_promedio`

Compara con `tabu_tenure_inicial`: si el promedio es sustancialmente mayor que el inicial, el algoritmo pasó más tiempo en modo "alta presión" (muchos ciclos detectados). Si es menor que el inicial, el algoritmo exploró con libertad la mayor parte del tiempo.

---

---

## Comparativa con ABC Simple

La tabla siguiente ubica las dos Búsquedas Tabú en el contexto del tercer algoritmo del proyecto, `busqueda_abejas_simple`.

| Aspecto | TS simple | RTS | ABC simple |
|---|---|---|---|
| **Paradigma** | Trayectoria única | Trayectoria única + escape | Población (N fuentes en paralelo) |
| **Memoria** | Lista tabú FIFO | Lista tabú dinámica + historial | Contadores `trials[i]` por fuente |
| **Diversificación** | Via prohibiciones tabú | Via escape + tenure dinámico | Via scouts aleatorias puras (Karaboga 2005) |
| **Selección** | Best-improvement en lote | Best-improvement en lote | Greedy por fuente (empleadas) + ruleta (observadoras) |
| **Evaluación en lote** | Siempre (todo el vecindario) | Siempre (todo el vecindario) | Solo en observadoras (N vecinos simultáneos) |
| **GPU** | No aplica (lotes ya vectorizados en CPU) | No aplica | Sí: `costo_lote_penalizado_ids` en observadoras |
| **Sesgo inter/intra** | `seleccionar_grupo_operadores_inter_intra` | Mismo helper; escape sin sesgo | Mismo helper en empleadas y observadoras; scouts sin sesgo |
| **p_inter dinámico** | `alpha_inter`/`p_inter` independientes | `alpha_inter`/`p_inter` independientes | Un solo `p_inter`; piso automático `max(p_inter, 0.8)` bajo violación |
| **Parámetros instance-aware** | Parcial (regla `√n` manual) | Total (todos los params reactivos) | Total (fórmulas para N, abandono, iteraciones, parada) |
| **Criterio de parada** | `iteraciones_max` + `max_iter_sin_mejora` | Ídem | `iteraciones_eff` + `max_iter_sin_mejora` (siempre activo) |
| **Bug de imputación de mejora** | No aplica (una mejora por iter) | No aplica | Corregido: `registrar_mejora` dentro del mismo `if` de detección |
| **Script de experimentos** | `run_tabu_simple_automatico.py` | `run_tabu_reactiva_automatico.py` | `run_abc_simple_automatico.py` |
| **Dimensiones del grid** | 1D (tenure) | 4D (factor_aum × umbral_esc × p_inter × factor_red) | 4D (factor_fuentes × factor_abandono × p_inter × factor_iter) |
| **Total corridas grid** | 276 | 34,500 | 27,600 |

---

## Documentación relacionada

- `docs/busqueda_tabu_simple.md` — descripción completa del TS simple
- `docs/busqueda_tabu_reactiva.md` — descripción completa del RTS
- `docs/abc_simple.md` — descripción completa del ABC simple (Karaboga 2005 para CARP)
- `docs/colonia_abejas.md` — versión extendida del ABC del proyecto
- `docs/generacion_vecinos.md` — catálogo de los 9 operadores de vecindario
