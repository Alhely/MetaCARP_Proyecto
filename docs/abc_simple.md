# ABC Simple (Karaboga 2005 canónico) para CARP

## 1. Introducción y motivación

`busqueda_abejas_simple` implementa el algoritmo **Artificial Bee Colony (ABC)** de **Karaboga (2005)** en su versión más fiel al artículo original, ajustada al problema CARP. Su propósito es servir como **baseline didáctico** y referencia de comparación frente a la versión más elaborada del proyecto (`busqueda_abejas`), que ya incorpora varios añadidos heurísticos (sesgo inter/intra-ruta a lo largo de las tres fases, rastreo separado de mejor factible, scouts dirigidos hacia la mejor solución, etc.).

La intención de tener dos implementaciones en el repositorio responde a un argumento metodológico de la tesis: aislar cuánto del rendimiento observado en `busqueda_abejas` proviene del **núcleo ABC** y cuánto de las **modificaciones heurísticas** acumuladas. `busqueda_abejas_simple` permite responder esa pregunta con un experimento controlado.

### Diferencias clave con `busqueda_abejas`

| Aspecto | `busqueda_abejas` (extendida) | `busqueda_abejas_simple` (canónica) |
|---|---|---|
| Scouts | Vecino de la mejor fuente actual | **Solución aleatoria pura** (greedy por capacidad) |
| Sesgo inter/intra | `pesos_inter_bias` en empleadas, observadoras y scouts | `seleccionar_grupo_operadores_inter_intra` solo en empleadas y observadoras |
| Mejor factible | Rastreado separado del mejor general | Un único mejor (penalización guía la búsqueda) |
| Imputación `registrar_mejora` | Comparada al final del ciclo, imputada al último movimiento aceptado | Comparada **en el mismo `if`** del vecino aceptado |
| Criterio de parada | Solo `iteraciones` (tope duro) | `iteraciones` + `max_iter_sin_mejora` (siempre activo, calibrado a `max(50, 3·n)`) |
| Backend de generación | Configurable (`labels` o `ids`) | Siempre `ids` (más rápido y consistente) |
| Parámetro `alpha_inter` | Expuesto como parámetro independiente | **Eliminado**: el algoritmo aplica `max(p_inter, 0.8)` automáticamente bajo violación |
| Parámetros numéricos | Valores absolutos fijos | **Instance-aware**: fórmulas en función de `n_tareas`; también admiten factores de escala |

---

## 2. Los tres roles de abejas

| Tipo | Cantidad por ciclo | Fuente que visita | Acción |
|---|---|---|---|
| **Empleadas** | `num_fuentes` | Su fuente asignada (siempre la misma) | Genera un vecino; reemplaza la fuente si mejora |
| **Observadoras** | `num_fuentes` | Elegidas por ruleta sobre `1/(1+obj)` | Genera un vecino; reemplaza la fuente si mejora |
| **Scouts** | Variable | Fuente con `trials[i] >= limite_abandono` | Reemplaza la fuente con una **solución aleatoria pura** |

La diferencia conceptual fundamental: las empleadas explotan localmente, las observadoras **concentran esfuerzo en las mejores** (ruleta), los scouts **diversifican** abandonando regiones agotadas con puntos completamente nuevos del espacio.

---

## 3. Modificaciones sobre Karaboga (2005)

| Categoría | Decisión | Justificación |
|---|---|---|
| **Se MANTIENE** | Tres fases empleadas → observadoras → scouts | Estructura nuclear del ABC |
| **Se MANTIENE** | Ruleta por fitness inverso `1/(1+obj)` en observadoras | Fórmula original Karaboga |
| **Se MANTIENE** | Scouts puramente aleatorios (greedy por capacidad) | El sello distintivo del ABC canónico; generan un punto sin sesgo histórico. La aleatoriedad es una **permutación uniforme** de las tareas (`rng.shuffle`, Fisher–Yates) seguida de asignación greedy por capacidad |
| **Se MANTIENE** | Comparación greedy en empleadas y observadoras | Misma regla de actualización del paper |
| **Se SIMPLIFICA** | Una sola lista `mejor_sol_ids` (no se rastrea mejor factible aparte) | Penalización ya guía la búsqueda hacia región factible; segundo rastreo añadía complejidad sin beneficio en instancias estudiadas |
| **Se SIMPLIFICA** | Backend de generación fijo en IDs | Evita la rama dual `labels`/`ids` que complica el código sin acelerar nada relevante |
| **Se AÑADE** | Sesgo `seleccionar_grupo_operadores_inter_intra` en empleadas y observadoras | Sin él, ABC era incapaz de reparar capacidad en instancias con violación inicial. Es el mismo helper que usan SA / TS simple / RTS, lo cual asegura **comparabilidad metodológica** |
| **Se AÑADE** | `max_iter_sin_mejora` siempre activo (calibrado a `max(50, 3·n)` por defecto) | Permite parar pronto sin ejecutar las `iteraciones` completas; ahorra tiempo en plateaus largos |
| **Se CORRIGE** | `registrar_mejora` se invoca **dentro del mismo `if`** que detecta vecino mejor que el global | En `busqueda_abejas`, la comparación se hace al final del ciclo contra `ultimo_movimiento_aceptado`, que puede no ser el operador real responsable. Aquí `sum(operadores_mejoraron.values()) ≤ mejoras` siempre, con la diferencia exacta atribuida a scouts (que no tienen operador asociado) |
| **Se ELIMINA** | Parámetro `alpha_inter` | Sustituido por el piso automático `max(p_inter, 0.8)` bajo violación. El usuario configura un único punto (`p_inter`); el umbral 0.8 bajo violación es garantía interna |

---

## 4. Firma de `busqueda_abejas_simple`

```python
busqueda_abejas_simple(
    inicial_obj,                      # objeto inicial (lista de soluciones candidatas o pickle)
    data,                             # diccionario de datos de la instancia CARP
    G,                                # grafo NetworkX
    *,
    # --- Parámetros instance-aware (None → fórmula en función de n_tareas) ---
    iteraciones: int | None = None,          # None → max(200, 20·n)
    num_fuentes: int | None = None,          # None → max(10, round(2·√n))
    limite_abandono: int | None = None,      # None → max(15, n // 2)
    max_iter_sin_mejora: int | None = None,  # None → max(50, 3·n)
    # --- Factores de escala (sobrescriben la fórmula default; ignorados si se pasa valor absoluto) ---
    factor_fuentes: float | None = None,     # num_fuentes   = max(10, round(f·√n))
    factor_abandono: float | None = None,    # limite_abandono = max(15, round(f·n))
    factor_iter: int | float | None = None,  # iteraciones   = max(200, round(f·n))
    # --- Parámetros de sesgo y aleatoridad ---
    p_inter: float = 0.6,                    # P(inter) cuando la solución es factible
    semilla: int | None = None,
    operadores: Iterable[str] = OPERADORES_POPULARES,
    marcador_depot_etiqueta: str | None = None,
    usar_gpu: bool = False,
    guardar_historial: bool = True,
    guardar_csv: bool = False,
    ruta_csv: str | None = None,
    nombre_instancia: str = "instancia",
    repeticion: int | None = None,
    root: str | None = None,
    usar_penalizacion_capacidad: bool = True,
    lambda_capacidad: float | None = None,
    extra_csv: dict[str, object] | None = None,
    max_iter_sin_mejora_kick: int | None = None,
    intensificador: Callable | None = None,
    **_ignorado_kwargs,
) -> AbejasSimpleResult
```

> **Nota:** `alpha_inter` fue eliminado de la firma. El algoritmo aplica automáticamente `max(p_inter, 0.8)` como probabilidad inter-ruta cuando la solución actual viola capacidad. El parámetro `p_inter` base controla el régimen factible; el piso 0.8 bajo violación es una garantía interna.

### Tabla de parámetros

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `iteraciones` | `int \| None` | `None` → `max(200, 20·n)` | Número de ciclos completos (empleadas + observadoras + scouts) |
| `num_fuentes` | `int \| None` | `None` → `max(10, round(2·√n))` | Fuentes de alimento (soluciones activas) en paralelo |
| `limite_abandono` | `int \| None` | `None` → `max(15, n // 2)` | Intentos fallidos consecutivos antes de mandar fuente al scout. En el programa experimental (`_calibracion_2knob_20260601.py`) este parámetro se calibra como segundo parámetro más influyente; el default instance-aware es un buen punto de partida pero puede ser un multiplicador distinto de `n // 2` según la instancia. |
| `max_iter_sin_mejora` | `int \| None` | `None` → `max(50, 3·n)` | Criterio de parada anticipada por estancamiento; siempre activo |
| `factor_fuentes` | `float \| None` | `None` | Si se pasa, `num_fuentes = max(10, round(f·√n))`. Solo actúa si `num_fuentes` es `None` |
| `factor_abandono` | `float \| None` | `None` | Si se pasa, `limite_abandono = max(15, round(f·n))`. Solo actúa si `limite_abandono` es `None` |
| `factor_iter` | `int \| float \| None` | `None` | Si se pasa, `iteraciones = max(200, round(f·n))`. Solo actúa si `iteraciones` es `None` |
| `p_inter` | `float` [0,1] | `0.6` | P(elegir inter-ruta) cuando la solución es factible. Bajo violación se aplica `max(p_inter, 0.8)` |
| `semilla` | `int \| None` | `None` | `None` = aleatoria del sistema (corridas no reproducibles; recomendado en experimentos paralelos) |
| `operadores` | `Iterable[str]` | `OPERADORES_POPULARES` | Subset de los 9 operadores activos |
| `usar_gpu` | `bool` | `False` | Si `True`, evaluación en lote (fase observadoras) intenta usar CuPy |
| `usar_penalizacion_capacidad` | `bool` | `True` | Si `True`, objetivo = costo + λ × violación |
| `lambda_capacidad` | `float \| None` | `None` | λ explícito. `None` = automático (~10× mediana deadhead) |
| `max_iter_sin_mejora_kick` | `int \| None` | `None` | Umbral de ciclos consecutivos sin mejora para disparar el hook. Si `None`, el hook nunca se activa (no altera el comportamiento). Solo tiene efecto cuando `intensificador` también se provee. |
| `intensificador` | `Callable \| None` | `None` | Hook opcional de intensificación. Cuando se dispara, se llama con `intensificador(sol_ids, mejor_ids, ctx, lam, rng, encoding, md)` sobre la fuente de menor objetivo. ABC usa backend IDs, por lo que el hook compatible es `hook_pr_ids` de `path_relinking_limpio_20260531`. Con `None` (default) el comportamiento es idéntico al anterior. |

### Precedencia de parámetros instance-aware

```
valor absoluto (e.g. iteraciones=500)
    ↓ si None
factor de escala (e.g. factor_iter=20)
    ↓ si None
fórmula default (e.g. max(200, 20·n))
```

---

## 5. Comportamiento del `p_inter` dinámico

El algoritmo mide la **violación media** de las fuentes activas al inicio de cada ciclo y antes de la fase observadoras. Si esa violación media es positiva, aplica automáticamente:

```
p_efectivo = max(p_inter, 0.8)
```

Si la violación media es cero (todas las fuentes son factibles), se respeta `p_inter` tal como lo pasó el usuario. Este comportamiento se registra en el campo `p_inter_max_efectivo` del resultado y en la columna homónima del CSV.

El parámetro `alpha_inter` que tenía `busqueda_abejas` fue eliminado porque exponía un segundo punto de la curva que el usuario no necesita controlar: la decisión relevante es el sesgo en régimen factible (`p_inter`); el piso 0.8 bajo violación es una elección de diseño del algoritmo, no un hiperparámetro.

---

## 6. `AbejasSimpleResult` — campos del resultado

| Campo | Tipo | Significado |
|---|---|---|
| `mejor_solucion` | `list[list[str]]` | Mejor solución encontrada en formato de etiquetas con depósito |
| `mejor_costo` | `float` | Costo PURO (sin penalización) de la mejor solución |
| `solucion_inicial_referencia` | `list[list[str]]` | Solución inicial elegida como referencia |
| `costo_solucion_inicial` | `float` | Costo puro de la solución inicial |
| `mejora_absoluta` | `float` | `costo_inicial − costo_mejor` (positivo ⇒ mejora) |
| `mejora_porcentaje_inicial_vs_final` | `float` | Mejora porcentual respecto al inicial |
| `tiempo_segundos` | `float` | Tiempo total de ejecución |
| `iteraciones_totales` | `int` | Ciclos completos ejecutados (≤ `iteraciones_efectivas`) |
| `iteraciones_sin_mejora_final` | `int` | Contador de estancamiento al cierre |
| `fuentes_alimento` | `int` | `num_fuentes_efectivo` (replicado por trazabilidad) |
| `scouts_reinicios` | `int` | Veces que una fuente fue reemplazada por solución aleatoria |
| `mejoras` | `int` | Veces que el mejor global mejoró durante la corrida |
| `semilla` | `int \| None` | Semilla efectiva del RNG |
| `backend_evaluacion` | `str` | `"cpu"` o `"gpu"` |
| `usar_penalizacion_capacidad` | `bool` | Si la penalización estuvo activa |
| `lambda_capacidad` | `float` | λ efectivo aplicado |
| `mejor_solucion_factible_final` | `bool` | `True` si la mejor solución respeta la restricción de capacidad |
| `aceptaciones_solucion_infactible` | `int` | Cuántas aceptaciones aterrizaron en vecino infactible |
| `iteraciones_con_violacion` | `int` | Ciclos con violación media de fuentes > 0 |
| `fraccion_iter_con_violacion` | `float` | `iteraciones_con_violacion / iteraciones_totales` |
| `operadores_propuestos` | `dict[str,int]` | Conteo por operador (propuesto) |
| `operadores_aceptados` | `dict[str,int]` | Conteo por operador (aceptado) |
| `operadores_mejoraron` | `dict[str,int]` | Conteo por operador (mejoraron el mejor global) |
| `operadores_trayectoria_mejor` | `dict[str,int]` | Snapshot de `aceptados` en el momento de la última mejora |
| `historial_mejor_costo` | `list[float]` | Trayectoria del mejor costo por ciclo (si `guardar_historial=True`) |
| `archivo_csv` | `str \| None` | Ruta del CSV generado, o `None` |
| `n_tareas` | `int` | Número de tareas requeridas (`len(ctx.u_arr)`), variable de escala `n` |
| `iteraciones_efectivas` | `int` | Valor de `iteraciones` realmente usado en el bucle |
| `num_fuentes_efectivo` | `int` | Valor de `num_fuentes` realmente usado |
| `limite_abandono_efectivo` | `int` | Valor de `limite_abandono` realmente usado |
| `max_iter_sin_mejora_efectivo` | `int \| None` | Valor de `max_iter_sin_mejora` realmente usado |
| `p_inter_max_efectivo` | `float` | Valor de P(inter) aplicado bajo violación (`max(p_inter, 0.8)`) |

---

## 7. Columnas del CSV de salida

El CSV lo escribe `guardar_resultado_csv` cuando `guardar_csv=True`. La etiqueta de metaheurística es `"busqueda_abejas_simple"`. El CSV no incluye las columnas `id_corrida` ni `config_id` (convención del proyecto).

### Columnas de identificación y parámetros (19 columnas)

| Columna | Descripción |
|---|---|
| `metaheuristica` | Siempre `"busqueda_abejas_simple"` |
| `instancia` | Nombre de la instancia |
| `bks_referencia` | Valor BKS de la literatura |
| `bks_origen` | Fuente del BKS |
| `gap_bks_porcentaje` | `(mejor_costo − BKS) / BKS × 100` |
| `repeticion` | Número de repetición dentro del experimento |
| `semilla` | Semilla del RNG |
| `tiempo_segundos` | Duración real de la corrida |
| `mejor_costo` | Costo PURO de la mejor solución |
| `costo_solucion_inicial` | Costo de la solución inicial de referencia |
| `mejora_absoluta` | `costo_inicial − mejor_costo` |
| `mejora_porcentaje` | Mejora porcentual |
| `iteraciones` | Valor absoluto de `iteraciones` pasado por el usuario (`""` si se dejó en `None`) |
| `num_fuentes` | Valor absoluto de `num_fuentes` pasado por el usuario (`""` si `None`) |
| `limite_abandono` | Valor absoluto de `limite_abandono` pasado por el usuario (`""` si `None`) |
| `max_iter_sin_mejora` | Valor absoluto de `max_iter_sin_mejora` pasado por el usuario (`""` si `None`) |
| `factor_fuentes` | Factor de escala pasado por el usuario (`""` si no se usó) |
| `factor_abandono` | Factor de escala pasado por el usuario (`""` si no se usó) |
| `factor_iter` | Factor de escala pasado por el usuario (`""` si no se usó) |

### Columnas de p_inter y penalización (4 columnas)

| Columna | Descripción |
|---|---|
| `p_inter` | Valor BASE de P(inter) pasado por el usuario |
| `p_inter_max_efectivo` | P(inter) aplicada bajo violación (`max(p_inter, 0.8)`) |
| `usar_penalizacion_capacidad` | Si la penalización estuvo activa |
| `lambda_capacidad` | λ efectivo de penalización |

### Columnas de valores efectivos (5 columnas)

| Columna | Descripción |
|---|---|
| `iteraciones_efectivas` | Iteraciones realmente usadas en el bucle |
| `num_fuentes_efectivo` | Fuentes realmente usadas |
| `limite_abandono_efectivo` | Límite de abandono realmente usado |
| `max_iter_sin_mejora_efectivo` | Criterio de parada realmente usado |
| `n_tareas` | Número de tareas requeridas (`n`) |

### Columnas de operadores (36 columnas)

Formato `<categoria>_<operador>`. Categorías: `propuesto`, `aceptado`, `mejoraron`, `trayectoria_mejor`. Operadores: los 9 de `OPERADORES_POPULARES`. Se generan con `contador.resumen_csv()`.

### Columnas de estadísticas de corrida (9 columnas)

| Columna | Descripción |
|---|---|
| `iteraciones_totales` | Ciclos completos ejecutados |
| `iteraciones_sin_mejora_final` | Contador de estancamiento al cierre |
| `scouts_reinicios` | Fuentes reiniciadas con solución aleatoria |
| `mejoras` | Actualizaciones del mejor global |
| `aceptaciones_solucion_infactible` | Aceptaciones que aterrizaron en vecino infactible |
| `iteraciones_con_violacion` | Ciclos con violación media > 0 |
| `fraccion_iter_con_violacion` | `iteraciones_con_violacion / iteraciones_totales` |
| `mejor_solucion_factible_final` | Si la mejor solución es factible |
| `mejor_solucion_tr_legible` | Representación textual de la solución |
| `reporte_detalle_deadheading` | Desglose de costos de arrastre por ruta |
| `costo_total_desde_reporte` | Verificación cruzada del costo |

> **Total aproximado:** ~68 columnas (19 identificación + 4 p_inter + 5 efectivos + 36 operadores + 11 estadísticas).

---

## 8. Ejemplo de uso básico

```python
from metacarp import busqueda_abejas_simple_desde_instancia

resultado = busqueda_abejas_simple_desde_instancia(
    "gdb1",
    p_inter=0.6,
    # Parámetros instance-aware: se calculan automáticamente a partir de n_tareas.
    # Para gdb1 (n≈22): iteraciones≈440, num_fuentes≈10, limite_abandono≈15, max_iter_sin_mejora≈66
)

print(f"Mejor costo: {resultado.mejor_costo:.2f}")
print(f"Mejora vs inicial: {resultado.mejora_porcentaje_inicial_vs_final:.2f} %")
print(f"Iteraciones reales: {resultado.iteraciones_totales}")
print(f"Scouts disparados: {resultado.scouts_reinicios}")
print(f"Factible final: {resultado.mejor_solucion_factible_final}")
print(f"n_tareas: {resultado.n_tareas}")
print(f"num_fuentes efectivo: {resultado.num_fuentes_efectivo}")
```

Con factores de escala explícitos (modo experimento):

```python
resultado = busqueda_abejas_simple_desde_instancia(
    "gdb1",
    factor_fuentes=2.0,   # num_fuentes = max(10, round(2·√n))
    factor_abandono=0.5,  # limite_abandono = max(15, round(0.5·n))
    factor_iter=20,        # iteraciones = max(200, 20·n)
    p_inter=0.6,
    guardar_csv=True,
    ruta_csv="resultados/abc_simple_gdb1.csv",
    nombre_instancia="gdb1",
    repeticion=1,
)
```

Con valores absolutos (reproducibilidad exacta entre instancias):

```python
resultado = busqueda_abejas_simple_desde_instancia(
    "gdb1",
    iteraciones=500,
    num_fuentes=20,
    limite_abandono=30,
    max_iter_sin_mejora=100,
    p_inter=0.6,
)
```

---

## 9. Guía del grid search: `run_abc_simple_automatico.py`

El script `scripts/run_abc_simple_automatico.py` barre el espacio de hiperparámetros en 4 dimensiones mediante factores de escala en función de `n_tareas`:

| Dimensión | Valores barridos | Fórmula aplicada |
|---|---|---|
| `factor_fuentes` | `{1.5, 2.0, 3.0, 4.0}` | `num_fuentes = max(10, round(f·√n))` |
| `factor_abandono` | `{0.25, 0.5, 0.75, 1.0}` | `limite_abandono = max(15, round(f·n))` |
| `p_inter` | `{0.4, 0.5, 0.6, 0.7, 0.8}` | P(inter) base en régimen factible |
| `factor_iter` | `{15, 20, 30}` | `iteraciones = max(200, round(f·n))` |

**Total del grid:** 4 × 4 × 5 × 3 = 240 combos × 23 instancias × 5 repeticiones = **27,600 corridas**.

`max_iter_sin_mejora` no se barre: se deja en `None` para que la función lo calibre automáticamente a `max(50, 3·n)` por instancia.

Las semillas son aleatorias del sistema (no deterministas) para que las 5 repeticiones por configuración sean estadísticamente independientes.

```bash
# Corrida completa del grid (puede tardar horas; recomendado con --workers)
python scripts/run_abc_simple_automatico.py

# Con paralelismo (N procesos simultáneos)
python scripts/run_abc_simple_automatico.py --workers 8

# Fijar p_inter y barrer solo las otras 3 dimensiones (960 corridas menos)
python scripts/run_abc_simple_automatico.py --p-inter 0.6

# Directorio de salida personalizado
python scripts/run_abc_simple_automatico.py --salida-dir experimentos/abc_simple_grid
```

---

## 10. Por qué la GPU SOLO acelera la evaluación

La generación de vecinos opera sobre listas de Python con operadores combinatorios (`relocate_inter`, `swap_intra`, `2opt_star`, etc.). Estas operaciones son **secuenciales y data-dependent**: cada operador decide qué hacer en función de la longitud de las rutas y de elecciones aleatorias. No hay matrices ni reducciones vectoriales que paralelizar.

La **evaluación**, en cambio, sí es trivialmente paralelizable: cada vecino se reduce a una suma de distancias precomputadas (`dist[u, v] + costo_servicio`). La función `costo_lote_penalizado_ids` empaqueta los `num_fuentes` vecinos de la **fase observadoras** en arrays planos y delega la reducción a NumPy (CPU) o CuPy (GPU). Con GPU la aceleración es notable en instancias grandes y con `num_fuentes` alto; en instancias pequeñas (n < 50) el overhead de copiar arrays a la GPU puede dominar y conviene dejar `usar_gpu=False`.

En las **empleadas** los vecinos se evalúan individualmente con `costo_rapido_ids` (escalar) porque la decisión de aceptar/rechazar se toma por fuente y no se beneficiaría de vectorizar (cada decisión depende del resultado anterior).
