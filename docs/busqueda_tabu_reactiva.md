# Reactive Tabu Search (RTS) para CARP

La Búsqueda Tabú Reactiva (*Reactive Tabu Search*, RTS) implementada en `metacarp.busqueda_tabu_reactiva` extiende la Búsqueda Tabú clásica con tres mecanismos de adaptación dinámica introducidos por Battiti & Tecchiolli (1994): tenencia tabú auto-ajustable, memoria de soluciones visitadas para detectar ciclos, y un mecanismo de escape que perturba fuertemente la búsqueda cuando se queda atrapada.

---

## Introducción conceptual

### Limitación del TS clásico

El TS simple usa una tenencia tabú fija durante toda la corrida. El problema es que la "presión tabú" óptima varía según la fase de la búsqueda:

- En una zona con muchos mínimos locales cercanos, una tenencia corta provoca ciclos: el algoritmo rebota entre las mismas pocas soluciones.
- En una zona de exploración libre, una tenencia larga prohibe movimientos útiles innecesariamente, ralentizando el progreso.

El RTS detecta estos estados automáticamente y ajusta la tenencia en consecuencia, sin intervención del usuario.

### Los tres pilares del RTS

#### 1. Tenencia tabú dinámica

La tenencia ya no es un parámetro fijo: varía durante la ejecución dentro del rango `[tabu_tenure_min, tabu_tenure_max]`:

- Cuando se detecta que la solución actual **ya fue visitada antes** (señal de ciclo), la tenencia **crece** multiplicándose por `factor_aumento` (> 1). Más prohibiciones = mayor presión para explorar regiones nuevas.
- Cuando han pasado `iter_sin_repeticion_para_reducir` iteraciones **sin detectar repeticiones**, la tenencia **decrece** multiplicándose por `factor_reduccion` (< 1). La búsqueda está explorando libremente; no necesita tanta presión tabú.

La deque que implementa la lista tabú tiene `maxlen` de solo lectura en Python, por lo que cada cambio de tenencia reconstruye la deque preservando los elementos más recientes (`_ajustar_maxlen_deque`).

#### 2. Memoria de soluciones visitadas

Para detectar repeticiones en O(1) se calcula un **hash canónico** de cada solución visitada y se almacena en un diccionario `historial`. El hash ordena globalmente las rutas antes de hacer `hash()` (los vehículos son intercambiables: una solución con las rutas en distinto orden de lista tiene el mismo costo), pero preserva el orden interno de cada ruta (no se canonicalizan reversas).

```python
def _hash_solucion(sol: list[list[str]]) -> int:
    rutas_canon = tuple(sorted(tuple(r) for r in sol))
    return hash(rutas_canon)
```

> **Decisión conservadora:** solo se identifican como "la misma solución" estructuras que sean indiscutiblemente equivalentes (mismo conjunto de rutas en cualquier orden). No se canonicalizan reversas dentro de cada ruta porque introduciría falsos positivos en variantes dirigidas del CARP.

#### 3. Mecanismo de escape

Si una misma solución se repite `umbral_repeticiones_escape` veces o más (ciclo "duro" que el aumento de tenencia no logra romper), se dispara un escape:

1. Se aplican `num_movimientos_escape` movimientos aleatorios consecutivos sobre la solución actual, ignorando la lista tabú (el objetivo es saltar lejos en el espacio).
2. Se **limpia completamente** la lista tabú y el historial de soluciones.
3. Se reinicia `iter_sin_repeticion = 0` e `iter_sin_mejora = 0` y se registra la nueva solución en el historial.

El reset incondicional de `iter_sin_mejora` (independientemente de si el escape mejora el mejor global) es una decisión deliberada: el escape es un evento de diversificación, no de explotación. Contar el escape como "iteración sin mejora" sería injusto con el algoritmo: la región a la que llega el escape es nueva y puede tardar varios ciclos normales en mostrar progreso. Resetear el contador le da ese tiempo.

La limpieza total tras el escape está justificada: después de varios movimientos aleatorios estamos en una región potencialmente lejana. Las prohibiciones y el historial de la zona anterior no son útiles allí; resetear desde cero permite detectar ciclos en la nueva región desde el principio.

> **Nota:** durante el escape no se aplica el sesgo `alpha_inter` / `p_inter`. Los movimientos aleatorios del escape usan la lista completa de operadores con selección uniforme. El sesgo sí actúa en todas las iteraciones del bucle principal de búsqueda guiada.

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
  n          = len(ctx.encoding.id_to_label)
  sqrt_n     = sqrt(n)
  iter_max   = max(50, 20·n)             si iteraciones_max es None
  sin_mejora = max(20, 5·n)              si max_iter_sin_mejora es None
  tam_vec    = max(20, 2·n)              si tam_vecindario es None
  tenure_ini = max(3, round(sqrt_n))     si tabu_tenure_inicial es None
  tenure_min = 3                         si tabu_tenure_min es None
  tenure_max = max(15, round(3·sqrt_n))  si tabu_tenure_max es None
  iter_pac   = max(5, round(2·sqrt_n))   si iter_sin_repeticion_para_reducir es None
  n_escape   = max(3, n//10)             si num_movimientos_escape es None
  |
  v
Seleccionar mejor solución inicial
  |
  v
Inicializar:
  sol_actual    = sol_inicial
  tenure_actual = tenure_ini
  lista_tabu    = deque(maxlen=tenure_actual)  +  set_tabu = set()  +  counter_tabu = Counter()
  historial     = { hash(sol_inicial): {ultima_vista: -1, veces_vista: 1} }
  iter_sin_rep  = 0
  |
  v
+--------------------------------------------------------------+
|  BUCLE PRINCIPAL: mientras iteracion < iter_max            |
|  Y iter_sin_mejora < max_sin_mejora                        |
|                                                            |
|  -- Selección de grupo de operadores (sesgo p_inter) --    |
|  (idéntico al TS simple y SA)                              |
|                                                            |
|  -- Generación + evaluación del lote (vectorizada) --      |
|  -- Selección best-improvement no-tabú con aspiración --   |
|  -- Movimiento al vecino elegido --                        |
|  -- Actualización de la lista tabú FIFO --                 |
|  -- Actualización del mejor global --                      |
|                                                            |
|  === MECANISMO REACTIVO DEL TENURE ===                     |
|  h = _hash_solucion(sol_actual)                            |
|                                                            |
|  si h EN historial (REPETICIÓN):                           |
|      historial[h]["veces_vista"] += 1                      |
|      iter_sin_rep = 0                                      |
|      nuevo_tenure = min(tenure_max,                        |
|                     max(tenure+1, round(tenure·f_aum)))    |
|      si nuevo_tenure > tenure:                             |
|          tenure ← nuevo_tenure                             |
|          reconstruir deque con nueva maxlen                |
|          num_aumentos += 1                                 |
|                                                            |
|      si veces_vista >= umbral_escape:                      |
|          num_escapes += 1                                  |
|          para _ en range(n_escape):                        |
|              aplicar movimiento aleatorio (sin tabú)       |
|          limpiar lista_tabu y set_tabu                     |
|          limpiar historial                                 |
|          iter_sin_rep = 0                                  |
|          registrar nueva sol en historial                  |
|          si costo_actual < mejor_costo:                    |
|              actualizar mejor global                       |
|                                                            |
|  si h NO EN historial (SOLUCIÓN NUEVA):                    |
|      historial[h] = {ultima_vista: iter, veces_vista: 1}  |
|      iter_sin_rep += 1                                     |
|      si iter_sin_rep >= iter_paciencia:                    |
|          nuevo_tenure = max(tenure_min,                    |
|                          min(tenure-1, round(tenure·f_red)))|
|          si nuevo_tenure < tenure:                         |
|              tenure ← nuevo_tenure                         |
|              reconstruir deque con nueva maxlen            |
|              num_reducciones += 1                          |
|          iter_sin_rep = 0                                  |
|                                                            |
|  iteracion += 1                                            |
+--------------------------------------------------------------+
  |
  v
Calcular métricas (mejora_absoluta, mejora_porcentaje, tenure_promedio)
  |
  v
Opcional: guardar CSV
  |
  v
Retornar BusquedaTabuReactivaResult
  |
  v
FIN
```

---

## La función `busqueda_tabu_reactiva`

### Firma exacta

```python
def busqueda_tabu_reactiva(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    iteraciones_max: int | None = None,
    max_iter_sin_mejora: int | None = None,
    tam_vecindario: int | None = None,
    tabu_tenure_inicial: int | None = None,
    tabu_tenure_min: int | None = None,
    tabu_tenure_max: int | None = None,
    factor_aumento: float = 1.2,
    factor_reduccion: float = 0.9,
    iter_sin_repeticion_para_reducir: int | None = None,
    umbral_repeticiones_escape: int = 3,
    num_movimientos_escape: int | None = None,
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
    max_iter_sin_mejora_kick: int | None = None,
    intensificador: Callable | None = None,
    **_ignorado_kwargs: object,
) -> BusquedaTabuReactivaResult
```

> **Nota:** Los parámetros `id_corrida` y `config_id` son absorbidos por `**_ignorado_kwargs` y **no se escriben en el CSV**. Esta es la convención del proyecto.

### Tabla de parámetros

| Parámetro | Tipo | Default / Fórmula instance-aware | Descripción |
|---|---|---|---|
| `inicial_obj` | `Any` | — | Objeto pickle con la(s) solución(es) inicial(es). |
| `data` | `Mapping[str, Any]` | — | Datos de la instancia CARP. |
| `G` | `nx.Graph` | — | Grafo de la instancia cargado desde GEXF. |
| `iteraciones_max` | `int \| None` | `None` → `max(50, 20·n)` | Cota dura de iteraciones (criterio de parada 1). |
| `max_iter_sin_mejora` | `int \| None` | `None` → `max(20, 5·n)` | Estancamiento: iters consecutivas sin mejorar (criterio 2). |
| `tam_vecindario` | `int \| None` | `None` → `max(20, 2·n)` | Vecinos generados por iteración (best-improvement sobre el lote). |
| `tabu_tenure_inicial` | `int \| None` | `None` → `max(3, round(√n))` | Tenencia tabú de arranque. Regla clásica de Glover-Laguna. |
| `tabu_tenure_min` | `int \| None` | `None` → `3` | Cota inferior de la tenencia. Evita listas tabú efímeras. |
| `tabu_tenure_max` | `int \| None` | `None` → `max(15, round(3·√n))` | Cota superior de la tenencia. Limita el exceso de prohibiciones. El piso 15 evita que instancias pequeñas (n < 25) usen un techo demasiado bajo y el mecanismo reactivo no tenga espacio real para actuar. |
| `factor_aumento` | `float` | `1.2` | Multiplicador del tenure al detectar repetición. Debe ser `> 1.0`. |
| `factor_reduccion` | `float` | `0.9` | Multiplicador del tenure al pasar tiempo sin repeticiones. Debe estar en `(0, 1)`. |
| `iter_sin_repeticion_para_reducir` | `int \| None` | `None` → `max(5, round(2·√n))` | Iteraciones sin repetición antes de reducir el tenure. |
| `umbral_repeticiones_escape` | `int` | `3` | Veces que una solución debe repetirse para disparar el escape. Debe ser `>= 2`. |
| `num_movimientos_escape` | `int \| None` | `None` → `max(3, n // 10)` | Movimientos aleatorios aplicados durante el escape. |
| `semilla` | `int \| None` | `None` | Semilla para `random.Random`. `None` produce corridas no reproducibles. |
| `operadores` | `Iterable[str]` | `OPERADORES_POPULARES` | Conjunto de operadores de vecindario habilitados. |
| `marcador_depot_etiqueta` | `str \| None` | `None` | Etiqueta del nodo depósito. `None` la lee del contexto. |
| `usar_gpu` | `bool` | `False` | Flag de GPU. Solo para trazabilidad; el bucle principal ya es eficiente en CPU. |
| `backend_vecindario` | `Literal["labels", "ids"]` | `"labels"` | Representación interna para generar vecinos. |
| `guardar_historial` | `bool` | `True` | Si `True`, guarda `historial_mejor_costo` e `historial_tenure` en el resultado. |
| `guardar_csv` | `bool` | `False` | Si `True`, escribe una fila en el CSV al finalizar. |
| `ruta_csv` | `str \| None` | `None` | Ruta del CSV. Si `None`, se genera automáticamente. |
| `nombre_instancia` | `str` | `"instancia"` | Nombre de la instancia para el CSV y la caché del contexto. |
| `repeticion` | `int \| None` | `None` | Número de repetición dentro de un experimento. |
| `root` | `str \| None` | `None` | Directorio raíz alternativo para los archivos de la instancia. |
| `extra_csv` | `dict[str, object] \| None` | `None` | Columnas adicionales para el CSV. |
| `alpha_inter` | `float \| None` | `None` → `0.8` | P(elegir inter-ruta) cuando la solución viola capacidad. |
| `p_inter` | `float \| None` | `None` → `0.6` | P(elegir inter-ruta) cuando la solución es factible. |
| `max_iter_sin_mejora_kick` | `int \| None` | `None` | Umbral de iteraciones consecutivas sin mejora para disparar el hook del intensificador. Si `None`, el hook nunca se activa. Solo tiene efecto cuando `intensificador` también se provee. |
| `intensificador` | `Callable \| None` | `None` | Hook opcional de intensificación. Si se provee, se invoca con `intensificador(sol, mejor_global, ctx, lam, rng, encoding, md)` cuando `iter_sin_mejora >= max_iter_sin_mejora_kick`. Con `None` (default) el comportamiento es idéntico al anterior. |

### Función de conveniencia: `busqueda_tabu_reactiva_desde_instancia`

```python
def busqueda_tabu_reactiva_desde_instancia(
    nombre_instancia: str,
    *,
    root: str | None = None,
    # mismos parámetros que busqueda_tabu_reactiva, sin inicial_obj, data, G
) -> BusquedaTabuReactivaResult
```

---

## `BusquedaTabuReactivaResult`

Dataclass inmutable (`frozen=True, slots=True`). Extiende los campos del TS simple con métricas específicas del comportamiento reactivo.

### Campos comunes con el TS simple

Los mismos campos que `BusquedaTabuSimpleResult` (ver `docs/busqueda_tabu_simple.md`), más `historial_tenure: list[int]` (historial del tenure efectivo al inicio de cada iteración; solo si `guardar_historial=True`).

### Campos específicos de RTS

| Campo | Tipo | Descripción |
|---|---|---|
| `tenure_final` | `int` | Valor del tenure al terminar la corrida. Indica en qué régimen se quedó el algoritmo. |
| `tenure_promedio` | `float` | Promedio del tenure a lo largo de toda la ejecución. Cuantifica la "presión tabú media". |
| `tenure_max_alcanzado` | `int` | Máximo valor del tenure observado durante la corrida. |
| `tenure_min_alcanzado` | `int` | Mínimo valor del tenure observado durante la corrida. |
| `num_repeticiones_detectadas` | `int` | Iteraciones en las que el hash de la solución actual coincidió con uno ya visto. |
| `num_escapes_realizados` | `int` | Veces que se disparó el mecanismo de escape. |
| `num_aumentos_tenure` | `int` | Veces que el tenure creció por detección de ciclo. |
| `num_reducciones_tenure` | `int` | Veces que el tenure decreció por ausencia de repeticiones. |
| `tenure_inicial_aplicado` | `int` | Valor efectivo de `tabu_tenure_inicial` usado tras resolver el default instance-aware. |
| `tenure_min_aplicado` | `int` | Valor efectivo de `tabu_tenure_min`. |
| `tenure_max_aplicado` | `int` | Valor efectivo de `tabu_tenure_max`. |
| `alpha_inter_aplicado` | `float` | Valor efectivo de `alpha_inter`. |
| `p_inter_aplicado` | `float` | Valor efectivo de `p_inter`. |
| `iteraciones_con_violacion` | `int` | Iteraciones del bucle principal (no del escape) donde la solución violaba capacidad. |

---

## Columnas del CSV de salida

El CSV se guarda en `experimentos/tabu_reactive_small_20260517/` cuando se usa `run_tabu_reactiva_automatico.py`. La etiqueta de metaheurística en el CSV es `"tabu_reactiva"`.

El CSV de RTS contiene todas las columnas del TS simple más las columnas específicas reactivas listadas a continuación.

### Columnas de parámetros (comparables con TS simple)

Las mismas columnas de identificación y estadísticas que el TS simple, incluyendo `tabu_tenure` (que en RTS contiene el **valor inicial** del tenure para comparabilidad con el TS simple).

### Columnas específicas de RTS (16 columnas adicionales)

| Columna | Descripción |
|---|---|
| `tenure_inicial` | Tenure inicial efectivo (igual que `tabu_tenure` en este CSV) |
| `tenure_min_aplicado` | Cota inferior efectiva del tenure |
| `tenure_max_aplicado` | Cota superior efectiva del tenure |
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

### Columnas del sesgo inter/intra (iguales que TS simple)

`alpha_inter`, `p_inter`, `iteraciones_con_violacion`, `fraccion_iter_con_violacion`.

---

## Ejemplos de uso

### Opción 1: función de conveniencia (recomendada)

```python
from metacarp.busqueda_tabu_reactiva import busqueda_tabu_reactiva_desde_instancia

resultado = busqueda_tabu_reactiva_desde_instancia(
    "gdb1",
    # Si se pasan None (o se omiten), se calculan los valores instance-aware:
    # iteraciones_max = max(50, 20·n)
    # max_iter_sin_mejora = max(20, 5·n)
    # etc.
    factor_aumento=1.2,
    factor_reduccion=0.9,
    umbral_repeticiones_escape=3,
    semilla=42,
    guardar_historial=True,
    guardar_csv=True,
    ruta_csv="resultados/tabu_reactiva_gdb1.csv",
    repeticion=1,
)

print(f"Mejor costo                 : {resultado.mejor_costo:.2f}")
print(f"Tiempo                      : {resultado.tiempo_segundos:.2f} s")
print(f"Iteraciones                 : {resultado.iteraciones_totales}")
print(f"Tenure inicial              : {resultado.tenure_inicial_aplicado}")
print(f"Tenure final                : {resultado.tenure_final}")
print(f"Tenure promedio             : {resultado.tenure_promedio:.2f}")
print(f"Repeticiones detectadas     : {resultado.num_repeticiones_detectadas}")
print(f"Escapes realizados          : {resultado.num_escapes_realizados}")
print(f"Aumentos de tenure          : {resultado.num_aumentos_tenure}")
print(f"Reducciones de tenure       : {resultado.num_reducciones_tenure}")

# Visualizar la evolución del tenure
if resultado.historial_tenure:
    for i, (c, t) in enumerate(zip(resultado.historial_mejor_costo,
                                    resultado.historial_tenure)):
        if i % 50 == 0:
            print(f"  iter={i:4d}  costo={c:.2f}  tenure={t}")
```

### Opción 2: carga manual de recursos

```python
from metacarp.busqueda_tabu_reactiva import busqueda_tabu_reactiva
from metacarp.instances import load_instances
from metacarp.cargar_grafos import cargar_objeto_gexf
from metacarp.cargar_soluciones_iniciales import cargar_solucion_inicial

nombre     = "gdb1"
data       = load_instances(nombre)
G          = cargar_objeto_gexf(nombre)
inicial_obj = cargar_solucion_inicial(nombre)

resultado = busqueda_tabu_reactiva(
    inicial_obj,
    data,
    G,
    # Pasar None = usar defaults instance-aware (recomendado para instancias nuevas)
    iteraciones_max=None,
    max_iter_sin_mejora=None,
    tam_vecindario=None,
    tabu_tenure_inicial=None,
    tabu_tenure_min=None,
    tabu_tenure_max=None,
    iter_sin_repeticion_para_reducir=None,
    num_movimientos_escape=None,
    factor_aumento=1.2,
    factor_reduccion=0.9,
    umbral_repeticiones_escape=3,
    semilla=42,
    guardar_historial=True,
    guardar_csv=False,
    nombre_instancia=nombre,
    # alpha_inter=None  ->  usa 0.8 (mismo default que SA)
    # p_inter=None      ->  usa 0.6 (mismo default que SA)
)
```

### Script de experimentos automático

```bash
# Grid 4D: factor_aumento × umbral_escape × p_inter × factor_reduccion
# 5 × 4 × 5 × 3 = 300 combos × 23 instancias × 5 reps = 34,500 corridas
python scripts/run_tabu_reactiva_automatico.py --salida-dir experimentos/rts_grid

# Con paralelismo (N procesos simultáneos)
python scripts/run_tabu_reactiva_automatico.py --salida-dir experimentos/rts_grid --workers 8

# Con más repeticiones
python scripts/run_tabu_reactiva_automatico.py --salida-dir experimentos/rts_grid --repeticiones 5
```

Las semillas son aleatorias del sistema (sin `--semilla-base`) para que las repeticiones sean estadísticamente independientes. Los CSV se escriben directamente en `<salida-dir>/`.

---

## Pseudocódigo

```
RTS(sol_inicial, tenure_ini, tenure_min, tenure_max, f_aum, f_red,
    iter_pac, umbral_esc, n_esc, iteraciones_max, max_sin_mejora,
    tam_vec, alpha_inter, p_inter):

  s       ← sol_inicial
  s*      ← s;  c* ← costo(s)
  tenure  ← tenure_ini
  T       ← deque(maxlen=tenure)  +  T_set = {}
  H       ← { hash(s): {veces: 1, ultima: -1} }  // historial
  rep_cnt ← 0      // iteraciones consecutivas sin repetición
  sin_mej ← 0;  iter ← 0

  mientras iter < iter_max Y sin_mej < max_sin_mejora:

    // Selección de grupo de operadores (sesgo p_inter)
    grupo ← seleccionar_grupo(alpha_inter, p_inter, viol(s))
    N     ← generar_lote(s, grupo, tam_vec)
    evaluar_lote(N)

    // Best-improvement no-tabú con aspiración clásica
    s_elg ← argmin { costo(v) : v ∈ N, no_tabu(v) O aspira(v, c*) }
    si todos tabú:  s_elg ← argmin { costo(v) : v ∈ N }

    // Actualizar lista tabú
    insertar _clave(s_elg) en T y T_set (con mantenimiento FIFO)
    s ← s_elg

    // Actualizar mejor global
    si costo(s) < c*:  s* ← s;  c* ← costo(s);  sin_mej ← 0
    si no:             sin_mej += 1

    // === Mecanismo reactivo ===
    h ← hash(s)
    si h EN H:
        H[h]["veces"] += 1
        rep_cnt ← 0
        tenure ← min(tenure_max, max(tenure+1, round(tenure·f_aum)))
        reconstruir deque con nuevo tenure

        si H[h]["veces"] >= umbral_esc:
            // ESCAPE
            para _ en range(n_esc):
                s ← aplicar_movimiento_aleatorio(s)  // ignora lista tabú
            limpiar T, T_set, H
            rep_cnt ← 0
            H[hash(s)] ← {veces: 1}
    si no:
        H[h] ← {veces: 1, ultima: iter}
        rep_cnt += 1
        si rep_cnt >= iter_pac:
            tenure ← max(tenure_min, min(tenure-1, round(tenure·f_red)))
            reconstruir deque con nuevo tenure
            rep_cnt ← 0

    iter += 1

  retornar s*, c*
```

---

## Guía de ajuste de parámetros

| Parámetro | Efecto con valor bajo | Efecto con valor alto | Recomendación |
|---|---|---|---|
| `factor_aumento` | Tenure crece lentamente ante ciclos; menos protección | Tenure sube agresivamente; puede prohibir demasiado | `1.1–1.3`. Default `1.2` es un buen equilibrio. |
| `factor_reduccion` | Tenure baja rápidamente cuando hay exploración libre | Tenure baja lentamente; la presión tabú disminuye despacio | `0.85–0.95`. Default `0.9`. |
| `umbral_repeticiones_escape` | Escape muy frecuente; la búsqueda se vuelve casi aleatoria | Escape muy raro; el ciclo puede persistir mucho tiempo | `2–5`. Default `3`. |
| `num_movimientos_escape` | Salto corto; puede no escapar del pozo de atracción | Salto largo; buena diversificación pero pierde contexto | `None` (instance-aware: `max(3, n//10)`). |
| `tabu_tenure_max` | Rango dinámico estrecho; poco efecto reactivo | Tenencias muy largas; demasiadas prohibiciones | `None` (instance-aware: `max(10, round(3·√n))`). |
| `iter_sin_repeticion_para_reducir` | Paciencia corta; el tenure baja muy rápido | Paciencia larga; el tenure tarda en relajarse | `None` (instance-aware: `max(5, round(2·√n))`). |

---

## Decisiones de diseño y comparación con TS simple

| Aspecto | TS simple | RTS |
|---|---|---|
| Tenencia tabú | Fija (`tabu_tenure`) | Dinámica en `[min, max]` |
| Detección de ciclos | No (implícita via tenencia) | Explícita con hash canónico |
| Escape | No | Sí: movimientos aleatorios + limpieza de memoria |
| Sesgo inter/intra | Sí (helper compartido) | Sí (mismo helper; el escape no aplica sesgo) |
| Parámetros instance-aware | Parcial (regla `√n` manual) | Total: todos los params reactivos dependen de `n` |
| Complejidad de memoria | O(tenure) por lista tabú | O(tenure) + O(H) por historial; H puede crecer mucho sin escapes |

---

## Referencias

- Battiti, R., & Tecchiolli, G. (1994). "The Reactive Tabu Search." *ORSA Journal on Computing*, 6(2), 126–140.
- Glover, F. (1986). "Future paths for integer programming and links to artificial intelligence." *Computers & Operations Research*, 13(5), 533–549.
- Glover, F., & Laguna, M. (1997). *Tabu Search*. Kluwer Academic Publishers.
