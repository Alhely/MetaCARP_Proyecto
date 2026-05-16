# Generación de vecinos en MetaCARP

Este documento describe en detalle el sistema de generación de vecinos del proyecto
MetaCARP: desde la representación de una solución, pasando por los 9 operadores de
vecindad disponibles, hasta la integración completa con las metaheurísticas del
paquete.

---

## Tabla de contenidos

1. [Introducción conceptual](#1-introducción-conceptual)
2. [Los 9 operadores de vecindad](#2-los-9-operadores-de-vecindad)
3. [La estructura `MovimientoVecindario`](#3-la-estructura-movimientovecindario)
4. [Las funciones `generar_vecino` y `generar_vecino_ids`](#4-las-funciones-generar_vecino-y-generar_vecino_ids)
5. [Ejemplo completo de uso](#5-ejemplo-completo-de-uso)
6. [Integración con las metaheurísticas](#6-integración-con-las-metaheurísticas)
7. [Referencia rápida](#7-referencia-rápida)

---

## 1. Introducción conceptual

### ¿Qué es un vecino en optimización combinatoria?

En optimización combinatoria, una **solución vecina** (o simplemente "vecino") es
cualquier solución que se puede obtener a partir de la solución actual aplicando un
cambio pequeño y bien definido. El conjunto de todos los vecinos posibles de una
solución se denomina su **vecindario**.

Este concepto es central en las metaheurísticas de búsqueda local: en lugar de
explorar todo el espacio de soluciones (exponencialmente grande), el algoritmo navega
de solución en solución moviéndose por el vecindario a cada paso.

### Por qué el vecindario es central en las metaheurísticas

- **Búsqueda Tabú**: en cada iteración genera un lote de vecinos, elige el mejor no
  prohibido y lo convierte en la nueva solución actual.
- **Abejas (ABC)**: cada "abeja empleada" y cada "abeja observadora" generan un vecino
  de su fuente de alimento y lo aceptan si es mejor.
- **Cuckoo Search**: los "huevos de cucú" se generan como vecinos perturbados y
  compiten con el nido actual.
- **Recocido Simulado (SA)**: en cada iteración se genera un vecino y se acepta
  determinísticamente si mejora, o con probabilidad decreciente si empeora.

La calidad del operador de vecindad determina directamente la velocidad de convergencia
y la calidad de la solución final.

### Representación de una solución CARP

Una solución del Capacitated Arc Routing Problem (CARP) es una **lista de rutas**. En
el paquete metacarp, el tipo concreto es `list[list[str]]`:

```python
# Ejemplo de solución con 2 rutas y 5 tareas en total
solucion = [
    ["D", "TR1", "TR3", "TR5", "D"],   # ruta 0: depot → TR1 → TR3 → TR5 → depot
    ["D", "TR2", "TR4", "D"],           # ruta 1: depot → TR2 → TR4 → depot
]
```

Convenciones:
- `"D"` es el **marcador de depósito** (configurable, por defecto `"D"`). Aparece al
  inicio y al final de cada ruta indicando que el vehículo parte y regresa al depósito.
- `"TR<n>"` son las **etiquetas de tareas requeridas** (Required Tasks). Cada tarea
  corresponde a un arco del grafo CARP que debe ser recorrido exactamente una vez.
- Las etiquetas de tareas no requeridas siguen el mismo formato (p.ej. `"TNR2"`), pero
  los operadores las tratan de la misma forma que las tareas requeridas.

El multiconjunto total de tareas se conserva tras cualquier operador: ningún vecino
añade ni elimina tareas; solo cambia su orden o distribución entre rutas.

### Normalización interna

Antes de que cualquier operador actúe, el módulo `vecindarios.py` elimina los
marcadores `"D"` con `normalizar_para_vecindario`, dejando solo las etiquetas de
tareas:

```python
# Entrada:  [["D","TR1","TR3","D"], ["D","TR2","D"]]
# Salida:   [["TR1","TR3"], ["TR2"]]
```

Al devolver el vecino, `desnormalizar_con_deposito` los restaura si
`devolver_con_deposito=True` (valor por defecto).

---

## 2. Los 9 operadores de vecindad

El módulo `vecindarios.py` implementa 9 operadores agrupados en dos familias. Todos se
listan en la constante `OPERADORES_POPULARES`:

```python
OPERADORES_POPULARES = (
    "relocate_intra",
    "swap_intra",
    "2opt_intra",
    "relocate_inter",
    "swap_inter",
    "2opt_star",
    "cross_exchange",
    "or_opt_2",
    "or_opt_3",
)
```

---

### 2.1 `relocate_intra` — Reubicación intra-ruta

**Tipo**: intra-ruta

**Descripcion**: extrae la tarea que ocupa la posición `i` de una ruta y la reinserta
en la posición `j` de la **misma** ruta. El resto de las tareas se desplazan para
cerrar el hueco y abrir espacio respectivamente.

**Requisito**: la ruta debe tener al menos 2 tareas. Los índices `i` y `j` deben ser
distintos.

**Diagrama antes/después** (ruta con 4 tareas, `i=0`, `j=2`):

```
Antes:   [ TR1, TR2, TR3, TR4 ]
                                    ← pop(0): extrae TR1
         [ TR2, TR3, TR4 ]
                                    ← insert(2, TR1): inserta TR1 en posición 2
Después: [ TR2, TR3, TR1, TR4 ]
```

**Cuándo conviene**: cuando una tarea está en una posición ineficiente dentro de la
ruta y servirla en otro momento de la secuencia reduce el deadheading (traslado en
vacío entre tareas).

---

### 2.2 `swap_intra` — Intercambio intra-ruta

**Tipo**: intra-ruta

**Descripcion**: intercambia directamente las tareas en las posiciones `i` y `j` de
la **misma** ruta. Ninguna otra tarea cambia de posición.

**Requisito**: la ruta debe tener al menos 2 tareas. Los índices `i` y `j` deben ser
distintos.

**Diagrama antes/después** (`i=0`, `j=3`):

```
Antes:   [ TR1, TR2, TR3, TR4 ]
                                    ← swap(0, 3)
Después: [ TR4, TR2, TR3, TR1 ]
```

**Cuándo conviene**: cuando dos tareas específicas de la misma ruta tienen mejor
eficiencia al servirse en orden inverso. Es un movimiento conservador que no altera la
longitud de la ruta.

---

### 2.3 `2opt_intra` — Inversión de segmento intra-ruta

**Tipo**: intra-ruta

**Descripcion**: invierte (voltea) el subsegmento de tareas entre las posiciones `i`
y `j` (ambas inclusive) dentro de la **misma** ruta.

**Requisito**: la ruta debe tener al menos 3 tareas. Se garantiza `i < j`.

**Diagrama antes/después** (`i=1`, `j=3`):

```
Antes:   [ TR1, TR2, TR3, TR4, TR5 ]
         segmento [1:4] = [ TR2, TR3, TR4 ]
                                    ← reversed: [ TR4, TR3, TR2 ]
Después: [ TR1, TR4, TR3, TR2, TR5 ]
```

**Cuándo conviene**: cuando un subsegmento de la ruta se puede servir en sentido
contrario con menor costo de traslado. Es la generalización del operador 2-opt clásico
del TSP al dominio de arcos.

---

### 2.4 `relocate_inter` — Reubicación inter-ruta

**Tipo**: inter-ruta

**Descripcion**: extrae la tarea en posición `i` de la ruta origen `ra` y la inserta
en la posición `j` de la ruta destino `rb` (diferente a `ra`). La tarea cambia de
ruta.

**Requisito**: la ruta origen debe tener al menos 1 tarea. `ra != rb`.

**Diagrama antes/después** (ruta A: 3 tareas, ruta B: 2 tareas; `i=2`, `j=1`):

```
Antes:   ruta_A = [ TR1, TR2, TR3 ]    ruta_B = [ TR4, TR5 ]
                                    ← pop ruta_A[2]: extrae TR3
         ruta_A = [ TR1, TR2 ]
                                    ← insert ruta_B[1]: inserta TR3 en posición 1
Después: ruta_A = [ TR1, TR2 ]    ruta_B = [ TR4, TR3, TR5 ]
```

**Cuándo conviene**: para equilibrar la carga entre rutas, o cuando una tarea tiene
mejor afinidad geográfica con las tareas de otra ruta. Puede vaciar completamente una
ruta si solo tenía una tarea.

---

### 2.5 `swap_inter` — Intercambio inter-ruta

**Tipo**: inter-ruta

**Descripcion**: intercambia la tarea en posición `i` de la ruta `ra` con la tarea en
posición `j` de la ruta `rb`. Ambas rutas conservan su número de tareas.

**Requisito**: ambas rutas deben tener al menos 1 tarea. `ra != rb`.

**Diagrama antes/después** (`i=0`, `j=1`):

```
Antes:   ruta_A = [ TR1, TR2 ]    ruta_B = [ TR3, TR4 ]
                                    ← swap ruta_A[0] ↔ ruta_B[1]
Después: ruta_A = [ TR4, TR2 ]    ruta_B = [ TR3, TR1 ]
```

**Cuándo conviene**: para redistribuir tareas entre rutas sin modificar la cardinalidad
de ninguna. Es menos disruptivo que `relocate_inter` y apropiado cuando las rutas ya
tienen una distribución de carga equilibrada.

---

### 2.6 `2opt_star` — Intercambio de colas inter-ruta

**Tipo**: inter-ruta

**Descripcion**: divide cada ruta en una cabeza y una cola por un punto de corte, y
luego intercambia las colas entre las dos rutas. La ruta A conserva su cabeza y adopta
la cola de B; la ruta B conserva su cabeza y adopta la cola de A.

Formalmente:
- `ruta_A = cabeza_A + cola_A`  (corte en `cut_a`: `cabeza_A = ruta_A[:cut_a+1]`, `cola_A = ruta_A[cut_a+1:]`)
- `ruta_B = cabeza_B + cola_B`  (corte en `cut_b`: `cabeza_B = ruta_B[:cut_b+1]`, `cola_B = ruta_B[cut_b+1:]`)
- Resultado: `ruta_A' = cabeza_A + cola_B`, `ruta_B' = cabeza_B + cola_A`

**Requisito**: ambas rutas deben tener al menos 1 tarea. `ra != rb`.

**Diagrama antes/después** (`cut_a=1`, `cut_b=1`):

```
Antes:   ruta_A = [ TR1, TR2, TR3, TR4 ]
         ruta_B = [ TR5, TR6, TR7 ]

         cabeza_A = [TR1, TR2]    cola_A = [TR3, TR4]
         cabeza_B = [TR5, TR6]    cola_B = [TR7]

Después: ruta_A' = [ TR1, TR2, TR7 ]
         ruta_B' = [ TR5, TR6, TR3, TR4 ]
```

**Cuándo conviene**: cuando dos rutas se "cruzan" geográficamente en la segunda mitad
de su recorrido. Intercambiar colas puede eliminar esos cruces y reducir el
deadheading. Es la extensión del clásico 2-opt del TSP al caso de rutas múltiples.

---

### 2.7 `cross_exchange` — Intercambio de segmentos inter-ruta

**Tipo**: inter-ruta

**Descripcion**: extrae un segmento contiguo `[i..j]` de la ruta `ra` y un segmento
contiguo `[k..l]` de la ruta `rb` (ambos índices inclusive), y los intercambia entre
sí.

Formalmente:
- `seg_A = ruta_A[i:j+1]`
- `seg_B = ruta_B[k:l+1]`
- `ruta_A' = ruta_A[:i] + seg_B + ruta_A[j+1:]`
- `ruta_B' = ruta_B[:k] + seg_A + ruta_B[l+1:]`

**Requisito**: ambas rutas deben tener al menos 2 tareas. `ra != rb`. Se garantiza
`i < j` y `k < l`.

**Diagrama antes/después** (`i=1`, `j=2`, `k=0`, `l=1`):

```
Antes:   ruta_A = [ TR1, TR2, TR3, TR4 ]
         ruta_B = [ TR5, TR6, TR7 ]

         seg_A = [TR2, TR3]    seg_B = [TR5, TR6]

         ruta_A[:1] = [TR1]    ruta_A[3:] = [TR4]
         ruta_B[:0] = []       ruta_B[2:] = [TR7]

Después: ruta_A' = [ TR1, TR5, TR6, TR4 ]
         ruta_B' = [ TR2, TR3, TR7 ]
```

**Cuándo conviene**: cuando bloques de tareas adyacentes tienen mejor encaje en la
ruta contraria. Genera vecinos más disruptivos que `swap_inter` (que solo mueve una
tarea a la vez), por lo que es útil para escapar de mínimos locales profundos.

---

### 2.8 `or_opt_2` — Reubicación de par inter-ruta

**Tipo**: inter-ruta

**Descripcion**: extrae un bloque de **2 tareas consecutivas** en la posición `i` de
la ruta origen `ra` y lo inserta en la posición `j` de la ruta destino `rb` (diferente
a `ra`). Las dos tareas mantienen su orden relativo al moverse.

**Requisito**: la ruta origen debe tener al menos 2 tareas. `ra != rb`.

**Diagrama antes/después** (ruta A: 4 tareas, ruta B: 2 tareas; `i=1`, `j=1`):

```
Antes:   ruta_A = [ TR1, TR2, TR3, TR4 ]    ruta_B = [ TR5, TR6 ]
                                    ← pop ruta_A[1:3]: extrae [TR2, TR3]
         ruta_A = [ TR1, TR4 ]
                                    ← insert ruta_B[1]: inserta [TR2, TR3] en posición 1
Después: ruta_A = [ TR1, TR4 ]    ruta_B = [ TR5, TR2, TR3, TR6 ]
```

**Cuándo conviene**: cuando dos tareas consecutivas de una ruta tienen mejor afinidad
geográfica con las tareas de otra ruta. Más potente que `relocate_inter` porque mueve
un bloque completo, lo que puede reducir el deadheading en ambas rutas simultáneamente.
Es un operador Or-opt clásico (Tsitsiklis, 1992) extendido al dominio multi-ruta.

---

### 2.9 `or_opt_3` — Reubicación de trío inter-ruta

**Tipo**: inter-ruta

**Descripcion**: extrae un bloque de **3 tareas consecutivas** en la posición `i` de
la ruta origen `ra` y lo inserta en la posición `j` de la ruta destino `rb` (diferente
a `ra`). Las tres tareas mantienen su orden relativo.

**Requisito**: la ruta origen debe tener al menos 3 tareas. `ra != rb`.

**Diagrama antes/después** (ruta A: 5 tareas, ruta B: 2 tareas; `i=1`, `j=0`):

```
Antes:   ruta_A = [ TR1, TR2, TR3, TR4, TR5 ]    ruta_B = [ TR6, TR7 ]
                                    ← pop ruta_A[1:4]: extrae [TR2, TR3, TR4]
         ruta_A = [ TR1, TR5 ]
                                    ← insert ruta_B[0]: inserta [TR2, TR3, TR4] en posición 0
Después: ruta_A = [ TR1, TR5 ]    ruta_B = [ TR2, TR3, TR4, TR6, TR7 ]
```

**Cuándo conviene**: para mover subsecuencias largas entre rutas cuando tres tareas
consecutivas forman un bloque geográficamente compacto que encaja mejor en otra ruta.
Es el movimiento más disruptivo de la familia Or-opt: puede reestructurar la distribución
de tareas significativamente, útil para escapar de mínimos locales profundos que los
operadores de una sola tarea no pueden salvar.

---

## 3. La estructura `MovimientoVecindario`

`MovimientoVecindario` es una clase de datos definida en `vecindarios.py` que describe
con precisión qué movimiento se aplicó, sobre qué rutas, en qué posiciones y qué
tareas fueron desplazadas.

```python
@dataclass(frozen=True, slots=True)
class MovimientoVecindario:
    operador: str
    ruta_a: int | None = None
    ruta_b: int | None = None
    i: int | None = None
    j: int | None = None
    k: int | None = None
    l: int | None = None
    id_movidos: tuple[int, ...] = ()
    labels_movidos: tuple[str, ...] = ()
    backend_solicitado: str = "labels"
    backend_real: str = "cpu"
```

### Descripcion de cada campo

| Campo | Tipo | Descripcion |
|---|---|---|
| `operador` | `str` | Nombre del operador aplicado, p.ej. `"swap_inter"` o `"cross_exchange"`. |
| `ruta_a` | `int \| None` | Indice (0-based) de la primera ruta involucrada. |
| `ruta_b` | `int \| None` | Indice de la segunda ruta. Solo se rellena para operadores inter-ruta. |
| `i` | `int \| None` | Primer indice de posicion dentro de `ruta_a`. Para `2opt_star` es el punto de corte de la ruta A. |
| `j` | `int \| None` | Segundo indice: destino en `relocate_intra/inter`, segundo punto de swap, fin de segmento en `2opt_intra`, o punto de corte de la ruta B en `2opt_star`. |
| `k` | `int \| None` | Inicio del segmento en `ruta_b` para `cross_exchange`. `None` en todos los demás operadores. |
| `l` | `int \| None` | Fin del segmento en `ruta_b` para `cross_exchange`. `None` en todos los demás operadores. |
| `id_movidos` | `tuple[int, ...]` | IDs enteros de las tareas que cambiaron de posición o de ruta. Disponible cuando se usa el backend `"ids"`. |
| `labels_movidos` | `tuple[str, ...]` | Etiquetas (`"TR1"`, `"TR3"`...) de las tareas desplazadas. Disponible cuando se pasa un `SearchEncoding`. |
| `backend_solicitado` | `str` | Backend pedido por el llamador: `"cpu"` o `"gpu"`. |
| `backend_real` | `str` | Backend que ejecutó realmente el movimiento. Hoy siempre `"cpu"` (el backend GPU es un placeholder). |

### Que significa `frozen=True`

`frozen=True` hace que el objeto sea **inmutable**: una vez creado, ninguno de sus
campos puede modificarse. Si se intenta asignar `mov.ruta_a = 5` Python lanzará un
`FrozenInstanceError`. Esta garantía es importante porque:

1. El objeto se puede usar como **clave de diccionario** (es hashable), lo que permite
   indexarlo directamente en la lista tabú de Búsqueda Tabú.
2. Se puede pasar entre funciones sin riesgo de mutación accidental, lo que hace el
   código más seguro en entornos de múltiples funciones.

### Que significa `slots=True`

`slots=True` le indica a Python que reserve espacio **fijo** en memoria para los
atributos del objeto, en lugar de usar un diccionario interno (`__dict__`). El
resultado es:

- Menos memoria por objeto (importante cuando se crean miles de `MovimientoVecindario`
  por corrida).
- Acceso a atributos más rápido (lookup directo en lugar de búsqueda en hash).

---

## 4. Las funciones `generar_vecino` y `generar_vecino_ids`

El módulo expone dos funciones principales para generar vecinos.

---

### 4.1 `generar_vecino`

**Firma**:

```python
def generar_vecino(
    solucion: Sequence[Sequence[Hashable]],
    *,
    rng: random.Random | None = None,
    operadores: Iterable[str] = OPERADORES_POPULARES,
    marcador_depot: str = "D",
    devolver_con_deposito: bool = True,
    usar_gpu: bool = False,
    backend: Literal["labels", "ids"] = "labels",
    encoding: SearchEncoding | None = None,
) -> tuple[list[list[str]], MovimientoVecindario]:
```

**Parametros**:

| Parametro | Tipo | Por defecto | Descripcion |
|---|---|---|---|
| `solucion` | `Sequence[Sequence[Hashable]]` | — | Solucion actual en formato etiquetas. Puede incluir o no el marcador `"D"`. |
| `rng` | `random.Random \| None` | `None` | Generador de numeros aleatorios. Si es `None` se crea uno nuevo (no reproducible). Pasar un `random.Random(semilla)` garantiza reproducibilidad. |
| `operadores` | `Iterable[str]` | `OPERADORES_POPULARES` | Nombres de los operadores entre los que se elige aleatoriamente. Puede pasarse un subconjunto para restringir la búsqueda. |
| `marcador_depot` | `str` | `"D"` | Token de deposito a eliminar antes de operar y (si aplica) a restaurar en la salida. |
| `devolver_con_deposito` | `bool` | `True` | Si `True`, la solucion vecina devuelta incluye `"D"` al inicio y fin de cada ruta. |
| `usar_gpu` | `bool` | `False` | Flag de backend GPU. Hoy actua como placeholder: el campo `backend_real` del movimiento siempre sera `"cpu"`. |
| `backend` | `Literal["labels", "ids"]` | `"labels"` | Modo de operacion interno: `"labels"` opera sobre strings directamente; `"ids"` codifica a enteros, aplica el operador sobre IDs y decodifica. |
| `encoding` | `SearchEncoding \| None` | `None` | Codificacion entera de la instancia. Obligatorio cuando `backend="ids"`. Opcional pero recomendado con `backend="labels"` para rellenar `labels_movidos`. |

**Retorna**: `tuple[list[list[str]], MovimientoVecindario]`
- El primer elemento es la solucion vecina en formato etiquetas.
- El segundo es el descriptor del movimiento aplicado.

**Flujo interno**:

1. Si `backend="ids"`: codifica la solucion con `encode_solution`, llama a
   `generar_vecino_ids` y decodifica el resultado con `decode_solution`.
2. Si `backend="labels"`: elimina el marcador de deposito con
   `normalizar_para_vecindario`, elige un operador al azar, selecciona indices
   aleatorios validos para ese operador (con hasta 500 reintentos si los elegidos no
   son aplicables), aplica el operador, y restaura el deposito si se pide.

---

### 4.2 `generar_vecino_ids`

**Firma**:

```python
def generar_vecino_ids(
    solucion_ids: Sequence[Sequence[int]],
    *,
    rng: random.Random | None = None,
    operadores: Iterable[str] = OPERADORES_POPULARES,
    usar_gpu: bool = False,
    encoding: SearchEncoding | None = None,
) -> tuple[list[list[int]], MovimientoVecindario]:
```

**Parametros**:

| Parametro | Tipo | Por defecto | Descripcion |
|---|---|---|---|
| `solucion_ids` | `Sequence[Sequence[int]]` | — | Solucion como listas de IDs enteros (sin marcador de deposito). Producida por `encode_solution`. |
| `rng` | `random.Random \| None` | `None` | Generador de numeros aleatorios. |
| `operadores` | `Iterable[str]` | `OPERADORES_POPULARES` | Operadores habilitados para la eleccion aleatoria. |
| `usar_gpu` | `bool` | `False` | Placeholder de backend GPU. |
| `encoding` | `SearchEncoding \| None` | `None` | Si se proporciona, rellena `labels_movidos` en el `MovimientoVecindario` mediante `decode_task_ids`. |

**Retorna**: `tuple[list[list[int]], MovimientoVecindario]`
- Vecino como listas de IDs enteros.
- Descriptor del movimiento con `id_movidos` siempre disponible.

### Diferencia entre los backends

| Aspecto | `backend="labels"` | `backend="ids"` |
|---|---|---|
| Tipo de solucion interna | `list[list[str]]` | `list[list[int]]` |
| Conversion necesaria | Ninguna | `encode_solution` + `decode_solution` |
| `id_movidos` disponible | No (tupla vacia) | Si (siempre) |
| `labels_movidos` disponible | No | Si (si se pasa `encoding`) |
| Velocidad comparativa | Linea base | Mas rapido en instancias grandes (comparacion de enteros) |
| Requisito | Solo la solucion | `SearchEncoding` obligatorio |
| Uso tipico | Prototipos, depuracion | Produccion con metaheuristicas |

---

## 5. Ejemplo completo de uso

El siguiente codigo muestra el flujo completo: cargar la instancia, construir el
contexto de evaluacion, generar un vecino y decidir si aceptarlo.

```python
import random

from metacarp.instances import load_instances
from metacarp.cargar_grafos import cargar_objeto_gexf
from metacarp.cargar_soluciones_iniciales import cargar_solucion_inicial
from metacarp.busqueda_indices import build_search_encoding, encode_solution
from metacarp.evaluador_costo import construir_contexto, costo_rapido
from metacarp.vecindarios import (
    generar_vecino,
    OPERADORES_POPULARES,
    MovimientoVecindario,
)

# --- 1. Cargar la instancia ---
nombre = "gdb1"
data = load_instances(nombre)
G = cargar_objeto_gexf(nombre)
inicial_obj = cargar_solucion_inicial(nombre)

# 'inicial_obj' puede ser un dict con varias soluciones candidatas;
# usamos la primera disponible como punto de partida.
if isinstance(inicial_obj, dict):
    solucion_actual = next(iter(inicial_obj.values()))
else:
    solucion_actual = inicial_obj

# --- 2. Construir el contexto de evaluacion (solo una vez) ---
ctx = construir_contexto(data, G=G)

# --- 3. Construir el encoding para el backend ids (opcional pero recomendado) ---
encoding = build_search_encoding(data)

# --- 4. Generador de numeros aleatorios reproducible ---
rng = random.Random(42)

# --- 5. Evaluar la solucion actual ---
costo_actual = costo_rapido(solucion_actual, ctx)
print(f"Costo inicial: {costo_actual:.2f}")

# --- 6. Generar un vecino con el backend 'labels' ---
vecino, movimiento = generar_vecino(
    solucion_actual,
    rng=rng,
    operadores=OPERADORES_POPULARES,
    marcador_depot="D",
    devolver_con_deposito=True,
    backend="labels",
    encoding=encoding,   # rellena labels_movidos en el MovimientoVecindario
)

print(f"Operador aplicado: {movimiento.operador}")
print(f"Ruta afectada (ruta_a): {movimiento.ruta_a}")

# --- 7. Evaluar el vecino ---
costo_vecino = costo_rapido(vecino, ctx)
print(f"Costo del vecino: {costo_vecino:.2f}")

# --- 8. Aceptar el vecino si mejora (greedy) ---
if costo_vecino < costo_actual:
    solucion_actual = vecino
    costo_actual = costo_vecino
    print("Vecino aceptado.")
else:
    print("Vecino rechazado (no mejora).")

# --- Variante con backend 'ids' (mas rapido) ---
rutas_ids = encode_solution(solucion_actual, encoding)

from metacarp.vecindarios import generar_vecino_ids
from metacarp.busqueda_indices import decode_solution

vecino_ids, mov_ids = generar_vecino_ids(
    rutas_ids,
    rng=rng,
    operadores=OPERADORES_POPULARES,
    encoding=encoding,
)

# Decodificar de vuelta a etiquetas con deposito
vecino_labels = decode_solution(vecino_ids, encoding, con_deposito=True)
print(f"Tareas movidas (IDs): {mov_ids.id_movidos}")
print(f"Tareas movidas (etiquetas): {mov_ids.labels_movidos}")
```

---

## 6. Integración con las metaheurísticas

Todas las metaheuristicas del paquete llaman a `generar_vecino` con la misma firma,
variando solo los parametros de control. La siguiente tabla resume como cada una lo
usa.

### Parametros tipicos de `generar_vecino` por metaheuristica

| Metaheuristica | Modulo | `iteraciones` (u.e.) | `backend` tipico | Operadores activos | Genera vecinos en lote |
|---|---|---|---|---|---|
| Busqueda Tabu | `busqueda_tabu.py` | 400 | `"labels"` (por defecto) | Todos (`OPERADORES_POPULARES`) | Si (`tam_vecindario=25` por iteracion) |
| Abejas (ABC) | `abejas.py` | 250 | `"labels"` (por defecto) | Todos (`OPERADORES_POPULARES`) | Si (`num_fuentes=16` empleadas + observadoras) |
| Cuckoo Search | `cuckoo_search.py` | 260 | `"labels"` (por defecto) | Todos (`OPERADORES_POPULARES`) | Si (`num_nidos=20` cuckoos) |
| Recocido Simulado (SA) | `recocido_simulado.py` | `L = n²` iteraciones por nivel (adaptativo a la instancia) | `"labels"` (por defecto) | Todos (`OPERADORES_POPULARES`) con selección por dado (`p_inter`) | No (1 vecino por evaluacion) |

### Detalle por metaheuristica

**Busqueda Tabu** (`busqueda_tabu`):
- Genera `tam_vecindario` (por defecto 25) vecinos por iteracion llamando a
  `generar_vecino` en un bucle.
- Evalua el lote completo con `costo_lote_penalizado_ids`.
- Elige el mejor vecino no tabu (o el mejor tabu si cumple el criterio de aspiracion).
- Registra el movimiento elegido en el diccionario `tabu_hasta` por `tenure_tabu`
  iteraciones.
- La clave tabu se construye a partir de `(mov.operador, mov.ruta_a, mov.ruta_b,
  mov.i, mov.j, mov.k, mov.l, tuple(mov.labels_movidos))`.

**Abejas (ABC)** (`busqueda_abejas`):
- Mantiene `num_fuentes` (por defecto 16) soluciones activas en paralelo.
- **Fase empleadas**: genera 1 vecino por fuente (lote de `num_fuentes`).
- **Fase observadoras**: selecciona fuentes con probabilidad proporcional a su calidad
  y genera otro lote de `num_fuentes` vecinos.
- **Fase scout**: fuentes que no mejoraron en `limite_abandono` (por defecto 35)
  intentos consecutivos se reemplazan con vecinos de la mejor fuente.

**Cuckoo Search** (`cuckoo_search`):
- Mantiene `num_nidos` (por defecto 20) soluciones activas.
- Por iteracion genera 1 "cuckoo" (vecino perturbado) por nido usando multiples
  llamadas consecutivas a `generar_vecino` (vuelo de Levy discreto).
- Los `floor(pa_abandono * num_nidos)` (por defecto `floor(0.25 * 20) = 5`) peores
  nidos se abandonan y reemplazan con vecinos del mejor nido.

**Recocido Simulado** (`recocido_simulado`):
- Genera 1 vecino por evaluacion.
- Acepta siempre si mejora; si empeora, acepta con probabilidad `exp(-delta/T)`.
- `T` decrece geometricamente por factor `alpha` en cada nivel; la condicion de parada es `T < temperatura_minima`.
- La longitud de la cadena de Markov es `L = n²`, donde `n` es el numero de arcos requeridos de la instancia.
- `temperatura_inicial` y `temperatura_minima` pueden fijarse manualmente o calcularse de forma automatica desde la instancia (`temperatura_inicial = 20 · d_max / n`, `temperatura_minima = 20 · d_max / n²`).
- **Mecanismo de dado (`p_inter`)**: antes de elegir el operador en cada iteracion, se lanza un numero aleatorio en `[0,1)`. Si es menor que `p_efectiva` (= `alpha_inter = 0.8` cuando la solucion viola capacidad, o `p_inter` cuando es factible), se elige del grupo inter-ruta; en caso contrario, del grupo intra-ruta. Esto equilibra diversificacion (entre rutas) e intensificacion (dentro de ruta).

---

## 7. Referencia rápida

### Tabla resumen de los 9 operadores

| Nombre | Tipo | Complejidad del movimiento | Cuándo preferirlo |
|---|---|---|---|
| `relocate_intra` | Intra-ruta | Mueve 1 tarea dentro de la misma ruta. Requiere >= 2 tareas en la ruta. | Cuando una tarea especifica esta en una posicion ineficiente y moverla reduce deadheading. |
| `swap_intra` | Intra-ruta | Intercambia 2 tareas dentro de la misma ruta. Requiere >= 2 tareas. | Cuando el orden de dos tareas especificas no es optimo y el resto de la ruta es aceptable. |
| `2opt_intra` | Intra-ruta | Invierte un segmento de >= 2 tareas dentro de la misma ruta. Requiere >= 3 tareas. | Cuando la ruta se "cruza" a si misma y servir un subsegmento en sentido inverso lo corrige. |
| `relocate_inter` | Inter-ruta | Mueve 1 tarea de una ruta a otra. Puede vaciar la ruta origen. | Para reequilibrar carga entre rutas o acercar una tarea a sus vecinas geograficas. |
| `swap_inter` | Inter-ruta | Intercambia 1 tarea de cada ruta. Ambas rutas conservan su cardinalidad. | Para redistribuir tareas entre rutas sin cambiar su tamano. Menos disruptivo que relocate. |
| `2opt_star` | Inter-ruta | Intercambia las colas de dos rutas desde un punto de corte. | Cuando dos rutas se cruzan en su segunda mitad y el intercambio de colas elimina el cruce. |
| `cross_exchange` | Inter-ruta | Intercambia un segmento de tareas de cada ruta. Los segmentos pueden tener tamanos distintos. | Para mover bloques de tareas adyacentes entre rutas. El operador mas disruptivo; util para escapar de minimos locales profundos. |
| `or_opt_2` | Inter-ruta | Mueve un bloque de 2 tareas consecutivas de una ruta a otra. Requiere >= 2 tareas en la ruta origen. | Cuando un par de tareas consecutivas tiene mejor afinidad geografica con otra ruta. Mas potente que relocate_inter al mover el bloque completo. |
| `or_opt_3` | Inter-ruta | Mueve un bloque de 3 tareas consecutivas de una ruta a otra. Requiere >= 3 tareas en la ruta origen. | Cuando tres tareas consecutivas forman un bloque geograficamente compacto que encaja mejor en otra ruta. El mas disruptivo de la familia Or-opt. |

### Tabla de requisitos minimos por operador

| Nombre | Rutas no vacias requeridas | Tareas minimas por ruta |
|---|---|---|
| `relocate_intra` | 1 (la misma) | >= 2 en `ruta_a` |
| `swap_intra` | 1 (la misma) | >= 2 en `ruta_a` |
| `2opt_intra` | 1 (la misma) | >= 3 en `ruta_a` |
| `relocate_inter` | 1 no vacia (origen), 1 cualquiera (destino) | >= 1 en `ruta_a`, >= 0 en `ruta_b` |
| `swap_inter` | 2 no vacias distintas | >= 1 en `ruta_a` y `ruta_b` |
| `2opt_star` | 2 no vacias distintas | >= 1 en `ruta_a` y `ruta_b` |
| `cross_exchange` | 2 con al menos 2 tareas cada una | >= 2 en `ruta_a` y `ruta_b` |
| `or_opt_2` | 1 no vacia (origen), 1 cualquiera (destino) | >= 2 en `ruta_a` |
| `or_opt_3` | 1 no vacia (origen), 1 cualquiera (destino) | >= 3 en `ruta_a` |

Si los requisitos no se cumplen para el operador elegido aleatoriamente, la funcion
`generar_vecino` (y `generar_vecino_ids`) reintenta con otro operador. El limite de
reintentos es 500; si se supera, se lanza `RuntimeError`.
