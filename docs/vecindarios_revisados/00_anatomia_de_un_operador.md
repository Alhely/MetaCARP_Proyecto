# Anatomía de un operador de vecindario

Todos los operadores de `vecindarios.py` comparten la misma infraestructura de soporte. Este documento describe esas piezas comunes una sola vez para no repetirlas en cada documento de operador.

---

## Tabla de contenidos

1. [Contexto: qué ve el operador](#1-contexto-qué-ve-el-operador)
2. [`_copy_solution` — copia profunda obligatoria](#2-_copy_solution--copia-profunda-obligatoria)
3. [`normalizar_para_vecindario` — eliminar el marcador de depósito](#3-normalizar_para_vecindario--eliminar-el-marcador-de-depósito)
4. [`desnormalizar_con_deposito` — restaurar el marcador de depósito](#4-desnormalizar_con_deposito--restaurar-el-marcador-de-depósito)
5. [Bucle de reintentos en `generar_vecino` y `generar_vecino_ids`](#5-bucle-de-reintentos-en-generar_vecino-y-generar_vecino_ids)
6. [`MovimientoVecindario` — descriptor inmutable del movimiento](#6-movimientovecindario--descriptor-inmutable-del-movimiento)
7. [`_moved_ids` — extracción de tareas desplazadas](#7-_moved_ids--extracción-de-tareas-desplazadas)

---

## 1. Contexto: qué ve el operador

Una solución CARP en el paquete tiene el formato `list[list[str]]`, con el marcador de depósito `"D"` al inicio y al final de cada ruta:

```python
solucion = [
    ["D", "TR1", "TR3", "TR5", "D"],
    ["D", "TR2", "TR4", "D"],
]
```

**Los operadores `op_*` nunca reciben este formato.** Reciben la solución ya normalizada —sin `"D"`— producida por `normalizar_para_vecindario`. El ciclo completo de transformaciones es:

```
solucion con "D"
      │
      ▼  normalizar_para_vecindario()
rutas sin "D"   ──► _copy_solution() ──► movimiento puro ──► vecino sin "D"
                                                                    │
                                                                    ▼  desnormalizar_con_deposito()
                                                             vecino con "D"
```

---

## 2. `_copy_solution` — copia profunda obligatoria

**Referencia:** [`vecindarios.py:158–161`](../../metacarp/vecindarios.py#L158)

```python
def _copy_solution(sol: Sequence[Sequence[Hashable]]) -> list[list[str]]:
    return [[x for x in r] for r in sol]
```

### Por qué es imprescindible

Python pasa listas por referencia. Si un operador hiciera `sol[r].pop(i)` directamente sobre la solución recibida, estaría modificando la solución original que el algoritmo (Búsqueda Tabú, Recocido Simulado, etc.) mantiene como estado actual. El error resultante sería silencioso y extremadamente difícil de depurar: la solución "actual" cambiaría de forma inadvertida cada vez que se genera un vecino.

`_copy_solution` construye una **nueva lista de listas**: cada fila es una nueva lista independiente con los mismos elementos. Modificar `s[r]` dentro del operador no afecta en absoluto a `sol`.

### Lo que hace exactamente

La comprensión de lista `[[x for x in r] for r in sol]` es funcionalmente equivalente a `[list(r) for r in sol]`. Genera una copia superficial de cada ruta: los elementos `x` son strings, tipos inmutables en Python, por lo que copiar la referencia al string es suficiente —no hay riesgo de aliasing en los elementos individuales.

---

## 3. `normalizar_para_vecindario` — eliminar el marcador de depósito

**Referencia:** [`vecindarios.py:111–127`](../../metacarp/vecindarios.py#L111)

```python
def normalizar_para_vecindario(
    solucion: Sequence[Sequence[Hashable]],
    *,
    marcador_depot: str = "D",
) -> list[list[str]]:
    out: list[list[str]] = []
    for ruta in solucion:
        fila = [str(x).strip() for x in ruta
                if str(x).strip() and not _is_depot_token(x, marcador_depot)]
        out.append(fila)
    return out
```

### Transformación

```
Entrada:  [["D","TR1","TR3","D"], ["D","TR2","D"]]
Salida:   [["TR1","TR3"],         ["TR2"]]
```

La comparación usa `_is_depot_token`, que convierte a mayúsculas y elimina espacios antes de comparar. Esto hace que `"d"`, `"  D  "` y `"D"` se traten igual.

### Nota sobre rutas vacías

Si una ruta solo contiene `"D"` (p. ej. `["D", "D"]`), la normalización produce una lista vacía `[]`. Esto es intencional: `_rutas_con_indices` filtra después estas rutas vacías para que los operadores no intenten operar sobre ellas.

---

## 4. `desnormalizar_con_deposito` — restaurar el marcador de depósito

**Referencia:** [`vecindarios.py:139–147`](../../metacarp/vecindarios.py#L139)

```python
def desnormalizar_con_deposito(
    rutas: Sequence[Sequence[Hashable]],
    *,
    marcador_depot: str = "D",
) -> list[list[str]]:
    md = str(marcador_depot).strip().upper() or "D"
    return [[md, *[str(x).strip() for x in r], md] for r in rutas]
```

### Transformación

```
Entrada:  [["TR3","TR1"], ["TR2","TR4"]]
Salida:   [["D","TR3","TR1","D"], ["D","TR2","TR4","D"]]
```

Se aplica a **todas** las rutas, incluidas las que quedaron vacías tras el movimiento. Una ruta vacía `[]` se convierte en `["D", "D"]`, lo que representa un vehículo que parte y regresa al depósito sin servir ninguna tarea.

Esta función solo se invoca cuando `devolver_con_deposito=True` (valor por defecto de `generar_vecino`). En `generar_vecino_ids` no se llama, porque la representación indexada no incluye el marcador de depósito.

---

## 5. Bucle de reintentos en `generar_vecino` y `generar_vecino_ids`

**Referencia:** [`vecindarios.py:1014–1018`](../../metacarp/vecindarios.py#L1014) (modo labels) y [`vecindarios.py:769–772`](../../metacarp/vecindarios.py#L769) (modo ids).

```python
intentos = 0
while True:
    intentos += 1
    if intentos > 500:
        raise RuntimeError(
            "No se pudo generar un vecino: solución demasiado pequeña para los operadores."
        )
    op = rng.choice(ops)
    ...
    if not cand:
        continue   # <-- reintenta con otro operador
    ...
    if i == j:
        continue   # <-- reintenta con otros índices
    ...
    # Movimiento válido encontrado: rompe el bucle al llegar al return
    return vec, mov
```

### Por qué no se devuelve `null`

En el pseudocódigo original de la tesis aparecía `Return null` o `Return null` como salida cuando un operador no era aplicable. En el código real esto nunca ocurre: el bucle simplemente intenta con otro operador o con nuevos índices. El único caso en que la función termina con error es cuando se superan los 500 intentos, lo que indica una solución degenerada imposible de explorar con los operadores configurados.

### Causas de `continue`

| Causa | Ejemplo |
|---|---|
| Ninguna ruta activa (`activos` vacío) | Todas las rutas están vacías (caso patológico). |
| Ninguna ruta candidata para el operador | `2opt_intra` necesita `>= 3` tareas; ninguna ruta cumple. |
| Índices iguales | `relocate_intra` sortea `i == j`; el movimiento no generaría un vecino distinto. |
| Caso intra sin margen para `or_opt_k` | `or_opt_2` con una sola ruta de exactamente 2 tareas; tras extraer el bloque no quedaría posición distinta. |

---

## 6. `MovimientoVecindario` — descriptor inmutable del movimiento

**Referencia:** [`vecindarios.py:69–83`](../../metacarp/vecindarios.py#L69)

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

### Campos por operador

| Campo | Intra (relocate/swap/2opt) | Inter (relocate/swap) | 2opt_star | cross_exchange | or_opt_2 / or_opt_3 |
|---|---|---|---|---|---|
| `ruta_a` | ruta única | ruta origen | ruta A | ruta A | ruta origen |
| `ruta_b` | `None` | ruta destino | ruta B | ruta B | ruta destino |
| `i` | pos. origen / inicio seg. | pos. en ruta_a | `cut_a` | inicio seg. en A | inicio bloque en ruta_a |
| `j` | pos. destino / fin seg. | pos. en ruta_b | `cut_b` | fin seg. en A | pos. inserción en ruta_b |
| `k` | `None` | `None` | `None` | inicio seg. en B | tamaño del bloque (2 o 3) |
| `l` | `None` | `None` | `None` | fin seg. en B | `None` |

> **Nota:** Para `or_opt_2` y `or_opt_3`, el campo `k` almacena el tamaño del bloque (`2` o `3`), no un índice en `ruta_b`. Esto permite que `_moved_ids` recupere el bloque desplazado como `rutas[ruta_a][i : i+k]`.

### Por qué `frozen=True` importa para Búsqueda Tabú

La lista tabú de `busqueda_tabu.py` usa `MovimientoVecindario` como clave de diccionario. Esto solo es posible porque el objeto es **hashable**: `frozen=True` hace que el dataclass genere automáticamente `__hash__` a partir de todos sus campos. Si el objeto fuera mutable, no podría usarse como clave.

### `backend_real` siempre es `"cpu"`

El flag `usar_gpu=True` existe en la firma de `generar_vecino` y `generar_vecino_ids`, pero el backend GPU es un **placeholder**: la función `_aplicar_backend_gpu_placeholder` ([vecindarios.py:725](../../metacarp/vecindarios.py#L725)) siempre devuelve `backend_real = "cpu"`. El campo `backend_solicitado` registra lo que pidió el llamador, y `backend_real` registra lo que realmente ejecutó.

---

## 7. `_moved_ids` — extracción de tareas desplazadas

**Referencia:** [`vecindarios.py:661–714`](../../metacarp/vecindarios.py#L661)

Esta función privada recibe la solución en formato de IDs enteros **antes de aplicar el movimiento** y el `MovimientoVecindario` resultante, y devuelve una tupla con los IDs de las tareas que cambiaron de posición o de ruta.

```python
def _moved_ids(op: str, rutas: Sequence[Sequence[int]],
               mov: MovimientoVecindario) -> tuple[int, ...]:
    ...
```

La lógica varía por operador:

| Operador | Qué devuelve |
|---|---|
| `relocate_intra` | La tarea en `rutas[ruta_a][i]` (solo la tarea movida). |
| `swap_intra` / `2opt_intra` | Todas las tareas del rango `[min(i,j) .. max(i,j)]`. |
| `relocate_inter` | La tarea en `rutas[ruta_a][i]`. |
| `swap_inter` | Las tareas en `rutas[ruta_a][i]` y `rutas[ruta_b][j]`. |
| `2opt_star` | Las colas completas: `rutas[ruta_a][cut_a+1:]` y `rutas[ruta_b][cut_b+1:]`. |
| `cross_exchange` | Ambos segmentos: `rutas[ruta_a][i:j+1]` y `rutas[ruta_b][k:l+1]`. |
| `or_opt_2` / `or_opt_3` | El bloque: `rutas[ruta_a][i : i+k]` donde `k` es el tamaño del bloque. |

`_moved_ids` solo se llama en `generar_vecino_ids`. En el modo `labels` de `generar_vecino`, `id_movidos` y `labels_movidos` quedan como tuplas vacías a menos que se pase un `encoding`.
