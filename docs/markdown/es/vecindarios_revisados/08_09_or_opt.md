# `op_or_opt_2` y `op_or_opt_3` — Reubicación de bloque inter-ruta (Or-opt)

**Familia:** inter-ruta | **Archivo fuente:**
- `op_or_opt_2`: [`vecindarios.py:433–452`](../../metacarp/vecindarios.py#L433)
- `op_or_opt_3`: [`vecindarios.py:470–488`](../../metacarp/vecindarios.py#L470)
- Lógica genérica `_or_opt_k`: [`vecindarios.py:507–579`](../../metacarp/vecindarios.py#L507)
- Selección de índices en `generar_vecino`: [`vecindarios.py:1119–1153`](../../metacarp/vecindarios.py#L1119)

---

## 1. Qué hacen estos operadores

Ambos operadores forman la familia **Or-opt**: extraen un bloque de tareas **consecutivas** de una ruta origen y lo insertan —en el mismo orden, sin invertir— en otra posición de una ruta destino.

- `op_or_opt_2`: bloque de **2** tareas consecutivas.
- `op_or_opt_3`: bloque de **3** tareas consecutivas.

La diferencia entre `or_opt_k` y `cross_exchange` es que aquí el bloque se mueve sin intercambiarse por otro segmento: la ruta destino solo recibe el bloque, no entrega nada a cambio.

**Cuándo conviene:** cuando un par o trío de tareas consecutivas tiene mejor afinidad geográfica con las tareas de otra ruta. El movimiento bloque-completo puede reducir el deadheading en ambas rutas simultáneamente. Es el operador Or-opt clásico (Tsitsiklis, 1992) extendido al dominio multi-ruta.

---

## 2. Código Python real de los operadores puros

### `op_or_opt_2` — [`vecindarios.py:433–452`](../../metacarp/vecindarios.py#L433)

```python
def op_or_opt_2(
    sol: Sequence[Sequence[Hashable]],
    ra: int,   # Índice de la ruta origen
    i: int,    # Inicio del bloque de 2 tareas en la ruta origen (toma [i, i+1])
    rb: int,   # Índice de la ruta destino
    j: int,    # Posición de inserción en la ruta destino
) -> list[list[str]]:
    """
    Or-opt (k=2): mueve un bloque de 2 tareas consecutivas de ra[i..i+1] a rb[j].
    El bloque se inserta en el mismo orden (no se invierte).
    """
    s = _copy_solution(sol)
    # Extrae el bloque de 2 tareas consecutivas comenzando en la posición i
    bloque = s[ra][i : i + 2]
    # Elimina el bloque de la ruta origen
    del s[ra][i : i + 2]
    # Inserta cada elemento del bloque en orden a partir de la posición j
    s[rb][j:j] = bloque
    return s
```

### `op_or_opt_3` — [`vecindarios.py:470–488`](../../metacarp/vecindarios.py#L470)

```python
def op_or_opt_3(
    sol: Sequence[Sequence[Hashable]],
    ra: int,
    i: int,
    rb: int,
    j: int,
) -> list[list[str]]:
    """
    Or-opt (k=3): mueve un bloque de 3 tareas consecutivas de ra[i..i+2] a rb[j].
    El bloque se inserta en el mismo orden (no se invierte).
    """
    s = _copy_solution(sol)
    bloque = s[ra][i : i + 3]
    del s[ra][i : i + 3]
    s[rb][j:j] = bloque
    return s
```

### Nota sobre la asignación de slice `s[rb][j:j] = bloque`

La expresión `s[rb][j:j]` selecciona un slice **vacío** en la posición `j`. Asignarle `bloque` inserta todos los elementos de `bloque` en la posición `j` sin sobreescribir ningún elemento existente. Es la forma idiomática de Python para insertar una sublista completa en una posición arbitraria sin usar un bucle.

---

## 3. La función genérica `_or_opt_k`

[`vecindarios.py:507–579`](../../metacarp/vecindarios.py#L507)

`_or_opt_k` generaliza la lógica de selección aleatoria para un tamaño de bloque `k` arbitrario. Es la función que realmente implementa la lógica de candidatas, sorteos y caso especial intra. Los operadores `op_or_opt_2` y `op_or_opt_3` son los movimientos puros; `_or_opt_k` es el orquestador.

```python
def _or_opt_k(
    solucion: Sequence[Sequence[Hashable]],
    k: int,
    rng: random.Random,
    marcador_depot: str = "D",
) -> tuple[list[list[str]], MovimientoVecindario]:
    rutas = normalizar_para_vecindario(solucion, marcador_depot=marcador_depot)
    op_nombre = f"or_opt_{k}"

    cand_origen = [idx for idx, r in enumerate(rutas) if len(r) >= k]
    if not cand_origen:
        return (
            desnormalizar_con_deposito(rutas, marcador_depot=marcador_depot),
            MovimientoVecindario(operador=op_nombre),
        )

    ra = rng.choice(cand_origen)
    destinos = [x for x in range(len(rutas)) if x != ra]
    if not destinos:
        destinos = [ra]
    rb = rng.choice(destinos)

    na = len(rutas[ra])
    i = rng.randrange(0, na - k + 1)

    if rb == ra:
        tam_destino_post = na - k
        if tam_destino_post <= 0:
            return (
                desnormalizar_con_deposito(rutas, marcador_depot=marcador_depot),
                MovimientoVecindario(operador=op_nombre),
            )
        j = rng.randrange(0, tam_destino_post + 1)
        if j == i:
            j = (j + 1) % (tam_destino_post + 1)
    else:
        j = rng.randrange(0, len(rutas[rb]) + 1)

    if k == 2:
        vec = op_or_opt_2(rutas, ra, i, rb, j)
    elif k == 3:
        vec = op_or_opt_3(rutas, ra, i, rb, j)
    else:
        s = _copy_solution(rutas)
        bloque = s[ra][i : i + k]
        del s[ra][i : i + k]
        s[rb][j:j] = bloque
        vec = s

    mov = MovimientoVecindario(operador=op_nombre, ruta_a=ra, ruta_b=rb, i=i, j=j, k=k)
    return desnormalizar_con_deposito(vec, marcador_depot=marcador_depot), mov
```

> **Nota:** `_or_opt_k` se llama cuando el código externo invoca a `_aplicar_or_opt_2` o `_aplicar_or_opt_3` directamente. El camino habitual en metaheurísticas pasa por `generar_vecino`, que tiene su propio bloque de selección de índices equivalente (descrito en la sección 4).

---

## 4. Bloque de selección de índices en `generar_vecino`

[`vecindarios.py:1119–1153`](../../metacarp/vecindarios.py#L1119)

```python
elif op in ("or_opt_2", "or_opt_3"):
    k_blk = 2 if op == "or_opt_2" else 3
    cand_origen = [x for x in activos if len(rutas[x]) >= k_blk]
    if not cand_origen:
        continue
    ra = rng.choice(cand_origen)
    destinos = [x for x in range(len(rutas)) if x != ra]
    if not destinos:
        destinos = [ra]
    rb = rng.choice(destinos)
    na = len(rutas[ra])
    i = rng.randrange(0, na - k_blk + 1)
    if rb == ra:
        tam_destino_post = na - k_blk
        if tam_destino_post <= 0:
            continue
        j = rng.randrange(0, tam_destino_post + 1)
        if j == i:
            continue
    else:
        j = rng.randrange(0, len(rutas[rb]) + 1)
    if k_blk == 2:
        vec = op_or_opt_2(rutas, ra, i, rb, j)
    else:
        vec = op_or_opt_3(rutas, ra, i, rb, j)
    mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=i, j=j, k=k_blk)
```

El bloque equivalente en `generar_vecino_ids` está en [`vecindarios.py:877–919`](../../metacarp/vecindarios.py#L877).

### Distribuciones de los sorteos

| Sorteo | Llamada | Distribución |
|---|---|---|
| Ruta origen `ra` | `rng.choice(cand_origen)` | Uniforme discreta sobre rutas con `>= k` tareas. |
| Ruta destino `rb` | `rng.choice(destinos)` | Uniforme discreta sobre rutas distintas a `ra`; si solo hay una ruta, `destinos = [ra]`. |
| Inicio del bloque `i` | `rng.randrange(0, na - k + 1)` | Uniforme en `{0, ..., na-k}`. Garantiza que caben `k` tareas desde `i`. |
| Posición de inserción `j` (inter) | `rng.randrange(0, len(rutas[rb]) + 1)` | Uniforme en `{0, ..., len(rb)}`. El `+1` permite insertar al final. |
| Posición de inserción `j` (intra) | `rng.randrange(0, tam_destino_post + 1)` | Uniforme en `{0, ..., na-k}` (posiciones válidas tras extraer el bloque). |

### Campo `k` en `MovimientoVecindario`

Para `or_opt_2` y `or_opt_3`, el campo `k` del `MovimientoVecindario` **no** es el inicio de un segmento en `ruta_b` (como en `cross_exchange`), sino el **tamaño del bloque** desplazado (`2` o `3`). Esto permite que `_moved_ids` recupere el bloque exacto como `rutas[ruta_a][i : i+k]`.

### Caso intra: `rb == ra`

Si la solución tiene una sola ruta (caso patológico), `destinos` queda vacío y se fuerza `destinos = [ra]`, permitiendo la inserción en la misma ruta. En este caso el código verifica dos condiciones adicionales:

1. `tam_destino_post = na - k_blk`. Si `tam_destino_post <= 0`, tras extraer el bloque no queda espacio para reinsertar en una posición distinta: se ejecuta `continue`.
2. `if j == i: continue`. Si la posición de inserción coincide con la de extracción, el resultado sería idéntico a la solución original.

---

## 5. Traza paso a paso — `or_opt_2`

**Solución inicial:**

```
solucion = [
    ["D", "TR1", "TR2", "TR3", "TR4", "D"],
    ["D", "TR5", "TR6", "D"],
]
```

**Parámetros elegidos:** `ra = 0`, `i = 1`, `rb = 1`, `j = 1` (bloque de 2 tareas)

**Paso 1 — Normalizar (quitar `"D"`):**

```
rutas = [
    ["TR1", "TR2", "TR3", "TR4"],   # ruta 0 (ra)
    ["TR5", "TR6"],                  # ruta 1 (rb)
]
```

**Paso 2 — Copiar (`_copy_solution`):**

```
s = [
    ["TR1", "TR2", "TR3", "TR4"],
    ["TR5", "TR6"],
]
```

**Paso 3 — Extraer bloque de 2: `bloque = s[0][1:3]` → `["TR2", "TR3"]`:**

**Paso 4 — Eliminar bloque de ruta origen: `del s[0][1:3]`:**

```
s = [
    ["TR1", "TR4"],
    ["TR5", "TR6"],
]
```

**Paso 5 — Insertar bloque en ruta destino: `s[1][1:1] = ["TR2", "TR3"]`:**

```
vec (normalizado) = [
    ["TR1", "TR4"],
    ["TR5", "TR2", "TR3", "TR6"],
]
```

**Paso 6 — Registrar movimiento:**

```python
mov = MovimientoVecindario("or_opt_2", ruta_a=0, ruta_b=1, i=1, j=1, k=2)
```

**Paso 7 — Desnormalizar (restaurar `"D"`):**

```
vecino final = [
    ["D", "TR1", "TR4", "D"],
    ["D", "TR5", "TR2", "TR3", "TR6", "D"],
]
```

**Diagrama antes/después:**

```
Antes:
  ruta 0 (ra): ["TR1", "TR2", "TR3", "TR4"]
                        |──bloque──|
  ruta 1 (rb): ["TR5", "TR6"]

Después:
  ruta 0:      ["TR1", "TR4"]
  ruta 1:      ["TR5", "TR2", "TR3", "TR6"]
                        |──bloque──|
```

---

## 6. Traza paso a paso — `or_opt_3`

**Solución inicial:**

```
solucion = [
    ["D", "TR1", "TR2", "TR3", "TR4", "TR5", "D"],
    ["D", "TR6", "TR7", "D"],
]
```

**Parámetros elegidos:** `ra = 0`, `i = 0`, `rb = 1`, `j = 0` (bloque de 3 tareas)

**Paso 1 — Normalizar:**

```
rutas = [
    ["TR1", "TR2", "TR3", "TR4", "TR5"],
    ["TR6", "TR7"],
]
```

**Paso 2 — Copiar:**

```
s = [["TR1", "TR2", "TR3", "TR4", "TR5"], ["TR6", "TR7"]]
```

**Paso 3 — Extraer bloque de 3: `bloque = s[0][0:3]` → `["TR1", "TR2", "TR3"]`:**

**Paso 4 — Eliminar bloque: `del s[0][0:3]`:**

```
s = [["TR4", "TR5"], ["TR6", "TR7"]]
```

**Paso 5 — Insertar bloque: `s[1][0:0] = ["TR1", "TR2", "TR3"]`:**

```
vec (normalizado) = [
    ["TR4", "TR5"],
    ["TR1", "TR2", "TR3", "TR6", "TR7"],
]
```

**Paso 6 — Registrar movimiento:**

```python
mov = MovimientoVecindario("or_opt_3", ruta_a=0, ruta_b=1, i=0, j=0, k=3)
```

**Paso 7 — Desnormalizar:**

```
vecino final = [
    ["D", "TR4", "TR5", "D"],
    ["D", "TR1", "TR2", "TR3", "TR6", "TR7", "D"],
]
```

---

## 7. Requisitos y precondiciones reales

| Condición | `or_opt_2` | `or_opt_3` | Razón |
|---|---|---|---|
| Tareas mín. en `ruta_a` (origen) | `>= 2` | `>= 3` | `rng.randrange(0, na - k + 1)` requiere `na >= k`. |
| Tareas mín. en `ruta_b` (destino) | `>= 0` | `>= 0` | El destino puede estar vacío; el bloque se inserta como contenido único. |
| `ra != rb` | preferido | preferido | Si solo hay una ruta, se permite intra con las condiciones adicionales de `j != i` y `tam_destino_post > 0`. |
| Número de rutas no vacías | `>= 1` | `>= 1` | Si no hay rutas en `activos`, el bucle ejecuta `continue`. |

---

## 8. Contraste "Pseudocódigo vs. realidad"

Un pseudocódigo ingenuo mostraría:

```
s' ← Copy(s)
bloque ← r'_a[i : i+k]
del r'_a[i : i+k]
r'_b[j:j] ← bloque
m ← (or_opt_k, a, b, i, j)
Return (s', m)
```

Lo que omitía:

- La normalización previa que elimina `"D"` de todas las rutas.
- El filtrado de candidatas: `cand_origen = [x for x in activos if len(rutas[x]) >= k_blk]` (requisito de `k` tareas mínimas en la ruta origen).
- La lógica de selección de destino: si solo hay una ruta, `destinos = [ra]` (caso intra habilitado).
- El cálculo especial de `j` en el caso intra (`rb == ra`): el número de posiciones válidas cambia tras extraer el bloque, y se verifica `j != i`.
- El `+1` en `rng.randrange(0, len(rutas[rb]) + 1)` para permitir insertar al final de la ruta destino.
- El hecho de que el campo `k` del `MovimientoVecindario` almacena el **tamaño del bloque** (no el inicio de un segmento en `ruta_b`).
- El bucle de hasta 500 reintentos.
- La desnormalización final.

---

## 9. Pseudocódigos corregidos en LaTeX

### `or_opt_2`

```latex
\begin{algorithm}
\DontPrintSemicolon
\caption{\texttt{or\_opt\_2} --- Mover un bloque de 2 tareas consecutivas a otra ruta}
\KwIn{Solución con depósito $s$;\; generador aleatorio $\mathit{rng}$;\; máximo de intentos $T_{\max}=500$}
\KwOut{Vecino con depósito $s'$;\; registro de movimiento $m$}
$\hat{s} \leftarrow \textsc{NormalizarQuitarDepósito}(s)$
\tcp{elimina el token "D" de inicio y fin de cada ruta}
$t \leftarrow 0$\;
\Repeat{movimiento válido encontrado}{
    $t \leftarrow t + 1$\;
    \lIf{$t > T_{\max}$}{\textbf{lanzar} \texttt{RuntimeError}}
    $\mathcal{C} \leftarrow \{k : |\hat{r}_k| \geq 2\}$
    \tcp{rutas candidatas como origen: al menos 2 tareas (tamaño del bloque)}
    \lIf{$\mathcal{C} = \emptyset$}{\textbf{continue}}
    $\mathit{ra} \leftarrow \mathit{rng}.\textsc{choice}(\mathcal{C})$\;
    $\mathcal{D} \leftarrow \{x : x \neq \mathit{ra}\}$
    \tcp{rutas destino distintas a la origen; si solo hay una, $\mathcal{D} = \{\mathit{ra}\}$}
    \lIf{$\mathcal{D} = \emptyset$}{$\mathcal{D} \leftarrow \{\mathit{ra}\}$}
    $\mathit{rb} \leftarrow \mathit{rng}.\textsc{choice}(\mathcal{D})$\;
    $n_a \leftarrow |\hat{r}_{\mathit{ra}}|$\;
    $i \leftarrow \mathit{rng}.\textsc{randrange}(0,\, n_a - 2 + 1)$
    \tcp{inicio del bloque; garantiza que caben 2 tareas desde $i$}
    \eIf{$\mathit{rb} = \mathit{ra}$}{
        $\tau \leftarrow n_a - 2$
        \tcp{posiciones disponibles tras extraer el bloque}
        \lIf{$\tau \leq 0$}{\textbf{continue}}
        $j \leftarrow \mathit{rng}.\textsc{randrange}(0,\, \tau + 1)$\;
        \lIf{$j = i$}{\textbf{continue} \tcp*{no genera vecino distinto}}
    }{
        $j \leftarrow \mathit{rng}.\textsc{randrange}(0,\, |\hat{r}_{\mathit{rb}}| + 1)$
        \tcp{posición de inserción; $+1$ permite insertar al final}
    }
    $\hat{s}' \leftarrow \textsc{CopiaProfunda}(\hat{s})$
    \tcp{Python pasa listas por referencia; sin copia se corrompería $\hat{s}$}
    $\mathit{bloque} \leftarrow \hat{r}'_{\mathit{ra}}[i\,{:}\,i{+}2]$\;
    $\text{eliminar } \hat{r}'_{\mathit{ra}}[i\,{:}\,i{+}2]$\;
    $\hat{r}'_{\mathit{rb}}[j\,{:}\,j] \leftarrow \mathit{bloque}$
    \tcp{inserción sin sobreescribir: slice vacío $[j:j]$ abre espacio en la posición $j$}
    $m \leftarrow \textsc{MovimientoVecindario}(\texttt{or\_opt\_2},\; \mathit{ruta\_a}=\mathit{ra},\; \mathit{ruta\_b}=\mathit{rb},\; i=i,\; j=j,\; k=2)$\;
    $s' \leftarrow \textsc{RestaurarDepósito}(\hat{s}')$
    \tcp{añade "D" al inicio y fin de cada ruta antes de devolver}
    \Return $(s',\, m)$\;
}
\end{algorithm}
```

### `or_opt_3`

```latex
\begin{algorithm}
\DontPrintSemicolon
\caption{\texttt{or\_opt\_3} --- Mover un bloque de 3 tareas consecutivas a otra ruta}
\KwIn{Solución con depósito $s$;\; generador aleatorio $\mathit{rng}$;\; máximo de intentos $T_{\max}=500$}
\KwOut{Vecino con depósito $s'$;\; registro de movimiento $m$}
$\hat{s} \leftarrow \textsc{NormalizarQuitarDepósito}(s)$
\tcp{elimina el token "D" de inicio y fin de cada ruta}
$t \leftarrow 0$\;
\Repeat{movimiento válido encontrado}{
    $t \leftarrow t + 1$\;
    \lIf{$t > T_{\max}$}{\textbf{lanzar} \texttt{RuntimeError}}
    $\mathcal{C} \leftarrow \{k : |\hat{r}_k| \geq 3\}$
    \tcp{rutas candidatas como origen: al menos 3 tareas (tamaño del bloque)}
    \lIf{$\mathcal{C} = \emptyset$}{\textbf{continue}}
    $\mathit{ra} \leftarrow \mathit{rng}.\textsc{choice}(\mathcal{C})$\;
    $\mathcal{D} \leftarrow \{x : x \neq \mathit{ra}\}$
    \tcp{rutas destino distintas a la origen; si solo hay una, $\mathcal{D} = \{\mathit{ra}\}$}
    \lIf{$\mathcal{D} = \emptyset$}{$\mathcal{D} \leftarrow \{\mathit{ra}\}$}
    $\mathit{rb} \leftarrow \mathit{rng}.\textsc{choice}(\mathcal{D})$\;
    $n_a \leftarrow |\hat{r}_{\mathit{ra}}|$\;
    $i \leftarrow \mathit{rng}.\textsc{randrange}(0,\, n_a - 3 + 1)$
    \tcp{inicio del bloque; garantiza que caben 3 tareas desde $i$}
    \eIf{$\mathit{rb} = \mathit{ra}$}{
        $\tau \leftarrow n_a - 3$
        \tcp{posiciones disponibles tras extraer el bloque}
        \lIf{$\tau \leq 0$}{\textbf{continue}}
        $j \leftarrow \mathit{rng}.\textsc{randrange}(0,\, \tau + 1)$\;
        \lIf{$j = i$}{\textbf{continue} \tcp*{no genera vecino distinto}}
    }{
        $j \leftarrow \mathit{rng}.\textsc{randrange}(0,\, |\hat{r}_{\mathit{rb}}| + 1)$
        \tcp{posición de inserción; $+1$ permite insertar al final}
    }
    $\hat{s}' \leftarrow \textsc{CopiaProfunda}(\hat{s})$
    \tcp{Python pasa listas por referencia; sin copia se corrompería $\hat{s}$}
    $\mathit{bloque} \leftarrow \hat{r}'_{\mathit{ra}}[i\,{:}\,i{+}3]$\;
    $\text{eliminar } \hat{r}'_{\mathit{ra}}[i\,{:}\,i{+}3]$\;
    $\hat{r}'_{\mathit{rb}}[j\,{:}\,j] \leftarrow \mathit{bloque}$
    \tcp{inserción sin sobreescribir: slice vacío $[j:j]$ abre espacio en la posición $j$}
    $m \leftarrow \textsc{MovimientoVecindario}(\texttt{or\_opt\_3},\; \mathit{ruta\_a}=\mathit{ra},\; \mathit{ruta\_b}=\mathit{rb},\; i=i,\; j=j,\; k=3)$\;
    $s' \leftarrow \textsc{RestaurarDepósito}(\hat{s}')$
    \tcp{añade "D" al inicio y fin de cada ruta antes de devolver}
    \Return $(s',\, m)$\;
}
\end{algorithm}
```
