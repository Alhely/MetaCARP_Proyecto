# `op_relocate_intra` — Reubicación intra-ruta

**Familia:** intra-ruta | **Archivo fuente:** [`vecindarios.py:182–191`](../../metacarp/vecindarios.py#L182)

---

## 1. Qué hace el movimiento

Extrae la tarea que ocupa la posición `i` de una ruta y la reinserta en la posición `j` de **la misma ruta**. El resto de las tareas se desplazan para cerrar el hueco y abrir el espacio respectivamente.

El multiconjunto total de tareas se conserva: el operador no elimina ni duplica ninguna tarea.

**Cuándo conviene:** cuando una tarea está en una posición ineficiente dentro de la ruta y servirla en otro momento de la secuencia reduce el deadheading (traslado en vacío entre tareas consecutivas).

---

## 2. Código Python real del operador puro

[`vecindarios.py:182–191`](../../metacarp/vecindarios.py#L182)

```python
def op_relocate_intra(sol: Sequence[Sequence[Hashable]], r: int, i: int, j: int) -> list[list[str]]:
    """
    Relocate dentro de una ruta: mueve la tarea en posición i a la posición j.
    Requiere len(ruta) >= 2.
    """
    s = _copy_solution(sol)   # Copia para no alterar la solución original
    ruta = s[r]               # Referencia a la ruta r dentro de la copia
    x = ruta.pop(i)           # Extrae la tarea de la posición i (la lista se acorta)
    ruta.insert(j, x)         # Inserta la tarea extraída en la posición j
    return s
```

El operador recibe la solución **ya normalizada** (sin `"D"`). Internamente llama a `_copy_solution` para no modificar la solución original. Las operaciones `pop(i)` e `insert(j, x)` son de la lista estándar de Python.

---

## 3. Bloque de selección de índices en `generar_vecino`

[`vecindarios.py:1026–1037`](../../metacarp/vecindarios.py#L1026)

```python
if op == "relocate_intra":
    cand = [x for x in activos if len(rutas[x]) >= 2]
    if not cand:
        continue
    r = rng.choice(cand)
    n = len(rutas[r])
    i = rng.randrange(n)
    j = rng.randrange(n)
    if i == j:
        continue
    vec = op_relocate_intra(rutas, r, i, j)
    mov = MovimientoVecindario(op, ruta_a=r, i=i, j=j)
```

El bloque equivalente existe en `generar_vecino_ids` en las líneas [`vecindarios.py:781–793`](../../metacarp/vecindarios.py#L781).

### Distribuciones de los sorteos

| Sorteo | Llamada | Distribución |
|---|---|---|
| Ruta candidata | `rng.choice(cand)` | Uniforme discreta sobre rutas con `>= 2` tareas. |
| Posición origen `i` | `rng.randrange(n)` | Uniforme discreta en `{0, ..., n-1}`. |
| Posición destino `j` | `rng.randrange(n)` | Uniforme discreta en `{0, ..., n-1}`. |

Si `i == j`, el movimiento no generaría un vecino distinto: el código ejecuta `continue` y reintenta el bucle.

---

## 4. Traza paso a paso sobre una solución concreta con depósito

**Solución inicial:**

```
solucion = [
    ["D", "TR1", "TR2", "TR3", "TR4", "D"],
    ["D", "TR5", "TR6", "D"],
]
```

**Parámetros elegidos:** `r = 0`, `i = 0`, `j = 2`

**Paso 1 — Normalizar (quitar `"D"`):**

```
rutas = [
    ["TR1", "TR2", "TR3", "TR4"],   # ruta 0
    ["TR5", "TR6"],                  # ruta 1
]
```

**Paso 2 — Copiar (`_copy_solution`):**

```
s = [
    ["TR1", "TR2", "TR3", "TR4"],   # copia independiente
    ["TR5", "TR6"],
]
```

**Paso 3 — `s[0].pop(0)` → extrae `"TR1"`, ruta queda `["TR2", "TR3", "TR4"]`:**

```
s = [
    ["TR2", "TR3", "TR4"],
    ["TR5", "TR6"],
]
```

**Paso 4 — `s[0].insert(2, "TR1")` → inserta `"TR1"` en posición 2:**

```
vec (normalizado) = [
    ["TR2", "TR3", "TR1", "TR4"],
    ["TR5", "TR6"],
]
```

**Paso 5 — Registrar movimiento:**

```python
mov = MovimientoVecindario("relocate_intra", ruta_a=0, i=0, j=2)
```

**Paso 6 — Desnormalizar (restaurar `"D"`):**

```
vecino final = [
    ["D", "TR2", "TR3", "TR1", "TR4", "D"],
    ["D", "TR5", "TR6", "D"],
]
```

**Diagrama antes/después:**

```
Antes (ruta 0):   ["TR1", "TR2", "TR3", "TR4"]
                        pop(0) ──► "TR1" sale
                  ["TR2", "TR3", "TR4"]
                        insert(2, "TR1")
Después (ruta 0): ["TR2", "TR3", "TR1", "TR4"]
```

---

## 5. Requisitos y precondiciones reales

| Condición | Valor | Razón |
|---|---|---|
| Tareas mínimas en `ruta_a` | `>= 2` | Con solo 1 tarea, `pop(i)` dejaría la ruta vacía y `insert(j, x)` la repondría en la misma posición: no se genera vecino distinto. |
| `i != j` | obligatorio | Si `i == j`, `pop(i)` e `insert(j, x)` producen la ruta idéntica. |
| Número de rutas | `>= 1` | Es un operador intra; solo necesita una ruta. |

Cuando `not cand` (ninguna ruta tiene `>= 2` tareas) o cuando `i == j`, el bucle ejecuta `continue` y elige de nuevo.

---

## 6. Contraste "Pseudocódigo vs. realidad"

El pseudocódigo original de la tesis mostraba:

```
s' ← Copy(s)
e ← r'_a.pop(i)
r'_a.insert(j, e)
m ← (relocate_intra, a, i, j)
Return (s', m)
```

Lo que omitía:

- La normalización previa que elimina `"D"` de todas las rutas antes de que el operador vea la solución.
- El filtrado `cand = [r for r in activos if len(rutas[r]) >= 2]` antes de elegir `r`.
- El sorteo explícito de `i` y `j` con `rng.randrange(n)` y la condición `if i == j: continue`.
- El bucle de hasta 500 reintentos: el pseudocódigo mostraba `Return null` donde el código real hace `continue`.
- La desnormalización final que restaura `"D"` al devolver el vecino.
- La reconstrucción del `MovimientoVecindario` definitivo con `backend_solicitado` y `backend_real`.

---

## 7. Pseudocódigo corregido en LaTeX

```latex
\begin{algorithm}
\DontPrintSemicolon
\caption{\texttt{relocate\_intra} --- Mover una tarea a una posición diferente dentro de la misma ruta}
\KwIn{Solución con depósito $s$;\; generador aleatorio $\mathit{rng}$;\; máximo de intentos $T_{\max}=500$}
\KwOut{Vecino con depósito $s'$;\; registro de movimiento $m$}
$\hat{s} \leftarrow \textsc{NormalizarQuitarDepósito}(s)$
\tcp{elimina el token "D" de inicio y fin de cada ruta; los operadores trabajan solo con etiquetas de tareas}
$t \leftarrow 0$\;
\Repeat{movimiento válido encontrado}{
    $t \leftarrow t + 1$\;
    \lIf{$t > T_{\max}$}{\textbf{lanzar} \texttt{RuntimeError}}
    $\mathcal{C} \leftarrow \{k : |\hat{r}_k| \geq 2\}$
    \tcp{rutas candidatas: al menos 2 tareas (requisito del operador)}
    \lIf{$\mathcal{C} = \emptyset$}{\textbf{continue}}
    $a \leftarrow \mathit{rng}.\textsc{choice}(\mathcal{C})$;\quad $n \leftarrow |\hat{r}_a|$\;
    $i \leftarrow \mathit{rng}.\textsc{randrange}(n)$ \tcp{posición de origen, uniforme en $\{0,\ldots,n-1\}$}
    $j \leftarrow \mathit{rng}.\textsc{randrange}(n)$ \tcp{posición de destino, uniforme en $\{0,\ldots,n-1\}$}
    \lIf{$i = j$}{\textbf{continue} \tcp*{no genera vecino distinto}}
    $\hat{s}' \leftarrow \textsc{CopiaProfunda}(\hat{s})$ \tcp{Python pasa listas por referencia; sin copia se corrompería $\hat{s}$}
    $e \leftarrow \hat{r}'_a.\textsc{pop}(i)$\;
    $\hat{r}'_a.\textsc{insert}(j,\, e)$\;
    $m \leftarrow \textsc{MovimientoVecindario}(\texttt{relocate\_intra},\; \mathit{ruta\_a}=a,\; i=i,\; j=j)$\;
    $s' \leftarrow \textsc{RestaurarDepósito}(\hat{s}')$
    \tcp{añade "D" al inicio y fin de cada ruta antes de devolver}
    \Return $(s',\, m)$\;
}
\end{algorithm}
```
