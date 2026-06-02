# `op_swap_intra` — Intercambio intra-ruta

**Familia:** intra-ruta | **Archivo fuente:** [`vecindarios.py:206–213`](../../metacarp/vecindarios.py#L206)

---

## 1. Qué hace el movimiento

Intercambia directamente las tareas en las posiciones `i` y `j` de **la misma ruta**. Ninguna otra tarea cambia de posición ni de ruta. La longitud de la ruta se mantiene igual.

**Cuándo conviene:** cuando el orden de dos tareas específicas dentro de la misma ruta no es óptimo y su intercambio directo reduce el deadheading entre ellas, sin necesidad de desplazar el bloque completo que las rodea.

---

## 2. Código Python real del operador puro

[`vecindarios.py:206–213`](../../metacarp/vecindarios.py#L206)

```python
def op_swap_intra(sol: Sequence[Sequence[Hashable]], r: int, i: int, j: int) -> list[list[str]]:
    """Swap dentro de una ruta: intercambia posiciones i y j."""
    s = _copy_solution(sol)
    ruta = s[r]
    # Intercambio en una sola línea usando desempaquetado de tuplas de Python:
    # Python primero evalúa el lado derecho completo, luego asigna.
    ruta[i], ruta[j] = ruta[j], ruta[i]
    return s
```

La línea `ruta[i], ruta[j] = ruta[j], ruta[i]` es atómica desde la perspectiva de Python: el lado derecho se evalúa completamente antes de cualquier asignación, lo que evita el problema de sobreescribir `ruta[i]` antes de leer `ruta[j]`.

---

## 3. Bloque de selección de índices en `generar_vecino`

[`vecindarios.py:1039–1047`](../../metacarp/vecindarios.py#L1039)

```python
elif op == "swap_intra":
    cand = [x for x in activos if len(rutas[x]) >= 2]
    if not cand:
        continue
    r = rng.choice(cand)
    n = len(rutas[r])
    i, j = rng.sample(range(n), 2)
    vec = op_swap_intra(rutas, r, i, j)
    mov = MovimientoVecindario(op, ruta_a=r, i=i, j=j)
```

El bloque equivalente en `generar_vecino_ids` está en [`vecindarios.py:795–804`](../../metacarp/vecindarios.py#L795).

### Distribuciones de los sorteos

| Sorteo | Llamada | Distribución |
|---|---|---|
| Ruta candidata | `rng.choice(cand)` | Uniforme discreta sobre rutas con `>= 2` tareas. |
| Par de posiciones `(i, j)` | `rng.sample(range(n), 2)` | Uniforme discreta sin reemplazo sobre pares de `{0, ..., n-1}`. Garantiza `i != j` por construcción. |

A diferencia de `relocate_intra`, aquí se usa `rng.sample(range(n), 2)` en lugar de dos `rng.randrange(n)` independientes. Esto elimina la necesidad de verificar `i != j` con un `continue`: `sample` garantiza dos índices distintos directamente.

---

## 4. Traza paso a paso sobre una solución concreta con depósito

**Solución inicial:**

```
solucion = [
    ["D", "TR1", "TR2", "TR3", "TR4", "D"],
    ["D", "TR5", "TR6", "D"],
]
```

**Parámetros elegidos:** `r = 0`, `i = 0`, `j = 3`

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

**Paso 3 — `s[0][0], s[0][3] = s[0][3], s[0][0]` → intercambia `"TR1"` y `"TR4"`:**

```
vec (normalizado) = [
    ["TR4", "TR2", "TR3", "TR1"],
    ["TR5", "TR6"],
]
```

**Paso 4 — Registrar movimiento:**

```python
mov = MovimientoVecindario("swap_intra", ruta_a=0, i=0, j=3)
```

**Paso 5 — Desnormalizar (restaurar `"D"`):**

```
vecino final = [
    ["D", "TR4", "TR2", "TR3", "TR1", "D"],
    ["D", "TR5", "TR6", "D"],
]
```

**Diagrama antes/después:**

```
Antes (ruta 0):   ["TR1", "TR2", "TR3", "TR4"]
                   ^                       ^
                   i=0                   j=3
                        swap(0, 3)
Después (ruta 0): ["TR4", "TR2", "TR3", "TR1"]
```

---

## 5. Requisitos y precondiciones reales

| Condición | Valor | Razón |
|---|---|---|
| Tareas mínimas en `ruta_a` | `>= 2` | Con una sola tarea no hay par posible para intercambiar. |
| `i != j` | obligatorio | `rng.sample(range(n), 2)` lo garantiza: nunca devuelve el mismo índice dos veces. |
| Número de rutas | `>= 1` | Operador intra; solo necesita una ruta. |

---

## 6. Contraste "Pseudocódigo vs. realidad"

Un pseudocódigo ingenuo mostraría:

```
s' ← Copy(s)
swap(r'_a[i], r'_a[j])
m ← (swap_intra, a, i, j)
Return (s', m)
```

Lo que omitía:

- La normalización previa que elimina `"D"` de todas las rutas.
- El filtrado `cand = [r for r in activos if len(rutas[r]) >= 2]`.
- El uso de `rng.sample(range(n), 2)` en lugar de dos `randrange` independientes: la diferencia no es cosmética, ya que `sample` garantiza `i != j` sin reintento extra.
- El bucle de hasta 500 reintentos (en este operador solo actúa si `not cand`).
- La desnormalización final.

---

## 7. Pseudocódigo corregido en LaTeX

```latex
\begin{algorithm}
\DontPrintSemicolon
\caption{\texttt{swap\_intra} --- Intercambiar dos tareas dentro de la misma ruta}
\KwIn{Solución con depósito $s$;\; generador aleatorio $\mathit{rng}$;\; máximo de intentos $T_{\max}=500$}
\KwOut{Vecino con depósito $s'$;\; registro de movimiento $m$}
$\hat{s} \leftarrow \textsc{NormalizarQuitarDepósito}(s)$
\tcp{elimina el token "D" de inicio y fin de cada ruta}
$t \leftarrow 0$\;
\Repeat{movimiento válido encontrado}{
    $t \leftarrow t + 1$\;
    \lIf{$t > T_{\max}$}{\textbf{lanzar} \texttt{RuntimeError}}
    $\mathcal{C} \leftarrow \{k : |\hat{r}_k| \geq 2\}$
    \tcp{rutas candidatas: al menos 2 tareas}
    \lIf{$\mathcal{C} = \emptyset$}{\textbf{continue}}
    $a \leftarrow \mathit{rng}.\textsc{choice}(\mathcal{C})$;\quad $n \leftarrow |\hat{r}_a|$\;
    $(i,\, j) \leftarrow \mathit{rng}.\textsc{sample}(\{0,\ldots,n-1\},\, 2)$
    \tcp{dos índices distintos sin reemplazo (garantiza $i \neq j$ por construcción)}
    $\hat{s}' \leftarrow \textsc{CopiaProfunda}(\hat{s})$
    \tcp{Python pasa listas por referencia; sin copia se corrompería $\hat{s}$}
    $\hat{r}'_a[i],\, \hat{r}'_a[j] \leftarrow \hat{r}'_a[j],\, \hat{r}'_a[i]$
    \tcp{intercambio atómico: lado derecho se evalúa completo antes de asignar}
    $m \leftarrow \textsc{MovimientoVecindario}(\texttt{swap\_intra},\; \mathit{ruta\_a}=a,\; i=i,\; j=j)$\;
    $s' \leftarrow \textsc{RestaurarDepósito}(\hat{s}')$
    \tcp{añade "D" al inicio y fin de cada ruta antes de devolver}
    \Return $(s',\, m)$\;
}
\end{algorithm}
```
