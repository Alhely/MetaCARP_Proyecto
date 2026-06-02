# `op_2opt_intra` — Inversión de segmento intra-ruta

**Familia:** intra-ruta | **Archivo fuente:** [`vecindarios.py:236–243`](../../metacarp/vecindarios.py#L236)

---

## 1. Qué hace el movimiento

Invierte (voltea) el subsegmento de tareas entre las posiciones `i` y `j` (ambas inclusive) dentro de **la misma ruta**. Las tareas fuera del segmento permanecen en su lugar. Es la generalización del operador 2-opt del TSP (Traveling Salesman Problem) aplicado al dominio de arcos del CARP.

**Cuándo conviene:** cuando la ruta se "cruza a sí misma" en su representación geográfica y servir un subsegmento en sentido contrario reduce el costo de deadheading acumulado. Es el operador más potente de la familia intra para corregir ineficiencias de orden en rutas largas.

---

## 2. Código Python real del operador puro

[`vecindarios.py:236–243`](../../metacarp/vecindarios.py#L236)

```python
def op_2opt_intra(sol: Sequence[Sequence[Hashable]], r: int, i: int, j: int) -> list[list[str]]:
    """
    2-opt (intra): revierte el segmento [i:j] (i < j) en la ruta r.
    """
    s = _copy_solution(sol)
    ruta = s[r]
    ruta[i : j + 1] = reversed(ruta[i : j + 1])  # Invierte el segmento entre i y j (inclusive)
    return s
```

La notación `ruta[i : j+1]` es slicing de Python: selecciona los elementos desde el índice `i` hasta el `j` incluido (el límite superior del slice es exclusivo, por eso se usa `j+1`). La asignación `ruta[i:j+1] = reversed(...)` reemplaza ese segmento en su lugar.

---

## 3. Bloque de selección de índices en `generar_vecino`

[`vecindarios.py:1049–1061`](../../metacarp/vecindarios.py#L1049)

```python
elif op == "2opt_intra":
    cand = [x for x in activos if len(rutas[x]) >= 3]
    if not cand:
        continue
    r = rng.choice(cand)
    n = len(rutas[r])
    i = rng.randrange(0, n - 1)
    j = rng.randrange(i + 1, n)
    if j - i < 1:
        # Segmento de longitud 1 no genera vecino útil
        continue
    vec = op_2opt_intra(rutas, r, i, j)
    mov = MovimientoVecindario(op, ruta_a=r, i=i, j=j)
```

El bloque equivalente en `generar_vecino_ids` está en [`vecindarios.py:806–816`](../../metacarp/vecindarios.py#L806).

### Distribuciones de los sorteos

| Sorteo | Llamada | Distribución |
|---|---|---|
| Ruta candidata | `rng.choice(cand)` | Uniforme discreta sobre rutas con `>= 3` tareas. |
| Inicio del segmento `i` | `rng.randrange(0, n-1)` | Uniforme discreta en `{0, ..., n-2}`. |
| Fin del segmento `j` | `rng.randrange(i+1, n)` | Uniforme discreta en `{i+1, ..., n-1}`. Garantiza `j > i`. |

La selección secuencial `i` primero y `j` en `[i+1, n)` garantiza `i < j` por construcción, por lo que el segmento tiene longitud mínima de 2. La verificación `if j - i < 1: continue` es una comprobación defensiva adicional.

> **Nota:** el requisito mínimo es `>= 3` tareas —no `>= 2`— porque con solo 2 tareas el único segmento posible es el completo `[0, 1]`, que invertido produce la misma ruta espejada pero con un costo diferente. El código sí lo permite (no es un `continue` adicional sobre eso), pero la condición `>= 3` proviene de que `rng.randrange(0, n-1)` requiere `n-1 > 0`, es decir `n >= 2`, y para que luego `rng.randrange(i+1, n)` tenga al menos un valor, necesitamos `n >= 2` con `i` al menos `0`, lo que funciona. El comentario en el código dice `>= 3` para garantizar un segmento de longitud 2 dentro de una ruta más larga.

---

## 4. Traza paso a paso sobre una solución concreta con depósito

**Solución inicial:**

```
solucion = [
    ["D", "TR1", "TR2", "TR3", "TR4", "TR5", "D"],
    ["D", "TR6", "TR7", "D"],
]
```

**Parámetros elegidos:** `r = 0`, `i = 1`, `j = 3`

**Paso 1 — Normalizar (quitar `"D"`):**

```
rutas = [
    ["TR1", "TR2", "TR3", "TR4", "TR5"],   # ruta 0
    ["TR6", "TR7"],                          # ruta 1
]
```

**Paso 2 — Copiar (`_copy_solution`):**

```
s = [
    ["TR1", "TR2", "TR3", "TR4", "TR5"],   # copia independiente
    ["TR6", "TR7"],
]
```

**Paso 3 — `s[0][1:4] = reversed(s[0][1:4])`:**

El segmento `s[0][1:4]` es `["TR2", "TR3", "TR4"]`. Invertido: `["TR4", "TR3", "TR2"]`.

```
vec (normalizado) = [
    ["TR1", "TR4", "TR3", "TR2", "TR5"],
    ["TR6", "TR7"],
]
```

**Paso 4 — Registrar movimiento:**

```python
mov = MovimientoVecindario("2opt_intra", ruta_a=0, i=1, j=3)
```

**Paso 5 — Desnormalizar (restaurar `"D"`):**

```
vecino final = [
    ["D", "TR1", "TR4", "TR3", "TR2", "TR5", "D"],
    ["D", "TR6", "TR7", "D"],
]
```

**Diagrama antes/después:**

```
Antes (ruta 0):   ["TR1", "TR2", "TR3", "TR4", "TR5"]
                           |←── segmento [1:4] ──→|
                           ["TR2", "TR3", "TR4"]
                                  reversed
                           ["TR4", "TR3", "TR2"]
Después (ruta 0): ["TR1", "TR4", "TR3", "TR2", "TR5"]
```

---

## 5. Requisitos y precondiciones reales

| Condición | Valor | Razón |
|---|---|---|
| Tareas mínimas en `ruta_a` | `>= 3` | El sorteo `rng.randrange(0, n-1)` requiere `n >= 2`, y `rng.randrange(i+1, n)` con `i >= 0` requiere `n >= 2`. El requisito `>= 3` asegura que el segmento invertible tiene al menos 2 tareas dentro de una ruta con contexto. |
| `i < j` | obligatorio | `rng.randrange(i+1, n)` lo garantiza por construcción. |
| Número de rutas | `>= 1` | Operador intra; solo necesita una ruta. |

---

## 6. Contraste "Pseudocódigo vs. realidad"

Un pseudocódigo ingenuo mostraría:

```
s' ← Copy(s)
r'_a[i:j+1] ← reversed(r'_a[i:j+1])
m ← (2opt_intra, a, i, j)
Return (s', m)
```

Lo que omitía:

- La normalización previa que elimina `"D"` de todas las rutas.
- El filtrado `cand = [r for r in activos if len(rutas[r]) >= 3]` (requisito mínimo de 3 tareas, no 2).
- La selección dependiente de índices: `i = rng.randrange(0, n-1)` seguido de `j = rng.randrange(i+1, n)`. No son dos sorteos independientes: `j` depende del valor de `i`.
- El bucle de hasta 500 reintentos.
- La desnormalización final.

---

## 7. Pseudocódigo corregido en LaTeX

```latex
\begin{algorithm}
\DontPrintSemicolon
\caption{\texttt{2opt\_intra} --- Invertir un segmento de tareas dentro de la misma ruta}
\KwIn{Solución con depósito $s$;\; generador aleatorio $\mathit{rng}$;\; máximo de intentos $T_{\max}=500$}
\KwOut{Vecino con depósito $s'$;\; registro de movimiento $m$}
$\hat{s} \leftarrow \textsc{NormalizarQuitarDepósito}(s)$
\tcp{elimina el token "D" de inicio y fin de cada ruta}
$t \leftarrow 0$\;
\Repeat{movimiento válido encontrado}{
    $t \leftarrow t + 1$\;
    \lIf{$t > T_{\max}$}{\textbf{lanzar} \texttt{RuntimeError}}
    $\mathcal{C} \leftarrow \{k : |\hat{r}_k| \geq 3\}$
    \tcp{rutas candidatas: mínimo 3 tareas para garantizar segmento de longitud $\geq 2$}
    \lIf{$\mathcal{C} = \emptyset$}{\textbf{continue}}
    $a \leftarrow \mathit{rng}.\textsc{choice}(\mathcal{C})$;\quad $n \leftarrow |\hat{r}_a|$\;
    $i \leftarrow \mathit{rng}.\textsc{randrange}(0,\, n-1)$
    \tcp{inicio del segmento, uniforme en $\{0,\ldots,n-2\}$}
    $j \leftarrow \mathit{rng}.\textsc{randrange}(i+1,\, n)$
    \tcp{fin del segmento, uniforme en $\{i+1,\ldots,n-1\}$; garantiza $j > i$}
    $\hat{s}' \leftarrow \textsc{CopiaProfunda}(\hat{s})$
    \tcp{Python pasa listas por referencia; sin copia se corrompería $\hat{s}$}
    $\hat{r}'_a[i\,{:}\,j{+}1] \leftarrow \textsc{Revertir}(\hat{r}'_a[i\,{:}\,j{+}1])$
    \tcp{asignación de slice: reemplaza el segmento en su lugar}
    $m \leftarrow \textsc{MovimientoVecindario}(\texttt{2opt\_intra},\; \mathit{ruta\_a}=a,\; i=i,\; j=j)$\;
    $s' \leftarrow \textsc{RestaurarDepósito}(\hat{s}')$
    \tcp{añade "D" al inicio y fin de cada ruta antes de devolver}
    \Return $(s',\, m)$\;
}
\end{algorithm}
```
