# `op_cross_exchange` — Intercambio de segmentos inter-ruta

**Familia:** inter-ruta | **Archivo fuente:** [`vecindarios.py:385–409`](../../metacarp/vecindarios.py#L385)

---

## 1. Qué hace el movimiento

Extrae un segmento contiguo `[i..j]` (ambos inclusivos) de la ruta `ra` y un segmento contiguo `[k..l]` (ambos inclusivos) de la ruta `rb`, y los intercambia entre sí. Los dos segmentos pueden tener longitudes distintas. Es el operador inter-ruta más disruptivo del módulo.

Formalmente:

```
seg_A = ruta_A[i : j+1]
seg_B = ruta_B[k : l+1]

ruta_A' = ruta_A[:i] + seg_B + ruta_A[j+1:]
ruta_B' = ruta_B[:k] + seg_A + ruta_B[l+1:]
```

**Cuándo conviene:** cuando bloques de tareas adyacentes tienen mejor encaje en la ruta contraria. Al transferir segmentos completos (no tareas individuales), este operador puede reestructurar significativamente la distribución de carga y escapar de mínimos locales que los operadores de tarea única no alcanzan.

---

## 2. Código Python real del operador puro

[`vecindarios.py:385–409`](../../metacarp/vecindarios.py#L385)

```python
def op_cross_exchange(
    sol: Sequence[Sequence[Hashable]],
    ra: int,   # Índice de la ruta A
    i: int,    # Inicio del segmento en la ruta A (inclusive)
    j: int,    # Fin del segmento en la ruta A (inclusive)
    rb: int,   # Índice de la ruta B
    k: int,    # Inicio del segmento en la ruta B (inclusive)
    l: int,    # Fin del segmento en la ruta B (inclusive)
) -> list[list[str]]:
    """
    Cross-exchange: intercambia segmentos [i:j] de A con [k:l] de B.
    Índices inclusivos.
    """
    s = _copy_solution(sol)
    a = s[ra]
    b = s[rb]

    seg_a = a[i : j + 1]   # Extrae el segmento de A (de i hasta j inclusive)
    seg_b = b[k : l + 1]   # Extrae el segmento de B (de k hasta l inclusive)

    # Reconstruye ruta A: parte antes de i + segmento de B + parte después de j
    s[ra] = a[:i] + seg_b + a[j + 1 :]
    # Reconstruye ruta B: parte antes de k + segmento de A + parte después de l
    s[rb] = b[:k] + seg_a + b[l + 1 :]
    return s
```

Los slicings `a[:i]` y `a[j+1:]` producen nuevas listas. La concatenación con `+` produce una nueva lista resultante. Esto es seguro respecto a la copia porque `a` y `b` son referencias a las listas dentro de la copia `s`, no a la solución original.

---

## 3. Bloque de selección de índices en `generar_vecino`

[`vecindarios.py:1104–1117`](../../metacarp/vecindarios.py#L1104)

```python
elif op == "cross_exchange":
    if len(rutas) < 2:
        continue
    non_empty = [x for x in range(len(rutas)) if len(rutas[x]) >= 2]
    if len(non_empty) < 2:
        continue
    ra, rb = rng.sample(non_empty, 2)
    na, nb = len(rutas[ra]), len(rutas[rb])
    i = rng.randrange(0, na - 1)     # Inicio del segmento en A
    j = rng.randrange(i + 1, na)     # Fin del segmento en A (j > i)
    k = rng.randrange(0, nb - 1)     # Inicio del segmento en B
    l = rng.randrange(k + 1, nb)     # Fin del segmento en B (l > k)
    vec = op_cross_exchange(rutas, ra, i, j, rb, k, l)
    mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=i, j=j, k=k, l=l)
```

El bloque equivalente en `generar_vecino_ids` está en [`vecindarios.py:861–875`](../../metacarp/vecindarios.py#L861).

### Distribuciones de los sorteos

| Sorteo | Llamada | Distribución |
|---|---|---|
| Par de rutas `(ra, rb)` | `rng.sample(non_empty, 2)` | Uniforme discreta sin reemplazo sobre rutas con `>= 2` tareas. |
| Inicio seg. A (`i`) | `rng.randrange(0, na-1)` | Uniforme en `{0, ..., na-2}`. |
| Fin seg. A (`j`) | `rng.randrange(i+1, na)` | Uniforme en `{i+1, ..., na-1}`. Garantiza `j > i`. |
| Inicio seg. B (`k`) | `rng.randrange(0, nb-1)` | Uniforme en `{0, ..., nb-2}`. |
| Fin seg. B (`l`) | `rng.randrange(k+1, nb)` | Uniforme en `{k+1, ..., nb-1}`. Garantiza `l > k`. |

La selección de `i` primero y `j` condicionada a `j > i` (y análogamente para `k` y `l`) garantiza que ambos segmentos tengan longitud mínima de 2 elementos.

> **Nota:** el requisito mínimo es `>= 2` tareas en ambas rutas participantes (no `>= 1`). Con solo 1 tarea, `rng.randrange(0, na-1)` con `na=1` produciría `rng.randrange(0, 0)`, que lanza `ValueError`. El filtro `non_empty = [x for x in range(len(rutas)) if len(rutas[x]) >= 2]` previene este caso.

---

## 4. Traza paso a paso sobre una solución concreta con depósito

**Solución inicial:**

```
solucion = [
    ["D", "TR1", "TR2", "TR3", "TR4", "D"],
    ["D", "TR5", "TR6", "TR7", "D"],
]
```

**Parámetros elegidos:** `ra = 0`, `i = 1`, `j = 2`, `rb = 1`, `k = 0`, `l = 1`

**Paso 1 — Normalizar (quitar `"D"`):**

```
rutas = [
    ["TR1", "TR2", "TR3", "TR4"],   # ruta 0 (ra)
    ["TR5", "TR6", "TR7"],           # ruta 1 (rb)
]
```

**Paso 2 — Copiar (`_copy_solution`):**

```
s = [
    ["TR1", "TR2", "TR3", "TR4"],
    ["TR5", "TR6", "TR7"],
]
```

**Paso 3 — Extraer segmentos:**

```
a = s[0] = ["TR1", "TR2", "TR3", "TR4"]
b = s[1] = ["TR5", "TR6", "TR7"]

seg_a = a[1:3]  = ["TR2", "TR3"]
seg_b = b[0:2]  = ["TR5", "TR6"]
```

**Paso 4 — Reconstruir rutas con segmentos intercambiados:**

```
s[0] = a[:1] + seg_b + a[3:]
     = ["TR1"] + ["TR5", "TR6"] + ["TR4"]
     = ["TR1", "TR5", "TR6", "TR4"]

s[1] = b[:0] + seg_a + b[2:]
     = [] + ["TR2", "TR3"] + ["TR7"]
     = ["TR2", "TR3", "TR7"]
```

**Paso 5 — Registrar movimiento:**

```python
mov = MovimientoVecindario("cross_exchange", ruta_a=0, ruta_b=1, i=1, j=2, k=0, l=1)
```

**Paso 6 — Desnormalizar (restaurar `"D"`):**

```
vecino final = [
    ["D", "TR1", "TR5", "TR6", "TR4", "D"],
    ["D", "TR2", "TR3", "TR7", "D"],
]
```

**Diagrama antes/después:**

```
Antes:
  ruta 0: ["TR1", "TR2", "TR3", "TR4"]
                    |──seg_A──|
  ruta 1: ["TR5", "TR6", "TR7"]
           |──seg_B──|

Después (segmentos intercambiados):
  ruta 0: ["TR1", "TR5", "TR6", "TR4"]
                    |──seg_B──|
  ruta 1: ["TR2", "TR3", "TR7"]
           |──seg_A──|
```

---

## 5. Requisitos y precondiciones reales

| Condición | Valor | Razón |
|---|---|---|
| Tareas mínimas en `ruta_a` | `>= 2` | `rng.randrange(0, na-1)` requiere `na >= 2`. |
| Tareas mínimas en `ruta_b` | `>= 2` | `rng.randrange(0, nb-1)` requiere `nb >= 2`. |
| `ra != rb` | obligatorio | `rng.sample(non_empty, 2)` lo garantiza. |
| `i < j` y `k < l` | obligatorio | La selección dependiente de índices lo garantiza por construcción. |

---

## 6. Contraste "Pseudocódigo vs. realidad"

Un pseudocódigo ingenuo mostraría:

```
s' ← Copy(s)
seg_A ← r'_a[i:j+1];  seg_B ← r'_b[k:l+1]
r'_a ← r'_a[:i] + seg_B + r'_a[j+1:]
r'_b ← r'_b[:k] + seg_A + r'_b[l+1:]
m ← (cross_exchange, a, b, i, j, k, l)
Return (s', m)
```

Lo que omitía:

- La normalización previa que elimina `"D"` de todas las rutas.
- El requisito mínimo de `>= 2` tareas en ambas rutas (no `>= 1`).
- La selección dependiente de cuatro índices: `i` y `j` con `j > i`, y `k` y `l` con `l > k`, cada par usando sorteos condicionados.
- El hecho de que los cuatro campos `i`, `j`, `k`, `l` de `MovimientoVecindario` se utilizan todos (es el único operador que usa `l`).
- El bucle de hasta 500 reintentos.
- La desnormalización final.

---

## 7. Pseudocódigo corregido en LaTeX

```latex
\begin{algorithm}
\DontPrintSemicolon
\caption{\texttt{cross\_exchange} --- Intercambiar segmentos entre dos rutas}
\KwIn{Solución con depósito $s$;\; generador aleatorio $\mathit{rng}$;\; máximo de intentos $T_{\max}=500$}
\KwOut{Vecino con depósito $s'$;\; registro de movimiento $m$}
$\hat{s} \leftarrow \textsc{NormalizarQuitarDepósito}(s)$
\tcp{elimina el token "D" de inicio y fin de cada ruta}
$t \leftarrow 0$\;
\Repeat{movimiento válido encontrado}{
    $t \leftarrow t + 1$\;
    \lIf{$t > T_{\max}$}{\textbf{lanzar} \texttt{RuntimeError}}
    \lIf{$|\hat{s}| < 2$}{\textbf{continue}}
    $\mathcal{N} \leftarrow \{k : |\hat{r}_k| \geq 2\}$
    \tcp{rutas con al menos 2 tareas en ambas participantes (requisito del segmento)}
    \lIf{$|\mathcal{N}| < 2$}{\textbf{continue}}
    $(\mathit{ra},\, \mathit{rb}) \leftarrow \mathit{rng}.\textsc{sample}(\mathcal{N},\, 2)$
    \tcp{dos rutas distintas con $\geq 2$ tareas, sin reemplazo}
    $n_a \leftarrow |\hat{r}_{\mathit{ra}}|$;\quad $n_b \leftarrow |\hat{r}_{\mathit{rb}}|$\;
    $i \leftarrow \mathit{rng}.\textsc{randrange}(0,\, n_a - 1)$
    \tcp{inicio del segmento en A, uniforme en $\{0,\ldots,n_a-2\}$}
    $j \leftarrow \mathit{rng}.\textsc{randrange}(i+1,\, n_a)$
    \tcp{fin del segmento en A, uniforme en $\{i+1,\ldots,n_a-1\}$; garantiza $j > i$}
    $k \leftarrow \mathit{rng}.\textsc{randrange}(0,\, n_b - 1)$
    \tcp{inicio del segmento en B, uniforme en $\{0,\ldots,n_b-2\}$}
    $l \leftarrow \mathit{rng}.\textsc{randrange}(k+1,\, n_b)$
    \tcp{fin del segmento en B, uniforme en $\{k+1,\ldots,n_b-1\}$; garantiza $l > k$}
    $\hat{s}' \leftarrow \textsc{CopiaProfunda}(\hat{s})$
    \tcp{Python pasa listas por referencia; sin copia se corrompería $\hat{s}$}
    $\mathit{seg\_A} \leftarrow \hat{r}'_{\mathit{ra}}[i\,{:}\,j{+}1]$;\quad
    $\mathit{seg\_B} \leftarrow \hat{r}'_{\mathit{rb}}[k\,{:}\,l{+}1]$\;
    $\hat{r}'_{\mathit{ra}} \leftarrow \hat{r}'_{\mathit{ra}}[{:}i] \mathbin{\|} \mathit{seg\_B} \mathbin{\|} \hat{r}'_{\mathit{ra}}[j{+}1{:}]$\;
    $\hat{r}'_{\mathit{rb}} \leftarrow \hat{r}'_{\mathit{rb}}[{:}k] \mathbin{\|} \mathit{seg\_A} \mathbin{\|} \hat{r}'_{\mathit{rb}}[l{+}1{:}]$\;
    $m \leftarrow \textsc{MovimientoVecindario}(\texttt{cross\_exchange},\; \mathit{ruta\_a}=\mathit{ra},\; \mathit{ruta\_b}=\mathit{rb},\; i=i,\; j=j,\; k=k,\; l=l)$\;
    $s' \leftarrow \textsc{RestaurarDepósito}(\hat{s}')$
    \tcp{añade "D" al inicio y fin de cada ruta antes de devolver}
    \Return $(s',\, m)$\;
}
\end{algorithm}
```
