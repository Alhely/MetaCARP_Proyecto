# `op_two_opt_star` — Intercambio de colas inter-ruta (2-opt*)

**Familia:** inter-ruta | **Archivo fuente:** [`vecindarios.py:332–358`](../../metacarp/vecindarios.py#L332)

---

## 1. Qué hace el movimiento

Divide cada una de dos rutas en una **cabeza** y una **cola** según un punto de corte, y luego intercambia las colas entre las dos rutas. La ruta A conserva su cabeza y adopta la cola de B; la ruta B conserva su cabeza y adopta la cola de A.

Formalmente, dado `cut_a` para la ruta A y `cut_b` para la ruta B:

```
cabeza_A = ruta_A[: cut_a + 1]    cola_A = ruta_A[cut_a + 1 :]
cabeza_B = ruta_B[: cut_b + 1]    cola_B = ruta_B[cut_b + 1 :]

ruta_A' = cabeza_A + cola_B
ruta_B' = cabeza_B + cola_A
```

**Cuándo conviene:** cuando dos rutas se cruzan geográficamente en su segunda mitad. El intercambio de colas puede eliminar esos cruces y reducir el deadheading acumulado. Es la extensión del clásico 2-opt del TSP al caso de rutas múltiples.

---

## 2. Código Python real del operador puro

[`vecindarios.py:332–358`](../../metacarp/vecindarios.py#L332)

```python
def op_two_opt_star(
    sol: Sequence[Sequence[Hashable]],
    ra: int,      # Índice de la ruta A
    cut_a: int,   # Posición del corte en la ruta A (inclusive en la cabeza)
    rb: int,      # Índice de la ruta B
    cut_b: int,   # Posición del corte en la ruta B (inclusive en la cabeza)
) -> list[list[str]]:
    """
    2-opt* (inter): intercambia las colas después de los cortes.

    - A = [a0..a_cut] + tailA
    - B = [b0..b_cut] + tailB
    -> A' = [a0..a_cut] + tailB
    -> B' = [b0..b_cut] + tailA
    """
    s = _copy_solution(sol)
    a = s[ra]
    b = s[rb]

    # Dividir cada ruta en cabeza (hasta cut inclusive) y cola (desde cut+1)
    head_a, tail_a = a[: cut_a + 1], a[cut_a + 1 :]
    head_b, tail_b = b[: cut_b + 1], b[cut_b + 1 :]

    # Reconstruir rutas intercambiando colas
    s[ra] = head_a + tail_b   # La ruta A conserva su cabeza y adopta la cola de B
    s[rb] = head_b + tail_a   # La ruta B conserva su cabeza y adopta la cola de A
    return s
```

La suma de listas `head_a + tail_b` concatena las dos sublistas en una nueva lista. El slicing `a[:cut_a+1]` incluye el elemento en posición `cut_a`; `a[cut_a+1:]` toma todo desde `cut_a+1` hasta el final.

---

## 3. Bloque de selección de índices en `generar_vecino`

[`vecindarios.py:1092–1102`](../../metacarp/vecindarios.py#L1092)

```python
elif op == "2opt_star":
    if len(rutas) < 2:
        continue
    non_empty = [x for x in range(len(rutas)) if len(rutas[x]) > 0]
    if len(non_empty) < 2:
        continue
    ra, rb = rng.sample(non_empty, 2)
    cut_a = rng.randrange(len(rutas[ra]))
    cut_b = rng.randrange(len(rutas[rb]))
    vec = op_two_opt_star(rutas, ra, cut_a, rb, cut_b)
    mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=cut_a, j=cut_b)
```

El bloque equivalente en `generar_vecino_ids` está en [`vecindarios.py:848–859`](../../metacarp/vecindarios.py#L848).

### Distribuciones de los sorteos

| Sorteo | Llamada | Distribución |
|---|---|---|
| Par de rutas `(ra, rb)` | `rng.sample(non_empty, 2)` | Uniforme discreta sin reemplazo sobre pares de rutas no vacías. |
| Punto de corte `cut_a` | `rng.randrange(len(rutas[ra]))` | Uniforme discreta en `{0, ..., len(ra)-1}`. |
| Punto de corte `cut_b` | `rng.randrange(len(rutas[rb]))` | Uniforme discreta en `{0, ..., len(rb)-1}`. |

> **Nota sobre los campos de `MovimientoVecindario`:** para este operador, `i` almacena `cut_a` y `j` almacena `cut_b`. No hay campo de posición de destino en el sentido de `relocate`; los cortes determinan completamente el movimiento.

---

## 4. Traza paso a paso sobre una solución concreta con depósito

**Solución inicial:**

```
solucion = [
    ["D", "TR1", "TR2", "TR3", "TR4", "D"],
    ["D", "TR5", "TR6", "TR7", "D"],
]
```

**Parámetros elegidos:** `ra = 0`, `cut_a = 1`, `rb = 1`, `cut_b = 1`

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

**Paso 3 — Calcular cabezas y colas:**

```
head_a = s[0][:2]  = ["TR1", "TR2"]
tail_a = s[0][2:]  = ["TR3", "TR4"]

head_b = s[1][:2]  = ["TR5", "TR6"]
tail_b = s[1][2:]  = ["TR7"]
```

**Paso 4 — Intercambiar colas:**

```
s[0] = head_a + tail_b = ["TR1", "TR2"] + ["TR7"]       = ["TR1", "TR2", "TR7"]
s[1] = head_b + tail_a = ["TR5", "TR6"] + ["TR3", "TR4"] = ["TR5", "TR6", "TR3", "TR4"]
```

**Paso 5 — Registrar movimiento:**

```python
mov = MovimientoVecindario("2opt_star", ruta_a=0, ruta_b=1, i=1, j=1)
```

**Paso 6 — Desnormalizar (restaurar `"D"`):**

```
vecino final = [
    ["D", "TR1", "TR2", "TR7", "D"],
    ["D", "TR5", "TR6", "TR3", "TR4", "D"],
]
```

**Diagrama antes/después:**

```
Antes:
  ruta 0: ["TR1", "TR2" | "TR3", "TR4"]   cut_a=1
                          ^─── cola_A
  ruta 1: ["TR5", "TR6" | "TR7"]           cut_b=1
                          ^─── cola_B

Después:
  ruta 0: ["TR1", "TR2"] + ["TR7"]          = ["TR1", "TR2", "TR7"]
  ruta 1: ["TR5", "TR6"] + ["TR3", "TR4"]   = ["TR5", "TR6", "TR3", "TR4"]
```

---

## 5. Requisitos y precondiciones reales

| Condición | Valor | Razón |
|---|---|---|
| Tareas mínimas en `ruta_a` | `>= 1` | `rng.randrange(len(rutas[ra]))` requiere al menos un elemento. |
| Tareas mínimas en `ruta_b` | `>= 1` | Igual. |
| `ra != rb` | obligatorio | `rng.sample(non_empty, 2)` lo garantiza. |
| Número de rutas no vacías | `>= 2` | Si `len(non_empty) < 2`, el código ejecuta `continue`. |

> **Caso especial de cola vacía:** si `cut_a = len(ruta_a) - 1`, entonces `tail_a = []` (cola vacía). Esto es válido: la ruta A adoptaría toda la cola de B, y la ruta B terminaría con la cabeza de B únicamente. El código no previene este caso, por lo que es un movimiento legal.

---

## 6. Contraste "Pseudocódigo vs. realidad"

Un pseudocódigo ingenuo mostraría:

```
s' ← Copy(s)
head_A ← r'_a[:cut_a+1];  tail_A ← r'_a[cut_a+1:]
head_B ← r'_b[:cut_b+1];  tail_B ← r'_b[cut_b+1:]
r'_a ← head_A + tail_B
r'_b ← head_B + tail_A
m ← (2opt_star, a, b, cut_a, cut_b)
Return (s', m)
```

Lo que omitía:

- La normalización previa que elimina `"D"` de todas las rutas.
- El filtrado `non_empty = [x for x in range(len(rutas)) if len(rutas[x]) > 0]` (ambas rutas deben ser no vacías).
- El uso de `rng.sample(non_empty, 2)` en lugar de dos sorteos independientes.
- El hecho de que en `MovimientoVecindario`, `i` es `cut_a` y `j` es `cut_b` (no posiciones de tarea, sino puntos de corte de ruta).
- El bucle de hasta 500 reintentos.
- La desnormalización final.

---

## 7. Pseudocódigo corregido en LaTeX

```latex
\begin{algorithm}
\DontPrintSemicolon
\caption{\texttt{2opt\_star} --- Intercambiar las colas de dos rutas (2-opt*)}
\KwIn{Solución con depósito $s$;\; generador aleatorio $\mathit{rng}$;\; máximo de intentos $T_{\max}=500$}
\KwOut{Vecino con depósito $s'$;\; registro de movimiento $m$}
$\hat{s} \leftarrow \textsc{NormalizarQuitarDepósito}(s)$
\tcp{elimina el token "D" de inicio y fin de cada ruta}
$t \leftarrow 0$\;
\Repeat{movimiento válido encontrado}{
    $t \leftarrow t + 1$\;
    \lIf{$t > T_{\max}$}{\textbf{lanzar} \texttt{RuntimeError}}
    \lIf{$|\hat{s}| < 2$}{\textbf{continue}}
    $\mathcal{N} \leftarrow \{k : |\hat{r}_k| \geq 1\}$
    \tcp{rutas no vacías; ambas participantes necesitan al menos 1 tarea}
    \lIf{$|\mathcal{N}| < 2$}{\textbf{continue}}
    $(\mathit{ra},\, \mathit{rb}) \leftarrow \mathit{rng}.\textsc{sample}(\mathcal{N},\, 2)$
    \tcp{dos rutas distintas no vacías, sin reemplazo}
    $\mathit{cut\_a} \leftarrow \mathit{rng}.\textsc{randrange}(|\hat{r}_{\mathit{ra}}|)$
    \tcp{punto de corte en la ruta A, uniforme en $\{0,\ldots,|\hat{r}_{\mathit{ra}}|-1\}$}
    $\mathit{cut\_b} \leftarrow \mathit{rng}.\textsc{randrange}(|\hat{r}_{\mathit{rb}}|)$
    \tcp{punto de corte en la ruta B, uniforme en $\{0,\ldots,|\hat{r}_{\mathit{rb}}|-1\}$}
    $\hat{s}' \leftarrow \textsc{CopiaProfunda}(\hat{s})$
    \tcp{Python pasa listas por referencia; sin copia se corrompería $\hat{s}$}
    $\mathit{head\_A} \leftarrow \hat{r}'_{\mathit{ra}}[:\mathit{cut\_a}{+}1]$;\quad
    $\mathit{tail\_A} \leftarrow \hat{r}'_{\mathit{ra}}[\mathit{cut\_a}{+}1:]$\;
    $\mathit{head\_B} \leftarrow \hat{r}'_{\mathit{rb}}[:\mathit{cut\_b}{+}1]$;\quad
    $\mathit{tail\_B} \leftarrow \hat{r}'_{\mathit{rb}}[\mathit{cut\_b}{+}1:]$\;
    $\hat{r}'_{\mathit{ra}} \leftarrow \mathit{head\_A} \mathbin{\|} \mathit{tail\_B}$
    \tcp{ruta A adopta la cola de B}
    $\hat{r}'_{\mathit{rb}} \leftarrow \mathit{head\_B} \mathbin{\|} \mathit{tail\_A}$
    \tcp{ruta B adopta la cola de A}
    $m \leftarrow \textsc{MovimientoVecindario}(\texttt{2opt\_star},\; \mathit{ruta\_a}=\mathit{ra},\; \mathit{ruta\_b}=\mathit{rb},\; i=\mathit{cut\_a},\; j=\mathit{cut\_b})$\;
    $s' \leftarrow \textsc{RestaurarDepósito}(\hat{s}')$
    \tcp{añade "D" al inicio y fin de cada ruta antes de devolver}
    \Return $(s',\, m)$\;
}
\end{algorithm}
```
