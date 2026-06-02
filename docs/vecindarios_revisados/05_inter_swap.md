# `op_swap_inter` — Intercambio inter-ruta

**Familia:** inter-ruta | **Archivo fuente:** [`vecindarios.py:291–302`](../../metacarp/vecindarios.py#L291)

---

## 1. Qué hace el movimiento

Intercambia la tarea en la posición `i` de la ruta `ra` con la tarea en la posición `j` de la ruta `rb` (`ra != rb`). Ambas rutas conservan exactamente el mismo número de tareas antes y después del movimiento. Solo las dos tareas seleccionadas cambian de ruta.

**Cuándo conviene:** cuando una tarea específica de la ruta `ra` encaja mejor geográficamente con las tareas de la ruta `rb` y viceversa, sin que sea necesario alterar la distribución de carga (cardinalidad de las rutas). Es menos disruptivo que `relocate_inter` porque no cambia el tamaño de ninguna ruta.

---

## 2. Código Python real del operador puro

[`vecindarios.py:291–302`](../../metacarp/vecindarios.py#L291)

```python
def op_swap_inter(
    sol: Sequence[Sequence[Hashable]],
    ra: int,
    i: int,
    rb: int,
    j: int,
) -> list[list[str]]:
    """Swap (inter): intercambia una tarea entre rutas ra y rb."""
    s = _copy_solution(sol)
    # Intercambio simultáneo usando desempaquetado de tuplas (ver op_swap_intra)
    s[ra][i], s[rb][j] = s[rb][j], s[ra][i]
    return s
```

El intercambio `s[ra][i], s[rb][j] = s[rb][j], s[ra][i]` es atómico: Python evalúa completamente el lado derecho antes de realizar cualquier asignación. Esto funciona incluso si `ra == rb` (aunque el código de selección lo previene).

---

## 3. Bloque de selección de índices en `generar_vecino`

[`vecindarios.py:1080–1090`](../../metacarp/vecindarios.py#L1080)

```python
elif op == "swap_inter":
    if len(rutas) < 2:
        continue
    non_empty = [x for x in range(len(rutas)) if len(rutas[x]) > 0]
    if len(non_empty) < 2:
        continue
    ra, rb = rng.sample(non_empty, 2)
    i = rng.randrange(len(rutas[ra]))
    j = rng.randrange(len(rutas[rb]))
    vec = op_swap_inter(rutas, ra, i, rb, j)
    mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=i, j=j)
```

El bloque equivalente en `generar_vecino_ids` está en [`vecindarios.py:835–846`](../../metacarp/vecindarios.py#L835).

### Distribuciones de los sorteos

| Sorteo | Llamada | Distribución |
|---|---|---|
| Par de rutas `(ra, rb)` | `rng.sample(non_empty, 2)` | Uniforme discreta sin reemplazo sobre pares de rutas no vacías. Garantiza `ra != rb`. |
| Posición `i` en `ra` | `rng.randrange(len(rutas[ra]))` | Uniforme discreta en `{0, ..., len(ra)-1}`. |
| Posición `j` en `rb` | `rng.randrange(len(rutas[rb]))` | Uniforme discreta en `{0, ..., len(rb)-1}`. |

El uso de `rng.sample(non_empty, 2)` garantiza dos rutas distintas con al menos una tarea cada una. A diferencia de `relocate_inter`, aquí no se hace `rng.randrange` individual y luego `continue` si coinciden: el `sample` asegura distinción por construcción.

---

## 4. Traza paso a paso sobre una solución concreta con depósito

**Solución inicial:**

```
solucion = [
    ["D", "TR1", "TR2", "D"],
    ["D", "TR3", "TR4", "TR5", "D"],
]
```

**Parámetros elegidos:** `ra = 0`, `i = 0`, `rb = 1`, `j = 1`

**Paso 1 — Normalizar (quitar `"D"`):**

```
rutas = [
    ["TR1", "TR2"],             # ruta 0 (ra)
    ["TR3", "TR4", "TR5"],      # ruta 1 (rb)
]
```

**Paso 2 — Copiar (`_copy_solution`):**

```
s = [
    ["TR1", "TR2"],
    ["TR3", "TR4", "TR5"],
]
```

**Paso 3 — `s[0][0], s[1][1] = s[1][1], s[0][0]` → intercambia `"TR1"` y `"TR4"`:**

```
vec (normalizado) = [
    ["TR4", "TR2"],
    ["TR3", "TR1", "TR5"],
]
```

**Paso 4 — Registrar movimiento:**

```python
mov = MovimientoVecindario("swap_inter", ruta_a=0, ruta_b=1, i=0, j=1)
```

**Paso 5 — Desnormalizar (restaurar `"D"`):**

```
vecino final = [
    ["D", "TR4", "TR2", "D"],
    ["D", "TR3", "TR1", "TR5", "D"],
]
```

**Diagrama antes/después:**

```
Antes:
  ruta 0 (ra): ["TR1", "TR2"]          ruta 1 (rb): ["TR3", "TR4", "TR5"]
                  ^                                           ^
                  i=0                                        j=1
                          swap(ra[0], rb[1])
Después:
  ruta 0:      ["TR4", "TR2"]          ruta 1:      ["TR3", "TR1", "TR5"]
```

---

## 5. Requisitos y precondiciones reales

| Condición | Valor | Razón |
|---|---|---|
| Tareas mínimas en `ruta_a` | `>= 1` | Se necesita al menos una tarea para intercambiar. |
| Tareas mínimas en `ruta_b` | `>= 1` | Igual. |
| `ra != rb` | obligatorio | `rng.sample(non_empty, 2)` lo garantiza por construcción. |
| Número de rutas no vacías | `>= 2` | Si `len(non_empty) < 2`, el código ejecuta `continue`. |

---

## 6. Contraste "Pseudocódigo vs. realidad"

Un pseudocódigo ingenuo mostraría:

```
s' ← Copy(s)
swap(r'_a[i], r'_b[j])
m ← (swap_inter, a, b, i, j)
Return (s', m)
```

Lo que omitía:

- La normalización previa que elimina `"D"` de todas las rutas.
- El filtrado explícito `non_empty = [x for x in range(len(rutas)) if len(rutas[x]) > 0]`: ambas rutas deben ser no vacías.
- El uso de `rng.sample(non_empty, 2)` para elegir dos rutas distintas en un solo sorteo sin reemplazo.
- El bucle de hasta 500 reintentos.
- La desnormalización final.

---

## 7. Pseudocódigo corregido en LaTeX

```latex
\begin{algorithm}
\DontPrintSemicolon
\caption{\texttt{swap\_inter} --- Intercambiar una tarea entre dos rutas distintas}
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
    \tcp{rutas no vacías; ambas rutas participantes deben tener al menos 1 tarea}
    \lIf{$|\mathcal{N}| < 2$}{\textbf{continue}}
    $(\mathit{ra},\, \mathit{rb}) \leftarrow \mathit{rng}.\textsc{sample}(\mathcal{N},\, 2)$
    \tcp{dos rutas distintas no vacías, sin reemplazo; garantiza $\mathit{ra} \neq \mathit{rb}$}
    $i \leftarrow \mathit{rng}.\textsc{randrange}(|\hat{r}_{\mathit{ra}}|)$
    \tcp{posición en la ruta $\mathit{ra}$}
    $j \leftarrow \mathit{rng}.\textsc{randrange}(|\hat{r}_{\mathit{rb}}|)$
    \tcp{posición en la ruta $\mathit{rb}$}
    $\hat{s}' \leftarrow \textsc{CopiaProfunda}(\hat{s})$
    \tcp{Python pasa listas por referencia; sin copia se corrompería $\hat{s}$}
    $\hat{r}'_{\mathit{ra}}[i],\, \hat{r}'_{\mathit{rb}}[j] \leftarrow \hat{r}'_{\mathit{rb}}[j],\, \hat{r}'_{\mathit{ra}}[i]$
    \tcp{intercambio atómico: lado derecho se evalúa completo antes de asignar}
    $m \leftarrow \textsc{MovimientoVecindario}(\texttt{swap\_inter},\; \mathit{ruta\_a}=\mathit{ra},\; \mathit{ruta\_b}=\mathit{rb},\; i=i,\; j=j)$\;
    $s' \leftarrow \textsc{RestaurarDepósito}(\hat{s}')$
    \tcp{añade "D" al inicio y fin de cada ruta antes de devolver}
    \Return $(s',\, m)$\;
}
\end{algorithm}
```
