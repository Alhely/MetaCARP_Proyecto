# `op_relocate_inter` — Reubicación inter-ruta

**Familia:** inter-ruta | **Archivo fuente:** [`vecindarios.py:265–276`](../../metacarp/vecindarios.py#L265)

---

## 1. Qué hace el movimiento

Extrae la tarea en la posición `i` de la ruta origen `ra` y la inserta en la posición `j` de la ruta destino `rb`, siendo `ra != rb`. La tarea cambia de ruta. La ruta origen pierde una tarea y la ruta destino gana una.

**Cuándo conviene:** para equilibrar la carga entre rutas cuando una ruta está sobrecargada, o cuando una tarea tiene mejor afinidad geográfica con las tareas de otra ruta. Es el operador inter más sencillo y puede vaciar completamente la ruta origen si solo tenía una tarea.

---

## 2. Código Python real del operador puro

[`vecindarios.py:265–276`](../../metacarp/vecindarios.py#L265)

```python
def op_relocate_inter(
    sol: Sequence[Sequence[Hashable]],
    ra: int,   # Índice de la ruta origen
    i: int,    # Posición de la tarea a mover dentro de la ruta origen
    rb: int,   # Índice de la ruta destino
    j: int,    # Posición de inserción dentro de la ruta destino
) -> list[list[str]]:
    """Relocate (inter): mueve una tarea de ruta ra posición i hacia ruta rb posición j."""
    s = _copy_solution(sol)
    x = s[ra].pop(i)      # Extrae la tarea de la ruta origen
    s[rb].insert(j, x)   # La inserta en la ruta destino
    return s
```

---

## 3. Bloque de selección de índices en `generar_vecino`

[`vecindarios.py:1064–1078`](../../metacarp/vecindarios.py#L1064)

```python
elif op == "relocate_inter":
    if len(activos) < 1 or len(rutas) < 2:
        continue
    ra = rng.choice(activos)
    if not rutas[ra]:
        continue
    rb = rng.randrange(len(rutas))
    if ra == rb and len(rutas) < 2:
        continue
    i = rng.randrange(len(rutas[ra]))
    j = rng.randrange(len(rutas[rb]) + 1)
    if ra == rb:
        continue  # Garantiza que sea inter y no intra
    vec = op_relocate_inter(rutas, ra, i, rb, j)
    mov = MovimientoVecindario(op, ruta_a=ra, ruta_b=rb, i=i, j=j)
```

El bloque equivalente en `generar_vecino_ids` está en [`vecindarios.py:820–833`](../../metacarp/vecindarios.py#L820).

### Distribuciones de los sorteos

| Sorteo | Llamada | Distribución |
|---|---|---|
| Ruta origen `ra` | `rng.choice(activos)` | Uniforme discreta sobre rutas no vacías. |
| Ruta destino `rb` | `rng.randrange(len(rutas))` | Uniforme discreta sobre **todas** las rutas (incluyendo vacías). |
| Posición origen `i` | `rng.randrange(len(rutas[ra]))` | Uniforme discreta en `{0, ..., len(ra)-1}`. |
| Posición destino `j` | `rng.randrange(len(rutas[rb]) + 1)` | Uniforme discreta en `{0, ..., len(rb)}`. El `+1` permite insertar al final. |

> **Detalle de implementación:** la ruta destino `rb` se sortea de **todas** las rutas, incluidas las vacías. La comprobación `if ra == rb: continue` al final garantiza que el movimiento sea efectivamente inter-ruta. Esto es distinto a filtrar primero las rutas candidatas: primero se sortean los índices y luego se descarta si `ra == rb`.

---

## 4. Traza paso a paso sobre una solución concreta con depósito

**Solución inicial:**

```
solucion = [
    ["D", "TR1", "TR2", "TR3", "D"],
    ["D", "TR4", "TR5", "D"],
]
```

**Parámetros elegidos:** `ra = 0`, `i = 2`, `rb = 1`, `j = 1`

**Paso 1 — Normalizar (quitar `"D"`):**

```
rutas = [
    ["TR1", "TR2", "TR3"],   # ruta 0 (ra)
    ["TR4", "TR5"],           # ruta 1 (rb)
]
```

**Paso 2 — Copiar (`_copy_solution`):**

```
s = [
    ["TR1", "TR2", "TR3"],   # copia independiente
    ["TR4", "TR5"],
]
```

**Paso 3 — `s[0].pop(2)` → extrae `"TR3"`, ruta 0 queda `["TR1", "TR2"]`:**

```
s = [
    ["TR1", "TR2"],
    ["TR4", "TR5"],
]
```

**Paso 4 — `s[1].insert(1, "TR3")` → inserta `"TR3"` en posición 1 de ruta 1:**

```
vec (normalizado) = [
    ["TR1", "TR2"],
    ["TR4", "TR3", "TR5"],
]
```

**Paso 5 — Registrar movimiento:**

```python
mov = MovimientoVecindario("relocate_inter", ruta_a=0, ruta_b=1, i=2, j=1)
```

**Paso 6 — Desnormalizar (restaurar `"D"`):**

```
vecino final = [
    ["D", "TR1", "TR2", "D"],
    ["D", "TR4", "TR3", "TR5", "D"],
]
```

**Diagrama antes/después:**

```
Antes:
  ruta 0 (ra): ["TR1", "TR2", "TR3"]
  ruta 1 (rb): ["TR4", "TR5"]
                                          pop(ra, i=2) ──► "TR3" sale de ruta 0
  ruta 0:      ["TR1", "TR2"]
                                          insert(rb, j=1, "TR3")
Después:
  ruta 0:      ["TR1", "TR2"]
  ruta 1:      ["TR4", "TR3", "TR5"]
```

---

## 5. Requisitos y precondiciones reales

| Condición | Valor | Razón |
|---|---|---|
| Tareas mínimas en `ruta_a` (origen) | `>= 1` | `rng.randrange(len(rutas[ra]))` requiere que la ruta no esté vacía. |
| Tareas mínimas en `ruta_b` (destino) | `>= 0` | La ruta destino puede estar vacía; `j` puede ser 0 y la tarea se inserta como única. |
| `ra != rb` | obligatorio | El código verifica `if ra == rb: continue` después de sortear. |
| Número de rutas | `>= 2` | Debe existir al menos una ruta distinta a `ra` para que `rb != ra`. |

---

## 6. Contraste "Pseudocódigo vs. realidad"

Un pseudocódigo ingenuo mostraría:

```
s' ← Copy(s)
e ← r'_a.pop(i)
r'_b.insert(j, e)
m ← (relocate_inter, a, b, i, j)
Return (s', m)
```

Lo que omitía:

- La normalización previa que elimina `"D"` de todas las rutas.
- La lógica de selección: `ra` se elige entre las rutas **no vacías** (`activos`), pero `rb` se sortea entre **todas** las rutas (incluidas las vacías), con un `continue` posterior si `ra == rb`.
- El `+1` en `rng.randrange(len(rutas[rb]) + 1)`: necesario para poder insertar al final de la ruta destino.
- El bucle de hasta 500 reintentos.
- La desnormalización final.

---

## 7. Pseudocódigo corregido en LaTeX

```latex
\begin{algorithm}
\DontPrintSemicolon
\caption{\texttt{relocate\_inter} --- Mover una tarea de una ruta a otra ruta distinta}
\KwIn{Solución con depósito $s$;\; generador aleatorio $\mathit{rng}$;\; máximo de intentos $T_{\max}=500$}
\KwOut{Vecino con depósito $s'$;\; registro de movimiento $m$}
$\hat{s} \leftarrow \textsc{NormalizarQuitarDepósito}(s)$
\tcp{elimina el token "D" de inicio y fin de cada ruta}
$t \leftarrow 0$\;
\Repeat{movimiento válido encontrado}{
    $t \leftarrow t + 1$\;
    \lIf{$t > T_{\max}$}{\textbf{lanzar} \texttt{RuntimeError}}
    $\mathcal{A} \leftarrow \{k : |\hat{r}_k| \geq 1\}$
    \tcp{rutas con al menos 1 tarea (candidatas como origen)}
    \lIf{$|\mathcal{A}| < 1$ \textbf{o} $|\hat{s}| < 2$}{\textbf{continue}}
    $\mathit{ra} \leftarrow \mathit{rng}.\textsc{choice}(\mathcal{A})$
    \tcp{ruta origen: uniforme entre rutas no vacías}
    $\mathit{rb} \leftarrow \mathit{rng}.\textsc{randrange}(|\hat{s}|)$
    \tcp{ruta destino: uniforme entre \textbf{todas} las rutas (puede estar vacía)}
    \lIf{$\mathit{ra} = \mathit{rb}$}{\textbf{continue} \tcp*{debe ser inter-ruta}}
    $i \leftarrow \mathit{rng}.\textsc{randrange}(|\hat{r}_{\mathit{ra}}|)$
    \tcp{posición en la ruta origen}
    $j \leftarrow \mathit{rng}.\textsc{randrange}(|\hat{r}_{\mathit{rb}}| + 1)$
    \tcp{posición en la ruta destino; $+1$ permite insertar al final}
    $\hat{s}' \leftarrow \textsc{CopiaProfunda}(\hat{s})$
    \tcp{Python pasa listas por referencia; sin copia se corrompería $\hat{s}$}
    $e \leftarrow \hat{r}'_{\mathit{ra}}.\textsc{pop}(i)$\;
    $\hat{r}'_{\mathit{rb}}.\textsc{insert}(j,\, e)$\;
    $m \leftarrow \textsc{MovimientoVecindario}(\texttt{relocate\_inter},\; \mathit{ruta\_a}=\mathit{ra},\; \mathit{ruta\_b}=\mathit{rb},\; i=i,\; j=j)$\;
    $s' \leftarrow \textsc{RestaurarDepósito}(\hat{s}')$
    \tcp{añade "D" al inicio y fin de cada ruta antes de devolver}
    \Return $(s',\, m)$\;
}
\end{algorithm}
```
