# Vecindarios revisados — MetaCARP

Este directorio contiene la documentación corregida de los nueve operadores de vecindario del proyecto MetaCARP. Su propósito es complementar los pseudocódigos LaTeX de la tesis mostrando exactamente lo que el código Python **realmente** hace alrededor del movimiento puro, cinco pasos que los pseudocódigos originales omitían.

> **Audiencia:** la autora de la tesis y cualquier lector que compare un pseudocódigo LaTeX con `metacarp/vecindarios.py` y necesite entender la brecha entre ambos.

---

## Índice de documentos

| Archivo | Contenido |
|---|---|
| `00_anatomia_de_un_operador.md` | Piezas compartidas por todos los operadores: `_copy_solution`, normalización, desnormalización, bucle de reintentos y `MovimientoVecindario`. |
| `01_intra_relocate.md` | `op_relocate_intra` — reubicación de una tarea dentro de la misma ruta. |
| `02_intra_swap.md` | `op_swap_intra` — intercambio de dos tareas dentro de la misma ruta. |
| `03_intra_2opt.md` | `op_2opt_intra` — inversión de un segmento dentro de la misma ruta. |
| `04_inter_relocate.md` | `op_relocate_inter` — traslado de una tarea a otra ruta. |
| `05_inter_swap.md` | `op_swap_inter` — intercambio de una tarea entre dos rutas. |
| `06_inter_2opt_star.md` | `op_two_opt_star` — intercambio de colas entre dos rutas. |
| `07_inter_cross_exchange.md` | `op_cross_exchange` — intercambio de segmentos entre dos rutas. |
| `08_09_or_opt.md` | `op_or_opt_2` y `op_or_opt_3` más la función genérica `_or_opt_k`. |

---

## Lo que el pseudocódigo original no mostraba

El pseudocódigo LaTeX original de la autora presentaba un operador como `relocate_intra` en tres líneas:

```
s' ← Copy(s)
e ← r'_a.pop(i)
r'_a.insert(j, e)
```

Eso es el **movimiento puro**. Pero en el código real, cada llamada a `generar_vecino` ejecuta cinco pasos adicionales que rodean ese movimiento. Ninguno de los cinco aparecía en el pseudocódigo.

### Los cinco pasos ocultos

**Paso 1 — Normalización (quita-depósito).**
Antes de que ningún operador toque la solución, `generar_vecino` llama a `normalizar_para_vecindario` ([vecindarios.py:111](../../metacarp/vecindarios.py#L111)). Esta función recorre cada ruta y elimina todos los tokens iguales al marcador de depósito `"D"`. El operador opera exclusivamente sobre etiquetas de tareas puras (`TR1`, `TR2`, ...). Sin esta normalización, el `pop(i)` podría extraer un `"D"` en lugar de una tarea real.

```
Entrada a generar_vecino:  [["D","TR1","TR3","D"], ["D","TR2","TR4","D"]]
Tras normalizar:           [["TR1","TR3"],          ["TR2","TR4"]]
```

**Paso 2 — Filtrado de candidatas.**
Antes de elegir índices, el código filtra las rutas que cumplen el requisito mínimo del operador seleccionado. Por ejemplo, `relocate_intra` exige `len(ruta) >= 2`; `2opt_intra` exige `len(ruta) >= 3`. Si ninguna ruta cumple el requisito, se hace `continue` y se reintenta el bucle con otro operador.

**Paso 3 — Bucle de reintentos (hasta 500).**
`generar_vecino` no devuelve `null` ante un operador inaplicable: ejecuta un bucle `while True` con un contador de intentos. Si el operador elegido no es aplicable —ya sea porque no hay rutas candidatas, o porque `i == j` en `relocate_intra`, o porque no hay margen para `or_opt_k` en caso intra—, el código ejecuta `continue` y sortea de nuevo. El límite es 500 intentos; si se supera, se lanza `RuntimeError`. El pseudocódigo ingenuo mostraba `Return null` donde el código real reintenta.

**Paso 4 — Copia profunda.**
Cada función `op_*` llama internamente a `_copy_solution` ([vecindarios.py:158](../../metacarp/vecindarios.py#L158)) al inicio. Python pasa listas por referencia; sin una copia independiente, el `pop(i)` o el `ruta[i:j+1] = ...` modificaría la solución original en memoria, corrompiendo el estado del algoritmo.

**Paso 5 — Registro del movimiento y desnormalización.**
Tras aplicar el operador, se construye un `MovimientoVecindario` con el nombre del operador y los índices usados. Si se está en `generar_vecino_ids`, también se extraen los `id_movidos` con `_moved_ids` y, si hay `encoding`, se decodifican las `labels_movidos`. Finalmente, `desnormalizar_con_deposito` ([vecindarios.py:139](../../metacarp/vecindarios.py#L139)) restaura el `"D"` al inicio y al final de cada ruta antes de devolver el vecino.

```
Vecino normalizado:    [["TR3","TR1"], ["TR2","TR4"]]
Tras desnormalizar:    [["D","TR3","TR1","D"], ["D","TR2","TR4","D"]]
```

---

## Ciclo de vida completo de una llamada a `generar_vecino`

El siguiente diagrama muestra el flujo real, incluyendo los cinco pasos ocultos. Las líneas de puntos delimitan lo que ocurre dentro de cada función.

```
Llamador: generar_vecino(solucion, rng=rng, operadores=[...])
│
│  solucion = [["D","TR1","TR3","D"], ["D","TR2","TR4","D"]]
│
├─► normalizar_para_vecindario()
│      Elimina "D" de inicio y fin de cada ruta
│      rutas = [["TR1","TR3"], ["TR2","TR4"]]
│
├─► while True  (bucle de hasta 500 intentos)
│   │
│   ├─► rng.choice(ops)  →  op elegido, p. ej. "relocate_intra"
│   │
│   ├─► _rutas_con_indices(rutas)  →  activos = [0, 1]
│   │
│   ├─► Filtrar candidatas: cand = [r for r in activos if len(rutas[r]) >= 2]
│   │      cand = [0, 1]
│   │
│   ├─► rng.choice(cand)  →  r = 0
│   │   rng.randrange(n)  →  i = 0
│   │   rng.randrange(n)  →  j = 1
│   │   ¿i == j?  →  No  →  continuar
│   │
│   └─► op_relocate_intra(rutas, r=0, i=0, j=1)
│          │
│          ├─► _copy_solution(rutas)
│          │      s = [["TR1","TR3"], ["TR2","TR4"]]  (copia independiente)
│          │
│          ├─► s[0].pop(0)  →  "TR1",  s[0] = ["TR3"]
│          │
│          ├─► s[0].insert(1, "TR1")  →  s[0] = ["TR3","TR1"]
│          │
│          └─► return s = [["TR3","TR1"], ["TR2","TR4"]]
│
├─► mov = MovimientoVecindario("relocate_intra", ruta_a=0, i=0, j=1)
│
├─► (backend ids: _moved_ids + decode_task_ids)
│
└─► desnormalizar_con_deposito(vec)
       Añade "D" al inicio y fin de cada ruta
       return [["D","TR3","TR1","D"], ["D","TR2","TR4","D"]], mov
```

---

## Referencia al código fuente

Todos los documentos de esta carpeta citan el archivo:

- [`metacarp/vecindarios.py`](../../metacarp/vecindarios.py) — fuente de verdad única.

Los números de línea referenciados corresponden al archivo tal como está en el repositorio en el momento de redactar esta documentación. Si el archivo se modifica, verificar la coincidencia con los bloques citados.
