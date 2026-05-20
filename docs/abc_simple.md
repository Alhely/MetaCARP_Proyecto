# ABC Simple (Karaboga 2005 canónico) para CARP

## 1. Introducción y motivación

`busqueda_abejas_simple` implementa el algoritmo **Artificial Bee Colony (ABC)** de **Karaboga (2005)** en su versión más fiel al artículo original, ajustada al problema CARP. Su propósito es servir como **baseline didáctico** y referencia de comparación frente a la versión más elaborada del proyecto (`busqueda_abejas`), que ya incorpora varios añadidos heurísticos (sesgo inter/intra-ruta a lo largo de las tres fases, rastreo separado de mejor factible, scouts dirigidos hacia la mejor solución, etc.).

La intención de tener dos implementaciones en el repositorio responde a un argumento metodológico de la tesis: aislar cuánto del rendimiento observado en `busqueda_abejas` proviene del **núcleo ABC** y cuánto de las **modificaciones heurísticas** acumuladas. `busqueda_abejas_simple` permite responder esa pregunta con un experimento controlado.

### Diferencias clave con `busqueda_abejas`

| Aspecto | `busqueda_abejas` (extendida) | `busqueda_abejas_simple` (canónica) |
|---|---|---|
| Scouts | Vecino de la mejor fuente actual | **Solución aleatoria pura** (greedy por capacidad) |
| Sesgo inter/intra | `pesos_inter_bias` en empleadas, observadoras y scouts | `seleccionar_grupo_operadores_inter_intra` solo en empleadas y observadoras |
| Mejor factible | Rastreado separado del mejor general | Un único mejor (penalización guía la búsqueda) |
| Imputación `registrar_mejora` | Comparada al final del ciclo, imputada al último movimiento aceptado | Comparada **en el mismo `if`** del vecino aceptado |
| Criterio de parada | Solo `iteraciones` (tope duro) | `iteraciones` + `max_iter_sin_mejora` opcional |
| Backend de generación | Configurable (`labels` o `ids`) | Siempre `ids` (más rápido y consistente) |

---

## 2. Los tres roles de abejas

| Tipo | Cantidad por ciclo | Fuente que visita | Acción |
|---|---|---|---|
| **Empleadas** | `num_fuentes` | Su fuente asignada (siempre la misma) | Genera un vecino; reemplaza la fuente si mejora |
| **Observadoras** | `num_fuentes` | Elegidas por ruleta sobre `1/(1+obj)` | Genera un vecino; reemplaza la fuente si mejora |
| **Scouts** | Variable | Fuente con `trials[i] >= limite_abandono` | Reemplaza la fuente con una **solución aleatoria** |

La diferencia conceptual fundamental: las empleadas explotan localmente, las observadoras **concentran esfuerzo en las mejores** (ruleta), los scouts **diversifican** abandonando regiones agotadas.

---

## 3. Modificaciones sobre Karaboga (2005)

| Categoría | Decisión | Justificación |
|---|---|---|
| **Se MANTIENE** | Tres fases empleadas → observadoras → scouts | Estructura nuclear del ABC |
| **Se MANTIENE** | Ruleta por fitness inverso `1/(1+obj)` en observadoras | Fórmula original Karaboga |
| **Se MANTIENE** | Scouts puramente aleatorios | El sello distintivo del ABC canónico |
| **Se MANTIENE** | Comparación greedy en empleadas y observadoras | Misma regla de actualización del paper |
| **Se SIMPLIFICA** | Una sola lista `mejor_sol_ids` (no se rastrea mejor factible aparte) | Penalización ya guía la búsqueda hacia región factible; segundo rastreo añadía complejidad sin beneficio en instancias estudiadas |
| **Se SIMPLIFICA** | Backend de generación fijo en IDs | Evita la rama dual `labels`/`ids` que complica el código sin acelerar nada relevante |
| **Se AÑADE** | Sesgo `seleccionar_grupo_operadores_inter_intra` en empleadas y observadoras | Sin él, ABC era incapaz de reparar capacidad en instancias con violación inicial. Es el mismo helper que usan SA / TS simple / RTS, lo cual asegura **comparabilidad metodológica** |
| **Se AÑADE** | `max_iter_sin_mejora` opcional | Permite parar pronto sin ejecutar las `iteraciones` completas; ahorra tiempo en repeticiones con estancamiento temprano |
| **Se CORRIGE** | `registrar_mejora` se invoca **dentro del mismo `if`** que detecta vecino mejor que el global | En `busqueda_abejas`, la comparación se hace al final del ciclo contra `ultimo_movimiento_aceptado`, que puede no ser el operador real responsable. Aquí `sum(operadores_mejoraron.values()) ≤ mejoras` siempre, con la diferencia exacta atribuida a scouts (que no tienen operador asociado) |

---

## 4. Firma de `busqueda_abejas_simple`

```python
busqueda_abejas_simple(
    inicial_obj,             # objeto inicial (lista de soluciones candidatas o pickle)
    data,                    # diccionario de datos de la instancia CARP
    G,                       # grafo NetworkX
    *,
    iteraciones=300,
    num_fuentes=20,
    limite_abandono=30,
    max_iter_sin_mejora=None,
    semilla=None,
    operadores=OPERADORES_POPULARES,
    marcador_depot_etiqueta=None,
    usar_gpu=False,
    guardar_historial=True,
    guardar_csv=False,
    ruta_csv=None,
    nombre_instancia="instancia",
    repeticion=None,
    root=None,
    usar_penalizacion_capacidad=True,
    lambda_capacidad=None,
    alpha_inter=0.8,
    p_inter=0.6,
    extra_csv=None,
    **_ignorado_kwargs,
) -> AbejasSimpleResult
```

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `iteraciones` | int | 300 | Número de ciclos completos (empleadas + observadoras + scouts) |
| `num_fuentes` | int | 20 | Cuántas fuentes (soluciones activas) en paralelo |
| `limite_abandono` | int | 30 | Intentos fallidos consecutivos antes de mandar fuente al scout |
| `max_iter_sin_mejora` | int \| None | None | Si se especifica, criterio de parada anticipada por estancamiento |
| `semilla` | int \| None | None | None = aleatoria del sistema (para experimentos paralelos) |
| `operadores` | Iterable[str] | `OPERADORES_POPULARES` | Subset de los 9 operadores activos |
| `usar_gpu` | bool | False | Si True, evaluación de lote intenta usar CuPy |
| `usar_penalizacion_capacidad` | bool | True | Si True, objetivo = costo + λ × violación |
| `lambda_capacidad` | float \| None | None | None = automático (~10× mediana deadhead) |
| `alpha_inter` | float [0,1] | 0.8 | P(elegir inter) cuando hay violación |
| `p_inter` | float [0,1] | 0.6 | P(elegir inter) cuando la solución es factible |

---

## 5. `AbejasSimpleResult` — campos del resultado

| Campo | Tipo | Significado |
|---|---|---|
| `mejor_solucion` | `list[list[str]]` | Mejor solución encontrada en formato de etiquetas |
| `mejor_costo` | `float` | Costo PURO (sin penalización) de la mejor solución |
| `solucion_inicial_referencia` | `list[list[str]]` | Solución inicial elegida como referencia |
| `costo_solucion_inicial` | `float` | Costo puro de la solución inicial |
| `mejora_absoluta` | `float` | `costo_inicial − costo_mejor` |
| `mejora_porcentaje_inicial_vs_final` | `float` | Mejora porcentual respecto al inicial |
| `tiempo_segundos` | `float` | Tiempo total de ejecución |
| `iteraciones_totales` | `int` | Iteraciones realmente ejecutadas (≤ `iteraciones`) |
| `iteraciones_sin_mejora_final` | `int` | Contador al cierre del criterio de estancamiento |
| `fuentes_alimento` | `int` | = `num_fuentes` (replicado por trazabilidad) |
| `scouts_reinicios` | `int` | Veces que una fuente fue reemplazada por aleatoria |
| `mejoras` | `int` | Veces que el mejor global mejoró |
| `semilla` | `int \| None` | Semilla efectiva del RNG |
| `backend_evaluacion` | `str` | `"cpu"` o `"gpu"` |
| `usar_penalizacion_capacidad` | `bool` | Si la penalización estuvo activa |
| `lambda_capacidad` | `float` | λ efectivo aplicado |
| `mejor_solucion_factible_final` | `bool` | True si la mejor solución es factible |
| `aceptaciones_solucion_infactible` | `int` | Cuántas aceptaciones aterrizaron en vecino infactible |
| `iteraciones_con_violacion` | `int` | Ciclos en los que la violación media de fuentes era > 0 |
| `fraccion_iter_con_violacion` | `float` | `iteraciones_con_violacion / iteraciones_totales` |
| `operadores_propuestos` | `dict[str,int]` | Conteo por operador (categoría: propuesto) |
| `operadores_aceptados` | `dict[str,int]` | Conteo por operador (categoría: aceptado) |
| `operadores_mejoraron` | `dict[str,int]` | Conteo por operador (categoría: mejoraron el mejor global) |
| `operadores_trayectoria_mejor` | `dict[str,int]` | Snapshot del momento de la última mejora |
| `historial_mejor_costo` | `list[float]` | Trayectoria del mejor costo por iteración (si `guardar_historial=True`) |
| `archivo_csv` | `str \| None` | Ruta del CSV generado, o None |

---

## 6. Ejemplo de uso básico

```python
from metacarp import busqueda_abejas_simple_desde_instancia

resultado = busqueda_abejas_simple_desde_instancia(
    "gdb1",
    iteraciones=200,
    num_fuentes=15,
    limite_abandono=25,
    alpha_inter=0.8,
    p_inter=0.6,
)

print(f"Mejor costo: {resultado.mejor_costo:.2f}")
print(f"Mejora vs inicial: {resultado.mejora_porcentaje_inicial_vs_final:.2f} %")
print(f"Iteraciones reales: {resultado.iteraciones_totales}")
print(f"Scouts disparados: {resultado.scouts_reinicios}")
print(f"Factible final: {resultado.mejor_solucion_factible_final}")
```

Para parar pronto cuando deja de mejorar:

```python
resultado = busqueda_abejas_simple_desde_instancia(
    "gdb1",
    iteraciones=500,
    num_fuentes=20,
    max_iter_sin_mejora=50,   # corta tras 50 ciclos sin mejora
)
```

---

## 7. Guía de parámetros del grid search

El script `scripts/run_abc_simple_automatico.py` barre tres dimensiones manteniendo `alpha_inter = 0.8` fijo:

| Dimensión | Valores barridos | Significado |
|---|---|---|
| `num_fuentes` | `{10, 20, 30}` | Tamaño de la población activa. Más fuentes = más diversidad pero más coste por ciclo. La literatura típica usa 10-50. |
| `limite_abandono` | `{20, 35, 50}` | Cuán paciente es el algoritmo antes de reiniciar una fuente. Karaboga sugiere `dim·SN/2` como heurística; nuestro barrido cubre desde "agresivo" (20) hasta "muy paciente" (50). |
| `p_inter` | `{0.4, 0.5, 0.6, 0.7, 0.8}` | Sesgo hacia operadores inter-ruta en estado factible. Estudia si ABC se beneficia de más exploración inter-ruta (>0.6) o más refinamiento intra (<0.6). |

`alpha_inter` se mantiene en **0.8** por instrucción explícita: cuando hay violación queremos garantizar **al menos 80%** de probabilidad de elegir un operador inter-ruta (que es el único capaz de reparar capacidad).

Total del grid: 3 × 3 × 5 × 23 instancias × 5 repeticiones = **5,175 corridas**.

Reducir el grid a una sola dimensión es trivial:

```bash
# Fijar p_inter = 0.6 → solo barre num_fuentes × limite_abandono (45 corridas/instancia).
python scripts/run_abc_simple_automatico.py --p-inter 0.6
```

---

## 8. Por qué la GPU SOLO acelera la evaluación

La generación de vecinos opera sobre listas de Python con operadores combinatorios (`relocate_inter`, `swap_intra`, `2opt_star`, etc.). Estas operaciones son **secuenciales y data-dependent**: cada operador decide qué hacer en función de la longitud de las rutas y de elecciones aleatorias. No hay matrices ni reducciones vectoriales que paralelizar.

La **evaluación**, en cambio, sí es trivialmente paralelizable: cada vecino se reduce a una suma de distancias precomputadas (`dist[u, v] + costo_servicio`). La función `costo_lote_penalizado_ids` empaqueta los `num_fuentes` vecinos de la fase observadoras en arrays planos y delega la reducción a NumPy (CPU) o CuPy (GPU). Con GPU la aceleración es notable en instancias grandes y con `num_fuentes` alto; en instancias pequeñas (n < 50) el overhead de copiar arrays a la GPU puede dominar y conviene dejar `usar_gpu=False`.

En las **empleadas** evaluamos vecinos individualmente con `costo_rapido_ids` (escalar) porque la decisión de aceptar/rechazar se toma por fuente y no se beneficiaría de vectorizar (cada decisión depende del resultado anterior). Si se quisiera vectorizar también esta fase habría que rediseñarla como "best-of-N" en lugar de "greedy por fuente", lo cual cambiaría la semántica del algoritmo y lo alejaría aún más del ABC canónico.
