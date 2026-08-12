# MetaCARP — Historial Completo del Proyecto

**Periodo cubierto:** 11 de mayo de 2026 – 15 de julio de 2026
**Última actualización:** 15 de julio de 2026

Este documento reconstruye, en orden cronológico y con el mayor detalle
posible, todos los esfuerzos realizados en el proyecto MetaCARP: implementación
de metaheurísticas, calibración por grid search, campañas experimentales,
hallazgos de investigación (incluyendo la resolución de una discrepancia de
escala de costos que en un primer momento parecía un error del evaluador), y
la generación de material listo para publicación. Sirve como bitácora de
referencia para escribir la tesis/artículo y para auditar de dónde sale cada
cifra reportada.

---

## 1. Resumen ejecutivo

El proyecto implementa y compara **seis metaheurísticas** para el Problema de
Ruteo de Arcos con Capacidad (CARP): Recocido Simulado (SA), Búsqueda Tabú
simple (TS), Búsqueda Tabú Reactiva (RTS), Colonia Artificial de Abejas (ABC),
Cuckoo Search (CS) y Vibration Damping Optimization (VDO). Todas comparten
la misma infraestructura: 9 operadores de vecindario (3 intra-ruta, 6
inter-ruta), un objetivo penalizado por violación de capacidad, y — en la
fase final del proyecto — un protocolo experimental uniforme (presupuesto de
10⁶ evaluaciones o 300 s, lo primero que ocurra; Path Relinking como
mecanismo de intensificación).

El corpus de instancias final consta de **87 instancias** de CARPLIB:
23 pequeñas (17 `gdb` + 6 `kshs`) usadas como conjunto de calibración, y 64
de transferencia (34 `val`, 6 `gdb` adicionales, 24 `egl`) usadas para medir
qué tan bien escalan los métodos calibrados.

**Resultado headline** (gap medio contra BKS, corpus completo de 87
instancias, mejor de 5 repeticiones por instancia):

| MH | Gap medio | Mediana | BKS alcanzadas | Victorias (mejor/empate) |
|---|---|---|---|---|
| **SA** | **6.15%** | 0.85% | **34/87** | **67/87** |
| CS | 7.71% | 7.33% | 9/87 | 13/87 |
| TS | 9.30% | 6.03% | 19/87 | 2/87 |
| RTS | 9.49% | 4.71% | 23/87 | 3/87 |
| ABC | 17.29% | 5.65% | 18/87 | 2/87 |
| VDO | 28.29% | 13.82% | 7/87 | 0/87 |

SA es el método dominante en el corpus completo y en la clase val (gap medio
1.73%), pero pierde el liderato frente a Cuckoo Search en la clase egl grande
(SA 19.53% vs CS 12.3%), lo que motivó un análisis de operadores que explica
la causa: la aceptación de movimientos inter-ruta colapsa con el tamaño de
instancia (7.5% en gdb → 2.7% en val), dejando que los operadores intra-ruta
(2-opt, swap, relocate) carguen con más del 80–90% de la trayectoria que
produce la mejor solución.

---

## 2. Cronología detallada

### Fase 1 — Fundamentos e implementación inicial (11–20 de mayo de 2026)

- **11 de mayo**: commit inicial del repositorio. Se sube documentación en
  Markdown y los módulos de código de las metaheurísticas ya existentes.
  Aplanamiento de la estructura (`metacarp/` a la raíz). Primer `.gitignore`
  (excluye CSVs de resultados del control de versiones — convención que se
  mantiene todo el proyecto).
- **12 de mayo**: se agregan comparaciones de operadores con tablas y
  gráficos en notebook de análisis. Se simplifican y luego se enriquecen las
  columnas del CSV de resultados (metadatos, `mejor_costo`,
  `costo_solucion_inicial`, `gap_bks_porcentaje`, columnas de solución). Se
  corrige un bug de sesgo de operadores: el sistema debía preferir
  movimientos **inter-ruta** cuando hay violación de capacidad (antes el
  sesgo estaba invertido) — esta corrección es la que luego se formaliza como
  la regla `P(inter) = max(p_inter, 0.8)` bajo violación, usada por las 6 MH
  hasta el final del proyecto.
- **13 de mayo**: se reduce el grid search inicial de 354 a 72 configuraciones
  (3,312 corridas totales) por razones de costo computacional. Primeros
  pseudocódigos LaTeX en inglés de los 4 metaheurísticos existentes en ese
  momento.
- **14–15 de mayo**: refactorización de SA — calibración adaptativa
  instance-aware, criterio de parada clásico, corrección de rutas. Se
  experimenta con reheat agresivo (α=0.90, p_inter=0.65 fijos), se
  simplifica quitando el reheat temporalmente, y finalmente se **restaura el
  reheat con criterio de parada por reheats consecutivos sin mejora** — el
  diseño que se mantiene hasta el final del proyecto.
- **17–20 de mayo**: pseudocódigos LaTeX de SA y de los 9 operadores.
  Implementación de **Búsqueda Tabú simple y Reactiva** (TS, RTS) con
  soporte de multiprocessing; ampliación del grid de RTS a 4 dimensiones
  (factor_aumento × umbral_escape × p_inter × factor_reduccion). Se
  implementa **ABC** (`abejas_simple.py`, fiel a Karaboga 2005 con las
  extensiones CARP del proyecto: sesgo inter/intra compartido).

### Fase 2 — Cuckoo Search y exploración de variantes (20–29 de mayo de 2026)

- **20 de mayo**: **Cuckoo Search** se moderniza a las convenciones
  instance-aware del proyecto (discretización de vuelos de Lévy, ver
  Sección 4). Documentación y pseudocódigos LaTeX completos de las 5 MH
  existentes hasta ese punto (SA, TS, RTS, ABC, CS). Primer script de
  consolidación de resultados de las 5 MH.
- **23–25 de mayo**: notebook de análisis exploratorio de operadores,
  conclusiones y biblioteca de soluciones. Se implementan y evalúan varias
  **variantes experimentales** del mecanismo de selección de operadores y de
  intensificación, cada una con su propio notebook de análisis:
  - `strict_intra_inter_20260524`: variante de sesgo inter/intra más estricta,
    con mecanismo de "kick" (perturbación disruptiva) por estancamiento.
  - `lambda_grid_20260525`: grid de calibración del peso de penalización de
    capacidad λ, expuesto explícitamente en TS/RTS.
  - Corrección de bug: TS simple y RTS no estaban optimizando el objetivo
    penalizado (costo + λ·violación) correctamente — corregido el 25 de mayo.
  - `vecindarios_p_inter_exp_2026050524`: variante posicional del sesgo de
    selección de operadores.
- **27–29 de mayo**: dos variantes adicionales de intensificación evaluadas
  en notebooks propios:
  - `aos_pm_20260527` (Adaptive Operator Selection — Probability Matching):
    pesos adaptativos por operador según su tasa de éxito reciente.
  - `path_relinking_20260528`: primera versión experimental de **Path
    Relinking** como mecanismo de intensificación (caminata truncada desde la
    solución actual hacia la mejor global, guardando el mejor intermedio).
  - `p_inter_pr_20260528`: combinación de Path Relinking con el sesgo
    p_inter fijo.
  - `budget_20260528`: experimento de sensibilidad al presupuesto
    (evaluaciones ×25, λ ×10) para entender el comportamiento a mayor escala
    de cómputo.
  - `warmstart_greedy_20260529`: primera versión de arranque cálido con
    Path Scanning (PS) + evaluador greedy, y re-corrida de baselines (R1–R5)
    bajo el nuevo evaluador.
  - Redacción del primer borrador del capítulo de Experimentación (nivel
    tesis de maestría), con resultados completos R1–R5 y citas
    bibliográficas — 29 de mayo.

### Fase 3 — Corrección del evaluador y "costo corregido" (30 de mayo – 3 de junio de 2026)

- **30 de mayo**: se integra la **orientación greedy** como única lógica del
  evaluador de costos (unifica el criterio de orientación de aristas
  requeridas usado en todas las metaheurísticas). Script de corrida
  definitiva bajo este evaluador corregido (`correct_cost`).
- **31 de mayo**: se diseña el **programa de approaches con costo
  corregido**: tres variantes experimentales comparadas de forma sistemática
  sobre las 23 instancias pequeñas:
  - `solo_p_inter`: solo el sesgo de selección de operadores.
  - `binario_capacidad`: penalización binaria de violación de capacidad.
  - `pr_aislado`: Path Relinking limpio como único mecanismo de
    intensificación, aislado de otras variantes.
- **1 de junio**: se fija p_inter y se simplifican los 3 approaches a una
  configuración canónica única por MH. Se calibra el "segundo parámetro más
  influyente" de cada MH (grid 1D adicional tras fijar el principal) y se
  incorpora a la configuración canónica.
- **2 de junio**: se documenta la calibración de los parámetros restantes y
  el experimento canónico puro. Se explicitan las distribuciones de
  probabilidad de cada sorteo aleatorio del sistema (para reproducibilidad
  exacta). Se corrigen y documentan los 9 operadores de vecindario fiel al
  código. Se exportan los resultados de los 3 approaches a archivos de texto
  consolidados y se generan tablas exhaustivas para el capítulo de tesis.
- **3 de junio**: documento final extenso de resultados bajo costo
  corregido, convertido a LaTeX compilable para Overleaf. **Este es el cierre
  de la campaña de junio**, cuyos resultados (23 instancias pequeñas × 5 MH
  × 3 approaches × 5 repeticiones) siguen siendo la fuente de la calibración
  usada en julio.

*(Pausa de desarrollo entre el 3 de junio y el 9 de julio.)*

### Fase 4 — Vibration Damping Optimization y campaña de transferencia (9–12 de julio de 2026)

- **9 de julio**: se implementa la **sexta metaheurística, Vibration Damping
  Optimization (VDO)**, un análogo físico del SA que reemplaza la regla de
  Metropolis por la CDF de una distribución de Rayleigh
  (`p = 1 − exp(−A²/2σ²)`, constante durante todo el nivel de amplitud) y el
  enfriamiento geométrico por una ley de amortiguamiento de oscilador
  (`A(t) = A₀·exp(−γt/2)`). Reutiliza toda la infraestructura de operadores y
  evaluación de costo del proyecto. Se agrega también un *dispatcher*
  explícito de `metodo_seleccion` inter/intra en las utilidades compartidas.
- **10 de julio**: se diseña y ejecuta la **campaña val/egl** — el primer
  despliegue de SA/TS/RTS/ABC bajo un **presupuesto uniforme** (10⁶
  evaluaciones o 300 s, lo que ocurra primero) sobre las 64 instancias de
  transferencia (34 val + 6 gdb + 24 egl), con 5 repeticiones por instancia
  y semillas deterministas. Se corre en paralelo un **mini-grid de Cuckoo
  Search** con arranque cálido (warm-start) de Path Scanning sobre 3
  instancias representativas (val1A, val5C, egl-e1-A) para encontrar la
  configuración de transferencia óptima antes del despliegue completo.
  Documentación LaTeX por MH con detalle de grid search, y primeras tablas
  de mejores valores.
- **11–12 de julio**: se despliega la **campaña completa de Cuckoo Search**
  (config ganadora del mini-grid: `factor_pasos=0.25`, `p_inter=0.6`,
  warm-start Path-Scanning mejor-de-5) sobre las 64 instancias de
  transferencia, y se generan las tablas de mejores valores.
  - **Se detecta una discrepancia**: los gaps calculados en la familia `val`
    parecían indicar costos por debajo de lo físicamente posible o
    incoherencias severas (10–33% de "undercounting" aparente en las 34
    instancias val). La hipótesis inicial fue un error en los datos de
    instancia o en el evaluador.
  - **Investigación de causa raíz** (documentada en detalle en la Sección 5):
    se descarta el evaluador (verificado independientemente con un algoritmo
    de programación dinámica de orientaciones sobre el grafo crudo), se
    descartan los archivos de instancia (verificados byte-a-byte contra la
    fuente oficial de Valencia y contra el repositorio DIMACS HGS-CARP), y
    se identifica la causa real: el formato CARPLIB distingue dos costos por
    arista requerida (costo de *servicio* vs. costo de *tránsito*), y la
    familia `val` es la única donde ambos difieren. Se implementa la
    corrección de reporte `costo_comparable = costo + δ`.
  - Se genera la tabla LaTeX de resultados de Cuckoo Search por instancia con
    parámetros de la corrida ganadora (87 instancias, español).

### Fase 5 — Material de artículo en inglés (12–13 de julio de 2026)

- **12 de julio**: pseudocódigo de Cuckoo Search en inglés (`algorithm2e`),
  tabla de resultados en inglés. **Reorganización completa de la
  documentación del repositorio** en `docs/{markdown,latex}/{es,en}/` con
  historial de git preservado (21 archivos Markdown + 16 LaTeX en español
  movidos, 4 LaTeX en inglés reubicados). Tabla en inglés de los 9
  operadores de vecindario para artículo.
- **12–13 de julio**: se corrige la notación del pseudocódigo de Cuckoo a la
  convención original de Yang & Deb (2009) — el exponente de Lévy pasa de β
  a λ, y el peso de penalización de capacidad se renombra a ω para evitar el
  choque de símbolos. Se clarifica el presupuesto de evaluación/tiempo de
  forma explícita en el pseudocódigo (`evals < 10⁶ and time < 300 s`).
- **13 de julio**: se agregan las **soluciones explícitas como arreglos de
  arcos** `(u,v)` (en vez de marcadores `TRk`) a las tablas de resultados de
  Cuckoo Search, verificando que el multiconjunto de arcos de cada solución
  reproduce exactamente las aristas requeridas de su instancia (0
  discrepancias en 87 instancias). Se genera el mismo material — subsección
  de artículo con pseudocódigo, tabla de resultados con parámetros, tabla de
  soluciones explícitas en landscape — para **SA, TS, RTS y ABC**. Tablas
  adicionales en formato Word (.docx) para TS.

### Fase 6 — Campaña de VDO y análisis de operadores (14–15 de julio de 2026)

- **14 de julio**: se extiende el módulo VDO con los kwargs de campaña que
  le faltaban (`tiempo_limite_segundos`, `max_iter_sin_mejora_kick` +
  `intensificador` para Path Relinking en estancamiento, `max_resets`),
  siguiendo exactamente el patrón ya validado en SA. Se extiende el runner
  de la campaña val/egl para soportar `--solo-mh vdo`. Smoke-test aprobado.
  Se lanza la **campaña completa de VDO**: 320 corridas en las 64
  instancias val/gdb/egl + 115 corridas en las 23 pequeñas (nuevo script
  dedicado), mismo presupuesto uniforme, Path Relinking en estancamiento.
  **435/435 corridas exitosas, 0 fallos.**
  - Resultado: VDO es la metaheurística más débil del corpus (gap medio
    28.29%, 7/87 BKS, 0 victorias). Causa estructural identificada: con
    γ=0.05 fijo, en instancias grandes el presupuesto de evaluaciones se
    agota mucho antes de que la amplitud A decaiga por debajo del umbral
    A_min, dejando el mecanismo aceptando empeoramientos casi hasta el
    final de la corrida.
- **15 de julio**: se regeneran todas las tablas comparativas incluyendo VDO
  como sexta columna. **Decisión de alcance del artículo**: el paper se
  enfoca exclusivamente en **Recocido Simulado**; las demás 5 MH pasan a ser
  baselines de comparación.
  - Se genera el **análisis de operadores de vecindario para SA**,
    comparando la clase grande `egl` contra el baseline pequeño `gdb`: se
    cuantifica, por operador, el % de propuestas, tasa de aceptación,
    mejoras del incumbente por millón de propuestas, y % de participación en
    la trayectoria que produjo la mejor solución final. Se replica el mismo
    análisis para la clase `val`.
  - **Hallazgo central**: el ranking de operadores es estable entre clases
    (2-opt intra > swap intra > relocate intra >> operadores inter-ruta,
    con Or-opt-2/3 casi inertes), pero la tasa de aceptación de movimientos
    **inter-ruta colapsa monótonamente con el tamaño de instancia**: 7.5%
    (gdb) → 2.7% (val) → 1.9% (egl). La regla de transferencia que subía
    `p_inter` a 0.6 en clases grandes (razonando "más rutas ⇒ más peso
    inter-ruta") resulta contraproducente: una fracción creciente del
    presupuesto de evaluación se gasta en propuestas que Metropolis rechaza
    casi siempre.
  - Se depuran y corrigen dos pseudocódigos de SA que no compilaban en
    Overleaf (uso de `\mathbbm{1}` sin el paquete `bbm`; sintaxis inestable
    de `\leIf`/comentarios en macros de una línea de `algorithm2e`) y se
    corrige el valor de `T_min` reportado en la documentación (era
    `10⁻³`, el valor real instance-aware es `20·d_max/n²`).
  - Se generan tablas de resultados y de soluciones explícitas restringidas
    a `egl` y a `val` para SA, incluyendo variantes en landscape con
    márgenes mínimos, y se redactan en inglés los párrafos de calibración,
    reheating y las conclusiones enfocadas en SA (sin mencionar egl, por
    decisión de alcance del paper).
  - Se compila este documento de historial y el Excel consolidado de
    mejores resultados por instancia y metaheurística.

---

## 3. Arquitectura común a las seis metaheurísticas

Todas las MH del proyecto comparten:

- **Representación de solución**: lista de rutas (secuencias de tareas
  requeridas), con el depósito implícito en los extremos de cada ruta.
- **9 operadores de vecindario** (Tabla de operadores del artículo):
  3 intra-ruta (`relocate`, `swap`, `2-opt`) y 6 inter-ruta (`relocate`,
  `swap`, `2-opt*`, `cross-exchange`, `Or-opt-2`, `Or-opt-3`).
- **Selección de grupo de operador**: se elige el grupo inter-ruta con
  probabilidad `P(inter) = max(p_inter, 0.8)` si la solución actual viola
  capacidad, o `p_inter` en régimen factible; el operador específico dentro
  del grupo se sortea uniformemente.
- **Objetivo penalizado**: `f(s) = costo(s) + λ · violación_capacidad(s)`,
  con `λ = max(10·mediana(d), 10)` derivado de la matriz de distancias.
- **Path Relinking** (`path_relinking_limpio_20260531.py`) como mecanismo de
  intensificación: al estancarse (umbral configurable, 30 iteraciones/niveles
  en la campaña final), camina greedy desde la solución actual hacia la
  mejor global, reasignando en cada paso la tarea que minimiza el objetivo
  penalizado, y devuelve el mejor intermedio observado (nunca empeora).
- **Protocolo experimental final** (campaña julio 2026): presupuesto
  uniforme de 10⁶ evaluaciones de costo o 300 s de pared (lo primero que
  ocurra), 5 repeticiones por instancia, semillas deterministas derivadas de
  `(semilla_base, instancia, MH, repetición)`.

---

## 4. Detalle de cada metaheurística

### 4.1 Recocido Simulado (SA)

Metropolis clásico con temperatura constante por "plateau" (cadena de Markov
de `L = n²` iteraciones), enfriamiento geométrico `T ← αT`, y **reheating**:
tras `patience = 10` niveles sin mejorar el incumbente, `T` se restaura a
`ρ·T_init` (ρ=0.5); la búsqueda se detiene tras `R = 10` reheats consecutivos
sin mejora global, o al llegar a `T_min`, o al agotar el presupuesto.
Calibración instance-aware: `T_init = 20·d_max/n`, `T_min = 20·d_max/n²`,
`L = n²`.

**Grid search** (23 instancias pequeñas, mayo–junio 2026):
- Grilla principal: α ∈ {0.80,…,0.99} (11 valores) × p_inter ∈
  {0.4,…,0.8} (5 valores) = 55 combinaciones × 23 instancias × 2
  repeticiones = 2,530 corridas. Ganador: α=0.9, p_inter=0.5.
- Segundo knob: `max_reheats_sin_mejora` ∈ {3,5,10}; ganó 10 (gap medio
  2.90% vs 3.08% y 3.20%).
- Resultado en las 23 pequeñas: gap medio 0.08%, 21/23 BKS — la mejor MH de
  la campaña de junio.
- Transferencia a val/egl: α=0.95 y p_inter=0.6 en la clase egl (enfriamiento
  más lento y mayor peso inter-ruta); val mantiene α=0.9, p_inter=0.5.

### 4.2 Búsqueda Tabú simple (TS)

Best-improvement con lista tabú FIFO de tenure `θ`, muestreo de lote de `B`
vecinos aleatorios por iteración (evaluación vectorizada), criterio de
aspiración clásico.

**Grid search**: θ ∈ {5,10,15,20,25,30} × 23 instancias × 2 repeticiones =
276 corridas (ganador θ=25); segundo knob B ∈ {15,25,40} (ganó 40, gap medio
5.35% vs 8.84% y 13.32%). Resultado en 23 pequeñas: gap medio 1.38%, 12/23
BKS. Transferencia: θ = max(25, ⌊n/4⌉), B = max(40, n), p_inter=0.4 (val) /
0.6 (egl).

### 4.3 Búsqueda Tabú Reactiva (RTS)

Extiende TS con tenure dinámico (`θ ∈ [θmin, θmax]`, crece ×f₊ al detectar
ciclo, decrece ×f₋ tras estabilidad), memoria de soluciones visitadas
(hash canónico) y mecanismo de escape (movimientos aleatorios + limpieza de
memoria) tras repeticiones excesivas.

**Grid search**: la más grande de la campaña — f₊∈{1.05,1.1,1.2,1.3,1.4} ×
umbral_escape∈{2,3,5,8} × p_inter∈{0.4,…,0.8} × f₋∈{0.85,0.9,0.95} = 300
combinaciones × 23 × 5 reps = 34,500 corridas. Ganador: f₊=1.2, f₋=0.95
(confirmado en pase 1D: gap 5.10% vs 5.44% y 5.77%), p_inter=0.5. Resultado
en 23 pequeñas: gap medio 1.16%, 16/23 BKS. Transferencia: factores sin
cambio (autoadaptativos por diseño), solo p_inter=0.6 en egl.

### 4.4 Colonia Artificial de Abejas (ABC)

Implementación canónica de Karaboga (2005) — fases de abejas empleadas,
observadoras (ruleta proporcional al fitness) y scouts (reinicio aleatorio
tras `limite_abandono` intentos sin mejora) — con el sesgo inter/intra
compartido del proyecto como única extensión.

**Grid search**: factor_fuentes∈{1.5,2,3,4} × factor_abandono∈{0.25,0.5,
0.75,1} × p_inter∈{0.4,…,0.8} × factor_iter∈{15,20,30} = 240 × 23 × 5 =
27,600 corridas. Segundo knob: limite_abandono=60 (gap 9.53%). Ganador:
num_fuentes=30, limite_abandono=60, p_inter=0.5. Resultado en 23 pequeñas:
gap medio 1.32%, 13/23 BKS. Transferencia: limite_abandono = max(60, n),
p_inter=0.6 en egl.

### 4.5 Cuckoo Search (CS)

Adaptación discreta del algoritmo de Yang & Deb (2009): el vuelo de Lévy
continuo se traduce en una ráfaga de `L` movimientos locales encadenados,
`L = min(12, 1 + ⌊x^(1/λ)·P_base⌋)` con `x ~ |N(0,1)|`, reproduciendo el
perfil de cola pesada (mayoría de vuelos cortos, ocasionales largos). Cada
nido compite contra un nido aleatorio (reemplazo greedy); una fracción `p_a`
de los peores nidos se abandona y reconstruye por vuelo de Lévy desde el
mejor nido.

**Calibración en dos etapas**:
1. Grilla principal (junio, 23 pequeñas): num_nidos∈{10,15,25,35} ×
   factor_pasos∈{0.5,1,1.5,2} × p_inter∈{0.1,0.4,0.6,0.8,1} × p_a∈{0.1,
   0.15,0.2} = 240 combinaciones × 23 × 5 reps = 27,600 corridas. Ganador:
   p_a=0.15, λ=1.3, p_inter=0.1 (intra-heavy en instancias pequeñas).
2. Mini-grid de refinamiento (julio, 3 representativas + warm-start):
   factor_pasos∈{0.25,0.5,1} × p_inter∈{0.1,0.4,0.6} = 9 combinaciones sobre
   val1A/val5C/egl-e1-A. Ganador: factor_pasos=0.25, p_inter=0.6 (gap
   −10.82% en representativas vs +7.2% de la calibración de junio) —
   inversión completa del sesgo intra/inter al escalar.

Arranque cálido: el nido 0 se siembra con la mejor de 5 construcciones de
Path Scanning independientes; el resto son vecinos aleatorios de esa
semilla. Resultado corpus completo: gap medio 7.71%, 9/87 BKS; es la MH
ganadora en la clase egl (12.3% vs 19.5% de SA, 13/24 victorias).

### 4.6 Vibration Damping Optimization (VDO)

Implementada el 9 de julio de 2026 como sexta metaheurística. Análogo físico
del SA: la regla de Metropolis se reemplaza por la CDF de una distribución
de Rayleigh, `p_aceptar = 1 − exp(−A²/2σ²)`, constante durante todo el nivel
de amplitud (a diferencia de SA, no depende de la magnitud Δ del
empeoramiento); el enfriamiento geométrico se reemplaza por la ley de
amortiguamiento del oscilador, `A(t) = A₀·exp(−γt/2)`, que siempre parte de
A₀ (no acumula sobre la A actual). Calibración instance-aware: A₀=20·d_max/n,
A_min=20·d_max/n², σ=A₀/2, γ=0.05 (constante adimensional, sin calibrar por
grid search).

No participó en la campaña de calibración de junio; corrió directamente bajo
el protocolo de julio (435 corridas, 0 fallos) con la config de clase de SA
(p_inter 0.5 val / 0.6 egl) por no tener calibración propia. Resultado:
la MH más débil del corpus (gap medio 28.29%, 7/87 BKS). Causa diagnosticada:
con γ=0.05 fijo, el presupuesto de 10⁶ evaluaciones se agota en instancias
grandes mucho antes de que A decaiga por debajo de A_min — la corrida
termina con el mecanismo aún aceptando empeoramientos casi indiscriminadamente.
Pendiente identificado (no ejecutado): recalibrar γ por instancia para que
el amortiguamiento coincida con el presupuesto,
`γ = 2·ln(A₀/A_min)/max_niveles`.

---

## 5. La investigación de la escala de costos en `val` (11–12 de julio de 2026)

Este fue el hallazgo metodológico más significativo del proyecto y merece
registro detallado porque en un primer momento se interpretó como un posible
error en los datos.

**Síntoma**: al desplegar Cuckoo Search sobre las 34 instancias `val`, los
gaps reportados sugerían costos sistemáticamente por debajo de lo esperado,
en un rango de 10–33% de "undercounting" aparente respecto al BKS publicado.

**Hipótesis inicial descartadas, en orden de investigación**:
1. *Error en el evaluador de costos.* Se implementó un algoritmo de
   programación dinámica independiente que calcula la orientación óptima de
   cada arista requerida sobre el grafo crudo, y se comparó contra el
   evaluador del proyecto: coincidencia exacta. El evaluador es correcto en
   su escala.
2. *Error en los archivos de instancia.* Se descargaron los archivos `.dat`
   originales del repositorio de la Universidad de Valencia — comparación
   byte a byte con los archivos usados por el proyecto: idénticos.
3. *Instancia mal transcrita.* Se obtuvo `val1A` desde el repositorio
   canónico DIMACS HGS-CARP como tercera fuente independiente: mismos
   puntajes.

**Causa raíz** (confirmada leyendo el `README` oficial de CARPLIB,
Universitat de València): el formato CARPLIB distingue **dos costos** por
arista requerida — el costo de **servicio** (atravesar + realizar el
servicio, reportado en el campo de cabecera `COSTE_TOTAL_REQ`) y el costo de
**tránsito** (solo atravesar sin servir, columna `coste` de cada arista). En
las familias `gdb`, `kshs` y `egl` ambos coinciden; en la familia `val`
(Benavent et al. 1992, bccm) difieren. El evaluador del proyecto contabiliza
el servicio a costo de tránsito, por lo que su reporte difiere del BKS
canónico por una **constante por instancia**:
`δ = COSTE_TOTAL_REQ − Σ(costos de tránsito de las aristas requeridas)`.

**Verificación empírica**: `costo_evaluador + δ` reproduce el BKS exacto en
12 de las 34 instancias val, y nunca queda por debajo del BKS en ninguna de
las 34 (34/34 coherentes con la interpretación de causa raíz).

**Resolución adoptada**: en todo el material generado desde el 12 de julio,
los costos de la familia val se reportan como `costo_comparable = costo +
δ`, con δ calculado directamente desde los pickles de instancia (no de
tablas manuales) para trazabilidad completa. El evaluador del código **no se
modificó** — es correcto en su escala interna; el ajuste es puramente de
reporte para comparar contra el BKS publicado en la convención CARPLIB. Esta
lógica vive en la función `_deltas_val()` / `_deltas()`, replicada en todos
los generadores de tablas del proyecto.

---

## 6. Infraestructura y convenciones técnicas

- **Reproducibilidad**: semillas deterministas derivadas de
  `(semilla_base, instancia, MH, repetición)`; ejecución paralela con
  `ProcessPoolExecutor` y CSV parciales por worker consolidados por
  instancia, reproducible bit a bit independientemente del orden de
  ejecución.
- **CSVs de resultados**: excluidos del control de versiones
  (`.gitignore`); se versiona `config_fija.json` / `config_grid.json` por
  corrida para trazabilidad de la procedencia sin inflar el repositorio.
- **Detección de causa de parada** (`_detectar_cap_disparado`): cada corrida
  registra si terminó por tope de tiempo, tope de iteraciones, o
  convergencia natural — usado para verificar honestidad de la comparación
  entre MH bajo presupuesto uniforme.
- **Organización de la documentación** (reorganizada el 12 de julio):
  `docs/markdown/{es,en}/` y `docs/latex/{es,en}/`, con un índice
  (`docs/README.md`) que cataloga cada archivo.

---

## 7. Fuentes primarias por sección (trazabilidad)

| Sección | Script(s) / módulo(s) generador(es) |
|---|---|
| Campaña SA/TS/RTS/ABC/VDO val-egl | `scripts/run_val_egl_20260710.py` |
| Campaña VDO pequeñas | `scripts/run_vdo_small_20260714.py` |
| Campaña CS (mini-grid + despliegue) | `scripts/run_cs_val_egl_20260711.py` |
| Ajuste de escala δ (val) | `_deltas()` / `_deltas_val()` en los generadores de tabla |
| Tablas por MH (resultados + soluciones) | `scripts/_gen_tablas_mh_20260713.py` |
| Tabla de Cuckoo Search | `scripts/_gen_tabla_cuckoo_20260712.py` |
| Tabla comparativa 6 MH | `scripts/_gen_mejores_val_egl_20260712.py` |
| Tabla global de mejores soluciones | `scripts/_gen_tabla_soluciones_20260713.py` |
| Tablas SA restringidas a val/egl | `scripts/_gen_tablas_sa_egl_20260715.py` |
| Análisis de operadores de SA | `scripts/_gen_analisis_operadores_sa_20260715.py` |
| Excel consolidado | `scripts/_gen_excel_resumen_20260715.py` |
| Implementación VDO | `metacarp/vibration_damping.py` |
| Implementación SA | `metacarp/recocido_simulado.py` |
| Path Relinking (intensificación) | `metacarp/path_relinking_limpio_20260531.py` |

---

*Documento generado el 15 de julio de 2026 como parte de la documentación
final del proyecto MetaCARP.*
