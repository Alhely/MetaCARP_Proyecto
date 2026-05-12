# MetaCARP

Utilidades de tesis para resolver el **Problema de Enrutamiento de Arcos con Capacidad** (CARP: *Capacitated Arc Routing Problem*) usando metaheurísticas avanzadas con evaluación de costo vectorizada y aceleración opcional en GPU.

## ¿Qué es CARP?

El Problema de Enrutamiento de Arcos con Capacidad es un desafío clásico de optimización combinatoria: dado un grafo con nodos conectados por aristas etiquetadas (cada una con una demanda específica), diseñar rutas para vehículos de capacidad limitada que:

- Recorran todas las aristas requeridas (demanda > 0)
- Respeten la capacidad máxima del vehículo
- Minimicen la distancia total recorrida (incluyendo aristas de repositorio)

Es fundamental en logística, recogida de residuos, mantenimiento de carreteras y otras aplicaciones del mundo real donde los vehículos deben servir segmentos de red (aristas), no solo nodos aislados.

---

## Características principales

- **Cuatro metaheurísticas implementadas:**
  - Búsqueda Tabú (Tabu Search)
  - Colonia de Abejas Artificiales (Artificial Bee Colony)
  - Búsqueda del Cucú (Cuckoo Search)
  - Recocido Simulado (Simulated Annealing)

- **Evaluación de costo optimizada:**
  - Matriz de caminos mínimos precalculada (Dijkstra)
  - Vectorización rápida con NumPy (10×–50× más rápida que métodos ingenuos)
  - Soporte opcional de GPU con CuPy para evaluación en lote

- **Vecindarios eficientes:** 7 operadores de movimiento implementados para generar soluciones vecinas

- **Validación de factibilidad:** verificación de capacidades y cobertura de aristas

- **Gestión de instancias:** carga perezosa (lazy loading) desde archivos pickle para minimizar uso de memoria

- **Reportes detallados:** generación de reportes legibles por ruta, demanda y costo

---

## Estructura del proyecto

```
MetaCARP_Proyecto/
├── README.md                              # Este archivo
├── pyproject.toml                         # Configuración del paquete
│
├── docs/                                  # Documentación técnica de las metaheurísticas
│   ├── busqueda_tabu.md
│   ├── colonia_abejas.md
│   ├── cuckoo_search.md
│   ├── recocido_simulado.md
│   └── generacion_vecinos.md
│
├── scripts/                               # Scripts ejecutables
│   ├── testing.py                         # Demostración interactiva de la API
│   ├── experimentos.py                    # Campaña de experimentación (grid search)
│   └── __init__.py
│
├── Grafos/                                # Archivos de instancias (grafos GEXF e imágenes)
├── PickleInstances/                       # Instancias CARP serializadas
├── Matrices/                              # Matrices de caminos mínimos precalculadas
├── InitialSolution/                       # Soluciones iniciales precalculadas
│
├── Módulos principales (raíz del paquete):
│
│   Instancias y carga de datos:
│   ├── instances.py                       # Gestión de instancias CARP (lazy loading)
│   ├── cargar_grafos.py                   # Carga de grafos NetworkX y imágenes PNG
│   ├── cargar_matrices.py                 # Carga de matrices de distancias (Dijkstra)
│   ├── cargar_soluciones_iniciales.py     # Carga de soluciones de partida
│
│   Evaluación y factibilidad:
│   ├── costo_solucion.py                  # Cálculo de costo de una solución
│   ├── evaluador_costo.py                 # Evaluación rápida con GPU opcional
│   ├── factibilidad.py                    # Verificación de restricciones
│   ├── reporte_solucion.py                # Generación de reportes legibles
│
│   Búsqueda y encoding:
│   ├── busqueda_indices.py                # Codificación índice <-> etiqueta
│   ├── vecindarios.py                     # Operadores de vecindario (7 tipos)
│
│   Metaheurísticas:
│   ├── busqueda_tabu.py                   # Búsqueda Tabú
│   ├── abejas.py                          # Colonia de Abejas
│   ├── cuckoo_search.py                   # Búsqueda del Cucú
│   ├── recocido_simulado.py               # Recocido Simulado
│   ├── metaheuristicas_utils.py           # Utilidades compartidas
│
│   Utilerías:
│   ├── solucion_formato.py                # Validación y normalización de soluciones
│   ├── grafo_ruta.py                      # Operaciones de bajo nivel sobre grafos
│   └── __init__.py                        # Punto de entrada del paquete
│
├── resultados/                            # Resultados de experimentos previos (CSV)
└── experimentos*/                         # Directorios de experimentos particulares
```

---

## Instalación

### Requisitos previos

- **Python 3.10+** (recomendado 3.11 o 3.12)
- **pip** (gestor de paquetes Python)

### Instalación básica (CPU)

Clona el repositorio y asegúrate de estar en el directorio raíz:

```bash
git clone <https://url-del-repo>
cd MetaCARP_Proyecto
pip install -e .
```

La bandera `-e` instala el paquete en modo editable (cambios en el código se reflejan inmediatamente).

**Dependencias instaladas automáticamente:**
- `numpy >= 1.23` — computación numérica vectorizada
- `networkx >= 3.0` — manipulación de grafos
- `pillow >= 10.0` — manejo de imágenes (conversión PNG/JPEG)

### Instalación con soporte GPU (CUDA)

Si tienes una GPU NVIDIA con CUDA instalado, puedes acelerar la evaluación de soluciones:

**CUDA 12.x:**
```bash
pip install -e ".[gpu-cuda12]"
```

**CUDA 11.x:**
```bash
pip install -e ".[gpu-cuda11]"
```

> **Nota:** Se fija el rango de CuPy < 14 para evitar incompatibilidades con librerías CUDA 12. Consulta la documentación de CuPy si tienes problemas de importación.

Si CuPy no está disponible o falla la importación, el código hace fallback automático a CPU.

---

## Uso rápido

### 1. Demostración interactiva de la API

El script [scripts/testing.py](scripts/testing.py) muestra todas las funcionalidades del paquete:

```bash
python -m metacarp.scripts.testing
```

Este script demuestra:
- Carga de instancias y catálogos disponibles
- Formato y normalización de soluciones
- Cálculo de costo y verificación de factibilidad
- Encoding de búsqueda (índices numéricos)
- Generación de vecinos
- Ejecución de las 4 metaheurísticas

### 2. Ejecución de una metaheurística

```python
from metacarp import busqueda_tabu_desde_instancia

resultado = busqueda_tabu_desde_instancia(
    nombre_instancia="gdb19",
    num_iteraciones=500,
    usar_gpu=False,
    semilla=42
)

print(f"Mejor costo encontrado: {resultado.mejor_costo}")
print(f"Tiempo de ejecución: {resultado.tiempo_segundos:.2f}s")
print(f"Iteraciones ejecutadas: {resultado.iteraciones}")
```

### 3. Campaña de experimentación

El script [scripts/experimentos.py](scripts/experimentos.py) ejecuta grid search sistemático:

```bash
# Ejecución por defecto (todas las instancias, todas las metaheurísticas)
python -m metacarp.scripts.experimentos

# Instancias y metaheurísticas específicas
python -m metacarp.scripts.experimentos \
    --instancias gdb19 gdb20 \
    --metaheuristicas sa tabu \
    --iteraciones 300 200 \
    --repeticiones 5 \
    --seed 7

# Con penalización de capacidad personalizada
python -m metacarp.scripts.experimentos \
    --lambda-capacidad 500 \
    --salida-dir mis_resultados
```

Los resultados se guardan en arquivos CSV por metaheurística e instancia, con columnas de tiempo, costo, iteraciones y contador de operadores.

---

## Referencia de metaheurísticas

Cada metaheurística implementa la misma interfaz estándar:

### Búsqueda Tabú

Explora el espacio de soluciones moviéndose al mejor vecino disponible, incluso si es peor que la solución actual. Usa una lista tabú para evitar ciclos.

```python
from metacarp import busqueda_tabu_desde_instancia

resultado = busqueda_tabu_desde_instancia(
    nombre_instancia="gdb19",
    num_iteraciones=500,
    tamaño_lista_tabu=50,
    tamaño_vecindario=30,
    semilla=42,
    usar_gpu=False
)
```

**Parámetros recomendados:** ver [docs/busqueda_tabu.md](docs/busqueda_tabu.md)

### Colonia de Abejas Artificiales

Inspirada en el comportamiento de búsqueda de alimento en colmenas: abejas exploradoras descubren fuentes de alimento, abejas danzarinas reclutan otras al mejores soluciones.

```python
from metacarp import busqueda_abejas_desde_instancia

resultado = busqueda_abejas_desde_instancia(
    nombre_instancia="gdb19",
    num_iteraciones=500,
    tamaño_colonia=30,
    tamaño_vecindario=20,
    semilla=42,
    usar_gpu=False
)
```

**Parámetros recomendados:** ver [docs/colonia_abejas.md](docs/colonia_abejas.md)

### Búsqueda del Cucú

Inspirada en el parasitismo de anidación del cucú: explora el espacio usando vuelos de Lévy para saltos largos, con abandono probabilístico de soluciones pobres.

```python
from metacarp import cuckoo_search_desde_instancia

resultado = cuckoo_search_desde_instancia(
    nombre_instancia="gdb19",
    num_iteraciones=500,
    tamaño_nido=25,
    tamaño_vecindario=20,
    semilla=42,
    usar_gpu=False
)
```

**Parámetros recomendados:** ver [docs/cuckoo_search.md](docs/cuckoo_search.md)

### Recocido Simulado

Inspirada en el enfriamiento de metales: acepta soluciones peores con cierta probabilidad (dependiente de la temperatura), reduciendo la probabilidad con el tiempo.

```python
from metacarp import recocido_simulado_desde_instancia

resultado = recocido_simulado_desde_instancia(
    nombre_instancia="gdb19",
    num_iteraciones=500,
    temperatura_inicial=100.0,
    factor_enfriamiento=0.995,
    tamaño_vecindario=30,
    semilla=42,
    usar_gpu=False
)
```

**Parámetros recomendados:** ver [docs/recocido_simulado.md](docs/recocido_simulado.md)

---

## API principal

### Gestión de instancias

```python
from metacarp import load_instances, load_instance

# Listar todas las instancias disponibles
instancias = load_instances()
print(instancias.keys())

# Cargar una instancia específica
instancia = load_instance("gdb19")
print(instancia)
```

### Evaluación de costo

```python
from metacarp import (
    construir_contexto_desde_instancia,
    costo_rapido,
    costo_rapido_ids
)

# Construcción del contexto (una sola vez por ejecución)
context = construir_contexto_desde_instancia("gdb19", usar_gpu=False)

# Bajo en costo: evalúa una solución rápidamente
costo = costo_rapido(context, solucion)

# Variante con IDs numéricos
costo = costo_rapido_ids(context, indices_solucion)
```

### Verificación de factibilidad

```python
from metacarp import verificar_factibilidad_desde_instancia

resultado = verificar_factibilidad_desde_instancia(
    nombre_instancia="gdb19",
    solucion=solucion_etiquetas
)

if resultado.es_factible:
    print("✓ Solución válida")
else:
    print("✗ Solución inválida:")
    for detalle in resultado.detalles:
        print(f"  - Ruta {detalle.ruta_id}: {detalle.descripcion}")
```

### Reporte de solución

```python
from metacarp import reporte_solucion_desde_instancia

reporte = reporte_solucion_desde_instancia(
    nombre_instancia="gdb19",
    solucion=solucion_etiquetas
)

print(reporte)
```

Genera un reporte legible con:
- Rutas por vehículo
- Demanda atendida por ruta
- Costo por ruta
- Uso de capacidad

---

## Documentación detallada

Consulta los archivos en [docs/](docs/) para detalles técnicos profundos:

- [busqueda_tabu.md](docs/busqueda_tabu.md) — Algoritmo, pseudocódigo y ajuste de parámetros
- [colonia_abejas.md](docs/colonia_abejas.md) — Mecánica de la colonia y operadores
- [cuckoo_search.md](docs/cuckoo_search.md) — Vuelos de Lévy y abandono de nidos
- [recocido_simulado.md](docs/recocido_simulado.md) — Escalas de temperatura y enfriamiento
- [generacion_vecinos.md](docs/generacion_vecinos.md) — Los 7 operadores de vecindario

---

## Dependencias

| Paquete | Versión | Propósito |
|---------|---------|----------|
| `numpy` | ≥ 1.23 | Computación numérica vectorizada |
| `networkx` | ≥ 3.0 | Manejo de grafos y operaciones de rutas |
| `pillow` | ≥ 10.0 | Carga y manipulación de imágenes PNG/JPG |
| `cupy-cuda12x` | ≥ 13.0, < 14 | (Opcional) Aceleración GPU para CUDA 12.x |
| `cupy-cuda11x` | ≥ 12.0, < 14 | (Opcional) Aceleración GPU para CUDA 11.x |
| `pytest` | ≥ 7.0 | (Desarrollo) Testing |

---

## Estructura de resultados

Los experimentos generan tablas CSV con columnas:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `instancia` | str | Nombre de la instancia (ej. `gdb19`) |
| `metaheuristica` | str | Nombre del algoritmo (`sa`, `tabu`, `abejas`, `cuckoo`) |
| `semilla` | int | Semilla de reproducibilidad |
| `mejor_costo` | float | Mejor solución encontrada |
| `tiempo_segundos` | float | Tiempo total de ejecución |
| `iteraciones` | int | Número de iteraciones realizadas |
| `contador_operadores` | dict | Frecuencia de cada operador de vecindario |

---

## Casos de uso típicos

### Comparación de metaheurísticas

```bash
python -m metacarp.scripts.experimentos \
    --instancias gdb19 gdb21 \
    --metaheuristicas sa tabu abejas cuckoo \
    --repeticiones 10 \
    --salida-dir comparativa_2025
```

Luego analiza los CSV con pandas:

```python
import pandas as pd

df = pd.read_csv("comparativa_2025/resultados_sa_gdb19.csv")
print(df.describe())
```

### Ajuste de parámetros

```bash
python -m metacarp.scripts.experimentos \
    --instancias gdb19 \
    --metaheuristicas tabu \
    --tamaño-lista-tabu 30 50 100 \
    --tamaño-vecindario 20 30 50 \
    --iteraciones 500 \
    --repeticiones 3
```

### Reproducibilidad garantizada

Especifica `--seed` para obtener exactamente los mismos resultados:

```bash
python -m metacarp.scripts.experimentos \
    --seed 12345 \
    --instancias gdb19 \
    --metaheuristicas sa \
    --repeticiones 1
```

---

## Contribución

Si deseas contribuir mejoras, nuevas metaheurísticas o casos de prueba:

1. **Fork** el repositorio
2. Crea una rama temática: `git checkout -b feature/nueva-metaheuristica`
3. Documenta en docstrings estilo NumPy (ej. en [docs/](docs/))
4. Asegúrate de que el código siga la estructura existente:
   - Función principal: `<nombre>(...) -> <NombreResult>`
   - Wrapper: `<nombre>_desde_instancia(...) -> <NombreResult>`
   - Resultado: dataclass frozen con campos `mejor_solucion`, `mejor_costo`, `tiempo_segundos`, `iteraciones`
5. Commit y push: `git push origin feature/...`
6. Abre un Pull Request describiendo los cambios

---

## Autor y créditos

**Proyecto:** Alhely González Luna  
**Email:** alelygl@gmail.com  
**Licencia:** (Especificar según corresponda, ej. MIT, GPL-3.0, etc.)

Este proyecto es producto del trabajo de tesis sobre resolución del Problema de Enrutamiento de Arcos con Capacidad usando metaheurísticas avanzadas.

---

## Preguntas frecuentes

**P: ¿Qué instancias CARP incluye el proyecto?**
> Consulta `load_instances()` en Python o el directorio `PickleInstances/` para ver qué instancias están disponibles.

**P: ¿Cómo sé si mi GPU se está usando?**
> Ejecuta `from metacarp import gpu_disponible; print(gpu_disponible())`. Si devuelve `True`, CuPy está disponible. En scripts, usa `--verbose` (si está soportado) o revisa logs de CUDA.

**P: ¿Qué pasa si una solución no es factible?**
> La función `verificar_factibilidad_desde_instancia()` devuelve un objeto con detalles específicos del problema. Los metaheurísticos usan penalización de capacidad para guiar la búsqueda hacia regiones factibles.

**P: ¿Puedo usar mis propias instancias CARP?**
> Actualmente, las instancias se cargan desde archivos pickle en `PickleInstances/`. Para agregar instancias personalizadas, necesitarías extender `instances.py` y crear archivos pickle con la estructura esperada.

**P: ¿Por qué los tiempos varían entre ejecuciones incluso con la misma semilla?**
> En CPU puro con la misma semilla son reproducibles. En GPU, las operaciones CUDA tienen no-determinismo intrínseco que puede variar ligeramente.

---

**Para más información, consulta los scripts de ejemplo, la documentación en [docs/](docs/) y los comentarios en el código fuente.**