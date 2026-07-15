# Índice de documentación — MetaCARP

Estructura canónica a partir de julio 2026. Todos los documentos de tesis y referencia técnica viven bajo este árbol.

```
docs/
├── figuras/                    # Imágenes compartidas (PNG) para compilación LaTeX
├── latex/
│   ├── referencias.bib         # Bibliografía compartida (ambos idiomas)
│   ├── es/                     # Documentos LaTeX en español
│   │   ├── metaheuristicas/    # Capítulos de algoritmos (maestro + 6 MH)
│   │   └── pseudocodigos/      # Pseudocódigos (6 archivos, español)
│   └── en/                     # Documentos LaTeX en inglés
│       └── pseudocodigos/      # Pseudocódigos (4 archivos, inglés)
└── markdown/
    ├── es/                     # Documentación Markdown en español
    │   └── vecindarios_revisados/  # Operadores de vecindario (10 archivos)
    └── en/                     # (reservado para documentación en inglés)
```

---

## docs/latex/es/

| Archivo | Descripción |
|---------|-------------|
| `docs_final_costo_correcto_con_resultados.tex` | Documento hito junio 2026: campaña completa costo corregido, 23 instancias, resultados extensos (autogenerado por `scripts/_gen_docs_final_costo_correcto.py`) |
| `Capitulo_Experimentos.tex` | Capítulo de experimentación (narrativa cronológica mayo–junio 2026, incluye costo_fixed vía `\input`) |
| `experimentos_cuerpo.tex` | Cuerpo del capítulo (sección de introducción, mayo 2026) |
| `experimento_solo_p_inter_costo_fixed.tex` | Sección: experimento solo_p_inter (approach 1, junio 2026) |
| `experimento_binario_capacidad_costo_fixed.tex` | Sección: experimento binario_capacidad (approach 2, junio 2026) |
| `experimento_pr_aislado_costo_fixed.tex` | Sección: experimento pr_aislado (approach 3, junio 2026) |
| `Reporte_seleccion_operadores_y_su_efecto.tex` | Reporte monolítico: comparativa de estrategias de selección de operadores |
| `seccion_resultados_tesis.tex` | Sección de resultados (autogenerada por `scripts/_gen_seccion_resultados_tesis.py`) |
| `experimentos_full_combined_all.txt` | Documento LaTeX autocontenido combinado del capítulo de experimentación (capítulo + bibliografía embebida) |

### docs/latex/es/metaheuristicas/

| Archivo | Descripción |
|---------|-------------|
| `00_metaheuristicas_master.tex` | Documento maestro que incluye los demás |
| `recocido_simulado.tex` | Recocido Simulado (SA) |
| `busqueda_tabu_simple.tex` | Búsqueda Tabú Simple (TS) |
| `busqueda_tabu_reactiva.tex` | Búsqueda Tabú Reactiva (RTS) |
| `abc_simple.tex` | Colonia de Abejas Artificiales Simple (ABC) |
| `cuckoo_search.tex` | Cuckoo Search (CS) |
| `vibration_damping.tex` | Vibration Damping Optimization (VDO) |

### docs/latex/es/pseudocodigos/

| Archivo | Descripción |
|---------|-------------|
| `sa_simple_pseudocode.tex` | Pseudocódigo Recocido Simulado |
| `tabu_pseudocode.tex` | Pseudocódigo Búsqueda Tabú Simple y Reactiva |
| `abc_simple_pseudocode.tex` | Pseudocódigo ABC Simple |
| `cuckoo_pseudocode.tex` | Pseudocódigo Cuckoo Search (ES) |
| `neighbor_generation.tex` | Generación de vecinos con ejemplos |
| `operadores_vecindario.tex` | Los 9 operadores de vecindario (fragmento, incluir vía `\input`) |

---

## docs/latex/en/

Subsecciones de artículo (descripción + pseudocódigo `algorithm2e` +
síntesis de calibración) por metaheurística, y tablas listas para Overleaf.

| Archivo | Descripción |
|---------|-------------|
| `sa_pseudocode_section.tex` | Simulated Annealing: article subsection + pseudocode |
| `ts_pseudocode_section.tex` | Tabu Search: article subsection + pseudocode |
| `rts_pseudocode_section.tex` | Reactive Tabu Search: article subsection + pseudocode |
| `abc_pseudocode_section.tex` | Artificial Bee Colony: article subsection + pseudocode |
| `cuckoo_discretization_pseudocode.tex` | Cuckoo Search: Discretization and Pseudocode subsection |
| `neighborhood_operators_table.tex` | Table of the 9 shared neighborhood operators |

## docs/latex/en/pseudocodigos/

| Archivo | Descripción |
|---------|-------------|
| `cuckoo_pseudocode_en.tex` | Discrete Cuckoo Search pseudocode (English) |
| `metacarp_algorithms.tex` | SA, TS, ABC, Cuckoo pseudocode (English) |
| `deadhanging_nodes_figure.tex` | Figure: dead-hanging nodes structure |
| `initial_solution_structure.tex` | Figure: initial solution structure |

---

## docs/markdown/es/

| Archivo | Descripción |
|---------|-------------|
| `recocido_simulado.md` | Recocido Simulado: implementación, parámetros, calibración |
| `busqueda_tabu.md` | Búsqueda Tabú (versión extendida): pseudocódigo y ajuste |
| `busqueda_tabu_simple.md` | Búsqueda Tabú Simple (TS didáctico FIFO) |
| `busqueda_tabu_reactiva.md` | Búsqueda Tabú Reactiva (RTS): tenure dinámico y escape |
| `comparativa_tabu.md` | Comparativa TS simple vs RTS |
| `colonia_abejas.md` | Colonia de Abejas Artificiales (ABC versión extendida) |
| `abc_simple.md` | ABC Simple (Karaboga 2005 canónico) |
| `cuckoo_search.md` | Cuckoo Search: vuelos de Lévy y abandono de nidos |
| `generacion_vecinos.md` | Los 9 operadores de vecindario: descripción completa |
| `experimentacion.md` | Sistema de experimentación: `scripts/experimentos.py` y runners |

### docs/markdown/es/vecindarios_revisados/

Revisión detallada de cada operador (pseudocódigo fiel a `metacarp/vecindarios.py`):

| Archivo | Descripción |
|---------|-------------|
| `README.md` | Índice del subdirectorio |
| `00_anatomia_de_un_operador.md` | Anatomía y convenios comunes |
| `01_intra_relocate.md` | Intra-relocate |
| `02_intra_swap.md` | Intra-swap |
| `03_intra_2opt.md` | Intra-2opt |
| `04_inter_relocate.md` | Inter-relocate |
| `05_inter_swap.md` | Inter-swap |
| `06_inter_2opt_star.md` | Inter-2opt* |
| `07_inter_cross_exchange.md` | Inter-cross-exchange |
| `08_09_or_opt.md` | Or-opt-2 y Or-opt-3 |

---

## docs/figuras/

Imágenes PNG para los documentos LaTeX (8 figuras, secciones 1.3 y 1.4 del análisis de operadores).
