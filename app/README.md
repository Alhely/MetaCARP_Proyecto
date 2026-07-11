# App interactiva — MetaCARP (Approach 1: `solo_p_inter`)

Aplicación [Streamlit](https://streamlit.io) para **ejecutar las 5
metaheurísticas** del approach de mejor desempeño (`solo_p_inter`), **ajustar
sus parámetros** y **visualizar la mejor solución**, con una **animación en
tiempo real** del recorrido de cada vehículo.

## Qué incluye

- **Barra lateral:** elección de instancia (23 del corpus), metaheurística
  (SA, TS, RTS, ABC, Cuckoo), semilla y parámetros (con los valores canónicos
  del Approach 1 por defecto).
- **Pestaña «Resumen»:** mejor costo, BKS, gap, la tabla de rutas
  (tareas, demanda/capacidad, factibilidad, secuencia), la **solución detallada**
  (servicio + *dead-heading* por vehículo, con sus nodos) y el **mapeo de
  tareas a arcos** (cada `TR#` → par de nodos y demanda).
- **Pestaña «Mejor ruta»:** mapa interactivo (zoom/hover). Línea **sólida** =
  servicio de tarea; línea **punteada** = *dead-heading* (traslado); ★ = depósito.

Las rutas se reconstruyen a nivel de nodo con la **misma orientación greedy**
del evaluador (entrar por el extremo más cercano), de modo que el dibujo es
consistente con `mejor_costo`.

## Cómo ejecutarla

Desde la **raíz del proyecto** (`MetaCARP_Proyecto/`):

```bash
# 1) (recomendado) entorno virtual
python3 -m venv .venv && source .venv/bin/activate

# 2) dependencias de la app (además del paquete metacarp del proyecto)
pip install -r app/requirements.txt

# 3) lanzar
streamlit run app/carp_app.py
```

Se abre en el navegador (por defecto `http://localhost:8501`).

## Notas

- No escribe CSV: todas las corridas ocurren en memoria.
- El layout del grafo usa *kamada-kawai* (si hay `scipy`) o `spring_layout`
  como respaldo; las instancias no traen coordenadas reales, así que la
  disposición es ilustrativa pero respeta la topología (pesos = costo).
- Si tu versión de Streamlit muestra un aviso de deprecación sobre
  `use_container_width`, es inofensivo (la app funciona igual).

## Archivos

- `carp_app.py` — interfaz Streamlit (3 pestañas).
- `carp_viz.py` — lógica: catálogo, ejecución de MH, reconstrucción de rutas y
  figuras Plotly (estática y de animación).
- `requirements.txt` — dependencias.
