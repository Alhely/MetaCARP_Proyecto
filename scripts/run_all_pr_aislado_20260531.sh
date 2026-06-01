#!/bin/bash
# ============================================================================
# Master script del approach pr_aislado_20260531 (tercer approach del programa
# con costo corregido): aísla el efecto de Path Relinking LIMPIO (módulo
# path_relinking_limpio_20260531, sin frame-hacks ni kick aleatorio) sobre un
# selector elegible (p_inter o binario, ambos por defecto). PR se dispara en el
# estancamiento HACIA la mejor solución global, en lugar del kick aleatorio.
# Config canonica fija por MH (sin grid); el selector sigue siendo parametrico
# (dimensión del experimento: p_inter vs binario).
#
# Corre las 5 MH EN SECUENCIA (cada una satura la CPU con su ProcessPoolExecutor).
# Cada MH crea su carpeta con timestamp:
#   experimentos_costo_fixed/<mh>_pr_aislado_<YYYYMMDD-HHMM>/
#
# Uso:
#   bash scripts/run_all_pr_aislado_20260531.sh
#
# Variables opcionales (override desde el shell):
#   MHS="sa tabu_simple tabu_reactiva abc_simple cuckoo"
#   SELECTOR="ambos"          # p_inter | binario | ambos
#   REPS=5                    # repeticiones por instancia
#   WORKERS=<n>               # default: nproc
#   SALIDA_BASE=experimentos_costo_fixed
#   PYTHON_BIN="python"
# ============================================================================
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

MHS="${MHS:-sa tabu_simple tabu_reactiva abc_simple cuckoo}"
SELECTOR="${SELECTOR:-ambos}"
REPS="${REPS:-5}"
WORKERS="${WORKERS:-$(nproc 2>/dev/null || echo 1)}"
SALIDA_BASE="${SALIDA_BASE:-experimentos_costo_fixed}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p logs

echo "============================================================================"
echo "Approach pr_aislado_20260531 — Path Relinking limpio, config canonica fija"
echo "============================================================================"
echo "MHs          : $MHS"
echo "Selector     : $SELECTOR"
echo "Reps         : $REPS"
echo "Workers      : $WORKERS"
echo "Salida base  : $SALIDA_BASE"
echo "============================================================================"
echo

TIEMPO_INICIO=$(date +%s)

for mh in $MHS; do
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  LOG="logs/pr_aislado_${mh}_${TIMESTAMP}.log"
  echo "### $mh — lanzando (log: $LOG)"
  T0=$(date +%s)
  "$PYTHON_BIN" scripts/_pr_aislado_20260531_common.py \
    --mh "$mh" \
    --selector "$SELECTOR" \
    --reps "$REPS" \
    --workers "$WORKERS" \
    --salida-base "$SALIDA_BASE" \
    > "$LOG" 2>&1
  T1=$(date +%s)
  echo "    *** $mh completada en $((T1 - T0))s"
done

TIEMPO_FIN=$(date +%s)
DURACION=$((TIEMPO_FIN - TIEMPO_INICIO))
echo
echo "============================================================================"
echo "TODAS las MH COMPLETADAS en ${DURACION}s ($((DURACION / 60))m)"
echo "============================================================================"
ls -dt "$SALIDA_BASE"/*_pr_aislado_* 2>/dev/null | head -25
echo "Logs: logs/pr_aislado_*"
