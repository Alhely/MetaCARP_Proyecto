#!/bin/bash
# ============================================================================
# Master script del approach solo_p_inter_20260531 (el mas simple bajo el
# costo corregido): selector p_inter + seleccion uniforme de operador, sin
# PR/kick/AOS/budget. Grid search en dos fases (calibracion + final) por MH.
#
# Corre las 5 MH EN SECUENCIA (cada una satura la CPU internamente con su
# propio ProcessPoolExecutor, por eso NO se lanzan en paralelo: evitaria
# sobre-suscribir los nucleos). Cada MH crea su carpeta con timestamp:
#   experimentos_costo_fixed/<mh>_solo_p_inter_<YYYYMMDD-HHMM>/
#
# Uso:
#   bash scripts/run_all_solo_p_inter_20260531.sh
#
# Variables opcionales (override desde el shell):
#   MHS="sa tabu_simple tabu_reactiva abc_simple cuckoo"  # subset de MH
#   REPS_CAL=3                # repeticiones de la fase de calibracion
#   REPS_FIN=5                # repeticiones de la fase final
#   WORKERS=<n>               # procesos paralelos por MH (default: nproc)
#   SALIDA_BASE=experimentos_costo_fixed
#   PYTHON_BIN="python"       # binario de Python (puede ser path de conda env)
# ============================================================================
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

MHS="${MHS:-sa tabu_simple tabu_reactiva abc_simple cuckoo}"
REPS_CAL="${REPS_CAL:-3}"
REPS_FIN="${REPS_FIN:-5}"
WORKERS="${WORKERS:-$(nproc 2>/dev/null || echo 1)}"
SALIDA_BASE="${SALIDA_BASE:-experimentos_costo_fixed}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p logs

echo "============================================================================"
echo "Approach solo_p_inter_20260531 — grid search en dos fases"
echo "============================================================================"
echo "MHs          : $MHS"
echo "Reps cal/fin : $REPS_CAL / $REPS_FIN"
echo "Workers      : $WORKERS"
echo "Salida base  : $SALIDA_BASE"
echo "Python       : $PYTHON_BIN"
echo "============================================================================"
echo

TIEMPO_INICIO=$(date +%s)

for mh in $MHS; do
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  LOG="logs/solo_p_inter_${mh}_${TIMESTAMP}.log"
  echo "### $mh — lanzando (log: $LOG)"
  T0=$(date +%s)
  "$PYTHON_BIN" scripts/_solo_p_inter_20260531_common.py \
    --mh "$mh" \
    --reps-calibracion "$REPS_CAL" \
    --reps-final "$REPS_FIN" \
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
echo "Carpetas de resultados:"
ls -dt "$SALIDA_BASE"/*_solo_p_inter_* 2>/dev/null | head -25
echo
echo "Logs: logs/solo_p_inter_*"
