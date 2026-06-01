#!/bin/bash
# ============================================================================
# Master script del approach binario_capacidad_20260531 (segundo approach del
# programa con costo corregido): selector BINARIO determinista basado en
# capacidad (viola -> INTER, factible -> INTRA) + seleccion uniforme de
# operador, sin PR/kick/AOS/budget. Config canonica fija por MH (sin grid).
# A diferencia del approach 1, NO hay p_inter (el selector es determinista).
#
# Corre las 5 MH EN SECUENCIA (cada una satura la CPU internamente con su
# propio ProcessPoolExecutor). Cada MH crea su carpeta con timestamp:
#   experimentos_costo_fixed/<mh>_binario_capacidad_<YYYYMMDD-HHMM>/
#
# Uso:
#   bash scripts/run_all_binario_capacidad_20260531.sh
#
# Variables opcionales (override desde el shell):
#   MHS="sa tabu_simple tabu_reactiva abc_simple cuckoo"  # subset de MH
#   REPS=5                    # repeticiones por instancia
#   WORKERS=<n>               # procesos paralelos por MH (default: nproc)
#   SALIDA_BASE=experimentos_costo_fixed
#   PYTHON_BIN="python"       # binario de Python (puede ser path de conda env)
# ============================================================================
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

MHS="${MHS:-sa tabu_simple tabu_reactiva abc_simple cuckoo}"
REPS="${REPS:-5}"
WORKERS="${WORKERS:-$(nproc 2>/dev/null || echo 1)}"
SALIDA_BASE="${SALIDA_BASE:-experimentos_costo_fixed}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p logs

echo "============================================================================"
echo "Approach binario_capacidad_20260531 — config canonica fija (sin grid)"
echo "============================================================================"
echo "MHs          : $MHS"
echo "Reps         : $REPS"
echo "Workers      : $WORKERS"
echo "Salida base  : $SALIDA_BASE"
echo "Python       : $PYTHON_BIN"
echo "============================================================================"
echo

TIEMPO_INICIO=$(date +%s)

for mh in $MHS; do
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  LOG="logs/binario_capacidad_${mh}_${TIMESTAMP}.log"
  echo "### $mh — lanzando (log: $LOG)"
  T0=$(date +%s)
  "$PYTHON_BIN" scripts/_binario_capacidad_20260531_common.py \
    --mh "$mh" \
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
echo "Carpetas de resultados:"
ls -dt "$SALIDA_BASE"/*_binario_capacidad_* 2>/dev/null | head -25
echo
echo "Logs: logs/binario_capacidad_*"
