#!/bin/bash
# ============================================================================
# Master script para re-correr los 5 baselines bajo evaluador greedy.
#
# Lanza 25 grids (5 variantes R x 5 MH = 25 corridas), 575 ejecuciones cada
# una, total 2875 corridas. Por defecto las 5 MH de cada variante corren EN
# PARALELO; las 5 variantes corren EN SECUENCIA (de R1 a R5).
#
# Salida:
#   experimentos/re_greedy_20260529_<variante>_<mh>/
#   logs/re_greedy_<variante>_<mh>_<timestamp>.log
#
# Uso:
#   bash scripts/run_all_re_greedy_20260529.sh
#
# Variables opcionales (override desde el shell):
#   REPETICIONES=5            # corridas por (instancia, MH, variante)
#   VARIANTES="R1 R2 R3 R4 R5"   # subset de variantes a correr
#   MHS="sa tabu_simple tabu_reactiva abc_simple cuckoo"  # subset MH
#   PYTHON_BIN="python"       # binario de Python (puede ser conda env path)
# ============================================================================
set -e

# --- Localizamos el directorio raiz del proyecto desde la ubicacion del script.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

# --- Parametros por defecto (override desde el entorno antes de invocar).
REPETICIONES="${REPETICIONES:-5}"
VARIANTES="${VARIANTES:-R1 R2 R3 R4 R5}"
MHS="${MHS:-sa tabu_simple tabu_reactiva abc_simple cuckoo}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p logs

echo "============================================================================"
echo "Master re-corrida bajo evaluador greedy (re_greedy_20260529)"
echo "============================================================================"
echo "Variantes  : $VARIANTES"
echo "MHs        : $MHS"
echo "Reps       : $REPETICIONES"
echo "Python     : $PYTHON_BIN"
echo "Workdir    : $ROOT_DIR"
echo "============================================================================"
echo

TIEMPO_INICIO=$(date +%s)

# --- Bucle principal: por variante (secuencial) -> MHs (paralelas).
for variant in $VARIANTES; do
  echo
  echo "############################################################################"
  echo "### Variante $variant — lanzando las MHs en PARALELO"
  echo "############################################################################"
  T0=$(date +%s)

  PIDS=()
  for mh in $MHS; do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG="logs/re_greedy_${variant}_${mh}_${TIMESTAMP}.log"
    echo "  -> lanzando $variant $mh (log: $LOG)"
    "$PYTHON_BIN" scripts/_re_greedy_20260529_common.py \
      --variant "$variant" --mh "$mh" --repeticiones "$REPETICIONES" \
      > "$LOG" 2>&1 &
    PIDS+=($!)
  done

  # Esperamos a TODAS las MHs de esta variante antes de pasar a la siguiente.
  echo "  ... esperando a que terminen ${#PIDS[@]} procesos paralelos"
  wait "${PIDS[@]}"
  T1=$(date +%s)
  echo "  *** Variante $variant completada en $((T1 - T0))s"
done

TIEMPO_FIN=$(date +%s)
DURACION=$((TIEMPO_FIN - TIEMPO_INICIO))
echo
echo "============================================================================"
echo "TODAS las variantes COMPLETADAS en ${DURACION}s ($((DURACION / 60))m)"
echo "============================================================================"
echo "Carpetas de resultados:"
ls -d experimentos/re_greedy_20260529_* 2>/dev/null | head -25
echo
echo "Logs: logs/re_greedy_*"
