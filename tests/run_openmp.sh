#!/bin/bash

PROJECT_DIR="$(cd "$(dirname "$0")/.."; pwd)"
BIN="$PROJECT_DIR/bin"
DATA="$PROJECT_DIR/data"
RESULTS="$PROJECT_DIR/results"

OMP="$BIN/log_openmp"
LOGFILE="$DATA/hadoop_all_logs.txt"

if [ ! -f "$OMP" ]; then
    echo "[ERROR] OpenMP binary not found. Run: make openmp"
    exit 1
fi

echo "=== Running OpenMP tests ==="
echo "Input file: $LOGFILE"

mkdir -p "$RESULTS"

for threads in 1 2 4 6 8 12; do
    echo "Running with OMP_NUM_THREADS=$threads"
    export OMP_NUM_THREADS=$threads

    OUTFILE="$RESULTS/test_openmp_${threads}_threads.txt"

    START=$(date +%s.%N)
    $OMP "$LOGFILE" > "$OUTFILE"
    END=$(date +%s.%N)

    echo "Time for $threads threads: $(echo "$END - $START" | bc) sec"
done
