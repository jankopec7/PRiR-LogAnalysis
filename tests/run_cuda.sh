#!/bin/bash

PROJECT_DIR="$(cd "$(dirname "$0")/.."; pwd)"
BIN="$PROJECT_DIR/bin"
DATA="$PROJECT_DIR/data"
RESULTS="$PROJECT_DIR/results"

CUDA_BIN="$BIN/log_cuda"
LOGFILE="$DATA/hadoop_all_logs.txt"

if [ ! -f "$CUDA_BIN" ]; then
    echo "[ERROR] CUDA binary not found. Run: make cuda"
    exit 1
fi

echo "=== Running CUDA tests ==="
echo "Input file: $LOGFILE"

mkdir -p "$RESULTS"

OUTFILE="$RESULTS/test_cuda.txt"

START=$(date +%s.%N)
"$CUDA_BIN" "$LOGFILE" > "$OUTFILE"
END=$(date +%s.%N)

RUNTIME=$(echo "$END - $START" | bc)

echo "Time for CUDA execution: ${RUNTIME} sec"
echo "Output saved to $OUTFILE"
