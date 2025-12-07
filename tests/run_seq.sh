#!/bin/bash

# Always run from project root
PROJECT_DIR="$(cd "$(dirname "$0")/.."; pwd)"
BIN="$PROJECT_DIR/bin"
DATA="$PROJECT_DIR/data"
RESULTS="$PROJECT_DIR/results"

SEQ="$BIN/log_seq"
LOGFILE="$DATA/hadoop_all_logs.txt"

if [ ! -f "$SEQ" ]; then
    echo "[ERROR] Sequential binary not found. Run: make seq"
    exit 1
fi

echo "=== Running SEQ tests ==="
echo "Input file: $LOGFILE"

mkdir -p "$RESULTS"

START=$(date +%s.%N)
$SEQ "$LOGFILE" > "$RESULTS/test_seq_output.txt"
END=$(date +%s.%N)

echo "Execution time: $(echo "$END - $START" | bc) sec"
echo "Output saved to results/test_seq_output.txt"
