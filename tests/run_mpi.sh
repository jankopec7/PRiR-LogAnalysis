#!/bin/bash

PROJECT_DIR="$(cd "$(dirname "$0")/.."; pwd)"
BIN="$PROJECT_DIR/bin"
DATA="$PROJECT_DIR/data"
RESULTS="$PROJECT_DIR/results"

MPI="$BIN/log_mpi"
LOGFILE="$DATA/hadoop_all_logs.txt"

if [ ! -f "$MPI" ]; then
    echo "[ERROR] MPI binary not found. Run: make mpi"
    exit 1
fi

echo "=== Running MPI tests ==="
echo "Input file: $LOGFILE"

mkdir -p "$RESULTS"

for np in 1 2 4 6; do
    echo "Running with $np processes"
    OUTFILE="$RESULTS/test_mpi_${np}_proc.txt"

    START=$(date +%s.%N)
    mpirun -np $np "$MPI" "$LOGFILE" > "$OUTFILE"
    END=$(date +%s.%N)

    echo "Time for $np processes: $(echo "$END - $START" | bc) sec"
done
