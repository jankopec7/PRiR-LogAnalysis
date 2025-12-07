#!/bin/bash
#
# Kompleksowy benchmark: SEQ + OpenMP + MPI + CUDA
# PRiR – Analiza logów systemowych
#

PROJECT_DIR="$(cd "$(dirname "$0")/.."; pwd)"
BIN="$PROJECT_DIR/bin"
DATA="$PROJECT_DIR/data"
RESULTS="$PROJECT_DIR/results"

SEQ="$BIN/log_seq"
OMP="$BIN/log_openmp"
MPI="$BIN/log_mpi"
CUDA="$BIN/log_cuda"

LOGFILE="$DATA/hadoop_all_logs.txt"

# ===============================
# Sprawdzenie plików wykonywalnych
# ===============================

echo "=== Benchmark – SEQ / OpenMP / MPI / CUDA ==="
echo "Plik wejściowy: $LOGFILE"
echo

if [ ! -f "$SEQ" ] || [ ! -f "$OMP" ] || [ ! -f "$MPI" ] || [ ! -f "$CUDA" ]; then
    echo "[ERROR] Brakuje binariów. Uruchom:"
    echo "  make all"
    exit 1
fi

mkdir -p "$RESULTS"
mkdir -p "$RESULTS/outputs"

CSV="$RESULTS/benchmarks.csv"

# Nagłówek CSV
echo "method,workers,time_seconds" > "$CSV"

# Pomocnicza funkcja do pomiaru czasu
measure() {
    START=$(date +%s.%N)
    $1 > "$2"
    END=$(date +%s.%N)
    echo "$(echo "$END - $START" | bc)"
}

# ===============================
# SEQ
# ===============================
echo ">>> SEQ"
OUT="$RESULTS/outputs/seq.txt"
TIME=$(measure "$SEQ $LOGFILE" "$OUT")
echo "SEQ,1,$TIME" >> "$CSV"
echo "   SEQ time = $TIME s"
echo

# ===============================
# OpenMP
# ===============================
echo ">>> OpenMP tests"
for threads in 1 2 4 6 8 12; do
    echo "   OMP_NUM_THREADS=$threads"
    OUT="$RESULTS/outputs/openmp_${threads}.txt"
    TIME=$(START=$(date +%s.%N); OMP_NUM_THREADS=$threads $OMP $LOGFILE > "$OUT"; END=$(date +%s.%N); echo "$END - $START" | bc)
    echo "OpenMP,$threads,$TIME" >> "$CSV"
    echo "      time = $TIME s"
done
echo

# ===============================
# MPI – standard tests
# ===============================
echo ">>> MPI tests"
for np in 1 2 4 6; do
    echo "   mpirun -np $np"
    OUT="$RESULTS/outputs/mpi_${np}.txt"
    TIME=$(START=$(date +%s.%N); mpirun -np $np $MPI $LOGFILE > "$OUT"; END=$(date +%s.%N); echo "$END - $START" | bc)
    echo "MPI,$np,$TIME" >> "$CSV"
    echo "      time = $TIME s"
done
echo


# ===============================
# MPI – oversubscribe tests
# ===============================
echo ">>> MPI oversubscribe tests"
for np in 8 12; do
    echo "   mpirun --oversubscribe -np $np"
    OUT="$RESULTS/outputs/mpi_${np}_oversub.txt"
    TIME=$(START=$(date +%s.%N); mpirun --oversubscribe -np $np $MPI $LOGFILE > "$OUT"; END=$(date +%s.%N); echo "$END - $START" | bc)

    echo "MPI-oversub,$np,$TIME" >> "$CSV"

    echo "      time = $TIME s"
done
echo


# ===============================
# CUDA
# ===============================
echo ">>> CUDA"
OUT="$RESULTS/outputs/cuda.txt"
TIME=$(measure "$CUDA $LOGFILE" "$OUT")
echo "CUDA,1,$TIME" >> "$CSV"
echo "   CUDA time = $TIME s"
echo

# ===============================
# Podsumowanie
# ===============================
echo "=== Benchmark zakończony ==="
echo "Wyniki zapisano do:"
echo "  $CSV"
echo "Szczegółowe wyjścia:"
echo "  $RESULTS/outputs/"
