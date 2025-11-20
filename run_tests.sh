#!/bin/bash

# ================================================
#  run_tests.sh — testy wydajnościowe (wersja SEQ)
# ================================================

BIN="./bin/log_seq"
DATA="./data/hadoop_all_logs.txt"
OUTDIR="./results"

mkdir -p "$OUTDIR"

CSV="$OUTDIR/seq.csv"

# Nagłówek CSV
echo "method,time_seconds" > "$CSV"

echo "========================================="
echo "  Testy wydajnościowe — wersja SEQ"
echo "========================================="
echo "Plik: $DATA"
echo

# ------------------------------------------------
# Pomiar czasu wykonania
# ------------------------------------------------
echo "→ Uruchamianie log_seq ..."

# /usr/bin/time podaje czas rzeczywisty w sekundach — idealny do speedup
TIME=$(/usr/bin/time -f "%e" $BIN "$DATA" 2>&1 > "$OUTDIR/seq_output.txt")

echo "Czas wykonania SEQ: $TIME s"

# zapis do CSV
echo "seq,$TIME" >> "$CSV"

echo
echo "Wyniki zapisano w:"
echo "  $CSV"
echo "Pełny output programu zapisano w:"
echo "  $OUTDIR/seq_output.txt"
echo
echo "Test zakończony."