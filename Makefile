# ============================
# Analiza logów systemowych
# Programowanie Równoległe i Rozproszone
# ============================

# Kompilatory
CC     = gcc
MPICC  = mpicc
NVCC   = nvcc

# Flagi
CFLAGS   = -O3 -Wall
OMPFLAGS = -fopenmp

# Foldery
SRC_DIR     = src
DATA_DIR    = data
RESULTS_DIR = results
BIN_DIR     = bin

# Pliki wynikowe
SEQ    = $(BIN_DIR)/log_seq
OPENMP = $(BIN_DIR)/log_openmp
MPI    = $(BIN_DIR)/log_mpi
CUDA   = $(BIN_DIR)/log_cuda

# ============================
# Zadania główne
# ============================

all: prepare seq openmp mpi cuda

prepare:
	mkdir -p $(BIN_DIR) $(RESULTS_DIR)

# ============================
# Wersja sekwencyjna (CPU)
# ============================

seq: prepare
	$(CC) $(CFLAGS) -o $(SEQ) $(SRC_DIR)/log_seq.c

# ============================
# Wersja OpenMP
# ============================

openmp: prepare
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $(OPENMP) $(SRC_DIR)/log_openmp.c

# ============================
# Wersja MPI
# ============================

mpi: prepare
	$(MPICC) $(CFLAGS) -o $(MPI) $(SRC_DIR)/log_mpi.c

# ============================
# Wersja CUDA
# ============================

cuda: prepare
	$(NVCC) -O3 -o $(CUDA) $(SRC_DIR)/log_cuda.cu

# ============================
# Czyszczenie
# ============================

clean:
	rm -rf $(BIN_DIR)/*
	rm -f $(RESULTS_DIR)/*.txt

.PHONY: all prepare seq openmp mpi cuda clean
