# ============================
# PRiR – Analiza logów systemowych
# Makefile – SEQ / OpenMP / MPI / CUDA
# ============================

# Kompilatory
CC     = gcc
MPICC  = mpicc
NVCC   = nvcc

# Flagi
CFLAGS   = -O3 -Wall
OMPFLAGS = -fopenmp
MPI_WARN   = -Wno-unused-result

# Foldery
SRC_DIR     = src
DATA_DIR    = data
RESULTS_DIR = results
BIN_DIR     = bin

# Wynikowe pliki wykonywalne
SEQ_BIN    = $(BIN_DIR)/log_seq
OPENMP_BIN = $(BIN_DIR)/log_openmp
MPI_BIN    = $(BIN_DIR)/log_mpi
CUDA_BIN   = $(BIN_DIR)/log_cuda

# ============================
# Zadania główne
# ============================

all: prepare $(SEQ_BIN) $(OPENMP_BIN) $(MPI_BIN) $(CUDA_BIN)
	@echo "Kompilacja wszystkich modułów zakończona."

prepare:
	mkdir -p $(BIN_DIR) $(RESULTS_DIR)

# ============================
# Wersja sekwencyjna (CPU)
# ============================

seq: $(SEQ_BIN)

$(SEQ_BIN): $(SRC_DIR)/log_seq.c
	@echo "[SEQ]    Kompiluję wersję sekwencyjną..."
	$(CC) $(CFLAGS) -o $@ $<

# ============================
# Wersja OpenMP
# ============================

openmp: $(OPENMP_BIN)

$(OPENMP_BIN): $(SRC_DIR)/log_openmp.c
	@echo "[OpenMP] Kompiluję wersję OpenMP..."
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $<


# ============================
# Wersja MPI
# ============================

mpi: $(MPI_BIN)

$(MPI_BIN): $(SRC_DIR)/log_mpi.c
	@echo "[MPI]    Kompiluję wersję MPI..."
	$(MPICC) $(CFLAGS) $(MPI_WARN) -o $@ $<

# ============================
# Wersja CUDA
# ============================

cuda: $(CUDA_BIN)

$(CUDA_BIN): $(SRC_DIR)/log_cuda.cu
	@echo "[CUDA]   Kompiluję wersję CUDA..."
	$(NVCC) -O3 -o $@ $<

# ============================
# Czyszczenie
# ============================

clean:
	rm -rf $(BIN_DIR)/*
	rm -f $(RESULTS_DIR)/*.txt
	@echo "Wyczyszczono pliki binarne i wyniki."

.PHONY: all prepare clean seq openmp mpi cuda
