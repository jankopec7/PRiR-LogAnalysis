# ============================
# Analiza logów systemowych
# Programowanie Równoległe i Rozproszone
# ============================

# Kompilatory
CC = gcc
MPICC = mpicc
NVCC = nvcc

# Flagi
CFLAGS = -O3 -Wall
OMPFLAGS = -fopenmp

# Foldery
SRC_DIR = src
DATA_DIR = data
RESULTS_DIR = results
BIN_DIR = bin

# Pliki wynikowe
SEQ = $(BIN_DIR)/log_seq
OPENMP = $(BIN_DIR)/log_openmp
MPI = $(BIN_DIR)/log_mpi
CUDA = $(BIN_DIR)/log_cuda
GEN_LOGS = $(BIN_DIR)/gen_logs

# ============================

all: prepare seq openmp mpi cuda

prepare:
	mkdir -p $(BIN_DIR) $(RESULTS_DIR)

# ============================
# Wersja sekwencyjna
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
# Generator logów
# ============================
gen_logs: prepare
	$(CC) $(CFLAGS) -o $(GEN_LOGS) $(SRC_DIR)/gen_logs.c

generate_sample:
	$(GEN_LOGS) 1000 > $(DATA_DIR)/sample_logs.txt

# ============================
# Czyszczenie
# ============================
clean:
	rm -rf $(BIN_DIR) $(RESULTS_DIR)/*.txt
