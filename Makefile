CC = gcc
CFLAGS = -O3 -Wall

all: log_seq log_openmp

log_seq: src/log_seq.c
	$(CC) $(CFLAGS) -o log_seq src/log_seq.c

log_openmp: src/log_openmp.c
	$(CC) $(CFLAGS) -fopenmp -o log_openmp src/log_openmp.c

clean:
	rm -f log_seq log_openmp results/results_*.txt

