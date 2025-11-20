#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE 1024

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <log_file>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *filename = argv[1];
    FILE *f = fopen(filename, "r");
    if (!f) {
        if (rank == 0)
            perror("Cannot open file");
        MPI_Finalize();
        return 1;
    }

    // =============================
    // Liczymy rozmiar pliku i wyznaczamy zakres dla każdego procesu
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    long chunk_size = file_size / size;
    long start = rank * chunk_size;
    long end = (rank == size - 1) ? file_size : start + chunk_size;

    char *buffer = (char *)malloc(chunk_size + 1);
    if (!buffer) {
        perror("Cannot allocate memory");
        fclose(f);
        MPI_Finalize();
        return 1;
    }

    fseek(f, start, SEEK_SET);

    // Jeśli nie pierwszy proces, przesuwamy do początku następnej linii
    if (rank != 0) {
        if (fgets(buffer, MAX_LINE, f) == NULL) {
            // jeśli nie udało się przeczytać linii, kończymy w tym procesie
            fclose(f);
            free(buffer);
            MPI_Finalize();
            return 1;
        }
    }

    long bytes_read = 0;
    int local_info = 0, local_warn = 0, local_error = 0;
    char line[MAX_LINE];

    while (ftell(f) < end && fgets(line, MAX_LINE, f)) {
        bytes_read += strlen(line);

        if (strstr(line, "INFO")) local_info++;
        else if (strstr(line, "WARNING")) local_warn++;
        else if (strstr(line, "ERROR")) local_error++;
    }

    free(buffer);
    fclose(f);

    // =============================
    // Redukcja wyników do procesu 0
    int total_info = 0, total_warn = 0, total_error = 0;

    MPI_Reduce(&local_info, &total_info, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_warn, &total_warn, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_error, &total_error, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // =============================
    // Proces 0 wypisuje wyniki
    if (rank == 0) {
        printf("=== Analiza logów (MPI) ===\n");
        printf("Plik: %s\n", filename);
        printf("INFO: %d\nWARNING: %d\nERROR: %d\n", total_info, total_warn, total_error);

        // zapis wyników do pliku
        FILE *out = fopen("results/results_mpi.txt", "w");
        if (out) {
            fprintf(out, "Plik: %s\n", filename);
            fprintf(out, "INFO: %d\nWARNING: %d\nERROR: %d\n", total_info, total_warn, total_error);
            fclose(out);
        } else {
            perror("Cannot write results file");
        }
    }

    MPI_Finalize();
    return 0;
}

