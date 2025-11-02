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
            fprintf(stderr, "Użycie: %s <plik_logów>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    char *filename = argv[1];
    FILE *f = fopen(filename, "r");
    if (!f) {
        if (rank == 0)
            perror("Błąd otwarcia pliku");
        MPI_Finalize();
        return 1;
    }

    // TODO: podział pliku między procesy
    // Każdy proces czyta część danych i zlicza INFO/WARNING/ERROR

    int local_info = 0, local_warn = 0, local_error = 0;
    char line[MAX_LINE];

    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "INFO")) local_info++;
        else if (strstr(line, "WARNING")) local_warn++;
        else if (strstr(line, "ERROR")) local_error++;
    }
    fclose(f);

    int total_info = 0, total_warn = 0, total_error = 0;
    MPI_Reduce(&local_info, &total_info, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_warn, &total_warn, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_error, &total_error, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("=== Analiza logów (MPI) ===\n");
        printf("Plik: %s\n", filename);
        printf("INFO: %d\nWARNING: %d\nERROR: %d\n", total_info, total_warn, total_error);
    }

    MPI_Finalize();
    return 0;
}
