#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define LINE_SIZE 1024
#define MAX_LINES 1000000  // ograniczenie (możesz zwiększyć jeśli chcesz)

// Funkcja wczytuje wszystkie linie pliku do pamięci
int load_file(const char *filename, char ***lines_out) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Nie można otworzyć pliku");
        return -1;
    }

    char **lines = malloc(MAX_LINES * sizeof(char *));
    char buffer[LINE_SIZE];
    int count = 0;

    while (fgets(buffer, sizeof(buffer), file) && count < MAX_LINES) {
        lines[count] = strdup(buffer);
        count++;
    }

    fclose(file);
    *lines_out = lines;
    return count;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Użycie: %s <plik_logów>\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    char **lines;
    int total_lines = load_file(filename, &lines);
    if (total_lines <= 0) {
        fprintf(stderr, "Błąd: plik jest pusty lub nie udało się go wczytać.\n");
        return 1;
    }

    long count_info = 0, count_warn = 0, count_error = 0;

    double start_time = omp_get_wtime();

    #pragma omp parallel for reduction(+:count_info, count_warn, count_error)
    for (int i = 0; i < total_lines; i++) {
        if (strstr(lines[i], "INFO")) count_info++;
        if (strstr(lines[i], "WARNING")) count_warn++;
        if (strstr(lines[i], "ERROR")) count_error++;
    }

    double end_time = omp_get_wtime();

    printf("=== Analiza logów (OpenMP) ===\n");
    printf("Plik: %s\n", filename);
    printf("Liczba linii: %d\n", total_lines);
    printf("INFO: %ld\n", count_info);
    printf("WARNING: %ld\n", count_warn);
    printf("ERROR: %ld\n", count_error);
    printf("Czas wykonania: %.6f s\n", end_time - start_time);
    printf("Liczba wątków: %d\n", omp_get_max_threads());

    // zapis wyników do pliku
    FILE *out = fopen("results/results_openmp.txt", "w");
    if (out) {
        fprintf(out, "Plik: %s\n", filename);
        fprintf(out, "Liczba linii: %d\n", total_lines);
        fprintf(out, "INFO: %ld\n", count_info);
        fprintf(out, "WARNING: %ld\n", count_warn);
        fprintf(out, "ERROR: %ld\n", count_error);
        fprintf(out, "Czas wykonania: %.6f s\n", end_time - start_time);
        fprintf(out, "Liczba wątków: %d\n", omp_get_max_threads());
        fclose(out);
    } else {
        perror("Nie można zapisać wyników");
    }

    // zwalnianie pamięci
    for (int i = 0; i < total_lines; i++)
        free(lines[i]);
    free(lines);

    return 0;
}
