#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LINE_SIZE 1024

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Użycie: %s <plik_logów>\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Nie można otworzyć pliku");
        return 1;
    }

    char line[LINE_SIZE];
    long count_info = 0, count_warn = 0, count_error = 0;
    long total_lines = 0;

    while (fgets(line, sizeof(line), file)) {
        total_lines++;

        if (strstr(line, "INFO")) count_info++;
        if (strstr(line, "WARNING")) count_warn++;
        if (strstr(line, "ERROR")) count_error++;
    }

    fclose(file);

    printf("=== Analiza logów ===\n");
    printf("Plik: %s\n", filename);
    printf("Łączna liczba linii: %ld\n", total_lines);
    printf("INFO: %ld\n", count_info);
    printf("WARNING: %ld\n", count_warn);
    printf("ERROR: %ld\n", count_error);

    // zapis wyników do pliku
    FILE *out = fopen("results/results_seq.txt", "w");
    if (out) {
        fprintf(out, "Plik: %s\n", filename);
        fprintf(out, "Łączna liczba linii: %ld\n", total_lines);
        fprintf(out, "INFO: %ld\n", count_info);
        fprintf(out, "WARNING: %ld\n", count_warn);
        fprintf(out, "ERROR: %ld\n", count_error);
        fclose(out);
    } else {
        perror("Nie można zapisać wyników");
    }

    return 0;
}
