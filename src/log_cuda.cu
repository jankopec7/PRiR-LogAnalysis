#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void count_logs_kernel(char *data, int n, int *info, int *warn, int *error) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // TODO: Równoległe wyszukiwanie słów "INFO", "WARNING", "ERROR"

}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Użycie: %s <plik_logów>\n", argv[0]);
        return 1;
    }

    char *filename = argv[1];
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("Błąd otwarcia pliku");
        return 1;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *data = (char *)malloc(size + 1);
    fread(data, 1, size, f);
    fclose(f);
    data[size] = '\0';

    // TODO: Alokacja na GPU i uruchomienie kernela
    // Wyniki zliczania zwrócić do CPU

    printf("=== Analiza logów (CUDA) ===\n");
    printf("Plik: %s\n", filename);
    printf("INFO: ?\nWARNING: ?\nERROR: ?\n");

    free(data);
    return 0;
}
