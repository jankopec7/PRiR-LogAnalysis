#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define LINE_SIZE 512
#define MAX_LINES 200000

// ===============================
// DEVICE: funkcja pomocnicza
// ===============================
__device__ int contains_keyword(const char *line, const char *keyword) {
    int i = 0;
    while (line[i] != '\0') {
        int j = 0;
        while (keyword[j] != '\0' && line[i + j] != '\0' && line[i + j] == keyword[j]) {
            j++;
        }
        if (keyword[j] == '\0') return 1; // znaleziono
        i++;
    }
    return 0;
}

// ===============================
// KERNEL CUDA
// ===============================
__global__ void count_logs_kernel(char *data, int *line_starts, int num_lines, int *info, int *warn, int *error) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_lines) return;

    char *line = data + line_starts[idx];

    if (contains_keyword(line, "INFO")) atomicAdd(info, 1);
    else if (contains_keyword(line, "WARNING")) atomicAdd(warn, 1);
    else if (contains_keyword(line, "ERROR")) atomicAdd(error, 1);
}

// ===============================
// HOST: program główny
// ===============================
int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Użycie: %s <plik_logów>\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("Błąd otwarcia pliku");
        return 1;
    }

    // Wczytaj cały plik
    fseek(f, 0, SEEK_END);
    long filesize = ftell(f);
    rewind(f);

    char *data = (char *)malloc(filesize + 1);
    fread(data, 1, filesize, f);
    data[filesize] = '\0';
    fclose(f);

    // Znajdź początki linii
    int *line_starts = (int *)malloc(MAX_LINES * sizeof(int));
    int num_lines = 0;
    line_starts[num_lines++] = 0;

    for (long i = 0; i < filesize; i++) {
        if (data[i] == '\n' && i + 1 < filesize) {
            line_starts[num_lines++] = i + 1;
        }
        if (num_lines >= MAX_LINES) break;
    }

    // ===============================
    // Alokacja pamięci na GPU
    // ===============================
    char *d_data;
    int *d_line_starts;
    int *d_info, *d_warn, *d_error;

    cudaMalloc(&d_data, filesize + 1);
    cudaMalloc(&d_line_starts, num_lines * sizeof(int));
    cudaMalloc(&d_info, sizeof(int));
    cudaMalloc(&d_warn, sizeof(int));
    cudaMalloc(&d_error, sizeof(int));

    cudaMemcpy(d_data, data, filesize + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_line_starts, line_starts, num_lines * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_info, 0, sizeof(int));
    cudaMemset(d_warn, 0, sizeof(int));
    cudaMemset(d_error, 0, sizeof(int));

    // ===============================
    // Uruchom kernel
    // ===============================
    int threads = 256;
    int blocks = (num_lines + threads - 1) / threads;

    count_logs_kernel<<<blocks, threads>>>(d_data, d_line_starts, num_lines, d_info, d_warn, d_error);
    cudaDeviceSynchronize();

    // ===============================
    // Pobierz wyniki
    // ===============================
    int h_info, h_warn, h_error;
    cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_warn, d_warn, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_error, d_error, sizeof(int), cudaMemcpyDeviceToHost);

    // ===============================
    // Wynik
    // ===============================
    printf("=== Analiza logów (CUDA) ===\n");
    printf("Plik: %s\n", filename);
    printf("Liczba linii: %d\n", num_lines);
    printf("INFO: %d\n", h_info);
    printf("WARNING: %d\n", h_warn);
    printf("ERROR: %d\n", h_error);

    // ===============================
    // Sprzątanie
    // ===============================
    cudaFree(d_data);
    cudaFree(d_line_starts);
    cudaFree(d_info);
    cudaFree(d_warn);
    cudaFree(d_error);
    free(data);
    free(line_starts);

    return 0;
}
