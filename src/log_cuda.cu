#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// Makro do sprawdzania błędów CUDA
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel: każdy wątek przetwarza kilka linii logów
__global__ void count_levels_kernel(
    const char *buffer,
    const int *line_starts,
    int num_lines,
    int buffer_len,
    unsigned long long *info_count,
    unsigned long long *warn_count,
    unsigned long long *error_count)
{
    unsigned long long local_info  = 0;
    unsigned long long local_warn  = 0;
    unsigned long long local_error = 0;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int line_idx = tid; line_idx < num_lines; line_idx += stride) {
        int start = line_starts[line_idx];
        int end = (line_idx + 1 < num_lines) ? line_starts[line_idx + 1] : buffer_len;

        if (start >= buffer_len || start >= end)
            continue;

        int has_info = 0, has_warn = 0, has_error = 0;

        // Przeszukujemy linię znak po znaku
        for (int i = start; i < end; ++i) {
            char c = buffer[i];

            // INFO
            if (!has_info && c == 'I' && i + 3 < end) {
                if (buffer[i + 1] == 'N' &&
                    buffer[i + 2] == 'F' &&
                    buffer[i + 3] == 'O') {
                    has_info = 1;
                }
            }

            // WARN (pasuje też WARNING, bo początek to WARN)
            if (!has_warn && c == 'W' && i + 3 < end) {
                if (buffer[i + 1] == 'A' &&
                    buffer[i + 2] == 'R' &&
                    buffer[i + 3] == 'N') {
                    has_warn = 1;
                }
            }

            // ERROR
            if (!has_error && c == 'E' && i + 4 < end) {
                if (buffer[i + 1] == 'R' &&
                    buffer[i + 2] == 'R' &&
                    buffer[i + 3] == 'O' &&
                    buffer[i + 4] == 'R') {
                    has_error = 1;
                }
            }

            if (has_info && has_warn && has_error) {
                break;
            }
        }

        if (has_info)  local_info++;
        if (has_warn)  local_warn++;
        if (has_error) local_error++;
    }

    if (local_info > 0)  atomicAdd(info_count,  local_info);
    if (local_warn > 0)  atomicAdd(warn_count,  local_warn);
    if (local_error > 0) atomicAdd(error_count, local_error);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Użycie: %s <plik_logów>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *filename = argv[1];

    // --- Wczytanie pliku na CPU ---
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("Nie można otworzyć pliku");
        return EXIT_FAILURE;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        perror("fseek");
        fclose(f);
        return EXIT_FAILURE;
    }

    long file_size = ftell(f);
    if (file_size < 0) {
        perror("ftell");
        fclose(f);
        return EXIT_FAILURE;
    }
    rewind(f);

    char *buffer = (char *)malloc(file_size);
    if (!buffer) {
        fprintf(stderr, "Błąd: brak pamięci na buffer.\n");
        fclose(f);
        return EXIT_FAILURE;
    }

    size_t read_bytes = fread(buffer, 1, file_size, f);
    fclose(f);

    if (read_bytes != (size_t)file_size) {
        fprintf(stderr, "Uwaga: odczytano mniej bajtów niż rozmiar pliku.\n");
        file_size = (long)read_bytes;
    }

    // --- Wyznaczanie początków linii ---
    int approx_lines = 1;
    for (long i = 0; i < file_size; ++i) {
        if (buffer[i] == '\n') approx_lines++;
    }

    int *line_starts = (int *)malloc(approx_lines * sizeof(int));
    if (!line_starts) {
        fprintf(stderr, "Błąd: brak pamięci na line_starts.\n");
        free(buffer);
        return EXIT_FAILURE;
    }

    int num_lines = 0;
    line_starts[num_lines++] = 0;  // pierwsza linia startuje od 0

    for (long i = 0; i < file_size; ++i) {
        if (buffer[i] == '\n' && i + 1 < file_size) {
            line_starts[num_lines++] = (int)(i + 1);
        }
    }

    // --- Alokacja na GPU ---
    char *d_buffer = NULL;
    int  *d_line_starts = NULL;
    unsigned long long *d_info = NULL, *d_warn = NULL, *d_error = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_buffer, file_size));
    CUDA_CHECK(cudaMalloc((void **)&d_line_starts, num_lines * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_info,  sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc((void **)&d_warn,  sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc((void **)&d_error, sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemcpy(d_buffer, buffer, file_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_line_starts, line_starts,
                          num_lines * sizeof(int), cudaMemcpyHostToDevice));

    unsigned long long zero = 0;
    CUDA_CHECK(cudaMemcpy(d_info,  &zero, sizeof(unsigned long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_warn,  &zero, sizeof(unsigned long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_error, &zero, sizeof(unsigned long long), cudaMemcpyHostToDevice));

    // --- Uruchomienie kernela ---
    int blocks = (num_lines + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 1024) blocks = 1024;  // bezpieczeństwo

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    count_levels_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_buffer,
        d_line_starts,
        num_lines,
        (int)file_size,
        d_info,
        d_warn,
        d_error
    );

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // --- Odczyt wyników ---
    unsigned long long h_info = 0, h_warn = 0, h_error = 0;
    CUDA_CHECK(cudaMemcpy(&h_info,  d_info,  sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_warn,  d_warn,  sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_error, d_error, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    // --- Wypisanie wyników ---
    printf("=== Analiza logów (CUDA – poziomy logów) ===\n");
    printf("Plik: %s\n", filename);
    printf("Szacowana liczba linii (na podstawie \\n): %d\n", num_lines);
    printf("INFO:   %llu\n", (unsigned long long)h_info);
    printf("WARN:   %llu\n", (unsigned long long)h_warn);
    printf("ERROR:  %llu\n", (unsigned long long)h_error);
    printf("Czas kernela: %.3f ms\n", ms);

    // --- Sprzątanie ---
    free(buffer);
    free(line_starts);

    CUDA_CHECK(cudaFree(d_buffer));
    CUDA_CHECK(cudaFree(d_line_starts));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_warn));
    CUDA_CHECK(cudaFree(d_error));

    return 0;
}
