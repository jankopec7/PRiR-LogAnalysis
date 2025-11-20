#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <omp.h>

#define LINE_SIZE 2048
#define INITIAL_WORD_CAP 1024
#define TOP_N 10
#define MAX_LINES 1000000  // można zwiększyć w razie czego

typedef struct {
    char *word;
    long count;
} WordCount;

// ==============================
// Wczytanie pliku do tablicy linii
// ==============================
static int load_file(const char *filename, char ***lines_out) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Nie można otworzyć pliku");
        return -1;
    }

    char **lines = malloc(MAX_LINES * sizeof(char *));
    if (!lines) {
        fprintf(stderr, "Błąd: brak pamięci na tablicę linii.\n");
        fclose(file);
        return -1;
    }

    char buffer[LINE_SIZE];
    int count = 0;

    while (fgets(buffer, sizeof(buffer), file) && count < MAX_LINES) {
        lines[count] = strdup(buffer);
        if (!lines[count]) {
            fprintf(stderr, "Błąd: brak pamięci przy kopiowaniu linii.\n");
            // sprzątanie dotychczasowych linii
            for (int i = 0; i < count; i++) free(lines[i]);
            free(lines);
            fclose(file);
            return -1;
        }
        count++;
    }

    fclose(file);
    *lines_out = lines;
    return count;
}

// ==============================
// Funkcje związane z top-N słów
// ==============================

// Dodanie słowa do lokalnej tablicy (per wątek)
static void add_word_local(WordCount **arr, int *size, int *cap, const char *word) {
    if (word[0] == '\0') return;

    for (int i = 0; i < *size; i++) {
        if (strcmp((*arr)[i].word, word) == 0) {
            (*arr)[i].count++;
            return;
        }
    }

    if (*size >= *cap) {
        *cap *= 2;
        WordCount *tmp = realloc(*arr, (*cap) * sizeof(WordCount));
        if (!tmp) {
            fprintf(stderr, "Błąd: brak pamięci przy rozszerzaniu tablicy słów.\n");
            exit(1);
        }
        *arr = tmp;
    }

    (*arr)[*size].word = strdup(word);
    if (!(*arr)[*size].word) {
        fprintf(stderr, "Błąd: brak pamięci przy kopiowaniu słowa.\n");
        exit(1);
    }
    (*arr)[*size].count = 1;
    (*size)++;
}

// Merge jednego słowa z liczbą wystąpień do globalnej tablicy
static void merge_word_global(WordCount **global, int *gsize, int *gcap,
                              const char *word, long count) {
    for (int i = 0; i < *gsize; i++) {
        if (strcmp((*global)[i].word, word) == 0) {
            (*global)[i].count += count;
            return;
        }
    }

    if (*gsize >= *gcap) {
        *gcap *= 2;
        WordCount *tmp = realloc(*global, (*gcap) * sizeof(WordCount));
        if (!tmp) {
            fprintf(stderr, "Błąd: brak pamięci przy rozszerzaniu globalnej tablicy słów.\n");
            exit(1);
        }
        *global = tmp;
    }

    (*global)[*gsize].word = strdup(word);
    if (!(*global)[*gsize].word) {
        fprintf(stderr, "Błąd: brak pamięci przy kopiowaniu słowa globalnie.\n");
        exit(1);
    }
    (*global)[*gsize].count = count;
    (*gsize)++;
}

// Komparator do sortowania top-N
static int cmp_wordcount(const void *a, const void *b) {
    const WordCount *wa = (const WordCount *)a;
    const WordCount *wb = (const WordCount *)b;
    if (wa->count < wb->count) return 1;
    if (wa->count > wb->count) return -1;
    return strcmp(wa->word, wb->word);
}

// Tokenizacja linii: pominięcie timestampu i prefiksu, zamiana na małe litery, split po nie-alfanumerycznych
static void process_line_words(const char *line, WordCount **words, int *size, int *cap) {
    char buf[LINE_SIZE];

    const char *msg_start = line;
    const char *p;

    if ((p = strstr(line, "INFO")) != NULL) msg_start = p;
    else if ((p = strstr(line, "WARN")) != NULL) msg_start = p;
    else if ((p = strstr(line, "ERROR")) != NULL) msg_start = p;

    strncpy(buf, msg_start, LINE_SIZE - 1);
    buf[LINE_SIZE - 1] = '\0';

    for (int i = 0; buf[i] != '\0'; i++) {
        unsigned char c = (unsigned char)buf[i];
        if (isalnum(c)) {
            buf[i] = (char)tolower(c);
        } else {
            buf[i] = ' ';
        }
    }

    char *saveptr = NULL;
    char *token = strtok_r(buf, " ", &saveptr);
    while (token) {
        if (strlen(token) > 1) {
            add_word_local(words, size, cap, token);
        }
        token = strtok_r(NULL, " ", &saveptr);
    }
}

// ==============================
// Analiza logów: poziomy + godziny
// ==============================

static int parse_hour(const char *line) {
    int hour = -1;
    if (sscanf(line, "%*d-%*d-%*d %2d:%*d:%*d", &hour) == 1) {
        if (hour >= 0 && hour < 24) return hour;
    }
    return -1;
}

static int is_info_line(const char *line) {
    return strstr(line, "INFO") != NULL;
}

static int is_warn_line(const char *line) {
    if (strstr(line, "WARN") != NULL) return 1;
    return 0;
}

static int is_error_line(const char *line) {
    return strstr(line, "ERROR") != NULL;
}

// ==============================
// main – równoległa analiza OpenMP
// ==============================

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

    long g_total_lines = total_lines;
    long g_info = 0, g_warn = 0, g_error = 0;
    long g_hourly_info[24] = {0};
    long g_hourly_warn[24] = {0};
    long g_hourly_error[24] = {0};

    int g_words_cap = INITIAL_WORD_CAP;
    int g_words_size = 0;
    WordCount *g_words = malloc(g_words_cap * sizeof(WordCount));
    if (!g_words) {
        fprintf(stderr, "Błąd: brak pamięci na globalną tablicę słów.\n");
        for (int i = 0; i < total_lines; i++) free(lines[i]);
        free(lines);
        return 1;
    }

    double t_start = omp_get_wtime();

    // ======================
    // Sekcja równoległa
    // ======================
    #pragma omp parallel
    {
        long l_info = 0, l_warn = 0, l_error = 0;
        long l_hourly_info[24] = {0};
        long l_hourly_warn[24] = {0};
        long l_hourly_error[24] = {0};

        int l_words_cap = INITIAL_WORD_CAP;
        int l_words_size = 0;
        WordCount *l_words = malloc(l_words_cap * sizeof(WordCount));
        if (!l_words) {
            fprintf(stderr, "Błąd: brak pamięci na lokalną tablicę słów.\n");
            // brutalnie, ale w pracy projektowej OK:
            #pragma omp critical
            {
                fprintf(stderr, "Wątek %d: brak pamięci – przerywam.\n", omp_get_thread_num());
            }
            exit(1);
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < total_lines; i++) {
            const char *line = lines[i];

            int info = is_info_line(line);
            int warn = is_warn_line(line);
            int error = is_error_line(line);

            if (info)  l_info++;
            if (warn)  l_warn++;
            if (error) l_error++;

            int hour = parse_hour(line);
            if (hour >= 0 && hour < 24) {
                if (info)  l_hourly_info[hour]++;
                if (warn)  l_hourly_warn[hour]++;
                if (error) l_hourly_error[hour]++;
            }

            process_line_words(line, &l_words, &l_words_size, &l_words_cap);
        }

        // Scalanie lokalnych wyników do globalnych
        #pragma omp critical
        {
            g_info += l_info;
            g_warn += l_warn;
            g_error += l_error;

            for (int h = 0; h < 24; h++) {
                g_hourly_info[h]  += l_hourly_info[h];
                g_hourly_warn[h]  += l_hourly_warn[h];
                g_hourly_error[h] += l_hourly_error[h];
            }

            // Merge lokalnych słów do globalnej tablicy
            for (int k = 0; k < l_words_size; k++) {
                merge_word_global(&g_words, &g_words_size, &g_words_cap,
                                  l_words[k].word, l_words[k].count);
            }
        }

        // Sprzątanie lokalnej tablicy słów
        for (int k = 0; k < l_words_size; k++) {
            free(l_words[k].word);
        }
        free(l_words);
    }

    double t_end = omp_get_wtime();
    double elapsed = t_end - t_start;
    int threads = omp_get_max_threads();

    // Sortowanie top-N
    qsort(g_words, g_words_size, sizeof(WordCount), cmp_wordcount);
    int limit = (g_words_size < TOP_N) ? g_words_size : TOP_N;

    // ======================
    // Wypisywanie wyników
    // ======================
    printf("=== Analiza logów (OpenMP) ===\n");
    printf("Plik: %s\n", filename);
    printf("Liczba linii: %ld\n", g_total_lines);
    printf("INFO:   %ld\n", g_info);
    printf("WARN:   %ld\n", g_warn);
    printf("ERROR:  %ld\n", g_error);
    printf("Liczba wątków: %d\n", threads);
    printf("Czas wykonania (sekcje równoległej): %.6f s\n", elapsed);

    printf("\nStatystyki godzinowe:\n");
    printf("Godz   INFO      WARN      ERROR\n");
    for (int h = 0; h < 24; h++) {
        printf("%02d   %8ld  %8ld  %8ld\n",
               h, g_hourly_info[h], g_hourly_warn[h], g_hourly_error[h]);
    }

    printf("\nTop-%d słów:\n", TOP_N);
    for (int i = 0; i < limit; i++) {
        printf("%2d. %-20s %8ld\n", i + 1, g_words[i].word, g_words[i].count);
    }

    // Zapis do pliku
    FILE *out = fopen("results/results_openmp.txt", "w");
    if (out) {
        fprintf(out, "=== Analiza logów (OpenMP) ===\n");
        fprintf(out, "Plik: %s\n", filename);
        fprintf(out, "Liczba linii: %ld\n", g_total_lines);
        fprintf(out, "INFO:   %ld\n", g_info);
        fprintf(out, "WARN:   %ld\n", g_warn);
        fprintf(out, "ERROR:  %ld\n", g_error);
        fprintf(out, "Liczba wątków: %d\n", threads);
        fprintf(out, "Czas wykonania: %.6f s\n", elapsed);

        fprintf(out, "\nStatystyki godzinowe:\n");
        fprintf(out, "Godz   INFO      WARN      ERROR\n");
        for (int h = 0; h < 24; h++) {
            fprintf(out, "%02d   %8ld  %8ld  %8ld\n",
                    h, g_hourly_info[h], g_hourly_warn[h], g_hourly_error[h]);
        }

        fprintf(out, "\nTop-%d słów:\n", TOP_N);
        for (int i = 0; i < limit; i++) {
            fprintf(out, "%2d. %-20s %8ld\n", i + 1, g_words[i].word, g_words[i].count);
        }

        fclose(out);
    } else {
        perror("Nie można zapisać wyników do results/results_openmp.txt");
    }

    // Sprzątanie pamięci
    for (int i = 0; i < total_lines; i++) {
        free(lines[i]);
    }
    free(lines);

    for (int i = 0; i < g_words_size; i++) {
        free(g_words[i].word);
    }
    free(g_words);

    return 0;
}
