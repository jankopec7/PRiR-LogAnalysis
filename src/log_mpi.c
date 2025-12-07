#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define LINE_SIZE 4096
#define INITIAL_WORD_CAP 1024
#define TOP_N 10

// ===============================
// Struktury do liczenia słów
// ===============================
typedef struct {
    char *word;
    long count;
} WordCount;

static void add_word(WordCount **arr, int *size, int *cap, const char *word) {
    if (word[0] == '\0') return;

    for (int i = 0; i < *size; i++) {
        if (strcmp((*arr)[i].word, word) == 0) {
            (*arr)[i].count++;
            return;
        }
    }

    if (*size >= *cap) {
        *cap *= 2;
        *arr = realloc(*arr, (*cap) * sizeof(WordCount));
        if (!*arr) {
            fprintf(stderr, "Brak pamięci w add_word\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    (*arr)[*size].word = strdup(word);
    (*arr)[*size].count = 1;
    (*size)++;
}

static int cmp_wordcount(const void *a, const void *b) {
    const WordCount *wa = a, *wb = b;
    if (wa->count < wb->count) return 1;
    if (wa->count > wb->count) return -1;
    return strcmp(wa->word, wb->word);
}

static void process_line_words(const char *line, WordCount **words, int *size, int *cap) {
    char buf[LINE_SIZE];

    // znajdź początek treści (poziom logu)
    const char *msg = line;
    const char *p;
    if ((p = strstr(line, "INFO"))) msg = p;
    else if ((p = strstr(line, "WARN"))) msg = p;
    else if ((p = strstr(line, "ERROR"))) msg = p;

    strncpy(buf, msg, LINE_SIZE - 1);
    buf[LINE_SIZE - 1] = '\0';

    for (int i = 0; buf[i]; i++) {
        if (isalnum((unsigned char)buf[i])) buf[i] = tolower(buf[i]);
        else buf[i] = ' ';
    }

    char *saveptr = NULL;
    char *tok = strtok_r(buf, " ", &saveptr);
    while (tok) {
        if (strlen(tok) > 1)
            add_word(words, size, cap, tok);
        tok = strtok_r(NULL, " ", &saveptr);
    }
}

static int parse_hour(const char *line) {
    int h;
    if (sscanf(line, "%*d-%*d-%*d %2d:%*d:%*d", &h) == 1)
        if (h >= 0 && h < 24) return h;
    return -1;
}

static int is_info(const char *l) { return strstr(l, "INFO") != NULL; }
static int is_warn(const char *l) { return strstr(l, "WARN") != NULL; }
static int is_error(const char *l) { return strstr(l, "ERROR") != NULL; }

// ===============================================
// GŁÓWNY PROGRAM MPI
// ===============================================
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

    const char *filename = argv[1];
    FILE *f = fopen(filename, "r");
    if (!f) {
        if (rank == 0) perror("Nie można otworzyć pliku");
        MPI_Finalize();
        return 1;
    }

    // ========================
    // Podział pliku na chunk'i
    // ========================
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    long chunk = fsize / size;
    long start = rank * chunk;
    long end   = (rank == size - 1) ? fsize : (rank + 1) * chunk;

    // skok do początku chunku
    fseek(f, start, SEEK_SET);

    // jeżeli nie jesteśmy procesem 0, odetnij pierwszą niedokończoną linię
    if (rank != 0) {
        char tmp[LINE_SIZE];
        fgets(tmp, LINE_SIZE, f);
    }

    // =====================================
    // Lokalne liczniki
    // =====================================
    long local_info = 0, local_warn = 0, local_error = 0;
    long hourly_info[24] = {0};
    long hourly_warn[24] = {0};
    long hourly_error[24] = {0};

    WordCount *words = malloc(INITIAL_WORD_CAP * sizeof(WordCount));
    int words_size = 0, words_cap = INITIAL_WORD_CAP;

    char line[LINE_SIZE];

    while (ftell(f) < end && fgets(line, LINE_SIZE, f)) {

        int info = is_info(line);
        int warn = is_warn(line);
        int err  = is_error(line);

        if (info) local_info++;
        if (warn) local_warn++;
        if (err)  local_error++;

        int h = parse_hour(line);
        if (h >= 0) {
            if (info) hourly_info[h]++;
            if (warn) hourly_warn[h]++;
            if (err)  hourly_error[h]++;
        }

        process_line_words(line, &words, &words_size, &words_cap);
    }

    fclose(f);

    // ===========================
    // REDUKCJE
    // ===========================
    long total_info, total_warn, total_error;
    MPI_Reduce(&local_info, &total_info, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_warn, &total_warn, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_error, &total_error, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    long global_hourly_info[24];
    long global_hourly_warn[24];
    long global_hourly_error[24];

    MPI_Reduce(hourly_info,  global_hourly_info,  24, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(hourly_warn,  global_hourly_warn,  24, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(hourly_error, global_hourly_error, 24, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // ===========================
    // REDUKCJA histogramu słów
    // rank 0 zbiera
    // ===========================
    if (rank != 0) {
        // wyślij ile słów
        MPI_Send(&words_size, 1, MPI_INT, 0, 100, MPI_COMM_WORLD);

        // wyślij każde słowo + count
        for (int i = 0; i < words_size; i++) {
            int len = strlen(words[i].word) + 1;
            MPI_Send(&len, 1, MPI_INT, 0, 101, MPI_COMM_WORLD);
            MPI_Send(words[i].word, len, MPI_CHAR, 0, 102, MPI_COMM_WORLD);
            MPI_Send(&words[i].count, 1, MPI_LONG, 0, 103, MPI_COMM_WORLD);
        }
    }

    // MASTER – scalanie histogramów
    WordCount *global_words = NULL;
    int gw_size = 0, gw_cap = INITIAL_WORD_CAP;

    if (rank == 0) {
        global_words = malloc(gw_cap * sizeof(WordCount));

        // dodaj lokalne słowa mastera
        for (int i = 0; i < words_size; i++) {
            add_word(&global_words, &gw_size, &gw_cap, words[i].word);
            global_words[gw_size - 1].count = words[i].count;
        }

        // odbieraj z pozostałych procesów
        for (int r = 1; r < size; r++) {

            int count_words;
            MPI_Recv(&count_words, 1, MPI_INT, r, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < count_words; i++) {
                int len;
                char buffer[LINE_SIZE];
                long c;

                MPI_Recv(&len, 1, MPI_INT, r, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(buffer, len, MPI_CHAR, r, 102, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&c, 1, MPI_LONG, r, 103, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                add_word(&global_words, &gw_size, &gw_cap, buffer);
                global_words[gw_size - 1].count = c;
            }
        }

        // sortowanie top-N
        qsort(global_words, gw_size, sizeof(WordCount), cmp_wordcount);

        // ===========================
        // WYPISANIE WYNIKÓW MPI
        // ===========================
        printf("=== Analiza logów (MPI) ===\n");
        printf("Plik: %s\n", filename);
        printf("INFO:   %ld\n", total_info);
        printf("WARN:   %ld\n", total_warn);
        printf("ERROR:  %ld\n", total_error);

        printf("\nStatystyki godzinowe:\n");
        printf("Godz   INFO      WARN      ERROR\n");
        for (int h = 0; h < 24; h++) {
            printf("%02d   %8ld  %8ld  %8ld\n",
                   h, global_hourly_info[h],
                   global_hourly_warn[h],
                   global_hourly_error[h]);
        }

        printf("\nTop-%d słów:\n", TOP_N);
        int limit = gw_size < TOP_N ? gw_size : TOP_N;
        for (int i = 0; i < limit; i++) {
            printf("%2d. %-20s %8ld\n",
                   i + 1, global_words[i].word, global_words[i].count);
        }

        // zapis wyników
        FILE *out = fopen("results/results_mpi.txt", "w");
        if (out) {
            fprintf(out, "=== Analiza logów (MPI) ===\n");
            fprintf(out, "Plik: %s\n", filename);
            fprintf(out, "INFO:   %ld\n", total_info);
            fprintf(out, "WARN:   %ld\n", total_warn);
            fprintf(out, "ERROR:  %ld\n", total_error);

            fprintf(out, "\nStatystyki godzinowe:\n");
            fprintf(out, "Godz   INFO      WARN      ERROR\n");
            for (int h = 0; h < 24; h++) {
                fprintf(out, "%02d   %8ld  %8ld  %8ld\n",
                        h, global_hourly_info[h],
                        global_hourly_warn[h],
                        global_hourly_error[h]);
            }

            fprintf(out, "\nTop-%d słów:\n", TOP_N);
            for (int i = 0; i < limit; i++) {
                fprintf(out, "%2d. %-20s %8ld\n",
                        i + 1, global_words[i].word, global_words[i].count);
            }

            fclose(out);
        }
    }

    MPI_Finalize();
    return 0;
}
