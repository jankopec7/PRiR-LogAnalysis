#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define LINE_SIZE 2048
#define INITIAL_WORD_CAP 1024
#define TOP_N 10

typedef struct {
    char *word;
    long count;
} WordCount;

// Znajdź lub dodaj słowo do tablicy słów
static void add_word(WordCount **arr, int *size, int *cap, const char *word) {
    if (word[0] == '\0') return;

    // szukamy, czy słowo już jest w tablicy
    for (int i = 0; i < *size; i++) {
        if (strcmp((*arr)[i].word, word) == 0) {
            (*arr)[i].count++;
            return;
        }
    }

    // jeśli nie ma – dodajemy nowe
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

// Porównanie do sortowania top-N (malejąco po liczności, rosnąco po nazwie)
static int cmp_wordcount(const void *a, const void *b) {
    const WordCount *wa = (const WordCount *)a;
    const WordCount *wb = (const WordCount *)b;
    if (wa->count < wb->count) return 1;
    if (wa->count > wb->count) return -1;
    return strcmp(wa->word, wb->word);
}

// Prosta funkcja tokenizująca – zamienia słowa na małe litery i wycina znaki niealfanumeryczne
static void process_line_words(const char *line, WordCount **words, int *size, int *cap) {
    char buf[LINE_SIZE];
    // Szukamy poziomu logu, żeby pominąć timestamp i prefiks
    const char *msg_start = line;
    const char *p;

    if ((p = strstr(line, "INFO")) != NULL) msg_start = p;
    else if ((p = strstr(line, "WARN")) != NULL) msg_start = p;
    else if ((p = strstr(line, "ERROR")) != NULL) msg_start = p;

    // kopiujemy od msg_start
    strncpy(buf, msg_start, LINE_SIZE - 1);
    buf[LINE_SIZE - 1] = '\0';

    // zamieniamy na małe litery i zamieniamy separatory na spacje
    for (int i = 0; buf[i] != '\0'; i++) {
        unsigned char c = (unsigned char)buf[i];
        if (isalnum(c)) {
            buf[i] = (char)tolower(c);
        } else {
            buf[i] = ' ';
        }
    }

    // tokenizacja po spacji
    char *saveptr = NULL;
    char *token = strtok_r(buf, " ", &saveptr);
    while (token) {
        // pomijamy bardzo krótkie "słowa" jednoznakowe
        if (strlen(token) > 1) {
            add_word(words, size, cap, token);
        }
        token = strtok_r(NULL, " ", &saveptr);
    }
}

// Parsowanie godziny z formatu: "YYYY-MM-DD HH:MM:SS,ms ..."
static int parse_hour(const char *line) {
    int hour = -1;
    // Próbujemy sparsować "rok-mies-dzień godz:min:sek"
    if (sscanf(line, "%*d-%*d-%*d %2d:%*d:%*d", &hour) == 1) {
        if (hour >= 0 && hour < 24) return hour;
    }
    return -1;
}

// Sprawdzenie poziomu logu w linii
static int is_info_line(const char *line) {
    return strstr(line, "INFO") != NULL;
}

static int is_warn_line(const char *line) {
    // Obsługuje zarówno WARN jak i WARNING
    if (strstr(line, "WARN") != NULL) return 1;
    return 0;
}

static int is_error_line(const char *line) {
    return strstr(line, "ERROR") != NULL;
}

// Normalizacja nazwy poziomu dla filtrowania
typedef enum {
    FILTER_NONE = 0,
    FILTER_INFO,
    FILTER_WARN,
    FILTER_ERROR
} FilterLevel;

static FilterLevel parse_filter_level(const char *s) {
    if (!s) return FILTER_NONE;

    if (strcasecmp(s, "INFO") == 0) return FILTER_INFO;
    if (strcasecmp(s, "WARNING") == 0) return FILTER_WARN;
    if (strcasecmp(s, "WARN") == 0) return FILTER_WARN;
    if (strcasecmp(s, "ERROR") == 0) return FILTER_ERROR;
    return FILTER_NONE;
}

static const char *filter_name(FilterLevel f) {
    switch (f) {
        case FILTER_INFO:  return "INFO";
        case FILTER_WARN:  return "WARN";
        case FILTER_ERROR: return "ERROR";
        default:           return "NONE";
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Użycie: %s <plik_logów> [INFO|WARN|WARNING|ERROR]\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    FilterLevel filter = FILTER_NONE;

    if (argc >= 3) {
        filter = parse_filter_level(argv[2]);
        if (filter == FILTER_NONE) {
            fprintf(stderr, "Uwaga: nieznany poziom filtrowania '%s'. Filtrowanie zostanie pominięte.\n", argv[2]);
        }
    }

    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Nie można otworzyć pliku");
        return 1;
    }

    FILE *filter_out = NULL;
    if (filter != FILTER_NONE) {
        char outname[256];
        snprintf(outname, sizeof(outname), "results/filtered_%s.txt", filter_name(filter));
        filter_out = fopen(outname, "w");
        if (!filter_out) {
            perror("Nie można otworzyć pliku do zapisu przefiltrowanych wyników");
            // filtrowanie po prostu wyłączymy
            filter = FILTER_NONE;
        }
    }

    char line[LINE_SIZE];
    long total_lines = 0;
    long count_info = 0, count_warn = 0, count_error = 0;

    long hourly_info[24] = {0};
    long hourly_warn[24] = {0};
    long hourly_error[24] = {0};

    long filtered_lines = 0;

    // Struktury do top-N słów
    int words_cap = INITIAL_WORD_CAP;
    int words_size = 0;
    WordCount *words = malloc(words_cap * sizeof(WordCount));
    if (!words) {
        fprintf(stderr, "Błąd: brak pamięci na tablicę słów.\n");
        fclose(file);
        return 1;
    }

    while (fgets(line, sizeof(line), file)) {
        total_lines++;

        int info = is_info_line(line);
        int warn = is_warn_line(line);
        int error = is_error_line(line);

        if (info)  count_info++;
        if (warn)  count_warn++;
        if (error) count_error++;

        // Statystyka godzinowa
        int hour = parse_hour(line);
        if (hour >= 0 && hour < 24) {
            if (info)  hourly_info[hour]++;
            if (warn)  hourly_warn[hour]++;
            if (error) hourly_error[hour]++;
        }

        // Filtrowanie – jeśli włączone i linia pasuje, zapisujemy ją
        if (filter != FILTER_NONE) {
            int match = 0;
            if (filter == FILTER_INFO && info) match = 1;
            else if (filter == FILTER_WARN && warn) match = 1;
            else if (filter == FILTER_ERROR && error) match = 1;

            if (match && filter_out) {
                fputs(line, filter_out);
                filtered_lines++;
            }
        }

        // Tokenizacja do top-N słów
        process_line_words(line, &words, &words_size, &words_cap);
    }

    fclose(file);
    if (filter_out) fclose(filter_out);

    // Sortujemy tablicę słów
    qsort(words, words_size, sizeof(WordCount), cmp_wordcount);

    // ==========================
    // WYPISANIE WYNIKÓW
    // ==========================
    printf("=== Analiza logów (sekwencyjna) ===\n");
    printf("Plik: %s\n", filename);
    printf("Łączna liczba linii: %ld\n", total_lines);
    printf("INFO:   %ld\n", count_info);
    printf("WARN:   %ld\n", count_warn);
    printf("ERROR:  %ld\n", count_error);

    if (filter != FILTER_NONE) {
        printf("\nFiltrowanie: poziom %s\n", filter_name(filter));
        printf("Liczba linii spełniających kryterium: %ld\n", filtered_lines);
        printf("Plik wynikowy (jeśli zapis się powiódł): results/filtered_%s.txt\n",
               filter_name(filter));
    }

    printf("\nStatystyki godzinowe (liczba linii danego poziomu):\n");
    printf("Godz  INFO      WARN      ERROR\n");
    for (int h = 0; h < 24; h++) {
        printf("%02d   %8ld  %8ld  %8ld\n",
               h, hourly_info[h], hourly_warn[h], hourly_error[h]);
    }

    printf("\nTop-%d najczęściej występujących słów (po przetworzeniu treści logów):\n", TOP_N);
    int limit = (words_size < TOP_N) ? words_size : TOP_N;
    for (int i = 0; i < limit; i++) {
        printf("%2d. %-20s %8ld\n", i + 1, words[i].word, words[i].count);
    }

    // ==========================
    // ZAPIS WYNIKÓW DO PLIKU
    // ==========================
    FILE *out = fopen("results/results_seq.txt", "w");
    if (out) {
        fprintf(out, "=== Analiza logów (sekwencyjna) ===\n");
        fprintf(out, "Plik: %s\n", filename);
        fprintf(out, "Łączna liczba linii: %ld\n", total_lines);
        fprintf(out, "INFO:   %ld\n", count_info);
        fprintf(out, "WARN:   %ld\n", count_warn);
        fprintf(out, "ERROR:  %ld\n", count_error);

        if (filter != FILTER_NONE) {
            fprintf(out, "\nFiltrowanie: poziom %s\n", filter_name(filter));
            fprintf(out, "Liczba linii spełniających kryterium: %ld\n", filtered_lines);
            fprintf(out, "Plik wynikowy: results/filtered_%s.txt\n", filter_name(filter));
        }

        fprintf(out, "\nStatystyki godzinowe (liczba linii danego poziomu):\n");
        fprintf(out, "Godz  INFO      WARN      ERROR\n");
        for (int h = 0; h < 24; h++) {
            fprintf(out, "%02d   %8ld  %8ld  %8ld\n",
                    h, hourly_info[h], hourly_warn[h], hourly_error[h]);
        }

        fprintf(out, "\nTop-%d najczęściej występujących słów:\n", TOP_N);
        for (int i = 0; i < limit; i++) {
            fprintf(out, "%2d. %-20s %8ld\n", i + 1, words[i].word, words[i].count);
        }

        fclose(out);
    } else {
        perror("Nie można zapisać wyników do results/results_seq.txt");
    }

    // Sprzątanie pamięci
    for (int i = 0; i < words_size; i++) {
        free(words[i].word);
    }
    free(words);

    return 0;
}
