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

/* --------------------------- Utils --------------------------- */

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
        WordCount *tmp = realloc(*arr, (*cap) * sizeof(WordCount));
        if (!tmp) {
            fprintf(stderr, "Memory error expanding top-N table\n");
            exit(1);
        }
        *arr = tmp;
    }

    (*arr)[*size].word = strdup(word);
    (*arr)[*size].count = 1;
    (*size)++;
}

static int cmp_words(const void *a, const void *b) {
    const WordCount *wa = a, *wb = b;
    if (wa->count != wb->count)
        return (wb->count - wa->count);
    return strcmp(wa->word, wb->word);
}

static void tokenize_message(const char *line, WordCount **words, int *size, int *cap) {
    char buf[LINE_SIZE];

    const char *start = line;
    const char *p;

    if ((p = strstr(line, "INFO"))) start = p;
    else if ((p = strstr(line, "WARN"))) start = p;
    else if ((p = strstr(line, "ERROR"))) start = p;

    strncpy(buf, start, LINE_SIZE - 1);
    buf[LINE_SIZE - 1] = '\0';

    for (int i = 0; buf[i]; i++)
        buf[i] = isalnum((unsigned char)buf[i]) ? tolower(buf[i]) : ' ';

    char *save, *tok = strtok_r(buf, " ", &save);
    while (tok) {
        if (strlen(tok) > 1)
            add_word(words, size, cap, tok);
        tok = strtok_r(NULL, " ", &save);
    }
}

static int parse_hour(const char *line) {
    int hour = -1;
    if (sscanf(line, "%*d-%*d-%*d %2d:%*d:%*d", &hour) == 1)
        return (hour >= 0 && hour < 24) ? hour : -1;
    return -1;
}

/* --------------------------- Filter enum --------------------------- */

typedef enum { FILTER_NONE=0, FILTER_INFO, FILTER_WARN, FILTER_ERROR } Filter;

static Filter parse_filter(const char *s) {
    if (!s) return FILTER_NONE;
    if (!strcasecmp(s,"INFO")) return FILTER_INFO;
    if (!strcasecmp(s,"WARN")) return FILTER_WARN;
    if (!strcasecmp(s,"WARNING")) return FILTER_WARN;
    if (!strcasecmp(s,"ERROR")) return FILTER_ERROR;
    return FILTER_NONE;
}

static const char *filter_name(Filter f) {
    if (f==FILTER_INFO) return "INFO";
    if (f==FILTER_WARN) return "WARN";
    if (f==FILTER_ERROR) return "ERROR";
    return "NONE";
}

/* --------------------------- MAIN --------------------------- */

int main(int argc, char *argv[]) {

    if (argc < 2) {
        printf("Użycie:\n");
        printf("  %s <plik_logów>           - pełna analiza\n", argv[0]);
        printf("  %s <plik_logów> LEVEL     - filtr (INFO/WARN/ERROR)\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    Filter filt = (argc >= 3) ? parse_filter(argv[2]) : FILTER_NONE;

    /* ------------------ Open file ------------------ */
    FILE *f = fopen(filename, "r");
    if (!f) { perror("file"); return 1; }

    /* ---------- FULL ANALYSIS MODE ---------- */
    if (filt == FILTER_NONE) {

        long count_info = 0, count_warn = 0, count_error = 0;
        long hourly_info[24]={0}, hourly_warn[24]={0}, hourly_error[24]={0};

        int cap = INITIAL_WORD_CAP, size = 0;
        WordCount *words = malloc(cap * sizeof(WordCount));

        char line[LINE_SIZE];
        long total = 0;

        while (fgets(line, LINE_SIZE, f)) {
            total++;

            int hour = parse_hour(line);

            if (strstr(line,"INFO"))  { count_info++;  if (hour>=0) hourly_info[hour]++; }
            if (strstr(line,"WARN"))  { count_warn++;  if (hour>=0) hourly_warn[hour]++; }
            if (strstr(line,"ERROR")) { count_error++; if (hour>=0) hourly_error[hour]++; }

            tokenize_message(line, &words, &size, &cap);
        }

        qsort(words, size, sizeof(WordCount), cmp_words);

        printf("=== Pełna analiza (sekwencyjna) ===\n");
        printf("Plik: %s\n", filename);
        printf("Liczba linii: %ld\n", total);
        printf("INFO:   %ld\n", count_info);
        printf("WARN:   %ld\n", count_warn);
        printf("ERROR:  %ld\n\n", count_error);

        printf("Statystyki godzinowe:\n");
        printf("Godz   INFO      WARN      ERROR\n");
        for (int h=0; h<24; h++)
            printf("%02d   %8ld  %8ld  %8ld\n",
                   h, hourly_info[h], hourly_warn[h], hourly_error[h]);

        printf("\nTop-%d słów:\n", TOP_N);
        for (int i=0; i < (size<TOP_N?size:TOP_N); i++)
            printf("%2d. %-20s %8ld\n", i+1, words[i].word, words[i].count);

        FILE *out = fopen("results/results_seq.txt","w");
        if (out) {
            fprintf(out,"Pełna analiza pliku %s\n", filename);
            fprintf(out,"INFO=%ld WARN=%ld ERROR=%ld\n",count_info,count_warn,count_error);
            fclose(out);
        }

        for (int i=0;i<size;i++) free(words[i].word);
        free(words);
        fclose(f);
        return 0;
    }

    /* ---------- FILTER MODE ---------- */

    char outname[256];
    sprintf(outname, "results/filtered_%s.txt", filter_name(filt));
    FILE *fo = fopen(outname, "w");

    if (!fo) { perror("write"); fclose(f); return 1; }

    long filtered = 0;
    long hourly[24]={0};

    int cap = INITIAL_WORD_CAP, size = 0;
    WordCount *words = malloc(cap * sizeof(WordCount));

    char line[LINE_SIZE];

    while (fgets(line, LINE_SIZE, f)) {

        int match =
            (filt==FILTER_INFO  && strstr(line,"INFO")) ||
            (filt==FILTER_WARN  && strstr(line,"WARN")) ||
            (filt==FILTER_ERROR && strstr(line,"ERROR"));

        if (!match) continue;

        fputs(line, fo);
        filtered++;

        int hour = parse_hour(line);
        if (hour>=0) hourly[hour]++;

        tokenize_message(line, &words, &size, &cap);
    }

    qsort(words, size, sizeof(WordCount), cmp_words);

    printf("=== FILTR: %s ===\n", filter_name(filt));
    printf("Plik: %s\n", filename);
    printf("Liczba dopasowanych linii: %ld\n", filtered);
    printf("Wynik zapisano w: %s\n\n", outname);

    printf("Statystyki godzinowe (%s):\n", filter_name(filt));
    for (int h=0; h<24; h++)
        printf("%02d   %8ld\n", h, hourly[h]);

    printf("\nTop-%d słów dla %s:\n", TOP_N, filter_name(filt));
    for (int i=0; i < (size<TOP_N?size:TOP_N); i++)
        printf("%2d. %-20s %8ld\n", i+1, words[i].word, words[i].count);

    fclose(fo);
    fclose(f);
    for (int i=0;i<size;i++) free(words[i].word);
    free(words);

    return 0;
}
