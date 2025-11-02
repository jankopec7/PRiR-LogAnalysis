#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DEFAULT_LINES 1000000  
#define LOG_FILE "data/sample_logs.txt"

const char *statuses[] = {"INFO", "WARNING", "ERROR"};
const char *methods[] = {"GET", "POST", "PUT", "DELETE"};
const char *resources[] = {"/index.html", "/login", "/admin", "/api/data", "/home", "/dashboard"};

int main(int argc, char *argv[]) {
    long num_lines = DEFAULT_LINES;
    if (argc > 1) {
        num_lines = atol(argv[1]);
        if (num_lines <= 0) num_lines = DEFAULT_LINES;
    }

    FILE *f = fopen(LOG_FILE, "w");
    if (!f) {
        perror("Nie można otworzyć pliku do zapisu");
        return 1;
    }

    srand(time(NULL));

    for (long i = 0; i < num_lines; i++) {
        int hour = rand() % 24;
        int min = rand() % 60;
        int sec = rand() % 60;

        int status_idx = rand() % 3;
        int method_idx = rand() % 4;
        int resource_idx = rand() % 6;

        int response_code;
        switch (status_idx) {
            case 0: response_code = 200; break;  // INFO
            case 1: response_code = 401; break;  // WARNING
            case 2: response_code = 500; break;  // ERROR
        }

        int size = 100 + rand() % 1000;

        fprintf(f, "127.0.0.1 - - [02/Nov/2025:%02d:%02d:%02d +0000] \"%s %s HTTP/1.1\" %d %d \"-\" \"Mozilla/5.0\" %s\n",
                hour, min, sec, methods[method_idx], resources[resource_idx], response_code, size, statuses[status_idx]);
    }

    fclose(f);
    printf("Wygenerowano %ld linii logów w pliku %s\n", num_lines, LOG_FILE);
    return 0;
}
