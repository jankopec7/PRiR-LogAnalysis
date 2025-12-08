# Analiza logów systemowych – Projekt z przedmiotu Programowanie Równoległe i Rozproszone

## Opis projektu
Projekt dotyczy równoległego przetwarzania dużych plików tekstowych, takich jak dzienniki systemowe, raporty serwerowe czy dane telemetryczne.

Celem programu jest:
- Zliczanie wystąpień poziomów logów: **INFO**, **WARN**, **ERROR**
- Wyodrębnianie wierszy spełniających określone kryteria
- Tworzenie statystyk godzinowych
- Analiza najczęściej występujących słów (Top-N)
- Porównanie wydajności różnych paradygmatów programowania równoległego

Wszystkie implementacje korzystają z jednego pliku wejściowego:
**data/hadoop_all_logs.txt**

---

## Zakres implementacji

### **1. Wersja sekwencyjna (SEQ)**
Podstawowa implementacja wykorzystywana jako punkt odniesienia do pomiarów wydajności.

### **2. OpenMP**
- Przetwarzanie linii w wielu wątkach
- Lokalny słownik słów tworzony przez każdy wątek
- Redukcja wyników do struktur globalnych
- Statystyki godzinowe oraz Top-N słów

### **3. MPI**
- Podział pliku na fragmenty między procesy
- Każdy proces analizuje swój zakres danych
- Proces master (rank 0) odbiera wyniki i wykonuje scalanie
- Analogiczny model do OpenMP, ale z komunikacją między procesami

### **4. CUDA**
- Równoległe zliczanie poziomów INFO/WARN/ERROR na GPU
- Wysoka przepustowość przetwarzania danych
- Wersja rozszerzona o statystyki godzinowe (opcjonalnie)

---

## Kompilacja projektu

Projekt wykorzystuje Makefile umożliwiający szybkie kompilowanie wszystkich modułów.

### **Kompilacja wszystkich wersji:**
```bash
make all
```

### **Kompilacja jednej konkretnej wersji:**

```bash
make seq
make openmp
make mpi
make cuda
```

### **Czyszczenie wyników i binarek:**

```bash
make clean
```

---

## Uruchomienie programu

### **Wersja sekwencyjna:**

```bash
./bin/log_seq data/hadoop_all_logs.txt
```

### **OpenMP (z określoną liczbą wątków):**

```bash
export OMP_NUM_THREADS=8
./bin/log_openmp data/hadoop_all_logs.txt
```

### **MPI – przykładowo 4 procesy:**

```bash
mpirun -np 4 ./bin/log_mpi data/hadoop_all_logs.txt
```

### **MPI z oversubscribe (np. 12 procesów):**

```bash
mpirun --oversubscribe -np 12 ./bin/log_mpi data/hadoop_all_logs.txt
```

### **CUDA:**

```bash
./bin/log_cuda data/hadoop_all_logs.txt
```

---

## Testy automatyczne

W katalogu **tests/** znajdują się skrypty testowe:

- `run_seq.sh` — pomiar czasu wersji sekwencyjnej  
- `run_openmp.sh` — testy dla różnych wartości `OMP_NUM_THREADS`  
- `run_mpi.sh` — testy dla różnych konfiguracji MPI  
- `run_cuda.sh` — pomiar czasu wykonania GPU  
- `run_all.sh` — pełen benchmark uruchamiający wszystkie powyższe testy  

### **Uruchomienie pełnego zestawu benchmarków:**

```bash
./tests/run_all.sh
```

Wyniki zapisywane są automatycznie w:
- `results/benchmarks.csv`  
- `results/outputs/`  

---

## Źródło danych

Dane logów pochodzą z oficjalnego repozytorium **LogHub**:  
https://github.com/logpai/loghub

Zbiór Hadoop został opisany w publikacjach naukowych:

- Qingwei Lin, Hongyu Zhang, Jian-Guang Lou, Yu Zhang, Xuewei Chen.  
  *Log Clustering Based Problem Identification for Online Service Systems*.  
  ICSE 2016.

- Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu.  
  *Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics*.  
  ISSRE 2023.

---
