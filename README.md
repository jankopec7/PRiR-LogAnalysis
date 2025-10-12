# Analiza logów systemowych - Projekt z przedmiotu Programowanie Równoległe i Rozproszone

## Opis projektu
Projekt dotyczy równoległego przetwarzania dużych plików tekstowych, np. dzienników systemowych, raportów z serwerów, wyników pomiarów czy danych z czujników IoT.  

Celem programu jest analiza zawartości plików w celu:  
- Zliczenia częstości występowania określonych słów lub fraz (np. „ERROR”, „WARNING”, „INFO”)  
- Wyodrębnienia wierszy spełniających określone kryteria (np. tylko błędy lub tylko określony zakres dat)  
- Stworzenia statystyk dotyczących zdarzeń w czasie (np. liczba błędów na godzinę)  

## Zakres implementacji
- **OpenMP** – równoległa tokenizacja i lokalne słowniki; końcowa redukcja  
- **MPI** – podział plików między procesy, łączenie wyników  
- **CUDA/OpenCL** – przyspieszenie zliczania/histogramów na GPU  

## Dodatkowe wyniki do raportu
- Wykres top‑N słów  
- Przepustowość GB/s  
- Porównanie CPU/GPU  
- Krótkie wnioski o wąskich gardłach I/O
