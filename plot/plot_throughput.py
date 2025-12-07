import pandas as pd
import matplotlib.pyplot as plt
import os

# Ścieżka do benchmarków
CSV_PATH = os.path.join("results", "benchmarks.csv")

# Rozmiar pliku wejściowego (w bajtach)
FILE_SIZE_BYTES = 48595595
FILE_SIZE_GB = FILE_SIZE_BYTES / (1024 ** 3)

# Wczytanie CSV
bench = pd.read_csv(CSV_PATH)

# Obliczenie przepustowości
bench["throughput_gbps"] = FILE_SIZE_GB / bench["time_seconds"]

print("\n=== Throughput (GB/s) ===")
print(bench[["method", "workers", "throughput_gbps"]])

# Rysowanie wykresu
plt.figure(figsize=(10, 6))
plt.plot(
    bench["workers"],
    bench["throughput_gbps"],
    marker="o",
    linestyle="-",
    linewidth=2,
    markersize=7,
)

plt.xlabel("Workers (threads / processes / GPU)")
plt.ylabel("Throughput (GB/s)")
plt.title("Przepustowość przetwarzania logów (GB/s)")
plt.grid(True)

# Zapis wykresu
OUT_PATH = os.path.join("results", "benchmarks_throughput.png")
plt.savefig(OUT_PATH, dpi=150)

print(f"\nWykres zapisany jako: {OUT_PATH}")
