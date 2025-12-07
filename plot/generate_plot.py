#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
CSV = os.path.join(RESULTS_DIR, "benchmarks.csv")

df = pd.read_csv(CSV)

# ---------------------------------
# 0) Przepustowość – NAJPIERW!
# ---------------------------------
FILE_SIZE_BYTES = 48595595
FILE_SIZE_GB = FILE_SIZE_BYTES / (1024 ** 3)
df["throughput"] = FILE_SIZE_GB / df["time_seconds"]

# ---------------------------------
# 1) Podział na metody
# ---------------------------------
df_seq = df[df["method"] == "SEQ"]
df_omp = df[df["method"] == "OpenMP"]
df_mpi = df[df["method"] == "MPI"]
df_mpi_over = df[df["method"] == "MPI-oversub"]
df_cuda = df[df["method"] == "CUDA"]

t_seq = float(df_seq.iloc[0]["time_seconds"])

def speedup(sub):
    return t_seq / sub["time_seconds"]

# ---------------------------------
# 2) Czas wykonania
# ---------------------------------
plt.figure(figsize=(10, 6))
plt.plot(df_seq["workers"], df_seq["time_seconds"], "o-", label="SEQ")
plt.plot(df_omp["workers"], df_omp["time_seconds"], "o-", label="OpenMP")
plt.plot(df_mpi["workers"], df_mpi["time_seconds"], "o-", label="MPI")
plt.plot(df_mpi_over["workers"], df_mpi_over["time_seconds"], "o--", label="MPI oversub")
plt.plot(df_cuda["workers"], df_cuda["time_seconds"], "o", markersize=10, label="CUDA")

plt.title("Czas wykonania – SEQ vs OpenMP vs MPI vs CUDA")
plt.xlabel("Liczba wątków / procesów / GPU")
plt.ylabel("Czas [s]")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "benchmarks_time.png"), dpi=200)
plt.close()

# ---------------------------------
# 3) Speedup
# ---------------------------------
plt.figure(figsize=(10, 6))
plt.plot(df_omp["workers"], speedup(df_omp), "o-", label="OpenMP")
plt.plot(df_mpi["workers"], speedup(df_mpi), "o-", label="MPI")
plt.plot(df_mpi_over["workers"], speedup(df_mpi_over), "o--", label="MPI oversub")
plt.plot(df_cuda["workers"], speedup(df_cuda), "o", markersize=10, label="CUDA")

plt.axhline(1.0, linestyle="--", color="gray")
plt.title("Przyspieszenie (Speedup) względem wersji sekwencyjnej")
plt.xlabel("Workers")
plt.ylabel("Speedup = T_seq / T_parallel")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "benchmarks_speedup.png"), dpi=200)
plt.close()

# ---------------------------------
# 4) Efektywność
# ---------------------------------
plt.figure(figsize=(10, 6))
plt.plot(df_omp["workers"], speedup(df_omp) / df_omp["workers"], "o-", label="OpenMP")
plt.plot(df_mpi["workers"], speedup(df_mpi) / df_mpi["workers"], "o-", label="MPI")
plt.plot(df_mpi_over["workers"], speedup(df_mpi_over) / df_mpi_over["workers"],
         "o--", label="MPI oversub")

plt.title("Efektywność równoległa (Efficiency)")
plt.xlabel("Liczba wątków / procesów")
plt.ylabel("Efficiency = Speedup / Workers")
plt.ylim(0, 1.2)
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "benchmarks_efficiency.png"), dpi=200)
plt.close()

# ---------------------------------
# 5) Przepustowość
# ---------------------------------
plt.figure(figsize=(10, 6))
plt.plot(df_omp["workers"], df_omp["throughput"], "o-", label="OpenMP")
plt.plot(df_mpi["workers"], df_mpi["throughput"], "o-", label="MPI")
plt.plot(df_mpi_over["workers"], df_mpi_over["throughput"], "o--", label="MPI oversub")
plt.plot(df_cuda["workers"], df_cuda["throughput"], "o", markersize=10, label="CUDA")

plt.title("Przepustowość przetwarzania logów (GB/s)")
plt.xlabel("Workers (threads / processes / GPU)")
plt.ylabel("Throughput (GB/s)")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "benchmarks_throughput.png"), dpi=200)
plt.close()

print("Wykresy zapisane w:", RESULTS_DIR)
