#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
CSV = os.path.join(RESULTS_DIR, "benchmarks.csv")

df = pd.read_csv(CSV)

# Pobieramy SEQ jako baseline
t_seq = float(df[(df["method"] == "SEQ")]["time_seconds"])

# --- 1) WYKRES CZASÓW ---
plt.figure(figsize=(10, 6))
for method in ["SEQ", "OpenMP", "MPI", "CUDA"]:
    sub = df[df["method"] == method]
    plt.plot(sub["workers"], sub["time_seconds"],
             marker="o", label=method)

plt.title("Czas wykonania – SEQ vs OpenMP vs MPI vs CUDA")
plt.xlabel("Liczba wątków / procesów / GPU")
plt.ylabel("Czas [s]")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "benchmarks_time.png"), dpi=200)
plt.close()

# --- 2) WYKRES PRZYSPIESZENIA (SPEEDUP) ---
df["speedup"] = t_seq / df["time_seconds"]

plt.figure(figsize=(10, 6))
for method in ["OpenMP", "MPI", "CUDA"]:
    sub = df[df["method"] == method]
    plt.plot(sub["workers"], sub["speedup"],
             marker="o", label=method)

plt.axhline(1.0, color="gray", linestyle="--")
plt.title("Przyspieszenie (Speedup) względem wersji sekwencyjnej")
plt.xlabel("Liczba wątków / procesów / GPU")
plt.ylabel("Speedup = T_seq / T_parallel")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "benchmarks_speedup.png"), dpi=200)
plt.close()

# --- 3) WYKRES EFEKTYWNOŚCI ---
df["efficiency"] = df["speedup"] / df["workers"]

plt.figure(figsize=(10, 6))
for method in ["OpenMP", "MPI"]:
    sub = df[df["method"] == method]
    plt.plot(sub["workers"], sub["efficiency"],
             marker="o", label=method)

plt.title("Efektywność równoległa (Efficiency)")
plt.xlabel("Liczba wątków / procesów")
plt.ylabel("Efficiency = Speedup / Workers")
plt.ylim(0, 1.1)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "benchmarks_efficiency.png"), dpi=200)
plt.close()

print("=== Wykresy zapisane do results/ ===")
print(" - benchmarks_time.png")
print(" - benchmarks_speedup.png")
print(" - benchmarks_efficiency.png")
