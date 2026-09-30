#!/usr/bin/env python3
"""
Week 6 Lab - Exercise 3 plot: GFLOP/s versus channel count.

    python plot_layer.py conv_layer.csv layer_plot.png

The data plumbing is done. The four TODOs are the four things the plot
rubric grades. Fill them in.

Deps: pip install matplotlib   (Colab already has it)
"""
import sys, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "conv_layer.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "layer_plot.png"

C, gf = [], []
with open(src, newline="") as f:
    for row in csv.DictReader(f):
        C.append(int(row["C"]))
        gf.append(float(row["gflop_s"]))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(C, gf, marker="o", markersize=8, linewidth=2,
        label="Measured throughput")

# --- TODO A: mark your GPU's peak FP32 as a horizontal reference line ---
PEAK_TFLOPS = 48.7          # <- your card, from its spec sheet
ax.axhline(PEAK_TFLOPS * 1000, linestyle=":", color="gray",
           label=f"Peak FP32 ({PEAK_TFLOPS} TFLOP/s)")

# --- TODO B: log scale base 2 on x, with ticks at the real C values -----
ax.set_xscale("log", base=2)
ax.set_xticks(C); ax.set_xticklabels([str(c) for c in C])

# --- TODO C: label BOTH axes with quantity AND unit ---------------------
ax.set_xlabel("Channels C (C_in = C_out)")
ax.set_ylabel("Achieved throughput (GFLOP/s)")

# --- TODO D: title naming the experiment AND your GPU, plus a legend ----
ax.set_title("CNN layer throughput vs channel depth, 512x512, 3x3 - NVIDIA GeForce RTX 4080")
ax.legend()

ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(dst, dpi=150)
print("wrote", dst)
