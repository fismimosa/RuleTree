import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.ticker import LogLocator, FuncFormatter, ScalarFormatter

METRIC = "f1_score"

# ===== LOAD =====
files = glob.glob("real_benchmark_results/base/*.csv")

df = pd.concat(
    [
        pd.read_csv(f).assign(
            dataset=os.path.basename(f).replace(".csv", "").split("_")[2]
        )
        for f in files
    ],
    ignore_index=True
)

# ===== METADATA =====
meta = df.groupby("dataset")[["n_features", "n_records"]].first()
meta["complexity"] = meta["n_features"] * meta["n_records"]

# ordina dataset per complessità
datasets = meta.sort_values("complexity").index.tolist()

# ===== AGGREGATE =====
stats = df.groupby("dataset")[METRIC].mean()

x = np.arange(len(datasets))

plt.figure(figsize=(12, 6))

plt.bar(
    x,
    [stats[d] for d in datasets],
    color="#1f77b4"
)

# ===== LABELS =====
labels = [
    f"{d}\n(n={meta.loc[d, 'n_records']}, m={meta.loc[d, 'n_features']})"
    for d in datasets
]

plt.xticks(x, labels, rotation=45, ha="right")

plt.ylabel(METRIC)
plt.title(f"Base model performance across datasets (ordered by size)")
plt.grid(axis="y", alpha=0.3)

vals = np.array([stats[d] for d in datasets])

ax = plt.gca()

#plt.yscale("log")

# SOLO potenze di 10 (pulito ma non troppo scarno)
ax.yaxis.set_major_locator(ticker.LogLocator(base=2))

# niente tick secondari (fondamentale)
ax.yaxis.set_minor_locator(ticker.NullLocator())

# formato pulito
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.get_major_formatter().set_scientific(False)
ax.yaxis.get_major_formatter().set_useOffset(False)

plt.tight_layout()
plt.savefig("benchmark_plots/plot_" + METRIC + "_real.png")
plt.show()
