import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ===== SETTINGS =====
FILE_PATH = "synth_benchmark_results/B_20_benchmark_results_20260510-100954.csv"
METRIC = "fit_time"

# ===== LOAD =====
df = pd.read_csv(FILE_PATH)

# ===== GROUP =====
grouped = df.groupby(["n_features", "version"])[METRIC]

mean = grouped.mean().reset_index()
std = grouped.std().reset_index()

# ===== PLOT =====
plt.figure(figsize=(10, 7))

scale = 60 if METRIC=='fit_time' else 1

for version in df["version"].unique():
    m = mean[mean["version"] == version]
    s = std[std["version"] == version]

    m = m.sort_values("n_features")
    s = s.sort_values("n_features")

    plt.plot(
        m["n_features"],
        m[METRIC] / scale,
        marker='o',
        label=version
    )

    plt.fill_between(
        m["n_features"],
        (m[METRIC] - s[METRIC]) / scale,
        (m[METRIC] + s[METRIC]) / scale,
        alpha=0.2
    )

# ===== FORMAT =====
plt.xlabel("Number of Features")
plt.ylabel(METRIC + " (min)" if METRIC=='fit_time' else METRIC)
title = f"({df['n_records'][0]} records, {max(df['run'])} runs"
if df["batch_size"][0] != 0:
    title += f", batch_size={df['batch_size'][0]}"
plt.title(f"{METRIC} vs #features {title})")
plt.legend()
plt.grid(True)

plt.xscale("log")

features = sorted(df["n_features"].unique())

plt.xticks(features)
plt.gca().set_xticklabels(
    features,
    rotation=45,
    ha='right',
    fontsize=8
)

plt.minorticks_off()

plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(20))

y_min = (mean[METRIC] - std[METRIC]).min() / scale
y_max = (mean[METRIC] + std[METRIC]).max() / scale

plt.ylim(y_min * 0.95, y_max * 1.05)
plt.tight_layout()
plt.savefig("benchmark_plots/plot_" + METRIC + "_" + FILE_PATH.split("/")[1] + ".png")
plt.show()
