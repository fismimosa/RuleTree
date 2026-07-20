import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ===== SETTINGS =====
_path = "conclusi/sintetici/B20 2048 features"
_name= "B_20_benchmark_results_20260704-202247.csv"
FILE_PATH = f"{_path}/{_name}"
METRICS = ["fit_time", "peak_memory_mb", "f1_score"]
VERSIONS = ["Base", "MatrixV1", "MatrixV2"] # ["Base", "MatrixAB", "MatrixV2"]
DEPTHS=[1, 2, 3, 5, 8, 10]

# ===== LOAD =====
df_all = pd.read_csv(FILE_PATH)

for DEPTH in DEPTHS:
    for METRIC in METRICS:
        # Filtra solo la profondità desiderata
        # Filtra le versioni
        df = df_all[
            (df_all["max_depth"] == DEPTH) &
            (df_all["version"].isin(VERSIONS))
            ].copy()

        # ===== GROUP =====
        grouped = df.groupby(["n_features", "version"])[METRIC]

        mean = grouped.mean().reset_index()
        std = grouped.std().reset_index()

        # ===== PLOT =====
        plt.figure(figsize=(10, 7))

        scale = 1 if METRIC == 'fit_time' else 1

        stats = (
            df.groupby(["n_features", "version"])[METRIC]
            .agg(["mean", "std"])
            .fillna(0)
            .reset_index()
        )

        for version in stats["version"].unique():
            m = (
                stats[stats["version"] == version]
                .sort_values("n_features")
            )

            plt.plot(
                m["n_features"],
                m["mean"] / scale,
                marker="o",
                label=version
            )

            plt.fill_between(
                m["n_features"],
                (m["mean"] - m["std"]) / scale,
                (m["mean"] + m["std"]) / scale,
                alpha=0.2
            )
        # ===== FORMAT =====
        plt.xlabel("Number of Features")
        plt.ylabel(METRIC + " (min)" if METRIC == 'fit_time' else METRIC)

        title_parts = [
            f"{df['n_records'].iloc[0]} records",
            f"{df['run'].max()} runs",
            f"max_depth={DEPTH}"
        ]

        if "batch_size" in df.columns and pd.notna(df["batch_size"].iloc[0]):
            title_parts.append(f"batch_size={df['batch_size'].iloc[0]}")

        if "n_bins" in df.columns and pd.notna(df["n_bins"].iloc[0]):
            title_parts.append(f"#bins={df['n_bins'].iloc[0]}")

        title = "(" + ", ".join(map(str, title_parts)) + ")"

        plt.title(f"{METRIC} vs #features {title}")
        plt.legend()
        plt.grid(True)

        plt.xscale("log")
        plt.yscale("log")

        features = sorted(df["n_features"].unique())

        plt.xticks(features)
        plt.gca().set_xticklabels(
            features,
            rotation=45,
            ha='right',
            fontsize=8
        )

        plt.minorticks_off()

        #plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(20))

        y_min = (mean[METRIC] - std[METRIC]).min() / scale
        y_max = (mean[METRIC] + std[METRIC]).max() / scale

        plt.ylim(y_min * 0.95, y_max * 1.05)
        plt.tight_layout()
        plt.savefig(f"{_path}/plot_" + str(DEPTH) + "_" + METRIC + "_" + FILE_PATH.split("/")[len(FILE_PATH.split("/")) - 1] + ".png")
        plt.close()