import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ===== SETTINGS =====
_path = "conclusi/sintetici/B20 2048 features"
_name = "B_20_benchmark_results_20260704-202247.csv"
FILE_PATH = f"{_path}/{_name}"

METRICS = ["fit_time", "peak_memory_mb", "f1_score"]
VERSIONS = ["Base", "MatrixV1", "MatrixV2"]

FEATURES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]   # numero di features da fissare

# ===== LOAD =====
df_all = pd.read_csv(FILE_PATH)


for FEATURES_NUM in FEATURES:
    for METRIC in METRICS:

        # Fisso il numero di features e filtro le versioni
        df = df_all[
            (df_all["n_features"] == FEATURES_NUM) &
            (df_all["version"].isin(VERSIONS))
        ].copy()

        # ===== GROUP =====
        grouped = df.groupby(["max_depth", "version"])[METRIC]

        mean = grouped.mean().reset_index()
        std = grouped.std().reset_index()

        # ===== PLOT =====
        plt.figure(figsize=(10, 7))

        scale = 60 if METRIC == "fit_time" else 1

        stats = (
            df.groupby(["max_depth", "version"])[METRIC]
            .agg(["mean", "std"])
            .fillna(0)
            .reset_index()
        )

        for version in stats["version"].unique():

            m = (
                stats[stats["version"] == version]
                .sort_values("max_depth")
            )

            plt.plot(
                m["max_depth"],
                m["mean"] / scale,
                marker="o",
                label=version
            )

            plt.fill_between(
                m["max_depth"],
                (m["mean"] - m["std"]) / scale,
                (m["mean"] + m["std"]) / scale,
                alpha=0.2
            )

        # ===== FORMAT =====
        plt.xlabel("Maximum Depth")
        plt.ylabel(
            METRIC + " (min)" if METRIC == "fit_time" else METRIC
        )

        title_parts = [
            f"{df['n_records'].iloc[0]} records",
            f"{df['run'].max()} runs",
            f"n_features={FEATURES_NUM}"
        ]

        if "batch_size" in df.columns and pd.notna(df["batch_size"].iloc[0]):
            title_parts.append(
                f"batch_size={df['batch_size'].iloc[0]}"
            )

        if "n_bins" in df.columns and pd.notna(df["n_bins"].iloc[0]):
            title_parts.append(
                f"#bins={df['n_bins'].iloc[0]}"
            )

        title = "(" + ", ".join(map(str, title_parts)) + ")"

        plt.title(f"{METRIC} vs depth {title}")

        plt.legend()
        plt.grid(True)

        # niente log: depth è discreta
        depths = sorted(df["max_depth"].unique())

        plt.xticks(depths)

        plt.minorticks_off()

        plt.gca().yaxis.set_major_locator(
            ticker.MaxNLocator(20)
        )

        y_min = (mean[METRIC] - std[METRIC]).min() / scale
        y_max = (mean[METRIC] + std[METRIC]).max() / scale

        plt.ylim(y_min * 0.95, y_max * 1.05)

        plt.tight_layout()

        plt.savefig(
            f"{_path}/plot_features_{FEATURES_NUM}_{METRIC}_depth_"
            f"{_name}.png"
        )

        plt.close()