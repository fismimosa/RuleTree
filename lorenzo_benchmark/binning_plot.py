import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ===== SETTINGS =====
_path = "conclusi/sintetici/BINNING B20 2048 features"
_name = "BIN_B_[20]_benchmark_results_20260813-100920.csv"
FILE_PATH = f"{_path}/{_name}"

METRICS = ["fit_time", "peak_memory_mb", "f1_score"]
VERSIONS = ["Matrix AB", "Flat Matrix"]

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
        grouped = df.groupby(["n_bins", "version"])[METRIC]

        mean = grouped.mean().reset_index()
        std = grouped.std().reset_index()

        # ===== PLOT =====
        plt.figure(figsize=(10, 7))

        scale = 1 if METRIC == "fit_time" else 1

        stats = (
            df.groupby(["n_bins", "version"])[METRIC]
            .agg(["mean", "std"])
            .fillna(0)
            .reset_index()
        )

        for version in stats["version"].unique():

            m = (
                stats[stats["version"] == version]
                .sort_values("n_bins")
            )

            plt.plot(
                m["n_bins"],
                m["mean"] / scale,
                marker="o",
                label=version
            )

            plt.fill_between(
                m["n_bins"],
                (m["mean"] - m["std"]) / scale,
                (m["mean"] + m["std"]) / scale,
                alpha=0.2
            )

        # ===== FORMAT =====
        plt.xlabel("n_bins")
        plt.ylabel(
            METRIC + " (sec)" if METRIC == "fit_time" else METRIC
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


        title = "(" + ", ".join(map(str, title_parts)) + ")"

        plt.title(f"{METRIC} vs depth {title}")

        plt.legend()
        plt.grid(True)

        plt.xscale("log")
        plt.yscale("log")

        # niente log: depth è discreta
        depths = sorted(df["n_bins"].unique())

        plt.xticks(depths)
        plt.gca().set_xticklabels(
            depths,
            rotation=45,
            ha='right',
            fontsize=8
        )

        ax = plt.gca()

        # ===== X AXIS =====
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.get_major_formatter().set_scientific(False)
        ax.xaxis.get_major_formatter().set_useOffset(False)

        # ===== Y AXIS =====
        if METRIC == "f1_score":
            # F1 score: scala lineare, più leggibile
            ax.set_yscale("linear")
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        else:
            ax.set_yscale("log")

            # Major ticks: 1, 10, 100, 1000, ...
            ax.yaxis.set_major_locator(
                ticker.LogLocator(base=10, subs=(1.0,))
            )

            # Minor ticks: 2, 4, 6, 8 × 10^n
            ax.yaxis.set_minor_locator(
                ticker.LogLocator(base=10, subs=(2, 5))
            )

            # Etichette dei major tick
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: f"{x:g}")
            )

            ax.yaxis.set_minor_formatter(
                ticker.FuncFormatter(lambda x, pos: f"{x:g}")
            )

        # ===== GRID =====
        ax.grid(True, which="major", axis="both", linestyle="-", alpha=0.5)
        ax.grid(True, which="minor", axis="y", linestyle="--", alpha=0.25)

        y_min = (mean[METRIC] - std[METRIC]).min() / scale
        y_max = (mean[METRIC] + std[METRIC]).max() / scale

        plt.ylim(y_min * 0.95, y_max * 1.05)

        plt.tight_layout()

        plt.savefig(
            f"{_path}/plot_features_{FEATURES_NUM}_{METRIC}_depth_"
            f"{_name}.png"
        )

        plt.close()