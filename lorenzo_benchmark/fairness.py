import os
import time
import tracemalloc

import numpy as np
import pandas as pd
from fairlearn.metrics import (demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    equal_opportunity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio
)

from sklearn.model_selection import train_test_split

from RuleTree.stumps.classification.DecisionTreeStumpClassifierBase import \
    DecisionTreeStumpClassifierBase

from RuleTree.stumps.classification.matrix.DecisionTreeStumpClassifierFlatMatrix import \
    DecisionTreeStumpClassifierFlatMatrix

from RuleTree.tree.RuleTreeClassifier import RuleTreeClassifier
from benchmark.Hybrid.HybridReaders import read_titanic

from benchmark.evaluation_utils import evaluate_clf, evaluate_expl


# ==================================================
# SETTINGS
# ==================================================

DATASET_PATH = "../datasets/CLF/titanic.csv"
OUTPUT_FOLDER = "fairness_benchmark_results"

N_RUNS = 3

DEPTH = 3
BATCH_SIZE = 64
N_BINS = 128

PROTECTED_ATTRIBUTE_IDX = 3

# Valori di fairness da testare
FAIRNESS_WEIGHTS = [
    0.0,
    0.1,
    0.3,
    0.5,
    0.7,
    1.0,
    1.5,
    2.0,
    5.0,
    10.0,
    20.0
]

#[
#    0.0,
#    0.1,
#    0.2,
#    0.3,
#    0.5,
#    0.7,
#    1.0,
#    1.5,
#    2.0
#]


# ==================================================
# CHECK DATASET
# ==================================================

if not os.path.exists(DATASET_PATH):
    print(f"Error: {DATASET_PATH} not found.")
    exit(1)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

output_path = os.path.join(
    OUTPUT_FOLDER,
    "titanic_benchmark.csv"
)

# ==================================================
# LOAD DATASET
# ==================================================

print("\n==========================================")
print("===== FAIRNESS BENCHMARK =====")
print("==========================================")

print(f"\nDataset: {DATASET_PATH}")

dataset_name, df = read_titanic()
print(df.columns)
print(df.columns[PROTECTED_ATTRIBUTE_IDX])
X = df.iloc[:, :-1].copy().to_numpy()
y = df.iloc[:, -1].values

n_records = len(X)
n_features_original = X.shape[1]

print(f"Records: {n_records}")
print(f"Original features: {n_features_original}")


# ==================================================
# TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# ==================================================
# DATASET DIMENSIONS
# ==================================================

n_features = X_train.shape[1]

print(f"Final features: {n_features}")
print(
    f"Protected attribute index: "
    f"{PROTECTED_ATTRIBUTE_IDX}"
)

print(f"Depth: {DEPTH}")
print(f"Batch size: {BATCH_SIZE}")
print(f"N bins: {N_BINS}")

print(
    f"Fairness weights: "
    f"{FAIRNESS_WEIGHTS}"
)


# ==================================================
# RESULTS
# ==================================================

results = []


# ==================================================
# BENCHMARK
# ==================================================

for fairness_weight in FAIRNESS_WEIGHTS:

    print(
        "\n=========================================="
    )

    print(
        f"FAIRNESS WEIGHT = {fairness_weight}"
    )

    print(
        "=========================================="
    )

    models = [
        (
            "Flat Matrix",
            lambda fw=fairness_weight:
            RuleTreeClassifier(
                max_depth=DEPTH,
                base_stumps=DecisionTreeStumpClassifierFlatMatrix(
                    random_state=42,
                    batch_size=BATCH_SIZE,
                    n_bins=N_BINS,
                    protected_attribute_idx=PROTECTED_ATTRIBUTE_IDX,
                    fairness_weight=fw
                ),
                random_state=42
            )
        ),
        (
            "Base",
            lambda fw=fairness_weight:
                RuleTreeClassifier(
                    max_depth=DEPTH,
                    base_stumps=DecisionTreeStumpClassifierBase(
                        random_state=42,
                        protected_attribute_idx=PROTECTED_ATTRIBUTE_IDX,
                        fairness_weight=fw
                    ),
                    random_state=42
                )
        ),


    ]


    for run in range(1, N_RUNS + 1):

        for model_name, model_factory in models:

            print(
                f"Run {run}/{N_RUNS} - "
                f"{model_name} - "
                f"fairness={fairness_weight}"
            )

            clf = model_factory()


            # ==================================================
            # FIT TIME + MEMORY
            # ==================================================

            tracemalloc.start()

            start_time = time.time()

            try:

                clf.fit(
                    X_train,
                    y_train
                )

            except Exception as e:

                tracemalloc.stop()
                raise e

            end_time = time.time()

            current, peak = (
                tracemalloc.get_traced_memory()
            )

            tracemalloc.stop()

            fit_time = end_time - start_time

            peak_memory_mb = (
                peak / (1024 * 1024)
            )


            # ==================================================
            # PREDICTIONS
            # ==================================================

            y_pred = clf.predict(X_test)

            y_pred_proba = clf.predict_proba(X_test)

            print("y_test shape:", y_test.shape)
            print("y_test classes:", np.unique(y_test))
            print("y_pred_proba shape:", np.asarray(y_pred_proba).shape)
            print("y_pred_proba:", y_pred_proba[:5])


            # ==================================================
            # EVALUATION
            # ==================================================

            metrics = (
                    evaluate_clf(
                        y_test,
                        y_pred,
                        y_pred_proba
                    )
                    |
                    evaluate_expl(clf)
            )

            sensitive_test = X_test[:, PROTECTED_ATTRIBUTE_IDX]

            metrics["demographic_parity_difference"] = demographic_parity_difference(

                y_test, y_pred, sensitive_features=sensitive_test

            )

            metrics["demographic_parity_ratio"] = demographic_parity_ratio(

                y_test, y_pred, sensitive_features=sensitive_test

            )

            metrics["equal_opportunity_difference"] = equal_opportunity_difference(

                y_test, y_pred, sensitive_features=sensitive_test

            )

            metrics["equal_opportunity_ratio"] = equal_opportunity_ratio(

                y_test, y_pred, sensitive_features=sensitive_test

            )

            metrics["equalized_odds_difference"] = equalized_odds_difference(

                y_test, y_pred, sensitive_features=sensitive_test

            )

            metrics["equalized_odds_ratio"] = equalized_odds_ratio(

                y_test, y_pred, sensitive_features=sensitive_test

            )


            # ==================================================
            # RESULTS
            # ==================================================

            row = {

                "run": run,

                "version": model_name,

                "dataset": "titanic.csv",

                "fairness_weight": fairness_weight,

                "depth": DEPTH,

                "batch_size": (
                    BATCH_SIZE
                    if model_name == "Flat Matrix"
                    else None
                ),

                "n_bins": (
                    N_BINS
                    if model_name == "Flat Matrix"
                    else None
                ),

                "protected_attribute_idx":
                    PROTECTED_ATTRIBUTE_IDX,

                "fit_time": fit_time,

                "peak_memory_mb":
                    peak_memory_mb,

                "n_records":
                    n_records,

                "n_features_original":
                    n_features_original,

                "n_features":
                    n_features
            }

            row.update(metrics)

            results.append(row)


            # ==================================================
            # SAVE RESULTS
            # ==================================================

            df_results = pd.DataFrame(results)



            df_results.to_csv(
                output_path,
                index=False,
                float_format="%.5f"
            )


# ==================================================
# SUMMARY
# ==================================================

print(
    "\n=========================================="
)

print(
    "===== BENCHMARK COMPLETED ====="
)

print(
    "=========================================="
)

print(
    f"\nResults saved to:"
)

print(
    output_path
)

print(
    f"\nTotal results: "
    f"{len(df_results)}"
)

print(
    "\nSummary:"
)

summary = (
    df_results
    .groupby(
        ["fairness_weight", "version"]
    )
    .agg(
        fit_time_mean=("fit_time", "mean"),
        fit_time_std=("fit_time", "std"),
        memory_mean=("peak_memory_mb", "mean"),
        accuracy_mean=("accuracy", "mean")
    )
    .reset_index()
)

print(
    summary.to_string(index=False)
)