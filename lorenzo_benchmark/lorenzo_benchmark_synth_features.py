import time
import pandas as pd
import tracemalloc

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import datetime

from RuleTree.stumps.classification.lorenzo.DecisionTreeStumpClassifierBase import \
    DecisionTreeStumpClassifierBase
# Imports for the models
from RuleTree.tree.RuleTreeClassifier import RuleTreeClassifier
from RuleTree.stumps.classification.lorenzo.matrix.DecisionTreeStumpClassifierMatrixV1 import \
    DecisionTreeStumpClassifierMatrixV1
from RuleTree.stumps.classification.lorenzo.matrix.DecisionTreeStumpClassifierMatrixV2 import \
    DecisionTreeStumpClassifierMatrixV2
from benchmark.evaluation_utils import evaluate_clf, evaluate_expl

# Settings
NUM_RECORDS = 10000
NUM_FEATURES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]  # [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
INFORMATIVE_PERC = 0.8
REDUNDANT_PERC = 0.05
REPEATED_PERC = 0.05
N_RUNS = 5
BATCH_SIZE = 20
N_BINS= None
MAX_DEPTH = [1, 2, 3, 5, 8, 10]
STUMP_VERSIONS = [
    ("Base", DecisionTreeStumpClassifierBase),
    ("MatrixV1", DecisionTreeStumpClassifierMatrixV1),
    ("MatrixV2", DecisionTreeStumpClassifierMatrixV2)
]
FILE_NAME = "benchmark_results_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"

if BATCH_SIZE is not None:
    FILE_NAME = "B_" + str(BATCH_SIZE) + "_" + FILE_NAME

if N_BINS is not None:
    FILE_NAME = "BI_" + str(N_BINS) + "_" + FILE_NAME

results = []

# TODO: fare grafico dove sulle x c'è la variazione della batch size e sulla y tempo

try:
    for j in range(0, len(MAX_DEPTH)):
        for i in range(0, len(NUM_FEATURES)):
            print(f"--- Dataset with {NUM_FEATURES[i]} features, max_depth={MAX_DEPTH[j]} ---")

            # -------- RUN LOOP --------
            for run in range(1, N_RUNS + 1):
                X, y = make_classification(
                    n_samples=NUM_RECORDS,
                    n_features=NUM_FEATURES[i],
                    n_informative=round(INFORMATIVE_PERC * NUM_FEATURES[i]),
                    n_redundant=round(REDUNDANT_PERC * NUM_FEATURES[i]),
                    n_repeated=round(REPEATED_PERC * NUM_FEATURES[i]),
                    n_classes=2,
                    random_state=run
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
                for version_name, stump_class in STUMP_VERSIONS:
                    print(f"Run {run}/{N_RUNS} - {version_name}")

                    if version_name == "Base":
                        clf = RuleTreeClassifier(
                            max_depth=MAX_DEPTH[j],
                            base_stumps=stump_class(random_state=run),
                            random_state=run
                        )
                    else:
                        clf = RuleTreeClassifier(
                            max_depth=MAX_DEPTH[j],
                            base_stumps=stump_class(random_state=run, batch_size=BATCH_SIZE, n_bins=N_BINS),
                            random_state=run
                        )

                    # Measure memory and time for fit
                    tracemalloc.start()
                    start_time = time.time()
                    try:
                        clf.fit(X_train, y_train)
                    except Exception as e:
                        tracemalloc.stop()
                        raise e
                    end_time = time.time()
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    fit_time = end_time - start_time
                    peak_memory_mb = peak / (1024 * 1024)

                    # Predictions for metrics
                    y_pred = clf.predict(X_test)
                    y_pred_proba = clf.predict_proba(X_test)

                    # Evaluate metrics
                    metrics = evaluate_clf(y_test, y_pred, y_pred_proba) | evaluate_expl(clf)

                    # Build row
                    row = {
                        "n_features": NUM_FEATURES[i],
                        "batch_size": BATCH_SIZE,
                        "n_bins": N_BINS,
                        "n_records": NUM_RECORDS,
                        "run": run,
                        "version": version_name,
                        "fit_time": fit_time,
                        "peak_memory_mb": peak_memory_mb
                    }
                    row.update(metrics)

                    results.append(row)

except Exception as e:
    exit(1)
except KeyboardInterrupt:
    # Creare .csv
    df_results = pd.DataFrame(results)
    df_results.to_csv("synth_benchmark_results/" + FILE_NAME, index=False, float_format="%.5f")
    print("\n===== BENCHMARK SAVED to " + FILE_NAME + " =====")

# Creare .csv
df_results = pd.DataFrame(results)
df_results.to_csv("synth_benchmark_results/" + FILE_NAME, index=False, float_format="%.5f")
print("\n===== BENCHMARK SAVED to " + FILE_NAME + " =====")
