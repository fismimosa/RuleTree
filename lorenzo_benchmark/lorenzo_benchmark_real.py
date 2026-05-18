import os
import time
import pandas as pd
import tracemalloc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import datetime
import matplotlib.pyplot as plt

# Imports for the models
from RuleTree.stumps.classification.lorenzo.DecisionTreeStumpClassifierLorenzoBase import \
    DecisionTreeStumpClassifierLorenzoBase
from RuleTree.tree.RuleTreeClassifier import RuleTreeClassifier
from RuleTree.stumps.classification.lorenzo.DecisionTreeStumpClassifierLorenzoMatrixv1 import \
    DecisionTreeStumpClassifierLorenzoMatrixv1
from RuleTree.stumps.classification.lorenzo.DecisionTreeStumpClassifierLorenzoMatrixv2 import \
    DecisionTreeStumpClassifierLorenzoMatrixv2

# Import for evaluation
from benchmark.evaluation_utils import evaluate_clf, evaluate_expl

DATASET_FOLDER = "../datasets/CLF"
MAX_ROWS = 10000
N_RUNS = 5
BATCH_SIZE = None

results = []
skipped = []

# List of versions to test
stump_versions = [
    ("Base", DecisionTreeStumpClassifierLorenzoBase),
    ("MatrixV1", DecisionTreeStumpClassifierLorenzoMatrixv1),
    ("MatrixV2", DecisionTreeStumpClassifierLorenzoMatrixv2)
]

# Ensure we are in the right directory or dataset folder path is correct
if not os.path.exists(DATASET_FOLDER):
    print(f"Error: {DATASET_FOLDER} not found.")
    exit(1)

for file in sorted(os.listdir(DATASET_FOLDER)):
    results = []
    if not file.endswith(".csv"):
        continue

    print(f"\nProcessing dataset: {file}")
    FILE_NAME = "benchmark_results_" + file
    if BATCH_SIZE is not None:
        FILE_NAME = "B_" + str(BATCH_SIZE) + "_" + FILE_NAME

    try:
        path = os.path.join(DATASET_FOLDER, file)
        df = pd.read_csv(path)

        if len(df) > MAX_ROWS:
            print(f"Skipping {file}: too large ({len(df)} rows)")
            skipped.append(file)
            continue

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        if len(X) == 0:
            raise ValueError("Empty dataset")

        # Basic label encoding if y is string
        if y.dtype == object or isinstance(y[0], str):
            le = LabelEncoder()
            y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # -------- RUN LOOP --------

        for run in range(1, N_RUNS + 1):
            for version_name, stump_class in stump_versions:
                print(f"Run {run}/{N_RUNS} - {version_name}")

                if version_name != "Base" and BATCH_SIZE is not None:
                    clf = RuleTreeClassifier(
                        max_depth=2,
                        base_stumps=stump_class(random_state=42, batch_size=BATCH_SIZE),
                        random_state=42
                    )
                else:
                    clf = RuleTreeClassifier(
                        max_depth=2,
                        base_stumps=stump_class(random_state=42),
                        random_state=42
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
                    "run": run,
                    "version": version_name,
                    "fit_time": fit_time,
                    "peak_memory_mb": peak_memory_mb
                }
                row.update(metrics)

                results.append(row)

        df_results = pd.DataFrame(results)

        if not df_results.empty:
            df_results.to_csv("real_benchmark_results/" + FILE_NAME, index=False, float_format="%.5f")
            print("\n===== BENCHMARK SAVED to " + FILE_NAME + " =====")
        else:
            print("No datasets processed successfully!")

    except Exception as e:
        print(f"Skipping dataset {file} due to error: {type(e).__name__}: {e}")
        # traceback.print_exc()
        skipped.append(file)
        continue

print("\nSkipped datasets:", skipped)
