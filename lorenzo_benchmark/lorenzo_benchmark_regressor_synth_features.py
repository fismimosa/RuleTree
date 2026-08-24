import time
import pandas as pd
import tracemalloc

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import datetime

from RuleTree.tree.RuleTreeRegressor import RuleTreeRegressor
from RuleTree.stumps.classification.DecisionTreeStumpClassifierBase import \
    DecisionTreeStumpClassifierBase
from RuleTree.stumps.regression.DecisionTreeStumpRegressorBase import DecisionTreeStumpRegressorBase
from RuleTree.stumps.regression.matrix.DecisionTreeStumpRegressorFlatMatrix import DecisionTreeStumpRegressorFlatMatrix
from RuleTree.stumps.regression.matrix.DecisionTreeStumpRegressorMatrixAB import DecisionTreeStumpRegressorMatrixAB
# Imports for the models
from RuleTree.tree.RuleTreeClassifier import RuleTreeClassifier
from RuleTree.stumps.classification.matrix.DecisionTreeStumpClassifierMatrixAB import \
    DecisionTreeStumpClassifierMatrixAB
from RuleTree.stumps.classification.matrix.DecisionTreeStumpClassifierFlatMatrix import \
    DecisionTreeStumpClassifierFlatMatrix
from benchmark.evaluation_utils import evaluate_clf, evaluate_expl, evaluate_reg
from tests.test_rule_tree import evaluate_regressor

# Settings
NUM_RECORDS = 10000
NUM_FEATURES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]  # [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
INFORMATIVE_PERC = 0.8
REDUNDANT_PERC = 0.05
REPEATED_PERC = 0.05
N_RUNS = 5
BATCH_SIZE = [20] # [0] per non usare
N_BINS= [0] # [0] per non usare
MAX_DEPTH = [2] # [1, 2, 3, 5, 8, 10]
STUMP_VERSIONS = [
    ("Base", DecisionTreeStumpRegressorBase),
    ("Matrix AB", DecisionTreeStumpRegressorMatrixAB),
    ("Flat Matrix", DecisionTreeStumpRegressorFlatMatrix)
]
FILE_NAME = "reg_benchmark_results_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"

if BATCH_SIZE[0] != 0:
    FILE_NAME = "B_" + str(BATCH_SIZE) + "_" + FILE_NAME

if N_BINS[0] != 0:
    FILE_NAME = "BIN_" + FILE_NAME

results = []

# TODO: fare grafico dove sulle x c'è la variazione della batch size e sulla y tempo

try:
    for j in range(0, len(MAX_DEPTH)):
        for i in range(0, len(NUM_FEATURES)):
            for k in range(0, len(N_BINS)):
                for b in range(0, len(BATCH_SIZE)):
                    print(f"--- Dataset with {NUM_FEATURES[i]} features, max_depth={MAX_DEPTH[j]}, n_bins={N_BINS[k]}, batch_size={BATCH_SIZE[b]} ---")

                    # -------- RUN LOOP --------
                    for run in range(1, N_RUNS + 1):
                        X, y = make_regression(
                            n_samples=NUM_RECORDS,
                            n_features=NUM_FEATURES[i],
                            n_informative=round(INFORMATIVE_PERC * NUM_FEATURES[i]),
                            noise=0.0,
                            random_state=run
                        )

                        y = y - y.min() + 1 # devo rendere positivi i target sennò da errore sulla MSLR

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.3, random_state=42
                        )
                        for version_name, stump_class in STUMP_VERSIONS:
                            print(f"Run {run}/{N_RUNS} - {version_name}")

                            if version_name == "Base":
                                clf = RuleTreeRegressor(
                                    max_depth=MAX_DEPTH[j],
                                    base_stumps=stump_class(random_state=run),
                                    random_state=run
                                )
                            else:
                                clf = RuleTreeRegressor(
                                    max_depth=MAX_DEPTH[j],
                                    base_stumps=stump_class(random_state=run, batch_size=BATCH_SIZE[b], n_bins=N_BINS[k] if N_BINS[k] != 0 else None),
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

                            # Evaluate metrics
                            metrics = evaluate_reg(y_test, y_pred) | evaluate_expl(clf)

                            # Build row
                            row = {
                                "n_features": NUM_FEATURES[i],
                                "depth": MAX_DEPTH[j],
                                "batch_size": BATCH_SIZE[b],
                                "n_bins": N_BINS[k],
                                "n_records": NUM_RECORDS,
                                "run": run,
                                "version": version_name,
                                "fit_time": fit_time,
                                "peak_memory_mb": peak_memory_mb
                            }
                            row.update(metrics)

                            results.append(row)

except Exception as e:
    print("ERRORE:")
    raise
except KeyboardInterrupt:
    # Creare .csv
    df_results = pd.DataFrame(results)
    df_results.to_csv("synth_benchmark_results/" + FILE_NAME, index=False, float_format="%.5f")
    print("\n===== BENCHMARK SAVED to " + FILE_NAME + " =====")

# Creare .csv
df_results = pd.DataFrame(results)
df_results.to_csv("synth_benchmark_results/" + FILE_NAME, index=False, float_format="%.5f")
print("\n===== BENCHMARK SAVED to " + FILE_NAME + " =====")
