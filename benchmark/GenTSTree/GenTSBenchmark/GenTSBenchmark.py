import os
import pickle
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import psutil
from tqdm.auto import tqdm

from benchmark.GenTSTree.GenTSBenchmark.GenTSBenchmark_Hyper import get_hyperparameters
from benchmark.evaluation_utils import evaluate_clf

os.environ['OPENBLAS_NUM_THREADS'] = '32'
os.environ["OMP_NUM_THREADS"] = '32'
os.environ["OPENBLAS_NUM_THREADS"] = '32'
os.environ["MKL_NUM_THREADS"] = '32'
os.environ["NUMEXPR_NUM_THREADS"] = '32'
os.environ["VECLIB_MAXIMUM_THREADS"] = '32'

from benchmark.GenTSTree.GenTSReaders import small_datasets, medium_datasets, big_datasets

def run(method, hyper_dict, hyper_stump_dict, dataset_name, df: pd.DataFrame):
    hyper_dict = copy(hyper_dict)

    filename_hash = '|'.join([
        dataset_name,
        method,
        hashlib.md5('|'.join([f'{x}' for x in hyper_dict.values()]).encode()).hexdigest(),
        hashlib.md5('|'.join([f'{x}' for x in hyper_stump_dict.values()]).encode()).hexdigest(),
    ]) + '.csv'
    if os.path.exists(f"res_tmp/{dataset_name}/{filename_hash}"):
        return pd.read_csv(f"res_tmp/{dataset_name}/{filename_hash}")

    scores = copy(hyper_dict) | {'method': method}
    scores |= copy(hyper_stump_dict)

    y = df[df.columns[-1]].values
    X = StandardScaler().fit_transform(df[df.columns[:-1]].values)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scores_kv = []
    skf = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    hyper_dict_original = copy(hyper_dict)
    for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        hyper_dict = copy(hyper_dict_original)
        X_train_kf, X_val, y_train_kf, y_val = (X_train[train_index], X_train[test_index],
                                                y_train[train_index], y_train[test_index])
        if method == "DT":
            model = DecisionTreeStumpClassifier(**hyper_dict)
        elif method == "RF":
            model = RandomForestClassifier(**hyper_dict)
        elif method == "AB":
            model = AdaBoostClassifier(**hyper_dict)
        else:
            base_stumps = []
            if 'base_stumps' not in hyper_dict:
                raise ValueError(f"Base stumps not found in hyper_dict for {method}")
            for stump in hyper_dict["base_stumps"]:
                if stump == "PartialPivotTreeStumpClassifier":
                    base_stumps.append(PartialPivotTreeStumpClassifier(**hyper_stump_dict[stump]))
                elif stump == "PartialProximityTreeStumpClassifier":
                     base_stumps.append(PartialProximityTreeStumpClassifier(**hyper_stump_dict[stump]))
                elif stump == "ObliqueDecisionTreeStumpClassifier":
                    base_stumps.append(ObliqueDecisionTreeStumpClassifier(**hyper_stump_dict[stump]))
                elif stump == "ObliquePivotTreeStumpClassifier":
                    base_stumps.append(ObliquePivotTreeStumpClassifier(**hyper_stump_dict[stump]))
                elif stump == "PivotTreeStumpClassifier":
                    base_stumps.append(PivotTreeStumpClassifier(**hyper_stump_dict[stump]))
                elif stump == "MultiplePivotTreeStumpClassifier":
                    base_stumps.append(MultiplePivotTreeStumpClassifier(**hyper_stump_dict[stump]))
                elif stump == "MultipleObliquePivotTreeStumpClassifier":
                    base_stumps.append(MultipleObliquePivotTreeStumpClassifier(**hyper_stump_dict[stump]))
                elif stump == "DecisionTreeStumpClassifier":
                    base_stumps.append(DecisionTreeStumpClassifier(**hyper_stump_dict[stump]))
                else:
                    raise ValueError(f"Unknown stump type: {stump}")

            del hyper_dict["base_stumps"]
            model = RuleTreeClassifier(**hyper_dict, base_stumps=base_stumps)

        start = time.time()
        try:
            model.fit(X_train_kf, y_train_kf)
        except UnsupportedError:
            return pd.DataFrame()
        except Exception as e:
            traceback.print_exc()
            return pd.DataFrame()
        end = time.time()

        y_pred_val = model.predict(X_val)

        dict_res = {f'{k}_val': v for k, v in compute_measures(y_val, y_pred_val, model).items()}
        scores_kv.append(dict_res)

    hyper_dict = hyper_dict_original.copy()

    if method == "DT":
        model = DecisionTreeStumpClassifier(**hyper_dict)
    elif method == "RF":
        model = RandomForestClassifier(**hyper_dict)
    elif method == "AB":
        model = AdaBoostClassifier(**hyper_dict)
    else:
        raise ValueError(f"Unknown method {method}")

        del hyper_dict["base_stumps"]
        model = RuleTreeClassifier(**hyper_dict, base_stumps=base_stumps)

    try:
        process = psutil.Process(os.getpid())
        before = process.memory_info().rss
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        after = process.memory_info().rss

        before_pred = process.memory_info().rss
        start_pred = time.time()
        y_pred = model.predict(X_test)
        end_pred = time.time()
        after_pred = process.memory_info().rss
    except Exception as e:
        traceback.print_exc()
        return pd.DataFrame()

    scores |= (
            {'train_time': end - start, 'pred_time': end_pred - start_pred}
            | {'train_mem_MB': (after - before) / 1024 ** 2, 'pred_mem_MB': (after_pred - before_pred) / 1024 ** 2}
            | {f'{k}_test': v for k, v in evaluate_clf(y_test, y_pred, y_pred_proba=None).items()}
    )

    df_val = pd.DataFrame(scores_kv).mean().to_frame().T

    df = pd.DataFrame.from_dict([scores])
    df[df_val.columns] = df_val
    try:
        df.to_csv(f"res_tmp/{dataset_name}/{filename_hash}", index=False)
        pickle.dump(model, open(f"res_tmp/{dataset_name}/{filename_hash.replace('.csv', '.pickle')}", "wb"))
    except Exception as e:
        raise e

    return df


def main():
    datasets = dict() #small_datasets
    for dataset_fun in small_datasets + medium_datasets + big_datasets:
        dataset_name, df = dataset_fun()
        print(dataset_name, df.shape)
        for scaler_name, scaler_fun in scalers.items():
            if scaler_fun is None:
                datasets[dataset_name] = df
            else:
                df_scaled = pd.DataFrame(scaler_fun().fit_transform(df.iloc[:, :-1]).reshape(df.shape[0], -1),
                                         columns=df.columns[:-1])
                df_scaled[df.columns[-1]] = df.iloc[:, -1]
                datasets[f'{dataset_name}_{scaler_name}'] = df_scaled


    n_process = int(psutil.cpu_count(logical=True) / n_jobs)+2
    print(n_process)
    #n_process = 16

    processes = []
    dataframes = []
    single_t = False
    with ProcessPoolExecutor(max_workers=n_process) as executor:
        for dataset_name, df in datasets.items():
            if dataset_name in ['wine', 'ionosphere']:
                continue
            if not os.path.exists("res_tmp/"):
                os.mkdir("res_tmp/")
            if not os.path.exists("res_tmp/" + dataset_name):
                os.mkdir("res_tmp/" + dataset_name)

            if single_t:
                for model, hyper_dict, hyper_stump_dict in tqdm(list(get_hyperparameters(df)),
                                                                desc=f"{dataset_name}: submitting experiments"):
                    processes.append(run(model, hyper_dict, hyper_stump_dict, dataset_name, df))

                for process in tqdm(processes, desc=f"{dataset_name}: collecting results"):
                    dataframes.append(process)

            else:
                for model, hyper_dict, hyper_stump_dict in tqdm(list(get_hyperparameters(df)),
                                                                desc=f"{dataset_name}: submitting experiments"):
                    processes.append((dataset_name,
                                      executor.submit(run, model, hyper_dict, hyper_stump_dict, dataset_name, df)))

                #for process in tqdm(processes, desc=f"{dataset_name}: collecting results"):
                #    try:
                #        res = process.result()
                #        dataframes.append(res)
                #    except Exception as e:
                #        traceback.print_exc()


        it = tqdm(processes)
        for dataset_name, process in it:
            try:
                it.set_description(f"{dataset_name}: collecting results")
                res = process.result()
                dataframes.append(res)
            except Exception as e:
                traceback.print_exc()

        #pd.concat(dataframes, ignore_index=True).to_csv(f"res_tmp/hybrid_benchmark_{dataset_name}.csv", index=False)


if __name__ == '__main__':
    main()