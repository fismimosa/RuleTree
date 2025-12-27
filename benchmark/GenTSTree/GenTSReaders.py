import warnings

import numpy as np
from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import univariate

from tqdm.auto import tqdm

small_datasets = []

medium_datasets = []

big_datasets = []

def _reader_fun(dataset_name):
    X_train, y_train = load_classification(dataset_name, split="train", load_equal_length=True, load_no_missing=True)
    X_test, y_test = load_classification(dataset_name, split="test", load_equal_length=True, load_no_missing=True)

    if isinstance(X_train, list):
        X_train = np.array(X_train)
        X_test = np.array(X_test)

    return dataset_name, X_train[:, 0, :], y_train, X_test[:, 0, :], y_test

it = tqdm(univariate)
for dataset_name in it:
    it.set_description(f"Preparing: {dataset_name}")
    try:
        _, X_train, y_train, X_test, y_test = _reader_fun(dataset_name)
    except Exception as e:
        warnings.warn(f'Unable to load {dataset_name}. Is it irregular?')
        continue

    n_ts = X_train.shape[0] + X_test.shape[0]
    n_features = max(X_train.shape[1], X_test.shape[1])

    if n_ts < 200 and n_features < 500:
        small_datasets.append(lambda : _reader_fun(dataset_name))
    elif n_ts < 200 or n_features < 1000:
        medium_datasets.append(lambda : _reader_fun(dataset_name))
    else:
        big_datasets.append(lambda : _reader_fun(dataset_name))

if __name__ == "__main__":
    print(f'len(small_datasets) = {len(small_datasets)}')
    print(f'len(medium_datasets) = {len(medium_datasets)}')
    print(f'len(big_datasets) = {len(big_datasets)}')
    print(f'total = {len(small_datasets) + len(medium_datasets) + len(big_datasets)}/{len(univariate)}')
