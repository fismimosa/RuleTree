import numpy as np
from sklearn.model_selection import train_test_split

from benchmark.Hybrid2.HybridReaders import all_datasets

import pickle

if __name__ == "__main__":
    dataset_shapes = []
    for dataset_reader in all_datasets:
        dataset_name, df = dataset_reader()
        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1].values, df.iloc[:, -1].values,
                                                            test_size=0.2, random_state=42,
                                                            stratify=df.iloc[:, -1].values)
        dataset_shapes.append((
            dataset_name,
            X_train.shape[0],
            X_test.shape[0],
            X_train.shape[1],
            len(np.unique(df.iloc[:, -1].values))
        ))

    dataset_shapes.sort(key=lambda x: x[3])

    pickle.dump(dataset_shapes, open('dataset_shapes.pkl', 'wb'))

    for dataset_name, train_size, test_size, n_features, n_classes in dataset_shapes:
        print(f"{train_size},\t{test_size},\t{n_features},\t{n_classes}\t{dataset_name}")