import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_iris, make_blobs


@pytest.fixture(scope="session")
def iris():
    return load_iris(return_X_y=True)


@pytest.fixture(scope="session")
def diabetes():
    return load_diabetes(return_X_y=True)


@pytest.fixture(scope="session")
def blobs():
    return make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=0.5, random_state=0)


@pytest.fixture(scope="session")
def iris_mixed(iris):
    """Iris with petal length replaced by a categorical feature strongly related to the class."""
    X, y = iris
    color = np.where(X[:, 2] < 2.5, "red", np.where(X[:, 2] < 4.8, "green", "blue"))
    X_mixed = np.empty((len(y), 3), dtype=object)  # column_stack would turn the numbers into strings
    X_mixed[:, 0], X_mixed[:, 1:] = color, X[:, :2]
    return X_mixed, y
