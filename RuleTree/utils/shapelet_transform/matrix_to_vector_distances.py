import numpy as np
from numba import jit
from scipy.spatial import distance


@jit
def euclidean(matrix, vector):
    return np.sqrt(np.sum(np.power(matrix - vector, 2), axis=1))

@jit
def sqeuclidean(matrix, vector):
    return np.sum(np.power(matrix - vector, 2), axis=1)

@jit
def cityblock(matrix, vector):
    return np.sum(np.abs(matrix - vector), axis=1)

@jit(nopython=True)
def cosine(matrix, vector):
    res = np.zeros(matrix.shape[0])

    for i in range(matrix.shape[0]):
        try:
            res[i] = 1 - np.dot(matrix[i], vector) / np.linalg.norm(matrix[i]) * np.linalg.norm(vector)
        except:
            res[i] = np.nan
            #warnings.warn("Er")

    return np.nan_to_num(res)

if __name__ == "__main__":
    matrix = np.array([
        [0, 1, 0],
        [0, 0, 0],
    ], dtype=np.float64)

    vector = np.array([0, -1, 0], dtype=np.float64)

    print("Euclidean:", euclidean(matrix, vector), [distance.euclidean(el, vector) for el in matrix])
    print("SqEuclidean:", sqeuclidean(matrix, vector), [distance.sqeuclidean(el, vector) for el in matrix])
    print("Cosine:", cosine(matrix, vector), [distance.cosine(el, vector) for el in matrix])
    print("CityBlock:", cityblock(matrix, vector), [distance.cityblock(el, vector) for el in matrix])