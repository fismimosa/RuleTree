import time
import numpy as np
from jax import numpy as jnp, pmap
from jax import jit
from jax import scipy as jscipy
from jax import random

from fastdist import fastdist

from numba import jit as numba_jit, prange
import psutil
import os
from sklearn.base import TransformerMixin

@jit
def jensenshannon(XA, XB):
    m = (XA+XB)/2
    return jnp.sqrt((jscipy.special.kl_div(XA, m) + jscipy.special.kl_div(XB, m))/2)

def rolling_window(a: jnp.ndarray, window: int):
  idx = jnp.arange(len(a) - window + 1)[:, None] + jnp.arange(window)[None, :]
  return a[idx]

class Shapelet(TransformerMixin):
    def __init__(self, n_shapelets=100, sliding_window=50, selection='random', save_results=True, distance='euclidean',
                 random_state=42, n_jobs=1):
        super().__init__()

        if n_jobs == -1:
            n_jobs = psutil.cpu_count()

        self.n_shapelets = n_shapelets
        self.sliding_window = sliding_window
        self.selection = selection
        self.save_results = save_results
        self.distance = distance
        self.random_state = random_state
        self.n_jobs = n_jobs

        random.key(random_state)
        os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=" + str(n_jobs)
        os.environ["OPENBLAS_NUM_THREADS"] = str(n_jobs)
        os.environ["MKL_NUM_THREADS"] = str(n_jobs)
        os.environ["OMP_NUM_THREAD"] = str(n_jobs)

    def fit(self, X, y=None, **fit_params):
        # X.shape = (n_records, n_signals, n_obs)
        if X.shape[1] != self.n_shapelets:
            raise NotImplemented("Multivariate TS are not supported (yet).")
        pass

    def transform(self, X, y=None, **transform_params):
        pass


def _best_fit_classic_for(timeseries: np.ndarray, shapelets: np.ndarray):
    res = np.zeros((timeseries.shape[0], shapelets.shape[0]), dtype=np.float32)
    w = shapelets.shape[-1]

    for ts_idx, ts in enumerate(timeseries[:, 0, :]):
        ts_sw = np.lib.stride_tricks.sliding_window_view(ts, w)
        tmp = ts_sw[:, np.newaxis] - shapelets[:, 0, :]
        tmp = np.sum(tmp**2, axis=2)
        res[ts_idx, :] = tmp.min(axis=0)

    return res

@numba_jit(parallel=True)
def _best_fit_classic_for2(timeseries: np.ndarray, shapelets: np.ndarray):
    res = np.zeros((timeseries.shape[0], shapelets.shape[0]), dtype=np.float32)
    w = shapelets.shape[-1]

    for ts_idx in prange(timeseries.shape[0]):
        ts = timeseries[ts_idx, 0, :]
        ts_sw = np.lib.stride_tricks.sliding_window_view(ts, w)
        for shapelet_idx, shapelet in enumerate(shapelets[:, 0, :]):
            distance_matrix = np.sum((ts_sw - shapelet) ** 2, axis=1) ** .5
            res[ts_idx, shapelet_idx] = np.min(distance_matrix)

    return res




if __name__ == '__main__':
    random.key(42)
    X = np.random.rand(10000, 1, 1000).astype(np.float32)
    shapelets = np.random.rand(100, 1, 50).astype(np.float32)

    _best_fit_classic_for(X, shapelets)
    _best_fit_classic_for2(X, shapelets)

    st = Shapelet(n_jobs=1)

    start = time.time()
    jax_res = _best_fit_classic_for(X, shapelets)
    print("JAX_TIME", round(time.time()-start, 6))


    start = time.time()
    classic_res = _best_fit_classic_for2(X, shapelets)
    print("Classic_TIME", round(time.time()-start, 6))
