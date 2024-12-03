import time
import typing
from functools import partial

import jax
import numpy as np
from jax import numpy as jnp
from jax import jit
from jax import scipy as jscipy
from jax import random

from numba import jit as numba_jit
try:
  from jax._src.numpy.util import implements
except ImportError:
  from jax._src.numpy.util import _wraps as implements  # for jax < 0.4.25

from jax_scipy_spatial.distance import braycurtis, canberra, chebyshev, cityblock, correlation, cosine, euclidean, \
    hamming, jaccard, minkowski, russellrao, sqeuclidean

from scipy.spatial import distance
import psutil
import os
from sklearn.base import TransformerMixin

@jit
def jensenshannon(XA, XB):
    m = (XA+XB)/2
    return jnp.sqrt((jscipy.special.kl_div(XA, m) + jscipy.special.kl_div(XB, m))/2)

_distance_map = {
    "braycurtis": braycurtis,
    "canberra": canberra,
    "chebyshev": chebyshev,
    "cityblock": cityblock,
    "correlation": correlation,
    "cosine": cosine,
    "dice": None, # dissimilarity between boolean 1-D arrays
    "euclidean": euclidean,
    "hamming": hamming,
    "jaccard": jaccard,
    "jensenshannon": jensenshannon,
    "kulczynski1": None, # dissimilarity between boolean 1-D arrays
    "mahalanobis": None,
    "matching": hamming,
    "minkowski": minkowski,
    "rogerstanimoto": None,  # dissimilarity between boolean 1-D arrays
    "russellrao": russellrao,
    "seuclidean": None,
    "sokalmichener": None,  # dissimilarity between boolean 1-D arrays
    "sokalsneath": None,  # dissimilarity between boolean 1-D arrays
    "sqeuclidean": sqeuclidean,
    "yule": None,  # dissimilarity between boolean 1-D arrays
}


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
        os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads="+str(n_jobs)
        os.environ["OPENBLAS_NUM_THREADS"] = str(n_jobs)
        os.environ["MKL_NUM_THREADS"] = str(n_jobs)
        os.environ["OMP_NUM_THREAD"] = str(n_jobs)

    def _get_distance(self):
        if self.distance not in _distance_map or _distance_map[self.distance] is None:
            raise NotImplemented(f"Distance {self.distance} is not implemented (see: jax_scipy_spatial)")

        return _distance_map[self.distance]


    def fit(self, X, y=None, **fit_params):
        # X.shape = (n_records, n_signals, n_obs)
        if X.shape[1] != self.n_shapelets:
            raise NotImplemented("Multivariate TS are not supported (yet).")
        pass

    def transform(self, X, y=None, **transform_params):
        pass

@jit
def _best_fit_jax_for(timeseries: jnp.ndarray, shapelets: jnp.ndarray):
    res = jnp.zeros((timeseries.shape[0], shapelets.shape[0]), dtype=jnp.float32)
    positions = timeseries.shape[-1] - shapelets.shape[-1]
    d = Shapelet()._get_distance()

    for ts_idx, ts in enumerate(timeseries[:, 0, :]):
        for shapelet_idx, shapelet in enumerate(shapelets[:, 0, :]):
            dist_vector = np.zeros((positions,), dtype=jnp.float32)
            for start in range(positions + 1):
                dist_vector[start:start+1] = d(timeseries[ts_idx:ts_idx+1, 0, start:start + shapelets.shape[-1]],
                                       shapelets[shapelet_idx:shapelet_idx+1, 0, :])
            res[ts_idx, shapelet_idx] = jnp.argmin(dist_vector)

    return res

@numba_jit
def _best_fit_classic_for(timeseries: np.ndarray, shapelets: np.ndarray):
    res = np.zeros((timeseries.shape[0], shapelets.shape[0]), dtype=np.float32)
    positions = timeseries.shape[-1] - shapelets.shape[-1]

    for ts_idx, ts in enumerate(timeseries[:, 0, :]):
        for shapelet_idx, shapelet in enumerate(shapelets[:, 0, :]):
            dist_vector = np.zeros((positions,), dtype=np.float32)
            for start in range(positions + 1):
                dist_vector[start] = distance.euclidean(ts[start:start + shapelets.shape[-1]], shapelet)
            res[ts_idx, shapelet_idx] = np.argmin(dist_vector)

    return res




if __name__ == '__main__':
    random.key(42)
    X = np.random.rand(10000, 1, 1000).astype(np.float32)
    shapelets = np.random.rand(100, 1, 50).astype(np.float32)

    st = Shapelet(n_jobs=1)

    """start = time.time()
    jax_res = _best_fit_jax_for(X, shapelets)
    print("JAX_TIME", round(time.time()-start, 6))"""

    start = time.time()
    classic_res = _best_fit_classic_for(X, shapelets)
    print("Classic_TIME", round(time.time()-start, 6))
