import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder

from scipy import linalg
from scipy.special import logsumexp
from sklearn.compose import ColumnTransformer, make_column_selector


def estimate_gaussian_covariances(X, nk, means, resp, reg_covar=1e-06):
    """Estimate the full covariance matrices.
    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)
    X : array-like of shape (n_samples, n_features)
    nk : array-like of shape (n_components,)
    means : array-like of shape (n_components, n_features)
    reg_covar : float
    Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
    """
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        covariances[k].flat[:: n_features + 1] += reg_covar
    return covariances


def estimate_parameters(X, resp, reg_covar=1e-06):
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    covariances = estimate_gaussian_covariances(X, nk, means, resp)
    return nk, means, covariances


def compute_precision_cholesky(covariances):
    """Compute the Cholesky decomposition of the precisions.
    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
    """
    n_components, n_features, _ = covariances.shape
    precisions_chol = np.empty((n_components, n_features, n_features))
    for k, covariance in enumerate(covariances):
        try:
            cov_chol = linalg.cholesky(covariance, lower=True)
        except linalg.LinAlgError:
            raise ValueError("Value error")
        precisions_chol[k] = linalg.solve_triangular(
            cov_chol, np.eye(n_features), lower=True
        ).T
    return precisions_chol


def estimate_weighted_log_prob(X, means, precisions_chol, weights):
    """Estimate the weighted log-probabilities, log P(X | Z) + log weights.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    Returns
    -------
    weighted_log_prob : array, shape (n_samples, n_component)
    """
    return estimate_log_prob(X, means, precisions_chol) + estimate_log_weights(weights)


def estimate_log_prob(X, means, precisions_chol):
    """Estimate the log Gaussian probability.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    means : array-like of shape (n_components, n_features)
    precisions_chol : array-like
    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    # The determinant of the precision matrix from the Cholesky decomposition
    # corresponds to the negative half of the determinant of the full precision
    # matrix.
    # In short: det(precision_chol) = - det(precision) / 2
    log_det = compute_log_det_cholesky(precisions_chol, n_features)

    log_prob = np.empty((n_samples, n_components))
    for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
        y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
        log_prob[:, k] = np.sum(np.square(y), axis=1)

    # Since we are using the precision of the Cholesky decomposition,
    # `- 0.5 * log_det_precision` becomes `+ log_det_precision_chol`
    return -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det


def compute_log_det_cholesky(matrix_chol, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.
    Parameters
    ----------
    matrix_chol : array-like
        Cholesky decompositions of the matrices.
    n_features : int
        Number of features.
    Returns
    -------
    log_det_precision_chol : array-like of shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    n_components, _, _ = matrix_chol.shape
    log_det_chol = np.sum(
        np.log(matrix_chol.reshape(n_components, -1)[:, :: n_features + 1]), 1
    )

    return log_det_chol


def estimate_log_weights(weights):
    return np.log(weights)


def score_samples(X, means, precisions_chol, weights):
    """Compute the log-likelihood of each sample.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row
        corresponds to a single data point.
    Returns
    -------
    log_prob : array, shape (n_samples,)
        Log-likelihood of each sample in `X` under the current model.
    """

    return logsumexp(
        estimate_weighted_log_prob(X, means, precisions_chol, weights), axis=1
    )


def score(X, n_components, labels):
    """Compute the per-sample average log-likelihood of the given data X.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_dimensions)
        List of n_features-dimensional data points. Each row
        corresponds to a single data point.
    Returns
    -------
    log_likelihood : float
        Log-likelihood of `X` under the Gaussian mixture model.
    """

    n_samples = X.shape[0]

    resp = np.zeros((n_samples, n_components))
    resp[np.arange(n_samples), labels] = 1

    weights, means, covariances = estimate_parameters(X, resp, reg_covar=1e-06)
    weights /= n_samples

    precisions_chol = compute_precision_cholesky(covariances)

    return score_samples(X, means, precisions_chol, weights).mean()


def get_n_parameters(X, n_components):
    """Return the number of free parameters in the model."""
    n_features = X.shape[1]
    cov_params = n_components * n_features * (n_features + 1) / 2.0
    mean_params = n_features * n_components
    n_parameters = int(cov_params + mean_params + n_components - 1)
    return n_parameters


def bic(X, labels):
    """Bayesian information criterion for the current model on the input X.
    You can refer to this :ref:`mathematical section <aic_bic>` for more
    details regarding the formulation of the BIC used.
    Parameters
    ----------
    X : array of shape (n_samples, n_dimensions)
        The input samples.
    Returns
    -------
    bic : float
        The lower the better.
    """

    if X.dtype == np.dtype('O'):
        X = pd.DataFrame(X).infer_objects()
        ct = ColumnTransformer([("cat", OrdinalEncoder(), make_column_selector(dtype_include="object"))],
                               remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=1)
        X = ct.fit_transform(X)

    n_records = X.shape[0]
    n_components = len(np.unique(labels))
    bic_score = score(X, n_components, labels)
    n_free_params = get_n_parameters(X, n_components)
    bic_val = -2 * bic_score * n_records + n_free_params * np.log(n_records)
    return bic_val
