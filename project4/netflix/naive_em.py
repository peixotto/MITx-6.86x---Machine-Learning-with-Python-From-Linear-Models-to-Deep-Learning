"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    # raise NotImplementedError
    # setting the number of samples (n), dimension (d) and number of clusters (K)
    n,d = X.shape   # 250, 2

    mu_j = mixture[0]
    var_j = mixture[1]
    post = mixture[2]

    deno = (2 * np.pi * var_j) ** (-d / 2) # Denominator
    expo = np.exp( (np.linalg.norm(X[:, np.newaxis] - mu_j, axis = 2) ** 2) / (-2 * var_j) ) # Exponent

    soft = post * deno * expo         
    total = soft.sum(axis=1).reshape(n, 1)
    weighted = np.divide(soft, total)     
    LL = np.sum(np.log(total), axis=0)
    
    # code will output updated versions of a GaussianMixture (with means mu, variances var and mixing proportions p) 
    return weighted, float(LL)



def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    # raise NotImplementedError
    n,d = X.shape   # 250, 2
    K = post.shape[1]

    n_hat = post.sum(axis=0)
    p_hat = n_hat / n

    mu_deno = n_hat.reshape(K,1)
    mu_nume = np.dot( np.transpose(post), X)
    mu_hat = mu_nume / mu_deno

    var_deno = n_hat * d
    var_nume = np.linalg.norm(X[:, np.newaxis] - mu_hat, axis = 2) ** 2
    var_sum = (post * var_nume).sum(axis=0)
    var_hat = var_sum / var_deno

    return GaussianMixture(mu_hat, var_hat, p_hat)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_log_likelihood = None
    new_log_likelihood = None

    while (old_log_likelihood is None or new_log_likelihood - old_log_likelihood >= 10**-6 * abs(new_log_likelihood)):
        
        old_log_likelihood = new_log_likelihood
        post, new_log_likelihood = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, new_log_likelihood
