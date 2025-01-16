import time

import numpy as np

from acoustic_model.kmeans import KMeansClustering


class GMM:
    """
    Represents a Gaussian Mixture Model.
    """

    def __init__(self, n_components: int, observations: np.ndarray, means_init="kmeans"):
        """
        Initialises Gaussian Mixture Model.

        Args:
            n_components (int): number of components in GMM.
            observations (np.ndarray): vector of observations.
            means_init (str): type of mean initialisation in GMM.
        """
        self.n_components = n_components

        if means_init == 'kmeans':
            kmeans = KMeansClustering(n_components, observations)
            _, means, variances = kmeans.cluster()
            self.means = means
            self.variations = variances
        elif means_init == 'random':
            seed = int(time.time() * 1000)
            self.means = np.random.default_rng(seed).choice(observations, size=n_components, replace=False)
            self.variations = np.array([np.full((self.means[0].shape[0],), np.var(observations)) for i in range(n_components)])
        else:
            raise ValueError('Means initialization method must be either "kmeans" or "random".')

        self.c = np.full((n_components,), 1. / n_components)

    def __getitem__(self, observation: np.ndarray):
        """
        Glorified way of calling pdf method.

        Args:
            observation (np.ndarray): observation for which to compute the probability.

        Return:
            np.float64: probability of observation (result of calling pdf function).
        """
        return self.pdf(observation)

    def lth_gaussian_prob(self, observation, l: int):
        """
        Computes the probability of observation for only the concrete l-th gaussian.

        Args:
            observation (np.ndarray): observation for which to compute the probability.
            l (int): number of gaussian.

        Return:
            np.float64: probability of observation for only the concrete l-th gaussian.
        """
        probability = np.float64(1)
        mean_l = self.means[l]
        variation_l = self.variations[l]
        for i in range(mean_l.shape[0]):
            probability *= 1. / (variation_l[i] * np.sqrt(2 * np.pi)) * np.exp(-0.5 * observation[i] - mean_l[i])
        return self.c[l] * probability

    def pdf(self, observation: np.ndarray):
        """
        Computes the probability of observation.

        Args:
            observation (np.ndarray): observation for which to compute the probability.

        Return:
            np.float64: probability of observation.
        """
        probability = np.float64(0)
        for i in range(self.n_components):
            probability += self.lth_gaussian_prob(observation, i)
        return probability

    # Training happens to be in Baum-Welch algorithm inside HMM
    # and thus omitted here.

