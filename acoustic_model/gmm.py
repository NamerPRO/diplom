import time

import numpy as np
from sklearn import datasets

from acoustic_model.kmeans import KMeansClustering


class GMM:
    """
    Represents a Gaussian Mixture Model.
    """

    def __init__(self, n_components: int, observations: np.ndarray, means_init="kmeans", kmeans_max_iterations=1000, kmeans_attempts=5):
        """
        Initialises Gaussian Mixture Model.

        Args:
            n_components (int): number of components in GMM.
            observations (np.ndarray): vector of observations.
            means_init (str): type of mean initialisation in GMM: kmeans or random.
            kmeans_max_iterations (int): if means_init set to 'kmeans' is a number of iterations
                for this method, else ignored.
            kmeans_attempts (int): if means_init set to 'kmeans' is a number of attempts for
                this method, else ignored.
        """
        self.n_components = n_components
        self.observations = observations

        if means_init == 'kmeans':
            kmeans = KMeansClustering(n_components, observations)
            _, c, means, covmatrices = kmeans.cluster(kmeans_max_iterations, kmeans_attempts)
            self.means = means
            self.covmatrices = covmatrices
            self.c = c
        elif means_init == 'random':
            seed = int(time.time() * 1000)
            self.means = np.random.default_rng(seed).choice(observations, size=n_components, replace=False)
            self.covmatrices = np.array([np.atleast_2d(np.cov(observations.T)) for i in range(n_components)])
            self.c = np.full((n_components,), 1. / n_components)
            # self.covmatrices = np.array([np.full((self.means[0].shape[0],), np.var(observations)) for i in range(n_components)])
        else:
            raise ValueError('Means initialization method must be either "kmeans" or "random".')

    def __getitem__(self, observation: np.ndarray):
        """
        Glorified way of calling log_pdf method.

        Args:
            observation (np.ndarray): observation for which to compute the logarithmic probability.

        Return:
            np.float64: logarithmic probability of observation (result of calling pdf function).
        """
        return self.log_pdf(observation)

    def lth_gaussian_prob(self, observation, l: int):
        """
        Computes the probability of observation for only the concrete l-th gaussian.

        Args:
            observation (np.ndarray): observation for which to compute the probability.
            l (int): number of gaussian.

        Return:
            np.float64: probability of observation for only the concrete l-th gaussian.
        """
        n = self.means[l].shape[0]
        det = np.linalg.det(self.covmatrices[l])
        diff = observation - self.means[l]
        inv = np.linalg.inv(self.covmatrices[l])
        return self.c[l] * 1. / (np.pow(2 * np.pi, n / 2.) * np.sqrt(det)) * np.exp(-0.5 * np.dot(np.dot(diff, inv), diff))

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

    def log_lth_gaussian_prob(self, observation: np.ndarray, l: int):
        """
        Computes the logarithmic probability of observation for only the concrete l-th gaussian.

        Args:
            observation (np.ndarray): observation for which to compute the logarithmic probability.
            l (int): number of gaussian.

        Return:
            np.float64: logarithmic probability of observation for only the concrete l-th gaussian.
        """
        n = self.means[l].shape[0]
        log_det = np.log(np.linalg.det(self.covmatrices[l]))
        diff = observation - self.means[l]
        inv = np.linalg.inv(self.covmatrices[l])
        return np.log(self.c[l]) + -0.5 * n * np.log(2 * np.pi) - 0.5 * log_det - 0.5 * np.dot(diff, np.dot(inv, diff))

    def log_pdf(self, observation: np.ndarray):
        """
        Computes the logarithmic probability of observation.

        Args:
            observation (np.ndarray): observation for which to compute the logarithmic probability.

        Return:
            np.float64: logarithmic probability of observation.
        """
        log_probability = self.log_lth_gaussian_prob(observation, 0)
        for i in range(1, self.n_components):
            lth_log_prob = self.log_lth_gaussian_prob(observation, i)
            if log_probability == -np.inf and lth_log_prob == -np.inf:
                return -np.inf
            if log_probability < lth_log_prob:
                log_probability, lth_log_prob = lth_log_prob, log_probability
            log_probability += np.log(1 + np.exp(lth_log_prob - log_probability))
        return log_probability

    def train(self, max_iterations: int = 1000, eps: float = 1e-5, ignore_eps: bool = False):
        observations_n, observation_dim = self.observations.shape
        for it in range(max_iterations):
            # E-step
            responsibilities = np.ndarray((observations_n, self.n_components))
            means_data = np.zeros((self.n_components, observation_dim))
            cov_data = np.zeros((self.n_components, observation_dim, observation_dim))
            for i in range(observations_n):
                prob_sum = np.float64(0)
                for j in range(self.n_components):
                    responsibilities[i][j] = self.lth_gaussian_prob(self.observations[i], j)
                    prob_sum += responsibilities[i][j]
                responsibilities[i] /= prob_sum
                for j in range(self.n_components):
                    means_data[j] += responsibilities[i][j] * self.observations[i]
                    cov_data[j] += responsibilities[i][j] * ((self.observations[i] - self.means[j]) * (self.observations[i] - self.means[j])[:, np.newaxis])

            gamma = np.array([sum(x) for x in zip(*responsibilities)])

            # M-step
            new_means = np.ndarray(self.means.shape)
            new_covs = np.ndarray(self.covmatrices.shape)
            for i in range(self.n_components):
                new_means[i] = means_data[i] / gamma[i]
                new_covs[i] = cov_data[i] / gamma[i] + 1e-5 * np.eye(observation_dim)

            is_converged = np.linalg.norm(new_means - self.means) < eps and np.linalg.norm(new_covs - self.covmatrices) < eps and not ignore_eps

            self.c = gamma / observations_n
            self.means = new_means
            self.covmatrices = new_covs

            if is_converged:
                break


if __name__ == '__main__':
    observations, _ = datasets.make_blobs(n_features=39, n_samples=3000, centers=3)
    # print(observations)
    # y = np.zeros((100,))

    # plt.scatter(observations, y, c="black")

    # observations = np.array([[-5.1264101, 4.54832182, -4.58751138, 5.25010411],    x
    #                          [-4.92796138, 4.1509452, -7.95003681, -8.49320315],   x
    #                          [5.06547746, 0.47018908, 5.40772812, 0.51091962],
    #                          [-7.13369485, 5.0330122, -3.60152971, 5.29487825],
    #                          [-6.08928355, 3.30373881, -8.15888813, -9.10275598],
    #                          [6.60077888, -1.89502569, 4.97859433, 0.34524229],
    #                          [8.02113197, 0.4590251, 6.3297532, -0.05357864],
    #                          [-4.92348852, 4.34663022, -7.60703999, -8.42360852],  x
    #                          [-5.91335909, 4.48925547, -4.28575091, 6.90227208],   x
    #                          [7.47970946, -0.21634921, 5.52826701, 1.2531821]])

    gmm = GMM(3, observations, means_init="random")
    gmm.train()

    # arange = np.arange(-20, 20, 0.001)
    # plt.plot(arange, norm.pdf(arange, gmm.means[0], np.sqrt(gmm.covmatrices[0][0])))
    # plt.plot(arange, norm.pdf(arange, gmm.means[1], np.sqrt(gmm.covmatrices[1][0])))
    # plt.plot(arange, norm.pdf(arange, gmm.means[2], np.sqrt(gmm.covmatrices[2][0])))
    #
    # plt.show()

    print(gmm.c)
    print(gmm.means)
    print(gmm.covmatrices)