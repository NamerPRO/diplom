import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


class KMeansClustering:
    """
    Represents a K-Means clustering algorithm.
    """

    def __init__(self, n_clusters, observations):
        """
        Initialization of an instance of KMeansClustering.

        Args:
            observations (np.ndarray): A 1D numpy array of observations.
            n_clusters (int): The number of clusters.
        """
        self.n_clusters = n_clusters
        self.observations = observations

    def __total_intraclass_variance(self, clusters, means):
        """
        Calculates the total intraclass variance.

        Args:
            clusters (np.ndarray): A 1D numpy array of clusters.
            means (np.ndarray): A 1D numpy array of means.

        Return:
            np.float64: Total intraclass variance.
        """
        variance = 0
        for i in range(self.n_clusters):
            clustered_observations = self.observations[clusters == i]
            if clustered_observations.size == 0:
                continue
            variance += np.linalg.norm(clustered_observations - means[i])**2
        return variance

    def __distance(self, x, y):
        """
        Calculates the distance between two points.

        Args:
            x (np.ndarray): A 1D numpy array representing first point.
            y (np.ndarray): A 1D numpy array representing second point.

        Return:
            float: The distance between two points.
        """
        return np.sqrt(np.sum(np.square(x - y)))

    def __update_centroids(self, clusters):
        """
        Recalculates the centroids for each cluster.

        Args:
            clusters (np.ndarray): A 1D numpy array of clusters.

        Return:
            np.ndarray: A 1D numpy array of recalculated centroids for each cluster.
        """
        centroids = np.zeros((self.n_clusters, self.observations.shape[1]))
        for i in range(self.n_clusters):
            clustered_observations = self.observations[clusters == i]
            if clustered_observations.size == 0:
                continue
            centroids[i] = np.mean(clustered_observations, axis=0)
        return centroids

    def __get_cluster(self, k, centroids):
        """
        Calculates to which cluster observation k belongs.

        Args:
            k (int): The number of observation for which to calculate the cluster.
            centroids (np.ndarray): A 1D numpy array of centroids.

        Return:
            int: The cluster number to which observation belongs.
        """
        distances = np.zeros((self.n_clusters,))
        for i in range(self.n_clusters):
            distances[i] = self.__distance(centroids[i], self.observations[k])
        return np.argmin(distances)

    def cluster(self, max_iters=1000, attempts=5):
        """
        Implements the K-Means algorithm.

        Args:
            max_iters (int): The maximum number of iterations.
            attempts (int): The number of attempts. The best result will be returned.


        Return:
            dict: A dictionary containing the information from the best attempt which includes:
                - clusters (np.ndarray): A 1D numpy array of clusters. observations[i] observation
                    belongs to the cluster[i] cluster.
                - total_variance (float): Sum of variations of each cluster.
                - centroids (np.ndarray): A 1D numpy array of centroids for each cluster.
                - covmatrixes (np.ndarray): An array of covariance matrices for each cluster.
        """
        observations_n, dim = self.observations.shape
        clusters = np.zeros((observations_n,))
        best_values = { "clusters": None, "total_intraclass_variance": np.inf, "centroids": None }
        for attempt in range(attempts):
            seed = int(time.time() * 1000)
            centroids = np.random.default_rng(seed).choice(self.observations, size=self.n_clusters, replace=False)
            for i in range(max_iters):
                for cur_observation in range(observations_n):
                    clusters[cur_observation] = self.__get_cluster(cur_observation, centroids)
                new_centroids = self.__update_centroids(clusters)
                diff = new_centroids - centroids
                centroids = new_centroids
                if not diff.any():
                    break
            variance = self.__total_intraclass_variance(clusters, centroids)
            if best_values["total_intraclass_variance"] > variance:
                best_values["clusters"] = clusters.__deepcopy__(None)
                best_values["total_intraclass_variance"] = variance
                best_values["centroids"] = centroids
        clustered_observations = [self.observations[best_values["clusters"] == i] for i in range(self.n_clusters)]
        covmatrices = np.array([np.atleast_2d(np.cov(clustered_observations[i].T)) + 1e-5 * np.eye(dim) for i in range(self.n_clusters)])
        c = [len(clustered_observations[i]) / observations_n for i in range(self.n_clusters)]
        return best_values["clusters"], c, best_values["centroids"], covmatrices


if __name__ == "__main__":
    X, y = datasets.make_blobs(n_features=3)
    kmeans = KMeansClustering(n_clusters=3, observations=X)
    clusters, means, variances = kmeans.cluster(max_iters=1000)

    print(clusters)
    print(means)
    print(variances)

    plt.scatter(X[:, 0], X[:, 1], c=clusters)

    plt.show()

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()


