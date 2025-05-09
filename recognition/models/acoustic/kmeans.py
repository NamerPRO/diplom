import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


class KMeansClustering:
    """
    Представляет собой алгоритм кластеризации K-means.
    """

    def __init__(self, n_clusters, observations):
        """
        Инициализация экземпляра KMeansClustering.

        Аргументы:
            observations: Одномерный numpy массив наблюдений.
            n_clusters: Количество кластеров.
        """
        self.n_clusters = n_clusters
        self.observations = observations

    def __total_intraclass_variance(self, clusters, means):
        """
        Рассчитывает общую внутриклассовую дисперсию.

        Аргументы:
            clusters: Одномерный numpy массив кластеров.
            means: Одномерный массив значений.

        Возвращаемое значение:
            Общая внутриклассовая дисперсия.
        """
        variance = 0
        for i in range(self.n_clusters):
            clustered_observations = self.observations[clusters == i]
            if clustered_observations.size == 0:
                continue
            variance += np.linalg.norm(clustered_observations - means[i]) ** 2
        return variance

    def __distance(self, x, y):
        """
        Вычисляет расстояние между двумя точками.

        Аргументы:
            x: Одномерный numpy массив, представляющий первую точку.
            y: Одномерный numpy массив, представляющий вторую точку.

        Возвращаемое значение:
            Расстояние между двумя точками.
        """
        return np.sqrt(np.sum(np.square(x - y)))

    def __update_centroids(self, clusters):
        """
        Пересчитывает центроиды для каждого кластера.

        Аргументы:
            clusters: Одномерный numpy массив кластеров.

        Возвращаемое значение:
            Одномерный numpy массив пересчитанных центроидов для каждого кластера.
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
        Вычисляет, к какому кластеру относится наблюдение k.

        Аргументы:
            k: Число наблюдений, для которых необходимо рассчитать кластер.
            centroids: Одномерный numpy массив центроидов.

        Возвращаемое значение:
            Номер кластера, к которому принадлежит наблюдение.
        """
        distances = np.zeros((self.n_clusters,))
        for i in range(self.n_clusters):
            distances[i] = self.__distance(centroids[i], self.observations[k])
        return np.argmin(distances)

    def cluster(self, max_iters=1000, attempts=5):
        """
        Реализует алгоритм K-Means.

        Аргуметы:
            max_iters: Предельное число итераций. По-умолчанию: 1000.
            attempts: Количество попыток. Лучший результат будет возвращен.
                По-умолчанию: 5.

        Возвращаемое значение:
            Словарь, содержащий информацию из лучшей попытки, которая включает в себя:
                - clusters: Одномерный numpy массив кластеров. Наблюдение observations[i]
                    принадлежит кластеру cluster[i].
                - с: Коэффициенты, используемые для инициализации весов компонент в GMM,
                    если тип начальной инициализации GMM выбран 'kmeans'.
                - centroids: Одномерный массив центроидов для каждого кластера.
                - covmatrixes: Массив ковариационных матриц для каждого кластера.
        """
        observations_n, dim = self.observations.shape
        clusters = np.zeros((observations_n,))
        best_values = {"clusters": None, "total_intraclass_variance": np.inf, "centroids": None}
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
        covmatrices = np.array(
            [np.atleast_2d(np.cov(clustered_observations[i].T)) + 1e-5 * np.eye(dim) for i in range(self.n_clusters)])
        c = [len(clustered_observations[i]) / observations_n for i in range(self.n_clusters)]
        return best_values["clusters"], c, best_values["centroids"], covmatrices


if __name__ == "__main__":
    X, y = datasets.make_blobs(n_features=3)
    kmeans = KMeansClustering(n_clusters=3, observations=X)
    clusters, _, means, variances = kmeans.cluster(max_iters=1000)

    print(clusters)
    print(means)
    print(variances)

    plt.scatter(X[:, 0], X[:, 1], c=clusters)

    plt.show()

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
