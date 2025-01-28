from typing import List

import numpy as np

import utils.log_math as lmath
from utils.counter.counters import Counter, FrequenciesCounter
from utils.ngram import NGram
from utils.counter.node import Node


class AdditiveSmoothing:

    def __init__(self, counter: Counter, addition: int = 0) -> None:
        self.__counter = counter
        self.__unseen_words = [
            np.pow(self.__counter.unique_ngrams_count[0], i + 1) - self.__counter.unique_ngrams_count[i] for i in
            range(self.__counter.n)]
        self.__addition = addition

    def get_probabilities(self, ngram: NGram) -> List[float]:
        counts = self.__counter.get_count_n(ngram)
        probabilities: List[float] = [0 for _ in range(len(ngram))]
        probabilities[0] = np.log((counts[0] + self.__addition) / self.__counter.get_total_count())
        for i in range(1, len(ngram)):
            value = 0 if counts[i - 1] == -1 else counts[i - 1]
            favorable_count = counts[i] + self.__addition
            total_count = value + self.__addition * self.__unseen_words[i]
            if favorable_count == 0 or total_count == 0:
                probabilities[i] = -np.inf
            else:
                probabilities[i] = np.log(favorable_count / total_count)
        return probabilities


class SimpleGoodTuringSmoothing:

    def __init__(self, counter: FrequenciesCounter) -> None:
        self.__counter = counter
        self.__coefficients = self.__linear_fit_from_dict()
        self.__probabilities: List[List[float]] = [[] for _ in range(counter.n)]
        self.__compute_probabilities()

    def __compute_probabilities(self) -> None:
        probability_sums = [-np.inf for i in range(self.__counter.n)]
        total_log_count = np.log(self.__counter.get_total_count())
        for i in range(self.__counter.n):
            frequency_max, _ = next(reversed(self.__counter.frequencies[i].items()))
            self.__probabilities[i].append(self.__get_log_zr(i, 1) - total_log_count)
            probability_sums[i] = lmath.log_sum(probability_sums[i], self.__probabilities[i][0])
            for r in range(1, frequency_max + 1):
                log_new_r = np.log(r + 1) + self.__get_log_zr(i, r + 1) - self.__get_log_zr(i, r)
                smoothed_probability = log_new_r - total_log_count
                self.__probabilities[i].append(smoothed_probability)
                probability_sums[i] = lmath.log_sum(probability_sums[i], smoothed_probability)
        self.__normalize(probability_sums)

    def __normalize(self, probabilities_sums):
        for i in range(self.__counter.n):
            unseen_ngrams_count = np.pow(self.__counter.unique_ngrams_count[0], i + 1) - \
                                  self.__counter.unique_ngrams_count[i]
            self.__probabilities[i][0] -= -np.inf if unseen_ngrams_count == 0 else probabilities_sums[i] + np.log(
                unseen_ngrams_count)
            for j in range(1, len(self.__probabilities[i])):
                self.__probabilities[i][j] -= probabilities_sums[i] + np.log(self.__counter.frequencies[i][j])

    def __get_log_zr(self, i, r):
        return self.__coefficients[i][1] + self.__coefficients[i][0] * np.log(r)

    def __linear_fit_from_dict(self):
        coefficients = []
        normalized_frequencies = self.__normalize_frequencies()
        for i in range(self.__counter.n):
            keys = normalized_frequencies[i][0]
            values = normalized_frequencies[i][1]
            A = np.vstack([keys, np.ones_like(keys)]).T
            c = np.linalg.lstsq(A, values, rcond=None)[0]
            coefficients.append(c)
        return coefficients

    def __normalize_frequencies(self):
        normalized_frequencies = []
        for i in range(self.__counter.n):
            x = []
            y = []
            q, r, nr, t = 0, -1, -1, -1
            for key, value in self.__counter.frequencies[i].items():
                if value <= 0:
                    continue
                if r == -1:
                    r, nr = key, value
                    continue
                t = key
                x.append(np.log(r))
                y.append(np.log(nr / (0.5 * (t - q))))
                q = r
                r, nr = t, value
            x.append(np.log(r))
            y.append(np.log(nr / (r - q)))
            normalized_frequencies.append((x, y))
        return normalized_frequencies

    def get_probabilities(self, ngram: NGram) -> List[float]:
        occurrences = self.__counter.get_count_n(ngram)
        probabilities: List[float] = [-np.inf for _ in range(len(ngram))]
        for i in range(len(ngram)):
            probabilities[i] = self.__probabilities[i][occurrences[i]]
        return probabilities


class KatzSmoothing:

    def __init__(
            self,
            counter: FrequenciesCounter,
            k: int = 5
    ) -> None:
        self.__counter = counter
        self.__k = k
        self.__probabilities: List[List[float]] = [[] for _ in range(counter.n)]
        self.__alphas = self.__compute_coefficients()

    def __dr(self, r: int, i: int) -> float:
        if r > self.__k:
            return 1
        good_turing_r = (r + 1) * self.__counter.frequencies[i][r + 1] / self.__counter.frequencies[i][r]
        top = good_turing_r / self.__counter.frequencies[i][r] - (self.__k + 1) * self.__counter.frequencies[i][
            self.__k + 1] / self.__counter.frequencies[i][1]
        bottom = 1 - (self.__k + 1) * self.__counter.frequencies[i][self.__k + 1] / self.__counter.frequencies[i][1]
        return top / bottom

    def get_probabilities(self, ngram: NGram) -> List[float]:
        counts = self.__counter.get_count_n(ngram)
        probabilities = [self.__get_probabilities_inner(counts, len(ngram) - i - 1) for i in range(self.__counter.n)]
        return np.log(probabilities)

    def __get_probabilities_inner(self, counts: List[int], i: int):
        if counts[i] > 0:
            c = self.__dr(counts[i], i) * counts[i]
            return c / self.__counter.total_ngrams_count[i]
        else:
            return self.__alphas[i] * self.__get_probabilities_inner(counts, i - 1)

    def __compute_coefficients(self):
        probabilities_sum: List[float] = [0 for i in range(self.__counter.n)]
        self.__compute_p_katz_inner(self.__counter.root, probabilities_sum, 0)
        alphas: List[float] = [0 for i in range(self.__counter.n - 1)]
        for i in range(1, self.__counter.n):
            alphas[i - 1] = (1 - probabilities_sum[i]) / (1 - probabilities_sum[i - 1])
        return alphas

    def __compute_p_katz_inner(self, node: Node, probabilities_sum: List[float], i: int) -> None:
        for token, child in node.children.items():
            probabilities_sum[i] += self.__dr(child.value, i) * child.value / self.__counter.total_ngrams_count[i]
            self.__compute_p_katz_inner(child, probabilities_sum, i + 1)

