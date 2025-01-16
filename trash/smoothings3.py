from collections import deque, defaultdict
from copy import deepcopy
from typing import List, Dict, Deque, Tuple

import numpy as np

import utility.log_math as lmath
from language_model.counters import Counter, FrequenciesCounter
from language_model.ngram import NGram
from language_model.node import Node


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
        self.__top = None
        self.__bottom = None

        self.__compute_coefficients()
        # print(self.__top)
        # print(self.__bottom)

    def __dr(self, r: int, i: int) -> float:
        if r > self.__k:
            return 1
        good_turing_r = (r + 1) * self.__get_frequencies_safe(i, r + 1) / self.__get_frequencies_safe(i, r)
        top = good_turing_r / r - (self.__k + 1) * self.__get_frequencies_safe(i, self.__k + 1) / \
              self.__get_frequencies_safe(i, 1)
        bottom = 1 - (self.__k + 1) * self.__get_frequencies_safe(i, self.__k + 1) / self.__get_frequencies_safe(i, 1)
        return top / bottom

    def __get_frequencies_safe(self, i: int, r: int):
        return self.__probabilities[i][r] if r in self.__probabilities[i] else 1e-12

    def get_probabilities(self, ngram: NGram) -> List[float]:
        # counts = self.__counter.get_count_n(ngram)
        # probabilities = [self.__get_probability_inner(ngram) for i in range(self.__counter.n)]

        x = self.__get_probability_inner(ngram)
        # print('p', x)
        return np.log(x)

    # def __get_probabilities_inner(self, counts: List[int], i: int):
    #     if counts[i] > 0:
    #         c = self.__dr(counts[i], i) * counts[i]
    #         return c / self.__counter.total_ngrams_count[i - 1]
    #     else:
    #         return self.__alphas[i - 1] * self.__get_probabilities_inner(counts, i - 1)

    def __get_probability_inner(self, ngram: NGram) -> float:
        n = len(ngram)
        counts = self.__counter.get_count_n(ngram)
        r = counts[n - 1]
        if r > 0 or n == 1:
            ngram.shorten()
            x = counts[n - 2] if n > 1 else self.__counter.unique_ngrams_count[0]
            return self.__p_katz(r, n, x)
        if n > 1 and counts[n - 2] == 0:
            ngram.shorten()
            return self.__get_probability_inner(ngram)
        context = deepcopy(ngram)
        context.shorten(direction="right")
        ngram.shorten()
        return self.__get_alpha(context) * self.__get_probability_inner(ngram)

    def __p_katz(self, r: int, n: int, cnt: int) -> float:
        # print('dr=', self.__dr(r, n - 1))
        return self.__dr(r, n - 1) * r / cnt

    def __compute_coefficients(self):
        top: Dict[Deque[str], float] = defaultdict(float)
        bottom: Dict[Deque[str], float] = defaultdict(float)
        self.__compute_alpha_parameter(self.__counter.root, deque(), top, bottom)
        self.__top, self.__bottom = top, bottom

    def __get_alpha(self, context: NGram) -> float:
        if self.__top is None or self.__bottom is None:
            raise ValueError("Model must be trained first")
        # print("a=", self.__top[tuple(context.as_deque())] / self.__bottom[tuple(context.as_deque())])
        return self.__top[tuple(context.as_deque())] / self.__bottom[tuple(context.as_deque())]

    def __compute_alpha_parameter(self, node: Node, ngram: Deque[str], top: Dict[Tuple[str], float], bottom: Dict[Tuple[str], float]) -> None:
        for token, child in node.children.items():
            ngram.append(token)
            n = len(ngram) + 1
            ngram2 = deepcopy(ngram)
            ngram2.popleft()
            for token2, child2 in child.children.items():
                ngram2.append(token2)
                context = NGram.from_words_list(ngram2)
                counts = self.__counter.get_count_n(context)
                if child2.value > 0 and child.value > 0:
                    top[tuple(ngram)] += self.__p_katz(child2.value, n, child.value)
                    if n > 2:
                        bottom[tuple(ngram)] += self.__p_katz(counts[-1], n, counts[-2])
                    else:
                        bottom[tuple(ngram)] += counts[-1] / self.__counter.get_total_count()
                ngram2.pop()
            top[tuple(ngram)] = 1 - top[tuple(ngram)]
            bottom[tuple(ngram)] = 1 - bottom[tuple(ngram)]
            if n < self.__counter.n:
                self.__compute_alpha_parameter(child, ngram, top, bottom)
            ngram.pop()
