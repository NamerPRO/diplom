from abc import ABC, abstractmethod
from typing import List, TypeVar

import numpy as np
from overrides import overrides

import utility.log_math as lmath
from language_model.counters import Counter, FrequenciesCounter
from language_model.ngram import NGram
from language_model.node import Node

T = TypeVar('T', bound=Counter)


class Smoothing(ABC):

    def __init__(self, counter: T):
        self._counter = counter

    # @abstractmethod
    # def _compute_probabilities(self) -> None:
    #     pass

    @abstractmethod
    def get_probabilities(self, ngram: NGram) -> List[float]:
        pass


# add so it calculates probability for unknown words when additive smoothing but not standard
class AdditiveSmoothing(Smoothing):

    def __init__(self, counter: Counter, addition: int = 0) -> None:
        super().__init__(counter)
        self.__root = Node()
        self.__unseen_words = [
            np.pow(self._counter.unique_ngrams_count[0], i + 1) - self._counter.unique_ngrams_count[i] for i in
            range(self._counter.n)]
        self.__addition = addition
        # self._compute_probabilities()
        # self.__unknown_token_probabilities: List[float] = [-np.inf if addition == 0 else np.log(
        #     self.__addition / ((self.__addition + 1) * self._counter.get_total_count(i + 1))) for i in
        #                                                    range(self._counter.n)]

    # @overrides
    # def _compute_probabilities(self) -> None:
    #     self.__compute_inner_probabilities(self._counter.root, self.__root, 0)
    #     total_tokens_count = self._counter.get_total_count()
    #     for token, child in self.__root.children.items():
    #         if self._counter.root[token].value > 0:
    #             child.value = np.log((self._counter.root[token].value + self.__addition) / (
    #                         total_tokens_count + self.__addition * self.__unseen_words[0]))
    #         else:
    #             child.value = -np.inf

    # def __compute_inner_probabilities(self, counter_node: Node, probabilities_node: Node, i: int) -> None:
    #     for token, child in counter_node.children.items():
    #         probabilities_node[token] = Node()
    #         self.__compute_inner_probabilities(child, probabilities_node[token], i + 1)
    #         if child.value > 0:  # if counter_node.value > 0 and child.value > 0:
                # value = counter_node.value if counter_node.value > 0 else 0
    #            value = 0 if counter_node.value == -1 else counter_node.value
                # probabilities_node[token].value = np.log((child.value + self.__addition) / (counter_node.value + self.__addition * self.__unseen_words[i]))
    #           probabilities_node[token].value = np.log( # <s> a <s>
    #           (child.value + self.__addition) / (value + self.__addition * self.__unseen_words[i]))
            # elif counter_node.value > 0: <s> <s> apple
            #     value = counter_node.value if counter_node.value > 0 else 0    a </s> </s> </s>
            #     probabilities_node[token].value = -np.inf if self.__addition == 0 else np.log(self.__addition / (counter_node.value + self.__addition * self.__unseen_words[i]))
    #       else:
    #           probabilities_node[token].value = -np.inf

    @overrides
    def get_probabilities(self, ngram: NGram) -> List[float]:
        counts = self._counter.get_count_n(ngram)
        probabilities: List[float] = [0 for _ in range(len(ngram))]
        probabilities[0] = np.log((counts[0] + self.__addition) / self._counter.get_total_count())
        for i in range(1, len(ngram)):
            value = 0 if counts[i - 1] == -1 else counts[i - 1]
            favorable_count = counts[i] + self.__addition
            total_count = value + self.__addition * self.__unseen_words[i]
            if favorable_count == 0 or total_count == 0:
                probabilities[i] = -np.inf
            else:
                probabilities[i] = np.log(favorable_count / total_count)
        return probabilities

    # @overrides
    # def get_probabilities(self, ngram: NGram) -> List[float]:
    #     probabilities: List[float] = [0 for i in range(len(ngram))]
    #     node = self.__root
    #     for i in range(len(ngram)):
    #         if ngram[i] not in node.children:
    #             for j in range(i, len(ngram)):
    #                 counts = self._counter.get_count_n(ngram)
    #                 probabilities[j] =
    #             return probabilities
    #         probabilities[i] = node[ngram[i]].value
    #         node = node[ngram[i]]
    #     return probabilities


class SimpleGoodTuringSmoothing(Smoothing):

    def __init__(self, counter: FrequenciesCounter) -> None:
        super().__init__(counter)
        self.__coefficients = self.__linear_fit_from_dict()
        self.__probabilities: List[List[float]] = [[] for _ in range(counter.n)]
        self._compute_probabilities()

    def _compute_probabilities(self) -> None:
        probability_sums = [-np.inf for i in range(self._counter.n)]
        total_log_count = np.log(self._counter.get_total_count())
        for i in range(self._counter.n):
            frequency_max, _ = next(reversed(self._counter.frequencies[i].items()))
            self.__probabilities[i].append(self.__get_log_zr(i, 1) - total_log_count)
            probability_sums[i] = lmath.log_sum(probability_sums[i], self.__probabilities[i][0])
            for r in range(1, frequency_max + 1):
                log_new_r = np.log(r + 1) + self.__get_log_zr(i, r + 1) - self.__get_log_zr(i, r)
                smoothed_probability = log_new_r - total_log_count
                self.__probabilities[i].append(smoothed_probability)
                probability_sums[i] = lmath.log_sum(probability_sums[i], smoothed_probability)
        self.__normalize(probability_sums)

    def __normalize(self, probabilities_sums):
        for i in range(self._counter.n):
            unseen_ngrams_count = np.pow(self._counter.unique_ngrams_count[0], i + 1) - \
                                  self._counter.unique_ngrams_count[i]
            self.__probabilities[i][0] -= -np.inf if unseen_ngrams_count == 0 else probabilities_sums[i] + np.log(
                unseen_ngrams_count)
            for j in range(1, len(self.__probabilities[i])):
                self.__probabilities[i][j] -= probabilities_sums[i] + np.log(self._counter.frequencies[i][j])

    def __get_log_zr(self, i, r):
        return self.__coefficients[i][1] + self.__coefficients[i][0] * np.log(r)

    def __linear_fit_from_dict(self):
        coefficients = []
        normalized_frequencies = self.__normalize_frequencies()
        for i in range(self._counter.n):
            keys = normalized_frequencies[i][0]
            values = normalized_frequencies[i][1]
            A = np.vstack([keys, np.ones_like(keys)]).T
            c = np.linalg.lstsq(A, values, rcond=None)[0]
            coefficients.append(c)
        return coefficients

    def __normalize_frequencies(self):
        normalized_frequencies = []
        for i in range(self._counter.n):
            x = []
            y = []
            q, r, nr, t = 0, -1, -1, -1
            for key, value in self._counter.frequencies[i].items():
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

    @overrides
    def get_probabilities(self, ngram: NGram) -> List[float]:
        occurrences = self._counter.get_count_n(ngram)
        probabilities: List[float] = [-np.inf for _ in range(len(ngram))]
        for i in range(len(ngram)):
            probabilities[i] = self.__probabilities[i][occurrences[i]]
        return probabilities
