import itertools
import logging
import time
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Tuple, Final, Set, Any, Optional

import numpy as np

import utils.log_math as lmath
from utils.counter.counters import Counter, FrequenciesCounter
from utils.ngram import NGram
from utils.counter.node import Node
from utils import arpafile
from utils.pyfoma_wfst import WFST


class AdditiveSmoothing:
    __eps: Final[int] = 1e-15

    def __init__(
            self,
            counter: Counter,
            vocabulary_size: int,
            addition: float = 0
    ) -> None:
        self.__counter = counter
        self.__vocabulary_size = vocabulary_size
        self.__addition = addition

    def get_probability(self, ngram: NGram) -> float:
        counts = self.__counter.get_count_n(ngram)
        favorable_count = counts[-1] + self.__addition
        value = counts[-2] if self.__counter.n > 1 else self.__counter.get_total_count()
        total_count = value + self.__addition * self.__vocabulary_size
        if total_count == 0:
            return -np.inf
        probability = favorable_count / total_count
        return -np.inf if probability < AdditiveSmoothing.__eps or total_count == 0 else np.log(probability)


class SimpleGoodTuringSmoothing:

    def __init__(self, counter: FrequenciesCounter, vocabulary: Set[str]) -> None:
        self.__counter = counter
        self.__vocabulary = vocabulary
        self.__sorted_frequencies = {k: self.__counter.frequencies[-1][k] for k in
                                     sorted(self.__counter.frequencies[-1]) if
                                     self.__counter.frequencies[-1][k] > 0}
        self.__coefficients = self.__linear_fit_from_dict()
        self.__probabilities: Dict[Tuple[str, ...], float] = {}
        self.__compute_probabilities()

    def __compute_probabilities(self) -> None:
        total_log_count = np.log(self.__counter.get_total_count(self.__counter.n))
        frequency_max, _ = next(reversed(self.__sorted_frequencies.items()))
        probabilities = {0: self.__get_log_zr(1) - total_log_count}
        for r in self.__sorted_frequencies.keys():
            log_new_r = np.log(r + 1) + self.__get_log_zr(r + 1) - self.__get_log_zr(r)
            probabilities[r] = log_new_r - total_log_count
        self.__normalize(probabilities)

    def __normalize(self, probabilities: Dict[int, Any]) -> None:
        for context in itertools.product(self.__vocabulary, repeat=self.__counter.n - 1):
            probabilities_sum = -np.inf
            for token in self.__vocabulary:
                ngram = context + (token,)
                count = self.__counter.get_count_n(NGram.from_words_list(ngram))[-1]
                probabilities_sum = lmath.log_sum(probabilities_sum, probabilities[count])
            for token in self.__vocabulary:
                ngram = context + (token,)
                count = self.__counter.get_count_n(NGram.from_words_list(ngram))[-1]
                self.__probabilities[ngram] = probabilities[count] - probabilities_sum

    def __get_log_zr(self, r):
        return self.__coefficients[1] + self.__coefficients[0] * np.log(r)

    def __linear_fit_from_dict(self):
        x, y = self.__preprocess_frequencies()
        A = np.vstack([x, np.ones_like(x)]).T
        return np.linalg.lstsq(A, y, rcond=None)[0]

    def __preprocess_frequencies(self):
        x, y, q, r, nr, t = [], [], 0, -1, -1, -1
        for key, value in self.__sorted_frequencies.items():
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
        return x, y

    def get_probability(self, ngram: NGram) -> float:
        return self.__probabilities[tuple(ngram.as_deque())]


class KatzSmoothing:
    __eps: Final[int] = 1e-15

    def __init__(
            self,
            counter: Optional[FrequenciesCounter],
            k: int = 5,
            reserved_probability: float = 0.001,
            path_to_arpa_file: Optional[str] = None
    ) -> None:
        self.__counter = counter
        self.__k = k
        self.__reserved_probability = reserved_probability
        self.__observed_ngrams_probabilities: Dict[int, Dict[str, float]] = defaultdict(dict)
        self.__alpha: Dict[str, float] = {}
        if path_to_arpa_file:
            self.__observed_ngrams_probabilities, self.__alpha = arpafile.read_arpa_lm(path_to_arpa_file)
            logging.log(logging.INFO, f"Katz language model was restored from ARPA file: {path_to_arpa_file}")
        else:
            self.__precompute_adjustment_values(self.__counter.root, NGram.empty())
            logging.log(logging.INFO, "Katz adjustments were successfully precomputed.")
            self.__precompute_probabilities(self.__counter.root, NGram.empty())
            save_lm_path = f'./language_model/KATZ_LM_{int(time.time())}.ARP'
            arpafile.write_arpa_lm(save_lm_path, self.__counter.n, (self.__observed_ngrams_probabilities, self.__alpha))
            logging.log(logging.INFO, f"Katz language model was saved to ARPA file: {save_lm_path}. You will be able to restore model from it later on.")

    def __precompute_probabilities(self, node: Node, ngram: NGram) -> None:
        for token, child in node.children.items():
            ngram.append(token)
            n = len(ngram)
            if child.value > 0:
                self.__observed_ngrams_probabilities[n][ngram.as_string()] = self.__p_katz(child.value, n, node.value) if n > 1\
                    else child.value / self.__counter.get_total_count() * (1 - self.__reserved_probability)
            self.__precompute_probabilities(child, ngram)
            ngram.shorten(direction='right')
            logging.log(logging.INFO, "Katz probabilities were successfully precomputed.")

    def __dr(self, r: int, i: int) -> float:
        if r > self.__k:
            return 1
        good_turing_r = (r + 1) * self.__get_frequencies_safe(i, r + 1) / self.__get_frequencies_safe(i, r)
        top = good_turing_r / r - (self.__k + 1) * self.__get_frequencies_safe(i, self.__k + 1) / \
              self.__get_frequencies_safe(i, 1)
        bottom = 1 - (self.__k + 1) * self.__get_frequencies_safe(i, self.__k + 1) / self.__get_frequencies_safe(i, 1)
        return self.__force_in_range(top / bottom)

    def __get_frequencies_safe(self, i: int, r: int):
        return self.__counter.frequencies[i][r] if r in self.__counter.frequencies[i] else KatzSmoothing.__eps

    # def get_probabilities(self, ngram: NGram) -> List[float]:
    #     n = len(ngram)
    #     probabilities = [0.0 for _ in range(n)]
    #     for i in range(n):
    #         probabilities[n - i - 1] = np.log(self.__get_probability_inner(deepcopy(ngram)))
    #         ngram.shorten(direction="right")
    #     return probabilities
    #
    # def get_probability(self, ngram: NGram) -> float:
    #     return np.log(self.__get_probability_inner(deepcopy(ngram)))

    def build_wfst(self) -> WFST:
        wfst = WFST(self.__counter.n)
        self.__wfst_insert_seen_ngrams(self.__counter.root, NGram.empty(), wfst)
        self.__wfst_insert_backoff(wfst)
        unknown_token = NGram.get_sys_token('unknown')
        wfst.add_state(unknown_token)
        wfst.add_arc(unknown_token, WFST.TRUE_EPS, (WFST.TRUE_EPS, WFST.TRUE_EPS), 0)
        wfst.add_arc(WFST.TRUE_EPS, unknown_token, (unknown_token, unknown_token), self.__reserved_probability)
        return wfst

    def __wfst_insert_backoff(self, wfst: WFST) -> None:
        for context, value in self.__alpha.items():
            ngram_context = NGram.from_words_list(context.split(' '))
            counts = self.__counter.get_count_n(ngram_context)
            n = len(ngram_context) + 1
            if counts[n - 2] == 0:
                self.__build_part(wfst, ngram_context, WFST.EPS, weight=1, is_backoff=True)
            else:
                self.__build_part(wfst, ngram_context, WFST.EPS, is_backoff=True)

    def __wfst_insert_seen_ngrams(self, node: Node, ngram: NGram, wfst: WFST, n: int = 1) -> None:
        for token, child in node.children.items():
            ngram.append(token)
            if child.value > 0:
                weight = self.__observed_ngrams_probabilities[n][ngram.as_string()]
                self.__build_part(wfst, ngram, ngram[-1], weight, is_backoff=False)
            self.__wfst_insert_seen_ngrams(child, ngram, wfst, n + 1)
            ngram.shorten(direction='right')

    def __build_part(self, wfst: WFST, ngram: NGram, label: str, weight: float = None, is_backoff=False) -> None:
        context = deepcopy(ngram)
        cur_state = context.as_string() if is_backoff else context.shorten(direction="right").as_string()
        weight = weight or self.__get_alpha(context)
        if not cur_state:
            cur_state = WFST.TRUE_EPS
        if not wfst.has_state(cur_state):
            wfst.add_state(cur_state)
            if cur_state.endswith(NGram.get_sys_token('end')):
                wfst.mark_as_final(cur_state, 0)
        if is_backoff:
            next_state = context.shorten().as_string()
        elif len(ngram) == self.__counter.n:
            next_state = deepcopy(ngram).shorten().as_string()
        else:
            next_state = ngram.as_string()
        if not next_state:
            next_state = WFST.TRUE_EPS
        if cur_state == WFST.TRUE_EPS and next_state == NGram.get_sys_token('start')\
                or label == NGram.get_sys_token('start'):
            return
        no_state_flag = not wfst.has_state(next_state)
        if no_state_flag or not wfst.has_arc(cur_state, next_state):
            state = wfst.add_state(next_state)
            if not wfst.is_final(cur_state):
                wfst.add_arc(cur_state, state.name, (label, label), weight)
        if next_state.endswith(NGram.get_sys_token('end')):
            wfst.mark_as_final(next_state, 0)

    def __p_katz(self, r: int, n: int, cnt: int) -> float:
        return self.__dr(r, n - 1) * r / cnt

    def __get_alpha(self, context: NGram) -> float:
        return self.__alpha[context.as_string()]

    def __force_in_range(self, x: float, floor: int = __eps, ceil: int = 1) -> float:
        return max(min(x, ceil), floor)

    def __precompute_adjustment_values(self, node: Node, ngram: NGram) -> None:
        for token, child in node.children.items():
            ngram.append(token)
            n = len(ngram) + 1
            context = deepcopy(ngram)
            context.shorten()
            top, bottom = 0, 0
            for token2, child2 in child.children.items():
                context.append(token2)
                counts = self.__counter.get_count_n(context)
                if child2.value > 0:
                    top += self.__p_katz(child2.value, n, child.value)
                    if n > 2:
                        bottom += self.__p_katz(counts[-1], n - 1, counts[-2])
                    else:
                        bottom += counts[-1] / self.__counter.get_total_count() * (1 - self.__reserved_probability)
                context.shorten(direction='right')
            self.__alpha[ngram.as_string()] = self.__force_in_range(1 - top) / self.__force_in_range(1 - bottom)
            if n < self.__counter.n:
                self.__precompute_adjustment_values(child, ngram)
            ngram.shorten(direction='right')
