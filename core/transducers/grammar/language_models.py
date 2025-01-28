import itertools
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Optional, cast, Callable, Final, Tuple, Dict, Any

import numpy as np
from overrides import override

import utils.log_math as lmath
from utils import arpafile
from utils.counter.counters import Counter, FrequenciesCounter
from utils.counter.node import Node
from utils.ngram import NGram
from utils.wfst import WFST


def get_probability_additive(ngram: NGram, counter: Counter, addition: int = 0, vocabulary_size: int = 0) -> float:
    counts = counter.get_count_n(ngram)
    favorable_count = counts[-1] + addition
    value = counts[-2] if counter.n > 1 else counter.get_total_count()
    total_count = value + addition * vocabulary_size
    if total_count == 0:
        return np.inf
    probability = favorable_count / total_count
    return np.inf if probability < 1e-15 or total_count == 0 else -np.log(probability)


class BaseLM(ABC):
    def __init__(
            self,
            n: int,
            counter: Counter,
            is_acceptable_character: Optional[Callable[[str], bool]] = None
    ) -> None:
        self._n = n
        self._counter = counter
        self._model_probabilities = None
        self.__is_acceptable_character = is_acceptable_character or BaseLM.__is_acceptable_character
        if self._n == 1:
            self._vocabulary = {NGram.get_sys_token('unknown')}
        else:
            self._vocabulary = {NGram.get_sys_token('start'), NGram.get_sys_token('end'),
                                NGram.get_sys_token('unknown')}

    @staticmethod
    def __is_acceptable_character(character: str) -> bool:
        return character.isalpha() or character in {'<', '>', '/', '.'}

    def __corpus_reader(self, ngram: NGram, corpus_path: str, initial_character: str, case: str = 'count') -> Tuple[
        float, float]:
        perplexity, n = 0.0, 0
        with open(corpus_path, 'r', encoding='utf8') as f:
            char = initial_character
            while char:
                while not self.__is_acceptable_character(char) and char:
                    char = f.read(1)
                if char:
                    word = char
                    char = f.read(1)
                    while self.__is_acceptable_character(char):
                        word += char
                        char = f.read(1)
                    word = word.lower()
                    self._vocabulary.add(word)
                    ngram.update(word)
                    if case == 'count':
                        self._counter.count_n(ngram)
                    elif case == 'perplexity':
                        perplexity -= self.get_probability(ngram)
                        n += 1
        return (0, 0) if case == 'count' else (perplexity, n)

    def _count_all_ngrams(self, corpus_path: str, ngrams_counter: Counter, initial_character: str = '$') -> None:
        ngram = NGram(self._n)
        self.__corpus_reader(ngram, corpus_path, initial_character, case='count')
        for i in range(self._n):
            ngram.update(NGram.get_sys_token(name='end'))
            ngrams_counter.count_n(ngram)

    @property
    def vocabulary(self):
        return self._vocabulary

    def oov_rate(self, path_to_test_corpus: str) -> float:
        counter = Counter(self._n)
        self._count_all_ngrams(path_to_test_corpus, counter)
        difference, count = self._counter.difference_and_count(counter)
        return difference / count * 100

    def perplexity(self, path_to_test_corpus: str, initial_character: str = '$') -> float:
        ngram = NGram(self._n)
        perplexity, n = self.__corpus_reader(ngram, path_to_test_corpus, initial_character, case='perplexity')
        ngram.update('</s>')
        return np.exp(perplexity * 1. / (n + 1))

    def _prepare_ngram(self, ngram: NGram) -> NGram:
        if len(ngram) != self._n:
            raise ValueError(f'{self._n}-gram model expected ngram of length {self._n}, but len={len(ngram)} given.')
        for i in range(len(ngram)):
            if ngram[i] not in self._vocabulary:
                ngram[i] = NGram.get_sys_token('unknown')
        return ngram

    @abstractmethod
    def get_probability(self, ngram: NGram) -> float:
        pass


class NoSmoothingLM(BaseLM):

    def __init__(
            self,
            n: int,
            corpus_path: str,
            is_acceptable_character: Optional[Callable[[str], bool]] = None
    ) -> None:
        super().__init__(n, Counter(n), is_acceptable_character)
        self._count_all_ngrams(corpus_path, self._counter)

    @override
    def get_probability(self, ngram: NGram) -> float:
        prepared_ngram = self._prepare_ngram(ngram)
        return get_probability_additive(prepared_ngram, self._counter, addition=0, vocabulary_size=len(self.vocabulary))


class LaplaceSmoothingLM(BaseLM):

    def __init__(
            self,
            n: int,
            corpus_path: str,
            is_acceptable_character: Optional[Callable[[str], bool]] = None
    ) -> None:
        super().__init__(n, Counter(n), is_acceptable_character)
        self._count_all_ngrams(corpus_path, self._counter)

    @override
    def get_probability(self, ngram: NGram) -> float:
        prepared_ngram = self._prepare_ngram(ngram)
        return get_probability_additive(prepared_ngram, self._counter, addition=1, vocabulary_size=len(self.vocabulary))


class AdditiveSmoothingLM(BaseLM):

    def __init__(
            self,
            n: int,
            corpus_path: str,
            addition: int,
            is_acceptable_character: Optional[Callable[[str], bool]] = None
    ) -> None:
        if addition <= 0:
            raise ValueError(
                'Addition size must be positive. If you intend to use addition=0 consider using NoSmoothingLM.')
        super().__init__(n, Counter(n), is_acceptable_character)
        self._count_all_ngrams(corpus_path, self._counter)
        self.__addition = addition

    @override
    def get_probability(self, ngram: NGram) -> float:
        prepared_ngram = self._prepare_ngram(ngram)
        return get_probability_additive(prepared_ngram, self._counter, self.__addition,
                                        vocabulary_size=len(self.vocabulary))


class SimpleGoodTuringSmoothingLM(BaseLM):

    def __init__(
            self,
            n: int,
            corpus_path: str,
            is_acceptable_character: Optional[Callable[[str], bool]] = None
    ) -> None:
        super().__init__(n, FrequenciesCounter(n), is_acceptable_character)
        self._count_all_ngrams(corpus_path, self._counter)
        frequencies_counter = cast(FrequenciesCounter, self._counter)
        self.__sorted_frequencies = {k: frequencies_counter.frequencies[-1][k] for k in
                                     sorted(frequencies_counter.frequencies[-1]) if
                                     frequencies_counter.frequencies[-1][k] > 0}
        self.__coefficients = self.__linear_fit_from_dict()
        self.__probabilities: Dict[Tuple[str, ...], float] = {}
        self.__compute_probabilities()

    def __compute_probabilities(self) -> None:
        total_log_count = np.log(self._counter.get_total_count(self._counter.n))
        frequency_max, _ = next(reversed(self.__sorted_frequencies.items()))
        probabilities = {0: self.__get_log_zr(1) - total_log_count}
        for r in self.__sorted_frequencies.keys():
            log_new_r = np.log(r + 1) + self.__get_log_zr(r + 1) - self.__get_log_zr(r)
            probabilities[r] = log_new_r - total_log_count
        self.__normalize(probabilities)

    def __normalize(self, probabilities: Dict[int, Any]) -> None:
        for context in itertools.product(self._vocabulary, repeat=self._counter.n - 1):
            probabilities_sum = -np.inf
            for token in self._vocabulary:
                ngram = context + (token,)
                count = self._counter.get_count_n(NGram.from_words_list(ngram))[-1]
                probabilities_sum = lmath.log_sum(probabilities_sum, probabilities[count])
            for token in self._vocabulary:
                ngram = context + (token,)
                count = self._counter.get_count_n(NGram.from_words_list(ngram))[-1]
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

    @override
    def get_probability(self, ngram: NGram) -> float:
        return -self.__probabilities[tuple(ngram.as_deque())]


class KatzSmoothingLM(BaseLM):
    EPS: Final[float] = 1e-9
    BACKOFF_DISAMBIGUITY_SYMBOL: Final[str] = '#0'

    def __init__(
            self,
            n: int,
            is_acceptable_character: Optional[Callable[[str], bool]] = None,
            counter: Optional[Counter] = None
    ) -> None:
        super().__init__(n, counter, is_acceptable_character)
        self.__wfst: Optional[WFST] = None
        self.__observed_ngrams_probabilities: Optional[Dict[int, Dict[str, float]]] = None
        self.__alpha: Optional[Dict[str, float]] = None
        self.__k = None
        self.__reserved_probability = None

    @classmethod
    def from_train_corpus(
            cls,
            n: int,
            corpus_path: str,
            k: int = 5,
            reserved_probability: float = 0.01,
            is_acceptable_character: Optional[Callable[[str], bool]] = None
    ) -> 'KatzSmoothingLM':
        obj = KatzSmoothingLM(n, is_acceptable_character, FrequenciesCounter(n))
        obj.__k = k
        obj.__reserved_probability = reserved_probability
        obj._count_all_ngrams(corpus_path, obj._counter)
        obj._corpus_path = corpus_path
        obj.__observed_ngrams_probabilities = defaultdict(dict)
        obj.__alpha = {}
        logging.info("Precomputations for Katz smoothing started...")
        obj.__precompute_adjustment_values(obj._counter.root, NGram.empty())
        obj.__precompute_probabilities(obj._counter.root, NGram.empty())
        obj.__reserved_probability = -np.log(obj.__reserved_probability)
        save_lm_path = f'./language_model/KATZ_LM_{int(time.time())}.ARP'
        arpafile.write_arpa_lm(save_lm_path, obj._n,
                               (obj.__observed_ngrams_probabilities, obj.__alpha, obj.__reserved_probability))
        logging.log(logging.INFO,
                    f"All done! Katz language model was saved to ARPA file: {save_lm_path}. You will be able to restore model from it later on.")
        return obj

    @classmethod
    def from_arpa_file(
            cls,
            arpa_file_path: str
    ) -> 'KatzSmoothingLM':
        probabilities, alpha, vocabulary, n, reserved_probability = arpafile.read_arpa_lm(arpa_file_path)
        obj = KatzSmoothingLM(n)
        obj.__observed_ngrams_probabilities = probabilities
        obj.__alpha = alpha
        obj._vocabulary = vocabulary
        obj._arpa_file_path = arpa_file_path
        obj.reserved_probability = -np.log(reserved_probability)
        logging.info(f"Katz language model was restored from ARPA file: {arpa_file_path}")
        return obj

    def build_wfst(self) -> WFST:
        if self.__wfst is None:
            logging.info("Building WFST...")
            self.__wfst = self.__build_wfst_inner()
            self.vocabulary.remove(NGram.get_sys_token('start'))
            self.vocabulary.remove(NGram.get_sys_token('end'))
            self.vocabulary.add(KatzSmoothingLM.BACKOFF_DISAMBIGUITY_SYMBOL)
            logging.info(
                "WFST was successfully built. These logs will not appear again, because same WFST will be returned on next method call.")
        return self.__wfst

    def __precompute_probabilities(self, node: Node, ngram: NGram) -> None:
        for token, child in node.children.items():
            ngram.append(token)
            n = len(ngram)
            if child.value > 0:
                self.__observed_ngrams_probabilities[n][ngram.as_string()] = -np.log(self.__p_katz(child.value, n,
                                                                                                   node.value)) if n > 1 \
                    else -np.log(child.value / self._counter.get_total_count() * (1 - self.__reserved_probability))
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
        frequencies_counter = cast(FrequenciesCounter, self._counter)
        return frequencies_counter.frequencies[i][r] if r in frequencies_counter.frequencies[i] else KatzSmoothingLM.EPS

    def __build_wfst_inner(self) -> WFST:
        wfst = WFST(NGram(self._n - 1).as_string())
        self.__wfst_insert_seen_ngrams(wfst)
        self.__wfst_insert_backoff(wfst)
        unknown_token = NGram.get_sys_token('unknown')
        wfst.add_state(unknown_token)
        wfst.add_arc(unknown_token, WFST.EPS, (WFST.EPS, WFST.EPS), 0)
        wfst.add_arc(WFST.EPS, unknown_token, (unknown_token, unknown_token), self.__reserved_probability)
        wfst.remove_epsilons()
        return wfst

    def __wfst_insert_backoff(self, wfst: WFST) -> None:
        for context, value in self.__alpha.items():
            ngram_context = NGram.from_words_list(context.split(' '))
            self.__build_part(wfst, ngram_context, (KatzSmoothingLM.BACKOFF_DISAMBIGUITY_SYMBOL, WFST.EPS),
                              is_backoff=True)

    def __wfst_insert_seen_ngrams(self, wfst: WFST) -> None:
        for n in self.__observed_ngrams_probabilities:
            for ngram_str, weight in self.__observed_ngrams_probabilities[n].items():
                ngram = NGram.from_words_list(ngram_str.split(' '))
                self.__build_part(wfst, ngram, ngram[-1], weight, is_backoff=False)

    def __build_part(self, wfst: WFST, ngram: NGram, label: str | Tuple[str, str], weight: float = None,
                     is_backoff=False) -> None:
        context = deepcopy(ngram)
        cur_state = context.as_string() if is_backoff else context.shorten(direction="right").as_string()
        weight = weight or self.__get_alpha(context)
        if not cur_state:
            cur_state = WFST.EPS
        if not wfst.has_state(cur_state):
            wfst.add_state(cur_state)
            if cur_state.endswith(NGram.get_sys_token('end')):
                wfst.mark_as_final(cur_state, 0)
        if is_backoff:
            next_state = context.shorten().as_string()
        elif len(ngram) == self._n:
            next_state = deepcopy(ngram).shorten().as_string()
        else:
            next_state = ngram.as_string()
        if not next_state:
            next_state = WFST.EPS
        if cur_state == WFST.EPS and next_state == NGram.get_sys_token('start') \
                or label == NGram.get_sys_token('start'):
            return
        no_state_flag = not wfst.has_state(next_state)
        if no_state_flag or not wfst.has_arc(cur_state, next_state):
            state = wfst.add_state(next_state)
            if not wfst.is_final(cur_state):
                if label == NGram.get_sys_token('end'):
                    label = WFST.EPS
                wfst.add_arc(cur_state, state, (label, label) if isinstance(label, str) else label, weight)
        if next_state.endswith(NGram.get_sys_token('end')):
            wfst.mark_as_final(next_state, 0)

    def __p_katz(self, r: int, n: int, cnt: int) -> float:
        return self.__dr(r, n - 1) * r / cnt

    def __get_alpha(self, context: NGram) -> float:
        key = context.as_string()
        return self.__alpha[key] if key in self.__alpha else 0

    def __force_in_range(self, x: float, floor: int = EPS, ceil: int = 1) -> float:
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
                counts = self._counter.get_count_n(context)
                if child2.value > 0:
                    top += self.__p_katz(child2.value, n, child.value)
                    if n > 2:
                        bottom += self.__p_katz(counts[-1], n - 1, counts[-2])
                    else:
                        bottom += counts[-1] / self._counter.get_total_count() * (1 - self.__reserved_probability)
                context.shorten(direction='right')
            self.__alpha[ngram.as_string()] = -np.log(
                self.__force_in_range(1 - top) / self.__force_in_range(1 - bottom))
            if n < self._n:
                self.__precompute_adjustment_values(child, ngram)
            ngram.shorten(direction='right')

    @override
    def get_probability(self, ngram: NGram) -> float:
        prepared_ngram = self._prepare_ngram(ngram)
        return self.__get_probability_inner(prepared_ngram)

    def __get_probability_inner(self, ngram: NGram) -> float:
        n = len(ngram)
        if n == 1 and ngram[0] == NGram.get_sys_token('unknown'):
            return self.__reserved_probability
        if ngram.as_string() in self.__observed_ngrams_probabilities[n]:
            return self.__observed_ngrams_probabilities[n][ngram.as_string()]
        context = deepcopy(ngram)
        context.shorten(direction="right")
        ngram.shorten()
        return self.__get_alpha(context) + self.__get_probability_inner(ngram)


def xxx(cc: str) -> bool:
    return cc.isalpha() or cc.isdigit() or cc == '_' or cc == '.'


if __name__ == '__main__':
    path = 'C:/Users/PeterA/Desktop/vkr/test/corpus.txt'
    path2 = 'C:/Users/PeterA/Desktop/vkr/test/2.txt'
    arpa = './language_model/katz_lm_1737559161.ARP'

    # gt = SimpleGoodTuringSmoothingLM(
    #     n=2,
    #     corpus_path='C:/Users/PeterA/Desktop/vkr/test/corpus.txt',
    #     is_acceptable_character=xxx
    # )
    # print(gt.get_probability(NGram.from_words_list(['you', 'you'])))

    model = KatzSmoothingLM.from_arpa_file(
        arpa_file_path="./language_model/KATZ_LM_1737926250.ARP",
    )
    print(model.vocabulary)

    # model = KatzSmoothingLM.from_train_corpus(
    #     n=2,
    #     corpus_path='C:/Users/PeterA/Desktop/vkr/test/corpus.txt',
    #     k=5,
    #     reserved_probability=0.1,
    #     is_acceptable_character=xxx
    # )
    prob = model.get_probability(NGram.from_words_list(['you', 'cake']))
    print(prob)
    grammar_wfst = model.build_wfst()
    grammar_wfst.view()

    exit(0)

    # model = KatzSmoothingLM.from_arpa_file(
    #     n=3,
    #     arpa_file_path="./language_model/KATZ_LM_1737650077.ARP",
    #     k=5,
    #     reserved_probability=0.1,
    #     is_acceptable_character=xxx
    # )

    # model.train()
    print(model.vocabulary)
    wfst = model.build_wfst()
    wfst.view()
    time.sleep(1)
    wfst.determinize()
    wfst.minimize()
    wfst.view()

    exit(0)

    # model0 = SimpleGoodTuringSmoothingLM(n=3, corpus_path=path, is_acceptable_character=xxx)
    # model0.train()

    # print(model0.vocabulary)
    # print(np.exp(model0.get_probability(['word_1', 'word_2', 'word_3'])))
    # print(np.exp(model0.get_probability(['word_1', 'word_10', 'word_3'])))
    # print(np.exp(model0.get_probability(['word_10', 'word_20', 'word_3'])))
    # print(np.exp(model0.get_probability(['word_10', 'word_5', 'word_7'])))
    # print(np.exp(model0.get_probability(['word_2', 'word_2', 'word_3'])))
    # print(np.exp(model0.get_probability(['word_4', 'word_2', 'word_5'])))
    # print(np.exp(model0.get_probability(['word_8', 'word_7', 'word_1'])))

    # voc = model0.vocabulary

    # ind = 0
    # for i in voc:
    #     for j in voc:
    #         probs = 0
    #         for k in voc:
    #             lp = model0.get_probability([i, j, k])
    #             print(f"print(np.exp(model0.get_probability(['{i}', '{j}', '{k}']))) = {np.exp(lp)}")
    #             probs += np.exp(lp)
    #         print(ind, '=', probs, '|', [i, j, k])
    #         if not math.isclose(probs, 1):
    #             print("...")
    #         ind += 1

    modelx = KatzSmoothingLM(n=3, corpus_path=path, is_acceptable_character=xxx, reserved_probability=0.01)
    modelx.train()
    print(f'perplexity={modelx.perplexity(path2)}')

    print(modelx.vocabulary)
    print(np.exp(modelx.get_probability(['a', 'word_3', 'c'])))
    print(np.exp(modelx.get_probability(['a', 'word_3', 'word_3'])))
    print(np.exp(modelx.get_probability(['word_1', 'b', 'c'])))
    print(np.exp(modelx.get_probability(['word_1', 'word_2', 'c'])))
    print(np.exp(modelx.get_probability(['a', 'b', 'word_3'])))
    print(np.exp(modelx.get_probability(['word_1', 'b', 'word_3'])))
    print('===')

    print(np.exp(modelx.get_probability(['word_1', 'word_2', 'word_3'])))
    print(np.exp(modelx.get_probability(['word_1', 'word_10', 'word_3'])))
    print(np.exp(modelx.get_probability(['word_10', 'word_20', 'word_3'])))
    print(np.exp(modelx.get_probability(['word_10', 'word_5', 'word_7'])))
    print(np.exp(modelx.get_probability(['word_2', 'word_2', 'word_3'])))
    print(np.exp(modelx.get_probability(['word_4', 'word_2', 'word_5'])))
    print(np.exp(modelx.get_probability(['word_8', 'word_7', 'word_1'])))

    exit()

    model1 = AdditiveSmoothingLM(n=3, corpus_path=path, addition=1)
    model1.train()
    voc = model1.vocabulary
    print(voc)
    print(np.exp(model1.get_probability(['apple', 'apple', 'apple'])))
    print(np.exp(model1.get_probability(['apple', '</s>', 'apple'])))

    model2 = KatzSmoothingLM(n=3, corpus_path=path, k=5, unknown_probability=0.001)
    model2.train()
    voc = model2.vocabulary
    print(voc)
    print(np.exp(model2.get_probability(['apple', 'orange', 'potatoes'])))

    ind = 0
    for i in voc:
        for j in voc:
            probs = 0
            for k in voc:
                lp = model0.get_probability([i, j, k])
                print(f"print(np.exp(model0.get_probability(['{i}', '{j}', '{k}']))) = {np.exp(lp)}")
                probs += np.exp(lp)
            print(ind, '=', probs, '|', [i, j, k])
            if not math.isclose(probs, 1):
                print("...")
            ind += 1

    # print(model2.oov_rate(path2))
