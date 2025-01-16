import math
from abc import ABC, abstractmethod
from typing import Optional, List, Union, Deque, cast, Callable, Final, Any, Tuple

import numpy as np
from overrides import override

from language_model.counters import Counter, FrequenciesCounter
from language_model.ngram import NGram
from language_model.smoothings import AdditiveSmoothing, SimpleGoodTuringSmoothing, KatzSmoothing


class BaseLM(ABC):
    UNTRAINED_MODEL_EXCEPTION_MESSAGE: Final[str] = "Model must be trained first."

    def __init__(
            self,
            n: int,
            corpus_path: str,
            counter: Counter,
            is_acceptable_character: Optional[Callable[[str], bool]] = None
    ) -> None:
        self._n = n
        self._corpus_path = corpus_path
        self._counter = counter
        self._model_probabilities = None
        self.__is_acceptable_character = is_acceptable_character or BaseLM.__is_acceptable_character
        if self._n == 1:
            self.__vocabulary = {NGram.get_sys_token('unknown')}
        else:
            self.__vocabulary = {NGram.get_sys_token('start'), NGram.get_sys_token('end'),
                                 NGram.get_sys_token('unknown')}

    @staticmethod
    def __is_acceptable_character(character: str) -> bool:
        return character.isalpha() or character in {'<', '>', '/'}

    def __corpus_reader(self, ngram: NGram, corpus_path: str, initial_character: str, case: str = 'count') -> Tuple[float, float]:
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
                    self.__vocabulary.add(word)
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
        return self.__vocabulary

    def oov_rate(self, path_to_test_corpus: str) -> float:
        if self._model_probabilities is None:
            raise ValueError(BaseLM.UNTRAINED_MODEL_EXCEPTION_MESSAGE)
        counter = Counter(self._n)
        self._count_all_ngrams(path_to_test_corpus, counter)
        difference, count = self._counter.difference_and_count(counter)
        return difference / count * 100

    def perplexity(self, path_to_test_corpus: str, initial_character: str = '$') -> float:
        if self._model_probabilities is None:
            raise ValueError(BaseLM.UNTRAINED_MODEL_EXCEPTION_MESSAGE)
        ngram = NGram(self._n)
        perplexity, n = self.__corpus_reader(ngram, path_to_test_corpus, initial_character, case='perplexity')
        ngram.update('</s>')
        return np.exp(perplexity * 1. / (n + 1))

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError('Must be implemented in subclass.')

    def get_probability(self, ngram: Union[NGram, List[str], Deque[str]]) -> List[float]:
        if self._model_probabilities is None:
            raise ValueError(BaseLM.UNTRAINED_MODEL_EXCEPTION_MESSAGE)
        if len(ngram) != self._n:
            raise ValueError(f'{self._n}-gram model expected ngram of length {self._n}, but len={len(ngram)} given.')
        for i, item in enumerate(ngram):
            if item not in self.__vocabulary:
                ngram[i] = NGram.get_sys_token('unknown')
        if not isinstance(ngram, NGram):
            ngram = NGram.from_words_list(ngram)
        return self._model_probabilities.get_probability(ngram)


class NoSmoothingLM(BaseLM):

    def __init__(
            self,
            n: int,
            corpus_path: str,
            is_acceptable_character: Optional[Callable[[str], bool]] = None
    ) -> None:
        super().__init__(n, corpus_path, Counter(n), is_acceptable_character)

    @override
    def train(self):
        self._count_all_ngrams(self._corpus_path, self._counter)
        vocabulary_size = len(self.vocabulary)
        self._model_probabilities = AdditiveSmoothing(self._counter, vocabulary_size=vocabulary_size)


class LaplaceSmoothingLM(BaseLM):

    def __init__(
            self,
            n: int,
            corpus_path: str,
            is_acceptable_character: Optional[Callable[[str], bool]] = None
    ) -> None:
        super().__init__(n, corpus_path, Counter(n), is_acceptable_character)

    @override
    def train(self) -> None:
        self._count_all_ngrams(self._corpus_path, self._counter)
        vocabulary_size = len(self.vocabulary)
        self._model_probabilities = AdditiveSmoothing(self._counter, addition=1, vocabulary_size=vocabulary_size)


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
        super().__init__(n, corpus_path, Counter(n), is_acceptable_character)
        self.__addition = addition

    @override
    def train(self) -> None:
        self._count_all_ngrams(self._corpus_path, self._counter)
        vocabulary_size = len(self.vocabulary)
        self._model_probabilities = AdditiveSmoothing(self._counter, vocabulary_size, self.__addition)


class SimpleGoodTuringSmoothingLM(BaseLM):

    def __init__(
            self,
            n: int,
            corpus_path: str,
            is_acceptable_character: Optional[Callable[[str], bool]] = None
    ) -> None:
        super().__init__(n, corpus_path, FrequenciesCounter(n), is_acceptable_character)

    @override
    def train(self) -> None:
        self._count_all_ngrams(self._corpus_path, self._counter)
        self._model_probabilities = SimpleGoodTuringSmoothing(cast(FrequenciesCounter, self._counter), self.vocabulary)


class KatzSmoothingLM(BaseLM):
    eps: Final[float] = 1e-9

    def __init__(
            self,
            n: int,
            corpus_path: str,
            k: int = 5,
            reserved_probability: float = 0.001,
            is_acceptable_character: Optional[Callable[[str], bool]] = None
    ) -> None:
        if reserved_probability >= 1 - KatzSmoothingLM.eps or reserved_probability <= -KatzSmoothingLM.eps:
            raise ValueError("Probability reserved for unknown tokens must be in range (0, 1)")
        super().__init__(n, corpus_path, FrequenciesCounter(n), is_acceptable_character)
        self.__k = k
        self.__unknown_probability = reserved_probability

    @override
    def train(self) -> None:
        self._count_all_ngrams(self._corpus_path, self._counter)
        self._model_probabilities = KatzSmoothing(cast(FrequenciesCounter, self._counter), self.__k,
                                                  self.__unknown_probability)


def xxx(cc: str) -> bool:
    return cc.isalpha() or cc.isdigit() or cc == '_'


if __name__ == '__main__':
    path = 'C:/Users/PeterA/Desktop/vkr/test/1.txt'
    path2 = 'C:/Users/PeterA/Desktop/vkr/test/2.txt'

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
