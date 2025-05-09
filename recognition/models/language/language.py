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


def get_probability_additive(ngram: NGram, counter: Counter, addition: int = 0, vocabulary_size: int = 0) -> float:
    """
    Метод вычисляющий вероятность n-граммы согласно принципу аддитивного сглаживания.
    Подходит также для сглаживания по Лапласу, так как такое сглаживание является
    частным случаем аддитивного, и для метода без сглаживания, так как аддитивное
    сглаживание вырождается в таковый при значении параметра сглаживания равного 0.

    Аргументы:
        ngram: N-грамма, вероятность которой нужно вычислить.
        counter: Предварительно заполненный счетчик n-грамм.
        addition: Параметр сглаживания. По-умолчанию: 0 (нет сглаживания).
            Для сглаживания по лапдасу установить равным 1.
        vocabulary_size: Размер словаря, то есть количество уникальных слов
            в тренировочном корпусе текста.

    Возвращаемое значнение:
        Вычисленная вероятность n-граммы.
    """
    counts = counter.get_count_n(ngram)
    favorable_count = counts[-1] + addition
    value = counts[-2] if counter.n > 1 else counter.get_total_count()
    total_count = value + addition * vocabulary_size
    if total_count == 0:
        return np.inf
    probability = favorable_count / total_count
    return np.inf if probability < 1e-15 or total_count == 0 else -np.log(probability)


class BaseLM(ABC):
    """
    Представляет базовый абстрактный класс языковой модели, служащий основой
    для всех реализованных в работе языковых моделей.
    """

    def __init__(
            self,
            n: int,
            counter: Counter,
            is_acceptable_character: Optional[Callable[[str], bool]] = None
    ) -> None:
        """
        Инициализация базовых параметров языковой модели.

        Аргументы:
            n: Размерность языковой модели (длина n-грамм в модели).
            counter: Счетчик n-грамм.
            is_acceptable_character: Возврящает True для символов, составляющих n-грамму, и False для
                символов-разделителей n-грамм. Согласно правилам, заложенным в этот метод, из корпуса текста
                будут выделяться n-граммы. Метод по-умолчанию представлен в данном классе ниже.
        """
        self.n = n
        self._counter = counter
        self._model_probabilities = None
        self.__is_acceptable_character = is_acceptable_character or BaseLM.__is_acceptable_character
        if self.n == 1:
            self._vocabulary = {NGram.get_sys_token('unknown')}
        else:
            self._vocabulary = {NGram.get_sys_token('start'), NGram.get_sys_token('end'),
                                NGram.get_sys_token('unknown')}

    @staticmethod
    def __is_acceptable_character(character: str) -> bool:
        """
        Метод, используемый для выделения n-грамм, в случае, если в конструкор
        не была передана своя реализация метода. В реализации по-умолчанию
        полагается, что символ character принадлежит n-грамме, если есть
        буква, либо один из символов: '<', '>', '/', '.'.

        Аргументы:
            character: Символ, который необходимо проверить на принадлежность
                к n-грамме.

        Возвращаемое значение:
            True, если символ принадлежит n-грамме. Иначе False.
        """
        return character.isalpha() or character in {'<', '>', '/', '.'}

    def __corpus_reader(self, ngram: NGram, ngrams_counter: Counter | None, corpus_path: str, initial_character: str, case: str = 'count', update_vocabulary: bool = True) -> Tuple[float, float]:
        """
        Вспомогательный метод, который по корпусу текста может заполнить счетчик n-грамм информацией о количестве каждой
        n-граммы в тексте (параметр case='count'), а также вычислить данные для метрики perplexity (case='perplexity').

        Аргументы:
            ngram: Объект, в котором поочередно в порядке следования хранятся все n-граммы из
                корпуса текста. На каждом шаге соответствующая n-грамма поступает в счетчик,
                что приводит к его обновлению.
            ngrams_counter: Счетчик n-грамм, который будет инкрементироваться по мере считывания
                новых n-грамм из корпуса текста.
            corpus_path: Путь до корпуса текста, который требуется проагнализировать.
            initial_character: Начальный символ. Не должен встречаться в корписе текста.
            case: 'count', если требуется заполнить счетчик n-грамм информацией,
                'perplexity', если требуется посчитать данные для метрики perplexity.
                По-умолчанию: count.
            update_vocabulary: Если True, то словарь языковой модели будет увеличваться по мере
                считывания из корпуса текста новых слов. Иначе размер словаря не будет изменяться.
                По-умолчанию: True.

        Возвращаемое значение:
            Кортеж, содержащий данные для метрики perplexity, либо (0, 0), если
            требовалось заполнить счетчик.
        """
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
                    if update_vocabulary:
                        self._vocabulary.add(word)
                    ngram.update(word)
                    if case == 'count':
                        ngrams_counter.count_n(ngram)
                    elif case == 'perplexity':
                        perplexity += self.get_probability(ngram)
                        n += 1
        return (0, 0) if case == 'count' else (perplexity, n)

    def _count_all_ngrams(self, corpus_path: str, ngrams_counter: Counter, initial_character: str = '$', update_vocabulary: bool = True) -> None:
        """
        Метод, заполняющий счетчик n-грамм информацией о количестве n-грамм в корпусе текста.

        Аргументы:
            corpus_path: Путь до корпуса текста, который нужно проанализировать.
            ngrams_counter: Счетчик n-грамм, который нужно заполнить.
            initial_character: Начальный символ. Не должен встречаться в корписе текста.
                По-умолчанию: '$'.
            update_vocabulary: Если True, то словарь языковой модели будет увеличваться по мере
                считывания из корпуса текста новых слов. Иначе размер словаря не будет изменяться.
                По-умолчанию: True.
        """
        ngram = NGram(self.n)
        self.__corpus_reader(ngram, ngrams_counter, corpus_path, initial_character, case='count', update_vocabulary=update_vocabulary)
        for i in range(self.n):
            ngram.update(NGram.get_sys_token(name='end'))
            ngrams_counter.count_n(ngram)

    @property
    def vocabulary(self):
        """
        Возвращает vocabulry.
        """
        return self._vocabulary

    def oov_rate(self, path_to_test_corpus: str) -> float:
        """
        Метод вычисляющий Out-of-Vocabulary метрику. Это показатель, отражающий
        процентную составляющую слов, отсутствующих в словаре модели (OOV-слов).

        Формально вычисляется как:
            OOVRate = (Количество OOV-слов) / (Общее количество слов в тексте) * 100%.

        Аргументы:
            path_to_test_corpus: Путь до корпуса текста, который нужно проанализировать
                и по которому нужно посчитать Out-of-Vocabulary метрику.

        Возвращаемое значение:
            Посчитанная OOV метрика.
        """
        counter = Counter(self.n)
        self._count_all_ngrams(path_to_test_corpus, counter, update_vocabulary=False)
        difference, count = self._counter.difference_and_count(counter)
        return difference / count * 100

    def perplexity(self, path_to_test_corpus: str, initial_character: str = '$') -> float:
        """
        Метод, вычисляющий метрику perplexity. Является ключевым показателем качества
        языковой модели. Измеряет, насколько хорошо модель предсказывает вероятности
        n-грамм из тестовго корпуса текста path_to_test_corpus.

        Формально вычисляется как:
            Perplexity = exp(1/N * ∑_{i=1}^N(-log P(w_i|w_{i-n+1},...,w_{i-1}))),
                где n - размерность языковой модели.

        Аргументы:
            path_to_test_corpus: Путь до корпуса текста, по которому нужно
                вычислить метрику perplexity.
            initial_character: Начальный символ. Не должен встречаться в корписе текста.
                По-умолчанию: '$'.

        Возвращаемое значение:
            Посчитанная метрика perplexity.
        """
        ngram = NGram(self.n)
        perplexity, n = self.__corpus_reader(ngram, None, path_to_test_corpus, initial_character, case='perplexity', update_vocabulary=False)
        ngram.update('</s>')
        perplexity += self.get_probability(ngram)
        return np.exp(perplexity * 1. / (n + 1))

    def _prepare_ngram(self, ngram: NGram) -> NGram:
        """
        Метод проверяющий n-грамму на валидность путем сравнения
        длины этой n-граммы с размерностью языковой модели, а
        также подготавливающий ее к передаче языковой модели
        путем замены всех незвестные модели слов специальным
        токеном <unk>.

        Аргументы:
            ngram: N-грамма, которую необходимо проверить и подготовить.

        Возвращаемое значение:
            Проверенная и подготоовленная n-грамма.
        """
        if len(ngram) != self.n:
            raise ValueError(f'{self.n}-gram model expected ngram of length {self.n}, but len={len(ngram)} given.')
        for i in range(len(ngram)):
            if ngram[i] not in self._vocabulary:
                ngram[i] = NGram.get_sys_token('unknown')
        return ngram

    @abstractmethod
    def get_probability(self, ngram: NGram) -> float:
        """
        Абстрактный метод, переопределяемый в дочерних классах.
        Должен возвращать вероятность переданной ему на вход n-граммы.

        Аргмуметны:
            ngram: N-грамма, вероятность которой нужно вычислить.

        Возвращаемое значение:
            Вычисленная вероятность n-граммы.
        """
        pass


class NoSmoothingLM(BaseLM):
    """
    Представляет класс языковой модели, вычисляющий вероятности
    n-грамм без использования какого бы ни было метода сглаживания.
    """

    def __init__(
            self,
            n: int,
            corpus_path: str,
            is_acceptable_character: Optional[Callable[[str], bool]] = None
    ) -> None:
        """
        Инициализация параметров языковой модели.

        Аргументы:
            n: Размерность языковой модели (длина n-грамм в модели).
            corpus_path: Путь до корпуса текста, который нужно проанализировать и
                по которому следует заполнить считчик n-грамм.
            is_acceptable_character: Возврящает True для символов, составляющих n-грамму, и False для
                символов-разделителей n-грамм. Согласно правилам, заложенным в этот метод, из корпуса текста
                будут выделяться n-граммы. По-умолчанию: None (использовать метод из суперкласса).
        """
        super().__init__(n, Counter(n), is_acceptable_character)
        self._count_all_ngrams(corpus_path, self._counter)

    @override
    def get_probability(self, ngram: NGram) -> float:
        """
        Метод, возвращающий вероятность переданной ему на вход n-граммы
        без использования каких-либо методов сгладживания.

        Аргмуметны:
            ngram: N-грамма, вероятность которой нужно вычислить.

        Возвращаемое значение:
            Вычисленная вероятность n-граммы.
        """
        prepared_ngram = self._prepare_ngram(ngram)
        return get_probability_additive(prepared_ngram, self._counter, addition=0, vocabulary_size=len(self.vocabulary))


class LaplaceSmoothingLM(BaseLM):
    """
    Представляет класс языковой модели, вычисляющий вероятности
    n-грамм с использованием метода сглаживания по Лапласу.
    """

    def __init__(
            self,
            n: int,
            corpus_path: str,
            is_acceptable_character: Optional[Callable[[str], bool]] = None
    ) -> None:
        """
        Инициализация параметров языковой модели.

        Аргументы:
            n: Размерность языковой модели (длина n-грамм в модели).
            corpus_path: Путь до корпуса текста, который нужно проанализировать и
                по которому следует заполнить считчик n-грамм.
            is_acceptable_character: Возврящает True для символов, составляющих n-грамму, и False для
                символов-разделителей n-грамм. Согласно правилам, заложенным в этот метод, из корпуса текста
                будут выделяться n-граммы. По-умолчанию: None (использовать метод из суперкласса).
        """
        super().__init__(n, Counter(n), is_acceptable_character)
        self._count_all_ngrams(corpus_path, self._counter)

    @override
    def get_probability(self, ngram: NGram) -> float:
        """
        Метод, возвращающий вероятность переданной ему на вход n-граммы
        с использованием метода сглаживания по Лапласу.

        Аргмуметны:
            ngram: N-грамма, вероятность которой нужно вычислить.

        Возвращаемое значение:
            Вычисленная вероятность n-граммы.
        """
        prepared_ngram = self._prepare_ngram(ngram)
        return get_probability_additive(prepared_ngram, self._counter, addition=1, vocabulary_size=len(self.vocabulary))


class AdditiveSmoothingLM(BaseLM):
    """
    Представляет класс языковой модели, вычисляющий вероятности
    n-грамм с использованием метода аддитивного сглаживания.
    """

    def __init__(
            self,
            n: int,
            corpus_path: str,
            addition: int,
            is_acceptable_character: Optional[Callable[[str], bool]] = None
    ) -> None:
        """
        Инициализация параметров языковой модели.

        Аргументы:
            n: Размерность языковой модели (длина n-грамм в модели).
            corpus_path: Путь до корпуса текста, который нужно проанализировать и
                по которому следует заполнить считчик n-грамм.
            is_acceptable_character: Возврящает True для символов, составляющих n-грамму, и False для
                символов-разделителей n-грамм. Согласно правилам, заложенным в этот метод, из корпуса текста
                будут выделяться n-граммы. По-умолчанию: None (использовать метод из суперкласса).
        """
        if addition <= 0:
            raise ValueError(
                'Addition size must be positive. If you intend to use addition=0 consider using NoSmoothingLM.')
        super().__init__(n, Counter(n), is_acceptable_character)
        self._count_all_ngrams(corpus_path, self._counter)
        self.__addition = addition

    @override
    def get_probability(self, ngram: NGram) -> float:
        """
        Метод, возвращающий вероятность переданной ему на вход n-граммы
        с использованием метода аддитивного сглаживания.

        Аргмуметны:
            ngram: N-грамма, вероятность которой нужно вычислить.

        Возвращаемое значение:
            Вычисленная вероятность n-граммы.
        """
        prepared_ngram = self._prepare_ngram(ngram)
        return get_probability_additive(prepared_ngram, self._counter, self.__addition,
                                        vocabulary_size=len(self.vocabulary))


class SimpleGoodTuringSmoothingLM(BaseLM):
    """
    Представляет класс языковой модели, вычисляющий вероятности
    n-грамм с использованием метода сглаживания по Гуду-Тьюрингу.
    """

    def __init__(
            self,
            n: int,
            corpus_path: str,
            is_acceptable_character: Optional[Callable[[str], bool]] = None
    ) -> None:
        """
        Инициализация параметров языковой модели.

        Аргументы:
            n: Размерность языковой модели (длина n-грамм в модели).
            corpus_path: Путь до корпуса текста, который нужно проанализировать и
                по которому следует заполнить считчик n-грамм.
            is_acceptable_character: Возврящает True для символов, составляющих n-грамму, и False для
                символов-разделителей n-грамм. Согласно правилам, заложенным в этот метод, из корпуса текста
                будут выделяться n-граммы. По-умолчанию: None (использовать метод из суперкласса).
        """
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
        """
        Метод, выполняющий ненормализованный предподсчет логарифмических
        вероятностей n-грамм.

        Формально вычисляется:
            log P(x:c(x)=r) = log(r+1) + log S(N_{r+1}) - log S(N_{r}),
                где x - n-грамма, c(x) - количество раз, которое эта n-грамма
                встречается в корпусе текста, а log S(N_r) = log Z_{r} и
                вычисляется методом __get_log_zr.
        """
        total_log_count = np.log(self._counter.get_total_count(self._counter.n))
        frequency_max, _ = next(reversed(self.__sorted_frequencies.items()))
        probabilities = {0: self.__get_log_zr(1) - total_log_count}
        for r in self.__sorted_frequencies.keys():
            log_new_r = np.log(r + 1) + self.__get_log_zr(r + 1) - self.__get_log_zr(r)
            probabilities[r] = log_new_r - total_log_count
        self.__normalize(probabilities)

    def __normalize(self, probabilities: Dict[int, Any]) -> None:
        """
        Метод выполняющий нормализацию вероятностей.

        Аргументы:
            probabilities: Вероятности, подсчитанные в метооде __compute_probabilities,
                которые нужно нормализовать.
        """
        for context in itertools.product(self._vocabulary, repeat=self._counter.n - 1):
            probabilities_sum = np.inf
            for token in self._vocabulary:
                ngram = context + (token,)
                count = self._counter.get_count_n(NGram.from_words_list(ngram))[-1]
                probabilities_sum = lmath.log_sum(probabilities_sum, -probabilities[count])
            for token in self._vocabulary:
                ngram = context + (token,)
                count = self._counter.get_count_n(NGram.from_words_list(ngram))[-1]
                self.__probabilities[ngram] = -probabilities[count] - probabilities_sum

    def __get_log_zr(self, r):
        """
        Вспомогательный метод, который вычисляет log S(N_r) = log Z_{r},
        используемый в методе __compute_probabilities для вычисления
        логарифмических вероятностей.

        Аргументы:
            r: Параметр, который используется для вычисления log Z_r.

        Возвращаемое значение:
            Вычисленный log Z_r.
        """
        return self.__coefficients[1] + self.__coefficients[0] * np.log(r)

    def __linear_fit_from_dict(self):
        """
        Выполняет линейную регрессию по словарю.
        """
        x, y = self.__preprocess_frequencies()
        A = np.vstack([x, np.ones_like(x)]).T
        return np.linalg.lstsq(A, y, rcond=None)[0]

    def __preprocess_frequencies(self):
        """
        Выполняет предварительную обработку параметров r и N_r,
        что потом используются для построения линейной
        регрессии.

        Формально на выходе получается кортеж списков из log r и log Z_r,
            где Z_r = N_r / (0.5 * (t - q)), где q, r, t есть три
            последовательных величины для которых N_q, N_r и N_t
            не равны 0.

        Возвращаемое значение:
            Данные, используемые для построения по ним линейной
            регрессии в методе __linear_fit_from_dict.
        """
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
        """
        Метод, возвращающий вероятность переданной ему на вход n-граммы
        с использованием метода сглаживания по Гуду-Тьюрингу.

        Аргмуметны:
            ngram: N-грамма, вероятность которой нужно вычислить.

        Возвращаемое значение:
            Вычисленная вероятность n-граммы.
        """
        return self.__probabilities[tuple(ngram.as_deque())]


class KatzSmoothingLM(BaseLM):
    """
    Представляет класс языковой модели, вычисляющий вероятности
    n-грамм с использованием метода сглаживания по Кацу.

    Константы:
        EPS: Малое число для сравнения чисел с плавающей точкой.
    """
    EPS: Final[float] = 1e-9

    def __init__(
            self,
            n: int,
            is_acceptable_character: Optional[Callable[[str], bool]] = None,
            counter: Optional[Counter] = None
    ) -> None:
        """
        Инициализация параметров языковой модели.

        Аргументы:
            n: Размерность языковой модели (длина n-грамм в модели).
            corpus_path: Путь до корпуса текста, который нужно проанализировать и
                по которому следует заполнить считчик n-грамм.
            is_acceptable_character: Возврящает True для символов, составляющих n-грамму, и False для
                символов-разделителей n-грамм. Согласно правилам, заложенным в этот метод, из корпуса текста
                будут выделяться n-граммы. По-умолчанию: None (использовать метод из суперкласса).
            counter: Счетчик.
        """
        super().__init__(n, counter, is_acceptable_character)
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
        """
        Создать языковую модель с методом сглаживания вероятностей по Кацу,
        из тренировочного корписа и сохранить ее в ARPA файл.

        Аргументы:
            n: Размерность языковой модели.
            corpus_path: Путь до корпуса текст, на котором нужно натренировать модель.
            k: Порог отсечения. Определяет, какие n-граммы встречаются достаточно часто,
                чтобы использовать их несглаженные вероятности, а какие подвергнуть
                сглаживанию.
            reserved_probability: Вероятностная масса, зарезервированная под неизвестные токены.
            is_acceptable_character: Возврящает True для символов, составляющих n-грамму, и False для
                символов-разделителей n-грамм. Согласно правилам, заложенным в этот метод, из корпуса текста
                будут выделяться n-граммы. По-умолчанию: None (использовать метод из суперкласса).

        Возвращаемое значение:
            Созданная языковая модель.
        """
        obj = KatzSmoothingLM(n, is_acceptable_character, FrequenciesCounter(n))
        obj.__k = k
        obj.__reserved_probability = reserved_probability
        obj._count_all_ngrams(corpus_path, obj._counter)
        obj._corpus_path = corpus_path
        obj.__observed_ngrams_probabilities = defaultdict(dict)
        obj.__alpha = {}
        obj.__precompute_adjustment_values(obj._counter.root, NGram.empty())
        obj.__precompute_probabilities(obj._counter.root, NGram.empty())
        obj.__reserved_probability = -np.log(obj.__reserved_probability)
        save_lm_path = f'./language_model/KATZ_LM_{int(time.time())}.ARP'
        arpafile.write_arpa_lm(save_lm_path, obj.n,
                               (obj.__observed_ngrams_probabilities, obj.__alpha, obj.__reserved_probability))
        logging.log(logging.INFO,
                    f"Языковая модель со сглаживанием по Кацу сохранена в ARPA-файл: {save_lm_path}. Позже Вы сможете восстановить из него языковую модель.")
        return obj

    @classmethod
    def from_arpa_file(
            cls,
            arpa_file_path: str
    ) -> 'KatzSmoothingLM':
        """
        Получить объект языковой модели с методом сглаживания вероятностей по Кацу
        из ARPA-файла.

        Аргументы:
            arpa_file_path: Путь до ARPA-файла, по которому следует
                получить объект языковой модели.

        Возвращаемое значение:
            Созданная языковая модель.
        """
        probabilities, alpha, vocabulary, n, reserved_probability = arpafile.read_arpa_lm(arpa_file_path)
        obj = KatzSmoothingLM(n)
        obj.__observed_ngrams_probabilities = probabilities
        obj.__alpha = alpha
        obj._vocabulary = vocabulary
        obj._arpa_file_path = arpa_file_path
        obj.reserved_probability = reserved_probability
        logging.info(f"Языковая модель со сглаживанием по Кацу была восстановлена из ARPA-файла: {arpa_file_path}")
        return obj

    def __precompute_probabilities(self, node: Node, ngram: NGram) -> None:
        """
        Вспомогательный метод, который предподсчитывает вероятности по Катцу.
        """
        for token, child in node.children.items():
            ngram.append(token)
            n = len(ngram)
            if child.value > 0:
                self.__observed_ngrams_probabilities[n][ngram.as_string()] = -np.log(self.__p_katz(child.value, n,
                                                                                                   node.value)) if n > 1 \
                    else -np.log(child.value / self._counter.get_total_count() * (1 - self.__reserved_probability))
            self.__precompute_probabilities(child, ngram)
            ngram.shorten(direction='right')
        logging.log(logging.INFO, "Вероятности по Кацу были успешно предпосчитаны.")

    def __dr(self, r: int, i: int) -> float:
        """
        Вспомогательный метод, который вычисляет коэффициент дисконтирования,
        который корректирует вероятности низкочастотных n-грамм.
        """
        if r > self.__k:
            return 1
        good_turing_r = (r + 1) * self.__get_frequencies_safe(i, r + 1) / self.__get_frequencies_safe(i, r)
        top = good_turing_r / r - (self.__k + 1) * self.__get_frequencies_safe(i, self.__k + 1) / \
              self.__get_frequencies_safe(i, 1)
        bottom = 1 - (self.__k + 1) * self.__get_frequencies_safe(i, self.__k + 1) / self.__get_frequencies_safe(i, 1)
        return self.__force_in_range(top / bottom)

    def __get_frequencies_safe(self, i: int, r: int):
        """
        Метод получающий количество i-грамм, которые встречались в тренировочном
        корпусе текста ровно r раз. Если такое количество равно 0, возвращает
        значение константы EPS, чтобы предотвратить ошибку деления на 0.

        Аргументы:
            i: Размерность i-грамм.
            r: Количество раз, которое должны встречаться в корпусе текста i-граммы.
        """
        frequencies_counter = cast(FrequenciesCounter, self._counter)
        return frequencies_counter.frequencies[i][r] if r in frequencies_counter.frequencies[i] else KatzSmoothingLM.EPS

    def __p_katz(self, r: int, n: int, cnt: int) -> float:
        """
        Метод вычисляющий вероятность по Кацу.
        """
        return self.__dr(r, n - 1) * r / cnt

    def __get_alpha(self, context: NGram) -> float:
        """
        Метод возвращает коэффициент растяжения для заданной истории n-граммы.
        """
        key = context.as_string()
        return self.__alpha[key] if key in self.__alpha else 0

    def __force_in_range(self, x: float, floor: int = EPS, ceil: int = 1) -> float:
        """
        Вспомогательный метод, который принудительно поддерживает значение
        x в заданном диапазоне [floor, ceil]. Если x выходит из диапазона
        слева, возвращает floor. Если x выходит из диапазона справа, то
        возвращает ceil. Иначе возвращает x.

        Аргументы:
            x: Значение, которое требуется принудительно поддержать в диапазоне.
            floor: Левая граница диапазона.
            ceil: Правая граница диапазона.

        Возвращаемое значение:
            floor, если x выходит из диапазона слева, ceil, если x выходит из
            диапазона справа, иначе x.
        """
        return max(min(x, ceil), floor)

    def __precompute_adjustment_values(self, node: Node, ngram: NGram) -> None:
        """
        Вспомогательный метод, который рекурсивно подсчитывает коэффициенты
        растяжения.
        """
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
            if n < self.n:
                self.__precompute_adjustment_values(child, ngram)
            ngram.shorten(direction='right')

    @override
    def get_probability(self, ngram: NGram) -> float:
        """
        Метод, возвращающий вероятность переданной ему на вход n-граммы
        с использованием метода сглаживания по Кацу.

        Аргмуметны:
            ngram: N-грамма, вероятность которой нужно вычислить.

        Возвращаемое значение:
            Вычисленная вероятность n-граммы.
        """
        prepared_ngram = self._prepare_ngram(ngram)
        return self.__get_probability_inner(deepcopy(prepared_ngram))

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


if __name__ == '__main__':
    path = 'C:/Users/PeterA/Desktop/vkr/test/corpus.txt'
    path2 = 'C:/Users/PeterA/Desktop/vkr/test/phones.txt'
    arpa = './language_model/katz_lm_1737559161.ARP'

    gt = SimpleGoodTuringSmoothingLM(
        n=2,
        corpus_path='C:/Users/PeterA/Desktop/vkr/test/_____attempt14/language/corpus.txt'
    )
    print(gt.get_probability(NGram.from_words_list(['включить', 'включить'])))
    print(gt.get_probability(NGram.from_words_list(['включить', 'лампу'])))
    print(gt.get_probability(NGram.from_words_list(['лампу', 'включить'])))
    print(gt.get_probability(NGram.from_words_list(['лампу', 'лампу'])))

    exit(0)

    model = KatzSmoothingLM.from_train_corpus(
        n=3,
        corpus_path='C:/Users/PeterA/Desktop/vkr/test/corpus.txt',
        k=5,
        reserved_probability=0.01
    )
    voc = model.vocabulary
    ind = 0
    for i in voc:
        for j in voc:
            probs = 0
            for k in voc:
                lp = model.get_probability(NGram.from_words_list([i, j, k]))
                print(f"print(np.exp(model0.get_probability(['{i}', '{j}', '{k}']))) = {np.exp(-lp)}")
                probs += np.exp(-lp)
            print(ind, '=', probs, '|', [i, j, k])
            if not math.isclose(probs, 1):
                print("...")
            ind += 1

    # model = KatzSmoothingLM.from_arpa_file(
    #     arpa_file_path="./language_model/KATZ_LM_1738025541.ARP",
    # )
    print(model.vocabulary)

    # model = KatzSmoothingLM.from_train_corpus(
    #     n=2,
    #     corpus_path='C:/Users/PeterA/Desktop/vkr/test/corpus.txt',
    #     k=5,
    #     reserved_probability=0.1,
    #     is_acceptable_character=xxx
    # )
    # prob = model.get_probability(NGram.from_words_list(['ache', 'ache']))
    # print(prob)
    # grammar_wfst = model.build_wfst()
    # grammar_wfst.view()

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
