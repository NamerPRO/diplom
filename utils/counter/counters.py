from collections import deque, defaultdict
from copy import deepcopy
from typing import List, Deque, Tuple

from overrides import overrides

from utils.ngram import NGram
from utils.counter.node import Node


class Counter:
    """
    Представляет класс-счетчик n-грамм.
    """

    def __init__(self, n: int) -> None:
        """
        Инициализация параметров класса-счетчика.

        Аргументы:
            n: Максимальная длина n-граммы.
        """
        self.n = n
        self._root = Node()
        self.unique_ngrams_count: List[int] = [0 for _ in range(n)]

    @property
    def root(self) -> 'Node':
        """
        Возвращает root.
        """
        return self._root

    def count_n(self, ngram: NGram) -> None:
        """
        Подсчитывает переданную n-грамму и все m-граммы
        меньшего порядка.

        Аргументы:
            ngram: N-грамма для подсчета, передаваемая методу.
        """
        node = self._root
        i = 0
        while i < len(ngram) and ngram[i] in node.children:
            self._calculate(node[ngram[i]], i, ngram)
            node = node[ngram[i]]
            i += 1
        while i < len(ngram):
            node[ngram[i]] = Node(value=0)
            self._calculate(node[ngram[i]], i, ngram)
            node = node[ngram[i]]
            i += 1

    def _is_valid_ngram(self, ngram: NGram, n: int) -> bool:
        """
        Выполняет валидацию n-граммы.

        Аргументы:
            ngram: N-грамма, которую нужно проверить.
            n: Длина такой n-граммы

        Возвращаемое значение:
            True, если результат валидации положителен. False иначе.
        """
        if self.n == 1 and ngram[n] == '</s>':
            return False
        if n > 0 and ngram[n - 1] == NGram.get_sys_token(name="end"):
            return False
        return True

    def _calculate(self, node: Node, _n: int, ngram: NGram) -> None:
        """
        Вспомогательный метод, вызываемый в методе count_n,
        осуществляющий подсчет n-граммы.

        Аргументы:
            node: Узел префиксного дерева, значение, хранящееся в котором,
                нужно инкрементировать.
            _n: Размерность подсчитываемой n-граммы.
            ngram: N-грамма, используемая при подсчете.
        """
        if not self._is_valid_ngram(ngram, _n):
            node.value = 0
            return
        if node.value == 0:
            self.unique_ngrams_count[_n] += 1
        if _n == 0:
            self._root.value += 1
        node.value += 1

    def get_count_n(self, ngram: NGram) -> List[int]:
        """
        Возвращает количество раз, которое встречается n-грамма и
        все m-граммы, меньшего порядка.

        Аргументы:
            ngram: N-грамма, по которой формируется список, элементы
                которого есть количество раз, которое встречается n-грамма
                и все m-граммы меньшего порядка.

        Возвращаемое значение:
            Список, элементы которого есть:
                количество раз, которое встречается n-грамма и все m-граммы меньшего порядка.
        """
        counts: List[int] = [0 for _ in range(len(ngram))]
        node = self._root
        for i in range(len(ngram)):
            if ngram[i] not in node.children:
                return counts
            counts[i] = node[ngram[i]].value
            node = node[ngram[i]]
        return counts

    def get_total_count(self, n: int = 1) -> int:
        """
        Возвращает суммарное количество n-грамм.

        Аргументы:
            n: Длина n-грамм, суммарное количество которых
                нужно вернуть.

        Возвращаемое значение:
            Суммарное количество n-грамм.
        """
        if n == self.n:
            return self.root.value - self.n + 1
        return self.root.value + (0 if n == 1 else 1)

    def difference_and_count(self, counter: 'Counter') -> (int, int):
        """
        Вычисляет, насколько счетчик 'counter' отличается от 'self'. Счетчик 1
        отличается от счетчика 2 n-граммами, которые он содержит, но не
        содержит счетчик 2.

        Аргументы:
            counter: Счетчик, для сравнения со счетчиком 'self'.

        Возвращаемое значние:
            Кортеж, содержащий элементы:
                - Величина, показывающая количество n-грамм, содержащихся в 'counter',
                    но не содержащихся в 'self'.
                - Величина, показывающая количество уникальных n-грамм в счетчике 'counter'.
        """
        return self.__compute_difference_and_count(counter.root, 0, 0, deque())

    def __compute_difference_and_count(self, node: 'Node', difference: int, count: int, ngram: Deque[str]) -> (int, int):
        """
        Вспомогательный метод, вызываемый 'difference_and_count', осуществляющий
        рекурсивный подсчет, насколько два счетчика отличаются друг от дргуа.
        """
        for token, child in node.children.items():
            ngram.append(token)
            i = len(ngram) - 1
            ngrams_count_in_another_counter = self.get_count_n(NGram.from_words_list(ngram))[-1]
            difference, count = self.__compute_difference_and_count(child, difference, count, ngram)
            if i + 1 == self.n and self._is_valid_ngram(NGram.from_words_list(ngram), i):
                count += 1
                difference += ngrams_count_in_another_counter == 0
            ngram.pop()
        return difference, count

    def __str__(self) -> str:
        """
        Возвращает строковое представление объекта.

        Возвращаемое значение:
            str: Строка, описывающая счетчик.
        """
        ngrams_list = [[] for _ in range(self.n)]
        ngram = deque()
        self.__get_all_ngrams(self._root, ngrams_list, ngram)
        resp = f"{self.__class__.__name__}=("
        for i in range(self.n):
            resp += "["
            for j in range(len(ngrams_list[i])):
                resp += f"{ngrams_list[i][j][0]}: {ngrams_list[i][j][1]},"
            resp += "],"
        return resp + ")"

    def __get_all_ngrams(self, node: 'Node', ngrams_list: List[List[Tuple[NGram, int]]], ngram: Deque[str]) -> None:
        """
        Вспомогательный метод, используемый в методе __str()__,
        записывающий в список ngrams_list все n-граммы, которые
        были подсчитаны счетчиком вместе с количеством раз, которым
        они встречались при подсчете.

        Аргументы:
            node: Узел префиксного дерева.
            ngrams_list: Список, в котором окажутся все n-граммы, подсчитанные счетчиком.
            ngram: вспомогательная переменная, которая на каждом шаге алгоритма содержит
                соответствующую n-грамму.
        """
        for token, child in node.children.items():
            ngram.append(token)
            gram = NGram.from_words_list(deepcopy(ngram))
            ngrams_count = self.get_count_n(gram)[-1]
            if ngrams_count > 0:
                ngrams_list[len(ngram) - 1].append((gram, ngrams_count))
            self.__get_all_ngrams(child, ngrams_list, ngram)
            ngram.pop()


class FrequenciesCounter(Counter):
    def __init__(self, n: int) -> None:
        """
        Инициализация параметров класса-счетчика, который дополнительно
        считает "частоту": количество n-грамм, которые встречаются
        ровно заданное число раз.

        Аргументы:
            n: Максимальная длина n-граммы.
        """
        super().__init__(n)
        self.frequencies = [defaultdict(int) for i in range(n)]
        self.total_ngrams_count: List[int] = [0 for i in range(n)]

    @overrides
    def _calculate(self, node: Node, _n: int, ngram: NGram) -> None:
        """
        Вспомогательный метод, вызываемый в методе count_n родительского
        класса Counter, осуществляющий подсчет n-граммы и их "частот".

        Аргументы:
            node: Узел префиксного дерева, значение, хранящееся в котором,
                нужно инкрементировать.
            _n: Размерность подсчитываемой n-граммы.
            ngram: N-грамма, используемая при подсчете.
        """
        if not self._is_valid_ngram(ngram, _n):
            node.value = 0
            return
        if node.value == 0:
            self.unique_ngrams_count[_n] += 1
        self.total_ngrams_count[_n] += 1
        if _n == 0:
            self._root.value += 1
        self.frequencies[_n][node.value] -= 1
        node.value += 1
        self.frequencies[_n][node.value] += 1
