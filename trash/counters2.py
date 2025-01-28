from collections import deque
from copy import deepcopy
from typing import List, Deque, Tuple

from overrides import overrides
from sortedcontainers import SortedDict

from utils.ngram import NGram
from utils.counter.node import Node


class Counter:

    def __init__(self, n: int) -> None:
        self.n = n
        self._root = Node()
        self.unique_ngrams_count: List[int] = [0 for _ in range(n)]

    @property
    def root(self) -> 'Node':
        return self._root

    def count_n(self, ngram: NGram) -> None:
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
        if ngram[n] == NGram.get_sys_token(name="start"):
            return False
        if n > 0 and ngram[n - 1] == NGram.get_sys_token(name="end") or n == 0 and ngram[n] == NGram.get_sys_token(name="end"):
            return False
        return True

    def _calculate(self, node: Node, _n: int, ngram: NGram) -> None:
        if not self._is_valid_ngram(ngram, _n):
            node.value = -1
            return
        if node.value == 0:
            self.unique_ngrams_count[_n] += 1
        if _n == 0:
            self._root.value += 1
        node.value += 1

    def get_count_n(self, ngram: NGram) -> List[int]:
        counts: List[int] = [0 for _ in range(len(ngram))]
        node = self._root
        for i in range(len(ngram)):
            if ngram[i] not in node.children:
                for j in range(i, len(ngram)):
                    counts[j] = 0 if self._is_valid_ngram(ngram, j) else -1
                return counts
            counts[i] = node[ngram[i]].value
            node = node[ngram[i]]
        return counts

    def get_total_count(self, n: int = 1) -> int:
        return self.root.value + (0 if n == 1 else 1)

    def difference(self, counter: 'Counter') -> List[int]:
        difference: List[int] = [0 for _ in range(counter.n)]
        ngram = deque()
        self.__compute_difference(counter.root, difference, ngram)
        return difference

    def __compute_difference(self, node: 'Node', difference: List[int], ngram: Deque[str]) -> None:
        for token, child in node.children.items():
            ngram.append(token)
            ngrams_count_in_another_counter = self.get_count_n(NGram.from_words_list(ngram))[-1]
            if ngrams_count_in_another_counter == 0:
                difference[len(ngram) - 1] += 1
            self.__compute_difference(child, difference, ngram)
            ngram.pop()

    def __str__(self) -> str:
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
        super().__init__(n)
        self.frequencies: List[SortedDict[int, int]] = [SortedDict() for _ in range(n)]

    @overrides
    def _calculate(self, node: Node, _n: int, ngram: NGram) -> None:
        if not self._is_valid_ngram(ngram, _n):
            node.value = -1
            return
        if node.value == 0:
            self.unique_ngrams_count[_n] += 1
        if _n == 0:
            self._root.value += 1
        if node.value not in self.frequencies[_n]:
            self.frequencies[_n][node.value] = -1
        else:
            self.frequencies[_n][node.value] -= 1
        node.value += 1
        if node.value not in self.frequencies[_n]:
            self.frequencies[_n][node.value] = 1
        else:
            self.frequencies[_n][node.value] += 1


# class AdditiveCounter(Counter):
#     def __init__(self, n: int, addition: int = 1) -> None:
#         super().__init__(n)
#         if addition <= 0:
#             raise ValueError(
#                 f'Addition must be positive, but {addition} is given. If you intend to use addition=0, use Counter(n: int) instead.')
#         self.addition = addition
#
#     @overrides
#     def _calculate(self, node: Node, _n: int, ngram: NGram) -> None:
#         if not self._is_valid_ngram(ngram, _n):
#             node.value = -1
#             return
#         if node.value == 0:
#             self.unique_ngrams_count[_n] += 1
#         if _n == 0:
#             self._root.value += 1
#             if node.value == 0:
#                 self._root.value += self.addition
#         node.value += 1 + self.addition if node.value == 0 else 1
