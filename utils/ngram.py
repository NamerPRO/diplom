from collections import deque
from copy import deepcopy
from itertools import islice
from typing import Final, Dict, Literal, List, Tuple, Union, Deque


class NGram:
    Token = Literal['start', 'end', 'silence', 'unknown']  # NOSONAR
    __sys_tokens: Final[Dict[Token, str]] = {
        'start': '<s>',
        'end': '</s>',
        'silence': '<sil>',
        'unknown': '<unk>'
    }

    def __init__(self, n: int) -> None:
        self.__n = n
        self.__ngram = deque([NGram.get_sys_token(name='start') for _ in range(n)])

    @staticmethod
    def empty() -> 'NGram':
        return NGram(0)

    @classmethod
    def from_words_list(cls, words: Union[Deque[str], List[str], Tuple[str, ...]]) -> 'NGram':
        if not words:
            raise ValueError('Expected list of words, but %s is given.' % words)
        ngram = cls(n=len(words))
        if isinstance(words, list) or isinstance(words, tuple):
            ngram.__ngram = deque(words)
        else:
            ngram.__ngram = deepcopy(words)
        return ngram

    @staticmethod
    def get_sys_token(name: Token = 'unknown') -> str:
        return NGram.__sys_tokens[name]

    @staticmethod
    def is_sys_token(name: Token) -> bool:
        return name in NGram.__sys_tokens

    @staticmethod
    def set_sys_token(tokens: Dict[Token, str]) -> None:
        for token, value in tokens.items():
            NGram.__sys_tokens[token] = value

    def update(self, word: str) -> None:
        self.__ngram.popleft()
        self.__ngram.append(word)

    def append(self, word: str) -> None:
        self.__n += 1
        self.__ngram.append(word)

    def shorten(self, direction="left") -> 'NGram':
        self.__n -= 1
        if direction == "left":
            self.__ngram.popleft()
        else:
            self.__ngram.pop()
        return self

    def to_ngrams(self):
        return [tuple(islice(self.__ngram, 0, i)) for i in range(1, len(self.__ngram) + 1)]

    def as_deque(self) -> Deque[str]:
        return self.__ngram

    def as_string(self):
        res = ''
        for i, word in enumerate(self.__ngram):
            res += f'{word} ' if i != len(self.__ngram) - 1 else word
        return res

    def __str__(self) -> str:
        res = '('
        for word in self.__ngram:
            res += f' {word}'
        return f'{res})'

    def __len__(self) -> int:
        return self.__n

    def __getitem__(self, item: int) -> str:
        return self.__ngram[item]

    def __setitem__(self, key: int, value: str) -> None:
        self.__ngram[key] = value

