from collections import deque
from copy import deepcopy
from typing import Final, Dict, Literal, List, Tuple, Union, Deque


class NGram:
    """
    Класс, представляющий n-грамму.
    """
    Token = Literal['start', 'end', 'silence', 'unknown']
    __sys_tokens: Final[Dict[Token, str]] = {
        'start': '<s>',
        'end': '</s>',
        'silence': '<sil>',
        'unknown': '<unk>'
    }

    def __init__(self, n: int) -> None:
        """
        Инициализация n-граммы. N-грамма инициализируется последовательностью
        из 'n' токенов NGram.get_sys_token(name='start'), которая по-умолчанию
        возвращает '<s>'.

        Аргументы:
            n: Длина n-граммы.
        """
        self.__n = n
        self.__ngram = deque([NGram.get_sys_token(name='start') for _ in range(n)])

    @staticmethod
    def empty() -> 'NGram':
        """
        Создает 0-грамму, то есть n-грамму длины 0.

        Возвращаемое значение:
            N-грамма длины 0.
        """
        return NGram(0)

    @classmethod
    def from_words_list(cls, words: Union[Deque[str], List[str], Tuple[str, ...]]) -> 'NGram':
        """
        Создает n-грамму из двусторонней очереди, списка или кортежа.
        Длина n-граммы определяется из количества токенов в 'words'.

        Аргументы:
            words: Двусторонняя очередь, список или кортеж, из чего
                создается n-грамма.

        Возвращаемое значение:
            Созданная по 'words' n-грамма.
        """
        if not isinstance(words, (list, tuple, deque)):
            raise ValueError(f'Ожидался один из следующих типов: двусторонняя очередь, список или кортеж. Но {words} подан.')
        ngram = cls(n=len(words))
        if isinstance(words, (list, tuple)):
            ngram.__ngram = deque(words)
        else:
            ngram.__ngram = deepcopy(words)
        return ngram

    @staticmethod
    def get_sys_token(name: Token = 'unknown') -> str:
        """
        Возврящает специальный токен по 'name':
            - 'start' - токен начала предложения.
                По-умолчанию: '<s>'.
            - 'end' - токен конца предложения.
                По-умолчанию: '</s>'.
            - 'silence' - токен тишины.
                По-умолчания: '<sil>'.
            - 'unknown' - токен неизвестного слова.
                По-умолчания: <unk>.

        Аргументы:
            name: Строка, в зависимости от которой будет
                возвращен служебный токен (см. описание выше).

        Возвращаемое значение:
            Специальный токен.
        """
        return NGram.__sys_tokens[name]

    @staticmethod
    def matches_sys_token(name: str) -> bool:
        """
        Проверяет, соответствует ли строке 'name'
        специальный токен.

        Аргументы:
            name: Стока, которая проверяется на соответствие
                специальному токену.

        Возвращаемое значение:
            True, если строке 'name' соостветствует специальный токен.
            False в противном случае.
        """
        return name in NGram.__sys_tokens

    @staticmethod
    def set_sys_token(tokens: Dict[Token, str]) -> None:
        """
        Заменяет специальные токены по информации из
        словаря 'tokens'.

        Аргументы:
            tokens: Словарь, ключами которого могут быть
                'start', 'end', 'silence', 'unknown', а
                значениями, строки, которыми надо переопределить
                стандартные токены начала предложения, конца
                предложения, тишины и неизвестного слова
                соответственно.
        """
        for token, value in tokens.items():
            NGram.__sys_tokens[token] = value

    def update(self, word: str) -> None:
        """
        Убирает самый левый токен в n-грамме и
        добавляет токен 'word' как самый правый.

        Аргументы:
            word: Токен, который нужно добавить в n-грамму.
        """
        self.__ngram.popleft()
        self.__ngram.append(word)

    def append(self, word: str) -> None:
        """
        Добавляет токен 'word' как самый правый. Размер
        n-граммы при этом увеличивается на 1.

        Аргументы:
            word: Токен, который нужно добавить в n-грамму.
        """
        self.__n += 1
        self.__ngram.append(word)

    def shorten(self, direction="left") -> 'NGram':
        """
        Удаляет либо самый левый токен из n-граммы (direction='left'),
        либо самый правый токен из n-граммы (direction='right'). Размер
        n-граммы при этом уменьшается на 1.

        Аргументы:
            direction: Определяет направление, с котрого нужно удалить
                токен из n-граммы. Если 'left', то удаляет самый левый
                токен, иначе - самый правый.
        """
        self.__n -= 1
        if direction == "left":
            self.__ngram.popleft()
        else:
            self.__ngram.pop()
        return self

    def as_deque(self) -> Deque[str]:
        """
        Возвращает n-грамму, представленную в виде
        двунаправленной очереди.

        Возвращаемое значение:
            Двунаправленная очередь, представляющая n-грамму.
        """
        return self.__ngram

    def as_string(self):
        """
        Возвращает строковое предствление n-граммы.

        Возвращаемое значение:
            Строковое представление n-граммы.
        """
        res = ''
        for i, word in enumerate(self.__ngram):
            res += f'{word} ' if i != len(self.__ngram) - 1 else word
        return res

    def __str__(self) -> str:
        """
        Возвращает строковое предствление n-граммы.

        Возвращаемое значение:
            Строковое представление n-граммы.
        """
        res = '('
        for word in self.__ngram:
            res += f' {word}'
        return f'{res})'

    def __len__(self) -> int:
        """
        Возвращает длину n-граммы.

        Возвращаемое значение:
            Длина n-граммы.
        """
        return self.__n

    def __getitem__(self, item: int) -> str:
        """
        Возвращает 'i'-ый элемент n-граммы, где в роли
        'i' выстудает 'item'.

        Аргументы:
            item: номер элемента n-граммы, который
                необходимо вернуть.

        Возвращаемое значение:
            Заданный элемент n-граммы.
        """
        return self.__ngram[item]

    def __setitem__(self, key: int, value: str) -> None:
        """
        Устанавливает значение переменной 'value' для
        'key'-того элемента n-граммы.

        Аргументы:
            key: Ключ, по которому будет сохранено значение.
            value: Значение, которое будет сохранено.
        """
        self.__ngram[key] = value
