import logging
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Set

from utils.ngram import NGram


def read_arpa_lm(path: str) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float], Set[str], int, float]:
    """
    Считывает данные о языковой модели из ARPA-файла.

    Аргументы:
        path: Путь до ARPA-файла, содержащего информацию о языковой модели.

    Возвращаемое значение:
        Кортеж, содержащий информацию о языковой модели.

    Исключения:
        FileNotFoundError: Если по указанному пути файл отсутствует.
    """
    lmdata = (defaultdict(dict), {})
    fdata: List[int] = []
    vocabulary: Set[str] = set()
    lm_n, reserved_probability = 0, 0.0
    try:
        with open(path, 'r') as f:
            f.readline()
            line = f.readline()
            while line.strip(' ') != '\n':
                fdata.append(int(line.split('=')[1]))
                line = f.readline()
            lm_n = len(fdata)
            for i, n in enumerate(fdata):
                if i > 0:
                    f.readline()
                f.readline()
                for _ in range(n):
                    if i < len(fdata) - 1:
                        prob, ngram, alpha = f.readline().split('\t')
                        if i == 0:
                            vocabulary.add(ngram)
                            if ngram == NGram.get_sys_token('unknown'):
                                reserved_probability = float(prob)
                                continue
                        lmdata[0][i + 1][ngram] = float(prob)
                        lmdata[1][ngram] = float(alpha)
                    else:
                        prob, ngram = f.readline().split('\t')
                        lmdata[0][i + 1][ngram[:-1]] = float(prob)
    except Exception as e:
        logging.log(logging.ERROR, 'Произошла ошибка чтения ARPA-файла. Он валидный?')
        raise e
    return lmdata[0], lmdata[1], vocabulary, lm_n, reserved_probability


def write_arpa_lm(path: str, n: int, data: Tuple[Dict[int, Dict[str, float]], Dict[str, float], float]) -> None:
    """
    Записывает данные о языковой модели в ARPA-файл.

    Аргументы:
        path: Путь до ARPA-файла, в который требуется записать информацию о языковой модели.
        n: Размерность n-грамм в языковой модели, которая будет записана.
        data: Данные о языковой модели, которые необходимо записать.

    Исключения:
        FileNotFoundError: Если по указанному пути файл отсутствует.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write('\\data\\\n')
            for i in range(n):
                count = len(data[0][i + 1])
                if i == 0:
                    count += 1
                f.write(f'ngram {i + 1}={count}\n')
            for i in range(n):
                f.write(' \n')
                f.write(f'\\{i + 1}-grams:\n')
                if i == 0:
                    f.write(f'{data[2]}\t{NGram.get_sys_token('unknown')}\t0\n')
                for (ngram, prob) in data[0][i + 1].items():
                    if i < n - 1:
                        backoff = data[1][ngram] if ngram in data[1] else 0
                        f.write(f'{prob}\t{ngram}\t{backoff}\n')
                    else:
                        f.write(f'{prob}\t{ngram}\n')
            f.write(' \n')
            f.write('\\end\\\n')
    except Exception as e:
        logging.log(logging.ERROR, 'Произошла ошибка записи в ARPA-файл.')
        raise e
