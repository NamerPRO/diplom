import logging
import os
from collections import defaultdict
from typing import List, Dict, Tuple


def read_arpa_lm(path: str) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
    lmdata = (defaultdict(dict), {})
    fdata: List[int] = []
    try:
        with open(path, 'r') as f:
            f.readline()
            line = f.readline()
            while line.strip(' ') != '\n':
                fdata.append(int(line.split('=')[1]))
                line = f.readline()
            for i, n in enumerate(fdata):
                if i > 0:
                    f.readline()
                f.readline()
                for _ in range(n):
                    if i < len(fdata) - 1:
                        prob, ngram, alpha = f.readline().split('\t')
                        lmdata[0][i + 1][ngram] = float(prob)
                        lmdata[1][ngram] = float(alpha)
                    else:
                        prob, ngram = f.readline().split('\t')
                        lmdata[0][i + 1][ngram[:-1]] = float(prob)
    except Exception as e:
        logging.log(logging.ERROR, 'An error occurred while attempting to read ARPA file. Is it valid?')
        raise e
    return lmdata

def write_arpa_lm(path: str, n: int, data: Tuple[Dict[int, Dict[str, float]], Dict[str, float]]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write('\\data\\\n')
            for i in range(n):
                f.write(f'ngram {i + 1}={len(data[0][i + 1])}\n')
            for i in range(n):
                f.write(' \n')
                f.write(f'\\{i + 1}-grams:\n')
                for (ngram, prob) in data[0][i + 1].items():
                    if i < n - 1:
                        backoff = data[1][ngram] if ngram in data[1] else 0
                        f.write(f'{prob}\t{ngram}\t{backoff}\n')
                    else:
                        f.write(f'{prob}\t{ngram}\n')
            f.write(' \n')
            f.write('\\end\\\n')
    except Exception as e:
        logging.log(logging.ERROR, 'An error occurred while attempting to write in ARPA file.')
        raise e