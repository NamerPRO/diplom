import numpy as np

eps = 1e-10


def log_sum(x, y):
    """
    Вычисляет -log(x+y) для x и y.

    Аргументы:
        x: Первый компонент суммы.
        y: Второй компонент суммы.

    Возвращаемое значение:
        -log(x+y) для заданных x и y.
    """
    if x == np.inf and y == np.inf:
        return np.inf
    return min(x, y) - np.log(1 + np.exp(-np.abs(x - y)))


def log_sum_arr(x):
    """
    Вычисляет -log(x_1+...+x_n), где x_i - i-ый элемент
    списка x.

    Аргументы:
        x: список, логарифмическую сумму элементнов которого
            нужно найти.

    Возвращаемое значение:
        -log(x_1+...+x_n) для списка x.
    """
    lsum = np.inf
    for j in x:
        lsum = log_sum(lsum, j + eps)
    return lsum


def negative_log_sum(x, y):
    """
    Вычисляет log(x+y) для x и y.

    Аргументы:
        x: Первый компонент суммы.
        y: Второй компонент суммы.

    Возвращаемое значение:
        log(x+y) для заданных x и y.
    """
    if x == -np.inf and y == -np.inf:
        return -np.inf
    if x > y:
        return x + np.log(1 + np.exp(y - x))
    return y + np.log(1 + np.exp(x - y))
