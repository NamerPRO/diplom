from collections import defaultdict

def substring_search(string: str, substring: str) -> bool:
    """
    Метод выполняет поиск подстроки в строке согласно алгоритму
    Бойера-Мура-Хорспула. Если подстрока содержится в строке,
    метод возвращает True. Иначе метод возвращает False.

    Аргументы:
        string: Строка, в которой выполняется поиск подстроки.
        substring: Подстрока, которая ищется в строке.

    Возвращаемое значение:
        True, если подстрока содержится в строке. False иначе.
    """
    l = len(substring)
    table = defaultdict(lambda: l)
    for i in range(l - 1):
        table[substring[i]] = l - i - 1
    i = l - 1
    while i < len(string):
        k = 0
        while k < l and substring[l - 1 - k] == string[i - k]:
            k += 1
        if k == l:
            return True
        else:
            i += table[string[i]]
    return False


if __name__ == '__main__':
    print(substring_search('fvkkkk', 'vkkk'))
    print(substring_search('hewllohehewllowllo', 'hello'))
    print(substring_search('hewllohehellowllo', 'hello'))