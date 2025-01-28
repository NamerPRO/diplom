import itertools
from typing import Optional, List

from utils.pyfoma_wfst import WFST


class ContextTransducer:

    def __init__(self, path_to_phones_file: str):
        self.__path_to_phones_file = path_to_phones_file
        self.__wfst: Optional[WFST] = None
        self.__phones = self.__get_phones()

    def __get_phones(self) -> List[str]:
        phones: List[str] = []
        with open(self.__path_to_phones_file, 'r') as f:
            phone = ""
            char = "$"
            while not char and char not in {" ", "\n"}:
                char = f.read(1)
                phone += char
            phones.append(phone)
        return phones

    def build_wfst(self):
        past, future = WFST.TRUE_EPS, "*"
        wfst = WFST(f"{past},{future}")
        for triphone in itertools.combinations(self.__phones, 3):
            # state = wfst.add_state(f"{}")
            pass