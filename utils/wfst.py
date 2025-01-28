import os.path
import re
import webbrowser
from collections import defaultdict
from typing import Final, Dict, Tuple

from graphviz import Source
from rustfst import VectorFst, SymbolTable, Tr
from rustfst.algorithms.minimize import MinimizeConfig
from rustfst.string_paths_iterator import StringPathsIterator


class WFST:
    EPS: Final[str] = '<eps>'
    __SYMBOL_TABLE: Final[SymbolTable] = SymbolTable()

    def __init__(self, start_state_name: str | None = None, render_state_names_str: bool = True) -> None:
        self.__wfst = VectorFst()
        self.__render_state_names_str = render_state_names_str

        self.__states_str_int_mapping: Dict[str, int] = defaultdict(lambda: -1)
        if self.__render_state_names_str:
            self.__states_int_str_mapping: Dict[int, str] = defaultdict(str)

        start_state = self.__wfst.add_state()
        if not start_state_name:
            start_state_name = str(start_state)
        self.__wfst.set_start(start_state)
        self.__states_str_int_mapping[start_state_name] = start_state
        if self.__render_state_names_str:
            self.__states_int_str_mapping[start_state] = start_state_name

        self.__can_render_str_state_names = self.__render_state_names_str

    @property
    def wfst(self) -> VectorFst:
        return self.__wfst

    def add_state(self, state_name: str | None = None, ignore_state_name_exists: bool = True) -> int:
        if state_name in self.__states_str_int_mapping:
            if ignore_state_name_exists:
                return self.__states_str_int_mapping[state_name]
            raise ValueError(f'State {state_name} already exists')
        state = self.__wfst.add_state()
        if not state_name:
            state_name = str(state)
        self.__states_str_int_mapping[state_name] = state
        if self.__render_state_names_str:
            self.__states_int_str_mapping[state] = state_name
        return state

    @property
    def start_state(self) -> int:
        return self.__wfst.start()

    def get_state(self, state_name: str) -> int:
        return self.__states_str_int_mapping[state_name]

    def mark_as_final(self, state_name: str | int, final_weight: float = 0.0) -> None:
        if isinstance(state_name, int):
            self.__wfst.set_final(state_name, final_weight)
            return
        if state_name not in self.__states_str_int_mapping:
            raise ValueError(f'State {state_name} does not exist. Thus cannot be marked as final.')
        state = self.__states_str_int_mapping[state_name]
        self.__wfst.set_final(state, final_weight)

    def has_state(self, state_name: str) -> bool:
        return state_name in self.__states_str_int_mapping

    def has_arc(self, source: str | int, dest: str | int) -> bool:
        source_state = self.__states_str_int_mapping[source] if isinstance(source, str) else source
        dest_state = self.__states_str_int_mapping[dest] if isinstance(dest, str) else dest
        for transition in self.__wfst.trs(source_state):
            if transition.next_state == dest_state:
                return True
        return False

    def is_start(self, state_name: str | int) -> bool:
        if isinstance(state_name, int):
            return self.__wfst.is_start(state_name)
        return self.__wfst.is_start(self.__states_str_int_mapping[state_name])

    def is_final(self, state_name: str | int) -> bool:
        if isinstance(state_name, int):
            return self.__wfst.is_final(state_name)
        return self.__wfst.is_final(self.__states_str_int_mapping[state_name])

    def add_arc(self, source: str | int, dest: str | int, label: Tuple[str, str], weight: float = 0.0) -> None:
        if isinstance(source, str) and source not in self.__states_str_int_mapping or isinstance(dest,
                                                                                                 str) and dest not in self.__states_str_int_mapping:
            raise ValueError(f'States {source} or/and {dest} do(es) not exist. Thus ark cannot be added.')
        source_state = self.__states_str_int_mapping[source] if isinstance(source, str) else source
        dest_state = self.__states_str_int_mapping[dest] if isinstance(dest, str) else dest
        WFST.__SYMBOL_TABLE.add_symbol(label[0])
        WFST.__SYMBOL_TABLE.add_symbol(label[1])
        self.__wfst.add_tr(source_state,
                           Tr(WFST.__SYMBOL_TABLE.find(label[0]), WFST.__SYMBOL_TABLE.find(label[1]), weight,
                              dest_state))

    def paths(self) -> StringPathsIterator:
        return self.__wfst.string_paths()

    def determinize(self) -> None:
        self.__can_render_str_state_names = False
        self.__wfst.determinize()

    def minimize(self, allow_nondet: bool = False) -> None:
        self.__can_render_str_state_names = False
        self.__wfst.minimize(MinimizeConfig(allow_nondet=allow_nondet))

    def remove_epsilons(self) -> None:
        self.__can_render_str_state_names = False
        self.__wfst.rm_epsilon()

    def compose(self, other: 'WFST') -> 'WFST':
        composed_wfst = self.__wfst.compose(other.__wfst)
        wfst = WFST()
        wfst.__wfst = composed_wfst
        wfst.__ilabel_table = None
        wfst.__olabel_table = None
        wfst.__can_render_str_state_names = False
        wfst.__render_state_names_str = False
        return wfst

    def save(self, path: str = './wfst.fst'):
        self.__wfst.write(path)

    def view(self, path: str = './wfst.dot', tmp_file_name: str = './wfst_tmp.dot') -> None:
        if not self.__can_render_str_state_names or not self.__render_state_names_str:
            self.__wfst.draw(path, WFST.__SYMBOL_TABLE, WFST.__SYMBOL_TABLE)
        else:
            def replacer(m):
                text = m.group(1)
                if '/' in text:
                    state, weight = text.split('/', 1)
                    return f'"{self.__states_int_str_mapping[int(state)]}/{weight}"'
                else:
                    return f'"{self.__states_int_str_mapping[int(text)]}"'

            self.__wfst.draw(tmp_file_name, WFST.__SYMBOL_TABLE, WFST.__SYMBOL_TABLE)
            with open(path, 'w') as wf, open(tmp_file_name, 'r') as rf:
                for line in rf:
                    if not line[0].isdigit():
                        wf.write(line)
                        continue
                    replaced_line = re.sub('"(.*)"', replacer, line)
                    wf.write(replaced_line)
            os.remove(tmp_file_name)
        source = Source.from_file(path)
        source.render(os.path.splitext(path)[0], format='pdf', cleanup=True)
        webbrowser.open_new_tab(f'file://{os.path.abspath(f'{os.path.splitext(path)[0]}.pdf')}')
