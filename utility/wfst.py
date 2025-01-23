from collections import defaultdict
from typing import Set, Dict, Tuple, List, Final

from pyfoma import State, FST

from language_model.ngram import NGram


class WFST:
    EPS: Final[str] = ''
    TRUE_EPS: Final[str] = 'ε'
    TRANSITION_EPS: Final[Tuple[str, str]] = ('', '')

    def __init__(self, n: int) -> None:
        self.n = n
        self.__wfst = FST()
        self.__states_mapping: Dict[str, State] = defaultdict(State)
        self.__wfst.initialstate.name = NGram(n - 1).as_string()
        self.__states_mapping[self.__wfst.initialstate.name] = self.__wfst.initialstate

    def set_alphabet(self, alphabet: Set[str]) -> None:
        self.__wfst.alphabet = alphabet

    def add_state(self, name: str = None, ignore_if_exists: bool = True) -> State:
        if name in self.__states_mapping:
            if ignore_if_exists:
                return self.__states_mapping[name]
            raise ValueError(f'State {name} already exists.')
        state = State(name=name)
        self.__states_mapping[name] = state
        self.__wfst.states.add(state)
        return state

    @property
    def start_state(self):
        return self.__wfst.initialstate

    def get_state(self, name: str) -> State:
        return self.__states_mapping[name]

    def mark_as_final(self, name: str, final_weight: float = 0.0) -> None:
        state = self.__states_mapping[name]
        state.finalweight = final_weight
        self.__wfst.finalstates.add(state)

    def has_state(self, name: str) -> bool:
        return name in self.__wfst.states

    def has_arc(self, source: str, dest: str) -> bool:
        for state in self.arcs(source):
            if state.name == dest:
                return True
        return False

    def is_start(self, name: str) -> bool:
        return self.__wfst.initialstate.name == name

    def is_final(self, name: str) -> bool:
        return self.__states_mapping[name] in self.__wfst.finalstates

    def add_arc(self, source: str, dest: str, label: Tuple[str, str], weight: float = 0.0) -> None:
        source_state = self.__states_mapping[source]
        dest_state = self.__states_mapping[dest]
        source_state.add_transition(dest_state, label, weight)

    def remove_arc(self, source: str, transition: Tuple[str, str]):
        source_state = self.__states_mapping[source]
        if transition in source_state.transitions:
            source_state.transitions.pop(transition)

    def arcs(self, name: str) -> Set[State]:
        return self.__states_mapping[name].transitions

    def transduce(self, chain: str, weights = True) -> List:
        return list(self.__wfst.generate(chain, weights))

    def view(self) -> None:
        FST.render(self.__wfst)

    def compose(self, other: 'WFST') -> 'WFST':
        composed_wfst = self.__wfst @ other.__wfst
        return composed_wfst

    def minimize(self) -> None:
        # noinspection PyUnresolvedReferences
        self.__wfst.minimize()

    def determinize(self) -> None:
        # noinspection PyUnresolvedReferences
        self.__wfst.determinize()
