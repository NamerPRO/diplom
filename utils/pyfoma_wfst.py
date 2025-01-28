from typing import Set, Dict, Tuple, List, Final

from pyfoma import State, FST


class WFST:
    EPS: Final[str] = ''
    TRUE_EPS: Final[str] = 'ε'
    TRANSITION_EPS: Final[Tuple[str, str]] = ('', '')

    def __init__(
            self,
            initial_state_name: str = '0'
    ) -> None:
        self.__wfst = FST()
        self.__states_mapping: Dict[str, State] = {}
        self.__wfst.initialstate.name = initial_state_name
        self.__states_mapping[self.__wfst.initialstate.name] = self.__wfst.initialstate
        self.__unspecified_name_replacement: int = 1

    def set_alphabet(self, alphabet: Set[str]) -> None:
        self.__wfst.alphabet = alphabet

    def expand_alphabet(self, token: str) -> None:
        self.__wfst.alphabet.add(token)

    def add_state(self, name: str = None, ignore_if_exists: bool = True) -> State:
        if name is None:
            name = str(self.__unspecified_name_replacement)
            self.__unspecified_name_replacement += 1
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

    def get_state(self, name: str) -> State | None:
        return self.__states_mapping[name] if name in self.__states_mapping else None

    def mark_as_final(self, name: str, final_weight: float = 0.0) -> None:
        if name not in self.__states_mapping:
            raise ValueError(f'State {name} does not exist. Thus cannot be marked as final.')
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
        if source not in self.__states_mapping or dest not in self.__states_mapping:
            raise ValueError(f'States {source} or/and {dest} do(es) not exist. Thus ark cannot be added.')
        source_state = self.__states_mapping[source]
        dest_state = self.__states_mapping[dest]
        source_state.add_transition(dest_state, label, weight)

    def remove_arc(self, source: str, transition: Tuple[str, str]):
        if source not in self.__states_mapping:
            raise ValueError(f'State {source} does not exist. Thus ark cannot be removed.')
        source_state = self.__states_mapping[source]
        if transition in source_state.transitions:
            source_state.transitions.pop(transition)

    def arcs(self, name: str) -> Set[State]:
        if name not in self.__states_mapping:
            raise ValueError(f'State {name} does not exist. Thus it has no arks.')
        return self.__states_mapping[name].transitions

    def transduce(self, chain: str, weights=True) -> List:
        return list(self.__wfst.generate(chain, weights))

    def view(self) -> None:
        FST.render(self.__wfst)

    def compose(self, other: 'WFST') -> None:
        # noinspection PyUnresolvedReferences
        self.__wfst.compose(other.__wfst)

    def minimize(self) -> None:
        # noinspection PyUnresolvedReferences
        self.__wfst.minimize()

    def determinize(self) -> None:
        # noinspection PyUnresolvedReferences
        self.__wfst = self.__wfst.determinize()

    def remove_epsilons(self) -> None:
        # noinspection PyUnresolvedReferences
        self.__wfst.epsilon_remove()
