import heapq
from dataclasses import dataclass
from typing import Set, Dict, Tuple, Optional

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


class WFST:
    __UNSPECIFIED = -1
    EPS = '<eps>'

    @dataclass
    class Transition:
        input_label: str
        output_label: str
        weight: float

    def __init__(self) -> None:
        self.__states: Set[int] = set()

        self.__last_state: int = WFST.__UNSPECIFIED
        self.start_state: int = WFST.__UNSPECIFIED
        self.final_states: Optional[Set[int]] = None

        self.__graph: Dict[int, Tuple[int, WFST.Transition]] = {}

    def add_state(self):
        self.__last_state += 1
        self.__states.add(self.__last_state)
        return self.__last_state

    def has_state(self, state: int) -> bool:
        return state in self.__states

    def has_arc(self, source: int, dest: int) -> bool:
        for state, _ in self.__graph[source]:
            if state == dest:
                return True
        return False

    def is_start(self, state: int) -> bool:
        return self.start_state == state

    def is_final(self, state: int) -> bool:
        return state in self.final_states

    def add_arc(self, source: int, dest: int, transition: Transition) -> None:
        self.__graph[source] = (dest, transition)

    def arcs(self, state: int) -> Tuple[int, Transition]:
        return self.__graph[state]

    def next(self, state: int, input_label: str) -> Tuple[int, str, float]:
        transition: Tuple[int, str, float] = (0, '', -np.inf)
        for next_state, arc in self.arcs(state):
            if arc.input_label in { WFST.EPS, input_label } and transition[2] < arc.weight:
                output_label = '' if arc.output_label == WFST.EPS else arc.output_label
                transition = (next_state, output_label, arc.weight)
        if transition[2] == -np.inf:
            raise ValueError(f'Cannot move from state={state} because no arc specified for input_label={input_label}.')
        return transition

    # def plot(self):
    #     nx_graph = nx.DiGraph(self.__graph)
    #     nx.draw(nx_graph, with_labels=True)
    #     plt.show()
