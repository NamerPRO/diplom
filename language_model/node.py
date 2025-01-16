from typing import Dict


class Node:

    def __init__(self, value: int = 0) -> None:
        self.value = value
        self.children: Dict[str, 'Node'] = {}

    def __getitem__(self, item: str) -> 'Node':
        return self.children[item]

    def __setitem__(self, key: str, value: 'Node') -> None:
        self.children[key] = value