from dataclasses import dataclass
from typing import List, Optional, Tuple

from search_lib.core import Action, Problem, State

_ROADS = [
    ("Oradea", "Zerind", 71),
    ("Oradea", "Sibiu", 151),
    ("Zerind", "Arad", 75),
    ("Arad", "Sibiu", 140),
    ("Arad", "Timisoara", 118),
    ("Timisoara", "Lugoj", 111),
    ("Lugoj", "Mehadia", 70),
    ("Mehadia", "Drobeta", 75),
    ("Drobeta", "Craiova", 120),
    ("Craiova", "Rimnicu Vilcea", 146),
    ("Craiova", "Pitesti", 138),
    ("Sibiu", "Fagaras", 99),
    ("Sibiu", "Rimnicu Vilcea", 80),
    ("Rimnicu Vilcea", "Pitesti", 97),
    ("Fagaras", "Bucharest", 211),
    ("Pitesti", "Bucharest", 101),
    ("Bucharest", "Giurgiu", 90),
    ("Bucharest", "Urziceni", 85),
    ("Urziceni", "Hirsova", 98),
    ("Urziceni", "Vaslui", 142),
    ("Hirsova", "Eforie", 86),
    ("Vaslui", "Iasi", 92),
    ("Iasi", "Neamt", 87),
]

ROMANIA_GRAPH: dict[str, dict[str, int]] = {}
for a, b, cost in _ROADS:
    ROMANIA_GRAPH.setdefault(a, {})[b] = cost
    ROMANIA_GRAPH.setdefault(b, {})[a] = cost

# Distância em linha reta até Bucharest
SLD_TO_BUCHAREST = {
    "Arad": 366, "Bucharest": 0, "Craiova": 160, "Drobeta": 242,
    "Eforie": 161, "Fagaras": 176, "Giurgiu": 77, "Hirsova": 151,
    "Iasi": 226, "Lugoj": 244, "Mehadia": 241, "Neamt": 234,
    "Oradea": 380, "Pitesti": 100, "Rimnicu Vilcea": 193, "Sibiu": 253,
    "Timisoara": 329, "Urziceni": 80, "Vaslui": 199, "Zerind": 374,
}

@dataclass(frozen=True)
class RomaniaState(State):
    """Um estado do problema da Romênia é uma tupla de cidades visitadas, do início até a cidade atual."""
    city: str
    
class MoveToCity(Action):
    """Move da cidade atual para a cidade `dst`."""
    def __init__(self, dst: str):
        self.dst = dst

    def apply(self, state: State) -> Optional[Tuple[State, float]]:
        if not isinstance(state, RomaniaState):
            return None
        cost = ROMANIA_GRAPH[state.city].get(self.dst)
        if cost is None:
            return None  # não há estrada direta
        return RomaniaState(self.dst), cost
    def __eq__(self, other: object) -> bool:
        return isinstance(other, MoveToCity) and self.dst == other.dst

    def __hash__(self) -> int:
        return hash(self.dst)

    def __repr__(self) -> str:
        return f"MOVER({self.dst})"

class Romania(Problem):
    """Problema da Romênia: encontrar o caminho mais curto de uma cidade inicial para uma cidade final."""

    def __init__(self, start: str = "Arad", goal: str = "Bucharest"):
        if start not in ROMANIA_GRAPH or goal not in ROMANIA_GRAPH:
            raise ValueError("Cidades inválidas.")
        self.start = start
        self.goal = goal
        self.initial_state = RomaniaState(start)

    def is_goal(self, state: State) -> bool:
        return isinstance(state, RomaniaState) and state.city == self.goal

    def actions(self, state: State) -> List[Action]:
        if not isinstance(state, RomaniaState):
            return []
        return [MoveToCity(d) for d in ROMANIA_GRAPH[state.city]]

    def heuristic(self, state: State) -> float:
        if not isinstance(state, RomaniaState):
            return 0.0
        if self.goal != "Bucharest":
            return 0.0  # sem tabela para outros destinos: vira custo uniforme
        return float(SLD_TO_BUCHAREST[state.city])
