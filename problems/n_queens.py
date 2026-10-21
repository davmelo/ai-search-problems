from dataclasses import dataclass
from typing import List, Optional, Tuple

from search_lib.core import Action, Problem, State

@dataclass(frozen=True)
class QueensState(State):
    """queens[r] é a coluna da rainha na linha r; as linhas são preenchidas em ordem."""
    queens: Tuple[int, ...]
    n: int

class PlaceQueen(Action):
    """Coloca uma rainha na coluna `col` da próxima linha livre."""

    def __init__(self, col: int):
        self.col = col

    def apply(self, state: State) -> Optional[Tuple[State, float]]:
        if not isinstance(state, QueensState):
            return None
        row = len(state.queens)
        if row >= state.n or not 0 <= self.col < state.n:
            return None
        for r, c in enumerate(state.queens):
            if c == self.col or abs(c - self.col) == row - r:
                return None  # mesma coluna ou mesma diagonal
        return QueensState(state.queens + (self.col,), state.n), 1.0

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PlaceQueen) and self.col == other.col

    def __hash__(self) -> int:
        return hash(self.col)

    def __repr__(self) -> str:
        return f"RAINHA({self.col})"

class NQueens(Problem):
    """N rainhas em um tabuleiro n x n, sem que nenhuma ataque outra."""

    def __init__(self, n: int = 8):
        if n < 1:
            raise ValueError("Precisa de n >= 1.")
        self.n = n
        self.initial_state = QueensState((), n)
        self._moves = [PlaceQueen(c) for c in range(n)]

    def actions(self, state: State) -> List[Action]:
        if not isinstance(state, QueensState):
            return []
        return [m for m in self._moves if m.apply(state) is not None]

    def is_goal(self, state: State) -> bool:
        return isinstance(state, QueensState) and len(state.queens) == self.n

    def heuristic(self, state: State) -> float:
        """Admissível: cada ação coloca exatamente uma rainha, então faltam n - colocadas."""
        if not isinstance(state, QueensState):
            return 0.0
        return float(self.n - len(state.queens))
