from dataclasses import dataclass
from typing import List, Optional, Tuple

from search_lib.core import Action, Problem, State


@dataclass(frozen=True)
class HanoiState(State):
    """Um pino é uma tupla de discos, do fundo para o topo (disco maior = número maior)."""
    pegs: Tuple[Tuple[int, ...], ...]


class MoveDisk(Action):
    """Move o disco do topo do pino `src` para o pino `dst`."""

    def __init__(self, src: int, dst: int):
        self.src = src
        self.dst = dst

    def apply(self, state: State) -> Optional[Tuple[State, float]]:
        if not isinstance(state, HanoiState):
            return None
        if self.src == self.dst:
            return None
        src_peg = state.pegs[self.src]
        dst_peg = state.pegs[self.dst]
        if not src_peg:
            return None
        disk = src_peg[-1]
        if dst_peg and dst_peg[-1] < disk:
            return None  # não pode pôr disco maior sobre menor

        pegs = list(state.pegs)
        pegs[self.src] = src_peg[:-1]
        pegs[self.dst] = dst_peg + (disk,)
        return HanoiState(tuple(pegs)), 1.0

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MoveDisk) and (self.src, self.dst) == (other.src, other.dst)

    def __hash__(self) -> int:
        return hash((self.src, self.dst))

    def __repr__(self) -> str:
        return f"MOVER({self.src}->{self.dst})"


class Hanoi(Problem):
    """Torre de Hanoi com `n_disks` discos e `n_pegs` pinos (padrão: 3).

    Estado inicial: todos os discos no pino 0. Objetivo: todos no último pino.
    """

    def __init__(self, n_disks: int = 3, n_pegs: int = 3):
        if n_disks < 1 or n_pegs < 3:
            raise ValueError("Precisa de n_disks >= 1 e n_pegs >= 3.")
        self.n_disks = n_disks
        self.n_pegs = n_pegs
        empty = tuple(() for _ in range(n_pegs - 1))
        self.initial_state = HanoiState((tuple(range(n_disks, 0, -1)),) + empty)
        self.goal = HanoiState(empty + (tuple(range(n_disks, 0, -1)),))
        self._moves = [
            MoveDisk(s, d) for s in range(n_pegs) for d in range(n_pegs) if s != d
        ]

    def actions(self, state: State) -> List[Action]:
        if not isinstance(state, HanoiState):
            return []
        return [m for m in self._moves if m.apply(state) is not None]

    def is_goal(self, state: State) -> bool:
        return state == self.goal

    def heuristic(self, state: State) -> float:
        """Admissível: cada disco fora do pino final precisa de pelo menos 1 movimento."""
        if not isinstance(state, HanoiState):
            return 0.0
        return float(self.n_disks - len(state.pegs[-1]))