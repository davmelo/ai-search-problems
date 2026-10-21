from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from search_lib.core import Action, Problem, State

FACE_ORDER = ("U", "R", "F", "D", "L", "B")
SOLVED_STICKERS = tuple(color for color in range(6) for _ in range(4))
DEFAULT_MOVES = tuple(face + suffix for face in FACE_ORDER for suffix in ("", "'"))

# Cada ciclo (a, b, c, d) move a->b->c->d->a em um giro horário.
_CYCLES = {
    "U": ((0, 1, 3, 2), (4, 8, 16, 20), (5, 9, 17, 21)),
    "R": ((1, 22, 13, 9), (3, 20, 15, 11), (4, 5, 7, 6)),
    "F": ((2, 4, 13, 19), (3, 6, 12, 17), (8, 9, 11, 10)),
    "D": ((6, 22, 18, 10), (7, 23, 19, 11), (12, 13, 15, 14)),
    "L": ((0, 8, 12, 23), (2, 10, 14, 21), (16, 17, 19, 18)),
    "B": ((0, 18, 15, 5), (1, 16, 14, 7), (20, 21, 23, 22)),
}


@dataclass(frozen=True)
class RubikState(State):
    """24 adesivos nas faces U, R, F, D, L, B, quatro por face.

    Em cada face, ordem de leitura olhando de fora. Cores de 0 a 5.
    A validação confere as cores, sem decidir a solvabilidade física.
    """
    stickers: Tuple[int, ...] = SOLVED_STICKERS

    def __post_init__(self) -> None:
        if type(self.stickers) is not tuple or sorted(self.stickers) != list(SOLVED_STICKERS):
            raise ValueError("Use uma tupla com quatro adesivos de cada cor (0 a 5).")


class RotateFace(Action):
    """Giro horário (R), anti-horário (R') ou meia-volta (R2). Custo 1."""

    def __init__(self, move: str) -> None:
        if not move or move[0] not in FACE_ORDER or move[1:] not in ("", "'", "2"):
            raise ValueError(f"Movimento inválido: {move!r}.")
        self.move = move
        self.face = move[0]
        self.turns = {"": 1, "'": 3, "2": 2}[move[1:]]

    def apply(self, state: State) -> Optional[Tuple[State, float]]:
        if not isinstance(state, RubikState):
            return None
        stickers = list(state.stickers)
        for _ in range(self.turns):
            for a, b, c, d in _CYCLES[self.face]:
                stickers[a], stickers[b], stickers[c], stickers[d] = (
                    stickers[d], stickers[a], stickers[b], stickers[c]
                )
        return RubikState(tuple(stickers)), 1.0

    def __repr__(self) -> str:
        return f"ROTATE({self.move})"


class Rubik2x2(Problem):
    """Cubo 2x2 com movimentos configuráveis e orientação global fixa.

    Por padrão permite quartos de volta nas seis faces, nos dois sentidos.
    """

    def __init__(
        self,
        initial_state: Optional[RubikState] = None,
        allowed_moves: Iterable[str] = DEFAULT_MOVES,
        goal: Optional[RubikState] = None,
    ) -> None:
        self.goal = goal or RubikState()
        self.initial_state = initial_state or self.goal
        self._moves = [RotateFace(move) for move in dict.fromkeys(allowed_moves)]

    def actions(self, state: State) -> List[Action]:
        if not isinstance(state, RubikState):
            return []
        return list(self._moves)

    def is_goal(self, state: State) -> bool:
        return isinstance(state, RubikState) and state == self.goal

    def heuristic(self, state: State) -> float:
        """Admissível: cada giro pode corrigir no máximo 12 adesivos."""
        if not isinstance(state, RubikState):
            return 0.0
        misplaced = sum(a != b for a, b in zip(state.stickers, self.goal.stickers))
        return float((misplaced + 11) // 12)
