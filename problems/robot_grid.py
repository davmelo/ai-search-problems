from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, List, Optional, Tuple

from search_lib.core import Action, Problem, State

Position = Tuple[int, int]
DEFAULT_ACTIONS = ("left", "right", "forward")


class Direction(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


@dataclass(frozen=True)
class RobotState(State):
    """Posição e direção; (0, 0) é o canto superior esquerdo do grid."""
    x: int
    y: int
    direction: Direction = Direction.NORTH

    def __post_init__(self) -> None:
        if type(self.direction) is not Direction:
            raise ValueError("direction deve ser um valor de Direction.")


class RobotAction(Action):
    """Gira 90 graus ou avança uma célula. Cada ação custa 1."""

    def __init__(
        self, name: str, width: int, height: int, obstacles: Iterable[Position] = ()
    ) -> None:
        if name not in DEFAULT_ACTIONS or width < 1 or height < 1:
            raise ValueError("Ação desconhecida ou dimensões inválidas.")
        self.name = name
        self.width = width
        self.height = height
        self.obstacles = frozenset(obstacles)
        if any(not self._inside(x, y) for x, y in self.obstacles):
            raise ValueError("Obstáculo fora do grid.")

    def _inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def apply(self, state: State) -> Optional[Tuple[State, float]]:
        if not isinstance(state, RobotState):
            return None
        if not self._inside(state.x, state.y) or (state.x, state.y) in self.obstacles:
            return None
        if self.name in ("left", "right"):
            delta = -1 if self.name == "left" else 1
            direction = Direction((state.direction + delta) % 4)
            return RobotState(state.x, state.y, direction), 1.0

        dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[state.direction]
        nx, ny = state.x + dx, state.y + dy
        if not self._inside(nx, ny) or (nx, ny) in self.obstacles:
            return None  # não pode atravessar paredes ou sair do grid
        return RobotState(nx, ny, state.direction), 1.0

    def __repr__(self) -> str:
        return f"ROBOT({self.name})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, RobotAction)
            and (self.name, self.width, self.height, self.obstacles)
            == (other.name, other.width, other.height, other.obstacles)
        )

    def __hash__(self) -> int:
        return hash((self.name, self.width, self.height, self.obstacles))


class RobotGrid(Problem):
    """Robô em grid com tamanho, obstáculos e ações configuráveis.

    Por padrão, basta chegar à posição objetivo, em qualquer direção.
    """

    def __init__(
        self,
        width: int = 5,
        height: int = 5,
        initial_state: Optional[RobotState] = None,
        goal: Optional[RobotState] = None,
        obstacles: Iterable[Position] = (),
        allowed_actions: Iterable[str] = DEFAULT_ACTIONS,
        require_goal_direction: bool = False,
    ) -> None:
        if width < 1 or height < 1:
            raise ValueError("width e height devem ser positivos.")
        self.initial_state = initial_state or RobotState(0, 0)
        self.goal = goal or RobotState(width - 1, height - 1)
        self.require_goal_direction = require_goal_direction
        self.obstacles = frozenset(obstacles)
        if any(not (0 <= x < width and 0 <= y < height) for x, y in self.obstacles):
            raise ValueError("Obstáculo fora do grid.")
        for state in (self.initial_state, self.goal):
            if not (0 <= state.x < width and 0 <= state.y < height):
                raise ValueError("Início ou objetivo fora do grid.")
            if (state.x, state.y) in self.obstacles:
                raise ValueError("Início e objetivo devem estar em células livres.")
        self._moves = [
            RobotAction(name, width, height, self.obstacles)
            for name in dict.fromkeys(allowed_actions)
        ]

    def actions(self, state: State) -> List[Action]:
        if not isinstance(state, RobotState):
            return []
        return [m for m in self._moves if m.apply(state) is not None]

    def is_goal(self, state: State) -> bool:
        if not isinstance(state, RobotState):
            return False
        return (state.x, state.y) == (self.goal.x, self.goal.y) and (
            not self.require_goal_direction or state.direction == self.goal.direction
        )

    def heuristic(self, state: State) -> float:
        """Manhattan é admissível: ignora os giros e os obstáculos."""
        if not isinstance(state, RobotState):
            return 0.0
        return float(abs(state.x - self.goal.x) + abs(state.y - self.goal.y))
