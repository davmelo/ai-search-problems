from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from search_lib.core import Action, Problem, State


@dataclass(frozen=True)
class SlidingState(State):
    """Estado do quebra-cabeça deslizante.

    O tabuleiro é representado como uma tupla linear (row-major), onde o valor 0
    indica o espaço vazio. Estados são imutáveis e hasháveis — exigido pela
    busca em grafo.

    Exemplo para 3×3 resolvido:
        board = (1, 2, 3, 4, 5, 6, 7, 8, 0)
        rows = 3, cols = 3
    """

    board: Tuple[int, ...]
    rows: int
    cols: int


# Mapeamento: direção → (delta_row, delta_col) do espaço até a peça vizinha
_DELTA: dict[str, Tuple[int, int]] = {
    "UP": (1, 0),
    "DOWN": (-1, 0),
    "LEFT": (0, 1),
    "RIGHT": (0, -1),
}


@dataclass(frozen=True)
class SlidingAction(Action):
    """Move a peça adjacente ao espaço vazio, deslizando-a para ocupá-lo.

    A `direction` refere-se ao sentido em que a **peça** se move:
        "UP"    → a peça abaixo do espaço sobe para ele
        "DOWN"  → a peça acima do espaço desce para ele
        "LEFT"  → a peça à direita do espaço desliza para a esquerda
        "RIGHT" → a peça à esquerda do espaço desliza para a direita
    """

    direction: str

    def apply(self, state: State) -> Optional[Tuple[State, float]]:
        """Retorna (novo_estado, custo=1.0) ou None se a ação for inválida."""
        if not isinstance(state, SlidingState):
            return None

        rows, cols = state.rows, state.cols
        blank = state.board.index(0)
        blank_row, blank_col = divmod(blank, cols)

        delta = _DELTA.get(self.direction)
        if delta is None:
            return None

        dr, dc = delta
        piece_row = blank_row + dr
        piece_col = blank_col + dc

        if not (0 <= piece_row < rows and 0 <= piece_col < cols):
            return None  # movimento fora dos limites

        piece = piece_row * cols + piece_col

        # Cria novo tabuleiro trocando espaço e peça (sem alterar o original)
        new_board = list(state.board)
        new_board[blank], new_board[piece] = new_board[piece], new_board[blank]
        return SlidingState(tuple(new_board), rows, cols), 1.0


class SlidingPuzzle(Problem):
    """Quebra-cabeça deslizante parametrizável (N-Puzzle).

    Suporta grades de qualquer dimensão, mas os tamanhos validados e usados
    nos testes são: 3×3 (8-puzzle), 4×4 (15-puzzle) e 4×3 (11-puzzle).

    Parâmetros
    ----------
    initial : tupla linear (row-major) com as peças; 0 representa o espaço.
    rows    : número de linhas da grade (padrão 3).
    cols    : número de colunas da grade (padrão 3).

    O estado objetivo é sempre (1, 2, ..., rows*cols-1, 0).
    """

    def __init__(
        self,
        initial: Tuple[int, ...],
        rows: int = 3,
        cols: int = 3,
    ) -> None:
        if not isinstance(rows, int) or isinstance(rows, bool) or not isinstance(cols, int) or isinstance(cols, bool):
            raise TypeError("rows e cols devem ser inteiros.")
        if rows <= 0 or cols <= 0:
            raise ValueError("rows e cols devem ser estritamente positivos.")
        if rows * cols < 2:
            raise ValueError("O quebra-cabeça deve ter pelo menos 2 células.")

        total = rows * cols
        if len(initial) != total:
            raise ValueError(
                f"Tabuleiro deve ter {total} elementos para uma grade {rows}x{cols}."
            )
        if sorted(initial) != list(range(total)):
            raise ValueError(
                f"Tabuleiro deve conter exatamente os valores 0..{total - 1}."
            )
        self.initial_state: SlidingState = SlidingState(initial, rows, cols)
        goal_board: Tuple[int, ...] = tuple(range(1, total)) + (0,)
        self.goal: SlidingState = SlidingState(goal_board, rows, cols)
        self._rows = rows
        self._cols = cols
        self._goal_coords: Tuple[Tuple[int, int], ...] = tuple(
            divmod(v - 1, cols) if v > 0 else (0, 0) for v in range(total)
        )
        self._all_actions: List[SlidingAction] = [
            SlidingAction(d) for d in ("UP", "DOWN", "LEFT", "RIGHT")
        ]

    # ------------------------------------------------------------------
    # Interface obrigatória de Problem
    # ------------------------------------------------------------------

    def actions(self, state: State) -> List[Action]:
        """Retorna todas as acoes validas para o estado dado."""
        if not isinstance(state, SlidingState):
            return []
        return [a for a in self._all_actions if a.apply(state) is not None]

    def is_goal(self, state: State) -> bool:
        """Verifica se o estado e o objetivo."""
        if not isinstance(state, SlidingState):
            return False
        return state == self.goal

    def heuristic(self, state: State) -> float:
        """Heuristica padrao: distancia de Manhattan (admissivel, >= misplaced)."""
        return self.manhattan_distance(state)

    # ------------------------------------------------------------------
    # Heuristicas
    # ------------------------------------------------------------------

    def misplaced_tiles(self, state: State) -> float:
        """Conta pecas fora do lugar, excluindo o espaco vazio.

        Admissivel: cada peca fora do lugar precisa de pelo menos 1 movimento.
        """
        if not isinstance(state, SlidingState):
            return 0.0
        goal_board = self.goal.board
        return float(
            sum(
                1
                for i, v in enumerate(state.board)
                if v != 0 and v != goal_board[i]
            )
        )

    def manhattan_distance(self, state: State) -> float:
        """Soma das distancias de Manhattan de cada peca ate sua posicao-objetivo.

        Admissivel: cada peca precisa de pelo menos |delta_row| + |delta_col|
        movimentos para chegar a posicao correta. Nunca superestima o custo real.
        Vale 0 no estado objetivo.
        """
        if not isinstance(state, SlidingState):
            return 0.0
        cols = self._cols
        coords = self._goal_coords
        total = 0.0
        for i, v in enumerate(state.board):
            if v == 0:
                continue
            r_tgt, c_tgt = coords[v]
            r_cur, c_cur = divmod(i, cols)
            total += abs(r_cur - r_tgt) + abs(c_cur - c_tgt)
        return total

    # ------------------------------------------------------------------
    # Utilitario: gerador de instancias resolviveis (random walk reverso)
    # ------------------------------------------------------------------

    @staticmethod
    def random_instance(
        rows: int = 3,
        cols: int = 3,
        steps: int = 10,
        seed: Optional[int] = None,
    ) -> "SlidingPuzzle":
        """Cria uma instancia com embaralhamento real de `steps` movimentos.

        Parte do estado-objetivo e aplica `steps` movimentos sem desfazer
        imediatamente o movimento anterior (evitando vai-e-volta), garantindo
        um espalhamento muito mais profundo e realista pelo tabuleiro.
        """
        rng = random.Random(seed)
        total = rows * cols
        goal_board: Tuple[int, ...] = tuple(range(1, total)) + (0,)
        current = SlidingState(goal_board, rows, cols)
        dummy = SlidingPuzzle(goal_board, rows, cols)
        actions = dummy._all_actions

        oposto = {
            "UP": "DOWN",
            "DOWN": "UP",
            "LEFT": "RIGHT",
            "RIGHT": "LEFT",
        }
        ultima_acao: Optional[str] = None

        for _ in range(steps):
            valid = [a for a in actions if a.apply(current) is not None]
            # Evita desfazer o movimento que acabou de ser feito
            if ultima_acao is not None:
                sem_volta = [a for a in valid if a.direction != oposto.get(ultima_acao)]
                candidatas = sem_volta if sem_volta else valid
            else:
                candidatas = valid

            action = rng.choice(candidatas)
            ultima_acao = action.direction
            result = action.apply(current)
            assert result is not None
            current, _ = result

        return SlidingPuzzle(current.board, rows, cols)
