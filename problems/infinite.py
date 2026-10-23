import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from search_lib.core import Action, Problem, State


@dataclass(frozen=True)
class IntState(State):
    """Um inteiro (positivo, zero ou negativo). Não há limite superior nem inferior."""
    value: int


class Op(Action):
    """Operação aritmética sobre o valor: soma, multiplicação ou subtração de `k`.

    kind: "add" (v + k), "mul" (v * k) ou "sub" (v - k).
    """

    KINDS = ("add", "mul", "sub")

    def __init__(self, kind: str, k: int, cost: float = 1.0):
        if kind not in self.KINDS:
            raise ValueError(f"kind deve ser um de {self.KINDS}, recebido: {kind!r}")
        if cost <= 0:
            raise ValueError("cost deve ser positivo")
        self.kind = kind
        self.k = k
        self.cost = cost

    def apply(self, state: State) -> Optional[Tuple[State, float]]:
        if not isinstance(state, IntState):
            return None
        v = state.value
        if self.kind == "add":
            new_value = v + self.k
        elif self.kind == "mul":
            new_value = v * self.k
        else:
            new_value = v - self.k
        return IntState(new_value), self.cost

    def _key(self) -> Tuple[str, int, float]:
        return (self.kind, self.k, self.cost)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Op) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        symbol = {"add": "+", "mul": "*", "sub": "-"}[self.kind]
        return f"({symbol}{self.k})"


DEFAULT_OPS: Tuple[Op, ...] = (Op("add", 1), Op("mul", 2), Op("sub", 3))


class InfiniteNumbers(Problem):
    """Alcançar `target` a partir de `start` aplicando operações aritméticas.

    O espaço de estados é INFINITO (os inteiros): sempre dá para gerar novos
    estados, e uma busca sem critério de parada pode nunca terminar quando o
    objetivo é inalcançável. Por isso, em testes de "sem solução" use DLS
    (com limite de profundidade) em vez de BFS/UCS/A*.

    Exemplo clássico: start=1, target=10, ops = (+1, *2, -3).
    """

    def __init__(
        self,
        target: int,
        start: int = 1,
        ops: Optional[Sequence[Op]] = None,
    ):
        self.ops: Tuple[Op, ...] = tuple(ops) if ops is not None else DEFAULT_OPS
        if not self.ops:
            raise ValueError("É preciso ao menos uma operação.")
        self.target = target
        self.initial_state = IntState(start)
        self.goal = IntState(target)

        # Constantes usadas pela heurística (calculadas uma vez).
        self._min_cost = min(op.cost for op in self.ops)
        max_mul = max((op.k for op in self.ops if op.kind == "mul"), default=1)
        max_add = max((op.k for op in self.ops if op.kind == "add"), default=0)
        max_sub = max((op.k for op in self.ops if op.kind == "sub"), default=0)
        # De um valor v >= 1, um passo chega a no máximo v * growth.
        self._growth = max(max_mul, 1 + max_add)
        self._max_sub = max_sub

    def actions(self, state: State) -> List[Action]:
        return list(self.ops)

    def is_goal(self, state: State) -> bool:
        return state == self.goal

    def heuristic(self, state: State) -> float:
        """Estimativa admissível do número de passos (vezes o menor custo).

        - Acima do alvo: só operações "sub" reduzem, cada uma em até `max_sub`.
        - Abaixo do alvo (valor positivo): cada passo multiplica o valor por no
          máximo `growth`, então são necessários pelo menos log_growth(alvo/valor)
          passos.
        - Em qualquer outro caso, devolve 0 (sempre seguro).
        """
        if not isinstance(state, IntState):
            return 0.0
        v = state.value
        if v == self.target:
            return 0.0

        steps = 0
        if v > self.target:
            if self._max_sub > 0:
                steps = math.ceil((v - self.target) / self._max_sub)
        elif v >= 1 and self._growth >= 2:
            reach = v
            while reach < self.target:
                reach *= self._growth
                steps += 1
        return steps * self._min_cost