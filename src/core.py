from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any


class State(ABC):
    """Classe abstrata para representar o estado. Deve ser imutável/hashável."""
    pass


class Action(ABC):
    """Classe abstrata para representação de ações."""
    
    @abstractmethod
    def apply(self, state: State) -> Optional[Tuple[State, float]]:
        """Aplica a ação no estado. Retorna o par (next_state, cost) ou None se for inválida."""
        pass


class SearchNode:
    """Nó interno da árvore de busca. Controla pai, ação geradora, custo acumulado e profundidade."""
    
    def __init__(
        self, 
        state: State, 
        parent: Optional['SearchNode'] = None, 
        action: Optional[Action] = None, 
        cost: float = 0.0
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.depth: int = (parent.depth + 1) if parent is not None else 0

    def get_path(self) -> List[Action]:
        """Reconstrói a lista de ações da raiz até este nó."""
        path = []
        curr = self
        while curr.action is not None:
            path.append(curr.action)
            curr = curr.parent
        return list(reversed(path))


@dataclass
class SearchMetrics:
    """Coleta de métricas de desempenho da busca."""
    nodes_expanded: int = 0
    nodes_generated: int = 0
    max_frontier_size: int = 0
    execution_time: float = 0.0


@dataclass
class SearchResult:
    """Resultado da execução da busca contendo o nó solução e as métricas."""
    solution_node: Optional[SearchNode]
    metrics: SearchMetrics


class Problem(ABC):
    """Representa o espaço de estados, transições e teste de objetivo."""

    @abstractmethod
    def actions(self, state: State) -> List[Action]:
        """Retorna o conjunto de ações aplicáveis em um estado."""
        pass

    @abstractmethod
    def is_goal(self, state: State) -> bool:
        """Verifica se o estado atinge o objetivo."""
        pass

    def heuristic(self, state: State) -> float:
        """Retorna o valor heurístico estimado até o objetivo (opcional)."""
        return 0.0


class SearchStrategy(ABC):
    """Estratégia genérica de busca."""
    
    def __init__(self, problem: Problem):
        self.problem = problem

    @abstractmethod
    def search(self, initial_state: State) -> SearchResult:
        """Executa a busca a partir de um estado inicial."""
        pass