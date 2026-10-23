from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import pytest

from search_lib import BFS
from search_lib.core import Action, Problem, SearchNode, SearchStrategy, State
from problems.hanoi import Hanoi
from strategies import UCS, AStar, GraphSearch
from tests.test_helpers import resolver

# --- problema de teste: grafo ponderado pequeno ------------------------------
#
#   A --1-- B --1-- D --1-- G        caminho A-B-D-G: 3 passos, custo 3
#   A ------5------ C ------1-- G    caminho A-C-G:   2 passos, custo 6
#
# BFS acha A-C-G (2 passos, custo 6); o ótimo em custo é A-B-D-G (custo 3).

EDGES: Dict[str, Dict[str, float]] = {
    "A": {"B": 1, "C": 5},
    "B": {"A": 1, "D": 1},
    "C": {"A": 5, "G": 1},
    "D": {"B": 1, "G": 1},
    "G": {"C": 1, "D": 1},
}


@dataclass(frozen=True)
class Node(State):
    name: str


class Go(Action):
    def __init__(self, dest: str):
        self.dest = dest

    def apply(self, state: State) -> Optional[Tuple[State, float]]:
        if not isinstance(state, Node):
            return None
        cost = EDGES.get(state.name, {}).get(self.dest)
        return None if cost is None else (Node(self.dest), float(cost))

    def __repr__(self) -> str:
        return f"->{self.dest}"


class WeightedGraph(Problem):
    def __init__(self, goal: str = "G", h: Optional[Mapping[str, float]] = None):
        self.goal = goal
        self._h: Mapping[str, float] = h or {}

    def actions(self, state: State) -> List[Action]:
        if not isinstance(state, Node):
            return []
        return [Go(d) for d in EDGES.get(state.name, {})]

    def is_goal(self, state: State) -> bool:
        return isinstance(state, Node) and state.name == self.goal

    def heuristic(self, state: State) -> float:
        if not isinstance(state, Node):
            return 0.0
        return self._h.get(state.name, 0.0)


# Heurística admissível para o grafo acima (nunca superestima o custo real até G)
H_ADMISSIVEL: Dict[str, float] = {"A": 3, "B": 2, "C": 1, "D": 1, "G": 0}


def nomes(node: SearchNode) -> List[str]:
    caminho = ["A"]
    for a in node.get_path():
        assert isinstance(a, Go)
        caminho.append(a.dest)
    return caminho


def h_por_parametro(state: State) -> float:
    assert isinstance(state, Node)
    return H_ADMISSIVEL[state.name]


def test_bfs_nao_e_otimo_em_custo() -> None:
    node = resolver(BFS(WeightedGraph()), Node("A"))
    assert node.cost == 6 and nomes(node) == ["A", "C", "G"]


def test_ucs_acha_custo_otimo() -> None:
    node = resolver(UCS(WeightedGraph()), Node("A"))
    assert node.cost == 3
    assert nomes(node) == ["A", "B", "D", "G"]


def test_astar_acha_custo_otimo_com_heuristica_admissivel() -> None:
    node = resolver(AStar(WeightedGraph(h=H_ADMISSIVEL)), Node("A"))
    assert node.cost == 3


def test_astar_expande_no_maximo_o_que_ucs_expande() -> None:
    problem = WeightedGraph(h=H_ADMISSIVEL)
    ucs = UCS(problem).search(Node("A")).metrics.nodes_expanded
    astar = AStar(problem).search(Node("A")).metrics.nodes_expanded
    assert astar <= ucs


def test_astar_aceita_heuristica_por_parametro() -> None:
    problem = WeightedGraph()  # sem h no problema
    node = resolver(AStar(problem, heuristic=h_por_parametro), Node("A"))
    assert node.cost == 3


def test_astar_otimo_com_heuristica_admissivel_mas_inconsistente() -> None:
    # h(B)=2 > custo(B->D) + h(D) = 1, logo h é inconsistente (mas admissível).
    # Este teste só confirma o custo final; não força a reabertura de estados.
    h: Dict[str, float] = {"A": 0, "B": 2, "C": 0, "D": 0, "G": 0}
    node = resolver(AStar(WeightedGraph(h=h)), Node("A"))
    assert node.cost == 3


def test_sem_solucao_retorna_none() -> None:
    problem = WeightedGraph(goal="Z")
    estrategias: List[SearchStrategy] = [
        UCS(problem),
        AStar(problem),
        GraphSearch(problem, "bfs"),
    ]
    for strategy in estrategias:
        assert strategy.search(Node("A")).solution_node is None


def test_objetivo_ja_no_estado_inicial() -> None:
    node = resolver(UCS(WeightedGraph()), Node("G"))
    assert node.cost == 0 and node.get_path() == []


def test_graph_search_bfs_acha_menos_passos() -> None:
    node = resolver(GraphSearch(WeightedGraph(), "bfs"), Node("A"))
    assert node.depth == 2


def test_graph_search_dfs_termina_e_acha_uma_solucao() -> None:
    node = resolver(GraphSearch(WeightedGraph(), "dfs"), Node("A"))
    assert isinstance(node.state, Node) and node.state.name == "G"


def test_graph_search_modo_invalido() -> None:
    with pytest.raises(ValueError):
        GraphSearch(WeightedGraph(), "xyz")


# --- Hanoi: o que o tree-BFS não aguenta -------------------------------------

@pytest.mark.parametrize("n", [4, 5])
def test_hanoi_otimo_com_graph_search_e_astar(n: int) -> None:
    problem = Hanoi(n_disks=n)
    inicial = problem.initial_state
    esperado = 2**n - 1
    assert resolver(GraphSearch(problem, "bfs"), inicial).depth == esperado
    assert resolver(UCS(problem), inicial).cost == esperado
    assert resolver(AStar(problem), inicial).cost == esperado


def test_astar_expande_no_maximo_o_que_ucs_em_hanoi() -> None:
    problem = Hanoi(n_disks=4)
    a_star = AStar(problem).search(problem.initial_state)
    ucs = UCS(problem).search(problem.initial_state)
    assert a_star.metrics.nodes_expanded <= ucs.metrics.nodes_expanded