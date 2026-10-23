import pytest

from search_lib import BFS
from strategies import UCS, AStar
from search_lib.core import SearchNode, SearchStrategy, State
from problems.romania import (
    ROMANIA_GRAPH,
    SLD_TO_BUCHAREST,
    MoveToCity,
    Romania,
    RomaniaState,
)

# --- Helper ---------------------------------------------------------------

def resolver(strategy: SearchStrategy, inicial: State) -> SearchNode:
    """Roda a busca e garante (com assert) que houve solução."""
    node = strategy.search(inicial).solution_node
    assert node is not None, "a busca deveria ter encontrado solução"
    return node


# --- Solução ótima ---------------------------------------------------------

def test_ucs_acha_solucao_otima():
    node = resolver(UCS(Romania()), RomaniaState("Arad"))
    assert node.cost == 418


def test_astar_acha_solucao_otima():
    node = resolver(AStar(Romania()), RomaniaState("Arad"))
    assert node.cost == 418


def test_astar_sem_heuristica_acha_solucao_otima():
    node = resolver(AStar(Romania(), heuristic=None), RomaniaState("Arad"))
    assert node.cost == 418


def test_bfs_acha_solucao_nao_otima():
    """BFS minimiza o número de ações (3), não o custo (450)."""
    node = resolver(BFS(Romania()), RomaniaState("Arad"))
    assert node.depth == 3
    assert node.cost == 450


@pytest.mark.parametrize("goal,custo", [("Sibiu", 140), ("Fagaras", 239)])
def test_custos_conhecidos(goal, custo):
    problem = Romania(start="Arad", goal=goal)
    node = resolver(AStar(problem), problem.initial_state)
    assert node.cost == custo


def test_inicio_igual_ao_destino():
    problem = Romania(start="Arad", goal="Arad")
    node = resolver(AStar(problem), problem.initial_state)
    assert node.depth == 0
    assert node.cost == 0.0


# --- Caminho ---------------------------------------------------------------

@pytest.mark.parametrize("strategy", [BFS, UCS, AStar])
def test_caminho_leva_ao_objetivo(strategy):
    problem = Romania()
    node = resolver(strategy(problem), problem.initial_state)
    state, total = problem.initial_state, 0.0
    for action in node.get_path():
        state, cost = action.apply(state)
        total += cost
    assert problem.is_goal(state)
    assert total == node.cost


# --- Ações -----------------------------------------------------------------

def test_acao_invalida_retorna_none():
    state = Romania().initial_state
    assert MoveToCity("Bucharest").apply(state) is None  # sem estrada direta


def test_acao_em_estado_de_outro_tipo_retorna_none():
    assert MoveToCity("Sibiu").apply(object()) is None


def test_acao_em_cidade_inexistente_retorna_none():
    assert MoveToCity("Sibiu").apply(RomaniaState("Atlantis")) is None


def test_acoes_em_cidade_inexistente_retorna_vazio():
    assert Romania().actions(RomaniaState("Atlantis")) == []


def test_apply_nao_altera_estado_original():
    state = Romania().initial_state
    copy = RomaniaState(state.city)
    result = MoveToCity("Timisoara").apply(state)
    assert state == copy
    assert result == (RomaniaState("Timisoara"), 118)


@pytest.mark.parametrize("city,n", [("Neamt", 1), ("Arad", 3), ("Sibiu", 4), ("Bucharest", 4)])
def test_numero_de_acoes_por_cidade(city, n):
    problem = Romania()
    assert len(problem.actions(RomaniaState(city))) == n


# --- Objetivo e construtor -------------------------------------------------

def test_is_goal():
    problem = Romania()
    assert problem.is_goal(RomaniaState("Bucharest"))
    assert not problem.is_goal(RomaniaState("Arad"))


@pytest.mark.parametrize("start,goal", [("Atlantis", "Bucharest"), ("Arad", "Atlantis")])
def test_construtor_rejeita_cidade_invalida(start, goal):
    with pytest.raises(ValueError):
        Romania(start=start, goal=goal)


# --- Heurística ------------------------------------------------------------

def test_heuristica_zero_no_objetivo():
    assert Romania().heuristic(RomaniaState("Bucharest")) == 0


def test_heuristica_zero_para_outros_destinos():
    problem = Romania(start="Arad", goal="Sibiu")
    assert problem.heuristic(RomaniaState("Arad")) == 0


def test_heuristica_cidade_invalida_retorna_zero():
    assert Romania().heuristic(RomaniaState("Atlantis")) == 0.0


@pytest.mark.parametrize("city", sorted(ROMANIA_GRAPH))
def test_heuristica_admissivel(city):
    """h(n) nunca passa do custo ótimo real de n até Bucharest."""
    problem = Romania(start=city, goal="Bucharest")
    node = resolver(UCS(problem), problem.initial_state)
    assert problem.heuristic(problem.initial_state) <= node.cost


def test_heuristica_consistente():
    """h(a) <= custo(a, b) + h(b) para toda estrada."""
    for a, vizinhos in ROMANIA_GRAPH.items():
        for b, custo in vizinhos.items():
            assert SLD_TO_BUCHAREST[a] <= custo + SLD_TO_BUCHAREST[b]


# --- Dados do grafo --------------------------------------------------------

def test_grafo_simetrico():
    for a, vizinhos in ROMANIA_GRAPH.items():
        for b, custo in vizinhos.items():
            assert ROMANIA_GRAPH[b][a] == custo


def test_grafo_tem_20_cidades_e_46_arestas_direcionadas():
    assert len(ROMANIA_GRAPH) == 20
    assert sum(len(v) for v in ROMANIA_GRAPH.values()) == 46


def test_tabela_sld_cobre_as_mesmas_cidades():
    assert set(SLD_TO_BUCHAREST) == set(ROMANIA_GRAPH)
