import pytest

from search_lib import BFS, DLS
from problems.hanoi import Hanoi, HanoiState, MoveDisk


@pytest.mark.parametrize("n", [1, 2, 3])
def test_bfs_acha_solucao_otima(n):
    problem = Hanoi(n_disks=n)
    result = BFS(problem).search(problem.initial_state)
    assert result.solution_node is not None
    assert result.solution_node.depth == 2**n - 1
    assert result.solution_node.cost == 2**n - 1


def test_dls_com_limite_otimo_acha_solucao():
    problem = Hanoi(n_disks=3)
    result = DLS(problem, depth_limit=7).search(problem.initial_state)
    assert result.solution_node is not None
    assert result.solution_node.depth == 7


def test_dls_com_limite_curto_nao_acha():
    problem = Hanoi(n_disks=3)
    result = DLS(problem, depth_limit=6).search(problem.initial_state)
    assert result.solution_node is None


def test_caminho_leva_ao_objetivo():
    problem = Hanoi(n_disks=3)
    node = BFS(problem).search(problem.initial_state).solution_node
    state = problem.initial_state
    for action in node.get_path():
        state, _ = action.apply(state)
    assert problem.is_goal(state)


def test_mais_pinos_reduz_solucao():
    # Com 4 pinos e 3 discos o ótimo é 5 (Frame-Stewart), menor que 2^3-1 = 7.
    problem = Hanoi(n_disks=3, n_pegs=4)
    result = BFS(problem).search(problem.initial_state)
    assert result.solution_node.depth == 5


def test_acao_invalida_retorna_none():
    problem = Hanoi(n_disks=3)
    state = problem.initial_state
    assert MoveDisk(1, 2).apply(state) is None   # origem vazia
    assert MoveDisk(0, 0).apply(state) is None   # mesmo pino
    state, _ = MoveDisk(0, 1).apply(state)       # disco 1 -> pino 1
    assert MoveDisk(0, 1).apply(state) is None   # disco 2 sobre disco 1


def test_apply_nao_altera_estado_original():
    problem = Hanoi(n_disks=3)
    state = problem.initial_state
    copia = HanoiState(state.pegs)
    MoveDisk(0, 2).apply(state)
    assert state == copia


def test_heuristica_admissivel_e_zero_no_objetivo():
    problem = Hanoi(n_disks=3)
    assert problem.heuristic(problem.goal) == 0
    assert problem.heuristic(problem.initial_state) <= 2**3 - 1