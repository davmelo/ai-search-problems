import pytest

from search_lib import BFS, DLS
from problems.hanoi import Hanoi, HanoiState, MoveDisk
from tests.test_helpers import resolver


@pytest.mark.parametrize("n", [1, 2, 3])
def test_bfs_acha_solucao_otima(n: int) -> None:
    problem = Hanoi(n_disks=n)
    node = resolver(BFS(problem), problem.initial_state)
    assert node.depth == 2**n - 1
    assert node.cost == 2**n - 1


def test_dls_com_limite_otimo_acha_solucao() -> None:
    problem = Hanoi(n_disks=3)
    node = resolver(DLS(problem, depth_limit=7), problem.initial_state)
    assert node.depth == 7


def test_dls_com_limite_curto_nao_acha() -> None:
    problem = Hanoi(n_disks=3)
    result = DLS(problem, depth_limit=6).search(problem.initial_state)
    assert result.solution_node is None


def test_caminho_leva_ao_objetivo() -> None:
    problem = Hanoi(n_disks=3)
    node = resolver(BFS(problem), problem.initial_state)
    state = problem.initial_state
    for action in node.get_path():
        result = action.apply(state)
        assert result is not None
        state = result[0]
    assert problem.is_goal(state)


def test_mais_pinos_reduz_solucao() -> None:
    # Com 4 pinos e 3 discos o ótimo é 5 (Frame-Stewart), menor que 2^3-1 = 7.
    problem = Hanoi(n_disks=3, n_pegs=4)
    node = resolver(BFS(problem), problem.initial_state)
    assert node.depth == 5


def test_acao_invalida_retorna_none() -> None:
    problem = Hanoi(n_disks=3)
    state = problem.initial_state
    assert MoveDisk(1, 2).apply(state) is None   # origem vazia
    assert MoveDisk(0, 0).apply(state) is None   # mesmo pino
    result = MoveDisk(0, 1).apply(state)         # disco 1 -> pino 1
    assert result is not None
    assert MoveDisk(0, 1).apply(result[0]) is None  # disco 2 sobre disco 1


def test_apply_nao_altera_estado_original() -> None:
    problem = Hanoi(n_disks=3)
    state = problem.initial_state
    copia = HanoiState(state.pegs)
    MoveDisk(0, 2).apply(state)
    assert state == copia


def test_heuristica_admissivel_e_zero_no_objetivo() -> None:
    problem = Hanoi(n_disks=3)
    assert problem.heuristic(problem.goal) == 0
    assert problem.heuristic(problem.initial_state) <= 2**3 - 1