import pytest

from search_lib import BFS
from strategies import UCS, AStar
from tests.test_strategies import resolver
from problems.n_queens import NQueens, PlaceQueen, QueensState


# --- Helpers ---------------------------------------------------------------

def sem_ataques(queens: tuple[int, ...]) -> bool:
    """True se nenhuma rainha ataca outra (coluna ou diagonal)."""
    for i in range(len(queens)):
        for j in range(i + 1, len(queens)):
            if queens[i] == queens[j] or abs(queens[i] - queens[j]) == j - i:
                return False
    return True


def solucoes(problem: NQueens, state: QueensState):
    """Enumera todos os estados objetivo usando só actions/apply (sem a lib de busca)."""
    if problem.is_goal(state):
        yield state
        return
    for action in problem.actions(state):
        novo_estado, _ = action.apply(state)
        yield from solucoes(problem, novo_estado)


SOLUCAO_4 = QueensState((1, 3, 0, 2), 4)  # uma das 2 soluções de n=4


# --- Construtor ------------------------------------------------------------

@pytest.mark.parametrize("n", [0, -1])
def test_construtor_rejeita_n_invalido(n):
    with pytest.raises(ValueError):
        NQueens(n)


def test_estado_inicial_e_tabuleiro_vazio():
    problem = NQueens(6)
    assert problem.initial_state == QueensState((), 6)


def test_estado_e_hashavel():
    estados = {QueensState((0, 2), 4), QueensState((0, 2), 4)}
    assert len(estados) == 1


# --- PlaceQueen.apply ------------------------------------------------------

def test_apply_valido_devolve_novo_estado_e_custo():
    state = QueensState((0,), 4)
    assert PlaceQueen(2).apply(state) == (QueensState((0, 2), 4), 1.0)


def test_apply_coluna_ocupada_retorna_none():
    assert PlaceQueen(0).apply(QueensState((0,), 4)) is None


def test_apply_diagonal_retorna_none():
    assert PlaceQueen(1).apply(QueensState((0,), 4)) is None


@pytest.mark.parametrize("col", [-1, 4])
def test_apply_coluna_fora_do_tabuleiro_retorna_none(col):
    assert PlaceQueen(col).apply(QueensState((), 4)) is None


def test_apply_tabuleiro_cheio_retorna_none():
    assert PlaceQueen(0).apply(SOLUCAO_4) is None


def test_apply_estado_de_outro_tipo_retorna_none():
    assert PlaceQueen(0).apply(object()) is None


def test_apply_nao_altera_estado_original():
    state = QueensState((0,), 4)
    copia = QueensState((0,), 4)
    novo_estado, _ = PlaceQueen(2).apply(state)
    assert state == copia
    assert novo_estado is not state


def test_acoes_com_mesma_coluna_sao_iguais():
    assert PlaceQueen(2) == PlaceQueen(2)
    assert hash(PlaceQueen(2)) == hash(PlaceQueen(2))
    assert PlaceQueen(2) != PlaceQueen(3)


# --- actions ---------------------------------------------------------------

def test_estado_vazio_tem_n_acoes():
    problem = NQueens(5)
    assert len(problem.actions(problem.initial_state)) == 5


def test_acoes_apos_uma_rainha():
    problem = NQueens(4)
    assert problem.actions(QueensState((0,), 4)) == [PlaceQueen(2), PlaceQueen(3)]


def test_beco_sem_saida_nao_tem_acoes():
    problem = NQueens(4)
    assert problem.actions(QueensState((0, 2), 4)) == []


def test_actions_estado_de_outro_tipo_retorna_vazio():
    assert NQueens(4).actions(object()) == []


# --- is_goal ---------------------------------------------------------------

def test_is_goal():
    problem = NQueens(4)
    assert problem.is_goal(SOLUCAO_4)
    assert not problem.is_goal(QueensState((1, 3, 0), 4))
    assert not problem.is_goal(problem.initial_state)


def test_is_goal_estado_de_outro_tipo():
    assert not NQueens(4).is_goal(object())


# --- Heurística ------------------------------------------------------------

def test_heuristica_vale_zero_no_objetivo():
    assert NQueens(4).heuristic(SOLUCAO_4) == 0.0


@pytest.mark.parametrize("n", [1, 4, 6, 8])
def test_heuristica_no_estado_inicial_vale_n(n):
    problem = NQueens(n)
    assert problem.heuristic(problem.initial_state) == n


def test_heuristica_estado_de_outro_tipo_vale_zero():
    assert NQueens(4).heuristic(object()) == 0.0


@pytest.mark.parametrize("n", [4, 5, 6])
def test_heuristica_admissivel(n):
    """h(inicial) nunca passa do custo real da solução ótima."""
    problem = NQueens(n)
    node = resolver(UCS(problem), problem.initial_state)
    assert problem.heuristic(problem.initial_state) <= node.cost


def test_heuristica_consistente():
    """h(s) <= custo(s, s') + h(s') para toda ação a partir de alguns estados."""
    problem = NQueens(6)
    for state in [problem.initial_state, QueensState((1,), 6), QueensState((1, 3), 6)]:
        for action in problem.actions(state):
            novo_estado, custo = action.apply(state)
            assert problem.heuristic(state) <= custo + problem.heuristic(novo_estado)


# --- Correção: contagem de soluções ----------------------------------------

@pytest.mark.parametrize("n,esperado", [(1, 1), (2, 0), (3, 0), (4, 2), (5, 10), (6, 4), (8, 92)])
def test_numero_de_solucoes(n, esperado):
    problem = NQueens(n)
    todas = list(solucoes(problem, problem.initial_state))
    assert len(todas) == esperado
    assert all(sem_ataques(s.queens) for s in todas)


# --- Busca -----------------------------------------------------------------

@pytest.mark.parametrize("strategy", [BFS, UCS, AStar])
@pytest.mark.parametrize("n", [1, 4, 6, 8])
def test_busca_encontra_solucao_valida(strategy, n):
    problem = NQueens(n)
    node = resolver(strategy(problem), problem.initial_state)
    assert node.depth == n
    assert node.cost == float(n)  # todo caminho até a solução tem n ações

    state = problem.initial_state
    for action in node.get_path():
        state, _ = action.apply(state)
    assert problem.is_goal(state)
    assert sem_ataques(state.queens)


@pytest.mark.parametrize("strategy", [BFS, UCS, AStar])
@pytest.mark.parametrize("n", [2, 3])
def test_busca_sem_solucao_retorna_none(strategy, n):
    problem = NQueens(n)
    assert strategy(problem).search(problem.initial_state).solution_node is None
