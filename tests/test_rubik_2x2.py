import pytest

from search_lib import BFS, DLS
from search_lib.core import SearchNode, SearchStrategy, State
from problems.rubik_2x2 import FACE_ORDER, RotateFace, Rubik2x2, RubikState
from strategies import AStar, GraphSearch, UCS


def resolver(strategy: SearchStrategy, inicial: State) -> SearchNode:
    node = strategy.search(inicial).solution_node
    assert node is not None, "a busca deveria ter encontrado solução"
    return node


def mover(state: RubikState, move: str) -> RubikState:
    result = RotateFace(move).apply(state)
    assert result is not None
    next_state, cost = result
    assert isinstance(next_state, RubikState)
    assert cost == 1.0
    return next_state


@pytest.mark.parametrize("strategy_type", [UCS, AStar, GraphSearch])
def test_solucao_otima_e_caminho(strategy_type: type[SearchStrategy]) -> None:
    initial = mover(mover(RubikState(), "R"), "U")
    problem = Rubik2x2(initial_state=initial)
    node = resolver(strategy_type(problem), initial)
    assert node.cost == node.depth == 2
    state: State = initial
    for action in node.get_path():
        result = action.apply(state)
        assert result is not None
        state, _ = result
    assert problem.is_goal(state)


@pytest.mark.parametrize(
    "move,expected",
    [
        ("R", (0, 2, 0, 2, 1, 1, 1, 1, 2, 3, 2, 3, 3, 5, 3, 5, 4, 4, 4, 4, 0, 5, 0, 5)),
        ("U", (0, 0, 0, 0, 5, 5, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 4, 4, 4, 4, 5, 5)),
        ("F", (0, 0, 4, 4, 0, 1, 0, 1, 2, 2, 2, 2, 1, 1, 3, 3, 4, 3, 4, 3, 5, 5, 5, 5)),
    ],
)
def test_giros_conhecidos(move: str, expected: tuple[int, ...]) -> None:
    assert mover(RubikState(), move).stickers == expected


@pytest.mark.parametrize("face", FACE_ORDER)
def test_giro_inverso_quatro_giros_e_meia_volta(face: str) -> None:
    original = mover(mover(RubikState(), "R"), "U")
    assert mover(mover(original, face), face + "'") == original
    assert mover(original, face + "2") == mover(mover(original, face), face)
    state = original
    for _ in range(4):
        state = mover(state, face)
    assert state == original


def test_giros_preservam_as_pecas_de_canto() -> None:
    corners = (
        (3, 4, 9), (2, 8, 17), (0, 16, 21), (1, 20, 5),
        (13, 11, 6), (12, 19, 10), (14, 23, 18), (15, 7, 22),
    )

    def pieces(state: RubikState) -> set[frozenset[int]]:
        return {frozenset(state.stickers[i] for i in corner) for corner in corners}

    state = RubikState()
    expected = pieces(state)
    for move in ("R", "U'", "F", "D2", "L'", "B"):
        state = mover(state, move)
        assert pieces(state) == expected


def test_acao_invalida_e_imutabilidade() -> None:
    state = RubikState()
    assert RotateFace("R").apply(State()) is None
    mover(state, "R")
    assert state == RubikState()
    assert len({state, RubikState()}) == 1
    with pytest.raises(ValueError):
        RotateFace("X")


def test_heuristica_admissivel_e_zero_no_objetivo() -> None:
    problem = Rubik2x2()
    assert problem.heuristic(problem.goal) == 0
    state = problem.goal
    for move in ("R", "U"):
        state = mover(state, move)
        assert 0 < problem.heuristic(state) <= resolver(UCS(problem), state).cost


def test_acoes_restritas_sem_solucao() -> None:
    problem = Rubik2x2(initial_state=mover(RubikState(), "R"), allowed_moves=("U", "U'"))
    for strategy in (UCS(problem), AStar(problem), GraphSearch(problem)):
        assert strategy.search(problem.initial_state).solution_node is None


def test_objetivo_personalizado_e_tree_search_rasa() -> None:
    goal = mover(RubikState(), "R2")
    problem = Rubik2x2(initial_state=RubikState(), goal=goal, allowed_moves=("R2",))
    assert resolver(BFS(problem), problem.initial_state).cost == 1
    assert resolver(DLS(problem, depth_limit=1), problem.initial_state).cost == 1
    assert DLS(problem, depth_limit=0).search(problem.initial_state).solution_node is None
    assert problem.heuristic(goal) == 0


def test_busca_com_outro_inicio_e_tipo_incompativel() -> None:
    problem = Rubik2x2()
    assert resolver(AStar(problem), mover(RubikState(), "F")).cost == 1
    assert resolver(AStar(problem), problem.goal).cost == 0
    assert problem.initial_state == problem.goal
    assert problem.actions(State()) == []
    assert not problem.is_goal(State())
    assert problem.heuristic(State()) == 0


def test_parametros_invalidos() -> None:
    with pytest.raises(ValueError):
        RubikState(())
    with pytest.raises(ValueError):
        RubikState((0,) * 24)
    with pytest.raises(ValueError):
        Rubik2x2(allowed_moves=("R3",))
