import pytest

from search_lib import BFS, DLS
from search_lib.core import SearchNode, SearchStrategy, State
from problems.robot_grid import Direction, RobotAction, RobotGrid, RobotState
from strategies import AStar, GraphSearch, UCS


def resolver(strategy: SearchStrategy, inicial: State) -> SearchNode:
    node = strategy.search(inicial).solution_node
    assert node is not None, "a busca deveria ter encontrado solução"
    return node


@pytest.mark.parametrize("strategy_type", [UCS, AStar, GraphSearch])
def test_solucao_otima_e_caminho(strategy_type: type[SearchStrategy]) -> None:
    problem = RobotGrid(3, 2, initial_state=RobotState(0, 0, Direction.EAST))
    node = resolver(strategy_type(problem), problem.initial_state)
    assert node.cost == node.depth == 4  # dois avanços, giro, avanço
    state: State = problem.initial_state
    for action in node.get_path():
        result = action.apply(state)
        assert result is not None
        state, _ = result
    assert problem.is_goal(state)


def test_objetivo_com_direcao_final() -> None:
    problem = RobotGrid(
        3, 2, initial_state=RobotState(0, 0, Direction.EAST),
        goal=RobotState(2, 1, Direction.WEST), require_goal_direction=True,
    )
    assert resolver(AStar(problem), problem.initial_state).cost == 5
    assert not problem.is_goal(RobotState(2, 1, Direction.SOUTH))


def test_giros_e_avanco() -> None:
    state = RobotState(1, 1)
    assert RobotAction("left", 3, 3).apply(state) == (RobotState(1, 1, Direction.WEST), 1.0)
    assert RobotAction("right", 3, 3).apply(state) == (RobotState(1, 1, Direction.EAST), 1.0)
    for direction, position in zip(Direction, ((1, 0), (2, 1), (1, 2), (0, 1))):
        assert RobotAction("forward", 3, 3).apply(RobotState(1, 1, direction)) == (
            RobotState(*position, direction), 1.0
        )


def test_acao_invalida_retorna_none() -> None:
    assert RobotAction("forward", 2, 2).apply(RobotState(0, 0)) is None
    assert RobotAction("forward", 2, 2, [(1, 0)]).apply(
        RobotState(0, 0, Direction.EAST)
    ) is None
    assert RobotAction("left", 2, 2).apply(RobotState(-1, 0)) is None
    assert RobotAction("forward", 2, 2).apply(State()) is None


def test_apply_nao_altera_estado_original() -> None:
    state = RobotState(1, 1)
    for name in ("left", "right", "forward"):
        RobotAction(name, 3, 3).apply(state)
    assert state == RobotState(1, 1)
    assert len({state, RobotState(1, 1)}) == 1


def test_obstaculos_e_caso_sem_solucao() -> None:
    problem = RobotGrid(
        3, 2, initial_state=RobotState(0, 0, Direction.EAST),
        goal=RobotState(2, 0), obstacles=[(1, 0)],
    )
    assert resolver(UCS(problem), problem.initial_state).cost == 7  # contorna por baixo
    assert len(problem.actions(problem.initial_state)) == 2
    blocked = RobotGrid(3, 1, obstacles=[(1, 0)])
    for strategy in (UCS(blocked), AStar(blocked), GraphSearch(blocked)):
        assert strategy.search(blocked.initial_state).solution_node is None


def test_heuristica_admissivel_e_zero_no_objetivo() -> None:
    problem = RobotGrid(2, 2, require_goal_direction=True)
    assert problem.heuristic(problem.goal) == 0
    for x in range(2):
        for y in range(2):
            for direction in Direction:
                state = RobotState(x, y, direction)
                assert 0 <= problem.heuristic(state) <= resolver(UCS(problem), state).cost


def test_acoes_personalizadas_e_tree_search_rasa() -> None:
    problem = RobotGrid(
        2, 1, initial_state=RobotState(0, 0, Direction.EAST), allowed_actions=("forward",)
    )
    assert resolver(BFS(problem), problem.initial_state).cost == 1
    assert resolver(DLS(problem, depth_limit=1), problem.initial_state).cost == 1
    assert DLS(problem, depth_limit=0).search(problem.initial_state).solution_node is None
    blocked = RobotGrid(2, 1, allowed_actions=())
    assert GraphSearch(blocked).search(blocked.initial_state).solution_node is None


def test_busca_com_outro_inicio_e_tipo_incompativel() -> None:
    problem = RobotGrid(2, 2)
    assert resolver(AStar(problem), problem.goal).cost == 0
    assert problem.initial_state == RobotState(0, 0)
    assert problem.actions(State()) == []
    assert not problem.is_goal(State())
    assert problem.heuristic(State()) == 0


def test_parametros_invalidos() -> None:
    with pytest.raises(ValueError):
        RobotGrid(width=0)
    with pytest.raises(ValueError):
        RobotGrid(2, 2, obstacles=[(2, 0)])
    with pytest.raises(ValueError):
        RobotGrid(2, 2, obstacles=[(0, 0)])
    with pytest.raises(ValueError):
        RobotGrid(allowed_actions=("jump",))
