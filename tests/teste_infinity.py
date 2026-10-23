from typing import List

import pytest

from search_lib import DLS
from search_lib.core import SearchStrategy
from problems.hanoi import Hanoi
from problems.infinite import InfiniteNumbers, IntState, Op
from strategies import UCS, AStar, GraphSearch
from tests.test_helpers import resolver


# --- soluções ótimas conhecidas (ops padrão: +1, *2, -3; start = 1) ----------

@pytest.mark.parametrize(
    "target, passos",
    [(1, 0), (10, 4), (7, 4), (25, 6), (100, 8), (0, 3), (-5, 2)],
)
def test_numero_minimo_de_passos(target: int, passos: int) -> None:
    problem = InfiniteNumbers(target)
    inicial = problem.initial_state
    assert resolver(GraphSearch(problem, "bfs"), inicial).depth == passos
    assert resolver(UCS(problem), inicial).cost == passos
    assert resolver(AStar(problem), inicial).cost == passos


def test_caminho_leva_ao_objetivo() -> None:
    problem = InfiniteNumbers(25)
    node = resolver(AStar(problem), problem.initial_state)
    state = problem.initial_state
    for action in node.get_path():
        result = action.apply(state)
        assert result is not None
        state = result[0]
    assert problem.is_goal(state)


def test_start_diferente_de_um() -> None:
    problem = InfiniteNumbers(target=50, start=20)
    esperado = resolver(UCS(problem), problem.initial_state).cost
    assert resolver(AStar(problem), problem.initial_state).cost == esperado
    assert esperado <= 11  # 20 -> 40 (*2) e depois 10 somas de +1


# --- custos diferentes: BFS deixa de ser ótimo --------------------------------

def test_custos_diferentes_ucs_e_astar_batem_e_bfs_nao() -> None:
    ops = (Op("add", 1, cost=1.0), Op("mul", 2, cost=5.0))
    problem = InfiniteNumbers(target=8, ops=ops)
    inicial = problem.initial_state
    # Com *2 custando 5, somar 1 sete vezes (custo 7) vence qualquer caminho que
    # use *2. O de menos passos (1->2->4->8, 3 passos) custa 1 + 5 + 5 = 11.
    assert resolver(UCS(problem), inicial).cost == 7
    assert resolver(AStar(problem), inicial).cost == 7
    bfs = resolver(GraphSearch(problem, "bfs"), inicial)
    assert bfs.depth == 3 and bfs.cost == 11  # menos passos, porém mais caro


# --- espaço infinito -----------------------------------------------------------

def test_espaco_e_ilimitado() -> None:
    problem = InfiniteNumbers(target=10)
    mul = Op("mul", 2)
    state = problem.initial_state
    for _ in range(40):
        assert problem.actions(state)  # sempre há ações disponíveis
        result = mul.apply(state)
        assert result is not None
        state = result[0]
    assert state == IntState(2**40)


def test_objetivo_inalcancavel_dls_termina_sem_solucao() -> None:
    # Só soma 2 a partir de 1: só gera números ímpares, então 4 é inalcançável.
    # BFS/UCS/A* nunca terminariam aqui; o DLS para pelo limite de profundidade.
    problem = InfiniteNumbers(target=4, ops=(Op("add", 2),))
    result = DLS(problem, depth_limit=8).search(problem.initial_state)
    assert result.solution_node is None


def test_dls_com_limite_suficiente_acha_solucao() -> None:
    problem = InfiniteNumbers(target=10)
    result = DLS(problem, depth_limit=4).search(problem.initial_state)
    assert result.solution_node is not None
    assert result.solution_node.depth == 4


# --- ações -------------------------------------------------------------------

def test_operacoes_basicas() -> None:
    estado = IntState(6)
    assert Op("add", 4).apply(estado) == (IntState(10), 1.0)
    assert Op("mul", 3).apply(estado) == (IntState(18), 1.0)
    assert Op("sub", 10).apply(estado) == (IntState(-4), 1.0)
    assert estado == IntState(6)  # estado original intacto


def test_apply_com_estado_de_outro_problema_retorna_none() -> None:
    assert Op("add", 1).apply(Hanoi(2).initial_state) is None


def test_operacao_invalida_levanta_erro() -> None:
    with pytest.raises(ValueError):
        Op("pow", 2)
    with pytest.raises(ValueError):
        Op("add", 1, cost=0)
    with pytest.raises(ValueError):
        InfiniteNumbers(target=5, ops=())


def test_acoes_comparaveis_e_hashaveis() -> None:
    assert Op("add", 1) == Op("add", 1)
    assert Op("add", 1) != Op("add", 2)
    assert len({Op("add", 1), Op("add", 1), Op("mul", 2)}) == 2


# --- heurística --------------------------------------------------------------

def test_heuristica_zero_no_objetivo() -> None:
    problem = InfiniteNumbers(target=37)
    assert problem.heuristic(problem.goal) == 0


@pytest.mark.parametrize("target", [10, 25])
def test_heuristica_admissivel(target: int) -> None:
    """h(v) nunca pode superar a distância real de v até o alvo."""
    problem = InfiniteNumbers(target)
    for valor in range(-5, 30):
        real = resolver(UCS(InfiniteNumbers(target, start=valor)), IntState(valor)).cost
        assert problem.heuristic(IntState(valor)) <= real, (valor, target)


def test_heuristica_respeita_menor_custo() -> None:
    ops = (Op("add", 1, cost=0.5), Op("mul", 2, cost=3.0))
    problem = InfiniteNumbers(target=8, ops=ops)
    # Só valores <= alvo: sem operação "sub", acima do alvo ele é inalcançável
    # (e a busca nunca terminaria).
    for valor in range(-5, 9):
        real = resolver(
            UCS(InfiniteNumbers(8, start=valor, ops=ops)), IntState(valor)
        ).cost
        assert problem.heuristic(IntState(valor)) <= real, valor


def test_astar_com_e_sem_heuristica_chegam_ao_mesmo_custo() -> None:
    problem = InfiniteNumbers(target=100)
    inicial = problem.initial_state
    estrategias: List[SearchStrategy] = [UCS(problem), AStar(problem)]
    custos = {resolver(e, inicial).cost for e in estrategias}
    assert custos == {8.0}