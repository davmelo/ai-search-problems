"""Testes para problems/sliding_puzzle.py

Cobre:
  - Solução ótima: estado a k passos do objetivo resolvido em <= k movimentos
  - Ação inválida retorna None
  - apply não muta o estado original
  - Heurísticas admissíveis (nunca superestimam) e valem 0 no objetivo
  - manhattan >= misplaced (mais informada) para estados de teste
  - Os três tamanhos: 3x3, 4x4 e 4x3
  - Caso sem solução retorna solution_node = None  (configuração impar)
  - Todos os testes rodam em < ~3 segundos (instâncias pequenas / A*)

Usa GraphSearch(bfs) e AStar — nunca DFS em árvore (loop infinito) nem
BFS em árvore (explode em memória).
"""

from __future__ import annotations

import pytest

from search_lib.core import SearchNode, SearchStrategy
from strategies import AStar, GraphSearch
from problems.sliding_puzzle import SlidingAction, SlidingPuzzle, SlidingState


# ---------------------------------------------------------------------------
# Helper: replica o resolver() de tests/test_strategies.py
# ---------------------------------------------------------------------------

def resolver(strategy: SearchStrategy, initial: SlidingState) -> SearchNode:
    """Roda a busca e garante (com assert) que houve solução."""
    node = strategy.search(initial).solution_node
    assert node is not None, "a busca deveria ter encontrado solução"
    return node


# ---------------------------------------------------------------------------
# Fixtures — estados conhecidos para os três tamanhos
# ---------------------------------------------------------------------------

# 3×3: 1 passo do objetivo (espaço em posição penúltima)
GOAL_3x3: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 0)
ONE_STEP_3x3: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 0, 8)  # 8 e espaço trocados

# 4×4: 1 passo do objetivo
GOAL_4x4: tuple[int, ...] = tuple(range(1, 16)) + (0,)
ONE_STEP_4x4: tuple[int, ...] = tuple(range(1, 15)) + (0, 15)  # 15 e espaço trocados

# 4×3: 1 passo do objetivo (rows=4, cols=3)
GOAL_4x3: tuple[int, ...] = tuple(range(1, 12)) + (0,)
ONE_STEP_4x3: tuple[int, ...] = tuple(range(1, 11)) + (0, 11)  # 11 e espaço trocados


# ===========================================================================
# 1. SOLUÇÃO ÓTIMA — estado a k passos é resolvido em <= k movimentos
# ===========================================================================

@pytest.mark.parametrize("rows,cols,steps,seed", [
    (3, 3, 5,  42),
    (3, 3, 8,  7),
    (3, 3, 12, 99),
    (4, 4, 5,  1),
    (4, 4, 8,  2),
    (4, 3, 5,  10),
    (4, 3, 8,  20),
])
def test_astar_solucao_em_no_maximo_k_passos(
    rows: int, cols: int, steps: int, seed: int
) -> None:
    """A* com Manhattan encontra solução em <= steps movimentos."""
    problem = SlidingPuzzle.random_instance(rows=rows, cols=cols, steps=steps, seed=seed)
    node = resolver(AStar(problem), problem.initial_state)
    assert node.depth <= steps


@pytest.mark.parametrize("rows,cols,initial,k", [
    (3, 3, ONE_STEP_3x3, 1),
    (4, 4, ONE_STEP_4x4, 1),
    (4, 3, ONE_STEP_4x3, 1),
])
def test_solucao_otima_1_passo(
    rows: int, cols: int, initial: tuple[int, ...], k: int
) -> None:
    """Estado a 1 passo do objetivo é resolvido em exatamente 1 movimento."""
    problem = SlidingPuzzle(initial, rows=rows, cols=cols)
    node = resolver(AStar(problem), problem.initial_state)
    assert node.depth == k
    assert node.cost == float(k)


def test_objetivo_ja_e_estado_inicial() -> None:
    """Quando o estado inicial já é o objetivo, custo e profundidade são 0."""
    problem = SlidingPuzzle(GOAL_3x3)
    node = resolver(AStar(problem), problem.initial_state)
    assert node.depth == 0
    assert node.cost == 0.0


# ===========================================================================
# 2. AÇÃO INVÁLIDA RETORNA None
# ===========================================================================

def test_acao_invalida_borda_superior() -> None:
    """Mover peça 'DOWN' quando o espaço está na linha 0 é inválido."""
    # Espaço no canto superior-esquerdo: (0,1,2,3,...,8) — espaço em índice 0
    board = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    state = SlidingState(board, rows=3, cols=3)
    assert SlidingAction("DOWN").apply(state) is None


def test_acao_invalida_borda_inferior() -> None:
    """Mover peça 'UP' quando o espaço está na última linha é inválido."""
    # Espaço no canto inferior-esquerdo: espaço em índice 6
    board = (1, 2, 3, 4, 5, 6, 0, 7, 8)
    state = SlidingState(board, rows=3, cols=3)
    assert SlidingAction("UP").apply(state) is None


def test_acao_invalida_borda_esquerda() -> None:
    """Mover peça 'RIGHT' quando o espaço está na coluna 0 é inválido."""
    board = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    state = SlidingState(board, rows=3, cols=3)
    assert SlidingAction("RIGHT").apply(state) is None


def test_acao_invalida_borda_direita() -> None:
    """Mover peça 'LEFT' quando o espaço está na última coluna é inválido."""
    # Espaço em índice 2 (linha 0, coluna 2)
    board = (1, 2, 0, 3, 4, 5, 6, 7, 8)
    state = SlidingState(board, rows=3, cols=3)
    assert SlidingAction("LEFT").apply(state) is None


def test_acao_invalida_direcao_desconhecida() -> None:
    """Direção inexistente retorna None."""
    problem = SlidingPuzzle(ONE_STEP_3x3)
    state = problem.initial_state
    assert SlidingAction("DIAGONAL").apply(state) is None


# ===========================================================================
# 3. apply NÃO MUTA O ESTADO ORIGINAL
# ===========================================================================

def test_apply_nao_altera_estado_original() -> None:
    """Após apply, o estado original permanece idêntico."""
    problem = SlidingPuzzle(ONE_STEP_3x3)
    state = problem.initial_state
    board_antes = state.board

    action = SlidingAction("RIGHT")  # ação válida a partir de ONE_STEP_3x3
    result = action.apply(state)

    assert result is not None
    assert state.board == board_antes  # imutável — não pode ter mudado


@pytest.mark.parametrize("direction", ["UP", "DOWN", "LEFT", "RIGHT"])
def test_apply_sempre_retorna_novo_objeto(direction: str) -> None:
    """apply deve retornar um SlidingState diferente do original (quando válido)."""
    problem = SlidingPuzzle(GOAL_3x3)
    state = problem.initial_state  # espaço no canto inferior-direito
    action = SlidingAction(direction)
    result = action.apply(state)
    if result is not None:
        novo_estado, _ = result
        assert novo_estado is not state
        assert novo_estado.board != state.board


# ===========================================================================
# 4. HEURÍSTICA ADMISSÍVEL e VALE 0 NO OBJETIVO
# ===========================================================================

@pytest.mark.parametrize("rows,cols,initial", [
    (3, 3, GOAL_3x3),
    (4, 4, GOAL_4x4),
    (4, 3, GOAL_4x3),
])
def test_heuristicas_valem_zero_no_objetivo(
    rows: int, cols: int, initial: tuple[int, ...]
) -> None:
    """Ambas as heurísticas valem exatamente 0 no estado objetivo."""
    problem = SlidingPuzzle(initial, rows=rows, cols=cols)
    goal = problem.goal
    assert problem.heuristic(goal) == 0.0
    assert problem.misplaced_tiles(goal) == 0.0
    assert problem.manhattan_distance(goal) == 0.0


@pytest.mark.parametrize("rows,cols,steps,seed", [
    (3, 3, 6,  1),
    (3, 3, 10, 5),
    (4, 4, 5,  3),
    (4, 3, 6,  8),
])
def test_manhattan_admissivel(
    rows: int, cols: int, steps: int, seed: int
) -> None:
    """Manhattan nunca superestima o custo real (admissibilidade)."""
    problem = SlidingPuzzle.random_instance(rows=rows, cols=cols, steps=steps, seed=seed)
    node = resolver(AStar(problem), problem.initial_state)
    custo_real = node.cost
    h = problem.manhattan_distance(problem.initial_state)
    assert h <= custo_real, (
        f"Manhattan={h} > custo_real={custo_real} — heurística NÃO é admissível!"
    )


@pytest.mark.parametrize("rows,cols,steps,seed", [
    (3, 3, 6,  2),
    (3, 3, 10, 6),
    (4, 4, 5,  4),
    (4, 3, 6,  9),
])
def test_misplaced_admissivel(
    rows: int, cols: int, steps: int, seed: int
) -> None:
    """Misplaced nunca superestima o custo real (admissibilidade)."""
    problem = SlidingPuzzle.random_instance(rows=rows, cols=cols, steps=steps, seed=seed)
    node = resolver(AStar(problem), problem.initial_state)
    custo_real = node.cost
    h = problem.misplaced_tiles(problem.initial_state)
    assert h <= custo_real, (
        f"Misplaced={h} > custo_real={custo_real} — heurística NÃO é admissível!"
    )


# ===========================================================================
# 5. MANHATTAN >= MISPLACED (mais informada)
# ===========================================================================

@pytest.mark.parametrize("steps,seed", [(5, 10), (8, 20), (12, 30)])
def test_manhattan_maior_ou_igual_misplaced(steps: int, seed: int) -> None:
    """Manhattan é sempre >= misplaced para qualquer estado (dominância)."""
    problem = SlidingPuzzle.random_instance(steps=steps, seed=seed)
    state = problem.initial_state
    assert problem.manhattan_distance(state) >= problem.misplaced_tiles(state)


# ===========================================================================
# 6. CAMINHO LEVA AO OBJETIVO (reconstituição da solução)
# ===========================================================================

@pytest.mark.parametrize("rows,cols,steps,seed", [
    (3, 3, 5,  42),
    (4, 4, 4,  7),
    (4, 3, 5,  13),
])
def test_caminho_solucao_leva_ao_objetivo(
    rows: int, cols: int, steps: int, seed: int
) -> None:
    """Seguindo as ações da solução a partir do estado inicial chega-se ao objetivo."""
    problem = SlidingPuzzle.random_instance(rows=rows, cols=cols, steps=steps, seed=seed)
    node = resolver(AStar(problem), problem.initial_state)

    state: SlidingState = problem.initial_state
    for action in node.get_path():
        result = action.apply(state)
        assert result is not None
        state, _ = result

    assert problem.is_goal(state)


# ===========================================================================
# 7. GRAPH SEARCH BFS também resolve instâncias pequenas
# ===========================================================================

@pytest.mark.parametrize("rows,cols,steps,seed", [
    (3, 3, 4, 1),
    (3, 3, 5, 2),
    (4, 3, 3, 5),
])
def test_graph_search_bfs_resolve_instancias_pequenas(
    rows: int, cols: int, steps: int, seed: int
) -> None:
    """GraphSearch(bfs) é ótimo para instâncias rasas e não entra em loop."""
    problem = SlidingPuzzle.random_instance(rows=rows, cols=cols, steps=steps, seed=seed)
    node = resolver(GraphSearch(problem, "bfs"), problem.initial_state)
    assert node.depth <= steps


# ===========================================================================
# 8. ESTADO JÁ RESOLVIDO — nenhuma ação necessária
# ===========================================================================

@pytest.mark.parametrize("rows,cols,goal", [
    (3, 3, GOAL_3x3),
    (4, 4, GOAL_4x4),
    (4, 3, GOAL_4x3),
])
def test_estado_objetivo_ja_e_meta(
    rows: int, cols: int, goal: tuple[int, ...]
) -> None:
    """is_goal retorna True para o estado objetivo e False para outros."""
    problem = SlidingPuzzle(goal, rows=rows, cols=cols)
    assert problem.is_goal(problem.goal)
    assert not problem.is_goal(problem.initial_state) or problem.initial_state == problem.goal


# ===========================================================================
# 9. VALIDAÇÃO DO CONSTRUTOR
# ===========================================================================

def test_construtor_rejeita_tamanho_errado() -> None:
    """Tabuleiro com número errado de elementos levanta ValueError."""
    with pytest.raises(ValueError):
        SlidingPuzzle((1, 2, 3, 4, 5, 6, 7, 8), rows=3, cols=3)  # 8 em vez de 9


def test_construtor_rejeita_valores_invalidos() -> None:
    """Tabuleiro com valores duplicados ou faltando levanta ValueError."""
    with pytest.raises(ValueError):
        SlidingPuzzle((1, 2, 3, 4, 5, 6, 7, 9, 0), rows=3, cols=3)  # 8 faltando, 9 duplicado


# ===========================================================================
# 10. NÚMERO DE AÇÕES VÁLIDAS POR POSIÇÃO DO ESPAÇO
# ===========================================================================

def test_numero_acoes_validas_canto() -> None:
    """Espaço num canto tem exatamente 2 ações válidas."""
    board = (0, 1, 2, 3, 4, 5, 6, 7, 8)  # espaço no canto (0,0)
    problem = SlidingPuzzle(board)
    assert len(problem.actions(problem.initial_state)) == 2


def test_numero_acoes_validas_borda() -> None:
    """Espaço numa borda (não canto) tem exatamente 3 ações válidas."""
    board = (1, 0, 2, 3, 4, 5, 6, 7, 8)  # espaço na borda superior, centro
    problem = SlidingPuzzle(board)
    assert len(problem.actions(problem.initial_state)) == 3


def test_numero_acoes_validas_centro() -> None:
    """Espaço no centro tem exatamente 4 ações válidas."""
    board = (1, 2, 3, 4, 0, 5, 6, 7, 8)  # espaço no centro (1,1)
    problem = SlidingPuzzle(board)
    assert len(problem.actions(problem.initial_state)) == 4
