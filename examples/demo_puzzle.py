"""Demonstração visual do Sliding Puzzle resolvido por A*.

Execute com:
    python examples/demo_puzzle.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from problems.sliding_puzzle import SlidingPuzzle, SlidingState
from strategies import AStar, UCS, GraphSearch


def desenhar_tabuleiro(state: SlidingState) -> str:
    """Gera uma representação gráfica em ASCII do tabuleiro."""
    rows, cols = state.rows, state.cols
    linha_div = "+" + "---+" * cols
    linhas = [linha_div]

    for r in range(rows):
        linha_pecas = []
        for c in range(cols):
            val = state.board[r * cols + c]
            if val == 0:
                linha_pecas.append("   ")  # espaço vazio destacado
            else:
                linha_pecas.append(f"{val:2d} ")
        linhas.append("|" + "|".join(linha_pecas) + "|")
        linhas.append(linha_div)

    return "\n".join(linhas)


def main():
    print("=" * 60)
    print("      DEMO VISUAL: SLIDING PUZZLE COM BUSCA A*")
    print("=" * 60)

    # Cria uma instância 3x3 a 6 passos do objetivo
    puzzle = SlidingPuzzle.random_instance(rows=3, cols=3, steps=6, seed=42)

    print("\n[ESTADO INICIAL]")
    print(desenhar_tabuleiro(puzzle.initial_state))
    print(f"Heurística Manhattan inicial : {puzzle.manhattan_distance(puzzle.initial_state):.0f}")
    print(f"Heurística Misplaced inicial : {puzzle.misplaced_tiles(puzzle.initial_state):.0f}")

    print("\nExecutando A* com heurística de Manhattan...")
    start = time.perf_counter()
    resultado = AStar(puzzle).search(puzzle.initial_state)
    duracao = time.perf_counter() - start

    node = resultado.solution_node
    if node is None:
        print("Nenhuma solução encontrada!")
        return

    caminho_acoes = node.get_path()
    print(f"\n[SUCESSO] Solução encontrada em {duracao*1000:.2f} ms!")
    print(f"Passos necessários       : {node.depth}")
    print(f"Custo total              : {node.cost}")
    print(f"Nós expandidos           : {resultado.metrics.nodes_expanded}")
    print(f"Nós gerados              : {resultado.metrics.nodes_generated}")
    print(f"Tamanho máx. fronteira   : {resultado.metrics.max_frontier_size}")

    print("\n" + "-" * 60)
    print("PASSO A PASSO DA RESOLUÇÃO:")
    print("-" * 60)

    estado_atual = puzzle.initial_state
    for i, acao in enumerate(caminho_acoes, start=1):
        prox_estado, _ = acao.apply(estado_atual)
        print(f"\nPasso {i}/{len(caminho_acoes)}: Ação -> Mover peça para '{acao.direction}'")
        print(desenhar_tabuleiro(prox_estado))
        print(f"  Dist. Manhattan restante: {puzzle.manhattan_distance(prox_estado):.0f}")
        estado_atual = prox_estado

    print("\nOBJETIVO ALCANÇADO COM SUCESSO!")


if __name__ == "__main__":
    main()
