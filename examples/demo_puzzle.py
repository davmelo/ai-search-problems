"""Demonstração visual do Sliding Puzzle resolvido por A*.

Suporta todos os tamanhos especificados no projeto:
  - 3x3 (8-puzzle)
  - 4x3 (11-puzzle)
  - 4x4 (15-puzzle)

Exemplos de uso:
    python examples/demo_puzzle.py                 # Roda menu / padrão 3x3
    python examples/demo_puzzle.py --rows 4 --cols 3
    python examples/demo_puzzle.py --rows 4 --cols 4
    python examples/demo_puzzle.py --all           # Demonstra os 3 tamanhos em sequência
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from problems.sliding_puzzle import SlidingPuzzle, SlidingState
from strategies import AStar


def desenhar_tabuleiro(state: SlidingState) -> str:
    """Gera uma representação gráfica em ASCII do tabuleiro com alinhamento dinâmico."""
    rows, cols = state.rows, state.cols
    # Determina a largura de cada célula com base no maior número
    max_val = rows * cols - 1
    cell_w = max(3, len(str(max_val)) + 1)
    
    linha_div = "+" + ("-" * cell_w + "+") * cols
    linhas = [linha_div]

    for r in range(rows):
        linha_pecas = []
        for c in range(cols):
            val = state.board[r * cols + c]
            if val == 0:
                linha_pecas.append(" " * cell_w)  # espaço vazio destacado
            else:
                linha_pecas.append(f"{val:^{cell_w}}")
        linhas.append("|" + "|".join(linha_pecas) + "|")
        linhas.append(linha_div)

    return "\n".join(linhas)


def resolver_e_exibir(rows: int, cols: int, steps: int = 6, seed: int = 42) -> None:
    print("\n" + "=" * 65)
    print(f"   DEMO VISUAL: GRADE {rows}x{cols} ({rows*cols-1}-PUZZLE) RESOLVIDO COM A*")
    print("=" * 65)

    puzzle = SlidingPuzzle.random_instance(rows=rows, cols=cols, steps=steps, seed=seed)

    print("\n[ESTADO INICIAL EMBARALHADO]")
    print(desenhar_tabuleiro(puzzle.initial_state))
    print(f"* Heuristica Manhattan inicial : {puzzle.manhattan_distance(puzzle.initial_state):.0f}")
    print(f"* Heuristica Misplaced inicial : {puzzle.misplaced_tiles(puzzle.initial_state):.0f}")

    print("\nExecutando A* com heuristica de Manhattan...")
    start = time.perf_counter()
    resultado = AStar(puzzle).search(puzzle.initial_state)
    duracao = time.perf_counter() - start

    node = resultado.solution_node
    if node is None:
        print("[ERRO] Nenhuma solucao encontrada!")
        return

    caminho_acoes = node.get_path()
    print(f"\n[SUCESSO] Solucao otima encontrada em {duracao*1000:.2f} ms!")
    print(f"* Passos necessarios       : {node.depth}")
    print(f"* Custo total              : {node.cost}")
    print(f"* Nos expandidos           : {resultado.metrics.nodes_expanded}")
    print(f"* Nos gerados              : {resultado.metrics.nodes_generated}")
    print(f"* Tamanho max. fronteira   : {resultado.metrics.max_frontier_size}")

    print("\n" + "-" * 65)
    print("PASSO A PASSO DA RESOLUCAO (Pecas deslizando ate o objetivo):")
    print("-" * 65)

    estado_atual = puzzle.initial_state
    for i, acao in enumerate(caminho_acoes, start=1):
        prox_estado, _ = acao.apply(estado_atual)
        print(f"\nPasso {i}/{len(caminho_acoes)}: Acao -> Mover peca para '{acao.direction}'")
        print(desenhar_tabuleiro(prox_estado))
        print(f"  -> Distancia Manhattan restante: {puzzle.manhattan_distance(prox_estado):.0f}")
        estado_atual = prox_estado

    print(f"\n[OK] OBJETIVO DA GRADE {rows}x{cols} ALCANCADO COM SUCESSO!\n")


def main():
    parser = argparse.ArgumentParser(description="Demonstração visual do Sliding Puzzle com A*")
    parser.add_argument("--rows", "-r", type=int, default=3, help="Linhas (ex: 3, 4)")
    parser.add_argument("--cols", "-c", type=int, default=3, help="Colunas (ex: 3, 4)")
    parser.add_argument("--steps", "-s", type=int, default=6, help="Passos de embaralhamento (padrão: 6)")
    parser.add_argument("--all", "-a", action="store_true", help="Demonstra todos os 3 tamanhos do trabalho (3x3, 4x3, 4x4)")

    args = parser.parse_args()

    if args.all:
        tamanhos = [(3, 3), (4, 3), (4, 4)]
        for r, c in tamanhos:
            resolver_e_exibir(rows=r, cols=c, steps=args.steps)
    else:
        resolver_e_exibir(rows=args.rows, cols=args.cols, steps=args.steps)


if __name__ == "__main__":
    main()
