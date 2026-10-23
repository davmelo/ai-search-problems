"""Exemplo de execução do problema das N-Rainhas."""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from problems.n_queens import NQueens
from search_lib import BFS, DFS
from strategies import UCS, AStar, GraphSearch


def main() -> None:
    print("=" * 60)
    print("  PROBLEMA DAS N-RAINHAS (N-Queens) - DEMO")
    print("=" * 60)

    for n in (4, 8):
        print(f"\n--- Tabuleiro {n}x{n} ---")
        problem = NQueens(n)

        # 1. Demonstração dos algoritmos
        for strat_class in (AStar, DFS, BFS, UCS, GraphSearch):
            strategy = strat_class(problem)
            t0 = time.perf_counter()
            result = strategy.search(problem.initial_state)
            dt = (time.perf_counter() - t0) * 1000

            node = result.solution_node
            status = "OK" if node else "Sem solução"
            nodes_exp = result.metrics.nodes_expanded
            cost_str = f"{node.cost:.1f}" if node else "0.0"
            print(
                f"[{strat_class.__name__:>11}] {status} | "
                f"Custo: {cost_str} | "
                f"Nós expandidos: {nodes_exp:>5} | "
                f"Tempo: {dt:>6.2f} ms"
            )

        # 2. Mostra visualmente uma solução
        res = AStar(problem).search(problem.initial_state)
        if res.solution_node:
            print(f"\nVisualização do tabuleiro resolvido para N={n} (A*):")
            print(NQueens.render(res.solution_node.state))


if __name__ == "__main__":
    main()
