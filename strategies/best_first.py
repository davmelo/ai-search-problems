import heapq
import itertools
import time
from typing import Callable, Optional

from search_lib.core import (
    Problem,
    SearchMetrics,
    SearchNode,
    SearchResult,
    SearchStrategy,
    State,
)

Heuristic = Callable[[State], float]


class AStar(SearchStrategy):
    """A* em versão graph-search.

    Ordena a fronteira por f(n) = g(n) + h(n). Se `heuristic` não for passada,
    usa `problem.heuristic`. É ótimo quando h é admissível. Se um estado for
    alcançado de novo por um caminho mais barato, ele é reinserido na fronteira,
    então o resultado continua correto mesmo com h inconsistente.

    Em empates de f, prefere o nó com menor h (mais perto do objetivo), o que
    costuma reduzir a quantidade de nós expandidos.

    Observação: `max_frontier_size` conta as entradas do heap, que podem incluir
    entradas obsoletas (estados já reinseridos com custo menor).
    """

    def __init__(self, problem: Problem, heuristic: Optional[Heuristic] = None):
        super().__init__(problem)
        self.heuristic = heuristic if heuristic is not None else problem.heuristic

    def search(self, initial_state: State) -> SearchResult:
        start = time.perf_counter()
        metrics = SearchMetrics()
        counter = itertools.count()  # desempate estável; evita comparar nós

        root = SearchNode(initial_state)
        metrics.nodes_generated += 1
        h0 = self.heuristic(initial_state)
        frontier = [(h0, h0, next(counter), root)]
        best_cost = {initial_state: 0.0}
        metrics.max_frontier_size = 1

        while frontier:
            _, _, _, node = heapq.heappop(frontier)

            # entrada obsoleta: já existe caminho melhor até este estado
            if node.cost > best_cost.get(node.state, float("inf")):
                continue

            metrics.nodes_expanded += 1

            if self.problem.is_goal(node.state):
                metrics.execution_time = time.perf_counter() - start
                return SearchResult(solution_node=node, metrics=metrics)

            for action in self.problem.actions(node.state):
                result = action.apply(node.state)
                if result is None:
                    continue
                next_state, step_cost = result
                new_cost = node.cost + step_cost
                if new_cost >= best_cost.get(next_state, float("inf")):
                    continue

                best_cost[next_state] = new_cost
                child = SearchNode(next_state, node, action, new_cost)
                h = self.heuristic(next_state)
                # desempate: menor f, depois menor h (mais perto do objetivo)
                heapq.heappush(frontier, (new_cost + h, h, next(counter), child))
                metrics.nodes_generated += 1
                metrics.max_frontier_size = max(
                    metrics.max_frontier_size, len(frontier)
                )

        metrics.execution_time = time.perf_counter() - start
        return SearchResult(solution_node=None, metrics=metrics)


class UCS(AStar):
    """Busca de Custo Uniforme: A* com h(n) = 0. Ótima para custos não negativos."""

    def __init__(self, problem: Problem):
        super().__init__(problem, heuristic=lambda state: 0.0)