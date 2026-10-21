import time
from collections import deque

from search_lib.core import (
    Problem,
    SearchMetrics,
    SearchNode,
    SearchResult,
    SearchStrategy,
    State,
)


class GraphSearch(SearchStrategy):
    """Busca em grafo (sem repetir estados), em largura ou em profundidade.

    mode="bfs" -> fronteira FIFO; mode="dfs" -> fronteira LIFO.
    O estado é marcado como visitado ao ser gerado, então cada estado entra
    na fronteira no máximo uma vez. O BFS continua ótimo com custos unitários;
    o DFS NÃO garante solução ótima.
    """

    MODES = ("bfs", "dfs")

    def __init__(self, problem: Problem, mode: str = "bfs"):
        super().__init__(problem)
        if mode not in self.MODES:
            raise ValueError(f"mode deve ser um de {self.MODES}, recebido: {mode!r}")
        self.mode = mode

    def search(self, initial_state: State) -> SearchResult:
        start = time.perf_counter()
        metrics = SearchMetrics()

        root = SearchNode(initial_state)
        metrics.nodes_generated += 1
        frontier = deque([root])
        visited = {initial_state}
        metrics.max_frontier_size = 1

        while frontier:
            node = frontier.popleft() if self.mode == "bfs" else frontier.pop()
            metrics.nodes_expanded += 1

            if self.problem.is_goal(node.state):
                metrics.execution_time = time.perf_counter() - start
                return SearchResult(solution_node=node, metrics=metrics)

            for action in self.problem.actions(node.state):
                result = action.apply(node.state)
                if result is None:
                    continue
                next_state, step_cost = result
                if next_state in visited:
                    continue
                visited.add(next_state)
                frontier.append(
                    SearchNode(next_state, node, action, node.cost + step_cost)
                )
                metrics.nodes_generated += 1
                metrics.max_frontier_size = max(
                    metrics.max_frontier_size, len(frontier)
                )

        metrics.execution_time = time.perf_counter() - start
        return SearchResult(solution_node=None, metrics=metrics)