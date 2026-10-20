from collections import deque
import time

from search_lib.core import (
    SearchNode, 
    SearchStrategy, 
    State, 
    SearchMetrics, 
    SearchResult
)


class BFS(SearchStrategy):
    """Busca em Largura em Árvore (Tree-BFS)."""

    def search(self, initial_state: State) -> SearchResult:
        start_time = time.perf_counter()
        metrics = SearchMetrics()

        root = SearchNode(initial_state)
        metrics.nodes_generated += 1

        frontier = deque([root])
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))

        while frontier:
            node = frontier.popleft()
            metrics.nodes_expanded += 1

            if self.problem.is_goal(node.state):
                metrics.execution_time = time.perf_counter() - start_time
                return SearchResult(solution_node=node, metrics=metrics)

            for action in self.problem.actions(node.state):
                res_action = action.apply(node.state)

                if res_action is not None:
                    next_state, action_cost = res_action
                    child = SearchNode(
                        state=next_state,
                        parent=node,
                        action=action,
                        cost=node.cost + action_cost
                    )
                    metrics.nodes_generated += 1
                    frontier.append(child)
                    metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))

        metrics.execution_time = time.perf_counter() - start_time
        return SearchResult(solution_node=None, metrics=metrics)