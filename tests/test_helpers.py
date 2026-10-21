from search_lib.core import SearchNode, SearchStrategy, State


def resolver(strategy: SearchStrategy, inicial: State) -> SearchNode:
    """Roda a busca e garante (com assert) que houve solução.

    Evita o erro de tipo "solution_node pode ser None" nos testes.
    """
    node = strategy.search(inicial).solution_node
    assert node is not None, "a busca deveria ter encontrado solução"
    return node