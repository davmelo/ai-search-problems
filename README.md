   # Problemas de busca para a search_lib (Opção C)

   Problemas de exemplo e testes para validar a `search_lib`
   (https://github.com/GustavoFelipeM/search_lib).

   ## Setup
```bash
   git clone https://github.com/davmelo/ai-search-problems.git
   cd ai-search-problems

   # a lib NÃO pode ficar numa pasta chamada search_lib
   git clone https://github.com/GustavoFelipeM/search_lib.git search_lib_repo

   py -m venv .venv
   source .venv/Scripts/activate   # Git Bash no Windows
   pip install -e ./search_lib_repo pytest matplotlib

   pytest
```

   ## Estrutura
   - `problems/`: um arquivo por problema (State, Action, Problem)
   - `strategies/`: UCS, A* e GraphSearch (além dos BFS/DFS/DLS da lib)
   - `tests/`: um `test_<problema>.py` para cada problema

   ## Regras do grupo
   - Uma branch por problema (`feat/sliding-puzzle`), entrando por pull request
   - Cada problema só é aceito com seu arquivo de teste
   - Não editar nada dentro de `search_lib_repo/` (é de outra equipe)
   - `Action.apply` devolve `(novo_estado, custo)` ou `None`