# Exemplos e Demonstrações Visuais

Este diretório contém scripts de demonstração dos problemas resolvidos pelas buscas.

## Sliding Puzzle (`demo_puzzle.py`)

Demonstra a resolução passo a passo do **Sliding Puzzle** utilizando o algoritmo **A\*** e a **Heurística de Manhattan**, exibindo o tabuleiro sendo resolvido no terminal com métricas completas de nós gerados, expandidos e tempo de execução.

### Como executar:

No terminal (com o ambiente virtual ativo):

- **Grade 3×3 (8-puzzle - padrão):**
  ```powershell
  python examples/demo_puzzle.py
  ```

- **Grade 4×3 (11-puzzle):**
  ```powershell
  python examples/demo_puzzle.py --rows 4 --cols 3
  ```

- **Grade 4×4 (15-puzzle):**
  ```powershell
  python examples/demo_puzzle.py --rows 4 --cols 4
  ```

- **Demonstrar os três tamanhos em sequência:**
  ```powershell
  python examples/demo_puzzle.py --all
  ```

---

## Como executar os Testes Automatizados (`pytest`)

Para validar a integridade de todas as heurísticas e garantias matemáticas:

- **Executar todos os testes do projeto (Hanoi + Sliding Puzzle + Estratégias):**
  ```powershell
  pytest -v
  ```

- **Executar exclusivamente a suíte do Sliding Puzzle:**
  ```powershell
  pytest tests/test_sliding_puzzle.py -v
  ```

