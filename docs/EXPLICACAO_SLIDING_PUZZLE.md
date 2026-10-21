# Explicação Técnica e Guia do Sliding Puzzle (N-Puzzle)

Este documento apresenta a fundamentação teórica, o guia de execução e a análise matemática formal da implementação do **Sliding Puzzle** (quebra-cabeça deslizante parametrizável), com ênfase no cálculo da **Heurística de Manhattan** em grades retangulares e quadradas.

---

## 1. Guia de Execução

### 1.1 Pré-requisitos e Ambiente

Certifique-se de que o ambiente virtual está configurado e ativado:

```powershell
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Linux / macOS:
source .venv/bin/activate
```

---

### 1.2 Execução dos Testes Automatizados (`pytest`)

A suíte de testes cobre garantias formais de admissibilidade, dominância, consistência, validação de construtor e resolução ótima via A* e Graph Search (BFS).

- **Executar toda a suíte de testes do projeto:**
  ```powershell
  pytest -v
  ```
  *(Atualmente com 98 testes passando em ~0.3s)*

- **Executar exclusivamente a suíte do Sliding Puzzle:**
  ```powershell
  pytest tests/test_sliding_puzzle.py -v
  ```

---

### 1.3 Demonstrações Visuais no Terminal (`demo_puzzle.py`)

O script `examples/demo_puzzle.py` permite visualizar a resolução passo a passo de instâncias embaralhadas, apresentando as métricas de busca (nós gerados, nós expandidos, profundidade e tempo):

- **Grade 3×3 clássica (8-puzzle - padrão):**
  ```powershell
  python examples/demo_puzzle.py
  ```

- **Grade retangular 4×3 (11-puzzle vertical):**
  ```powershell
  python examples/demo_puzzle.py --rows 4 --cols 3
  ```

- **Grade 4×4 clássica (15-puzzle):**
  ```powershell
  python examples/demo_puzzle.py --rows 4 --cols 4
  ```

- **Demonstração comparativa dos três formatos em sequência:**
  ```powershell
  python examples/demo_puzzle.py --all
  ```

---

## 2. A Heurística de Manhattan em Grades Retangulares

### 2.1 Representação de Estado

O estado do tabuleiro é mantido como uma tupla imutável linear (*row-major*) de tamanho $N = \text{rows} \times \text{cols}$, onde o valor `0` representa o espaço vazio.

Para converter qualquer índice linear $i \in [0, N-1]$ para suas coordenadas bidimensionais $(\text{row}, \text{col})$, utilizamos:

$$\text{row} = \lfloor i / \text{cols} \rfloor, \quad \text{col} = i \pmod{\text{cols}}$$

Em Python, essa operação é efetuada diretamente por:
```python
r, c = divmod(i, cols)
```

### 2.2 Coordenadas-Objetivo e Cálculo da Heurística

No estado objetivo canônico do Sliding Puzzle, as peças $1, 2, \dots, N-1$ ocupam os índices lineares $0, 1, \dots, N-2$, e o espaço vazio `0` ocupa o último índice ($N-1$):

- Para qualquer peça numerada $v > 0$, seu índice linear no objetivo é exatamente $v - 1$.
- Suas coordenadas de destino são:
  $$(r_{\text{meta}}, c_{\text{meta}}) = \text{divmod}(v - 1, \text{cols})$$

A distância de Manhattan de uma peça $v$ posicionada atualmente no índice $i$ é a distância em norma $L_1$:

$$d_M(v) = |r_{\text{atual}} - r_{\text{meta}}| + |c_{\text{atual}} - c_{\text{meta}}|$$

A heurística total de Manhattan do estado é a soma das distâncias de todas as peças reais ($v \neq 0$):

$$h(\text{estado}) = \sum_{v \in \text{peças}, v \neq 0} d_M(v)$$

> **Nota Teórica Fundamental:** O espaço vazio (`0`) **nunca** entra na soma da heurística. Mover uma peça física para o espaço vazio já troca ambos de posição em uma única ação de custo 1. Contar o espaço vazio contaria o mesmo deslocamento duas vezes, destruindo a admissibilidade da heurística.

---

## 3. Desmistificando o Equívoco da "Distância 3" do Revisor

### 3.1 O Caso Analisado

Considere o estado relatado no feedback de revisão:

$$\text{board} = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 11)$$

Avaliado para uma grade retangular $4 \times 3$ ($\text{rows} = 4, \text{cols} = 3$):

```
Índices Lineares (4x3):          Estado Atual:                  Estado Objetivo:
[ 0,  1,  2 ]                    [  1,  2,  3 ]                 [  1,  2,  3 ]
[ 3,  4,  5 ]                    [  4,  5,  6 ]                 [  4,  5,  6 ]
[ 6,  7,  8 ]                    [  7,  8,  9 ]                 [  7,  8,  9 ]
[ 9, 10, 11 ]                    [ 10,  0, 11 ]                 [ 10, 11,  0 ]
```

### 3.2 Cálculo Matemático Detalhado

1. **Peças de 1 a 10:**
   - Já estão nos índices $0, 1, \dots, 9$.
   - Suas posições objetivo são os mesmos índices $0, 1, \dots, 9$.
   - Distância de Manhattan de cada uma = $0.0$.

2. **Peça 11:**
   - **Posição atual:** índice linear $11$.
     $$\text{row}_{\text{atual}} = 11 // 3 = 3, \quad \text{col}_{\text{atual}} = 11 \% 3 = 2 \implies (3, 2)$$
   - **Posição objetivo:** índice linear $10$ (pois $v - 1 = 11 - 1 = 10$).
     $$\text{row}_{\text{meta}} = 10 // 3 = 3, \quad \text{col}_{\text{meta}} = 10 \% 3 = 1 \implies (3, 1)$$
   - **Distância de Manhattan:**
     $$d_M(11) = |3 - 3| + |2 - 1| = 0 + 1 = \mathbf{1.0}$$

3. **Espaço vazio (`0`):**
   - Não entra no somatório por definição canônica da heurística.

**Resultado da Heurística:**
$$h(\text{estado}) = 1.0$$

### 3.3 A Prova por Admissibilidade e a Inconsistência da "Distância 3"

Observe o que acontece a partir do estado atual ao aplicar uma única ação:
- A peça `11` desliza para a esquerda (`SlidingAction("LEFT")`), ocupando a posição onde estava o espaço vazio `0`.
- O novo estado torna-se:
  $$(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0)$$
  que é **exatamente o estado objetivo final**!

Portanto:
- O **custo real ótimo** de solução é $h^*(n) = 1$.
- Uma heurística admissível satisfaz, obrigatoriamente:
  $$h(n) \le h^*(n)$$
- Se o algoritmo retornasse **3.0** (como erroneamente sugerido no comentário de revisão):
  $$h(n) = 3.0 > h^*(n) = 1.0$$
  A heurística **superestimaria** o custo real em $200\%$, violando a admissibilidade e tornando impossível ao algoritmo A* garantir a descoberta da solução ótima!

### 3.4 De Onde Veio a Suposição Errada de "Distância 3"?

A suposição de que a distância seria 3 decorre de confusões conceituais recorrentes na transposição de tabuleiros 1D para grades 2D:

1. **Mistura Cruzada de Dimensões (cols=4 no estado e cols=3 na meta):**
   Esta é a explicação matemática exata para o surgimento do número 3: se alguém calcular a posição atual da peça 11 assumindo 4 colunas ($\text{divmod}(11, 4) = (2, 3)$) e a posição objetivo assumindo 3 colunas ($\text{divmod}(10, 3) = (3, 1)$), a distância calculada será:
   $$d = |2 - 3| + |3 - 1| = 1 + 2 = \mathbf{3.0}$$
   Misturar as dimensões do problema dessa forma produz exatamente o valor 3, mas é conceitualmente incorreto porque as dimensões da grade devem ser consistentes tanto no estado atual quanto no estado meta.
2. **Confusão com a Largura da Grade (`cols = 3`):**
   Em uma grade com 3 colunas, mover verticalmente altera o índice linear em $\pm 3$ posições ($\pm \text{cols}$). O revisor pode ter assumido intuitivamente que a presença do fator 3 nas dimensões da grade introduziria uma distância 3, confundindo a diferença de índices lineares com a distância ortogonal 2D.
3. **Confusão de Eixos (Inversão Linha/Coluna):**
   Se alguém calcular erroneamente `divmod(i, rows)` em vez de `divmod(i, cols)`, ou inverter a orientação da matriz, gerará coordenadas arbitrárias incompatíveis com a geometria da grade.
4. **Equívoco de Contagem do Espaço Vazio:**
   O espaço vazio está na célula $(3, 1)$ e sua meta seria $(3, 2)$ (distância 1). Mesmo que alguém cometesse o erro de incluir o espaço vazio na conta, o total seria $1 + 1 = 2$, e jamais 3.

---

## 4. Otimização de Desempenho Implementada

Na versão original, a cada avaliação de `manhattan_distance(state)` era reconstruído em tempo de execução um dicionário mapeando cada valor à sua posição-meta:

```python
# Abordagem anterior (lenta):
goal_pos: dict[int, int] = {v: i for i, v in enumerate(self.goal.board) if v != 0}
```

Como o algoritmo A* avalia a função heurística milhares de vezes durante a busca, essa alocação contínua de memória e geração de dicionários gerava overhead desnecessário de CPU e garbage collection.

### Solução Otimizada:
No construtor `__init__`, as coordenadas objetivo de todas as peças são pré-computadas em uma tupla imutável `_goal_coords`:

```python
self._goal_coords: Tuple[Tuple[int, int], ...] = tuple(
    divmod(v - 1, cols) if v > 0 else (0, 0) for v in range(total)
)
```

No cálculo de Manhattan, o acesso às coordenadas-meta da peça $v$ é feito em tempo constante $O(1)$ direto por indexação na tupla (`coords[v]`), sem alocar nenhuma estrutura adicional:

```python
r_tgt, c_tgt = coords[v]
r_cur, c_cur = divmod(i, cols)
total += abs(r_cur - r_tgt) + abs(c_cur - c_tgt)
```

---

## 5. Nota de Transparência Acadêmica

Em conformidade com as boas práticas de integridade acadêmica e transparência científica:

1. **Uso de Inteligência Artificial:**
   Ferramentas de Inteligência Artificial Generativa foram utilizadas como instrumento de apoio ao estudo, pesquisa bibliográfica teórica sobre algoritmos de busca no espaço de estados (Russel & Norvig, *Artificial Intelligence: A Modern Approach*) e auxílio na elaboração da documentação técnica e estruturação de cenários de teste automatizados.
2. **Autoria e Responsabilidade Técnica:**
   Todas as implementações matemáticas, modelos conceituais, testes de validação unitária e decisões de arquitetura de software foram revisadas, validadas e testadas rigorosamente pelo autor, assegurando total aderência aos requisitos e integridade científica do projeto.
