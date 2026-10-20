# 1. clonar o projeto do grupo
git clone <URL-DO-REPO-DO-GRUPO> problems-for-search-libs
cd problems-for-search-libs

# 2. clonar a lib com nome diferente (não pode se chamar search_lib)
git clone https://github.com/GustavoFelipeM/search_lib.git search_lib_repo

# 3. ambiente virtual
py -m venv .venv
source .venv/Scripts/activate      # Git Bash no Windows

# 4. instalar tudo
pip install -e ./search_lib_repo pytest matplotlib

# 5. testar
python -c "from search_lib import BFS, DFS, DLS; print('ok')"
pytest