Pipeline de Machine Learning - Análise Psicómetrica (RIASEC)

Este repositório contém um conjunto de scripts em Python para processamento de dados e treinamento de diversos algoritmos de Machine Learning. O objetivo é classificar perfis profissionais (Psicologia vs. Outros) com base em respostas do inventário RIASEC.

O projeto foi estruturado para permitir a execução individual de cada modelo ou uma execução em lote (pipeline unificado).

📋 Pré-requisitos

Certifique-se de ter o Python (3.8+) instalado. As dependências necessárias podem ser instaladas via pip:

pip install numpy pandas matplotlib seaborn scikit-learn pygam ordered-set


Nota: As bibliotecas pygam e ordered-set são essenciais para os scripts modelo_gam.py e prediction_rule_ensemble.py, respectivamente.

📊 Fonte dos Dados

Os dados utilizados neste projeto são públicos e foram retirados do Kaggle: https://www.kaggle.com/datasets/lucasgreenwell/holland-code-riasec-test-responses

O dataset contém respostas ao teste de personalidade RIASEC, que categoriza interesses profissionais em seis dimensões:

Realistic (Realista)

Investigative (Investigativo)

Artistic (Artístico)

Social (Social)

Enterprising (Empreendedor)

Conventional (Convencional)

O script de limpeza processa estes dados brutos para focar especificamente na distinção entre profissionais da área de Psicologia e outras áreas.

🚀 Passo 1: Preparação dos Dados

Antes de treinar os modelos, é necessário processar o arquivo bruto (data.csv) para gerar o dataset limpo e unificado que será utilizado por todos os algoritmos.

Certifique-se de que o arquivo bruto está no local correto (ex: tp3/data.csv) ou ajuste o caminho dentro do script limpeza_dados.py.

Execute o script de limpeza:

python limpeza_dados.py


O que este script faz:

Filtra profissionais com graduação completa.

Cria a variável alvo (target): 1 para Psicólogos, 0 para Outros.

Calcula os scores somados das dimensões R, I, A, S, E, C.

Saída: Gera o arquivo dataset_limpo_completo.tsv.

⚡ Passo 2: Treinar Todos os Modelos (Pipeline Unificado)

Para rodar todos os algoritmos em sequência utilizando o dataset limpo gerado no passo anterior, utilize o script orquestrador.

Certifique-se de que o arquivo dataset_limpo_completo.tsv está na mesma pasta (ou ajuste o caminho na variável CAMINHO_ARQUIVO dentro do script).

python main_treinar_todos.py


Este script irá:

Carregar o dataset limpo.

Executar cada algoritmo (KNN, Random Forest, GBM, GAM, etc.) sequencialmente.

Salvar gráficos de performance e relatórios .txt para cada modelo.

Exibir o progresso e erros no terminal.

🛠️ Passo 3: Executar Modelos Individualmente

Você pode rodar cada algoritmo isoladamente. Todos os scripts suportam o argumento --arquivo.

Sintaxe Básica

python nome_do_script.py --arquivo caminho/do/arquivo.tsv


Exemplos Específicos

1. Random Forest:

python random_forest.py --arquivo dataset_limpo_completo.tsv


2. Regressão Penalizada (Lasso/Ridge):
Este script aceita um argumento opcional para o nome da coluna alvo (padrão é 'target').

python regressao_penalizada.py --arquivo dataset_limpo_completo.tsv --target target


3. RuleFit Ensemble:
Pode ser rodado em modo de classificação (classify) ou regressão (regress).

python prediction_rule_ensemble.py --arquivo dataset_limpo_completo.tsv --modo classify


4. k-Nearest Neighbors (kNN):

python knn.py --arquivo dataset_limpo_completo.tsv


Modo de Teste (Dados Sintéticos)

Se você rodar qualquer script sem passar o argumento --arquivo, ele irá gerar dados sintéticos automaticamente para fins de teste de código.

# Roda com dados falsos gerados na hora
python gbte.py 


🧪 Testes Automatizados

O projeto inclui uma suite de testes desenvolvida com o framework unittest para garantir a integridade do processamento de dados e a funcionalidade básica dos modelos.

Para executar todos os testes disponíveis no projeto, utilize o comando de descoberta do unittest na raiz do repositório:

python -m unittest discover


Se os testes estiverem numa pasta específica (ex: tests/), o comando ajusta-se automaticamente ou pode ser especificado:

python -m unittest discover -s tests -p "test_*.py"


📂 Estrutura dos Arquivos

limpeza_dados.py: Script de ETL (Extração, Transformação e Carga).

main_treinar_todos.py: Orquestrador que chama todos os modelos.

Modelos:

knn.py: k-Nearest Neighbors.

random_forest.py: Random Forest Classifier.

gbte.py: HistGradientBoosting (Gradient Boosting Tree Ensemble).

arvore_decisao.py: Árvore de Decisão única com poda.

regressao_penalizada.py: Regressão Logística com penalidade L1/L2.

modelo_gam.py: Modelos Aditivos Generalizados (GAMs).

prediction_rule_ensemble.py: RuleFit (Regras + Modelo Linear).

📊 Saídas e Resultados

Ao final da execução, os scripts gerarão na pasta raiz:

Imagens PNG: Gráficos de importância das variáveis, curvas ROC, matrizes de confusão e estrutura de árvores.

Relatórios TXT: Métricas detalhadas (Acurácia, Recall, Precision, F1-Score).
