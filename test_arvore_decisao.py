import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import os
import warnings
from sklearn.tree import DecisionTreeClassifier

# Importa o script da Árvore de Decisão
# Certifique-se de que o arquivo arvore_decisao.py está na mesma pasta
import arvore_decisao

class TestArvoreDecisao(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Executado uma vez no início.
        Filtramos warnings para manter o console limpo (ex: avisos de depreciação do sklearn).
        """
        warnings.filterwarnings("ignore")

    def setUp(self):
        """
        Executado antes de CADA teste.
        Cria dados sintéticos simples para validação rápida das funções.
        """
        # 50 amostras, 5 features
        self.X_dummy = np.random.rand(50, 5)
        # Classes 0 e 1 balanceadas
        self.y_dummy = np.random.choice([0, 1], size=50)
        self.feature_names = [f"Feat_Teste_{i}" for i in range(5)]

    def test_geracao_dados_sinteticos(self):
        """
        OBJETIVO: Garantir que o script gera dados automaticamente (fallback) 
        quando nenhum arquivo é passado.
        """
        # Chama sem caminho de arquivo -> deve cair no make_classification
        X, y, feats = arvore_decisao.carregar_e_processar_dados(caminho_arquivo=None, n_samples=100)
        
        self.assertEqual(X.shape[0], 100, "Deveria ter gerado 100 linhas")
        self.assertEqual(len(y), 100, "O target deve ter 100 elementos")
        # O padrão do script para sintético é 10 features
        self.assertEqual(X.shape[1], 10, "Dados sintéticos padrão devem ter 10 colunas")
        self.assertIsInstance(feats, list)

    @patch('arvore_decisao.pd.read_csv')
    def test_carregar_arquivo_csv(self, mock_read_csv):
        """
        OBJETIVO: Testar o processamento de um CSV sem precisar de um arquivo real.
        Testamos especificamente:
        1. Leitura do pandas (simulada)
        2. One-Hot Encoding (converter texto para número)
        3. Separação X e y
        """
        # Cria um DataFrame falso na memória
        df_fake = pd.DataFrame({
            'Idade': [25, 30, 35, 40],
            'Categoria': ['A', 'B', 'A', 'B'], # Variável categórica para testar get_dummies
            'target': [0, 1, 0, 1]
        })
        mock_read_csv.return_value = df_fake

        X, y, feats = arvore_decisao.carregar_e_processar_dados('arquivo_fantasma.csv')
        
        # A coluna 'Categoria' vira 'Categoria_B' (pois drop_first=True remove a 'A')
        # Então esperamos 2 colunas finais: 'Idade' e 'Categoria_B'
        self.assertEqual(X.shape[1], 2, "Deveria ter 2 features após one-hot encoding")
        self.assertIn('Categoria_B', feats, "A codificação One-Hot falhou ou o nome mudou")
        
        # Garante que a função realmente tentou ler o arquivo
        mock_read_csv.assert_called_once()

    def test_treinamento_com_poda(self):
        """
        OBJETIVO: Teste de Integração do Modelo (Smoke Test).
        Verifica se a função 'treinar_arvore_com_poda' roda do início ao fim,
        executando o Cost Complexity Pruning e retornando um objeto válido.
        """
        # Executa com os dados dummy criados no setUp
        modelo = arvore_decisao.treinar_arvore_com_poda(self.X_dummy, self.y_dummy)
        
        # Verifica se retornou uma Árvore de Decisão do Scikit-Learn
        self.assertIsInstance(modelo, DecisionTreeClassifier, "O retorno deve ser um DecisionTreeClassifier")
        
        # Verifica se o modelo está treinado (possui o atributo 'tree_')
        self.assertTrue(hasattr(modelo, 'tree_'), "O modelo retornado não parece estar treinado (sem atributo tree_)")

    @patch('arvore_decisao.plt.savefig') # Impede salvar arquivo real
    @patch('arvore_decisao.plt.show')    # Impede abrir janela de plot
    @patch('arvore_decisao.plt.close')   # Impede fechar plot real
    def test_visualizar_resultados(self, mock_close, mock_show, mock_savefig):
        """
        OBJETIVO: Garantir que a função de visualização gera os gráficos sem erro de código.
        """
        # Precisamos de um modelo treinado para visualizar
        modelo = arvore_decisao.treinar_arvore_com_poda(self.X_dummy, self.y_dummy)
        
        try:
            arvore_decisao.visualizar_resultados(modelo, self.X_dummy, self.y_dummy, self.feature_names)
        except Exception as e:
            self.fail(f"A função visualizar_resultados falhou com o erro: {e}")
            
        # Verifica se tentou salvar os 2 gráficos (Estrutura e Features)
        self.assertTrue(mock_savefig.call_count >= 2, "Deveria ter tentado salvar pelo menos 2 imagens")

    def test_tratamento_erro_leitura(self):
        """
        OBJETIVO: Verificar se o script avisa o usuário corretamente 
        quando o arquivo não existe ou está corrompido.
        """
        # Simulamos um erro no pandas
        with patch('arvore_decisao.pd.read_csv', side_effect=Exception("Arquivo Inexistente")):
            # Esperamos que a função levante uma Exception com nossa mensagem personalizada
            with self.assertRaises(Exception) as context:
                arvore_decisao.carregar_e_processar_dados('arquivo_ruim.csv')
            
            # Verifica se a mensagem de erro contém o texto esperado do seu script
            self.assertIn("Erro crítico", str(context.exception))

if __name__ == '__main__':
    unittest.main()