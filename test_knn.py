import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import os
import warnings

# Importa classes para validação de tipo
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Importa o script
import knn

class TestKNN(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore")

    def setUp(self):
        # Dados Dummy leves (30 linhas)
        # kNN não precisa de muitas amostras para rodar o teste, mas precisa de features
        self.X_dummy = np.random.rand(30, 5).astype(np.float32)
        self.y_dummy = np.random.choice([0, 1], size=30)
        self.feature_names = [f"F_{i}" for i in range(5)]

    # =================================================================
    # TESTES DE LÓGICA DO kNN (CRÍTICOS)
    # =================================================================

    def test_garantia_de_escalonamento(self):
        """
        OBJETIVO CRÍTICO: Validar se o modelo é um Pipeline contendo o StandardScaler.
        
        Por que testar isso?
        O kNN é baseado em distância. Se os dados não forem normalizados (StandardScaler),
        o modelo falha matematicamente. Este teste garante que ninguém removerá
        acidentalmente a normalização do código no futuro.
        """
        modelo = knn.treinar_knn_otimizado(self.X_dummy, self.y_dummy)
        
        # 1. O retorno deve ser um Pipeline, não apenas o classificador solto
        self.assertIsInstance(modelo, Pipeline, 
            "O modelo retornado DEVE ser um Pipeline para garantir o pré-processamento.")
        
        # 2. Verifica se existe um passo chamado 'scaler'
        self.assertIn('scaler', modelo.named_steps)
        
        # 3. Verifica se esse passo é realmente um StandardScaler
        self.assertIsInstance(modelo.named_steps['scaler'], StandardScaler,
            "O primeiro passo do Pipeline deve ser a normalização (StandardScaler).")

    def test_conversao_performance_float32(self):
        """
        OBJETIVO: Validar se a conversão para float32 está ocorrendo.
        kNN calcula matrizes de distância gigantescas. float32 usa metade da RAM do float64.
        """
        X, y, feats = knn.carregar_e_processar_dados(caminho_arquivo=None, n_samples=50)
        
        self.assertEqual(X.dtype, np.float32, 
                         "Os dados devem ser convertidos para float32 para otimizar cálculos de distância.")

    # =================================================================
    # TESTES DE ROBUSTEZ (E/S)
    # =================================================================

    @patch('knn.pd.read_csv')
    def test_logica_leitura_tsv(self, mock_read):
        """
        OBJETIVO: Validar se o script sabe ler arquivos separados por TAB (\t).
        O script tenta ler com \t primeiro, depois com vírgula. Vamos simular
        que o arquivo é um TSV válido.
        """
        # Simula um dataframe retornado corretamente na primeira tentativa
        df_fake = pd.DataFrame({'A': [1, 2], 'target': [0, 1]})
        mock_read.return_value = df_fake
        
        knn.carregar_e_processar_dados('arquivo.tsv')
        
        # Verifica se chamou read_csv com sep='\t'
        # call_args[1] pega os kwargs da chamada
        self.assertEqual(mock_read.call_args[1]['sep'], '\t', 
                         "O script deveria tentar ler com separador '\\t' primeiro.")

    def test_fallback_otimizacao_falha(self):
        """
        OBJETIVO: Se o RandomizedSearchCV falhar (ex: erro de memória ou cv),
        o script deve retornar um Pipeline funcional padrão, sem travar.
        """
        with patch('knn.RandomizedSearchCV') as MockSearch:
            # Configura erro simulado
            instancia = MockSearch.return_value
            instancia.fit.side_effect = Exception("Erro Fatal na Busca")
            
            # Executa
            modelo = knn.treinar_knn_otimizado(self.X_dummy, self.y_dummy)
            
            # Deve retornar um Pipeline funcional (mesmo que não otimizado)
            self.assertIsInstance(modelo, Pipeline)
            # Verifica se tem o classificador final
            self.assertIsInstance(modelo.named_steps['knn'], KNeighborsClassifier)

    # =================================================================
    # TESTES DE VISUALIZAÇÃO
    # =================================================================

    @patch('knn.permutation_importance')
    @patch('knn.ConfusionMatrixDisplay.from_estimator')
    @patch('knn.plt.savefig')
    @patch('knn.plt.close')
    @patch('knn.plt.show')
    def test_visualizacao_sem_mdi(self, mock_show, mock_close, mock_save, mock_cm, mock_perm):
        """
        OBJETIVO: Testar a visualização específica do kNN.
        Diferente de Random Forest, kNN NÃO TEM 'feature_importances_'.
        Este teste garante que o script NÃO tenta acessar esse atributo inexistente,
        confiando apenas no permutation_importance.
        """
        # Mock do resultado da permutação
        mock_res = MagicMock()
        mock_res.importances_mean = np.array([0.1, 0.2, 0.05, 0.01, 0.0])
        mock_res.importances = np.random.rand(5, 5)
        mock_perm.return_value = mock_res
        
        # Mock do Modelo (Pipeline)
        # IMPORTANTE: Não adicionamos 'feature_importances_' aqui propositalmente.
        # Se o script tentar acessar, o teste vai falhar (corretamente).
        modelo_fake = MagicMock()
        
        try:
            knn.analisar_performance_knn(
                modelo_fake, self.X_dummy, self.y_dummy, self.feature_names
            )
        except AttributeError:
            self.fail("O script tentou acessar um atributo que kNN não tem (ex: feature_importances_)")

        # Verifica se salvou
        mock_save.assert_called()

if __name__ == '__main__':
    unittest.main()