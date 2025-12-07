import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import os
import warnings

# Importa as classes do sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

# Importa o seu script
import gbte

class TestGBTE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore")

    def setUp(self):
        # 50 amostras, 5 features numéricas
        self.X_dummy = np.random.rand(50, 5)
        self.y_dummy = np.random.choice([0, 1], size=50)
        self.feature_names = [f"Var_{i}" for i in range(5)]

    # =================================================================
    # TESTES BÁSICOS (Já existiam)
    # =================================================================

    def test_geracao_dados_sinteticos_fallback(self):
        X, y, feats = gbte.carregar_e_processar_dados(caminho_arquivo=None, n_samples=100)
        self.assertEqual(X.shape, (100, 20))
        self.assertEqual(len(y), 100)

    @patch('gbte.pd.read_csv')
    def test_processamento_one_hot_encoding(self, mock_read_csv):
        df_fake = pd.DataFrame({
            'Valor': [1.0, 2.0, 3.0, 4.0],
            'Cor': ['Azul', 'Vermelho', 'Azul', 'Verde'], 
            'target': [0, 1, 0, 1]
        })
        mock_read_csv.return_value = df_fake

        X, y, feats = gbte.carregar_e_processar_dados('dummy.csv')
        self.assertEqual(X.shape[1], 3)
        self.assertTrue(any("Cor_" in f for f in feats))

    def test_remocao_colunas_constantes(self):
        df_fake = pd.DataFrame({
            'Util': [1, 5, 2, 3],
            'Inutil': [0, 0, 0, 0],
            'target': [0, 1, 0, 1]
        })
        with patch('gbte.pd.read_csv', return_value=df_fake):
            X, y, feats = gbte.carregar_e_processar_dados('arquivo.csv')
            self.assertNotIn('Inutil', feats)
            self.assertIn('Util', feats)

    # =================================================================
    # NOVOS TESTES AVANÇADOS (Lógica e Robustez)
    # =================================================================

    def test_identificacao_inteligente_target(self):
        """
        OBJETIVO: Verificar a lógica que detecta a coluna alvo.
        Cenário 1: Existe coluna 'target' (deve ser usada, mesmo não sendo a última).
        Cenário 2: Não existe 'target' (deve usar a última).
        """
        # Cenário 1: 'target' explícito no meio
        df_1 = pd.DataFrame({
            'A': [1, 2], 'target': [0, 1], 'B': [3, 4]
        })
        
        # Cenário 2: Sem 'target', alvo é a última ('Y')
        df_2 = pd.DataFrame({
            'A': [1, 2], 'B': [3, 4], 'Y': [0, 1]
        })

        with patch('gbte.pd.read_csv') as mock_read:
            # Testa Cenário 1
            mock_read.return_value = df_1
            X1, y1, feats1 = gbte.carregar_e_processar_dados('f1.csv')
            # Se funcionou, 'B' está em features e 'A' está em features. 'target' saiu.
            self.assertIn('B', feats1) 
            
            # Testa Cenário 2
            mock_read.return_value = df_2
            X2, y2, feats2 = gbte.carregar_e_processar_dados('f2.csv')
            # Se funcionou, a última coluna 'Y' virou target e sumiu das features
            self.assertNotIn('Y', feats2)
            self.assertIn('B', feats2)

    def test_preservacao_nans_numericos(self):
        """
        OBJETIVO: O HistGradientBoosting aceita NaNs. Testamos se o pré-processamento
        mantém os NaNs em colunas numéricas (não deve preencher com 0 ou dropar linhas).
        """
        df_nan = pd.DataFrame({
            'feat_numerica': [1.0, np.nan, 3.0, 4.0], # Tem NaN
            'target': [0, 1, 0, 1]
        })
        
        with patch('gbte.pd.read_csv', return_value=df_nan):
            X, y, feats = gbte.carregar_e_processar_dados('nan.csv')
            
            # Verifica se o NaN ainda está lá no numpy array
            # A coluna 0 é a feat_numerica
            nans_presentes = np.isnan(X[:, 0]).sum()
            self.assertTrue(nans_presentes > 0, "O pré-processamento removeu indevidamente os NaNs numéricos!")

    def test_fallback_treinamento_erro_gridsearch(self):
        """
        OBJETIVO: Teste de Robustez.
        Simulamos uma falha crítica no RandomizedSearchCV (ex: erro de memória ou configuração).
        O código deve capturar o erro (except) e treinar um modelo simples (fit direto).
        """
        # Mock do objeto RandomizedSearchCV para lançar erro quando chamar .fit()
        with patch('gbte.RandomizedSearchCV') as MockSearch:
            instance = MockSearch.return_value
            instance.fit.side_effect = Exception("Erro Simulado no GridSearch")
            
            # Executa a função
            # Não deve lançar exceção para fora, pois o código trata isso
            modelo = gbte.treinar_hgbm_otimizado(self.X_dummy, self.y_dummy)
            
            # Verifica se retornou um classificador válido apesar do erro na busca
            self.assertIsInstance(modelo, HistGradientBoostingClassifier)
            # Verifica se o modelo tem atributos de treino (significa que rodou o fallback .fit())
            self.assertTrue(hasattr(modelo, 'classes_'))

    @patch('gbte.pd.read_csv')
    def test_logica_tentativa_leitura_tsv_csv(self, mock_read_csv):
        """
        OBJETIVO: Testar o bloco try/except na leitura de arquivos.
        Simulamos falha na primeira tentativa (TSV) e sucesso na segunda (CSV).
        """
        # Configura o mock para lançar erro na 1ª chamada e retornar DF na 2ª
        df_ok = pd.DataFrame({'a':[1], 'target':[0]})
        mock_read_csv.side_effect = [Exception("Erro TSV"), df_ok]
        
        X, y, f = gbte.carregar_e_processar_dados('arquivo.txt')
        
        # O mock deve ter sido chamado 2 vezes (1 falha + 1 sucesso)
        self.assertEqual(mock_read_csv.call_count, 2, "Deveria ter tentado ler duas vezes (fallback)")

    @patch('gbte.permutation_importance')
    @patch('gbte.plt.savefig')
    @patch('gbte.plt.show')
    @patch('gbte.plt.close') 
    def test_visualizacao_sem_historico_validacao(self, mock_close, mock_show, mock_save, mock_perm):
        """
        OBJETIVO: Testar se a plotagem funciona mesmo quando o modelo não gerou 
        dados de curva de aprendizado (validation_score_ é None).
        """
        # Mock do resultado da importância (para não quebrar essa parte)
        mock_res = MagicMock()
        mock_res.importances_mean = np.zeros(5)
        mock_res.importances = np.zeros((5,5))
        mock_perm.return_value = mock_res
        
        # Modelo sem score de validação
        modelo_incompleto = MagicMock()
        modelo_incompleto.validation_score_ = None # <--- Cenário de teste
        modelo_incompleto.train_score_ = [0.9]

        try:
            gbte.visualizar_performance_avancada(
                modelo_incompleto, self.X_dummy, self.y_dummy, self.X_dummy, self.y_dummy, self.feature_names
            )
        except Exception as e:
            self.fail(f"A visualização quebrou ao lidar com modelo sem histórico: {e}")
            
        mock_save.assert_called()

if __name__ == '__main__':
    unittest.main()