import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import os
import warnings

# Importa classes para validação de tipo
from sklearn.ensemble import RandomForestClassifier

# Importa o script
import random_forest

class TestRandomForest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Limpa warnings de bibliotecas para focar no resultado dos testes
        warnings.filterwarnings("ignore")

    def setUp(self):
        # Dados Dummy leves (50 linhas, 5 colunas)
        self.X_dummy = np.random.rand(50, 5).astype(np.float32)
        self.y_dummy = np.random.choice([0, 1], size=50)
        self.feature_names = [f"F_{i}" for i in range(5)]

    # =================================================================
    # TESTES DE ENGENHARIA DE DADOS
    # =================================================================

    def test_otimizacao_memoria_float32(self):
        """
        OBJETIVO: Validar a 'MELHORIA DE PERFORMANCE 1' documentada no script.
        O código promete converter os dados para float32 para economizar RAM.
        Este teste garante que isso está ocorrendo.
        """
        # Testa com dados sintéticos
        X, y, feats = random_forest.carregar_e_processar_dados(caminho_arquivo=None, n_samples=50)
        
        # Verifica se o tipo de dado é float32 (e não o float64 padrão do python)
        self.assertEqual(X.dtype, np.float32, 
                         "Os dados não foram convertidos para float32 para economia de memória.")

    @patch('random_forest.pd.read_csv')
    def test_one_hot_encoding_drop_first(self, mock_read):
        """
        OBJETIVO: Validar a lógica de 'drop_first=True' no One-Hot Encoding.
        Se temos uma coluna 'Cor' com ['Azul', 'Verde', 'Vermelho'], o modelo deve
        criar apenas 2 colunas novas (n-1) para evitar redundância matemática.
        """
        df_fake = pd.DataFrame({
            'Cor': ['Azul', 'Verde', 'Vermelho', 'Azul'], # 3 categorias únicas
            'target': [0, 1, 0, 1]
        })
        mock_read.return_value = df_fake

        X, y, feats = random_forest.carregar_e_processar_dados('dummy.csv')

        # Se temos 3 categorias, esperamos 2 colunas de features (ex: Cor_Verde, Cor_Vermelho)
        # Se drop_first fosse False, teríamos 3.
        self.assertEqual(X.shape[1], 2, 
                         f"Esperado 2 colunas (drop_first=True), mas obteve {X.shape[1]}.")

    # =================================================================
    # TESTES DO MODELO (TREINAMENTO)
    # =================================================================

    def test_treinamento_com_oob_score(self):
        """
        OBJETIVO: Validar se o modelo treinado calcula o 'Out-of-Bag Score'.
        Isso confirma que o parâmetro bootstrap=True e oob_score=True foram respeitados.
        """
        # Treina com dados sintéticos
        modelo = random_forest.treinar_rf_otimizada(self.X_dummy, self.y_dummy)
        
        # Verifica se é uma RF
        self.assertIsInstance(modelo, RandomForestClassifier)
        
        # Verifica se o atributo oob_score_ existe (só existe se o treino for configurado corretamente)
        self.assertTrue(hasattr(modelo, 'oob_score_'), 
                        "O modelo final não possui OOB Score. Verifique se bootstrap=True.")
        
        # Verifica se o score é um número válido (entre 0 e 1)
        self.assertTrue(0 <= modelo.oob_score_ <= 1, "O OOB Score está fora do intervalo [0,1]")

    def test_fallback_erro_otimizacao(self):
        """
        OBJETIVO: Robustez.
        Se o RandomizedSearchCV falhar (ex: dados insuficientes para CV),
        o código deve capturar o erro e treinar uma RF padrão sem travar.
        """
        with patch('random_forest.RandomizedSearchCV') as MockSearch:
            # Configura o mock para lançar erro ao chamar fit
            instancia_mock = MockSearch.return_value
            instancia_mock.fit.side_effect = Exception("Erro Simulado de CV")
            
            # Executa a função
            modelo = random_forest.treinar_rf_otimizada(self.X_dummy, self.y_dummy)
            
            # Não deve ter quebrado. Deve retornar um modelo treinado (fallback).
            # Sabemos que é o fallback se ele tiver o método predict
            self.assertTrue(hasattr(modelo, 'predict'))

    # =================================================================
    # TESTES DE VISUALIZAÇÃO (ESTRUTURA COMPLEXA)
    # =================================================================

    @patch('random_forest.permutation_importance')
    @patch('random_forest.ConfusionMatrixDisplay.from_estimator')
    @patch('random_forest.plt.savefig')
    @patch('random_forest.plt.close') # Mock close para não fechar nada real
    @patch('random_forest.plt.show')  # Mock show
    def test_analise_performance_sem_erros(self, mock_show, mock_close, mock_save, mock_cm, mock_perm):
        """
        OBJETIVO: Testar a função 'analisar_performance_rf'.
        Desafio: Esta função acessa 'model.estimators_' para calcular desvio padrão.
        Precisamos simular essa estrutura interna da Random Forest.
        """
        # 1. Mock do Resultado da Permutation Importance (simples)
        mock_res = MagicMock()
        mock_res.importances_mean = np.array([0.1, 0.2, 0.05, 0.01, 0.0])
        mock_res.importances = np.random.rand(5, 5)
        mock_perm.return_value = mock_res

        # 2. Mock do Modelo Complexo (Random Forest)
        modelo_fake = MagicMock()
        modelo_fake.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.2, 0.2]) # Importância média
        
        # Simula 3 árvores internas (estimators_)
        # Cada árvore precisa ter seu próprio array de feature_importances_
        arvore_1 = MagicMock(); arvore_1.feature_importances_ = np.random.rand(5)
        arvore_2 = MagicMock(); arvore_2.feature_importances_ = np.random.rand(5)
        arvore_3 = MagicMock(); arvore_3.feature_importances_ = np.random.rand(5)
        
        modelo_fake.estimators_ = [arvore_1, arvore_2, arvore_3]

        # 3. Executa a visualização
        try:
            random_forest.analisar_performance_rf(
                modelo_fake, self.X_dummy, self.y_dummy, self.feature_names
            )
        except AttributeError as e:
            self.fail(f"A visualização falhou ao acessar atributos internos da RF: {e}")

        # 4. Verificações
        mock_save.assert_called() # Tentou salvar
        mock_cm.assert_called()   # Tentou plotar matriz de confusão

if __name__ == '__main__':
    unittest.main()