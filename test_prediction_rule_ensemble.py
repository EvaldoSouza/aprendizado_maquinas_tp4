import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import os
import warnings

# Importa as classes do seu script
# Certifique-se de que o arquivo prediction_rule_ensemble.py está na mesma pasta
from prediction_rule_ensemble import (
    RuleFit, 
    CondicaoRegra, 
    Regra, 
    Winsorizer, 
    FriedScale, 
    carregar_e_processar_dados, 
    visualizar_regras
)

class TestRuleFit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore")

    def setUp(self):
        # Dados Dummy para testes rápidos
        self.X_dummy = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [100.0, 1000.0] # Outlier proposital
        ])
        self.y_dummy = np.array([0, 0, 1, 1, 1])
        self.feature_names = ["Feat_A", "Feat_B"]

    # =================================================================
    # 1. TESTES UNITÁRIOS DAS CLASSES AUXILIARES (Lógica Customizada)
    # =================================================================

    def test_condicao_regra_logica(self):
        """
        OBJETIVO: Testar se a classe CondicaoRegra aplica o filtro corretamente.
        Cenário: Feat_A (índice 0) <= 2.5
        """
        # Cria a condição: Feature 0 <= 2.5
        condicao = CondicaoRegra(feature_index=0, threshold=2.5, operator="<=", support=0.5)
        
        # Aplica aos dados dummy
        resultado = condicao.transform(self.X_dummy)
        
        # Esperado: [1, 1, 0, 0, 0] (Apenas 1.0 e 2.0 são <= 2.5)
        esperado = np.array([1, 1, 0, 0, 0])
        
        np.testing.assert_array_equal(resultado, esperado, 
                                      "A lógica da condição (CondicaoRegra) falhou na filtragem.")

    def test_regra_combinada(self):
        """
        OBJETIVO: Testar se a classe Regra combina múltiplas condições com AND lógico.
        Cenário: (Feat_A > 1.5) AND (Feat_B < 35.0)
        """
        c1 = CondicaoRegra(0, 1.5, ">", 0.1) # X[0] > 1.5 -> [0, 1, 1, 1, 1]
        c2 = CondicaoRegra(1, 35.0, "<=", 0.1) # X[1] <= 35 -> [1, 1, 1, 0, 0]
        
        regra = Regra([c1, c2], prediction_value=0.5)
        resultado = regra.transform(self.X_dummy)
        
        # AND lógico: [0, 1, 1, 1, 1] & [1, 1, 1, 0, 0] = [0, 1, 1, 0, 0]
        esperado = np.array([0, 1, 1, 0, 0])
        
        np.testing.assert_array_equal(resultado, esperado,
                                      "A combinação de regras (AND lógico) falhou.")

    def test_winsorizer_clipagem_outliers(self):
        """
        OBJETIVO: Verificar se o Winsorizer remove outliers extremos.
        Temos um valor 100.0 no X_dummy. Vamos tentar clipar os 20% superiores.
        """
        # trim_quantile=0.2 vai clipar os 20% maiores e menores
        winsor = Winsorizer(trim_quantile=0.2)
        winsor.train(self.X_dummy)
        X_trimmed = winsor.trim(self.X_dummy)
        
        # O valor 100.0 (última linha, col 0) deve ter sido reduzido
        # O valor máximo permitido será o percentil 80 (aprox 4.0 nos dados normais)
        valor_maximo_original = self.X_dummy[-1, 0] # 100.0
        valor_maximo_novo = X_trimmed[-1, 0]
        
        self.assertLess(valor_maximo_novo, valor_maximo_original,
                        "O Winsorizer falhou em reduzir o outlier extremo.")

    # =================================================================
    # 2. TESTES DE INTEGRAÇÃO DO MODELO RULEFIT
    # =================================================================

    def test_treinamento_classificacao(self):
        """
        OBJETIVO: Testar o pipeline completo do RuleFit em modo Classificação.
        Verifica se ele gera regras e ajusta o modelo linear.
        """
        # Configuração leve para teste rápido
        rf = RuleFit(
            rfmode='classify', 
            max_rules=10,       # Poucas regras
            tree_size=2,        # Árvores pequenas
            model_type='rl',    # Rules + Linear
            random_state=42,
            cv=2                # CV mínima
        )
        
        try:
            rf.fit(self.X_dummy, self.y_dummy, feature_names=self.feature_names)
        except Exception as e:
            self.fail(f"O treinamento do RuleFit falhou com erro: {e}")

        # Verificações
        self.assertTrue(len(rf.rule_ensemble.rules) > 0, 
                        "O modelo não extraiu nenhuma regra das árvores.")
        self.assertIsNotNone(rf.coef_, "O modelo não gerou coeficientes (Lasso/Logistic).")
        
        # Teste de Predição
        preds = rf.predict(self.X_dummy)
        self.assertEqual(len(preds), 5, "A predição retornou tamanho incorreto.")

    def test_treinamento_regressao(self):
        """
        OBJETIVO: Testar o pipeline completo em modo Regressão.
        """
        y_reg = np.array([1.5, 2.5, 3.5, 4.5, 50.0]) # Alvo contínuo
        
        rf = RuleFit(
            rfmode='regress',
            max_rules=5,
            tree_size=2,
            random_state=42,
            cv=2
        )
        
        rf.fit(self.X_dummy, y_reg)
        
        # O LassoCV gera intercept_ em regressão
        self.assertIsNotNone(rf.intercept_, "Faltou o intercepto do modelo de regressão.")
        
        # Predição
        preds = rf.predict(self.X_dummy)
        self.assertIsInstance(preds[0], (float, np.float32, np.float64))

    # =================================================================
    # 3. TESTES DE VISUALIZAÇÃO E UTILITÁRIOS
    # =================================================================

    @patch('prediction_rule_ensemble.plt.savefig')
    @patch('prediction_rule_ensemble.plt.close')
    @patch('prediction_rule_ensemble.pd.DataFrame.to_csv') # Mock para não criar CSV
    def test_visualizar_regras(self, mock_to_csv, mock_close, mock_savefig):
        """
        OBJETIVO: Garantir que a extração e plotagem das regras funciona.
        """
        # Treina modelo rápido
        rf = RuleFit(rfmode='classify', max_rules=5, tree_size=2, random_state=42, cv=2)
        rf.fit(self.X_dummy, self.y_dummy, feature_names=self.feature_names)
        
        # Força alguns coeficientes a serem não-zero para garantir que algo seja plotado
        # (Lasso às vezes zera tudo em dados dummy pequenos)
        rf.coef_ = np.ones(len(rf.rule_ensemble.rules) + 2) 

        try:
            visualizar_regras(rf, self.feature_names)
        except Exception as e:
            self.fail(f"A visualização de regras falhou: {e}")
            
        mock_savefig.assert_called()
        mock_to_csv.assert_called()

    def test_carregar_dados_sinteticos(self):
        """
        OBJETIVO: Testar a geração de dados sintéticos fallback.
        """
        X, y, feats = carregar_e_processar_dados(caminho_arquivo=None, n_samples=20)
        self.assertEqual(X.shape, (20, 10))
        self.assertEqual(len(y), 20)

if __name__ == '__main__':
    unittest.main()