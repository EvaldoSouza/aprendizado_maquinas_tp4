import unittest
import numpy as np
import os
import tempfile
import pandas as pd # Necessário para criar o arquivo TSV dummy

# Importamos a nova função carregar_dados_tsv também
from regressao_penalizada import (
    gerar_dataset_sintetico, 
    carregar_dados_tsv,
    preparar_dados, 
    treinar_modelo, 
    extrair_metricas
)

class TestPipelineRegressao(unittest.TestCase):
    """
    Classe de testes que herda de unittest.TestCase.
    """
    
    def setUp(self):
        """
        MÉTODO DE CONFIGURAÇÃO (FIXTURE)
        """
        self.n_samples = 200
        self.n_features = 20
        self.n_informative = 5
        
        self.X, self.y = gerar_dataset_sintetico(
            n_samples=self.n_samples, 
            n_features=self.n_features, 
            n_informative=self.n_informative
        )
        
        self.dados_split, self.scaler = preparar_dados(self.X, self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = self.dados_split

    def test_geracao_formato(self):
        """TESTE DE INTEGRIDADE (SHAPE CHECK)"""
        self.assertEqual(self.X.shape, (self.n_samples, self.n_features), 
                         "Erro Crítico: A matriz X tem dimensões diferentes do solicitado.")
        self.assertEqual(self.y.shape, (self.n_samples,), 
                         "Erro Crítico: O vetor alvo y tem dimensões erradas.")

    def test_carregamento_tsv(self):
        """
        TESTE DE I/O (LEITURA DE ARQUIVO):
        Verifica se conseguimos ler corretamente um arquivo TSV externo.
        Cria um arquivo temporário, escreve dados dummy, lê com a função e apaga.
        """
        # 1. Criar dados dummy num DataFrame
        df_dummy = pd.DataFrame({
            'feature1': [1.5, 2.5, 3.5],
            'feature2': [10, 20, 30],
            'classe_alvo': [0, 1, 0]
        })
        
        # 2. Criar arquivo temporário TSV
        # delete=False pois precisamos fechar o arquivo para o Windows permitir a leitura
        tmp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.tsv')
        try:
            df_dummy.to_csv(tmp_file.name, sep='\t', index=False)
            tmp_file.close() # Fecha escrita para permitir leitura
            
            # 3. Testar a função carregar_dados_tsv
            X_load, y_load = carregar_dados_tsv(tmp_file.name, 'classe_alvo')
            
            # 4. Asserções
            # O X deve ter 3 linhas e 2 colunas (feature1, feature2)
            self.assertEqual(X_load.shape, (3, 2), "Erro na dimensão do X carregado via TSV.")
            # O y deve ter 3 elementos
            self.assertEqual(y_load.shape, (3,), "Erro na dimensão do y carregado via TSV.")
            # Verifica se os valores do alvo batem
            np.testing.assert_array_equal(y_load, df_dummy['classe_alvo'].values)
            
        finally:
            # 5. Limpeza (Sempre apagar arquivos temporários)
            if os.path.exists(tmp_file.name):
                os.remove(tmp_file.name)

    def test_padronizacao(self):
        """TESTE DE PRÉ-PROCESSAMENTO"""
        media_train = np.mean(self.X_train, axis=0)
        std_train = np.std(self.X_train, axis=0)
        np.testing.assert_allclose(media_train, 0, atol=1e-1, 
                                   err_msg="A média dos dados padronizados não está centralizada em zero.")
        np.testing.assert_allclose(std_train, 1, atol=1e-1, 
                                   err_msg="O desvio padrão dos dados não foi normalizado para 1.")

    def test_modelo_l1_esparsidade(self):
        """TESTE DE LÓGICA DE NEGÓCIO (REGRESSÃO LASSO)"""
        modelo = treinar_modelo(
            self.X_train, self.y_train, 
            penalty='l1', solver='liblinear', cv=3
        )
        metrics = extrair_metricas(modelo, self.X_test, self.y_test)
        self.assertTrue(metrics['zeros'] > 0, 
                        "Falha na Lógica L1: O modelo deveria ter zerado coeficientes.")
        self.assertTrue(metrics['acuracia'] > 0.6, 
                        "Alerta de Qualidade: Acurácia do modelo está suspeitosamente baixa.")

    def test_modelo_l2_converge(self):
        """TESTE DE ESTABILIDADE (REGRESSÃO RIDGE)"""
        modelo = treinar_modelo(
            self.X_train, self.y_train, 
            penalty='l2', solver='lbfgs', cv=3
        )
        metricas = extrair_metricas(modelo, self.X_test, self.y_test)
        self.assertEqual(metricas['total_features'], self.n_features, 
                         "Erro Estrutural: O modelo final não possui o número correto de coeficientes.")

if __name__ == '__main__':
    unittest.main()