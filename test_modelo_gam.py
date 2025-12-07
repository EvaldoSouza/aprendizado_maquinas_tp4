import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import os
import warnings

# Importa o seu script. 
# Nota: O arquivo do script deve se chamar 'modelo_gam.py' e estar na mesma pasta.
import modelo_gam

class TestModeloGAM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Executado uma vez antes de todos os testes.
        Usamos para suprimir warnings de bibliotecas terceiras (como pygam) que poluem o log.
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    def setUp(self):
        """
        Executado antes de cada teste individualmente. 
        Prepara dados comuns para evitar repetição de código.
        """
        # Cria um dataset sintético pequeno (50 linhas, 5 colunas) para testes rápidos
        self.X_dummy = np.random.rand(50, 5) 
        
        # Força a primeira coluna (índice 0) a ser binária (apenas 0 e 1)
        # Isso é essencial para testar se o modelo sabe diferenciar colunas Lineares vs Splines
        self.X_dummy[:, 0] = np.random.choice([0, 1], size=50)
        
        self.y_dummy = np.random.choice([0, 1], size=50)
        self.feature_names = ['feat_binaria', 'feat_cont1', 'feat_cont2', 'feat_cont3', 'feat_cont4']

    def test_geracao_dados_sinteticos(self):
        """
        OBJETIVO: Verificar se o sistema gera dados automaticamente quando nenhum arquivo é fornecido.
        
        O que está sendo testado:
        1. Se as dimensões (linhas/colunas) batem com o solicitado (n_samples=100).
        2. Se os tipos de retorno (listas, arrays) estão corretos.
        """
        X, y, feats = modelo_gam.carregar_e_processar_dados(caminho_arquivo=None, n_samples=100)
        
        # Validações
        self.assertEqual(X.shape[0], 100, "Deveria ter gerado 100 amostras")
        self.assertEqual(len(y), 100, "O vetor alvo y deve ter o mesmo tamanho de X")
        self.assertIsInstance(feats, list, "Os nomes das features devem ser uma lista")
        self.assertEqual(len(feats), X.shape[1], "Deve haver um nome para cada coluna de X")

    @patch('modelo_gam.pd.read_csv')
    def test_carregar_arquivo_csv(self, mock_read_csv):
        """
        OBJETIVO: Testar a leitura de arquivos CSV sem precisar de um arquivo real no disco.
        
        Estratégia:
        - Usamos @patch para 'enganar' o script. Quando ele chamar pd.read_csv, 
          entregamos um DataFrame falso criado em memória.
        """
        # Cria um DataFrame falso que o pandas retornaria
        data = {
            'col1': np.random.rand(20),
            'col2': np.random.rand(20),
            'target': np.random.choice([0, 1], 20)
        }
        df_mock = pd.DataFrame(data)
        mock_read_csv.return_value = df_mock

        # Chama a função passando um caminho qualquer (o mock interceptará)
        X, y, feats = modelo_gam.carregar_e_processar_dados(caminho_arquivo='caminho_falso.csv')

        # Verificações
        self.assertEqual(X.shape, (20, 2), "Deveria ter 20 linhas e 2 colunas (removeu o target)")
        self.assertListEqual(feats, ['col1', 'col2'], "Os nomes das colunas devem bater com o CSV simulado")
        
        # Garante que o script realmente tentou ler o arquivo
        mock_read_csv.assert_called_once()

    def test_remocao_colunas_constantes(self):
        """
        OBJETIVO: Garantir que colunas que não variam (ex: tudo valor 1) sejam excluídas,
        pois elas quebram modelos estatísticos.
        """
        # Cria dataframe onde a col 'constante' é sempre 1
        df = pd.DataFrame({
            'var': [1, 2, 3, 4],       # Varia
            'constante': [1, 1, 1, 1], # Não varia
            'target': [0, 1, 0, 1]
        })

        # Mockamos o read_csv para retornar esse df específico
        with patch('modelo_gam.pd.read_csv', return_value=df):
            X, y, feats = modelo_gam.carregar_e_processar_dados('arquivo_fake.csv')
            
            # Validações
            self.assertNotIn('constante', feats, "A coluna constante deveria ter sido removida")
            self.assertEqual(X.shape[1], 1, "Só deve sobrar a coluna 'var'")

    def test_construcao_termos_dinamicamente(self):
        """
        OBJETIVO CRUCIAL: Testar a lógica inteligente que decide o tipo de função matemática.
        - Variáveis Binárias (0 ou 1) -> Devem usar termo Linear (l)
        - Variáveis Contínuas (float) -> Devem usar termo Spline (s)
        """
        # Usamos os dados criados no setUp (coluna 0 é binária, coluna 1 é contínua)
        terms, types = modelo_gam.construir_termos_dinamicamente(self.X_dummy)
        
        # Validações
        self.assertEqual(types[0], 'linear', "A primeira feature (binária) foi incorretamente classificada como spline")
        self.assertEqual(types[1], 'spline', "A segunda feature (contínua) foi incorretamente classificada como linear")
        
        # Verifica se retornou um objeto válido do pygam
        self.assertIsNotNone(terms)

    def test_treinamento_gam_smoke_test(self):
        """
        OBJETIVO: "Smoke Test" (Teste de Fumaça).
        Serve para garantir que o pipeline inteiro roda do início ao fim sem erros de sintaxe
        ou incompatibilidade de matrizes, mesmo que o modelo treinado seja "burro" (poucos dados).
        """
        # Executa o treino real com dataset pequeno (self.X_dummy)
        gam, scaler, t_types = modelo_gam.treinar_gam_automatico(self.X_dummy, self.y_dummy)
        
        # Validações básicas de existência
        self.assertTrue(hasattr(gam, 'predict'), "O objeto retornado não parece ser um modelo (sem método predict)")
        self.assertEqual(len(t_types), 5, "Deveria ter identificado 5 tipos de termos para as 5 features")
        self.assertIsNotNone(scaler, "O scaler não foi retornado")

    @patch('modelo_gam.plt.savefig')
    @patch('modelo_gam.plt.show')
    def test_plotagem_sem_erro(self, mock_show, mock_savefig):
        """
        OBJETIVO: Testar a função de gráficos.
        Como não temos interface gráfica (GUI) durante testes automatizados, usamos Mocks
        para impedir que plt.show() trave o teste ou plt.savefig() crie lixo no disco.
        """
        # Treina um modelo rápido para ter o que plotar
        gam, scaler, t_types = modelo_gam.treinar_gam_automatico(self.X_dummy, self.y_dummy)
        
        try:
            # Tenta rodar a função de plotagem
            modelo_gam.plotar_interpretacao(gam, scaler, self.feature_names, t_types, save_path='teste_plot.png')
        except Exception as e:
            self.fail(f"A função de plotagem falhou inesperadamente: {e}")
            
        # Verifica se o código tentou salvar o arquivo (o Mock interceptou a ação real)
        mock_savefig.assert_called()

if __name__ == '__main__':
    unittest.main()