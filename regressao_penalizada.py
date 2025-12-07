import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse  # Biblioteca para processar argumentos via linha de comando (Terminal)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

# ==========================================
# 1. Definição das Funções (Lógica Pura)
# ==========================================

def gerar_dataset_sintetico(n_samples=1000, n_features=20, n_informative=5, random_state=42):
    """
    Gera um conjunto de dados artificial para testes controlados.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=int(n_features * 0.2), # Cria 20% de colunas repetidas/redundantes
        n_classes=2,                       # Classificação binária (0 ou 1)
        random_state=random_state
    )
    return X, y

def carregar_dados_tsv(caminho_arquivo, nome_coluna_target='target'):
    """
    Lê um ficheiro TSV (separado por TABs) e prepara os dados para Machine Learning.
    """
    # 1. Tentativa de Leitura
    try:
        # sep='\t' indica que as colunas são divididas por tabulação, não vírgulas
        df = pd.read_csv(caminho_arquivo, sep='\t')
    except FileNotFoundError:
        # Lança exceção para ser capturada pelo wrapper
        raise FileNotFoundError(f"Erro Crítico: O ficheiro '{caminho_arquivo}' não foi encontrado.")
    except Exception as e:
        # Tenta ler como CSV normal se falhar como TSV
        try:
            df = pd.read_csv(caminho_arquivo, sep=',')
        except:
             raise e
    
    # 2. Validação da Coluna Alvo
    target_col = nome_coluna_target if nome_coluna_target in df.columns else df.columns[-1]
    
    # Separa o alvo (y) ANTES de mexer nas features para protegê-lo de alterações
    y = df[target_col].values
    
    # Cria uma cópia apenas com as features (removendo a coluna alvo)
    X_raw = df.drop(columns=[target_col])
    
    # 3. Tratamento de Dados Categóricos (Texto -> Números)
    cols_categoricas = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if cols_categoricas:
        print(f"   [Auto-Encoding] Colunas de texto detectadas: {cols_categoricas}")
        X_encoded = pd.get_dummies(X_raw, columns=cols_categoricas, drop_first=True)
    else:
        X_encoded = X_raw

    # 4. Conversão Final
    X = X_encoded.values.astype(float)
    
    return X, y

def preparar_dados(X, y, test_size=0.3, random_state=42):
    """
    Pipeline de preparação: Split + StandardScaler.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return (X_train_scaled, X_test_scaled, y_train, y_test), scaler

def treinar_modelo(X_train, y_train, penalty, solver, l1_ratio=None, cv=5, n_jobs=-1):
    """
    Configura e treina o modelo LogisticRegressionCV.
    """
    modelo = LogisticRegressionCV(
        cv=cv,             
        penalty=penalty,   
        solver=solver,     
        l1_ratios=[l1_ratio] if l1_ratio else None, 
        n_jobs=n_jobs,     
        random_state=42,   
        max_iter=5000,     
        tol=1e-3           
    )
    modelo.fit(X_train, y_train)
    return modelo

def extrair_metricas(modelo, X_test, y_test):
    """
    Gera um relatório simples de performance.
    """
    acc = modelo.score(X_test, y_test)
    coefs_zerados = np.sum(modelo.coef_ == 0)
    total_features = modelo.coef_.shape[1]
    
    return {
        "acuracia": acc,
        "zeros": coefs_zerados,
        "total_features": total_features,
        "melhor_C": modelo.C_[0]
    }

def plotar_comparacao(modelos_dict, save_path='comparacao_coeficientes.png'):
    """
    Cria um gráfico visual para comparar como cada modelo tratou os pesos das variáveis.
    """
    plt.figure(figsize=(10, 6))
    
    marcadores = ['s-', 'o-', '^-'] 
    
    for (nome, modelo), marker in zip(modelos_dict.items(), marcadores):
        plt.plot(modelo.coef_.flatten(), marker, label=nome, alpha=0.7)

    plt.axhline(0, color='black', linestyle='--', linewidth=0.8) 
    plt.title("Comparação de Coeficientes: Impacto da Regularização")
    plt.xlabel("Índice da Feature (Variável)")
    plt.ylabel("Magnitude do Peso")
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(save_path)
    print(f"Gráfico guardado com sucesso em: {save_path}")

# ==========================================
# FUNÇÃO WRAPPER (REFATORADA)
# ==========================================
def executar_pipeline(caminho_arquivo=None, target_col='target'):
    print("="*60)
    print(" INICIANDO REGRESSÃO PENALIZADA (LASSO/RIDGE/ELASTICNET)")
    print("="*60)
    
    # 1. Decisão da Fonte de Dados
    if caminho_arquivo:
        print(f"Modo: Ficheiro Real ({caminho_arquivo})")
        try:
            X, y = carregar_dados_tsv(caminho_arquivo, target_col)
            print(f"Dados carregados: {X.shape[0]} amostras, {X.shape[1]} features (após processamento)")
        except Exception as e:
            print(f"[ERRO] {e}")
            return # SEGURANÇA: Retorna em vez de matar o processo com exit()
    else:
        print("Modo: Dados Sintéticos (Padrão)")
        X, y = gerar_dataset_sintetico(n_samples=5000, n_features=50, n_informative=10)
    
    # Processamento padrão (Split + Scale)
    (X_train, X_test, y_train, y_test), _ = preparar_dados(X, y)
    
    # 2. Configuração dos Experimentos
    configs = [
        ("L2 (Ridge)", "l2", "lbfgs", None),      
        ("L1 (Lasso)", "l1", "saga", None),       
        ("ElasticNet", "elasticnet", "saga", 0.5) 
    ]
    
    resultados_modelos = {}
    
    # 3. Loop de Treinamento e Avaliação
    for nome, penalty, solver, l1_ratio in configs:
        print(f"A treinar modelo: {nome}...")
        try:
            modelo = treinar_modelo(X_train, y_train, penalty, solver, l1_ratio)
            metrics = extrair_metricas(modelo, X_test, y_test)
            resultados_modelos[nome] = modelo
            print(f"  -> Acurácia: {metrics['acuracia']:.4f} | Features Zeradas: {metrics['zeros']}/{metrics['total_features']}")
        except Exception as e:
            print(f"  -> Falha ao treinar {nome}: {e}")

    # 4. Geração do Gráfico Final
    if resultados_modelos:
        plotar_comparacao(resultados_modelos)
        
    print("\nProcesso finalizado com sucesso.")

# ==========================================
# Execução Principal
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Regressão Penalizada")
    
    parser.add_argument('--arquivo', type=str, help='Caminho para o ficheiro TSV')
    parser.add_argument('--target', type=str, default='target', help='Nome da coluna alvo')
    
    args = parser.parse_args()
    
    # Chamada via Wrapper
    executar_pipeline(caminho_arquivo=args.arquivo, target_col=args.target)