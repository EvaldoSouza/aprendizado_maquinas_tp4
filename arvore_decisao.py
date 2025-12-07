import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

def carregar_e_processar_dados(caminho_arquivo=None, n_samples=1000):
    """
    Carrega dados de um arquivo CSV/TSV ou gera dados sintéticos.
    """
    if caminho_arquivo:
        print(f"   [DADOS] Lendo arquivo: {caminho_arquivo}")
        try:
            try:
                df = pd.read_csv(caminho_arquivo, sep='\t')
            except:
                df = pd.read_csv(caminho_arquivo, sep=',')

            target_col = 'target' if 'target' in df.columns else df.columns[-1]
            print(f"   [DADOS] Coluna alvo identificada: '{target_col}'")
            
            df = df.dropna(subset=[target_col])
            df = pd.get_dummies(df, drop_first=True, dtype=int)
            
            X = df.drop(columns=[target_col])
            y = df[target_col].values
            
            cols_constantes = [col for col in X.columns if X[col].nunique() <= 1]
            if cols_constantes:
                print(f"   [AVISO] Removendo colunas constantes: {cols_constantes}")
                X = X.drop(columns=cols_constantes)
            
            feature_names = X.columns.tolist()
            X_values = X.values.astype(float)
            
            return X_values, y, feature_names
            
        except Exception as e:
            raise Exception(f"Erro crítico ao processar o arquivo: {e}")
    else:
        print("   [DADOS] Nenhum arquivo fornecido. Gerando dados sintéticos...")
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=n_samples, n_features=10, 
                                   n_informative=5, n_redundant=2, random_state=42)
        
        feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
        return X, y, feature_names

def treinar_arvore_com_poda(X_train, y_train):
    """
    Treina uma Árvore de Decisão utilizando 'Cost-Complexity Pruning' (ccp_alpha).
    """
    print("   [MODELO] Iniciando treinamento e otimização da árvore...")
    
    clf_base = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    
    path = clf_base.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    
    ccp_alphas = ccp_alphas[::5] if len(ccp_alphas) > 50 else ccp_alphas
    
    grid_search = GridSearchCV(
        estimator=clf_base,
        param_grid={'ccp_alpha': ccp_alphas},
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    try:
        grid_search.fit(X_train, y_train)
    except Exception as e:
        print(f"   [ERRO] Falha no GridSearch: {e}. Treinando modelo simples.")
        clf_base.fit(X_train, y_train)
        return clf_base

    print(f"   [MODELO] Melhor alpha de poda encontrado: {grid_search.best_params_['ccp_alpha']:.6f}")
    return grid_search.best_estimator_

def visualizar_resultados(model, X_test, y_test, feature_names, save_prefix='arvore_resultado'):
    """
    Gera visualizações: Árvore e Importância.
    """
    print("   [VISUALIZAÇÃO] Gerando gráficos...")
    
    # --- Plot 1: Estrutura da Árvore ---
    plt.figure(figsize=(20, 10))
    plot_tree(model, 
              feature_names=feature_names, 
              class_names=[str(c) for c in np.unique(y_test)], 
              filled=True, 
              rounded=True, 
              fontsize=10,
              max_depth=4)
    plt.title("Estrutura da Árvore de Decisão (Poda Estatística - Top 4 Níveis)")
    
    caminho_arvore = f"{save_prefix}_estrutura.png"
    plt.savefig(caminho_arvore)
    plt.close()
    
    # --- Plot 2: Importância das Features ---
    importancias = pd.Series(model.feature_importances_, index=feature_names)
    importancias = importancias.sort_values(ascending=False).head(15)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importancias.values, y=importancias.index, palette='viridis')
    plt.title("Importância das Variáveis (Feature Importance)")
    plt.xlabel("Importância Relativa")
    
    caminho_feat = f"{save_prefix}_features.png"
    plt.savefig(caminho_feat)
    plt.close()

    print(f"   [VISUALIZAÇÃO] Gráficos salvos: {caminho_arvore}, {caminho_feat}")

# ==========================================
# FUNÇÃO WRAPPER (REFATORADA)
# ==========================================
def executar_pipeline(caminho_arquivo=None):
    print("="*40)
    print(" INICIANDO PROCESSO DE MODELAGEM (ÁRVORE - MODULARIZADO)")
    print("="*40)

    try:
        # 1. Carregamento e Processamento
        X, y, feats = carregar_e_processar_dados(caminho_arquivo)
        
        # Verificação de segurança
        if len(y) < 50:
            raise ValueError("O dataset é muito pequeno (menos de 50 amostras) para inferência confiável.")

        # Divisão Treino/Teste (70% treino, 30% teste)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print(f"   [DADOS] Divisão concluída. Treino: {X_train.shape[0]} amostras, Teste: {X_test.shape[0]} amostras.")

        # 2. Treinamento com Poda Automática
        modelo = treinar_arvore_com_poda(X_train, y_train)

        # 3. Avaliação de Performance
        y_pred = modelo.predict(X_test)
        
        try:
            y_proba = modelo.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.5

        acc = accuracy_score(y_test, y_pred)
        depth = modelo.get_depth()
        n_leaves = modelo.get_n_leaves()

        print("-" * 30)
        print("RELATÓRIO DE PERFORMANCE:")
        print(f"  -> Acurácia:      {acc:.2%}")
        print(f"  -> AUC ROC:       {auc:.4f}")
        print(f"  -> Profundidade:  {depth}")
        print(f"  -> Nº de Folhas:  {n_leaves} (Complexidade do Modelo)")
        print("-" * 30)
        
        # 4. Visualização
        visualizar_resultados(modelo, X_test, y_test, feats)
        
        print("\nProcesso finalizado com sucesso.")

    except Exception as e:
        print(f"\n[ERRO FATAL] O programa foi interrompido: {e}")

# ==========================================
# Execução Principal (Terminal)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento de Árvore de Decisão Robusta")
    parser.add_argument('--arquivo', type=str, default=None, help="Caminho do arquivo CSV ou TSV")
    args = parser.parse_args()

    executar_pipeline(caminho_arquivo=args.arquivo)