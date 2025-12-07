import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

# ==============================================================================
# ESCOLHA DE ARQUITETURA: HistGradientBoostingClassifier
# ==============================================================================
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

def carregar_e_processar_dados(caminho_arquivo=None, n_samples=1000):
    """
    Carrega dados e realiza pré-processamento focado em robustez.
    """
    if caminho_arquivo:
        print(f"   [DADOS] Lendo arquivo: {caminho_arquivo}")
        try:
            try:
                df = pd.read_csv(caminho_arquivo, sep='\t')
            except:
                df = pd.read_csv(caminho_arquivo, sep=',')

            target_col = 'target' if 'target' in df.columns else df.columns[-1]
            df = df.dropna(subset=[target_col])
            
            print("   [DADOS] Aplicando One-Hot Encoding (get_dummies)...")
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
            raise Exception(f"Erro ao processar arquivo: {e}")
    else:
        print("   [DADOS] Gerando dados sintéticos (Make Classification)...")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=n_samples, n_features=20, 
                                   n_informative=10, n_redundant=5, random_state=42)
        feature_names = [f"Var_{i+1}" for i in range(X.shape[1])]
        return X, y, feature_names

def treinar_hgbm_otimizado(X_train, y_train):
    """
    Configura e treina o modelo HistGradientBoostingClassifier.
    """
    print("   [MODELO] Iniciando otimização (HistGradientBoosting)...")
    
    hgbm = HistGradientBoostingClassifier(
        random_state=42, 
        early_stopping=True
    )
    
    param_dist = {
        'max_iter': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 8, None],
        'max_leaf_nodes': [31, 63, 127],
        'l2_regularization': [0, 1.0, 10.0],
        'min_samples_leaf': [20, 40, 60]
    }
    
    random_search = RandomizedSearchCV(
        estimator=hgbm,
        param_distributions=param_dist,
        n_iter=15,
        scoring='roc_auc',
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    try:
        random_search.fit(X_train, y_train)
        print(f"   [MODELO] Melhores parâmetros: {random_search.best_params_}")
        return random_search.best_estimator_
    except Exception as e:
        print(f"   [ERRO] Falha na otimização: {e}. Treinando modelo padrão.")
        hgbm.fit(X_train, y_train)
        return hgbm

def visualizar_performance_avancada(model, X_test, y_test, X_train, y_train, feature_names, save_prefix='gbm_resultado'):
    """
    Gera gráficos de diagnóstico do modelo.
    """
    print("   [VISUALIZAÇÃO] Gerando gráficos...")
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    if hasattr(model, 'validation_score_') and model.validation_score_ is not None:
        plt.plot(model.train_score_, label='Score Treino')
        plt.plot(model.validation_score_, label='Score Validação', linestyle='--')
        plt.xlabel('Iterações (Árvores)')
        plt.ylabel('Score')
        plt.title('Histórico de Treinamento (Learning Curve)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, "Sem histórico disponível", ha='center')

    print("   [VISUALIZAÇÃO] Calculando importância por permutação (PARALELO)...")
    plt.subplot(1, 2, 2)
    
    result = permutation_importance(
        model, X_test, y_test, 
        n_repeats=5,
        random_state=42, 
        n_jobs=-1
    )
    
    sorted_idx = result.importances_mean.argsort()
    if len(sorted_idx) > 15: sorted_idx = sorted_idx[-15:]
    
    plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=np.array(feature_names)[sorted_idx])
    plt.title("Permutation Importance (Impacto no Teste)")
    plt.xlabel("Queda na Acurácia/Score")
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_performance.png")
    plt.close()
    print(f"   [VISUALIZAÇÃO] Gráfico salvo: {save_prefix}_performance.png")

# ==========================================
# FUNÇÃO WRAPPER (REFATORADA)
# ==========================================
def executar_pipeline(caminho_arquivo=None):
    print("="*60)
    print(" INICIANDO HIST-GRADIENT BOOSTING (MODULARIZADO)")
    print("="*60)

    try:
        # 1. Carregamento
        X, y, feats = carregar_e_processar_dados(caminho_arquivo)
        
        # Divisão 70/30 para garantir teste robusto
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # 2. Treino Otimizado
        model = treinar_hgbm_otimizado(X_train, y_train)

        # 3. Métricas Finais
        y_pred = model.predict(X_test)
        
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.5

        acc = accuracy_score(y_test, y_pred)
        
        print("-" * 30)
        print("RESULTADOS FINAIS:")
        print(f"  -> Acurácia:      {acc:.2%}")
        print(f"  -> AUC ROC:       {auc:.4f}")
        print(f"  -> Iterações:     {model.n_iter_} (Parou via Early Stopping?)")
        print("-" * 30)
        
        # 4. Visualização
        visualizar_performance_avancada(model, X_test, y_test, X_train, y_train, feats)
        
        print("\nProcesso concluído com sucesso.")
        
    except Exception as e:
        print(f"\n[ERRO CRÍTICO] Ocorreu um erro na execução: {e}")

# ==========================================
# Execução Principal (Terminal)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arquivo', type=str, default=None, help="Caminho do arquivo CSV/TSV")
    args = parser.parse_args()
    
    # Chamada via Wrapper
    executar_pipeline(caminho_arquivo=args.arquivo)