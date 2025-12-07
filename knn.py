import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import time

# ==============================================================================
# ARQUITETURA: k-NEAREST NEIGHBORS (kNN)
# ==============================================================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

def carregar_e_processar_dados(caminho_arquivo=None, n_samples=2000):
    """
    Carrega, prepara e NORMALIZA os dados.
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
            
            print("   [DADOS] Aplicando One-Hot Encoding...")
            df = pd.get_dummies(df, drop_first=True, dtype=int)
            
            X = df.drop(columns=[target_col])
            y = df[target_col].values
            
            cols_constantes = [col for col in X.columns if X[col].nunique() <= 1]
            if cols_constantes:
                X = X.drop(columns=cols_constantes)
            
            feature_names = X.columns.tolist()
            X_values = X.values.astype(np.float32)
            
            return X_values, y, feature_names
            
        except Exception as e:
            raise Exception(f"Erro ao ler arquivo: {e}")
    else:
        print("   [DADOS] Gerando dados sintéticos...")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=n_samples, n_features=20, 
                                   n_informative=10, n_redundant=2, 
                                   n_clusters_per_class=1,
                                   weights=[0.6, 0.4],
                                   random_state=42)
        feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
        return X.astype(np.float32), y, feature_names

def treinar_knn_otimizado(X_train, y_train):
    """
    Treina um kNN robusto utilizando RandomizedSearchCV e Pipeline.
    """
    print("   [MODELO] Configurando kNN com Otimização de Hiperparâmetros...")
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_jobs=-1))
    ])
    
    param_dist = {
        'knn__n_neighbors': list(range(3, 30, 2)),
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan', 'minkowski'],
        'knn__p': [1, 2]
    }
    
    print("   [MODELO] Iniciando RandomizedSearchCV (Paralelo)...")
    start_time = time.time()
    
    random_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=15,
        scoring='roc_auc',
        cv=5,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    try:
        random_search.fit(X_train, y_train)
        elapsed = time.time() - start_time
        print(f"   [MODELO] Otimização concluída em {elapsed:.2f} segundos.")
        print(f"   [MODELO] Melhores parâmetros: {random_search.best_params_}")
        
        return random_search.best_estimator_
        
    except Exception as e:
        print(f"   [ERRO] Falha na otimização: {e}. Usando modelo padrão.")
        pipe.fit(X_train, y_train)
        return pipe

def analisar_performance_knn(model, X_test, y_test, feature_names, save_prefix='knn_resultado'):
    """
    Gera análises visuais. 
    """
    print("   [VISUALIZAÇÃO] Gerando gráficos de Importância e Confusão...")
    
    plt.figure(figsize=(18, 8))
    
    # --- 1. Permutation Importance ---
    plt.subplot(1, 2, 1)
    print("   [VISUALIZAÇÃO] Calculando Permutation Importance...")
    
    result_perm = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    sorted_idx_perm = result_perm.importances_mean.argsort()[-15:]
    
    plt.boxplot(
        result_perm.importances[sorted_idx_perm].T,
        vert=False,
        labels=[feature_names[i] for i in sorted_idx_perm]
    )
    plt.title("Feature Importance (Permutation)")
    plt.xlabel("Queda na Acurácia")

    # --- 2. Matriz de Confusão ---
    ax_cm = plt.subplot(1, 2, 2)
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, 
        display_labels=["Classe 0", "Classe 1"],
        cmap=plt.cm.Greens,
        normalize='true',
        ax=ax_cm
    )
    plt.title("Matriz de Confusão Normalizada")

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_completo.png")
    plt.close()
    
    print(f"   [VISUALIZAÇÃO] Gráfico salvo: {save_prefix}_completo.png")

# ==========================================
# FUNÇÃO WRAPPER (REFATORADA)
# ==========================================
def executar_pipeline(caminho_arquivo=None):
    print("="*60)
    print(" INICIANDO k-NEAREST NEIGHBORS (OTIMIZADO - MODULARIZADO)")
    print("="*60)

    try:
        # 1. Carregamento
        X, y, feats = carregar_e_processar_dados(caminho_arquivo)
        
        # Divisão
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"   [INFO] Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

        # 2. Treino (Otimizado com Pipeline)
        model = treinar_knn_otimizado(X_train, y_train)

        # 3. Métricas
        print("-" * 30)
        print("PERFORMANCE NO CONJUNTO DE TESTE:")
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.5
            
        print(f"  -> Acurácia: {acc:.2%}")
        print(f"  -> AUC ROC:  {auc:.4f}")
        
        report = classification_report(y_test, y_pred)
        print("\n" + report)
        
        with open("knn_metrics.txt", "w") as f:
            f.write(f"Acuracia: {acc}\nAUC: {auc}\n\nRelatorio:\n{report}")
        print("   [INFO] Métricas salvas em knn_metrics.txt")
        print("-" * 30)
        
        # 4. Visualização
        analisar_performance_knn(model, X_test, y_test, feats)
        
        print("\nProcesso finalizado.")

    except Exception as e:
        print(f"\n[ERRO CRÍTICO] {e}")

# ==========================================
# Execução Principal (Terminal)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arquivo', type=str, default=None)
    args = parser.parse_args()
    
    executar_pipeline(caminho_arquivo=args.arquivo)