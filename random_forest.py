import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import time

# ==============================================================================
# ARQUITETURA: RANDOM FOREST
# ==============================================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, ConfusionMatrixDisplay

def carregar_e_processar_dados(caminho_arquivo=None, n_samples=2000):
    """
    Carrega e prepara os dados.
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
        X, y = make_classification(n_samples=n_samples, n_features=25, 
                                   n_informative=15, n_redundant=5, 
                                   weights=[0.7, 0.3],
                                   random_state=42)
        feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
        return X.astype(np.float32), y, feature_names

def treinar_rf_otimizada(X_train, y_train):
    """
    Treina uma Random Forest robusta utilizando Otimização Bayesiana ou Aleatória.
    """
    print("   [MODELO] Configurando Random Forest Paralela...")
    
    rf = RandomForestClassifier(
        random_state=42, 
        oob_score=True,
        class_weight='balanced'
    )
    
    param_dist = {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True]
    }
    
    print("   [MODELO] Iniciando RandomizedSearchCV (Paralelo)...")
    start_time = time.time()
    
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=10,
        scoring='roc_auc',
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    try:
        random_search.fit(X_train, y_train)
        elapsed = time.time() - start_time
        print(f"   [MODELO] Otimização concluída em {elapsed:.2f} segundos.")
        print(f"   [MODELO] Melhores parâmetros: {random_search.best_params_}")
        
        best_model = random_search.best_estimator_
        
        if best_model.oob_score:
            print(f"   [MODELO] OOB Score (Validação Interna): {best_model.oob_score_:.4f}")
            
        return best_model
        
    except Exception as e:
        print(f"   [ERRO] Falha na otimização: {e}. Usando modelo padrão.")
        rf.fit(X_train, y_train)
        return rf

def analisar_performance_rf(model, X_test, y_test, feature_names, save_prefix='rf_resultado'):
    """
    Gera análises avançadas comparando MDI vs Permutation Importance.
    """
    print("   [VISUALIZAÇÃO] Gerando gráficos comparativos de Importância e Confusão...")
    
    plt.figure(figsize=(18, 12))
    
    # --- 1. MDI Importance ---
    plt.subplot(2, 2, 1)
    importances_mdi = model.feature_importances_
    std_mdi = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices_mdi = np.argsort(importances_mdi)[-15:]
    
    plt.title("Feature Importance (MDI - Padrão)")
    plt.barh(range(len(indices_mdi)), importances_mdi[indices_mdi], xerr=std_mdi[indices_mdi], align="center", color='skyblue')
    plt.yticks(range(len(indices_mdi)), [feature_names[i] for i in indices_mdi])
    plt.xlabel("Importância Relativa (Gini)")
    
    # --- 2. Permutation Importance ---
    plt.subplot(2, 2, 2)
    print("   [VISUALIZAÇÃO] Calculando Permutation Importance (Isso pode demorar um pouco)...")
    
    result_perm = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
    )
    sorted_idx_perm = result_perm.importances_mean.argsort()[-15:]
    
    plt.boxplot(
        result_perm.importances[sorted_idx_perm].T,
        vert=False,
        labels=[feature_names[i] for i in sorted_idx_perm]
    )
    plt.title("Permutation Importance (Baseado em Teste)")
    plt.xlabel("Queda na Acurácia")

    # --- 3. Matriz de Confusão ---
    ax_cm = plt.subplot(2, 1, 2)
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, 
        display_labels=["Classe 0", "Classe 1"],
        cmap=plt.cm.Blues,
        normalize='true',
        ax=ax_cm
    )
    plt.title("Matriz de Confusão Normalizada (Recall por Classe)")

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_completo.png")
    plt.close()
    
    print(f"   [VISUALIZAÇÃO] Gráfico salvo: {save_prefix}_completo.png")

# ==========================================
# FUNÇÃO WRAPPER (REFATORADA)
# ==========================================
def executar_pipeline(caminho_arquivo=None):
    print("="*60)
    print(" INICIANDO RANDOM FOREST (PARALELA & ROBUSTA - MODULARIZADA)")
    print("="*60)

    try:
        # 1. Carregamento
        X, y, feats = carregar_e_processar_dados(caminho_arquivo)
        
        # Divisão
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print(f"   [INFO] Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

        # 2. Treino (Otimizado)
        model = treinar_rf_otimizada(X_train, y_train)

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
        
        with open("rf_metrics.txt", "w") as f:
            f.write(f"Acuracia: {acc}\nAUC: {auc}\n\nRelatorio:\n{report}")
        print("   [INFO] Métricas salvas em rf_metrics.txt")
        print("-" * 30)
        
        # 4. Visualização Avançada
        analisar_performance_rf(model, X_test, y_test, feats)
        
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