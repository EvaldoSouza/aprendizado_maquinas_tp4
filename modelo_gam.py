import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from pygam import LogisticGAM, s, f, l

def carregar_e_processar_dados(caminho_arquivo=None, n_samples=1000):
    """
    Carrega dados e realiza pré-processamento robusto.
    """
    if caminho_arquivo:
        try:
            df = pd.read_csv(caminho_arquivo, sep='\t')
            target_col = 'target' if 'target' in df.columns else df.columns[-1]
            
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
            raise Exception(f"Erro ao ler arquivo: {e}")
    else:
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=n_samples, n_features=10, 
                                   n_informative=5, n_redundant=2, random_state=42)
        
        X[:, 8] = np.where(X[:, 8] > 0, 1, 0) 
        X[:, 9] = np.where(X[:, 9] > 0, 1, 0)
        
        feature_names = [f"Feat_{i}" for i in range(X.shape[1])]
        return X, y, feature_names

def construir_termos_dinamicamente(X_train, n_splines=20):
    """
    CRUCIAL: Define se cada coluna deve ser tratada como Spline (curva) 
    ou Linear (binária/categórica).
    """
    terms = None
    term_types = []
    
    for i in range(X_train.shape[1]):
        col_values = X_train[:, i]
        n_unique = len(np.unique(col_values))
        
        if n_unique <= 2:
            current_term = l(i)
            term_types.append('linear')
        else:
            current_term = s(i, n_splines=n_splines)
            term_types.append('spline')
        
        if terms is None:
            terms = current_term
        else:
            terms += current_term
            
    return terms, term_types

def treinar_gam_automatico(X_train, y_train):
    """
    Configura e treina um GAM com seleção inteligente de termos.
    """
    print("   [GAM] Analisando estrutura dos dados...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = np.nan_to_num(X_train_scaled)

    n_samples = X_train_scaled.shape[0]
    n_splines_safe = min(20, max(4, n_samples // 10))
    
    terms, types = construir_termos_dinamicamente(X_train_scaled, n_splines=n_splines_safe)
    print(f"   [GAM] Termos definidos: {types.count('spline')} Splines, {types.count('linear')} Lineares.")

    search_lambdas = np.logspace(-3, 3, 5)
    
    gam = LogisticGAM(terms, max_iter=100, tol=1e-3)

    print("   [GAM] Iniciando treinamento (GridSearch)...")
    try:
        gam = gam.gridsearch(X_train_scaled, y_train, lam=search_lambdas, progress=False)
    except Exception as e:
        print(f"   [AVISO] GridSearch falhou ({e}). Tentando fit simples.")
        gam.fit(X_train_scaled, y_train)
    
    return gam, scaler, types

def plotar_interpretacao(gam, scaler, feature_names, term_types, save_path='gam_results.png'):
    """
    Plota as Dependências Parciais.
    """
    if not feature_names: return

    n_features = len(feature_names)
    cols = 3
    rows = (n_features // cols) + (1 if n_features % cols > 0 else 0)
    
    plt.figure(figsize=(15, 3.5 * rows))
    
    means = scaler.mean_
    scales = scaler.scale_
    
    for i, feature in enumerate(feature_names):
        plt.subplot(rows, cols, i + 1)
        
        try:
            XX = gam.generate_X_grid(term=i)
            pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
            x_plot = (XX[:, i] * scales[i]) + means[i]
            
            col_type = term_types[i] if i < len(term_types) else '?'
            
            if col_type == 'linear':
                plt.plot(x_plot, pdep, 'o-', label=f"{feature} (Lin)", color='darkgreen', lw=2)
            else:
                plt.plot(x_plot, pdep, label=f"{feature} (Spline)", color='blue', lw=2)
                plt.fill_between(x_plot, confi[:, 0], confi[:, 1], color='blue', alpha=0.1)

            plt.title(f"{feature}", fontsize=10, fontweight='bold')
            plt.grid(True, alpha=0.2, linestyle='--')
            
            if i % cols == 0: plt.ylabel('Log-Odds (Contribuição)')
            
        except Exception as e:
            plt.text(0.5, 0.5, f"Erro: {str(e)}", ha='center')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Gráficos salvos em: {save_path}")

# ==========================================
# FUNÇÃO WRAPPER (REFATORADA)
# ==========================================
def executar_pipeline(caminho_arquivo=None):
    print("="*60)
    print(" INICIANDO GENERALIZED ADDITIVE MODELS (GAM - MODULARIZADO)")
    print("="*60)

    try:
        # 1. Dados
        X, y, feats = carregar_e_processar_dados(caminho_arquivo)
        
        # Validação
        if len(y) < 20: 
            raise ValueError("Dados insuficientes.")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # 2. Treino (com detecção automática de estrutura)
        gam, scaler, t_types = treinar_gam_automatico(X_train, y_train)

        # 3. Avaliação
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = np.nan_to_num(X_test_scaled)
        
        acc = accuracy_score(y_test, gam.predict(X_test_scaled))
        try:
            # Tenta calcular probabilidade para AUC
            probs = gam.predict_proba(X_test_scaled)
            auc = roc_auc_score(y_test, probs)
        except:
            auc = 0.0

        print("-" * 30)
        print(f"Performance no Teste:")
        print(f"  -> Acurácia: {acc:.4f}")
        print(f"  -> AUC ROC:  {auc:.4f}")
        try:
            r2 = gam.statistics_.get('pseudo_r2', {}).get('McFadden', 0)
            print(f"  -> Pseudo R2: {r2:.4f}")
        except:
            pass
        print("-" * 30)

        # 4. Visualização Corrigida
        plotar_interpretacao(gam, scaler, feats, t_types)
        
        print("\nProcesso finalizado com sucesso.")

    except Exception as e:
         print(f"\n[ERRO FATAL] O programa foi interrompido: {e}")

# ==========================================
# Execução Principal (Terminal)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arquivo', type=str, default=None)
    args = parser.parse_args()
    
    executar_pipeline(caminho_arquivo=args.arquivo)