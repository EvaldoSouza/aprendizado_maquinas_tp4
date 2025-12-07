#
# ==========================================
# RuleFit Paralelo - Versão Estável (Correção de Serialização)
# Baseado em https://github.com/christophM/rulefit
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from functools import reduce
from ordered_set import OrderedSet
from joblib import Parallel, delayed

# ==========================================
# 1. Classes de Dados (Definidas no topo para Pickling)
# ==========================================

class CondicaoRegra:
    """Representa uma condição binária simples (ex: Feature_X > 5.5)."""
    def __init__(self, feature_index, threshold, operator, support, feature_name=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.operator = operator
        self.support = support
        self.feature_name = feature_name

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        feature = self.feature_name if self.feature_name else self.feature_index
        return "%s %s %s" % (feature, self.operator, self.threshold)

    def transform(self, X):
        # Otimização: uso direto de numpy array para evitar overhead
        col = X[:, self.feature_index]
        if self.operator == "<=":
            return (col <= self.threshold).astype(int)
        elif self.operator == ">":
            return (col > self.threshold).astype(int)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash((self.feature_index, self.threshold, self.operator, self.feature_name))


class Regra:
    """Classe que representa uma Regra composta por uma lista de Condicoes."""
    def __init__(self, rule_conditions, prediction_value):
        self.conditions = OrderedSet(rule_conditions)
        self.support = min([x.support for x in rule_conditions])
        self.prediction_value = prediction_value

    def transform(self, X):
        # Aplica todas as condições e faz o AND lógico (multiplicação)
        rule_applies = [condition.transform(X) for condition in self.conditions]
        return reduce(lambda x, y: x * y, rule_applies)

    def __str__(self):
        return " & ".join([x.__str__() for x in self.conditions])

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return sum([condition.__hash__() for condition in self.conditions])

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


# ==========================================
# 2. Funções Workers (Globais e sem dependência de NumPy Arrays de Objetos)
# ==========================================

def _transformar_batch_worker(regras_batch, X):
    """Worker para processar transformação em paralelo."""
    # regras_batch é uma LISTA de objetos Regra (seguro para pickle)
    return np.array([regra.transform(X) for regra in regras_batch]).T

def _extrair_regras_iterativo(tree, feature_names=None):
    """Extrai regras de uma única árvore (sklearn.tree._tree.Tree) de forma iterativa."""
    rules = []
    stack = [(0, [])] # Pilha: (node_id, conditions)
    
    # Cache attributes for speed
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value
    n_node_samples = tree.n_node_samples
    root_samples = float(n_node_samples[0])

    while stack:
        node_id, conditions = stack.pop()
        
        is_leaf = children_left[node_id] == children_right[node_id]
        
        if not is_leaf:
            feat_idx = feature[node_id]
            thresh = threshold[node_id]
            feat_name = feature_names[feat_idx] if feature_names is not None else feat_idx
            support = n_node_samples[node_id] / root_samples
            
            # Left child (<=)
            cond_left = CondicaoRegra(feat_idx, thresh, "<=", support, feat_name)
            stack.append((children_left[node_id], conditions + [cond_left]))
            
            # Right child (>)
            cond_right = CondicaoRegra(feat_idx, thresh, ">", support, feat_name)
            stack.append((children_right[node_id], conditions + [cond_right]))
        else:
            if len(conditions) > 0:
                pred_val = value[node_id][0][0]
                new_rule = Regra(conditions, pred_val)
                rules.append(new_rule)
                
    return rules

def _extrair_batch_worker(arvores_wrapper_batch, feature_names):
    """Worker para extrair regras de um lote de árvores."""
    regras_locais = []
    for tree_wrapper in arvores_wrapper_batch:
        # tree_wrapper é [estimator]
        regras_locais.extend(_extrair_regras_iterativo(tree_wrapper[0].tree_, feature_names))
    return regras_locais


# ==========================================
# 3. Pré-processamento e Ensemble
# ==========================================

class Winsorizer:
    def __init__(self, trim_quantile=0.0):
        self.trim_quantile = trim_quantile
        self.winsor_lims = None

    def train(self, X):
        self.winsor_lims = np.ones([2, X.shape[1]]) * np.inf
        self.winsor_lims[0, :] = -np.inf
        if self.trim_quantile > 0:
            for i_col in np.arange(X.shape[1]):
                lower = np.percentile(X[:, i_col], self.trim_quantile * 100)
                upper = np.percentile(X[:, i_col], 100 - self.trim_quantile * 100)
                self.winsor_lims[:, i_col] = [lower, upper]

    def trim(self, X):
        X_ = X.copy()
        X_ = np.where(X > self.winsor_lims[1, :], np.tile(self.winsor_lims[1, :], [X.shape[0], 1]),
                      np.where(X < self.winsor_lims[0, :], np.tile(self.winsor_lims[0, :], [X.shape[0], 1]), X))
        return X_


class FriedScale:
    def __init__(self, winsorizer=None):
        self.scale_multipliers = None
        self.winsorizer = winsorizer

    def train(self, X):
        if self.winsorizer is not None:
            X_trimmed = self.winsorizer.trim(X)
        else:
            X_trimmed = X

        scale_multipliers = np.ones(X.shape[1])
        for i_col in np.arange(X.shape[1]):
            num_uniq_vals = len(np.unique(X[:, i_col]))
            if num_uniq_vals > 2: 
                scale_multipliers[i_col] = 0.4 / (1.0e-12 + np.std(X_trimmed[:, i_col]))
        self.scale_multipliers = scale_multipliers

    def scale(self, X):
        if self.winsorizer is not None:
            return self.winsorizer.trim(X) * self.scale_multipliers
        else:
            return X * self.scale_multipliers


class EnsembleRegras:
    def __init__(self, tree_list, feature_names=None, n_jobs=None):
        self.tree_list = tree_list
        self.feature_names = feature_names
        self.n_jobs = n_jobs if n_jobs is not None else 1
        self.rules = OrderedSet()
        self._extrair_regras()
        self.rules = list(self.rules)

    def _extrair_regras(self):
        if not self.tree_list: return

        # CORREÇÃO CRÍTICA: Não usar np.array_split em objetos. Usar list slicing.
        n_total = len(self.tree_list)
        n_jobs = self.n_jobs if self.n_jobs > 0 else 1
        if self.n_jobs == -1: n_jobs = 4

        # Cria chunks usando listas puras do Python (seguro para pickle)
        chunk_size = int(np.ceil(n_total / n_jobs))
        chunks = [self.tree_list[i:i + chunk_size] for i in range(0, n_total, chunk_size)]
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_extrair_batch_worker)(chunk, self.feature_names) for chunk in chunks
        )
        
        for batch_rules in results:
            self.rules.update(batch_rules)

    def transform(self, X, coefs=None):
        n_total_rules = len(self.rules)
        if n_total_rules == 0:
             return np.zeros((X.shape[0], 0))

        rule_list_full = list(self.rules)
        indices_para_calcular = range(n_total_rules)
        
        if coefs is not None:
            indices_para_calcular = [i for i, c in enumerate(coefs) if c != 0]
            if not indices_para_calcular:
                return np.zeros((X.shape[0], n_total_rules))
            rules_to_process = [rule_list_full[i] for i in indices_para_calcular]
        else:
            rules_to_process = rule_list_full

        if not rules_to_process:
             return np.zeros((X.shape[0], n_total_rules))

        # CORREÇÃO CRÍTICA: List slicing em vez de numpy array_split
        n_proc = len(rules_to_process)
        if n_proc < 50:
             matriz_calculada = np.array([rule.transform(X) for rule in rules_to_process]).T
        else:
            n_jobs = self.n_jobs if self.n_jobs != -1 else 4
            chunk_size = int(np.ceil(n_proc / n_jobs))
            chunks = [rules_to_process[i:i + chunk_size] for i in range(0, n_proc, chunk_size)]
            
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_transformar_batch_worker)(chunk, X) for chunk in chunks
            )
            matriz_calculada = np.hstack(results)

        if coefs is None:
            return matriz_calculada
        else:
            res_full = np.zeros([X.shape[0], n_total_rules])
            res_full[:, indices_para_calcular] = matriz_calculada
            return res_full

    def __str__(self):
        return (map(lambda x: x.__str__(), self.rules)).__str__()


# ==========================================
# 4. Classe Principal RuleFit
# ==========================================

class RuleFit(BaseEstimator, TransformerMixin):
    def __init__(self, tree_size=4, sample_fract='default', max_rules=2000,
                 memory_par=0.01, rfmode='regress', lin_trim_quantile=0.025,
                 lin_standardise=True, exp_rand_tree_size=True, model_type='rl',
                 Cs=None, cv=3, tol=0.0001, max_iter=10000, n_jobs=-1, random_state=None):
        self.tree_size = tree_size
        self.sample_fract = sample_fract
        self.max_rules = max_rules
        self.memory_par = memory_par
        self.rfmode = rfmode
        self.lin_trim_quantile = lin_trim_quantile
        self.lin_standardise = lin_standardise
        self.exp_rand_tree_size = exp_rand_tree_size
        self.model_type = model_type
        self.Cs = Cs
        self.cv = cv
        self.tol = tol
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        self.tree_generator = None
        self.winsorizer = Winsorizer(trim_quantile=lin_trim_quantile)
        self.friedscale = FriedScale(self.winsorizer)

    def fit(self, X, y=None, feature_names=None):
        N = X.shape[0]
        if feature_names is None:
            self.feature_names = ["feature_" + str(x) for x in range(0, X.shape[1])]
        else:
            self.feature_names = feature_names

        # 1. Gerar Árvores
        if 'r' in self.model_type:
            n_estimators = int(np.ceil(self.max_rules / self.tree_size))
            sample_fract_ = min(0.5, (100 + 6 * np.sqrt(N)) / N)
            
            # Inicializa o gerador (sequencial)
            if self.rfmode == 'regress':
                self.tree_generator = GradientBoostingRegressor(n_estimators=n_estimators, max_leaf_nodes=self.tree_size, 
                                                              learning_rate=self.memory_par, subsample=sample_fract_, 
                                                              random_state=self.random_state, max_depth=100)
            else:
                self.tree_generator = GradientBoostingClassifier(n_estimators=n_estimators, max_leaf_nodes=self.tree_size, 
                                                               learning_rate=self.memory_par, subsample=sample_fract_, 
                                                               random_state=self.random_state, max_depth=100)
            
            # Treinamento
            if not self.exp_rand_tree_size:
                self.tree_generator.fit(X, y)
            else:
                np.random.seed(self.random_state)
                tree_sizes = np.random.exponential(scale=self.tree_size - 2, size=int(np.ceil(self.max_rules * 2 / self.tree_size)))
                tree_sizes = np.asarray([2 + np.floor(x) for x in tree_sizes], dtype=int)
                
                self.tree_generator.set_params(warm_start=True, n_estimators=0)
                curr_est = 0
                for size in tree_sizes:
                    if curr_est >= n_estimators: break
                    self.tree_generator.set_params(n_estimators=curr_est + 1, max_leaf_nodes=size, random_state=self.random_state)
                    self.tree_generator.fit(np.copy(X, order='C'), np.copy(y, order='C'))
                    curr_est += 1
                self.tree_generator.set_params(warm_start=False)

            # Extrai lista de estimadores (Flatten retorna numpy array, convertemos para lista de listas)
            tree_list = [[x] for x in self.tree_generator.estimators_.flatten()]
            
            self.rule_ensemble = EnsembleRegras(tree_list=tree_list, feature_names=self.feature_names, n_jobs=self.n_jobs)
            X_rules = self.rule_ensemble.transform(X)

        # 2. Linear
        if 'l' in self.model_type:
            self.winsorizer.train(X)
            if self.lin_standardise:
                self.friedscale.train(X)
                X_regn = self.friedscale.scale(X)
            else:
                X_regn = X.copy()

        # 3. Concatenar
        X_concat = np.zeros([X.shape[0], 0])
        if 'l' in self.model_type:
            X_concat = np.concatenate((X_concat, X_regn), axis=1)
        if 'r' in self.model_type:
            if X_rules.shape[0] > 0:
                X_concat = np.concatenate((X_concat, X_rules), axis=1)

        # 4. Lasso
        if self.rfmode == 'regress':
            self.lscv = LassoCV(cv=self.cv, n_jobs=self.n_jobs, random_state=self.random_state, tol=self.tol)
            self.lscv.fit(X_concat, y)
            self.coef_ = self.lscv.coef_
            self.intercept_ = self.lscv.intercept_
        else:
            self.lscv = LogisticRegressionCV(cv=self.cv, penalty='l1', solver='liblinear', 
                                           n_jobs=self.n_jobs, random_state=self.random_state, tol=self.tol)
            self.lscv.fit(X_concat, y)
            self.coef_ = self.lscv.coef_[0]
            self.intercept_ = self.lscv.intercept_[0]
        
        return self

    def predict(self, X):
        X_concat = self._prepare_data(X)
        return self.lscv.predict(X_concat)

    def predict_proba(self, X):
        X_concat = self._prepare_data(X)
        return self.lscv.predict_proba(X_concat)

    def _prepare_data(self, X):
        X_concat = np.zeros([X.shape[0], 0])
        if 'l' in self.model_type:
            if self.lin_standardise:
                X_concat = np.concatenate((X_concat, self.friedscale.scale(X)), axis=1)
            else:
                X_concat = np.concatenate((X_concat, X), axis=1)
        
        if 'r' in self.model_type:
            rule_coefs = self.coef_[-len(self.rule_ensemble.rules):]
            X_rules = self.rule_ensemble.transform(X, coefs=rule_coefs)
            if X_rules.shape[0] > 0:
                X_concat = np.concatenate((X_concat, X_rules), axis=1)
                
        return X_concat


# ==========================================
# 5. Funções de Execução e Visualização
# ==========================================

def carregar_e_processar_dados(caminho_arquivo=None, n_samples=1000):
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
                X = X.drop(columns=cols_constantes)
            
            feature_names = X.columns.tolist()
            X_values = X.values.astype(float)
            
            return X_values, y, feature_names
            
        except Exception as e:
            raise Exception(f"Erro crítico ao processar o arquivo: {e}")
    else:
        print("   [DADOS] Nenhum arquivo fornecido. Gerando dados sintéticos...")
        from sklearn.datasets import make_classification
        # Fix: 10 features para consistência com testes
        X, y = make_classification(n_samples=n_samples, n_features=10, 
                                   n_informative=5, n_redundant=2, random_state=42)
        feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
        return X, y, feature_names

def treinar_rulefit(X_train, y_train, feature_names, modo='classify'):
    print(f"   [MODELO] Iniciando treinamento RuleFit Paralelizado (Modo: {modo})...")
    
    model = RuleFit(
        rfmode=modo, 
        max_rules=2000, 
        tree_size=4, 
        cv=3, 
        random_state=42, 
        n_jobs=-1 
    )
    
    try:
        model.fit(X_train, y_train, feature_names=feature_names)
        n_rules = len(model.rule_ensemble.rules)
        n_active_rules = np.sum(model.coef_ != 0)
        print(f"   [MODELO] Treinamento concluído.")
        print(f"   [MODELO] Total de Regras Geradas: {n_rules}")
        print(f"   [MODELO] Regras Selecionadas: {n_active_rules}")
        return model
    except Exception as e:
        print(f"   [ERRO] Falha no treinamento RuleFit: {e}")
        raise e

def visualizar_regras(model, feature_names, save_prefix='rulefit_resultado'):
    print("   [VISUALIZAÇÃO] Gerando tabela de regras e gráficos...")
    
    rules = model.rule_ensemble.rules
    n_linear = len(feature_names) if 'l' in model.model_type else 0
    rule_coefs = model.coef_[-len(rules):]
    linear_coefs = model.coef_[:n_linear] if 'l' in model.model_type else []
    
    data_rules = []
    for i, r in enumerate(rules):
        if rule_coefs[i] != 0:
            data_rules.append({'Regra': str(r), 'Coeficiente': rule_coefs[i], 'Tipo': 'Regra'})
            
    for i, name in enumerate(feature_names):
        if i < len(linear_coefs) and linear_coefs[i] != 0:
             data_rules.append({'Regra': name, 'Coeficiente': linear_coefs[i], 'Tipo': 'Linear'})
             
    df_res = pd.DataFrame(data_rules)
    
    if df_res.empty:
        print("   [AVISO] O modelo Lasso zerou todos os coeficientes.")
        return

    df_res['Abs_Coef'] = df_res['Coeficiente'].abs()
    df_res = df_res.sort_values('Abs_Coef', ascending=False).head(20)
    
    def encurtar_texto(texto, limite=60):
        return texto if len(texto) <= limite else texto[:limite] + "..."
    
    df_res['Regra_Plot'] = df_res['Regra'].apply(encurtar_texto)
    
    plt.figure(figsize=(16, 10))
    sns.barplot(data=df_res, y='Regra_Plot', x='Coeficiente', hue='Tipo', dodge=False)
    plt.title("Top 20 Regras e Termos Lineares mais Importantes")
    plt.axvline(0, color='black', linewidth=0.8)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.subplots_adjust(left=0.3)
    
    caminho_img = f"{save_prefix}_importancia.png"
    plt.savefig(caminho_img, bbox_inches='tight')
    plt.close()
    
    caminho_csv = f"{save_prefix}_regras.csv"
    df_res.drop(columns=['Abs_Coef', 'Regra_Plot']).to_csv(caminho_csv, index=False)

    print(f"   [VISUALIZAÇÃO] Gráfico salvo: {caminho_img}")
    print(f"   [VISUALIZAÇÃO] Tabela de regras salva: {caminho_csv}")

def executar_pipeline(caminho_arquivo=None, modo='classify'):
    print("="*40)
    print(f" INICIANDO PROCESSO (RULEFIT PARALELO) - Modo: {modo}")
    print("="*40)

    try:
        X, y, feats = carregar_e_processar_dados(caminho_arquivo)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print(f"   [DADOS] Treino: {X_train.shape[0]}, Teste: {X_test.shape[0]}.")

        modelo = treinar_rulefit(X_train, y_train, feats, modo=modo)

        y_pred = modelo.predict(X_test)
        
        print("-" * 30)
        print("RELATÓRIO DE PERFORMANCE:")
        
        if modo == 'classify':
            acc = accuracy_score(y_test, y_pred)
            try:
                y_proba = modelo.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
            except:
                auc = 0.5
            print(f"  -> Acurácia:      {acc:.2%}")
            print(f"  -> AUC ROC:       {auc:.4f}")
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"  -> MSE:           {mse:.4f}")
            print(f"  -> R2 Score:      {r2:.4f}")
            
        print("-" * 30)

        visualizar_regras(modelo, feats)
        print("\nProcesso finalizado com sucesso.")

    except Exception as e:
        print(f"\n[ERRO FATAL]: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RuleFit Paralelo")
    parser.add_argument('--arquivo', type=str, default=None)
    parser.add_argument('--modo', type=str, default='classify', choices=['classify', 'regress'])
    args = parser.parse_args()

    executar_pipeline(caminho_arquivo=args.arquivo, modo=args.modo)