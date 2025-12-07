import os
import time

# Importando seus módulos já refatorados
import regressao_penalizada
import gbte
import knn
import random_forest
import prediction_rule_ensemble
import arvore_decisao
import modelo_gam

def executar_tudo():
    # --- CONFIGURAÇÃO ---
    # Defina aqui o caminho do arquivo que será usado por TODOS os scripts
    CAMINHO_ARQUIVO = "datasets/meu_dataset_treino.tsv" 
    
    # Verifica se o arquivo existe (Opcional: remove se quiser rodar sintético)
    if not os.path.exists(CAMINHO_ARQUIVO):
        print(f"AVISO: O arquivo '{CAMINHO_ARQUIVO}' não foi encontrado.")
        print("Os scripts irão gerar dados sintéticos automaticamente se suportado.\n")
        # Se quiser forçar que o arquivo exista, descomente a linha abaixo:
        # return 

    print(f"=== INICIANDO TREINAMENTO UNIFICADO ===")
    print(f"Arquivo alvo: {CAMINHO_ARQUIVO}\n")

    # Lista de algoritmos para rodar
    # Estrutura: (Módulo, Função Wrapper, Dicionário de Argumentos Extras)
    algoritmos = [
        (knn, "executar_pipeline", {}),
        (arvore_decisao, "executar_pipeline", {}),
        (random_forest, "executar_pipeline", {}),
        (gbte, "executar_pipeline", {}),
        (modelo_gam, "executar_pipeline", {}),
        # Regressão penalizada exige target_col (ajuste se necessário)
        (regressao_penalizada, "executar_pipeline", {"target_col": "target"}),
        # RuleFit pode rodar como classificador
        (prediction_rule_ensemble, "executar_pipeline", {"modo": "classify"}),
    ]

    for modulo, funcao_nome, kwargs in algoritmos:
        nome_algo = modulo.__name__
        print("\n" + "#"*80)
        print(f"### RODANDO MÓDULO: {nome_algo.upper()} ###")
        print("#"*80 + "\n")
        
        try:
            # Reflexão: Obtém a função dentro do módulo importado
            func = getattr(modulo, funcao_nome)
            
            # Chama a função passando o arquivo e quaisquer argumentos extras
            func(caminho_arquivo=CAMINHO_ARQUIVO, **kwargs)
            
            print(f"\n[SUCESSO] Módulo {nome_algo} finalizado.")
        except AttributeError:
            print(f"\n[ERRO DE CONFIGURAÇÃO] A função '{funcao_nome}' não existe em {nome_algo}.")
        except Exception as e:
            print(f"\n[FALHA DE EXECUÇÃO] Erro ao rodar {nome_algo}: {e}")
            # 'continue' garante que um erro em um script não pare os outros
            continue
        
        # Pequena pausa para garantir que os prints não se misturem no buffer
        time.sleep(1) 

    print("\n" + "="*80)
    print("=== TODOS OS TREINAMENTOS FORAM CONCLUÍDOS ===")
    print("="*80)

if __name__ == "__main__":
    executar_tudo()