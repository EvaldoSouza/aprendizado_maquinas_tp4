import pandas as pd
import numpy as np

def processar_dados_riasec(caminho_arquivo, output_file="dataset_limpo_completo.tsv"):
    print("--- Iniciando Processamento de Dados (Modo Unificado) ---")
    
    # 1. Carregamento
    try:
        df = pd.read_csv(caminho_arquivo, sep='\t')
    except FileNotFoundError:
        print(f"Erro: Arquivo {caminho_arquivo} não encontrado.")
        return

    # 2. Filtragem Inicial (Educação >= 3 e Major não nulo)
    df_clean = df[(df['education'] >= 3) & (df['major'].notnull())].copy()
    
    # 3. Definição das Classes (Regex)
    regex_psycho = '(?i)psy|couns|beh|psi|mental health'
    regex_exclude = '(?i)psychi|psiqui|anim'
    
    mask_psycho = (
        df_clean['major'].str.contains(regex_psycho, na=False) & 
        ~df_clean['major'].str.contains(regex_exclude, na=False)
    )
    
    # CRUCIAL: Nomeamos a coluna como 'target' para ser compatível 
    # automaticamente com seus scripts de ML sem precisar de argumentos extras.
    df_clean['target'] = np.where(mask_psycho, 1, 0) # 1=Psi, 0=Outros (Numérico é melhor para alguns modelos)
    
    print(f"Total de registros após filtros: {len(df_clean)}")
    print(f"Distribuição das Classes:\n{df_clean['target'].value_counts()}")

    # 4. Cálculo dos Scores RIASEC
    riasec_letters = list('RIASEC')
    
    for letra in riasec_letters:
        # Pega colunas que começam com a letra e são seguidas por um dígito
        cols_letra = [c for c in df_clean.columns if c.startswith(letra) and c[1:].isdigit()]
        
        if cols_letra:
            df_clean[letra] = df_clean[cols_letra].sum(axis=1)

    # 5. Seleção Final de Colunas
    # Selecionamos as perguntas individuais + Scores Somados + Target
    cols_perguntas = [c for c in df_clean.columns if any(c.startswith(x) and c[1:].isdigit() for x in riasec_letters)]
    cols_scores = riasec_letters
    
    # Removemos colunas "lixo" do dataset original, mantendo apenas o útil
    cols_finais = cols_perguntas + cols_scores + ['target']
    df_final = df_clean[cols_finais]

    # 6. Salvamento (Arquivo Único)
    print(f"Salvando dataset unificado: {df_final.shape}")
    df_final.to_csv(output_file, sep='\t', index=False)
    
    print(f"Sucesso! Arquivo salvo como '{output_file}'")
    print("Agora você pode usar este arquivo no seu script 'main_treinar_todos.py'")

if __name__ == "__main__":
    # Ajuste o caminho de entrada conforme sua pasta real
    processar_dados_riasec("tp4/data.csv")