# Arquivo: coletar_dados.py (VERSÃO CORRIGIDA)

import requests
import bs4 # beautifulsoup4
import pandas as pd
import os
import io

# --- Constantes ---
ONS_URL = "https://dados.ons.org.br/dataset/balanco-energia-subsistema"
CONSOLIDATED_FILE = "balanco_energia_consolidado.parquet"
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def run_full_etl():
    """Função principal que executa todo o pipeline de ETL."""
    print(">>> INICIANDO PROCESSO DE ETL DO BALANÇO ENERGÉTICO DA ONS <<<")

    try:
        print(f"[ETAPA 1/4] Buscando links de arquivos em: {ONS_URL}")
        response = requests.get(ONS_URL, headers=REQUEST_HEADERS, timeout=15)
        response.raise_for_status()
        soup = bs4.BeautifulSoup(response.text, 'html.parser')
        
        file_urls = [link.get('href') for link in soup.select("ul.resource-list a.resource-url-analytics") if link.get('href', '').endswith('.parquet')]
        
        if not file_urls:
            print("[ERRO] Nenhum arquivo .parquet encontrado na página. O site pode ter mudado.")
            return
        
        print(f"--- Encontrados {len(file_urls)} arquivos para baixar.")
    except Exception as e:
        print(f"[ERRO] Falha ao buscar informações no site da ONS: {e}")
        return

    try:
        print("\n[ETAPA 2/4] Baixando e lendo os arquivos parquet...")
        lista_dfs = [pd.read_parquet(io.BytesIO(requests.get(url, headers=REQUEST_HEADERS, timeout=90).content)) for url in file_urls]
        print("--- Download de todos os arquivos concluído.")
    except Exception as e:
        print(f"[ERRO] Falha durante o download de um dos arquivos: {e}")
        return

    try:
        print("\n[ETAPA 3/4] Consolidando e limpando os dados...")
        df = pd.concat(lista_dfs, ignore_index=True)
        
        df['din_instante'] = pd.to_datetime(df['din_instante'], errors='coerce')
        
        # ===== A CORREÇÃO ESTÁ AQUI =====
        colunas_valores = [
            'val_gerhidraulica', 'val_gertermica', 'val_gereolica', 
            'val_gersolar', 'val_carga', 'val_intercambio'  # <-- Adicionada a coluna que faltava
        ]
        
        for col in colunas_valores:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')
        
        df[colunas_valores] = df[colunas_valores].fillna(0)
        df.dropna(subset=['din_instante'], inplace=True)
        print("--- Limpeza e consolidação concluídas.")
    except Exception as e:
        print(f"[ERRO] Falha ao consolidar e limpar os dados: {e}")
        return

    try:
        print(f"\n[ETAPA 4/4] Salvando o arquivo consolidado como '{CONSOLIDATED_FILE}'...")
        df.to_parquet(CONSOLIDATED_FILE)
        print(f">>> SUCESSO! Arquivo '{CONSOLIDATED_FILE}' criado com {len(df):,} linhas.")
    except Exception as e:
        print(f"[ERRO] Falha ao salvar o arquivo final: {e}")
        return

if __name__ == "__main__":
    run_full_etl()