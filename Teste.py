# -*- coding: utf-8 -*-
"""
DASHBOARD A PARTIR DO NOTEBOOK ORIGINAL

Autor: Leandro
Data: 21 de junho de 2025
Descrição:
Este painel foi construído recriando as análises e visualizações
do notebook original 'untitled56.py', transformando-o em uma
aplicação Streamlit interativa.
"""

# ==============================================================================
# 1. IMPORTS E CONFIGURAÇÕES GERAIS
# ==============================================================================
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# --- Constantes e Configuração da Página ---
ONS_URL = "https://dados.ons.org.br/dataset/balanco-energia-subsistema"
RAW_DATA_DIR = "dados_ons_parquet_notebook" # Nova pasta para não conflitar
CONSOLIDATED_FILE = "balanco_energia_consolidado_notebook.parquet"
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
st.set_page_config(layout="wide", page_title="Meu Painel de Energia", page_icon="💡")

# ==============================================================================
# 2. MÓDULO DE COLETA E PROCESSAMENTO DE DADOS (Baseado nas Células 2 a 6)
# ==============================================================================
def run_full_etl_from_notebook_logic():
    """
    Executa o pipeline de ETL completo, baseado na lógica do notebook original.
    Retorna True em sucesso, False em falha.
    """
    try:
        with st.status("Iniciando ETL...", expanded=True) as status:
            # Raspagem e Download
            status.update(label="Etapa 1: Baixando dados da ONS...")
            response = requests.get(ONS_URL, headers=REQUEST_HEADERS, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            links_parquet = [link.get('href') for link in soup.select("ul.resource-list a.resource-url-analytics") if link.get('href', '').endswith('.parquet')]
            if not links_parquet:
                status.update(label="Nenhum link .parquet encontrado.", state="error")
                return False

            os.makedirs(RAW_DATA_DIR, exist_ok=True)
            lista_dfs = [pd.read_parquet(io.BytesIO(requests.get(link, headers=REQUEST_HEADERS, timeout=90).content)) for link in links_parquet]
            
            # Consolidação e Limpeza
            status.update(label="Etapa 2: Consolidando e limpando dados...")
            df = pd.concat(lista_dfs, ignore_index=True)
            df['din_instante'] = pd.to_datetime(df['din_instante'], errors='coerce')
            
            colunas_valores = ['val_gerhidraulica', 'val_gertermica', 'val_gereolica', 'val_gersolar', 'val_carga', 'val_intercambio']
            for col in colunas_valores:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')
            df[colunas_valores] = df[colunas_valores].fillna(0)
            df.dropna(subset=['din_instante'], inplace=True)

            status.update(label="Etapa 3: Salvando arquivo consolidado...")
            df.to_parquet(CONSOLIDATED_FILE)
            status.update(label="Processo ETL concluído!", state="complete")
        return True
    except Exception as e:
        st.error("Falha crítica durante o processo de ETL.")
        st.exception(e)
        if 'status' in locals(): status.update(label="Processo falhou.", state="error")
        return False

# ==============================================================================
# 3. MÓDULO DE PREPARAÇÃO DE DADOS E VISUALIZAÇÃO (Baseado nas Células de Análise)
# ==============================================================================

@st.cache_data
def load_and_prepare_data():
    """Lê o arquivo local e prepara todos os DataFrames para as análises."""
    df_consolidado = pd.read_parquet(CONSOLIDATED_FILE)
    df_consolidado['ano'] = pd.to_datetime(df_consolidado['din_instante']).dt.year

    # --- DataFrames para o SIN ---
    df_sin = df_consolidado[df_consolidado['nom_subsistema'] == 'SISTEMA INTERLIGADO NACIONAL'].copy()
    df_sin['geracao_renovavel'] = df_sin['val_gerhidraulica'] + df_sin['val_gereolica'] + df_sin['val_gersolar']
    df_sin['geracao_total'] = df_sin['geracao_renovavel'] + df_sin['val_gertermica']
    
    # --- DataFrame para Análise Anual Agregada ---
    analise_anual = df_sin.groupby('ano').agg(
        Hidraulica=('val_gerhidraulica', 'sum'), Termica=('val_gertermica', 'sum'),
        Eolica=('val_gereolica', 'sum'), Solar=('val_gersolar', 'sum'),
        geracao_renovavel=('geracao_renovavel', 'sum'), geracao_total=('geracao_total', 'sum')
    )
    analise_anual['percentual_renovavel'] = (analise_anual['geracao_renovavel'] / analise_anual['geracao_total']) * 100
    analise_anual['crescimento_anual_%'] = analise_anual['geracao_total'].pct_change() * 100

    # --- DataFrame para Análise Regional ---
    df_regional = df_consolidado[df_consolidado['nom_subsistema'] != 'SISTEMA INTERLIGADO NACIONAL'].copy()
    df_regional['geracao_renovavel'] = df_regional['val_gerhidraulica'] + df_regional['val_gereolica'] + df_regional['val_gersolar']
    df_regional['geracao_total'] = df_regional['geracao_renovavel'] + df_regional['val_gertermica']
    analise_regional_anual = df_regional.groupby(['ano', 'nom_subsistema']).agg(
        geracao_renovavel=('geracao_renovavel', 'sum'), geracao_total=('geracao_total', 'sum')
    ).reset_index()
    analise_regional_anual['percentual_renovavel'] = (analise_regional_anual['geracao_renovavel'] / analise_regional_anual['geracao_total']) * 100

    # --- DataFrame para Série Diária ---
    df_diario = df_sin.set_index('din_instante')[['val_gerhidraulica', 'val_gertermica', 'val_gereolica', 'val_gersolar']].resample('D').sum()
    df_diario.columns = ['Hidráulica', 'Térmica', 'Eólica', 'Solar']
    
    return analise_anual, analise_regional_anual, df_diario

def plotar_dashboard_geracao_total(analise_anual):
    """Cria o Dashboard 1 do notebook."""
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "table"}]], column_widths=[0.7, 0.3], subplot_titles=("Gráfico de Geração Total", "Tabela de Dados"))
    fig.add_trace(go.Bar(x=analise_anual.index, y=analise_anual['geracao_total'], text=analise_anual['geracao_total'].apply(lambda x: f'{x/1e6:,.2f}M'), textposition='outside', marker_color='darkblue', name='Geração Total'), row=1, col=1)
    fig.add_trace(go.Table(header=dict(values=['Ano', 'Geração Total (MWMED)'], fill_color='darkblue', font=dict(color='white')), cells=dict(values=[analise_anual.index, analise_anual['geracao_total']], format=[None, ",.0f"])), row=1, col=2)
    fig.update_layout(title_text='<b>Dashboard: Geração Total de Energia no Brasil por Ano</b>', showlegend=False, height=500)
    return fig

def plotar_dashboard_renovaveis(analise_anual):
    """Cria o Dashboard 2 do notebook."""
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scatter"}, {"type": "table"}]], column_widths=[0.6, 0.4], subplot_titles=("Gráfico de Participação", "Tabela de Dados"))
    fig.add_trace(go.Scatter(x=analise_anual.index, y=analise_anual['percentual_renovavel'], mode='lines+markers+text', text=[f'{p:.1f}%' for p in analise_anual['percentual_renovavel']], textposition='top center', marker_color='green', name='Renováveis %'), row=1, col=1)
    fig.add_trace(go.Table(header=dict(values=['Ano', 'Total Gerado', 'Total Renovável', '% Renovável'], fill_color='green', font=dict(color='white')), cells=dict(values=[analise_anual.index, analise_anual['geracao_total'], analise_anual['geracao_renovavel'], analise_anual['percentual_renovavel']], format=[None, ",.0f", ",.0f", ".1f"])), row=1, col=2)
    fig.update_layout(title_text='<b>Dashboard: Percentual de Energia Renovável na Matriz Energética</b>', showlegend=False, height=500)
    return fig

def plotar_serie_diaria_com_medias(df_diario):
    """Cria a Análise 3 do notebook, com médias móveis."""
    fig = go.Figure()
    cores = {'Hidráulica': 'blue', 'Térmica': 'red', 'Eólica': 'green', 'Solar': 'orange'}
    for fonte in df_diario.columns:
        media_30d = df_diario[fonte].rolling(window=30).mean()
        media_90d = df_diario[fonte].rolling(window=90).mean()
        fig.add_trace(go.Scatter(x=df_diario.index, y=df_diario[fonte], mode='lines', name=fonte, legendgroup=fonte, line=dict(width=1), opacity=0.3, marker_color=cores[fonte]))
        fig.add_trace(go.Scatter(x=df_diario.index, y=media_30d, mode='lines', name=f'{fonte} Média 30d', legendgroup=fonte, line=dict(width=2), marker_color=cores[fonte]))
        fig.add_trace(go.Scatter(x=df_diario.index, y=media_90d, mode='lines', name=f'{fonte} Média 90d', legendgroup=fonte, line=dict(width=2, dash='dash'), marker_color=cores[fonte]))
    fig.update_layout(height=700, title_text='<b>Análise Detalhada: Geração Diária com Tendências de Médias Móveis</b>', legend_title='<b>Fonte e Tendência</b>', xaxis_rangeslider_visible=True)
    return fig

# ==============================================================================
# 4. INTERFACE PRINCIPAL DA APLICAÇÃO
# ==============================================================================
def main():
    st.title("💡 Painel de Análise Energética (Baseado em Meu Código)")
    st.markdown(f"**Fonte dos Dados:** [Portal de Dados Abertos do ONS]({ONS_URL})")

    st.sidebar.title("Controle de Dados")
    if st.sidebar.button("Executar ETL (Baixar e Processar Dados)"):
        if run_full_etl_from_notebook_logic():
            st.sidebar.success("Dados processados com sucesso!")
            st.cache_data.clear() # Limpa o cache para forçar a releitura
            st.rerun() # Recarrega a página para o painel aparecer
        else:
            st.sidebar.error("O processo de ETL falhou.")
    
    st.sidebar.markdown("---")
    
    if not os.path.exists(CONSOLIDATED_FILE):
        st.info("👋 Bem-vindo! Para começar, clique no botão 'Executar ETL' na barra lateral para baixar e processar os dados da ONS.")
        st.stop()
    
    # --- Carrega os dados e exibe o painel ---
    analise_anual, analise_regional_anual, df_diario = load_and_prepare_data()
    
    st.sidebar.header("Minhas Análises")
    analise_selecionada = st.sidebar.radio(
        "Selecione uma análise para visualizar:",
        ("Visão Geral (Dashboards 1 e 2)", "Análise de Tendências Diárias", "Outras Análises") # Adicione mais opções aqui
    )
    
    if analise_selecionada == "Visão Geral (Dashboards 1 e 2)":
        st.header("Dashboards Gerais")
        st.plotly_chart(plotar_dashboard_geracao_total(analise_anual), use_container_width=True)
        st.plotly_chart(plotar_dashboard_renovaveis(analise_anual), use_container_width=True)
        
    elif analise_selecionada == "Análise de Tendências Diárias":
        st.header("Análise de Sazonalidade e Tendência")
        with st.expander("Clique para ler a interpretação original desta análise"):
            st.markdown("""
            **Como Interpretar Este Gráfico Detalhado:**
            As médias móveis suavizam os "ruídos" diários para revelar a tendência fundamental por trás dos dados.
            - **Geração Hidráulica (Azul) vs. Térmica (Vermelho):** Note como são inversamente proporcionais. Os picos de geração térmica coincidem com os vales da hidráulica, mostrando o uso das térmicas para segurança energética em períodos secos.
            - **Crescimento Eólico (Verde) e Solar (Laranja):** As médias móveis mostram claramente a trajetória de ascensão dessas fontes na matriz, ignorando a intermitência diária.
            Use o zoom na parte inferior para explorar períodos específicos.
            """)
        st.plotly_chart(plotar_serie_diaria_com_medias(df_diario), use_container_width=True)
        
    elif analise_selecionada == "Outras Análises":
        st.header("Outras Análises do Notebook")
        
        st.subheader("Análise de Crescimento Anual")
        fig_crescimento = px.bar(analise_anual, y='geracao_total', title="Crescimento da Geração Total")
        st.plotly_chart(fig_crescimento, use_container_width=True)
        
        st.subheader("Análise Regional de Renováveis")
        fig_regional = px.line(analise_regional_anual, x='ano', y='percentual_renovavel', color='nom_subsistema', title="Percentual Renovável por Subsistema")
        st.plotly_chart(fig_regional, use_container_width=True)


if __name__ == "__main__":
    main()