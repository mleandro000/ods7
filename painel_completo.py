# Arquivo: painel_completo.py
# Responsável por ler o arquivo local e exibir o dashboard interativo.

# ==============================================================================
# 1. IMPORTS E CONFIGURAÇÕES
# ==============================================================================
import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go

# --- Constantes e Configuração da Página ---
CONSOLIDATED_FILE = "balanco_energia_consolidado.parquet"
st.set_page_config(layout="wide", page_title="Análise Energética do Brasil", page_icon="🇧🇷")

# ==============================================================================
# 2. CARREGAMENTO E PREPARAÇÃO DE DADOS
# ==============================================================================
@st.cache_data
def load_and_prepare_data():
    """Lê o arquivo local e prepara todos os DataFrames para as análises."""
    df = pd.read_parquet(CONSOLIDATED_FILE)
    df['ano'] = pd.to_datetime(df['din_instante']).dt.year

    # Prepara dados do SIN
    df_sin = df[df['nom_subsistema'] == 'SISTEMA INTERLIGADO NACIONAL'].copy()
    fontes_renovaveis = ['val_gerhidraulica', 'val_gereolica', 'val_gersolar']
    df_sin['geracao_renovavel'] = df_sin[fontes_renovaveis].sum(axis=1)
    df_sin['geracao_total'] = df_sin['geracao_renovavel'] + df_sin['val_gertermica']

    # Agregação Anual Nacional
    analise_nacional_anual = df_sin.groupby('ano').agg(
        total_geral_brasil=('geracao_total', 'sum'),
        total_renovavel_brasil=('geracao_renovavel', 'sum'),
        Hidraulica=('val_gerhidraulica', 'sum'),
        Termica=('val_gertermica', 'sum'),
        Eolica=('val_gereolica', 'sum'),
        Solar=('val_gersolar', 'sum')
    ).reset_index()

    # Prepara Análise Regional
    df_regional = df[df['nom_subsistema'] != 'SISTEMA INTERLIGADO NACIONAL'].copy()
    df_regional['geracao_renovavel'] = df_regional[fontes_renovaveis].sum(axis=1)
    df_regional['geracao_total'] = df_regional['geracao_renovavel'] + df_regional['val_gertermica']
    
    analise_regional_anual = df_regional.groupby(['ano', 'nom_subsistema']).agg(
        geracao_renovavel_regiao=('geracao_renovavel', 'sum'),
        geracao_total_regiao=('geracao_total', 'sum')
    ).reset_index()
    
    # Análise Relativa: % de renováveis DENTRO da própria região
    mask = analise_regional_anual['geracao_total_regiao'] > 0
    analise_regional_anual['perc_renovavel_interno'] = 0.0
    analise_regional_anual.loc[mask, 'perc_renovavel_interno'] = (analise_regional_anual.loc[mask, 'geracao_renovavel_regiao'] / analise_regional_anual.loc[mask, 'geracao_total_regiao']) * 100

    # Análise Absoluta: Contribuição de cada região para o total de energia renovável do BRASIL
    analise_final_regional = pd.merge(analise_regional_anual, analise_nacional_anual[['ano', 'total_renovavel_brasil']], on='ano')
    analise_final_regional['contribuicao_perc_nacional'] = ((analise_final_regional['geracao_renovavel_regiao'] / analise_final_regional['total_renovavel_brasil']) * 100)
    
    # Prepara dados diários
    df_diario = df_sin.set_index('din_instante')[['val_gerhidraulica', 'val_gertermica', 'val_gereolica', 'val_gersolar']].resample('D').sum()
    df_diario.columns = ['Hidráulica', 'Térmica', 'Eólica', 'Solar']
    
    return analise_nacional_anual, analise_final_regional, df_diario

# ==============================================================================
# 3. FUNÇÕES DE PLOTAGEM
# ==============================================================================
def plot_matriz_energetica(df_anual):
    """Cria o gráfico de barras empilhadas da matriz energética anual."""
    df_plot = df_anual.set_index('ano')[['Hidraulica', 'Termica', 'Eolica', 'Solar']]
    fig = px.bar(df_plot, y=df_plot.columns, title='Composição da Matriz Energética por Fonte')
    fig.update_layout(barmode='stack', xaxis_title='Ano', yaxis_title='Geração (MWMED Somado)')
    return fig

def plot_analise_regional_relativa(df_regional):
    """Cria o gráfico da participação renovável DENTRO de cada subsistema, com eixos corrigidos."""
    fig = px.line(
        df_regional, x='ano', y='perc_renovavel_interno',
        facet_col='nom_subsistema', facet_col_wrap=2,
        color='nom_subsistema', markers=True, height=700,
        title='Análise Relativa: % de Renováveis na Matriz de CADA Subsistema'
    )
    fig.for_each_xaxis(lambda axis: axis.update(showticklabels=True, title='Ano'))
    fig.for_each_yaxis(lambda axis: axis.update(showticklabels=True, title='% Renovável Interno'))
    fig.update_layout(showlegend=False)
    return fig

def plot_analise_regional_absoluta(df_regional):
    """Cria o gráfico da contribuição absoluta de cada subsistema para o total de energia renovável do Brasil."""
    fig = px.bar(
        df_regional, x='ano', y='geracao_renovavel_regiao',
        color='nom_subsistema', title='Análise Absoluta: Contribuição de Cada Subsistema para a Geração Renovável do Brasil (MWMED)',
        labels={'ano': 'Ano', 'geracao_renovavel_regiao': 'Geração Renovável (MWMED)', 'nom_subsistema': 'Subsistema'}
    )
    fig.update_layout(barmode='stack')
    return fig

def plot_serie_diaria(df_diario):
    """Cria o gráfico de Série Diária com Médias Móveis."""
    fig = go.Figure()
    cores = {'Hidráulica': 'blue', 'Térmica': 'red', 'Eólica': 'green', 'Solar': 'orange'}
    for fonte in df_diario.columns:
        media_30d = df_diario[fonte].rolling(window=30).mean()
        media_90d = df_diario[fonte].rolling(window=90).mean()
        fig.add_trace(go.Scatter(x=df_diario.index, y=df_diario[fonte], mode='lines', name=fonte, legendgroup=fonte, line=dict(width=1), opacity=0.3, marker_color=cores[fonte]))
        fig.add_trace(go.Scatter(x=df_diario.index, y=media_30d, mode='lines', name=f'{fonte} Média 30d', legendgroup=fonte, line=dict(width=2), marker_color=cores[fonte]))
        fig.add_trace(go.Scatter(x=df_diario.index, y=media_90d, mode='lines', name=f'{fonte} Média 90d', legendgroup=fonte, line=dict(width=2, dash='dash'), marker_color=cores[fonte]))
    fig.update_layout(height=700, title_text='<b>Geração Diária com Tendências de Médias Móveis</b>', legend_title='<b>Fonte e Tendência</b>', xaxis_rangeslider_visible=True)
    return fig

# ==============================================================================
# 4. INTERFACE PRINCIPAL DA APLICAÇÃO
# ==============================================================================
def main():
    st.title("📊 Análise da Matriz Energética Brasileira e o ODS 7")

    if not os.path.exists(CONSOLIDATED_FILE):
        st.error(f"ERRO: Arquivo de dados '{CONSOLIDATED_FILE}' não encontrado!")
        st.warning("Por favor, execute o script `python coletar_dados.py` em seu terminal para gerar a base de dados.")
        st.stop()

    analise_nacional, analise_regional, df_diario = load_and_prepare_data()

    st.sidebar.title("Navegação")
    page_selection = st.sidebar.radio(
        "Selecione a Análise:",
        ("Sumário Executivo", "Análise da Matriz Energética", "Análise de Subsistemas", "Análise de Série Temporal")
    )

    if page_selection == "Sumário Executivo":
        st.header("Alinhamento da Matriz Energética Brasileira com o ODS 7")
        st.markdown("""
        O Brasil possui uma posição de destaque global com sua matriz energética predominantemente renovável. O **Objetivo de Desenvolvimento Sustentável (ODS) 7** da ONU visa "assegurar o acesso confiável, sustentável, moderno e a preço acessível à energia para todas e todos". 
        
        Este painel explora os dados de geração do Operador Nacional do Sistema Elétrico (ONS) para contextualizar o progresso do Brasil em relação a esta meta, com foco na diversificação, resiliência e nas particularidades regionais.
        """)
        st.info("Utilize o menu na barra lateral para navegar entre as diferentes seções da análise.")

    elif page_selection == "Análise da Matriz Energética":
        st.header("Composição da Matriz Energética Nacional (SIN)")
        st.markdown("""
        **Motivação:** Entender a contribuição de cada fonte de energia (Hidráulica, Térmica, Eólica, Solar) para a geração total do Sistema Interligado Nacional.
        **Análise:** O gráfico de barras empilhadas ilustra a forte dependência histórica da fonte hídrica, o papel complementar da geração térmica (frequentemente acionada para segurança do sistema) e, mais importante, a ascensão clara e exponencial das fontes eólica e solar na última década, diversificando a matriz.
        """)
        st.info("🎯 **Alinhamento Principal: ODS 7.2** (Manter elevada a participação de renováveis).")
        figura_matriz = plot_matriz_energetica(analise_nacional)
        st.plotly_chart(figura_matriz, use_container_width=True)

    elif page_selection == "Análise de Subsistemas":
        st.header("Análise Regional por Subsistemas")
        st.markdown("""
        O Brasil é dividido em 4 subsistemas operacionais (SE/CO, Sul, NE, Norte) com realidades energéticas muito distintas. Analisá-los separadamente é crucial para entender os desafios e oportunidades de cada região.
        """)
        
        st.subheader("Visão 1: Composição Interna de Cada Região")
        st.markdown("""
        Este primeiro gráfico responde: **"Dentro de cada região, qual a porcentagem de energia gerada que é renovável?"**. Ele mostra o quão "verde" é a matriz de cada subsistema.
        """)
        st.info("🎯 **Alinhamento Principal: ODS 7.1** (Acesso universal e equidade regional).")
        fig_relativa = plot_analise_regional_relativa(analise_regional)
        st.plotly_chart(fig_relativa, use_container_width=True)

        st.markdown("---")

        st.subheader("Visão 2: Contribuição de Cada Região para o Total Renovável do Brasil")
        st.markdown("""
        Este segundo gráfico responde à pergunta mais profunda: **"Do total de energia renovável gerado no Brasil, qual a contribuição (o 'peso') de cada região?"**. Aqui vemos a importância absoluta de cada subsistema. Por exemplo, o Nordeste não só tem uma matriz interna muito renovável, como também é um contribuinte massivo para o total de energia limpa do país, graças à sua força eólica.
        """)
        st.info("🎯 **Alinhamento Principal: ODS 7.b** (Expansão de infraestrutura para integração nacional).")
        fig_absoluta = plot_analise_regional_absoluta(analise_regional)
        st.plotly_chart(fig_absoluta, use_container_width=True)

    elif page_selection == "Análise de Série Temporal":
        st.header("Análise de Tendências Diárias com Médias Móveis")
        with st.expander("Clique aqui para entender as Médias Móveis e como interpretar este gráfico"):
            st.markdown("""
                **A Motivação:** Dados diários são "ruidosos". As **Médias Móveis (MM)** filtram essa volatilidade para revelar a tendência real.
                
                **Como Ler as Linhas:**
                - **Dado Diário (Área Sombreada):** A geração real de cada dia. Mostra a volatilidade.
                - **Média Móvel de 30 dias (Linha Sólida):** Revela a tendência mensal e a "dança" sazonal entre a geração **Hidráulica** e a **Térmica**.
                - **Média Móvel de 90 dias (Linha Tracejada):** Mostra a tendência de longo prazo, confirmando a ascensão estrutural da **Eólica** e **Solar** ao longo dos anos.
            """)
        
        fig_diario = plot_serie_diaria(df_diario)
        st.plotly_chart(fig_diario, use_container_width=True)

if __name__ == "__main__":
    main()