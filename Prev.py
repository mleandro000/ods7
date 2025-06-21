# Arquivo: painel_final_com_previsoes.py
# Painel unificado que lê a base de dados mestre e exibe todas as análises, incluindo previsões.

import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.tsa.api as smt

# --- Constantes e Configuração ---
CONSOLIDATED_FILE = "balanco_energia_consolidado.parquet"
st.set_page_config(layout="wide", page_title="Análise Energética do Brasil com Previsões", page_icon="🇧🇷")

# --- Módulo de Preparação de Dados (em cache) ---
@st.cache_data
def load_and_prepare_all_data():
    """
    Lê a base mestre e prepara TODOS os dataframes necessários para as análises.
    Retorna o dataframe original e os processados.
    """
    df = pd.read_parquet(CONSOLIDATED_FILE)
    df['ano'] = pd.to_datetime(df['din_instante']).dt.year

    # Prepara dados do SIN
    df_sin = df[df['nom_subsistema'] == 'SISTEMA INTERLIGADO NACIONAL'].copy()
    fontes_renovaveis = ['val_gerhidraulica', 'val_gereolica', 'val_gersolar']
    df_sin['geracao_renovavel'] = df_sin[fontes_renovaveis].sum(axis=1)
    df_sin['geracao_total'] = df_sin['geracao_renovavel'] + df_sin['val_gertermica']

    # Prepara Análise Anual (com base no SIN)
    analise_anual = df_sin.groupby('ano').agg(
        Hidraulica=('val_gerhidraulica', 'sum'), Termica=('val_gertermica', 'sum'),
        Eolica=('val_gereolica', 'sum'), Solar=('val_gersolar', 'sum'),
        total_renovavel=('geracao_renovavel', 'sum'), total_geral=('geracao_total', 'sum')
    ).reset_index()
    analise_anual['perc_renovavel_total'] = (analise_anual['total_renovavel'] / analise_anual['total_geral']) * 100
    analise_anual['perc_eolica'] = (analise_anual['Eolica'] / analise_anual['total_geral']) * 100
    analise_anual['perc_solar'] = (analise_anual['Solar'] / analise_anual['total_geral']) * 100
    analise_anual['crescimento_eolica'] = analise_anual['perc_eolica'].pct_change() * 100
    analise_anual['crescimento_solar'] = analise_anual['perc_solar'].pct_change() * 100

    # Prepara Análise Regional (com base no DataFrame completo)
    df_regional = df[df['nom_subsistema'] != 'SISTEMA INTERLIGADO NACIONAL'].copy()
    df_regional['geracao_renovavel'] = df_regional[fontes_renovaveis].sum(axis=1)
    df_regional['geracao_total'] = df_regional['geracao_renovavel'] + df_regional['val_gertermica']
    analise_regional_anual = df_regional.groupby(['ano', 'nom_subsistema']).agg(
        geracao_renovavel_regiao=('geracao_renovavel', 'sum'),
        geracao_total_regiao=('geracao_total', 'sum')
    ).reset_index()
    analise_regional_anual = pd.merge(analise_regional_anual, analise_anual[['ano', 'total_renovavel']], on='ano', suffixes=('', '_brasil'))
    
    # Prepara Dados Diários (com base no SIN)
    df_diario = df_sin.set_index('din_instante')[['val_gerhidraulica', 'val_gertermica', 'val_gereolica', 'val_gersolar']].resample('D').sum()
    df_diario.columns = ['Hidráulica', 'Térmica', 'Eólica', 'Solar']
    
    return df, analise_anual, analise_regional_anual, df_diario # Retornando df original também

# --- Funções de Previsão ---
def predict_linear_regression(df, target_column, current_year, forecast_until_year):
    """
    Realiza a previsão usando Regressão Linear Simples.
    Args:
        df (pd.DataFrame): DataFrame contendo os dados históricos.
        target_column (str): Nome da coluna a ser prevista.
        current_year (int): O último ano de dados históricos disponíveis.
        forecast_until_year (int): O ano até o qual se deseja prever.
    Returns:
        pd.DataFrame: DataFrame com os dados históricos e as previsões combinadas.
        pd.DataFrame: DataFrame apenas com as previsões futuras.
        float: Coeficiente angular (slope) do modelo.
        float: Intercepto do modelo.
    """
    model = LinearRegression()

    # Preparar dados para o modelo (ano como X, target_column como Y)
    X = df['ano'].values.reshape(-1, 1)
    y = df[target_column].values

    # Treinar o modelo
    model.fit(X, y)

    # Criar anos futuros para previsão
    future_years = np.arange(current_year + 1, forecast_until_year + 1).reshape(-1, 1)
    predictions = model.predict(future_years)

    # Criar DataFrame com as previsões
    df_predictions = pd.DataFrame({
        'ano': future_years.flatten(),
        target_column: predictions # Usar o nome da coluna original para facilitar a concatenação
    })
    
    # Combinar dados históricos e previsões
    df_combined = pd.concat([df[['ano', target_column]], df_predictions], ignore_index=True)
    return df_combined, df_predictions, model.coef_[0], model.intercept_

def predict_ses_for_daily_data(df_diario, forecast_days):
    """
    Realiza a previsão para dados diários usando Suavização Exponencial Simples (SES).
    Args:
        df_diario (pd.DataFrame): DataFrame com os dados diários, com 'din_instante' como índice.
        forecast_days (int): Número de dias para prever.
    Returns:
        pd.DataFrame: DataFrame com os dados diários originais e as previsões combinadas.
    """
    predictions_dfs = []
    
    for col in df_diario.columns:
        series = df_diario[col].dropna()
        
        # Ajustar o modelo SES
        # Usamos alpha=0.9 para dar mais peso às observações recentes
        fit = smt.SimpleExpSmoothing(series, initialization_method="estimated").fit(smoothing_level=0.9, optimized=False)
        
        # Gerar previsões
        forecast = fit.forecast(forecast_days)
        
        # Criar um DataFrame para esta coluna de previsão
        forecast_df = pd.DataFrame({
            col: forecast.values
        }, index=pd.to_datetime(forecast.index))
        
        predictions_dfs.append(forecast_df)
    
    # Combinar todas as previsões em um único DataFrame
    df_forecast = pd.concat(predictions_dfs, axis=1)
    
    # Combinar os dados históricos com as previsões
    df_combined = pd.concat([df_diario, df_forecast])
    return df_combined.sort_index()

def plot_serie_diaria(df_diario_original, df_diario_forecasted):
    """Função para criar o gráfico de Série Diária com Médias Móveis e Previsão."""
    fig = go.Figure()
    # Definindo um esquema de cores mais suave ou com um "gradiente" percebido
    # Usaremos tons que se complementam e são facilmente distinguíveis
    cores = {
        'Hidráulica': '#4c78a8',  # Azul médio
        'Térmica': '#e45756',     # Vermelho telha
        'Eólica': '#54a24b',      # Verde mais escuro
        'Solar': '#f89e47'        # Laranja vibrante
    }
    
    # Adicionar os dados históricos (linhas mais finas, opacas)
    for fonte in df_diario_original.columns:
        media_30d = df_diario_original[fonte].rolling(window=30).mean()
        media_90d = df_diario_original[fonte].rolling(window=90).mean()
        
        fig.add_trace(go.Scatter(x=df_diario_original.index, y=df_diario_original[fonte], mode='lines', 
                                 name=f'{fonte} (Histórico)', legendgroup=fonte, line=dict(width=1), opacity=0.5, marker_color=cores[fonte]))
        fig.add_trace(go.Scatter(x=df_diario_original.index, y=media_30d, mode='lines', 
                                 name=f'{fonte} Média 30d (Histórico)', legendgroup=fonte, line=dict(width=2), marker_color=cores[fonte], showlegend=False))
        fig.add_trace(go.Scatter(x=df_diario_original.index, y=media_90d, mode='lines', 
                                 name=f'{fonte} Média 90d (Histórico)', legendgroup=fonte, line=dict(width=2, dash='dash'), marker_color=cores[fonte], showlegend=False))

    # Adicionar as previsões (linhas tracejadas, mais grossas)
    for fonte in df_diario_forecasted.columns:
        if fonte in df_diario_original.columns: # Apenas para as colunas que foram previstas
            # A previsão SES já é uma série suavizada estendida
            # Plotamos a série completa do df_diario_forecasted para mostrar a continuação
            fig.add_trace(go.Scatter(x=df_diario_forecasted.index, y=df_diario_forecasted[fonte], mode='lines', 
                                     name=f'{fonte} (Previsão SES)', legendgroup=fonte, 
                                     line=dict(width=3, dash='dot'), marker_color=cores[fonte]))

    fig.update_layout(height=700, title_text='<b>Geração Diária com Tendências de Médias Móveis e Previsões (SES)</b>', 
                      legend_title='<b>Fonte e Tendência</b>', xaxis_rangeslider_visible=True)
    return fig


# --- Interface Principal ---
def main():
    st.title("📊 Análise Completa da Matriz Energética Brasileira com Previsões")
    st.markdown("Um projeto desenvolvido com **Gemini** para o estudo aprofundado da geração de energia no Brasil, com foco em previsões e alinhamento com os ODS da ONU.")

    if not os.path.exists(CONSOLIDATED_FILE):
        st.error(f"ERRO: Arquivo de dados mestre '{CONSOLIDATED_FILE}' não encontrado!")
        st.warning("Por favor, execute o script `python coletar_dados.py` em seu terminal para gerar a base de dados.")
        st.stop()

    # Carrega e prepara todos os dados de uma vez
    # Agora 'df_original' recebe o dataframe completo antes de qualquer processamento
    df_original, analise_anual, analise_regional, df_diario = load_and_prepare_all_data()

    # --- Metadados da Base de Dados para Enaltecer o Trabalho ---
    st.header("🔍 Visão Geral da Base de Dados (Nosso Esforço em Números!)")
    st.markdown("""
    Este projeto foi construído sobre uma base de dados robusta e detalhada, que exigiu um processo meticuloso de coleta, tratamento e engenharia de features.
    A complexidade dos dados multi-fonte e multi-temporal foi um desafio superado para garantir a precisão das análises e previsões.
    """)

    # Calcula o tamanho do arquivo
    file_size_bytes = os.path.getsize(CONSOLIDATED_FILE)
    file_size_mb = file_size_bytes / (1024 * 1024)

    # Informações sobre o DataFrame original
    num_linhas, num_colunas = df_original.shape
    periodo_inicio = df_original['din_instante'].min().strftime('%d/%m/%Y')
    periodo_fim = df_original['din_instante'].max().strftime('%d/%m/%Y')
    num_subsistemas = df_original['nom_subsistema'].nunique()
    cols_originais = ", ".join(df_original.columns.tolist())

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.metric(label="Tamanho do Arquivo da Base de Dados", value=f"{file_size_mb:.2f} MB")
        st.metric(label="Número Total de Registros", value=f"{num_linhas:,}".replace(",", "."))
        st.metric(label="Número de Subsistemas Analisados", value=f"{num_subsistemas}")
    with col_meta2:
        st.metric(label="Número de Colunas Originais", value=f"{num_colunas}")
        st.metric(label="Período Abrangido (Início)", value=periodo_inicio)
        st.metric(label="Período Abrangido (Fim)", value=periodo_fim)
    
    st.markdown(f"""
    <p>A base contém dados de <b>{num_subsistemas} subsistemas energéticos</b>, abrangendo um período vasto de <b>{pd.to_datetime(periodo_fim).year - pd.to_datetime(periodo_inicio).year + 1} anos</b>.
    Cada linha representa um registro horário de geração em diferentes fontes (hidráulica, térmica, eólica, solar) e subsistemas.
    As colunas principais incluem: `{cols_originais}`.
    </p>
    <p>O processamento envolveu a limpeza de dados, a agregação para diferentes granularidades (anual e diária) e a criação de novas features, como a porcentagem de energias renováveis e as taxas de crescimento, elementos cruciais para as análises avançadas e modelos preditivos apresentados.</p>
    """, unsafe_allow_html=True)
    st.markdown("---") # Separador visual


    # --- FILTRO: Excluir o ano de 2025 dos dados anuais antes de gerar as previsões ---
    # Isso garante que a base histórica para os modelos de regressão linear não inclua 2025.
    analise_anual = analise_anual[analise_anual['ano'] != 2025]

    # --- Geração das Previsões ---
    forecast_until_year = 2030
    # O current_year agora reflete o último ano *após* a exclusão de 2025, se aplicável
    current_year = analise_anual['ano'].max() 

    analise_anual_eolica_lr_combined, pred_eolica_lr, coef_eolica, intercept_eolica = predict_linear_regression(
        analise_anual, 'perc_eolica', current_year, forecast_until_year
    )
    analise_anual_solar_lr_combined, pred_solar_lr, coef_solar, intercept_solar = predict_linear_regression(
        analise_anual, 'perc_solar', current_year, forecast_until_year
    )
    analise_anual_renovavel_lr_combined, pred_renovavel_lr, coef_renovavel, intercept_renovavel = predict_linear_regression(
        analise_anual, 'perc_renovavel_total', current_year, forecast_until_year
    )

    # Previsões de SES para dados diários (Ex: 2 anos de previsão)
    forecast_days = 365 * (forecast_until_year - df_diario.index.max().year) # Prever até 2030 com base no último ano do df_diario
    df_diario_ses_combined = predict_ses_for_daily_data(df_diario, forecast_days)


    st.sidebar.title("Navegação")
    page = st.sidebar.radio("Selecione a Análise:",
        ["Visão Geral e ODS 7", "Análise de Crescimento", "Análise Regional", "Análise de Série Temporal", "Previsões e Conceitos"]
    )

    if page == "Visão Geral e ODS 7":
        st.header("Composição da Matriz e Participação Renovável")
        st.markdown("""
        Esta seção apresenta a composição da matriz energética do Sistema Interligado Nacional (SIN) e a participação percentual das fontes renováveis, alinhando-se com a meta **ODS 7.2** de manter elevada essa participação.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Geração por Fonte (MWMED)")
            # Gráfico de Geração por Fonte - Sem previsão para manter a didática
            fig_matriz = px.bar(analise_anual, x='ano', y=['Hidraulica', 'Termica', 'Eolica', 'Solar'])
            fig_matriz.update_layout(barmode='stack', xaxis_title='Ano', yaxis_title='Geração Somada (MWMED)')
            st.plotly_chart(fig_matriz, use_container_width=True)
        with col2:
            st.subheader("Participação Renovável Total (%) com Previsão até 2030")
            # Gráfico de Participação Renovável Total com Previsão
            fig_perc = px.line(analise_anual_renovavel_lr_combined, x='ano', y='perc_renovavel_total', 
                               title='Participação Renovável Total Histórica e Previsão (2030)',
                               markers=True, line_dash_map={'perc_renovavel_total': 'solid'})
            
            # Destacar a previsão
            fig_perc.add_trace(go.Scatter(x=pred_renovavel_lr['ano'], y=pred_renovavel_lr['perc_renovavel_total'],
                                           mode='lines+markers', name='Previsão (Regressão Linear)', 
                                           line=dict(dash='dot', color='red', width=3)))
            
            fig_perc.update_layout(xaxis_title='Ano', yaxis_title='% Renovável', showlegend=True)
            st.plotly_chart(fig_perc, use_container_width=True)

    elif page == "Análise de Crescimento":
        st.header("Taxa de Crescimento Anual da Participação (Eólica e Solar)")
        st.markdown("""
        **Motivação:** Medir a **velocidade** da expansão das fontes Eólica e Solar. Altas taxas de crescimento indicam um forte momento de investimento e adoção tecnológica, cruciais para a diversificação da matriz.
        """)
        st.info("🎯 **Alinhamento: ODS 7.2** (Aumentar substancialmente a participação) e **ODS 7.a** (Promover investimento).")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Participação Eólica (%) com Previsão até 2030")
            fig_eolica = px.line(analise_anual_eolica_lr_combined, x='ano', y='perc_eolica', 
                                 title='Participação Eólica Histórica e Previsão (2030)',
                                 markers=True, line_dash_map={'perc_eolica': 'solid'})
            
            fig_eolica.add_trace(go.Scatter(x=pred_eolica_lr['ano'], y=pred_eolica_lr['perc_eolica'],
                                             mode='lines+markers', name='Previsão (Regressão Linear)', 
                                             line=dict(dash='dot', color='red', width=3)))
            
            st.plotly_chart(fig_eolica, use_container_width=True)
        with col2:
            st.subheader("Participação Solar (%) com Previsão até 2030")
            fig_solar = px.line(analise_anual_solar_lr_combined, x='ano', y='perc_solar', 
                                 title='Participação Solar Histórica e Previsão (2030)',
                                 markers=True, line_dash_map={'perc_solar': 'solid'})
            
            fig_solar.add_trace(go.Scatter(x=pred_solar_lr['ano'], y=pred_solar_lr['perc_solar'],
                                             mode='lines+markers', name='Previsão (Regressão Linear)', 
                                             line=dict(dash='dot', color='red', width=3)))
            
            st.plotly_chart(fig_solar, use_container_width=True)

    elif page == "Análise Regional":
        st.header("Análise dos Subsistemas Energéticos")
        st.markdown("Análise da contribuição de geração renovável de cada região em duas perspectivas: **relativa** (interna) e **absoluta** (em relação ao total do Brasil).")
        st.info("🎯 **Alinhamento: ODS 7.1** (Acesso universal) e **ODS 7.b** (Infraestrutura).")

        st.subheader("Contribuição Absoluta para o Total Renovável do Brasil (MWMED)")
        # Manter este gráfico sem previsão para não sobrecarregar
        fig_abs = px.bar(analise_regional, x='ano', y='geracao_renovavel_regiao', color='nom_subsistema', title='Geração Renovável por Subsistema')
        fig_abs.update_layout(barmode='stack', xaxis_title='Ano', yaxis_title='Geração Renovável (MWMED)')
        st.plotly_chart(fig_abs, use_container_width=True)
        
    elif page == "Análise de Série Temporal":
        st.header("Análise de Tendências Diárias com Médias Móveis e Previsões")
        with st.expander("Clique aqui para entender as Médias Móveis, Suavização Exponencial Simples (SES) e como interpretar o gráfico"):
            st.markdown("""
            As **Médias Móveis (MM)** filtram a volatilidade diária ("ruído") para revelar a tendência real ("sinal").
            - **MM de 30 dias (Linha Sólida):** Mostra a tendência mensal, ideal para ver a "dança" sazonal entre Hidráulica e Térmica.
            - **MM de 90 dias (Linha Tracejada):** Mostra a tendência de longo prazo, confirmando a ascensão estrutural da Eólica e Solar.
            
            A **Suavização Exponencial Simples (SES)** é usada para gerar as previsões futuras (linhas pontilhadas). Ela atribui maior peso às observações mais recentes, tornando a previsão mais sensível às mudanças recentes na série.
            """)
        # Usar a função plot_serie_diaria que agora aceita o df original e o df com previsões
        fig_diario_pred = plot_serie_diaria(df_diario, df_diario_ses_combined) 
        st.plotly_chart(fig_diario_pred, use_container_width=True)

    elif page == "Previsões e Conceitos":
        st.header("Detalhes das Previsões e Conceitos Estatísticos")
        st.markdown("""
        Aqui você encontra os detalhes sobre os modelos estatísticos utilizados para as previsões e os resultados gerados, alinhados com os **Objetivos de Desenvolvimento Sustentável (ODS) da ONU**, especialmente o ODS 7.
        """)

        st.subheader("1. Previsões de Crescimento (Eólica, Solar e Participação Renovável Total)")
        st.markdown("""
        Para prever o crescimento da participação da energia eólica e solar, bem como a participação renovável total na matriz, utilizamos a **Regressão Linear Simples**. Este método é fundamental na estatística para identificar e modelar a relação linear entre duas variáveis.
        """)
        st.markdown("### Conceitos de Regressão Linear Simples:")
        st.markdown(r"""
        A Regressão Linear Simples busca ajustar uma linha reta aos seus dados. Esta linha é o 'melhor ajuste' no sentido de que minimiza a soma dos quadrados das distâncias verticais de cada ponto à linha. Essa técnica é conhecida como **Método dos Mínimos Quadrados**.

        A equação da linha de regressão é expressa como: $Y = \beta_0 + \beta_1 X$.
        * **$Y$ (Variável Dependente):** É a variável que queremos prever (e.g., % de participação eólica).
        * **$X$ (Variável Independente):** É a variável que usamos para fazer a previsão (o ano).
        * **$\beta_0$ (Intercepto):** Representa o valor previsto de $Y$ quando $X$ é zero.
        * **$\beta_1$ (Coeficiente Angular / Inclinação):** Indica a mudança média esperada em $Y$ para cada aumento de uma unidade em $X$. Ele mostra o quanto a participação da fonte aumenta ou diminui a cada ano.
        
        A **extrapolação** (previsão fora do intervalo de dados observados) deve ser feita com cautela, pois a tendência pode não se manter linear indefinidamente.
        """)

        st.markdown(f"#### Participação Eólica (%):")
        st.markdown(f"- Coeficiente Angular ($\beta_1$): `{coef_eolica:.4f}` (Aumento médio de `{coef_eolica:.2f}%` na participação eólica por ano)")
        st.markdown(f"- Intercepto ($\beta_0$): `{intercept_eolica:.4f}`")
        st.dataframe(pred_eolica_lr.head())

        st.markdown(f"#### Participação Solar (%):")
        st.markdown(f"- Coeficiente Angular ($\beta_1$): `{coef_solar:.4f}` (Aumento médio de `{coef_solar:.2f}%` na participação solar por ano)")
        st.markdown(f"- Intercepto ($\beta_0$): `{intercept_solar:.4f}`")
        st.dataframe(pred_solar_lr.head())

        st.markdown(f"#### Participação Renovável Total (%):")
        st.markdown(f"- Coeficiente Angular ($\beta_1$): `{coef_renovavel:.4f}` (Aumento médio de `{coef_renovavel:.2f}%` na participação renovável total por ano)")
        st.markdown(f"- Intercepto ($\beta_0$): `{intercept_renovavel:.4f}`")
        st.dataframe(pred_renovavel_lr.head())

        st.subheader("2. Previsões para a Variação Diária (Suavização Exponencial Simples - SES)")
        st.markdown("""
        Para as tendências diárias, onde a complexidade pode ser maior, além das Médias Móveis já utilizadas em seu painel, apresentamos a **Suavização Exponencial Simples (SES)** para previsões.
        """)
        st.markdown("### Conceitos de Suavização Exponencial Simples (SES):")
        st.markdown(r"""
        A SES atribui **pesos decrescentes exponencialmente** às observações mais antigas, fazendo com que dados mais recentes tenham um impacto maior na previsão.

        O conceito chave é o **Parâmetro de Suavização ($\alpha$)**, que varia entre 0 e 1.
        * **$\alpha$ Próximo de 1:** Dá muito peso às observações mais recentes. A série suavizada se ajusta rapidamente a novas mudanças.
        * **$\alpha$ Próximo de 0:** Dá mais peso às observações mais antigas. A série suavizada é mais 'lisa', menos sensível a flutuações recentes.
        
        A previsão para o próximo período é o valor suavizado do período atual.
        """)
        st.markdown("Previsões Diárias (últimas 5 previsões para cada fonte):")
        st.dataframe(df_diario_ses_combined.tail(5))
        st.markdown("*(Os valores 'NaN' nas colunas originais indicam os dados históricos para os quais a previsão é gerada. Os valores preenchidos nas colunas de previsão indicam os valores projetados.)*")


        st.subheader("3. Alinhamento com os Objetivos de Desenvolvimento Sustentável (ODS 7)")
        st.markdown("""
        As previsões que realizamos contribuem diretamente para o monitoramento e planejamento relacionados ao **ODS 7: Energia Limpa e Acessível**. Especificamente, elas se alinham com a meta:
        * **ODS 7.2: "Até 2030, aumentar substancialmente a participação de energias renováveis na matriz energética global"**.
        
        Ao projetar o crescimento da participação da energia eólica e solar, e o percentual total de energias renováveis, podemos avaliar se as tendências atuais são suficientes para atingir um aumento "substancial" até 2030. As previsões diárias, por sua vez, auxiliam no planejamento operacional e na otimização da infraestrutura para integrar cada vez mais essas fontes.
        """)


if __name__ == "__main__":
    main()