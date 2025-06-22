import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.tsa.api as smt
import json # BIBLIOTECA: 'Cozinheiro' de dados, prepara infos pra 'viagem'
from datetime import datetime # BIBLIOTECA: 'Relogio' e 'Calendario' pra registrar o tempo

# --- Constantes e Configuracao ---
# ENDEREÇO: Onde seu 'documento' principal está guardado.
# Se o arquivo não estiver lá, o app te avisa!
CONSOLIDATED_FILE = "balanco_energia_consolidado.parquet" 
st.set_page_config(layout="wide", page_title="Análise Energética do Brasil com Previsões", page_icon="🇧🇷")

# --- Módulo de Preparação de Dados (em cache) ---
@st.cache_data # MEMÓRIA TURBO: Guarda resultados pra não ter que refazer, tipo um atalho!
def load_and_prepare_all_data():
    """
    Carrega o 'Livro Mestre' (base de dados) e organiza TODOS os 'Cadernos de Análise'
    (dataframes) necessários. Retorna o 'Livro' original e os 'Cadernos' prontos.
    """
    if not os.path.exists(CONSOLIDATED_FILE):
        st.error(f"ERRO: Seu 'documento' mestre '{CONSOLIDATED_FILE}' sumiu!")
        st.warning("Por favor, verifique se o arquivo 'balanco_energia_consolidado.parquet' está na mesma 'gaveta' (pasta) do seu 'aplicativo' (script).")
        st.stop()

    df = pd.read_parquet(CONSOLIDATED_FILE) # Lendo o 'documento' em formato .parquet, que é super rápido!
    df['din_instante'] = pd.to_datetime(df['din_instante']) # CONVERSÃO: Data de texto para 'calendário de verdade'
    df['ano'] = df['din_instante'].dt.year # EXTRAÇÃO: Tirando só o 'Ano de safra' da data

    # CONVERSÃO: Colunas de geração viram números ANTES dos cálculos
    numeric_cols = ['val_gerhidraulica', 'val_gertermica', 'val_gereolica', 'val_gersolar']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') # TRATAMENTO: Tenta virar número, se não der, vira 'vazio'

    # DADOS SIN: Só o 'Brasilzão' da energia (Sistema Interligado Nacional)
    df_sin = df[df['nom_subsistema'] == 'SISTEMA INTERLIGADO NACIONAL'].copy()
    
    # FONTES RENOVÁVEIS: As 'fazendas de energia' (inclui água, vento, sol)
    fontes_renovaveis_total = ['val_gerhidraulica', 'val_gereolica', 'val_gersolar']
    df_sin['geracao_renovavel_total'] = df_sin[fontes_renovaveis_total].sum(axis=1) # SOMA: Produção total da 'fazenda verde'

    # NOVO: 'FAZENDAS DE ENERGIA NOVA': Só vento e sol (exclui água)
    fontes_novas_renovaveis = ['val_gereolica', 'val_gersolar']
    df_sin['geracao_novas_renovaveis'] = df_sin[fontes_novas_renovaveis].sum(axis=1) # SOMA: Produção das 'novas fazendas'

    df_sin['geracao_total'] = df_sin['geracao_renovavel_total'] + df_sin['val_gertermica'] # TOTAL: 'Fazenda verde' + 'Termoelétrica'

    # ANÁLISE ANUAL: 'Fechamento de contas' por ano
    analise_anual = df_sin.groupby('ano').agg(
        Hidraulica=('val_gerhidraulica', 'sum'), Termica=('val_gertermica', 'sum'),
        Eolica=('val_gereolica', 'sum'), Solar=('val_gersolar', 'sum'),
        total_renovavel=('geracao_renovavel_total', 'sum'), # Soma com água
        total_novas_renovaveis=('geracao_novas_renovaveis', 'sum'), # SÓ novas
        total_geral=('geracao_total', 'sum')
    ).reset_index() # ORGANIZAÇÃO: Devolve o ano como coluna

    # TRATAMENTO: Zera valores 'vazios' (NaN)
    analise_anual['total_hidraulica'] = analise_anual['Hidraulica'].fillna(0)
    analise_anual['total_termica'] = analise_anual['Termica'].fillna(0)
    analise_anual['total_eolica'] = analise_anual['Eolica'].fillna(0)
    analise_anual['total_solar'] = analise_anual['Solar'].fillna(0)
    
    # CÁLCULO: Geração total anual de todas as fontes
    analise_anual['geracao_total_anual'] = analise_anual['total_hidraulica'] + analise_anual['total_termica'] + \
                                            analise_anual['total_eolica'] + analise_anual['total_solar']

    total_geral_anual_safe = analise_anual['geracao_total_anual'].replace(0, np.nan) # PREVENÇÃO: Evita divisão por zero

    # CÁLCULOS: Porcentagem de participação no 'bolo' total
    analise_anual['perc_renovavel_total'] = (analise_anual['total_renovavel'] / total_geral_anual_safe) * 100
    analise_anual['perc_eolica'] = (analise_anual['Eolica'] / total_geral_anual_safe) * 100
    analise_anual['perc_solar'] = (analise_anual['Solar'] / total_geral_anual_safe) * 100
    analise_anual['perc_hidraulica'] = (analise_anual['Hidraulica'] / total_geral_anual_safe) * 100
    analise_anual['perc_termica'] = (analise_anual['Termica'] / total_geral_anual_safe) * 100
    
    analise_anual['perc_novas_renovaveis'] = (analise_anual['total_novas_renovaveis'] / total_geral_anual_safe) * 100 # NOVO: % Novas Renovaveis

    for col_perc in ['perc_renovavel_total', 'perc_eolica', 'perc_solar', 'perc_hidraulica', 'perc_termica', 'perc_novas_renovaveis']:
        analise_anual[col_perc] = analise_anual[col_perc].fillna(0) # TRATAMENTO: Zera % 'vazias'

    # CÁLCULOS: Crescimento percentual ano a ano
    analise_anual['crescimento_eolica'] = analise_anual['perc_eolica'].pct_change() * 100
    analise_anual['crescimento_solar'] = analise_anual['perc_solar'].pct_change() * 100
    analise_anual['crescimento_renovavel_total'] = analise_anual['perc_renovavel_total'].pct_change() * 100
    analise_anual['crescimento_novas_renovaveis'] = analise_anual['perc_novas_renovaveis'].pct_change() * 100 # NOVO: Crescimento Novas Renovaveis

    # DADOS REGIONAIS: O que acontece fora do 'Brasilzão' (exclui SIN)
    df_regional = df[df['nom_subsistema'] != 'SISTEMA INTERLIGADO NACIONAL'].copy()
    df_regional['geracao_renovavel'] = df_regional[fontes_renovaveis_total].sum(axis=1) # Produção 'verde' regional
    df_regional['geracao_total'] = df_regional['geracao_renovavel'] + df_regional['val_gertermica'] # Total regional
    
    # ANÁLISE REGIONAL ANUAL: Fechamento de contas por ano e por região
    analise_regional_anual = df_regional.groupby(['ano', 'nom_subsistema']).agg(
        geracao_renovavel_regiao=('geracao_renovavel', 'sum'),
        geracao_total_regiao=('geracao_total', 'sum')
    ).reset_index()
    analise_regional_anual = pd.merge(analise_regional_anual, analise_anual[['ano', 'total_renovavel']], on='ano', suffixes=('', '_brasil')) # JUNÇÃO: Adiciona total do Brasil
    
    # DADOS DIÁRIOS: Produção 'dia a dia' do SIN
    df_diario = df_sin.set_index('din_instante')[['val_gerhidraulica', 'val_gertermica', 'val_gereolica', 'val_gersolar']].resample('D').sum()
    df_diario.columns = ['Hidraulica', 'Termica', 'Eolica', 'Solar'] # RENOMEAR: Colunas com nomes 'amigáveis'
    
    return df, analise_anual, analise_regional_anual, df_diario

# --- Funções de Previsão ---
def predict_linear_regression(df, target_column, current_year, forecast_until_year):
    # REGRESSÃO LINEAR (RL): Tipo uma 'régua' pra ver a tendência futura
    model = LinearRegression() # MODELO: A 'régua' que vamos usar
    df_history = df[df['ano'] <= current_year] # HISTÓRICO: Dados que a 'régua' vai aprender
    X = df_history['ano'].values.reshape(-1, 1) # ENTRADA: O 'Ano' para a 'régua'
    y = df_history[target_column].values # SAÍDA: O que queremos prever (ex: % Eólica)
    model.fit(X, y) # APRENDIZADO: A 'régua' aprende com o passado
    future_years = np.arange(current_year + 1, forecast_until_year + 1).reshape(-1, 1) # FUTURO: Os 'Anos' que queremos prever
    predictions = model.predict(future_years) # PREVISÃO: Onde a 'régua' aponta no futuro
    df_predictions = pd.DataFrame({
        'ano': future_years.flatten(),
        target_column: predictions
    })
    df_predictions[target_column] = df_predictions[target_column].apply(lambda x: max(0, x)) # GARANTIA: % não pode ser negativa
    df_combined = pd.concat([df_history[['ano', target_column]], df_predictions], ignore_index=True) # JUNTAR: Histórico + Previsão
    return df_combined, df_predictions, model.coef_[0], model.intercept_

def predict_ses_for_daily_data(df_diario, forecast_days):
    # SUAVIZAÇÃO EXPONENCIAL SIMPLES (SES): 'Previsão de tempo' para o dia a dia
    predictions_dfs = []
    for col in df_diario.columns:
        series = df_diario[col].dropna() # SÉRIE: Pega os dados de cada fonte
        if series.empty: # VERIFICAÇÃO: Se não tem dados, pula
            continue
        try:
            fit = smt.SimpleExpSmoothing(series, initialization_method="estimated").fit(smoothing_level=0.9, optimized=False) # AJUSTE: O 'modelo' SES aprende com a série
            forecast = fit.forecast(forecast_days) # PROJEÇÃO: Gera a 'previsão de tempo'
        except Exception as e:
            st.warning(f"Não foi possível gerar previsão SES para {col}: {e}. Retornando 'nada'.") # ERRO: Avisa se não der
            forecast = pd.Series(np.nan, index=pd.date_range(start=series.index.max() + pd.Timedelta(days=1), periods=forecast_days)) # RETORNO: Retorna 'vazio' se falhar
        forecast_df = pd.DataFrame({
            col: forecast.values
        }, index=pd.to_datetime(forecast.index))
        predictions_dfs.append(forecast_df)
    if predictions_dfs:
        df_forecast = pd.concat(predictions_dfs, axis=1) # JUNTAR: Todas as previsões diárias
    else:
        df_forecast = pd.DataFrame() # VAZIO: Se não tiver previsão
    df_combined = pd.concat([df_diario, df_forecast]) # JUNTAR: Histórico Diário + Previsão
    return df_combined.sort_index() # ORGANIZA: Por data

def plot_serie_diaria(df_diario_original, df_diario_forecasted):
    """
    Função para criar o 'Boletim do Tempo' da energia: Gráfico de Série Diária com Medias Móveis e Previsão.
    Dividido em 'andares' para melhor visualização.
    """
    cores = { # PALETA: Cores para cada fonte
        'Hidraulica': '#4c78a8',
        'Termica': '#e45756',
        'Eolica': '#54a24b',
        'Solar': '#f89e47'
    }

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        row_titles=['Geração Hidráulica e Térmica', 'Geração Eólica e Solar']) # SUB-GRÁFICOS: Dois 'andares' no gráfico

    fontes_grandes = ['Hidraulica', 'Termica'] # FONTES: As 'grandes' da matriz
    for fonte in fontes_grandes:
        if fonte in df_diario_original.columns:
            media_30d = df_diario_original[fonte].rolling(window=30).mean() # MM 30d: Média dos últimos 30 dias ('tendência do mês')
            media_90d = df_diario_original[fonte].rolling(window=90).mean() # MM 90d: Média dos últimos 90 dias ('tendência do trimestre')
            
            fig.add_trace(go.Scatter(x=df_diario_original.index, y=df_diario_original[fonte], mode='lines', 
                                     name=f'{fonte} (Geração Histórica)', legendgroup=fonte, 
                                     line=dict(width=0), 
                                     fill='tozeroy', fillcolor=f'rgba({int(cores[fonte][1:3], 16)}, {int(cores[fonte][3:5], 16)}, {int(cores[fonte][5:7], 16)}, 0.1)'),
                            row=1, col=1) # ADICIONA: Geração histórica (preenchida)
            
            fig.add_trace(go.Scatter(x=df_diario_original.index, y=media_30d, mode='lines', 
                                     name=f'{fonte} Média 30d', legendgroup=fonte, 
                                     line=dict(width=2, color=cores[fonte]), showlegend=True), 
                            row=1, col=1) # ADICIONA: Média Móvel 30 dias
            
            fig.add_trace(go.Scatter(x=df_diario_original.index, y=media_90d, mode='lines', 
                                     name=f'{fonte} Média 90d', legendgroup=fonte, 
                                     line=dict(width=2, dash='dash', color=cores[fonte]), showlegend=True), 
                            row=1, col=1) # ADICIONA: Média Móvel 90 dias
            
            if fonte in df_diario_forecasted.columns and not df_diario_forecasted[fonte].isnull().all():
                fig.add_trace(go.Scatter(x=df_diario_forecasted.index, y=df_diario_forecasted[fonte], mode='lines', 
                                         name=f'{fonte} (Previsão SES)', legendgroup=fonte, 
                                         line=dict(width=3, dash='dot', color=cores[fonte]), showlegend=True),
                                 row=1, col=1) # ADICIONA: Previsão SES

    fontes_menores = ['Eolica', 'Solar'] # FONTES: As 'emergentes' da matriz
    for fonte in fontes_menores:
        if fonte in df_diario_original.columns:
            media_30d = df_diario_original[fonte].rolling(window=30).mean()
            media_90d = df_diario_original[fonte].rolling(window=90).mean()
            
            fig.add_trace(go.Scatter(x=df_diario_original.index, y=df_diario_original[fonte], mode='lines', 
                                     name=f'{fonte} (Geração Histórica)', legendgroup=fonte, 
                                     line=dict(width=0), 
                                     fill='tozeroy', fillcolor=f'rgba({int(cores[fonte][1:3], 16)}, {int(cores[fonte][3:5], 16)}, {int(cores[fonte][5:7], 16)}, 0.1)'),
                            row=2, col=1)
            
            fig.add_trace(go.Scatter(x=df_diario_original.index, y=media_30d, mode='lines', 
                                     name=f'{fonte} Média 30d', legendgroup=fonte, 
                                     line=dict(width=2, color=cores[fonte]), showlegend=True), 
                            row=2, col=1)
            
            fig.add_trace(go.Scatter(x=df_diario_original.index, y=media_90d, mode='lines', 
                                     name=f'{fonte} Média 90d', legendgroup=fonte, 
                                     line=dict(width=2, dash='dash', color=cores[fonte]), showlegend=True), 
                            row=2, col=1)
            
            if fonte in df_diario_forecasted.columns and not df_diario_forecasted[fonte].isnull().all():
                fig.add_trace(go.Scatter(x=df_diario_forecasted.index, y=df_diario_forecasted[fonte], mode='lines', 
                                         name=f'{fonte} (Previsão SES)', legendgroup=fonte, 
                                         line=dict(width=3, dash='dot', color=cores[fonte]), showlegend=True),
                                 row=2, col=1)

    fig.update_layout(height=800, title_text='<b>Boletim do Tempo da Energia: Geração Diária com Tendências e Previsões (SES) por Fonte</b>', 
                      legend_title='<b>Fonte e Tendência</b>', xaxis_rangeslider_visible=True,
                      hovermode="x unified") # LAYOUT: Título, legenda e slider de zoom
    
    # --- EXPLICAÇÃO DA ESCALA PARA LEIGOS ---
    fig.update_yaxes(title_text='Geração (MWmed) <br><sub>(Produção em Gigawatts Médios, tipo a "força" das usinas)</sub>', row=1, col=1)
    fig.update_yaxes(title_text='Geração (MWmed) <br><sub>(Produção em Gigawatts Médios, tipo a "força" das usinas)</sub>', row=2, col=1)

    return fig

def plot_participacao_anual_fontes(df_anual):
    """
    Desenha o 'Bolo da Energia' (gráfico de área empilhada) com a participação percentual anual das fontes.
    """
    df_plot = df_anual[['ano', 'perc_hidraulica', 'perc_termica', 'perc_eolica', 'perc_solar']].copy() # DADOS: Pega o ano e as porcentagens
    df_plot_melted = df_plot.melt(id_vars=['ano'], var_name='Fonte', value_name='Participacao (%)') # PREPARAÇÃO: 'Empilha' os dados
    
    df_plot_melted['Fonte'] = df_plot_melted['Fonte'].replace({ # RENOMEIA: Nomes 'amigáveis' para as fontes
        'perc_hidraulica': 'Hidráulica',
        'perc_termica': 'Térmica',
        'perc_eolica': 'Eólica',
        'perc_solar': 'Solar'
    })

    fig = px.area(df_plot_melted, 
                  x='ano', 
                  y='Participacao (%)', 
                  color='Fonte',
                  title='<b>Composição do Bolo da Energia: Participação Anual das Fontes na Matriz Elétrica Brasileira</b>',
                  labels={'Participacao (%)': 'Fatia do Bolo (%)', 'ano': 'Ano'},
                  hover_name='Fonte',
                  hover_data={'Participacao (%)': ':.2f'}
                 ) # GRÁFICO: Desenha o bolo!
    fig.update_layout(hovermode="x unified", yaxis_range=[0, 100]) # LAYOUT: Zoom no bolo de 0 a 100%
    return fig

def plot_crescimento_eolica(df_anual):
    # GRÁFICO: 'Termômetro' do crescimento da Eólica
    fig_eolica = go.Figure()
    fig_eolica.add_trace(go.Scatter(x=df_anual['ano'][1:], y=df_anual['crescimento_eolica'][1:],
                                  mode='lines+markers+text', name='Crescimento Eólica (%)',
                                  marker=dict(color='green'),
                                  text=[f'{c:.1f}%' for c in df_anual['crescimento_eolica'][1:]],
                                  textposition="top center"))
    fig_eolica.update_layout(
        title_text="<b>Termômetro do Vento: Crescimento Percentual Anual da Participação da Energia Eólica</b>",
        xaxis_title="Ano",
        yaxis_title="Crescimento Anual (%)", 
        xaxis=dict(tickmode='linear', dtick=1)
    )
    return fig_eolica

def plot_crescimento_solar(df_anual):
    # GRÁFICO: 'Termômetro' do crescimento da Solar
    fig_solar = go.Figure()
    fig_solar.add_trace(go.Scatter(x=df_anual['ano'][1:], y=df_anual['crescimento_solar'][1:],
                                  mode='lines+markers+text', name='Crescimento Solar (%)',
                                  marker=dict(color='orange'),
                                  text=[f'{c:.1f}%' for c in df_anual['crescimento_solar'][1:]],
                                  textposition="top center"))
    fig_solar.update_layout(
        title_text="<b>Termômetro do Sol: Crescimento Percentual Anual da Participação da Energia Solar</b>",
        xaxis_title="Ano",
        yaxis_title="Crescimento Anual (%)", 
        xaxis=dict(tickmode='linear', dtick=1)
    )
    return fig_solar

def plot_crescimento_renovavel_total(df_anual):
    # GRÁFICO: 'Termômetro' do crescimento de TODAS as renováveis
    fig_renovavel_total = go.Figure()
    fig_renovavel_total.add_trace(go.Scatter(x=df_anual['ano'][1:], y=df_anual['crescimento_renovavel_total'][1:],
                                   mode='lines+markers+text', name='Crescimento Renovável Total (%)',
                                   marker=dict(color='blue'),
                                   text=[f'{c:.1f}%' for c in df_anual['crescimento_renovavel_total'][1:]],
                                   textposition="top center"))
    fig_renovavel_total.update_layout(
        title_text="<b>Termômetro Verde: Crescimento Percentual Anual da Participação Total de Energias Renováveis</b>",
        xaxis_title="Ano",
        yaxis_title="Crescimento Anual (%)", 
        xaxis=dict(tickmode='linear', dtick=1)
    )
    return fig_renovavel_total

def plot_pizza_participacao_2024(df_anual):
    # GRÁFICO DE PIZZA: 'Fatias' da energia em 2024
    if 2024 in df_anual['ano'].values:
        df_2024 = df_anual[df_anual['ano'] == 2024].iloc[0] # Pega os dados de 2024
        participacao_renovavel = float(df_2024['perc_renovavel_total']) # Fatia 'verde'
        participacao_nao_renovavel = 100 - participacao_renovavel # Fatia 'não verde'
        fig_pizza = go.Figure(data=[go.Pie(labels=['Renováveis', 'Não Renováveis'],
                                           values=[participacao_renovavel, participacao_nao_renovavel],
                                           textinfo='percent',
                                           insidetextorientation='radial'
                                           )])
        fig_pizza.update_layout(title_text='<b>O Bolo Energético de 2024: Participação das Energias Renováveis vs. Não Renováveis</b>')
        return fig_pizza
    else:
        st.warning("Dados para o ano de 2024 não encontrados para o 'gráfico de pizza'.")
        return go.Figure()

def display_descriptive_stats(df, columns, title):
    # TABELA: Resumo das 'Médias e Desvios' dos dados
    st.subheader(f"Estatísticas Descritivas: {title}")
    stats_data = {}
    
    for col in columns:
        if col in df.columns:
            series = df[col].dropna()
            if not series.empty:
                stats_data[col] = {
                    'Média': float(series.mean()), # 'Nota' média
                    'Mediana': float(series.median()), # 'Nota' do meio
                    'Desvio Padrão': float(series.std()), # 'Espalhamento' das notas
                    'Variância': float(series.var()), # 'Bagunça' das notas
                    'Moda': float(series.mode().iloc[0]) if not series.mode().empty else float(np.nan) # 'Nota' mais comum
                }
            else:
                stats_data[col] = {key: float(np.nan) for key in ['Média', 'Mediana', 'Desvio Padrão', 'Variância', 'Moda']}
        else:
            stats_data[col] = {key: float(np.nan) for key in ['Média', 'Mediana', 'Desvio Padrão', 'Variância', 'Moda']}

    stats_df = pd.DataFrame.from_dict(stats_data, orient='index').T
    st.dataframe(stats_df.round(2))

# --- Interface Principal (O 'Painel de Controle') ---
def main():
    st.title("📊 Raio-X da Energia Brasileira: Análise e Previsões")
    # Removida a linha: st.markdown("Um 'aplicativo' feito com Gemini para entender a fundo a produção de energia no Brasil, com foco no futuro e alinhamento com os ODS da ONU (Objetivos de Desenvolvimento Sustentável).")
    st.markdown("Um 'aplicativo' para entender a fundo a produção de energia no Brasil, com foco no futuro e alinhamento com os **ODS da ONU** (Objetivos de Desenvolvimento Sustentável).")

    df_original, analise_anual, analise_regional_anual, df_diario = load_and_prepare_all_data() # CARREGA: Todas as 'contas' prontas

    st.header("🔍 Olhar Geral da Base de Dados (Nosso 'Caminhão de Dados'!)")
    st.markdown("""
    Este 'aplicativo' foi construído em cima de um 'caminhão de dados' forte e detalhado, que exigiu um 'garimpo' cuidadoso, tratamento e 'invenção' de novas 'medidas'.
    A 'mistura' de dados de várias fontes e tempos foi um 'desafio superado' pra garantir que as análises e previsões fossem 'na mosca'.
    """)

    if os.path.exists(CONSOLIDATED_FILE):
        file_size_bytes = os.path.getsize(CONSOLIDATED_FILE)
        file_size_mb = file_size_bytes / (1024 * 1024)
    else:
        file_size_mb = 0

    num_linhas, num_colunas = df_original.shape # QUANTIDADE: Linhas e colunas do 'caminhão'
    periodo_inicio = df_original['din_instante'].min().strftime('%d/%m/%Y') # INÍCIO: 'Data de fundação' dos dados
    periodo_fim = df_original['din_instante'].max().strftime('%d/%m/%Y') # FIM: 'Última atualização' dos dados
    num_subsistemas = df_original['nom_subsistema'].nunique() # QUANTIDADE: Quantas 'regiões energéticas'
    cols_originais = ", ".join(df_original.columns.tolist()) # NOMES: Quais eram as 'etiquetas' originais das colunas

    col_meta1, col_meta2 = st.columns(2) # COLUNAS: Dividindo a tela
    with col_meta1:
        st.metric(label="Tamanho do Arquivo da Base de Dados (Peso do Caminhão)", value=f"{file_size_mb:.2f} MB")
        st.metric(label="Número Total de Registros (Quantos 'Tijolos' de Dados)", value=f"{num_linhas:,}".replace(",", "."))
        st.metric(label="Número de Subsistemas Analisados (Quantas 'Regiões' Olhamos)", value=f"{num_subsistemas}")
    with col_meta2:
        st.metric(label="Número de Colunas Originais (Quantas 'Etiquetas' Tinha)", value=f"{num_colunas}")
        st.metric(label="Período Abrangido (Início) (Quando Começou o 'Filme')", value=periodo_inicio)
        st.metric(label="Período Abrangido (Fim) (Quando Acabou o 'Filme')", value=periodo_fim)
    
    st.markdown(f"""
    <p>O 'caminhão' tem dados de <b>{num_subsistemas} 'regiões energéticas'</b>, cobrindo um período GIGANTE de <b>{pd.to_datetime(periodo_fim, format='%d/%m/%Y').year - pd.to_datetime(periodo_inicio, format='%d/%m/%Y').year + 1} anos</b>.
    Cada linha é um 'tijolo' de dado que mostra a produção de energia por hora em diferentes 'fazendas' (água, carvão, vento, sol) e 'regiões'.
    As 'etiquetas' principais (colunas) são: `{cols_originais}`.
    </p>
    <p>O 'trabalho de campo' envolveu 'limpar a sujeira' dos dados, 'juntar as contas' por ano e por dia, e 'criar novas medidas' (features), como a porcentagem de energias 'verdes' e a 'velocidade' do crescimento. Isso tudo é crucial para as análises mais 'avançadas' e as 'apostas' futuras que você verá.</p>
    """, unsafe_allow_html=True)
    st.markdown("---")


    # Lógica de filtragem de ano incompleto para PREVISAO e EXIBIÇÃO DE GRÁFICOS HISTÓRICOS
    periodo_fim_dt = pd.to_datetime(periodo_fim, format='%d/%m/%Y')
    
    if analise_anual['ano'].max() == periodo_fim_dt.year and periodo_fim_dt.month < 12:
        analise_anual_para_exibicao = analise_anual[analise_anual['ano'] < periodo_fim_dt.year].copy() # Exclui ano incompleto
    else:
        analise_anual_para_exibicao = analise_anual.copy() # Usa todos os anos

    current_year_for_prediction = analise_anual_para_exibicao['ano'].max() # Último ano completo para previsão
    forecast_until_year = 2030 # Ano alvo da previsão

    # Chamada das funções de previsão com Regressão Linear (RL)
    analise_anual_eolica_lr_combined, pred_eolica_lr, coef_eolica, intercept_eolica = predict_linear_regression(
        analise_anual_para_exibicao, 'perc_eolica', current_year_for_prediction, forecast_until_year
    )
    analise_anual_solar_lr_combined, pred_solar_lr, coef_solar, intercept_solar = predict_linear_regression(
        analise_anual_para_exibicao, 'perc_solar', current_year_for_prediction, forecast_until_year
    )
    analise_anual_renovavel_lr_combined, pred_renovavel_lr, coef_renovavel, intercept_renovavel = predict_linear_regression(
        analise_anual_para_exibicao, 'perc_renovavel_total', current_year_for_prediction, forecast_until_year
    )
    analise_anual_novas_renovaveis_lr_combined, pred_novas_renovaveis_lr, coef_novas_renovaveis, intercept_novas_renovaveis = predict_linear_regression(
        analise_anual_para_exibicao, 'perc_novas_renovaveis', current_year_for_prediction, forecast_until_year
    )
    
    analise_anual_hidraulica_lr_combined, pred_hidraulica_lr, coef_hidraulica, intercept_hidraulica = predict_linear_regression(
        analise_anual_para_exibicao, 'perc_hidraulica', current_year_for_prediction, forecast_until_year
    )

    # Previsão diária com SES
    last_daily_date = df_diario.index.max()
    target_end_date = pd.to_datetime(f'{forecast_until_year}-12-31')
    forecast_days = (target_end_date - last_daily_date).days
    if forecast_days < 0:
        forecast_days = 0 
    
    df_diario_ses_combined = predict_ses_for_daily_data(df_diario, forecast_days)

    # Preparar df_comparison
    if forecast_until_year in analise_anual_eolica_lr_combined['ano'].values:
        data_2030_eolica = analise_anual_eolica_lr_combined[analise_anual_eolica_lr_combined['ano'] == forecast_until_year]['perc_eolica'].iloc[0]
        data_2030_solar = analise_anual_solar_lr_combined[analise_anual_solar_lr_combined['ano'] == forecast_until_year]['perc_solar'].iloc[0]
        data_2030_hidraulica = analise_anual_hidraulica_lr_combined[analise_anual_hidraulica_lr_combined['ano'] == forecast_until_year]['perc_hidraulica'].iloc[0]
        data_2030_novas_renovaveis = analise_anual_novas_renovaveis_lr_combined[analise_anual_novas_renovaveis_lr_combined['ano'] == forecast_until_year]['perc_novas_renovaveis'].iloc[0]
    else:
        data_2030_eolica, data_2030_solar, data_2030_hidraulica, data_2030_novas_renovaveis = 0.0, 0.0, 0.0, 0.0
    data_2030_eolica_solar_combinada = data_2030_eolica + data_2030_solar

    base_year_for_comparison_data = current_year_for_prediction
    if base_year_for_comparison_data in analise_anual_para_exibicao['ano'].values:
        data_base_eolica = analise_anual_para_exibicao[analise_anual_para_exibicao['ano'] == base_year_for_comparison_data]['perc_eolica'].iloc[0]
        data_base_solar = analise_anual_para_exibicao[analise_anual_para_exibicao['ano'] == base_year_for_comparison_data]['perc_solar'].iloc[0]
        data_base_hidraulica = analise_anual_para_exibicao[analise_anual_para_exibicao['ano'] == base_year_for_comparison_data]['perc_hidraulica'].iloc[0]
        data_base_eolica_solar_combinada = data_base_eolica + data_base_solar
        data_base_novas_renovaveis = analise_anual_para_exibicao[analise_anual_para_exibicao['ano'] == base_year_for_comparison_data]['perc_novas_renovaveis'].iloc[0]
    else:
        data_base_eolica, data_base_solar, data_base_hidraulica, data_base_novas_renovaveis = 0.0, 0.0, 0.0, 0.0
        data_base_eolica_solar_combinada = 0.0

    df_comparison_data = {
        'Fonte': ['Eólica', 'Solar', 'Eólica + Solar', 'Novas Renováveis (Eólica + Solar)', 'Hidráulica'],
        f'Participação em {base_year_for_comparison_data} (%)': [
            float(data_base_eolica), float(data_base_solar), float(data_base_eolica_solar_combinada), float(data_base_novas_renovaveis), float(data_base_hidraulica)
        ],
        f'Participação Projetada em {forecast_until_year} (%)': [
            float(data_2030_eolica), float(data_2030_solar), float(data_2030_eolica_solar_combinada), float(data_2030_novas_renovaveis), float(data_2030_hidraulica)
        ],
        'Diferença (p.p.)': [
            float(data_2030_eolica - data_base_eolica),
            float(data_2030_solar - data_base_solar),
            float(data_2030_eolica_solar_combinada - data_base_eolica_solar_combinada),
            float(data_2030_novas_renovaveis - data_base_novas_renovaveis),
            float(data_2030_hidraulica - data_base_hidraulica)
        ]
    }
    df_comparison = pd.DataFrame(df_comparison_data)


    # Removida a aba 'Relatório Completo (JSON)' da lista de abas
    tab_overview, tab_growth, tab_regional, tab_timeseries, tab_predictions, tab_2030_analysis = st.tabs(
        ["Visão Geral e ODS 7", "Análise de Crescimento", "Análise Regional", "Análise de Série Temporal", "Previsões e Conceitos", "Análise 2030: Eólica/Solar vs. Hidráulica"]
    )

    with tab_overview:
        st.header("Composição da Matriz e Participação Renovável")
        st.markdown("""
        Esta seção mostra a 'receita de bolo' da energia do **Sistema Interligado Nacional (SIN)** e a 'fatia' das fontes renováveis. Isso é importante pra ver se estamos seguindo o **ODS 7.2** da ONU, que fala em ter mais energia 'verde'.
        
        **Atenção na 'Fatia Verde Total':** A gente conta a energia da água (hidrelétrica) nessa 'fatia verde total'. Como a participação da água pode não crescer tanto ou até diminuir, o total de 'fatias verdes' pode não parecer um 'boom' tão grande. Por isso, olhe também a 'Fatia das Novas Renováveis' (vento e sol)!
        """)
        
        display_descriptive_stats(analise_anual_para_exibicao, ['perc_renovavel_total', 'perc_novas_renovaveis'], "Participação Renovável Total e Novas Renováveis (%)")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Geração por Fonte (MWmed) (Nossa 'Colheita' Anual)")
            df_gen_melted = analise_anual_para_exibicao.melt(id_vars=['ano'],
                                                     value_vars=['Hidraulica', 'Termica', 'Eolica', 'Solar'],
                                                     var_name='Fonte', value_name='Geração (MWmed)')
            fig_matriz = px.bar(df_gen_melted, x='ano', y='Geração (MWmed)', color='Fonte',
                                 title='<b>Colheita Anual de Energia por Tipo de Fazenda (MWmed)</b>',
                                 labels={'Geração (MWmed)': 'Produção em Gigawatts Médios <br><sub>(Gigawatts Médios, tipo a "força" das usinas)</sub>', 'ano': 'Ano'})
            fig_matriz.update_layout(barmode='stack', hovermode="x unified")
            st.plotly_chart(fig_matriz, use_container_width=True)

            st.subheader("Composição do Bolo da Energia: Participação Percentual Anual das Fontes")
            fig_plotly_stack = plot_participacao_anual_fontes(analise_anual_para_exibicao)
            st.plotly_chart(fig_plotly_stack, use_container_width=True)
            

        with col2:
            st.subheader("Fatia Verde Total: Histórico e Previsão até 2030")
            df_renovavel_plot = pd.DataFrame({
                'ano': analise_anual_renovavel_lr_combined['ano'],
                'perc_renovavel_total': analise_anual_renovavel_lr_combined['perc_renovavel_total'],
                'Tipo': ['Histórico'] * len(analise_anual_para_exibicao) + ['Previsão'] * len(pred_renovavel_lr)
            })

            fig_perc = px.line(df_renovavel_plot, x='ano', y='perc_renovavel_total', 
                               title='<b>Fatia Verde Total: Histórico e Aposta Futura (2030)</b>',
                               markers=True, color='Tipo', line_dash='Tipo',
                               color_discrete_map={'Histórico': 'blue', 'Previsão': 'red'})
            fig_perc.update_layout(xaxis_title='Ano', yaxis_title='% Renovável <br><sub>(Porcentagem no total, como uma fatia do bolo)</sub>', showlegend=True, hovermode="x unified")
            st.plotly_chart(fig_perc, use_container_width=True)

            st.subheader("Fatia das Novas Renováveis (Eólica + Solar): Histórico e Previsão até 2030")
            st.markdown("""
            Este gráfico foca especificamente no 'crescimento-foguete' das energias do Vento e do Sol, sem contar a Hidrelétrica. Essa 'medida' é mais 'esperta' pra ver o avanço das tecnologias 'verdes' mais recentes no Brasil.
            """)
            df_novas_renovaveis_plot = pd.DataFrame({
                'ano': analise_anual_novas_renovaveis_lr_combined['ano'],
                'perc_novas_renovaveis': analise_anual_novas_renovaveis_lr_combined['perc_novas_renovaveis'],
                'Tipo': ['Histórico'] * len(analise_anual_para_exibicao) + ['Previsão'] * len(pred_novas_renovaveis_lr)
            })
            fig_novas_perc = px.line(df_novas_renovaveis_plot, x='ano', y='perc_novas_renovaveis', 
                               title='<b>Fatia das Novas Renováveis (Eólica + Solar): Histórico e Aposta Futura (2030)</b>',
                               markers=True, color='Tipo', line_dash='Tipo',
                               color_discrete_map={'Histórico': 'purple', 'Previsão': 'orange'})
            fig_novas_perc.update_layout(xaxis_title='Ano', yaxis_title='% Novas Renováveis <br><sub>(Porcentagem no total, como uma fatia do bolo)</sub>', showlegend=True, hovermode="x unified")
            st.plotly_chart(fig_novas_perc, use_container_width=True)


            st.subheader("O Bolo Energético de 2024: Participação das Renováveis vs. Não Renováveis")
            fig_pizza = plot_pizza_participacao_2024(analise_anual_para_exibicao)
            st.plotly_chart(fig_pizza, use_container_width=True)

    with tab_growth:
        st.header("Termômetros de Crescimento: Velocidade da Expansão Eólica e Solar")
        st.markdown("""
        **Por que olhar isso?** Pra medir a **velocidade** que as 'fazendas' de Vento e Sol estão crescendo! Taxas altas indicam que tem muito 'dinheiro novo' sendo investido e muita gente usando essa tecnologia. Isso é 'ouro' pra mudar a nossa matriz energética.
        """)
        st.info("🎯 **Alinhamento: ODS 7.2** (Ter mais energia 'verde') e **ODS 7.a** (Promover 'dinheiro novo' e tecnologia).")
        
        display_descriptive_stats(analise_anual_para_exibicao, ['perc_eolica', 'perc_solar', 'perc_novas_renovaveis'], "Participação Eólica, Solar e Novas Renováveis (%)")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Fatia Eólica: Histórico e Previsão até 2030")
            df_eolica_plot = pd.DataFrame({
                'ano': analise_anual_eolica_lr_combined['ano'],
                'perc_eolica': analise_anual_eolica_lr_combined['perc_eolica'],
                'Tipo': ['Histórico'] * len(analise_anual_para_exibicao) + ['Previsão'] * len(pred_eolica_lr)
            })
            fig_eolica = px.line(df_eolica_plot, x='ano', y='perc_eolica', 
                                 title='<b>Fatia Eólica: Histórico e Aposta Futura (2030)</b>',
                                 markers=True, color='Tipo', line_dash='Tipo',
                                 color_discrete_map={'Histórico': 'green', 'Previsão': 'red'})
            fig_eolica.update_layout(xaxis_title='Ano', yaxis_title='% Eólica <br><sub>(Porcentagem no total, como uma fatia do bolo)</sub>', showlegend=True, hovermode="x unified")
            st.plotly_chart(fig_eolica, use_container_width=True)

            st.subheader("Termômetro do Vento: Crescimento Anual da Participação Eólica")
            fig_cres_eolica = plot_crescimento_eolica(analise_anual_para_exibicao)
            st.plotly_chart(fig_cres_eolica, use_container_width=True)
            
        with col2:
            st.subheader("Fatia Solar: Histórico e Previsão até 2030")
            df_solar_plot = pd.DataFrame({
                'ano': analise_anual_solar_lr_combined['ano'],
                'perc_solar': analise_anual_solar_lr_combined['perc_solar'],
                'Tipo': ['Histórico'] * len(analise_anual_para_exibicao) + ['Previsão'] * len(pred_solar_lr)
            })
            fig_solar = px.line(df_solar_plot, x='ano', y='perc_solar', 
                                 title='<b>Fatia Solar: Histórico e Aposta Futura (2030)</b>',
                                 markers=True, color='Tipo', line_dash='Tipo',
                                 color_discrete_map={'Histórico': 'orange', 'Previsão': 'red'})
            fig_solar.update_layout(xaxis_title='Ano', yaxis_title='% Solar <br><sub>(Porcentagem no total, como uma fatia do bolo)</sub>', showlegend=True, hovermode="x unified")
            st.plotly_chart(fig_solar, use_container_width=True)

            st.subheader("Termômetro do Sol: Crescimento Anual da Participação Solar")
            fig_cres_solar = plot_crescimento_solar(analise_anual_para_exibicao)
            st.plotly_chart(fig_cres_solar, use_container_width=True)

        st.subheader("Termômetro Verde: Crescimento Anual da Participação Renovável Total")
        fig_cres_renovavel = plot_crescimento_renovavel_total(analise_anual_para_exibicao)
        st.plotly_chart(fig_cres_renovavel, use_container_width=True)

        st.subheader("Termômetro da Novidade: Crescimento Anual da Participação de Novas Renováveis (Eólica + Solar)")
        fig_cres_novas_renovaveis = go.Figure()
        fig_cres_novas_renovaveis.add_trace(go.Scatter(x=analise_anual_para_exibicao['ano'][1:], y=analise_anual_para_exibicao['crescimento_novas_renovaveis'][1:],
                                                   mode='lines+markers+text', name='Crescimento Novas Renováveis (%)',
                                                   marker=dict(color='darkviolet'),
                                                   text=[f'{c:.1f}%' for c in analise_anual_para_exibicao['crescimento_novas_renovaveis'][1:]],
                                                   textposition="top center"))
        fig_cres_novas_renovaveis.update_layout(
            title_text="<b>Termômetro da Novidade: Crescimento Percentual Anual da Participação de Novas Renováveis</b>",
            xaxis_title="Ano",
            yaxis_title="Crescimento Anual (%)", 
            xaxis=dict(tickmode='linear', dtick=1)
        )
        st.plotly_chart(fig_cres_novas_renovaveis, use_container_width=True)


    with tab_regional:
        st.header("Raio-X das Regiões: Análise dos Subsistemas Energéticos")
        st.markdown("Aqui a gente vê a 'ajuda' que cada região dá na produção de energia 'verde'. Tem duas formas de ver: **a 'ajuda de casa'** (só da região) e **a 'ajuda pro Brasil'** (comparado com o total do país).")
        st.info("🎯 **Alinhamento: ODS 7.1** (Ter energia pra todo mundo) e **ODS 7.b** (Ter a 'fiação' e as 'usinas' modernas).")

        st.subheader("Produção da Fazenda Verde: Contribuição Absoluta por Região (MWmed)")
        fig_abs = px.bar(analise_regional_anual, x='ano', y='geracao_renovavel_regiao', color='nom_subsistema', title='<b>Produção da Fazenda Verde: Geração Renovável por Subsistema</b>')
        fig_abs.update_layout(barmode='stack', xaxis_title='Ano', 
                              yaxis_title='Geração Renovável (MWmed) <br><sub>(Produção em Gigawatts Médios, tipo a "força" das usinas)</sub>')
        st.plotly_chart(fig_abs, use_container_width=True)
        
    with tab_timeseries:
        st.header("Boletim do Tempo da Energia: Tendências Diárias e Previsões")
        st.markdown("""Essa 'previsão do tempo' detalha a produção de energia por dia, mostrando as 'ondas' e 'apostas' pro futuro. Você verá para cada fonte de energia (Água, Térmica, Vento e Sol):
        * A linha preenchida: a **produção real** de energia a cada dia.
        * Linhas contínuas e tracejadas: as **Médias Móveis**, que são como "filtros" para te mostrar a tendência da produção, tirando os "sobe e desce" do dia a dia.
        * A linha pontilhada: a **Previsão**, uma "aposta" de como a produção pode ser nos próximos dias.

        **DICA ESPERTA:** Para ver os detalhes de um período específico no gráfico, use a **"régua" (o controle deslizante)** que aparece na parte de baixo do gráfico. É como ter uma **"lupa"** para dar zoom no que te interessa!
        """)
        
        st.subheader("Notas Fiscais do Dia: Estatísticas Descritivas da Geração Diária por Fonte")
        display_descriptive_stats(df_diario, df_diario.columns.tolist(), "Geração Diária")

        with st.expander("Clique aqui pra entender as 'Médias Móveis', 'Suavização Exponencial Simples (SES)' e como ler o gráfico:"):
            st.markdown("""
            As **Médias Móveis (MM)** são tipo um 'filtro' que tira o 'barulho' do dia a dia e mostra a 'onda de verdade' (tendência). Para cada linha de energia (como a Hidráulica azul, ou a Térmica vermelha), você verá essas 'ondas':
            - **MM de 30 dias (Linha Contínua):** Mostra a 'tendência do mês', boa pra ver a 'dança' das usinas de Água e Térmicas.
            - **MM de 90 dias (Linha Tracejada):** Mostra a 'onda de longo prazo', confirmando a 'escalada' do Vento e do Sol.
            
            A **Suavização Exponencial Simples (SES)** é usada pras 'apostas' futuras (linhas pontilhadas). Ela dá mais 'peso' para o que aconteceu **recentemente**, fazendo a previsão 'reagir mais rápido' a novas 'mudanças de vento'.
            """)
        fig_diario_pred = plot_serie_diaria(df_diario, df_diario_ses_combined) 
        st.plotly_chart(fig_diario_pred, use_container_width=True)

    with tab_predictions:
        st.header("Detalhes das 'Apostas' e Conceitos 'Difíceis' (que a gente explica!)")
        st.markdown("""
        Aqui você encontra os detalhes sobre os 'modelos de adivinhação' (estatísticos) que usamos e os resultados. Tudo isso alinhado com os **ODS da ONU** (Objetivos de Desenvolvimento Sustentável), especialmente o **ODS 7** ('Energia Limpa e Acessível').
        """)

        st.subheader("1. 'Apostas' de Crescimento (Regressão Linear Simples - RLS)")
        st.markdown("""
        Pra 'apostar' no crescimento do Vento, do Sol, e da 'fatia verde total' (e das **Novas Renováveis**), usamos a **Regressão Linear Simples (RLS)**. Esse 'método' é como usar uma **régua** pra ver a 'linha reta' da relação entre duas coisas.
        """)
        st.markdown("### Conceitos de Regressão Linear Simples (RLS):")
        st.markdown(r"""
        A RLS tenta 'passar uma régua' pelos seus dados. Essa 'régua' é a que 'melhor se encaixa', ou seja, a que faz a 'soma dos erros' (distâncias de cada ponto até a régua) ser a menor possível. Isso é o **Método dos Mínimos Quadrados**.

        A 'equação da régua' é assim: $Y = \beta_0 + \beta_1 X$.
        * **$Y$ (O que queremos prever):** É o que a gente quer 'adivinhar' (tipo, a % de fatia eólica).
        * **$X$ (O que usamos pra prever):):** É a 'informação' que usamos pra 'adivinhar' (o ano).
        * **$\beta_0$ (Ponto de Partida):** É o valor 'adivinhado' de $Y$ quando $X$ é zero.
        * **$\beta_1$ (Inclinação da Régua / Velocidade):** Mostra o quanto $Y$ 'muda' a cada vez que $X$ aumenta um pouquinho. É a 'velocidade' que a fatia da energia aumenta ou diminui a cada ano.
        
        **Cuidado com a 'Aposta Longe Demais':** 'Apostar' fora dos dados que a gente já tem (extrapolação) deve ser feito com um 'pé atrás'. A 'linha reta' pode não continuar pra sempre!
        """)

        st.markdown(f"#### Fatia Eólica (%):")
        st.markdown(f"- Coeficiente Angular ($\beta_1$): `{coef_eolica:.4f}` (Aumento médio de `{coef_eolica:.2f}%` na fatia eólica por ano)")
        st.markdown(f"- Intercepto ($\beta_0$): `{intercept_eolica:.4f}`")
        st.dataframe(pred_eolica_lr.head())

        st.markdown(f"#### Fatia Solar (%):")
        st.markdown(f"- Coeficiente Angular ($\beta_1$): `{coef_solar:.4f}` (Aumento médio de `{coef_solar:.2f}%` na fatia solar por ano)")
        st.markdown(f"- Intercepto ($\beta_0$): `{intercept_solar:.4f}`")
        st.dataframe(pred_solar_lr.head())

        st.markdown(f"#### Fatia Renovável Total (%):")
        st.markdown(f"- Coeficiente Angular ($\beta_1$): `{coef_renovavel:.4f}` (Mudança média de `{coef_renovavel:.2f}%` na fatia renovável total por ano)")
        st.markdown(f"- Intercepto ($\beta_0$): `{intercept_renovavel:.4f}`")
        st.dataframe(pred_renovavel_lr.head())
        
        st.markdown(f"#### Fatia das Novas Renováveis (Eólica + Solar) (%):")
        st.markdown(f"- Coeficiente Angular ($\beta_1$): `{coef_novas_renovaveis:.4f}` (Aumento médio de `{coef_novas_renovaveis:.2f}%` na fatia das novas renováveis por ano)")
        st.markdown(f"- Intercepto ($\beta_0$): `{intercept_novas_renovaveis:.4f}`")
        st.dataframe(pred_novas_renovaveis_lr.head())


        st.markdown(f"#### Fatia Hidráulica (%):")
        st.markdown(f"- Coeficiente Angular ($\beta_1$): `{coef_hidraulica:.4f}` (Mudança média de `{coef_hidraulica:.2f}%` na fatia hidráulica por ano)")
        st.dataframe(pred_hidraulica_lr.head())


        st.subheader("2. 'Previsão do Tempo' para o Dia a Dia (Suavização Exponencial Simples - SES)")
        st.markdown("""
        Para as 'apostas' diárias, que podem ser mais 'temperamentais', além das 'Médias Móveis' que você já viu, usamos a **Suavização Exponencial Simples (SES)** para as previsões.
        """)
        st.markdown("### Conceitos de Suavização Exponencial Simples (SES):")
        st.markdown(r"""
        A SES dá **mais 'importância' (pesos) para o que aconteceu mais recentemente**, tipo uma 'memória fresca'.
        
        O 'botão' principal aqui é o **Parâmetro de Suavização ($\alpha$)**, que vai de 0 a 1.
        * **$\alpha$ Perto de 1:** Dá muito 'peso' para o que acabou de acontecer. A 'previsão' se ajusta rápido a novas 'mudanças de vento'.
        * **$\alpha$ Perto de 0:** Dá mais 'peso' para o que aconteceu há mais tempo. A 'previsão' fica mais 'lisinha', menos 'nervosa' com as flutuações rápidas.
        
        A 'aposta' para o próximo período é o valor 'suavizado' do período atual.
        """)
        st.markdown("Previsões Diárias (últimas 5 'apostas' para cada fonte):")
        st.dataframe(df_diario_ses_combined.tail(5))
        st.markdown("*(Os valores 'vazios' (NaN) nas colunas originais são onde a 'aposta' foi feita. Os valores preenchidos são as 'apostas'.)*")


        st.subheader("3. Nosso 'Ponto' no Mapa ODS 7 (Energia Limpa e Acessível)")
        st.markdown("""
        As 'apostas' que fazemos ajudam diretamente a 'monitorar' e 'planificar' o que está ligado ao **ODS 7: Energia Limpa e Acessível**. Em especial, batemos com a meta:
        * **ODS 7.2: "Até 2030, aumentar MUITO a participação de energias renováveis na matriz energética do mundo"**.
        
        Ao 'apostar' no crescimento do Vento e do Sol, e na porcentagem total de energias 'verdes', a gente consegue ver se estamos 'na rota certa' pra ter um aumento "muito grande" até 2030. As 'apostas' diárias, por sua vez, ajudam a 'arrumar a casa' e otimizar a 'fiação' pra integrar cada vez mais essas fontes.
        """)

    with tab_2030_analysis:
        st.header(f"Aposta para {forecast_until_year}: Fatia do Vento/Sol vs. Fatia da Água")
        st.markdown(f"""
        Pra entender a pergunta principal sobre a 'ajuda' futura do Vento e do Sol na energia e como isso pode 'trocar de lugar' com a energia da Água, vamos focar nas 'apostas' para o ano de **{forecast_until_year}**.

        **O que quer dizer 'trocar de lugar'?** Significa que a 'fatia' (participação percentual) das energias do Vento e do Sol vai aumentar no 'bolo da energia', enquanto a 'fatia' da Água pode diminuir em termos relativos (mesmo que a produção dela continue igual ou cresça mais devagar). Não é que as usinas de água vão sumir!

        **Importante sobre as 'Apostas':** Os valores 'apostados' são baseados numa **'Régua' Linear Simples**. É fundamental lembrar que essa 'régua' assume que tudo segue uma linha reta. Em um sistema 'complicado' como a energia, 'coisas novas' (políticas, tecnologias, economia) podem fazer o crescimento ser uma 'curva', não uma linha. Então, essas 'apostas' são uma 'ideia' baseada no passado e podem não pegar toda a 'movimentação' futura.
        """)
        
        if forecast_until_year in analise_anual_eolica_lr_combined['ano'].values:
            data_2030_eolica = analise_anual_eolica_lr_combined[analise_anual_eolica_lr_combined['ano'] == forecast_until_year]['perc_eolica'].iloc[0]
            data_2030_solar = analise_anual_solar_lr_combined[analise_anual_solar_lr_combined['ano'] == forecast_until_year]['perc_solar'].iloc[0]
            data_2030_hidraulica = analise_anual_hidraulica_lr_combined[analise_anual_hidraulica_lr_combined['ano'] == forecast_until_year]['perc_hidraulica'].iloc[0]
            data_2030_novas_renovaveis = analise_anual_novas_renovaveis_lr_combined[analise_anual_novas_renovaveis_lr_combined['ano'] == forecast_until_year]['perc_novas_renovaveis'].iloc[0]
        else:
            st.warning(f"Dados 'apostados' para o ano {forecast_until_year} não encontrados. Ajuste o 'ano alvo' ou a 'base de dados'.")
            data_2030_eolica, data_2030_solar, data_2030_hidraulica, data_2030_novas_renovaveis = 0.0, 0.0, 0.0, 0.0 
            
        data_2030_eolica_solar_combinada = data_2030_eolica + data_2030_solar

        base_year = current_year_for_prediction
        if base_year in analise_anual_para_exibicao['ano'].values:
            data_base_eolica = analise_anual_para_exibicao[analise_anual_para_exibicao['ano'] == base_year]['perc_eolica'].iloc[0]
            data_base_solar = analise_anual_para_exibicao[analise_anual_para_exibicao['ano'] == base_year]['perc_solar'].iloc[0]
            data_base_hidraulica = analise_anual_para_exibicao[analise_anual_para_exibicao['ano'] == base_year]['perc_hidraulica'].iloc[0]
            data_base_eolica_solar_combinada = data_base_eolica + data_base_solar
            data_base_novas_renovaveis = analise_anual_para_exibicao[analise_anual_para_exibicao['ano'] == base_year]['perc_novas_renovaveis'].iloc[0]
        else:
            st.warning(f"Dados do 'ano de partida' {base_year} não encontrados para comparação. Os valores de comparação podem ser 0.")
            data_base_eolica, data_base_solar, data_base_hidraulica, data_base_novas_renovaveis = 0.0, 0.0, 0.0, 0.0
            data_base_eolica_solar_combinada = 0.0


        st.subheader(f"1. Resumo da 'Aposta de Fatias' para {forecast_until_year}")
        st.markdown(f"""
        Com base nas 'apostas' da 'régua linear', esperamos que em **{forecast_until_year}** as 'fatias' no 'bolo da energia' sejam mais ou menos:
        * **Energia Eólica:** `{data_2030_eolica:.2f}%`
        * **Energia Solar:** `{data_2030_solar:.2f}%`
        * **Eólica + Solar (Juntas):** `{data_2030_eolica_solar_combinada:.2f}%`
        * **Novas Renováveis (Eólica + Solar):** `{data_2030_novas_renovaveis:.2f}%`
        * **Energia Hidráulica:** `{data_2030_hidraulica:.2f}%`
        """)

        st.subheader(f"2. Comparando as 'Fatias': {base_year} vs. {forecast_until_year}")
        df_comparison_data = {
            'Fonte': ['Eólica', 'Solar', 'Eólica + Solar', 'Novas Renováveis (Eólica + Solar)', 'Hidráulica'],
            f'Participação em {base_year} (%)': [
                float(data_base_eolica), float(data_base_solar), float(data_base_eolica_solar_combinada), float(data_base_novas_renovaveis), float(data_base_hidraulica)
            ],
            f'Participação Projetada em {forecast_until_year} (%)': [
                float(data_2030_eolica), float(data_2030_solar), float(data_2030_eolica_solar_combinada), float(data_2030_novas_renovaveis), float(data_2030_hidraulica)
            ],
            'Diferença (p.p.)': [
                float(data_2030_eolica - data_base_eolica),
                float(data_2030_solar - data_base_solar),
                float(data_2030_eolica_solar_combinada - data_base_eolica_solar_combinada),
                float(data_2030_novas_renovaveis - data_base_novas_renovaveis),
                float(data_2030_hidraulica - data_base_hidraulica)
            ]
        }
        df_comparison = pd.DataFrame(df_comparison_data)
        st.dataframe(df_comparison.style.format({
            f'Participação em {base_year} (%)': '{:,.2f}%',
            f'Participação Projetada em {forecast_until_year} (%)': '{:,.2f}%',
            'Diferença (p.p.)': '{:,.2f}'
        }), use_container_width=True)

        st.markdown(f"""
        Olhando a tabela e as 'apostas', a gente vê que:
        * A **'fatia' do Vento e do Sol (juntos)** deve **mudar em `{df_comparison[df_comparison['Fonte'] == 'Eólica + Solar']['Diferença (p.p.)'].iloc[0]:.2f}` pontos percentuais** de {base_year} para {forecast_until_year}.
        * A **'fatia' das Novas Renováveis (Vento + Sol)** deve **mudar em `{df_comparison[df_comparison['Fonte'] == 'Novas Renováveis (Eólica + Solar)']['Diferença (p.p.)'].iloc[0]:.2f}` pontos percentuais** no mesmo período. A gente espera que esse número seja positivo, mostrando a 'escalada' das fontes mais novas.
        * A **'fatia' da Água (Hidráulica)** deve **mudar em `{df_comparison[df_comparison['Fonte'] == 'Hidráulica']['Diferença (p.p.)'].iloc[0]:.2f}` pontos percentuais** no mesmo período. (Se o número for negativo, a 'fatia' dela diminui; se for positivo, aumenta.)

        Isso quer dizer que o crescimento do Vento e do Sol vai fazer uma **'troca de lugar' na composição do 'bolo da energia'**, com uma 'ajuda' percentual dessas fontes 'intermitentes' (que dependem do tempo) em relação à água, dependendo do histórico. Esse é um dado 'chave' para você 'ajustar as políticas', como você já me disse antes.
        """)

        st.subheader(f"3. Comparando as 'Fatias Chave' em {forecast_until_year}")
        df_2030_plot = pd.DataFrame({
            'Fonte': ['Hidráulica', 'Eólica', 'Solar', 'Novas Renováveis (Eólica + Solar)'],
            'Participação (%)': [float(data_2030_hidraulica), float(data_2030_eolica), float(data_2030_solar), float(data_2030_novas_renovaveis)],
            'Cor': ['#4c78a8', '#54a24b', '#f89e47', '#8A2BE2']
        })
        
        order = ['Hidráulica', 'Eólica', 'Solar', 'Novas Renováveis (Eólica + Solar)']
        df_2030_plot['Fonte'] = pd.Categorical(df_2030_plot['Fonte'], categories=order, ordered=True)
        df_2030_plot = df_2030_plot.sort_values('Fonte')

        fig_2030_comp = px.bar(
            df_2030_plot,
            x='Fonte',
            y='Participação (%)',
            title=f'<b>Aposta Futura: Participação Projetada na Matriz Elétrica Brasileira em {forecast_until_year}</b>',
            labels={'Participação (%)': 'Fatia do Bolo (%)'},
            color='Fonte',
            color_discrete_map={
                'Hidráulica': '#4c78a8',
                'Eólica': '#54a24b', 
                'Solar': '#f89e47', 
                'Novas Renováveis (Eólica + Solar)': '#8A2BE2'
            },
            text='Participação (%)'
        )
        fig_2030_comp.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_2030_comp.update_layout(yaxis_range=[0, max(df_2030_plot['Participação (%)']) * 1.1])
        st.plotly_chart(fig_2030_comp, use_container_width=True)

        st.subheader(f"4. Corrida das Fatias: Hidráulica vs. Novas Renováveis (Até {forecast_until_year})")
        
        df_eolica_solar_combined_proj = pd.DataFrame({
            'ano': analise_anual_novas_renovaveis_lr_combined['ano'],
            'Novas Renováveis (Eólica + Solar) (%)': analise_anual_novas_renovaveis_lr_combined['perc_novas_renovaveis']
        })
        
        df_comparison_evolution = pd.merge(df_eolica_solar_combined_proj, analise_anual_hidraulica_lr_combined, on='ano')
        df_comparison_evolution = df_comparison_evolution.rename(columns={'perc_hidraulica': 'Hidráulica (%)'})

        df_comparison_evolution_melted = df_comparison_evolution.melt(
            id_vars=['ano'], 
            value_vars=['Novas Renováveis (Eólica + Solar) (%)', 'Hidráulica (%)'],
            var_name='Fonte', 
            value_name='Participação (%)'
        )
        
        df_comparison_evolution_melted['Tipo'] = df_comparison_evolution_melted['ano'].apply(
            lambda x: 'Histórico' if x <= current_year_for_prediction else 'Previsão'
        )

        fig_evolution_comp = px.line(
            df_comparison_evolution_melted,
            x='ano',
            y='Participação (%)',
            color='Fonte',
            line_dash='Tipo',
            title=f'<b>Corrida das Fatias: Hidráulica vs. Novas Renováveis (Histórico e Aposta Futura até {forecast_until_year})</b>',
            labels={'Participação (%)': 'Fatia no Bolo da Energia (%)'},
            hover_data={'Participação (%)': ':.2f', 'Tipo': True},
            color_discrete_map={
                'Hidráulica (%)': '#4c78a8',
                'Novas Renováveis (Eólica + Solar) (%)': '#8A2BE2'
            }
        )
        fig_evolution_comp.update_layout(hovermode="x unified", yaxis_range=[0, 100])
        st.plotly_chart(fig_evolution_comp, use_container_width=True)


if __name__ == "__main__":
    main()