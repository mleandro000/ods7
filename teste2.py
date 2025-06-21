import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore') # Suprime warnings de bibliotecas

# --- CONFIGURAÇÃO DE LOGGING (NOVO) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURAÇÕES EXTERNAS (SIMULADAS VIA JSON STRINGS - EM PROD. SERIAM ARQUIVOS) ---
# config/policies.json
POLICIES_CONFIG_JSON = """
{
    "prime_plus": {
        "nome": "Prime Plus - Clientes Premium",
        "score_minimo": 850,
        "criterios": {
            "renda_minima": 15000.0,
            "comprometimento_maximo": 15,
            "tempo_relacionamento_minimo": 36,
            "restricoes_permitidas": false,
            "consultas_30d_max": 0,
            "idade_minima": 28,
            "idade_maxima": 60,
            "estabilidade_emprego_minima": 24,
            "utilizacao_cartao_maxima": 20,
            "probabilidade_inadimplencia_max": 0.03
        },
        "limites": {
            "credito_pessoal_max": 500000.0,
            "cartao_credito_max": 100000.0,
            "financiamento_max": 2000000.0
        },
        "condicoes": {
            "taxa_juros_base": 1.5,
            "prazo_maximo_meses": 60,
            "carencia_permitida": true,
            "garantia_exigida": false
        },
        "tipo_teste": null
    },
    "prime": {
        "nome": "Prime - Clientes Preferenciais",
        "score_minimo": 700,
        "criterios": {
            "renda_minima": 7000.0,
            "comprometimento_maximo": 25,
            "tempo_relacionamento_minimo": 18,
            "restricoes_permitidas": false,
            "consultas_30d_max": 1,
            "idade_minima": 25,
            "idade_maxima": 65,
            "estabilidade_emprego_minima": 12,
            "utilizacao_cartao_maxima": 40,
            "probabilidade_inadimplencia_max": 0.08
        },
        "limites": {
            "credito_pessoal_max": 200000.0,
            "cartao_credito_max": 50000.0,
            "financiamento_max": 800000.0
        },
        "condicoes": {
            "taxa_juros_base": 2.2,
            "prazo_maximo_meses": 48,
            "carencia_permitida": true,
            "garantia_exigida": false
        },
        "tipo_teste": null
    },
    "standard": {
        "nome": "Standard - Clientes Regulares",
        "score_minimo": 550,
        "criterios": {
            "renda_minima": 3000.0,
            "comprometimento_maximo": 35,
            "tempo_relacionamento_minimo": 9,
            "restricoes_permitidas": true,
            "consultas_30d_max": 2,
            "idade_minima": 21,
            "idade_maxima": 70,
            "estabilidade_emprego_minima": 6,
            "utilizacao_cartao_maxima": 60,
            "probabilidade_inadimplencia_max": 0.20
        },
        "limites": {
            "credito_pessoal_max": 50000.0,
            "cartao_credito_max": 20000.0,
            "financiamento_max": 300000.0
        },
        "condicoes": {
            "taxa_juros_base": 3.5,
            "prazo_maximo_meses": 36,
            "carencia_permitida": false,
            "garantia_exigida": true
        },
        "tipo_teste": null
    },
    "risk_based": {
        "nome": "Risk Based - Clientes com Restrições (Avaliação Caso a Caso)",
        "score_minimo": 400,
        "criterios": {
            "renda_minima": 1800.0,
            "comprometimento_maximo": 45,
            "tempo_relacionamento_minimo": 3,
            "restricoes_permitidas": true,
            "consultas_30d_max": 4,
            "idade_minima": 21,
            "idade_maxima": 65,
            "estabilidade_emprego_minima": 1,
            "utilizacao_cartao_maxima": 75,
            "probabilidade_inadimplencia_max": 0.40
        },
        "limites": {
            "credito_pessoal_max": 15000.0,
            "cartao_credito_max": 5000.0,
            "financiamento_max": 100000.0
        },
        "condicoes": {
            "taxa_juros_base": 6.0,
            "prazo_maximo_meses": 24,
            "carencia_permitida": false,
            "garantia_exigida": true
        },
        "tipo_teste": "Challenger"
    },
    "microcredito": {
        "nome": "Microcrédito - Apoio Social/Empreendedor",
        "score_minimo": 300,
        "criterios": {
            "renda_minima": 800.0,
            "comprometimento_maximo": 55,
            "tempo_relacionamento_minimo": 0,
            "restricoes_permitidas": true,
            "consultas_30d_max": 7,
            "idade_minima": 18,
            "idade_maxima": 70,
            "estabilidade_emprego_minima": 0,
            "utilizacao_cartao_maxima": 90,
            "probabilidade_inadimplencia_max": 0.60
        },
        "limites": {
            "credito_pessoal_max": 5000.0,
            "cartao_credito_max": 1000.0,
            "financiamento_max": 20000.0
        },
        "condicoes": {
            "taxa_juros_base": 8.0,
            "prazo_maximo_meses": 18,
            "carencia_permitida": false,
            "garantia_exigida": true
        },
        "tipo_teste": null
    }
}
"""

# config/score_weights.json
SCORE_WEIGHTS_CONFIG_JSON = """
{
    "primary_weights": {
        "comportamento_pagamento": 0.40,
        "endividamento_atual": 0.25,
        "relacionamento_bancario": 0.15,
        "dados_demograficos": 0.10,
        "consultas_recentes": 0.05,
        "dados_cadastrais": 0.05
    },
    "detailed_criteria_weights": {
        "comportamento_pagamento": {
            "historico_spc_serasa": 0.35,
            "pontualidade_12m": 0.25,
            "inadimplencia_historica": 0.20,
            "regularizacao_debitos": 0.15,
            "protestos_cartorio": 0.05
        },
        "endividamento_atual": {
            "comprometimento_renda": 0.40,
            "utilizacao_limite_cartao": 0.25,
            "numero_contratos_ativos": 0.15,
            "valor_total_dividas": 0.10,
            "divida_setor_bancario": 0.10
        },
        "relacionamento_bancario": {
            "tempo_conta_corrente": 0.30,
            "movimentacao_financeira": 0.25,
            "produtos_contratados": 0.20,
            "relacionamento_multiplo": 0.15,
            "conta_salario": 0.10
        },
        "dados_demograficos": {
            "renda_declarada": 0.35,
            "estabilidade_profissional": 0.25,
            "faixa_etaria": 0.15,
            "escolaridade": 0.15,
            "estado_civil": 0.10
        },
        "consultas_recentes": {
            "consultas_ultimos_30d": 0.50,
            "consultas_ultimos_90d": 0.30,
            "consultas_proprias": 0.20
        },
        "dados_cadastrais": {
            "atualizacao_cadastral": 0.40,
            "consistencia_dados": 0.30,
            "telefones_validados": 0.20,
            "endereco_confirmado": 0.10
        }
    },
    "segment_adjustments": {
        "pessoa_fisica": {"multiplier": 1.0, "min_score": 0, "max_score": 1000},
        "pessoa_juridica": {"multiplier": 1.1, "min_score": 0, "max_score": 1000},
        "microempresario": {"multiplier": 0.95, "min_score": 0, "max_score": 1000}
    }
}
"""

# config/lgd_params.json
LGD_PARAMS_CONFIG_JSON = """
{
    "credito_pessoal": 0.60,
    "cartao_credito": 0.70,
    "financiamento": 0.30,
    "microcredito": 0.80
}
"""

# utils/explanations.py
CRITERIA_EXPLANATIONS = {
    # CRITÉRIOS DE SCORE
    'comportamento_pagamento': {
        'nome': 'Comportamento de Pagamento',
        'racional': 'Avalia o histórico de como o cliente gerencia suas dívidas. É um dos fatores mais preditivos do risco de crédito, pois reflete a disciplina financeira passada.',
        'sub_criterios': {
            'historico_spc_serasa': 'Verifica a existência de restrições em órgãos de proteção ao crédito. Restrições indicam inadimplência registrada e impactam negativamente na capacidade de obter novos créditos.',
            'pontualidade_12m': 'Mede a regularidade e pontualidade dos pagamentos nos últimos 12 meses. Pagamentos em dia demonstram capacidade e intenção de honrar compromissos, sendo um forte indicativo de bom pagador.',
            'inadimplencia_historica': 'Analisa o histórico de débitos não pagos ou pagamentos com atraso significativo. Quanto maior e mais recente o histórico de inadimplência, maior o risco percebido pelo credor.',
            'regularizacao_debitos': 'Avalia a velocidade e a capacidade do cliente em regularizar débitos em atraso. Uma regularização rápida e completa é um bom sinal de recuperação de crédito.',
            'protestos_cartorio': 'Indica a existência de títulos protestados em cartório, que são dívidas legalmente reconhecidas como não pagas e representam um dos maiores riscos de crédito.'
        }
    },
    'endividamento_atual': {
        'nome': 'Endividamento Atual',
        'racional': 'Analisa o nível de endividamento do cliente em relação à sua capacidade de pagamento. Um endividamento excessivo pode indicar dificuldade futura em arcar com novas dívidas e aumentar o risco de default.',
        'sub_criterios': {
            'comprometimento_renda': 'Percentual da renda mensal líquida já comprometido com parcelas de outras dívidas. Um comprometimento alto (<30-40%) reduz a margem para novas obrigações financeiras.',
            'utilizacao_limite_cartao': 'Proporção do limite de crédito disponível em cartões que está sendo utilizada. Alta utilização (e.g., >70%) pode indicar dependência de crédito rotativo, falta de liquidez ou estresse financeiro.',
            'numero_contratos_ativos': 'Quantidade de operações de crédito ativas (empréstimos, financiamentos, crediários, etc.). Muitos contratos podem pulverizar a capacidade de pagamento e dificultar o controle financeiro.',
            'valor_total_dividas': 'Soma total das dívidas do cliente, incluindo todos os tipos de crédito. Um alto volume absoluto de dívida pode ser preocupante, mesmo com boa renda, se não houver ativos equivalentes.',
            'divida_setor_bancario': 'Concentração do endividamento no setor bancário. Alta concentração pode indicar poucos credores dispostos a emprestar, sugerindo um maior risco percebido pelo mercado, ou falta de diversificação de fontes de crédito.'
        }
    },
    'relacionamento_bancario': {
        'nome': 'Relacionamento Bancário',
        'racional': 'Avalia a profundidade e a qualidade da relação do cliente com instituições financeiras. Um relacionamento longo e diversificado pode indicar estabilidade, confiança e histórico de dados robusto.',
        'sub_criterios': {
            'tempo_conta_corrente': 'Duração do relacionamento com o banco principal ou a conta mais antiga. Contas antigas e ativas geralmente indicam maior estabilidade financeira e previsibilidade de fluxo.',
            'movimentacao_financeira': 'Volume e regularidade das transações financeiras na conta (entradas e saídas). Boa movimentação sugere atividade econômica, liquidez e capacidade de gestão financeira.',
            'produtos_contratados': 'Número e variedade de produtos e serviços bancários utilizados (investimentos, seguros, outros créditos, previdência). Diversificação indica engajamento com a instituição e potencial de relacionamento a longo prazo.',
            'relacionamento_multiplo': 'Se o cliente mantém relacionamento (e/ou dívidas) com múltiplos bancos. Pode ser um sinal de busca ativa por melhores condições, mas em excesso pode indicar fragilidade ou "pinga-pinga" de dívidas.',
            'conta_salario': 'Possuir uma conta salário vinculada ao banco. Indica fluxo de renda regular e primário através da instituição, facilitando a análise e a oferta de crédito consignado.'
        }
    },
    'dados_demograficos': {
        'nome': 'Dados Demográficos',
        'racional': 'Informações pessoais que, embora não diretamente financeiras, podem ser correlacionadas com a estabilidade e capacidade de pagamento. Usado com cautela para evitar vieses e garantir conformidade com a LGPD e leis antidiscriminação.',
        'sub_criterios': {
            'renda_declarada': 'Renda mensal declarada pelo cliente e, idealmente, comprovada. Serve como base fundamental para o cálculo da capacidade de pagamento e comprometimento de renda.',
            'estabilidade_profissional': 'Tempo de permanência no emprego atual ou na atividade profissional autônoma. Maior estabilidade sugere menor risco de desemprego e interrupção de renda, aumentando a previsibilidade.',
            'faixa_etaria': 'Idade do cliente. Determinadas faixas etárias (e.g., 25-55 anos) são historicamente associadas a maior estabilidade financeira, maturidade e menor probabilidade de default.',
            'escolaridade': 'Nível educacional formal. Pode indicar maior potencial de renda futura, estabilidade no mercado de trabalho e capacidade de gestão financeira, embora deva ser usada com cautela para evitar vieses.',
            'estado_civil': 'Estado civil do cliente. Clientes casados ou em união estável podem apresentar maior estabilidade financeira devido a co-responsabilidade e planejamento conjunto, mas o uso deve ser monitorado por vieses.'
        }
    },
    'consultas_recentes': {
        'nome': 'Consultas Recentes',
        'racional': 'Avalia a frequência com que o CPF/CNPJ do cliente foi consultado por outras instituições financeiras ou empresas. Muitas consultas em pouco tempo (hard pulls) podem indicar busca desesperada por crédito (credit-seeking behavior), um sinal de alerta.',
        'sub_criterios': {
            'consultas_ultimos_30d': 'Número de vezes que o crédito do cliente foi consultado nos últimos 30 dias. Um alto volume neste curto período é um forte indicador de risco crescente.',
            'consultas_ultimos_90d': 'Número de consultas nos últimos 90 dias. Complementa a análise dos 30 dias, mostrando uma tendência de busca por crédito.',
            'consultas_proprias': 'Quantidade de vezes que o próprio cliente consultou seu score/CPF (soft pulls). Geralmente é neutro ou ligeiramente positivo (indica consciência financeira), mas um excesso incomum pode ser preocupante.'
        }
    },
    'dados_cadastrais': {
        'nome': 'Dados Cadastrais',
        'racional': 'Verifica a qualidade, atualização e consistência das informações cadastrais do cliente. Dados desatualizados, inconsistentes ou incompletos podem indicar risco de fraude, dificuldade de contato ou falta de transparência.',
        'sub_criterios': {
            'atualizacao_cadastral': 'Recência da última atualização dos dados cadastrais do cliente. Dados recentemente atualizados indicam confiabilidade e facilidade de contato.',
            'consistencia_dados': 'Grau de concordância das informações do cliente entre diferentes fontes de dados (ex: dados bancários vs. dados de birôs). Inconsistências levantam bandeiras vermelhas e podem indicar fraude.',
            'telefones_validados': 'Número de telefones de contato confirmados e válidos. Facilita a comunicação e a cobrança, se necessário, reduzindo o risco operacional.',
            'endereco_confirmado': 'Confirmação do endereço residencial ou comercial do cliente. Um endereço estável e validado é um sinal de estabilidade e reduz o risco de "cliente fantasma".'
        }
    },
    # CRITÉRIOS DE POLÍTICA (GERAIS)
    'score_minimo': '**Critério de Política:** Score de crédito mínimo que um cliente deve ter para ser elegível a esta política. Clientes abaixo deste score são automaticamente reprovados para esta política. É um corte rígido.',
    'probabilidade_inadimplencia_max': '**Critério de Política:** Probabilidade de Inadimplência (PD) máxima permitida para o cliente se qualificar a esta política. Uma PD acima deste limite indica um risco muito elevado e inviabiliza a aprovação.',
    'renda_minima': '**Critério de Política:** Renda mensal bruta mínima exigida para a política. Garante que o cliente tem capacidade financeira para suportar o crédito e as parcelas.',
    'comprometimento_maximo': '**Critério de Política:** Percentual máximo da renda do cliente que pode já estar comprometida com outras dívidas. Essencial para avaliar a capacidade de endividamento adicional sem sobrecarregar o cliente.',
    'tempo_relacionamento_minimo': '**Critério de Política:** Período mínimo de relacionamento (em meses) que o cliente deve ter com a instituição financeira ou no mercado de crédito em geral. Tempo de relacionamento mais longo geralmente indica menor risco.',
    'restricoes_permitidas': '**Critério de Política:** Indica se esta política aceita clientes com histórico de restrições financeiras (como SPC/Serasa). Políticas de alto risco podem permitir, mas geralmente com condições mais duras (limites menores, taxas maiores).',
    'consultas_30d_max': '**Critério de Política:** Número máximo de consultas ao CPF/CNPJ do cliente por outras empresas nos últimos 30 dias. Muitas consultas podem ser um sinal de "caça a crédito" ou desespero financeiro.',
    'idade_minima': '**Critério de Política:** Idade mínima permitida para o cliente. As políticas geralmente têm faixas etárias ideais para otimizar o risco e a longevidade do crédito.',
    'idade_maxima': '**Critério de Política:** Idade máxima permitida para o cliente. Idades muito avançadas podem apresentar maior risco de longevidade ou capacidade de pagamento a longo prazo.',
    'estabilidade_emprego_minima': '**Critério de Política:** Tempo mínimo de permanência (em meses) no emprego ou na atividade profissional atual. Maior estabilidade profissional geralmente significa menor risco de desemprego e interrupção de renda.',
    'utilizacao_cartao_maxima': '**Critério de Política:** Percentual máximo da linha de crédito de cartões que o cliente pode estar utilizando. Altas utilizações podem indicar dependência de crédito rotativo e potencial risco de default.',
    'tipo_teste': '**Critério de Política (Champion/Challenger):** Indica se esta política é um "Champion" (a política atual padrão) ou "Challenger" (uma política experimental sendo testada contra a Champion). Em um sistema real, aplicações seriam roteadas entre elas para comparação de desempenho.'
    ,
    # LIMITES E CONDIÇÕES
    'credito_pessoal_max': '**Limite de Política:** Valor máximo base para crédito pessoal que pode ser concedido nesta política. O limite final para o cliente é ajustado dinamicamente com base no score e PD.',
    'cartao_credito_max': '**Limite de Política:** Valor máximo base para limite de cartão de crédito. O limite final para o cliente é ajustado dinamicamente com base no score e PD.',
    'financiamento_max': '**Limite de Política:** Valor máximo base para financiamentos (imóveis, veículos, etc.). O limite final para o cliente é ajustado dinamicamente.',
    'taxa_juros_base': '**Condição de Política:** Taxa de juros base da política (% ao mês). A taxa final concedida ao cliente será ajustada dinamicamente para refletir sua probabilidade de inadimplência (PD).',
    'prazo_maximo_meses': '**Condição de Política:** Número máximo de meses para o pagamento do crédito. Prazos mais longos podem diluir parcelas, mas aumentam o risco total e a perda esperada.',
    'carencia_permitida': '**Condição de Política:** Indica se a política permite um período de carência (sem pagamento de parcelas) no início do contrato. Pode ser um benefício para o cliente, mas com risco e custo associados para a instituição.',
    'garantia_exigida': '**Condição de Política:** Se a concessão do crédito exige alguma forma de garantia (e.g., imóvel, veículo). Garantias mitigam significativamente o risco em caso de inadimplência.',
    # LGD (Loss Given Default)
    'lgd_credito_pessoal': '**Perda Dado Inadimplência (LGD):** Percentual do valor do crédito pessoal que a instituição espera perder em caso de inadimplência, após considerar recuperações, garantias, etc. É um insumo crucial para o cálculo do Expected Loss.',
    'lgd_cartao_credito': '**Perda Dado Inadimplência (LGD):** Percentual do valor do cartão de crédito que a instituição espera perder em caso de inadimplência, considerando as características do produto (e.g., rotativo, sem garantia).',
    'lgd_financiamento': '**Perda Dado Inadimplência (LGD):** Percentual do valor do financiamento que a instituição espera perder em caso de inadimplência. Geralmente menor que outros produtos devido à presença de garantias (veículo, imóvel).',
    'lgd_microcredito': '**Perda Dado Inadimplência (LGD):** Percentual do valor do microcrédito que a instituição espera perder em caso de inadimplência. Frequentemente mais alto devido ao perfil do público e à ausência de garantias robustas.'
}


# --- DATACLASS PARA DADOS DO CLIENTE (MELHORIA NA ESTRUTURA DE DADOS) ---
@dataclass
class ClientData:
    name: str
    cpf: str
    monthly_income: float
    total_debt: float
    account_age_months: int
    restrictions: bool
    inquiries_30d: int
    age: int
    employment_stability_months: int
    credit_utilization: int
    payment_history: int
    historical_default_rate: int
    debt_regularization_speed: int
    protests: int
    open_accounts: int
    bank_debt_concentration: int
    monthly_turnover: float
    bank_products_count: int
    banks_relationship_count: int
    has_salary_account: bool
    education_level: int
    marital_status: int
    inquiries_90d: int
    self_inquiries: int
    days_since_update: int
    data_consistency_score: int
    validated_phones: int
    address_confirmed: bool
    # Novos campos para simular Open Finance / Dados Alternativos (NOVO)
    average_monthly_transactions: float = 0.0
    investment_balance: float = 0.0
    utility_bill_on_time_payment_ratio: float = 1.0 # 0 a 1
    # Campo para futuro treinamento de ML (NOVO)
    default_event: bool = False # Indicador se o cliente simulado "deu default"

    def to_dict(self):
        """Converte o objeto ClientData em um dicionário."""
        return self.__dict__

    def to_dataframe_features(self) -> pd.DataFrame:
        """
        Converte o objeto ClientData em um DataFrame com as features prontas para o modelo ML.
        CONCEITUAL: Esta é uma versão simplificada. Em um ambiente real, haveria um pipeline
        de feature engineering (OneHotEncoding, StandardScaler, etc.) aplicado aqui.
        """
        data = self.to_dict()
        # Remover campos não usados como features pelo modelo (ex: identificadores, o próprio default_event)
        features_to_exclude = ['name', 'cpf', 'default_event']
        features_dict = {k: v for k, v in data.items() if k not in features_to_exclude}
        # Garantir que a ordem das colunas seja a mesma que o modelo foi treinado
        # Em um sistema real, isso seria feito pelo preprocessor.feature_names_in_
        
        # Simulação de quais features o MockMLModel espera e em que ordem
        # Isso precisa ser consistente com o que MockMLModel "aprende"
        mock_feature_order = [
            'monthly_income', 'total_debt', 'account_age_months', 'restrictions', 
            'inquiries_30d', 'age', 'employment_stability_months', 'credit_utilization',
            'payment_history', 'historical_default_rate', 'debt_regularization_speed',
            'protests', 'open_accounts', 'bank_debt_concentration', 'monthly_turnover',
            'bank_products_count', 'banks_relationship_count', 'has_salary_account',
            'education_level', 'marital_status', 'inquiries_90d', 'self_inquiries',
            'days_since_update', 'data_consistency_score', 'validated_phones', 'address_confirmed',
            'average_monthly_transactions', 'investment_balance', 'utility_bill_on_time_payment_ratio'
        ]
        
        # Cria o DataFrame garantindo a ordem das colunas
        return pd.DataFrame([features_dict], columns=mock_feature_order)


# --- CLASSES DO MOTOR DE CRÉDITO ---

# CONCEITUAL: Mock de um Modelo de Machine Learning (ML)
# Em um ambiente real, 'model.joblib' e 'preprocessor.joblib' seriam arquivos serializados
# de modelos e pipelines de scikit-learn (ou Keras, PyTorch etc.).
class MockMLModel:
    """
    Simula um modelo de Machine Learning treinado para prever PD e Score.
    """
    def __init__(self):
        # Em um caso real, você carregaria o modelo e o preprocessor aqui:
        # self.preprocessor = joblib.load('models/preprocessor.joblib')
        # self.pd_model = joblib.load('models/pd_model.joblib')
        # self.score_transformer = joblib.load('models/score_transformer.joblib') # Ex: um StandardScaler invertido ou função de mapeamento

        logger.info("MockMLModel inicializado. (Simulando carregamento de modelo ML)")

    def predict_pd(self, features_df: pd.DataFrame) -> float:
        """
        Simula a previsão da Probabilidade de Inadimplência (PD) por um modelo ML.
        A PD é simulada com base em algumas features, como seriam em um modelo real.
        """
        # Em um cenário real, `features_df` seria pré-processado e então passado para `self.pd_model.predict_proba`
        # Ex: processed_features = self.preprocessor.transform(features_df)
        # pd_value = self.pd_model.predict_proba(processed_features)[:, 1][0]

        # Lógica simulada de PD baseada em algumas features chave (para demo)
        # Esta lógica AGORA USA DIRETAMENTE as features do DataFrame `features_df`
        income = features_df['monthly_income'].iloc[0]
        debt = features_df['total_debt'].iloc[0]
        restrictions = features_df['restrictions'].iloc[0]
        credit_util = features_df['credit_utilization'].iloc[0]
        account_age = features_df['account_age_months'].iloc[0]
        employment_stab = features_df['employment_stability_months'].iloc[0]
        inquiries_30d = features_df['inquiries_30d'].iloc[0]
        utility_ratio = features_df['utility_bill_on_time_payment_ratio'].iloc[0]
        investment_bal = features_df['investment_balance'].iloc[0]

        base_pd = 0.05 # PD inicial
        
        # Impacto da Renda
        if income < 1500: base_pd += 0.1
        elif income < 3000: base_pd += 0.05

        # Impacto da Dívida
        if income > 0 and debt / income > 0.8: base_pd += 0.15
        elif income == 0: base_pd += 0.3 # Penalidade forte se renda zero

        # Restrições
        if restrictions: base_pd += 0.2

        # Utilização de Crédito
        if credit_util > 80: base_pd += 0.08
        elif credit_util > 50: base_pd += 0.04

        # Tempo de Relacionamento/Emprego
        if account_age < 12: base_pd += 0.03
        if employment_stab < 12: base_pd += 0.03

        # Consultas Recentes
        if inquiries_30d > 2: base_pd += 0.05
        elif inquiries_30d > 0: base_pd += 0.02

        # Dados de Open Finance / Alternativos
        if utility_ratio < 0.7: base_pd += 0.05 # Pagamentos de contas atrasados
        if investment_bal > (debt / 2): base_pd -= 0.02 # Reduz PD se tiver muitos ativos

        # Adiciona um pouco de ruído para simular variação de modelo
        pd_value = base_pd * np.random.uniform(0.9, 1.1)
        
        return max(0.001, min(pd_value, 0.999)) # Garante range válido

    def predict_score(self, features_df: pd.DataFrame) -> int:
        """
        Simula a previsão do Score (0-1000) por um modelo ML, derivado da PD simulada.
        """
        pd_value = self.predict_pd(features_df)
        
        # Mapeamento PD para Score (exemplo FICO-like: PD baixa = score alto)
        # Usamos uma transformação logarítmica simplificada para simular a não-linearidade.
        # Ajuste os coeficientes (100 e 500) para escalar o score como desejar
        # (PD=0.001 -> score alto, PD=0.999 -> score baixo)
        
        # pd_scaled = np.log(pd_value / (1 - pd_value)) # Log-odds ou logit
        # score_value = int(500 - (pd_scaled * 50)) # Ajuste para faixa 0-1000, 500 = média
        
        # Versão mais simples de mapeamento PD para Score (linear inverso)
        score_value = 1000 - int(pd_value * 1000) # PD 0.1 -> 900, PD 0.5 -> 500
        
        # Adiciona um pequeno ruído para simular variância de modelo
        score_value = score_value + np.random.randint(-50, 50)
        
        return max(0, min(score_value, 1000)) # Garante score entre 0 e 1000

    # CONCEITUAL: Método para explicar a previsão (XAI)
    def explain_prediction(self, features_df: pd.DataFrame):
        """
        Simula a explicação da previsão do modelo usando técnicas XAI (e.g., SHAP, LIME).
        Retorna as features mais influentes e seu impacto simulado.
        """
        # Em um ambiente real, você geraria os valores SHAP ou LIME para as features.
        # Ex:
        # explainer = shap.TreeExplainer(self.pd_model)
        # shap_values = explainer.shap_values(self.preprocessor.transform(features_df))
        # feature_names = features_df.columns
        # return dict(zip(feature_names, shap_values[0])) # Retorna a importância de cada feature

        # Para a simulação, retorna algumas contribuições fictícias baseadas na lógica de PD
        explanations = []
        
        income = features_df['monthly_income'].iloc[0]
        debt = features_df['total_debt'].iloc[0]
        restrictions = features_df['restrictions'].iloc[0]
        credit_util = features_df['credit_utilization'].iloc[0]
        account_age = features_df['account_age_months'].iloc[0]
        employment_stab = features_df['employment_stability_months'].iloc[0]

        if income < 1500: explanations.append(('Renda Mensal', 'Impacto Negativo Forte (baixa renda)'))
        elif income < 3000: explanations.append(('Renda Mensal', 'Impacto Negativo (renda moderada)'))
        else: explanations.append(('Renda Mensal', 'Impacto Positivo (alta renda)'))

        if restrictions: explanations.append(('Restrições (SPC/Serasa)', 'Impacto Negativo MUITO Forte'))
        else: explanations.append(('Restrições (SPC/Serasa)', 'Impacto Positivo (sem restrições)'))

        if credit_util > 80: explanations.append(('Utilização de Crédito', 'Impacto Negativo Forte (alta utilização)'))
        elif credit_util > 50: explanations.append(('Utilização de Crédito', 'Impacto Negativo (média utilização)'))
        else: explanations.append(('Utilização de Crédito', 'Impacto Positivo (baixa utilização)'))
        
        if account_age < 12: explanations.append(('Tempo de Relacionamento', 'Impacto Negativo (curto tempo)'))
        else: explanations.append(('Tempo de Relacionamento', 'Impacto Positivo (longo tempo)'))

        if employment_stab < 12: explanations.append(('Estabilidade Profissional', 'Impacto Negativo (pouca estabilidade)'))
        else: explanations.append(('Estabilidade Profissional', 'Impacto Positivo (boa estabilidade)'))

        # Retorna as 3 mais influentes (simulado)
        return explanations[:3] # Retorna uma lista de tuplas (feature, impacto)


class CreditScoreCalculator:
    """
    Calculadora de score de crédito baseada em múltiplos fatores.
    Carrega pesos e critérios de configuração externa para manutenibilidade.
    AGORA COM ML INTEGRADO (MOCK).
    """
    def __init__(self, config_data: dict):
        self.primary_weights = config_data['primary_weights']
        self.detailed_criteria = config_data['detailed_criteria_weights']
        self.segment_adjustments = config_data['segment_adjustments']
        self.ml_model = MockMLModel() # Instancia o modelo ML aqui
        logger.info("CreditScoreCalculator inicializado com configurações externas e MockMLModel.")

    @st.cache_data(ttl=3600)
    def calculate_comprehensive_score(_self, client_data: ClientData, segment: str = 'pessoa_fisica') -> dict:
        """
        Calcula o score de crédito completo, agora usando o modelo ML.
        Retorna um dicionário com score final e detalhamento (simplificado para ML).
        """
        try:
            features_df = client_data.to_dataframe_features()
            final_score_value = _self.ml_model.predict_score(features_df)
            
            # Com ML, o "detalhamento do score" por fator é uma interpretação ou explicação do modelo (XAI).
            # Aqui, simulamos que o modelo ainda tem alguma noção das categorias originais.
            simulated_detailed_scores = {
                'comportamento_pagamento': final_score_value / 10 + np.random.randint(-10, 10),
                'endividamento_atual': final_score_value / 10 + np.random.randint(-10, 10),
                'relacionamento_bancario': final_score_value / 10 + np.random.randint(-10, 10),
                'dados_demograficos': final_score_value / 10 + np.random.randint(-10, 10),
                'consultas_recentes': final_score_value / 10 + np.random.randint(-10, 10),
                'dados_cadastrais': final_score_value / 10 + np.random.randint(-10, 10),
            }
            simulated_detailed_scores = {k: max(0, min(100, int(v))) for k, v in simulated_detailed_scores.items()}

            simulated_contributions = {
                'comportamento_pagamento': simulated_detailed_scores['comportamento_pagamento'] * _self.primary_weights['comportamento_pagamento'],
                'endividamento_atual': simulated_detailed_scores['endividamento_atual'] * _self.primary_weights['endividamento_atual'],
                'relacionamento_bancario': simulated_detailed_scores['relacionamento_bancario'] * _self.primary_weights['relacionamento_bancario'],
                'dados_demograficos': simulated_detailed_scores['dados_demograficos'] * _self.primary_weights['dados_demograficos'],
                'consultas_recentes': simulated_detailed_scores['consultas_recentes'] * _self.primary_weights['consultas_recentes'],
                'dados_cadastrais': simulated_detailed_scores['dados_cadastrais'] * _self.primary_weights['dados_cadastrais']
            }

            logger.info(f"Score ML calculado para CPF {client_data.cpf}: {final_score_value}")
            return {
                'score_final': final_score_value,
                'score_detalhado': simulated_detailed_scores,
                'contribuicao_pesos': simulated_contributions
            }
        except Exception as e:
            logger.error(f"Erro ao calcular score abrangente com ML para CPF {client_data.cpf}: {e}", exc_info=True)
            return {
                'score_final': 500,
                'score_detalhado': {k: 50 for k in _self.detailed_criteria.keys()},
                'contribuicao_pesos': {k: 0 for k in _self.primary_weights.keys()}
            }

    # Métodos _calculate_factor_score e calculate_base_score não são mais a fonte primária do score
    # mas poderiam ser mantidos para fins de teste ou comparação com o modelo antigo.
    def _calculate_factor_score(self, client_data: ClientData, factor_name: str) -> float:
        """Mantido para compatibilidade conceitual, mas seria removido em implementação ML real."""
        score = 0.0
        try:
            criteria = self.detailed_criteria[factor_name]
            # ... (Sua lógica original de cálculo de fator. Mantida para preencher o score_detalhado simulado se necessário) ...
            if factor_name == 'comportamento_pagamento':
                spc_score = 0 if client_data.restrictions else 100
                score += spc_score * criteria['historico_spc_serasa']
                score += client_data.payment_history * criteria['pontualidade_12m']
                inadimplencia_score = max(0, 100 - client_data.historical_default_rate)
                score += inadimplencia_score * criteria['inadimplencia_historica']
                score += client_data.debt_regularization_speed * criteria['regularizacao_debitos']
                protesto_score = max(0, 100 - client_data.protests * 20)
                score += protesto_score * criteria['protestos_cartorio']
            elif factor_name == 'endividamento_atual':
                renda = client_data.monthly_income
                dividas = client_data.total_debt
                comprometimento = (dividas / renda * 100) if renda > 0 else 100
                comprometimento_score = max(0, 100 - float(comprometimento))
                score += comprometimento_score * criteria['comprometimento_renda']
                utilizacao_score = max(0, 100 - client_data.credit_utilization)
                score += utilizacao_score * criteria['utilizacao_limite_cartao']
                contratos = client_data.open_accounts
                if 2 <= contratos <= 5: contratos_score = 100
                elif contratos < 2: contratos_score = contratos * 40
                else: contratos_score = max(0, 100 - (contratos - 5) * 15)
                score += contratos_score * criteria['numero_contratos_ativos']
                if renda > 0:
                    ratio_divida = min((dividas / (renda * 12)) * 100, 100)
                    divida_score = max(0, 100 - ratio_divida)
                else: divida_score = 0
                score += divida_score * criteria['valor_total_dividas']
                concentracao_score = max(0, 100 - client_data.bank_debt_concentration)
                score += concentracao_score * criteria['divida_setor_bancario']
            elif factor_name == 'relacionamento_bancario':
                tempo_score = min((client_data.account_age_months / 120) * 100, 100)
                score += tempo_score * criteria['tempo_conta_corrente']
                movimentacao_score = min((client_data.monthly_turnover / client_data.monthly_income) * 100 / 3, 100) if client_data.monthly_income > 0 else 0
                score += movimentacao_score * criteria['movimentacao_financeira']
                produtos_score = min(client_data.bank_products_count * 25, 100)
                score += produtos_score * criteria['produtos_contratados']
                bancos = client_data.banks_relationship_count
                if 2 <= bancos <= 4: relacionamento_score = 100
                elif bancos == 1: relacionamento_score = 60
                else: relacionamento_score = max(0, 100 - (bancos - 4) * 20)
                score += relacionamento_score * criteria['relacionamento_multiplo']
                salario_score = 100 if client_data.has_salary_account else 40
                score += salario_score * criteria['conta_salario']
            elif factor_name == 'dados_demograficos':
                renda = client_data.monthly_income
                if renda >= 10000: renda_score = 100
                elif renda >= 5000: renda_score = 80
                elif renda >= 3000: renda_score = 60
                elif renda >= 1500: renda_score = 40
                else: renda_score = 20
                score += renda_score * criteria['renda_declarada']
                tempo_emprego = client_data.employment_stability_months
                if tempo_emprego >= 36: estabilidade_score = 100
                elif tempo_emprego >= 24: estabilidade_score = 80
                elif tempo_emprego >= 12: estabilidade_score = 60
                else: estabilidade_score = tempo_emprego * 5
                score += estabilidade_score * criteria['estabilidade_profissional']
                idade = client_data.age
                if 25 <= idade <= 55: idade_score = 100
                elif 18 <= idade <= 65: idade_score = 80
                else: idade_score = 40
                score += idade_score * criteria['faixa_etaria']
                escolaridade_score = min(client_data.education_level * 20, 100)
                score += escolaridade_score * criteria['escolaridade']
                civil_score = 100 if client_data.marital_status == 1 else 70
                score += civil_score * criteria['estado_civil']
            elif factor_name == 'consultas_recentes':
                consultas_30d_score = max(0, 100 - client_data.inquiries_30d * 25)
                score += consultas_30d_score * criteria['consultas_ultimos_30d']
                consultas_90d_score = max(0, 100 - client_data.inquiries_90d * 10)
                score += consultas_90d_score * criteria['consultas_ultimos_90d']
                auto_consultas = client_data.self_inquiries
                if 1 <= auto_consultas <= 4: auto_score = 100
                elif auto_consultas == 0: auto_score = 60
                else: auto_score = max(0, 100 - (auto_consultas - 4) * 20)
                score += auto_score * criteria['consultas_proprias']
            elif factor_name == 'dados_cadastrais':
                dias_atualizacao = client_data.days_since_update
                if dias_atualizacao <= 30: atualizacao_score = 100
                elif dias_atualizacao <= 90: atualizacao_score = 80
                elif dias_atualizacao <= 180: atualizacao_score = 60
                else: atualizacao_score = 30
                score += atualizacao_score * criteria['atualizacao_cadastral']
                score += client_data.data_consistency_score * criteria['consistencia_dados']
                telefone_score = min(client_data.validated_phones * 50, 100)
                score += telefone_score * criteria['telefones_validados']
                endereco_score = 100 if client_data.address_confirmed else 40
                score += endereco_score * criteria['endereco_confirmado']
            
            return score
        except Exception as e:
            logger.warning(f"Erro no cálculo do fator '{factor_name}' para CPF {client_data.cpf}: {e}. Retornando 0 para o fator.")
            return 0.0

class RiskAnalyzer:
    """
    Analisador de risco de crédito com foco em probabilidade de inadimplência (PD) e Perda Esperada (EL).
    AGORA COM ML INTEGRADO (MOCK) para cálculo de PD.
    """
    def __init__(self, lgd_config: dict):
        self.risk_matrix = { # Mantido fixo para simplicidade, mas poderia ser externo
            (0, 400): 'MUITO_ALTO', (401, 500): 'ALTO', (501, 650): 'MEDIO',
            (651, 750): 'BAIXO', (751, 850): 'MUITO_BAIXO', (851, 1000): 'MUITO_BAIXO'
        }
        if 'active_lgd' not in st.session_state:
            st.session_state.active_lgd = lgd_config.copy()
        self.active_lgd = st.session_state.active_lgd
        logger.info("RiskAnalyzer inicializado com configurações de LGD externas.")

    @st.cache_data(ttl=3600)
    def calculate_risk_level(_self, score: int) -> str:
        """Calcula o nível de risco com base no score."""
        try:
            for (min_s, max_s), risk in _self.risk_matrix.items():
                if min_s <= score <= max_s:
                    return risk
            return 'INDEFINIDO'
        except Exception as e:
            logger.error(f"Erro ao calcular nível de risco para score {score}: {e}", exc_info=True)
            return 'INDEFINIDO'

    @st.cache_data(ttl=3600)
    def calculate_default_probability(_self, client_data: ClientData, score: int) -> float:
        """
        Calcula a probabilidade de inadimplência (PD) AGORA USANDO O MODELO ML (MOCK).
        """
        try:
            # Reutiliza o MockMLModel já carregado no CreditScoreCalculator
            ml_model = st.session_state.score_calculator.ml_model 
            
            # 1. Preparar features do ClientData para o modelo ML
            features_df = client_data.to_dataframe_features()

            # 2. Prever PD usando o modelo ML
            pd_value = ml_model.predict_pd(features_df)

            logger.info(f"PD ML calculada para CPF {client_data.cpf}: {pd_value:.2%}")
            return max(0.001, min(pd_value, 0.999))
        except Exception as e:
            logger.error(f"Erro ao calcular PD com ML para CPF {client_data.cpf}: {e}", exc_info=True)
            return 1.0


    @st.cache_data(ttl=3600)
    def calculate_expected_loss(_self, probability_of_default: float, exposure_at_default: float, product_type: str = 'credito_pessoal') -> float:
        """
        Calcula a Perda Esperada (EL = PD * EAD * LGD).
        """
        try:
            lgd = _self.active_lgd.get(product_type, _self.active_lgd['credito_pessoal'])
            return probability_of_default * exposure_at_default * lgd
        except Exception as e:
            logger.error(f"Erro ao calcular EL para PD {probability_of_default}, EAD {exposure_at_default}: {e}", exc_info=True)
            return exposure_at_default

    def calculate_dynamic_limits_and_rates(_self, policy_data: dict, client_score: int, client_pd: float) -> tuple[dict, dict]:
        """
        Ajusta os limites e taxas com base no score e PD do cliente.
        """
        adjusted_limits = {}
        adjusted_rates = {}

        for product, max_limit in policy_data['limites'].items():
            score_range_start = policy_data['score_minimo']
            score_factor = (client_score - score_range_start) / (1000 - score_range_start + 1e-9)
            score_factor = max(0.0, min(score_factor, 1.0))

            min_limit_factor_for_policy = 0.3
            final_limit_percentage = min_limit_factor_for_policy + (score_factor * (1.0 - min_limit_factor_for_policy))
            
            adjusted_limits[product] = max_limit * final_limit_percentage
            
            if 'pessoal' in product or 'financiamento' in product:
                adjusted_limits[product] = round(adjusted_limits[product] / 1000) * 1000
            elif 'cartao' in product:
                adjusted_limits[product] = round(adjusted_limits[product] / 100) * 100
            adjusted_limits[product] = max(100.0, adjusted_limits[product])
        
        base_rate = policy_data['condicoes']['taxa_juros_base']
        pd_sensitivity_factor = 2.0 
        pd_threshold = policy_data['criterios'].get('probabilidade_inadimplencia_max', 0.10)
        pd_delta = client_pd - pd_threshold
        
        if pd_delta > 0:
            rate_increase_percentage = pd_delta * pd_sensitivity_factor
            adjusted_rates['taxa_juros_base'] = base_rate * (1 + rate_increase_percentage)
        else:
            adjusted_rates['taxa_juros_base'] = base_rate
        
        adjusted_rates['taxa_juros_base'] = max(0.5, min(adjusted_rates['taxa_juros_base'], 15.0))
        
        adjusted_rates['prazo_maximo_meses'] = policy_data['condicoes']['prazo_maximo_meses']
        adjusted_rates['carencia_permitida'] = policy_data['condicoes']['carencia_permitida']
        adjusted_rates['garantia_exigida'] = policy_data['condicoes']['garantia_exigida']

        return adjusted_limits, adjusted_rates


class CreditPolicyEngine:
    """
    Motor de políticas de crédito com critérios granulares e funcionalidades de teste.
    """
    def __init__(self, policies_config_json: str):
        self._default_policies = json.loads(policies_config_json)
        
        if 'active_policies' not in st.session_state:
            st.session_state.active_policies = self._default_policies.copy()
        self.policies = st.session_state.active_policies
        logger.info("CreditPolicyEngine inicializado com políticas externas.")

    def reset_policy_to_default(self, policy_key: str) -> bool:
        """Reseta uma política específica para seus valores padrão."""
        if policy_key in self._default_policies:
            st.session_state.active_policies[policy_key] = self._default_policies[policy_key].copy()
            self.policies = st.session_state.active_policies
            logger.info(f"Política '{policy_key}' resetada para o padrão.")
            return True
        logger.warning(f"Tentativa de resetar política '{policy_key}' que não existe nos padrões.")
        return False

    @st.cache_data(ttl=3600)
    def evaluate_comprehensive_policy(_self, client_data: ClientData, policy_name: str) -> dict:
        """
        Avalia um cliente contra uma política específica.
        """
        try:
            _self.policies = st.session_state.active_policies

            if policy_name not in _self.policies:
                logger.error(f"Política '{policy_name}' não encontrada no motor de políticas.")
                raise ValueError(f'Política {policy_name} não encontrada')

            policy = _self.policies[policy_name]
            criterios = policy['criterios']

            score_calculator = st.session_state.score_calculator
            risk_analyzer = st.session_state.risk_analyzer

            score_result = score_calculator.calculate_comprehensive_score(client_data)
            score_calculado = score_result['score_final']
            
            prob_inadimplencia = risk_analyzer.calculate_default_probability(client_data, score_calculado)

            evaluation = {
                'politica': policy['nome'],
                'score_calculado': score_calculado,
                'prob_inadimplencia': prob_inadimplencia,
                'aprovado': True,
                'restricoes_violadas': [],
                'alertas': [],
                'recomendacoes': [],
                'limites_aprovados': {},
                'condicoes_aplicadas': {},
                'detalhamento_score': score_result['score_detalhado'],
                'contribuicao_pesos': score_result['contribuicao_pesos'],
                'expected_loss_total': 0.0,
                'revisao_humana_sugerida': False
            }

            if score_calculado < policy['score_minimo']:
                evaluation['aprovado'] = False
                evaluation['restricoes_violadas'].append(f"Score ({score_calculado}) abaixo do mínimo exigido ({policy['score_minimo']})")

            if prob_inadimplencia > criterios.get('probabilidade_inadimplencia_max', 1.0):
                evaluation['aprovado'] = False
                evaluation['restricoes_violadas'].append(f"Probabilidade de inadimplência ({prob_inadimplencia:.1%}) acima do máximo permitido ({criterios['probabilidade_inadimplencia_max']:.1%})")

            if client_data.monthly_income < criterios['renda_minima']:
                evaluation['aprovado'] = False
                evaluation['restricoes_violadas'].append(f"Renda (R$ {client_data.monthly_income:,.2f}) abaixo do mínimo exigido (R$ {criterios['renda_minima']:,.2f})")
            if client_data.monthly_income <= 0:
                evaluation['alertas'].append("Renda mensal igual a zero/negativa. Cálculos baseados em renda podem ser imprecisos.")

            comprometimento = (client_data.total_debt / client_data.monthly_income * 100) if client_data.monthly_income > 0 else 100
            if float(comprometimento) > criterios['comprometimento_maximo']:
                evaluation['aprovado'] = False
                evaluation['restricoes_violadas'].append(f"Comprometimento de renda ({comprometimento:.1f}%) acima do limite permitido ({criterios['comprometimento_maximo']}%)")

            if client_data.account_age_months < criterios['tempo_relacionamento_minimo']:
                evaluation['aprovado'] = False
                evaluation['restricoes_violadas'].append(f"Tempo de relacionamento ({client_data.account_age_months} meses) inferior ao mínimo exigido ({criterios['tempo_relacionamento_minimo']} meses)")

            if client_data.restrictions and not criterios['restricoes_permitidas']:
                evaluation['aprovado'] = False
                evaluation['restricoes_violadas'].append("Cliente possui restrições (SPC/Serasa) - não permitidas por esta política")
            elif client_data.restrictions and criterios['restricoes_permitidas']:
                evaluation['alertas'].append("Cliente possui restrições, mas a política permite com cautela.")

            if client_data.inquiries_30d > criterios['consultas_30d_max']:
                evaluation['aprovado'] = False
                evaluation['restricoes_violadas'].append(f"Consultas nos últimos 30 dias ({client_data.inquiries_30d}) acima do máximo permitido ({criterios['consultas_30d_max']})")
            elif client_data.inquiries_30d > 0 and client_data.inquiries_30d == criterios['consultas_30d_max']:
                evaluation['alertas'].append(f"Número de consultas recentes está no limite máximo permitido.")

            if not (criterios['idade_minima'] <= client_data.age <= criterios['idade_maxima']):
                evaluation['aprovado'] = False
                evaluation['restricoes_violadas'].append(f"Idade ({client_data.age}) fora da faixa permitida ({criterios['idade_minima']}-{criterios['idade_maxima']} anos)")

            if client_data.employment_stability_months < criterios['estabilidade_emprego_minima']:
                evaluation['aprovado'] = False
                evaluation['restricoes_violadas'].append(f"Estabilidade de emprego ({client_data.employment_stability_months} meses) abaixo do mínimo exigido ({criterios['estabilidade_emprego_minima']} meses)")

            if client_data.credit_utilization > criterios['utilizacao_cartao_maxima']:
                evaluation['aprovado'] = False
                evaluation['restricoes_violadas'].append(f"Utilização de cartão ({client_data.credit_utilization}%) acima do máximo permitido ({criterios['utilizacao_cartao_maxima']}%)")
            
            if evaluation['aprovado']:
                adjusted_limits, adjusted_rates = risk_analyzer.calculate_dynamic_limits_and_rates(
                    policy, score_calculado, prob_inadimplencia
                )
                evaluation['limites_aprovados'] = adjusted_limits
                evaluation['condicoes_aplicadas'] = adjusted_rates

                total_expected_loss = 0.0
                for product_key, limit_amount in evaluation['limites_aprovados'].items():
                    lgd_product_type = product_key.replace('_max', '').replace('_', '')
                    el = risk_analyzer.calculate_expected_loss(prob_inadimplencia, limit_amount, lgd_product_type)
                    evaluation[f'expected_loss_{lgd_product_type}'] = el
                    total_expected_loss += el
                evaluation['expected_loss_total'] = total_expected_loss
                
                if (prob_inadimplencia > policy['criterios']['probabilidade_inadimplencia_max'] * 0.8 or
                    score_calculado < policy['score_minimo'] + 50):
                    evaluation['alertas'].append("Decisão de crédito limítrofe. Sugestão de revisão humana para análise aprofundada.")
                    evaluation['revisao_humana_sugerida'] = True
                
                logger.info(f"Cliente CPF {client_data.cpf} APROVADO para {policy['nome']}. EL Total: R$ {evaluation['expected_loss_total']:.2f}")

            else:
                evaluation['limites_aprovados'] = {}
                evaluation['condicoes_aplicadas'] = {}
                evaluation['expected_loss_total'] = 0.0
                if not evaluation['restricoes_violadas']:
                    evaluation['recomendacoes'].append("Cliente não atende a todos os critérios para esta política. Considere políticas com requisitos mais flexíveis.")
                evaluation['recomendacoes'].append("Sugestões para melhorar o perfil de crédito: Reduzir endividamento, pagar contas em dia, evitar novas consultas de crédito excessivas. Busque educação financeira.")
                logger.info(f"Cliente CPF {client_data.cpf} REPROVADO para {policy['nome']}.")

            return evaluation

        except ValueError as e:
            logger.error(f"Erro de configuração ou dados inválidos na avaliação da política para CPF {client_data.cpf}: {e}", exc_info=True)
            return {
                'politica': 'Erro de Configuração de Política', 'score_calculado': 0, 'prob_inadimplencia': 1.0,
                'aprovado': False, 'restricoes_violadas': [str(e)], 'alertas': ["Erro interno na política. Contate o suporte."],
                'recomendacoes': [], 'limites_aprovados': {}, 'condicoes_aplicadas': {},
                'detalhamento_score': {}, 'contribuicao_pesos': {}, 'expected_loss_total': 0.0,
                'revisao_humana_sugerida': True
            }
        except Exception as e:
            logger.critical(f"Erro INESPERADO ao avaliar a política '{policy_name}' para CPF {client_data.cpf}: {e}", exc_info=True)
            return {
                'politica': 'Erro Inesperado na Avaliação', 'score_calculado': 0, 'prob_inadimplencia': 1.0,
                'aprovado': False, 'restricoes_violadas': [f"Erro interno: {e}"],
                'alertas': ["Erro inesperado na avaliação da política. Contate o suporte."], 'recomendacoes': [],
                'limites_aprovados': {}, 'condicoes_aplicadas': {}, 'detalhamento_score': {},
                'contribuicao_pesos': {}, 'expected_loss_total': 0.0,
                'revisao_humana_sugerida': True
            }


    @st.cache_data(ttl=3600)
    def find_best_policy(_self, client_data: ClientData) -> dict:
        """
        Encontra a melhor política de crédito para o cliente, avaliando sequencialmente.
        """
        try:
            _self.policies = st.session_state.active_policies
            policy_order = ['prime_plus', 'prime', 'standard', 'risk_based', 'microcrédito']
            best_policy_found = None
            all_policy_evaluations = {}

            for policy_name in policy_order:
                policy_evaluation = _self.evaluate_comprehensive_policy(client_data, policy_name)
                all_policy_evaluations[policy_name] = policy_evaluation

                if policy_evaluation.get('error'):
                    logger.warning(f"Falha na avaliação de política '{policy_name}' para CPF {client_data.cpf}: {policy_evaluation.get('restricoes_violadas', ['Desconhecido'])[0]}")
                    continue

                if policy_evaluation['aprovado']:
                    best_policy_found = policy_evaluation
                    logger.info(f"Melhor política encontrada para CPF {client_data.cpf}: {policy_name}")
                    break

            if best_policy_found:
                best_policy_found['all_policy_evaluations'] = all_policy_evaluations
                return best_policy_found
            else:
                score_calculator = st.session_state.score_calculator
                score_result = score_calculator.calculate_comprehensive_score(client_data)

                risk_analyzer = st.session_state.risk_analyzer
                prob_inadimplencia = risk_analyzer.calculate_default_probability(client_data, score_result['score_final'])

                final_result = {
                    'politica': 'Nenhuma Política Aplicável', 'score_calculado': score_result['score_final'],
                    'prob_inadimplencia': prob_inadimplencia, 'aprovado': False,
                    'restricoes_violadas': ['Nenhuma política de crédito disponível atende aos critérios do cliente.'],
                    'alertas': [], 'recomendacoes': ['Considere aprimorar seu perfil financeiro (reduzir dívidas, aumentar renda, manter bom histórico de pagamentos).'],
                    'limites_aprovados': {}, 'condicoes_aplicadas': {},
                    'detalhamento_score': score_result['score_detalhado'],
                    'contribuicao_pesos': score_result['contribuicao_pesos'], 'expected_loss_total': 0.0,
                    'revisao_humana_sugerida': True,
                    'all_policy_evaluations': all_policy_evaluations
                }
                logger.info(f"CPF {client_data.cpf} não se qualificou para nenhuma política.")
                return final_result
        except Exception as e:
            logger.critical(f"Erro crítico ao encontrar a melhor política para CPF {client_data.cpf}: {e}", exc_info=True)
            return {
                'politica': 'Falha na Seleção de Política', 'score_calculado': 0, 'prob_inadimplencia': 1.0,
                'aprovado': False, 'restricoes_violadas': [f"Erro interno: {e}"],
                'alertas': ['Não foi possível determinar uma política devido a um erro interno. Contate o suporte.'],
                'recomendacoes': [], 'limites_aprovados': {}, 'condicoes_aplicadas': {},
                'detalhamento_score': {}, 'contribuicao_pesos': {}, 'expected_loss_total': 0.0,
                'revisao_humana_sugerida': True,
                'all_policy_evaluations': {}
            }

class ExternalAPIIntegrator:
    """
    Simula a integração com APIs externas de consulta de crédito.
    """
    def __init__(self):
        self.endpoints = {
            'boa_vista': 'https://api.boavista.com.br/v1/consulta',
            'spc': 'https://api.spc.org.br/v2/consulta',
            'serasa': 'https://api.serasa.com.br/v1/score'
        }
        logger.info("ExternalAPIIntegrator inicializado.")

    @st.cache_data(ttl=3600)
    def mock_boa_vista_response(_self, cpf: str) -> ClientData:
        """
        Simula uma resposta de API externa, retornando um objeto ClientData.
        """
        try:
            monthly_income = float(np.random.normal(loc=5000, scale=3000))
            monthly_income = max(800.0, round(monthly_income, -2))
            
            total_debt = float(np.random.normal(loc=monthly_income*0.5, scale=monthly_income*1.5))
            total_debt = max(0.0, round(total_debt, -2))
            
            credit_utilization = int(np.random.normal(loc=40, scale=25))
            credit_utilization = max(0, min(100, credit_utilization))
            
            account_age_months = int(np.random.normal(loc=36, scale=24))
            account_age_months = max(1, min(120, account_age_months))
            
            employment_stability_months = int(np.random.normal(loc=24, scale=18))
            employment_stability_months = max(0, min(120, employment_stability_months))

            age = int(np.random.normal(loc=38, scale=10))
            age = max(18, min(70, age))

            restrictions = np.random.choice([True, False], p=[0.2, 0.8])
            default_event_chance = (
                0.01 +
                (total_debt / (monthly_income + 1) * 0.05) +
                (credit_utilization / 100 * 0.1) +
                (0.1 if restrictions else 0)
            )
            default_event = np.random.rand() < default_event_chance

            average_monthly_transactions = float(np.random.normal(loc=monthly_income*0.8, scale=monthly_income*0.3))
            average_monthly_transactions = max(0.0, round(average_monthly_transactions, -2))
            investment_balance = float(np.random.normal(loc=monthly_income*5, scale=monthly_income*10))
            investment_balance = max(0.0, round(investment_balance, -2))
            utility_bill_on_time_payment_ratio = float(np.random.beta(a=5, b=1) if np.random.rand() < 0.9 else np.random.beta(a=1, b=5))
            utility_bill_on_time_payment_ratio = max(0.0, min(1.0, utility_bill_on_time_payment_ratio))


            client_data_dict = {
                'name': f'Cliente Simulado {cpf[-4:]}',
                'cpf': cpf,
                'monthly_income': monthly_income,
                'total_debt': total_debt,
                'account_age_months': account_age_months,
                'restrictions': restrictions,
                'inquiries_30d': np.random.randint(0, 5),
                'age': age,
                'employment_stability_months': employment_stability_months,
                'credit_utilization': credit_utilization,
                'payment_history': np.random.randint(60, 100),
                'historical_default_rate': np.random.randint(0, 20),
                'debt_regularization_speed': np.random.randint(30, 100),
                'protests': np.random.randint(0, 2),
                'open_accounts': np.random.randint(1, 15),
                'bank_debt_concentration': np.random.randint(20, 90),
                'monthly_turnover': float(np.random.normal(loc=monthly_income * 1.5, scale=monthly_income * 0.5)),
                'bank_products_count': np.random.randint(1, 5),
                'banks_relationship_count': np.random.randint(1, 5),
                'has_salary_account': np.random.choice([True, False]),
                'days_since_update': np.random.randint(0, 365),
                'data_consistency_score': np.random.randint(50, 100),
                'validated_phones': np.random.randint(0, 3),
                'address_confirmed': np.random.choice([True, False]),
                'inquiries_90d': np.random.randint(0, 10),
                'self_inquiries': np.random.randint(0, 2),
                'education_level': np.random.randint(1, 5),
                'marital_status': np.random.choice([0, 1]),
                'average_monthly_transactions': average_monthly_transactions,
                'investment_balance': investment_balance,
                'utility_bill_on_time_payment_ratio': utility_bill_on_time_payment_ratio,
                'default_event': default_event
            }
            logger.info(f"Dados simulados gerados para CPF {cpf}.")
            return ClientData(**client_data_dict)
        except Exception as e:
            logger.error(f"Erro ao simular dados para CPF {cpf}: {e}", exc_info=True)
            return ClientData(
                name=f'ERRO_{cpf[-4:]}', cpf=cpf, monthly_income=0.0, total_debt=100000.0,
                account_age_months=0, restrictions=True, inquiries_30d=10, age=18,
                employment_stability_months=0, credit_utilization=100, payment_history=0,
                historical_default_rate=100, debt_regularization_speed=0, protests=5,
                open_accounts=0, bank_debt_concentration=100, monthly_turnover=0.0,
                bank_products_count=0, banks_relationship_count=0, has_salary_account=False,
                education_level=1, marital_status=0, inquiries_90d=20, self_inquiries=5,
                days_since_update=365, data_consistency_score=0, validated_phones=0,
                address_confirmed=False, default_event=True
            )

# --- Funções de Análise de Cenários e Portfólio ---

@st.cache_data(ttl=3600)
def generate_simulated_client_data_for_portfolio(num_clients: int = 1000) -> pd.DataFrame:
    """
    Gera um DataFrame com dados de clientes simulados, utilizando ExternalAPIIntegrator
    para garantir consistência e incluir o 'default_event'.
    """
    logger.info(f"Gerando {num_clients} clientes simulados para portfólio...")
    integrator = ExternalAPIIntegrator()
    data_list = []
    for i in range(num_clients):
        cpf = f"CPF_SIM_{i+1:06d}"
        client_data_obj = integrator.mock_boa_vista_response(cpf)
        data_list.append(client_data_obj.to_dict())
    
    df = pd.DataFrame(data_list)
    logger.info(f"Geração de portfólio concluída. Amostra:\n{df.head()}")
    return df

@st.cache_data(ttl=3600)
def analyze_policy_performance(policy_engine: CreditPolicyEngine, clients_df: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Analisa a performance das políticas de crédito contra um DataFrame de clientes.
    Retorna métricas e o DataFrame com resultados detalhados.
    """
    logger.info(f"Iniciando análise de performance para {len(clients_df)} clientes.")
    results = []
    
    for index, row_series in clients_df.iterrows():
        try:
            client_dict = row_series.to_dict()
            for field_name in ClientData.__annotations__.keys():
                if field_name not in client_dict:
                    client_dict[field_name] = None
            client_data_obj = ClientData(**client_dict)
            
            policy_eval = policy_engine.find_best_policy(client_data_obj)
            
            el_personal = policy_eval.get('expected_loss_credito_pessoal', 0.0)
            el_cartao = policy_eval.get('expected_loss_cartao_credito', 0.0)
            el_financiamento = policy_eval.get('expected_loss_financiamento', 0.0)
            el_micro = policy_eval.get('expected_loss_microcredito', 0.0)

            results.append({
                'CPF': client_data_obj.cpf,
                'Nome': client_data_obj.name,
                'Renda Mensal': client_data_obj.monthly_income,
                'Dívida Total': client_data_obj.total_debt,
                'Score Calculado': policy_eval['score_calculado'],
                'Prob. Inadimplência': policy_eval['prob_inadimplencia'],
                'Política Aplicada': policy_eval['politica'],
                'Aprovado': policy_eval['aprovado'],
                'Limites Aprovados': json.dumps(policy_eval['limites_aprovados']),
                'Taxa Juros Base Aprovada': policy_eval['condicoes_aplicadas'].get('taxa_juros_base', np.nan),
                'Expected Loss Total': policy_eval['expected_loss_total'],
                'Revisão Humana Sugerida': policy_eval.get('revisao_humana_sugerida', False),
                'Default Event (Simulado)': client_data_obj.default_event,
                'Restrições Violadas': ", ".join(policy_eval['restricoes_violadas']) if policy_eval['restricoes_violadas'] else "Nenhuma",
                'Alertas': ", ".join(policy_eval['alertas']) if policy_eval['alertas'] else "Nenhum",
                'EL_Pessoal': el_personal, 'EL_Cartao': el_cartao, 'EL_Financiamento': el_financiamento, 'EL_Microcredito': el_micro
            })
        except Exception as e:
            logger.error(f"Erro ao processar cliente {row_series.get('cpf', 'N/A')} no portfólio: {e}", exc_info=True)
            results.append({
                'CPF': row_series.get('cpf', 'ERRO'), 'Nome': row_series.get('name', 'ERRO'),
                'Renda Mensal': row_series.get('monthly_income', 0), 'Dívida Total': row_series.get('total_debt', 0),
                'Score Calculado': 0, 'Prob. Inadimplência': 1.0, 'Política Aplicada': 'ERRO',
                'Aprovado': False, 'Limites Aprovados': '{}', 'Taxa Juros Base Aprovada': np.nan,
                'Expected Loss Total': np.nan, 'Revisão Humana Sugerida': True, 'Default Event (Simulado)': True,
                'Restrições Violadas': 'Erro Interno', 'Alertas': 'Processamento Falhou',
                'EL_Pessoal': np.nan, 'EL_Cartao': np.nan, 'EL_Financiamento': np.nan, 'EL_Microcredito': np.nan
            })
    
    results_df = pd.DataFrame(results)
    logger.info(f"Análise de portfólio concluída. Total de resultados: {len(results_df)}")

    metrics = {}
    total_clients = len(results_df)
    
    metrics['Taxa de Aprovação Geral'] = (results_df['Aprovado'].sum() / total_clients) if total_clients > 0 else 0
    
    approved_df = results_df[results_df['Aprovado']]
    metrics['Score Médio (Aprovados)'] = approved_df['Score Calculado'].mean() if not approved_df.empty else 0
    metrics['Prob. Inadimplência Média (Aprovados)'] = approved_df['Prob. Inadimplência'].mean() if not approved_df.empty else 0
    metrics['Expected Loss Total (Aprovados)'] = approved_df['Expected Loss Total'].sum() if not approved_df.empty else 0
    metrics['Expected Loss Médio por Cliente (Aprovados)'] = approved_df['Expected Loss Total'].mean() if not approved_df.empty else 0
    
    metrics['Taxa de Revisão Humana'] = (results_df['Revisão Humana Sugerida'].sum() / total_clients) if total_clients > 0 else 0

    policy_approval_counts = results_df.groupby('Política Aplicada')['Aprovado'].value_counts().unstack(fill_value=0)
    policy_approval_rates_df = pd.DataFrame(index=policy_approval_counts.index)
    policy_approval_rates_df['Total Clientes na Política'] = policy_approval_counts.sum(axis=1)
    policy_approval_rates_df['Aprovados'] = policy_approval_counts.get(True, 0)
    policy_approval_rates_df['Taxa de Aprovação'] = (policy_approval_rates_df['Aprovados'] / policy_approval_rates_df['Total Clientes na Política']).fillna(0)
    
    policy_el_means = approved_df.groupby('Política Aplicada')['Expected Loss Total'].mean()
    policy_approval_rates_df['EL Médio Aprovado'] = policy_el_means
    
    logger.info("Métricas de performance calculadas.")
    return results_df, metrics, policy_approval_rates_df

# --- INTERFACE STREAMLIT ---

def main():
    st.set_page_config(
        page_title="Sistema de Análise de Crédito FICO-Like",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-card-green {
        background: linear-gradient(90deg, #28a745 0%, #218838 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-card-red {
        background: linear-gradient(90deg, #dc3545 0%, #c82333 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #ddd;
        padding: 0.5rem;
    }
    .stSelectbox>div>div {
        border-radius: 8px;
        border: 1px solid #ddd;
    }
    .risk-level-MUITO_ALTO { background-color: #dc3545; color: white; padding: 5px; border-radius: 5px; }
    .risk-level-ALTO { background-color: #fd7e14; color: white; padding: 5px; border-radius: 5px; }
    .risk-level-MEDIO { background-color: #ffc107; color: black; padding: 5px; border-radius: 5px; }
    .risk-level-BAIXO { background-color: #28a745; color: white; padding: 5px; border-radius: 5px; }
    .risk-level-MUITO_BAIXO { background-color: #17a2b8; color: white; padding: 5px; border-radius: 5px; }
    .info-button {
        background: none;
        border: none;
        padding: 0;
        margin-left: 5px;
        cursor: pointer;
        font-size: 0.9em;
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🏦 Sistema de Análise de Crédito <small>(FICO-Like)</small></h1>', unsafe_allow_html=True)

    # Inicializar objetos com st.session_state para manter o estado e carregar configurações
    if 'score_calculator' not in st.session_state:
        st.session_state.score_calculator = CreditScoreCalculator(json.loads(SCORE_WEIGHTS_CONFIG_JSON))
    
    if 'risk_analyzer' not in st.session_state:
        st.session_state.risk_analyzer = RiskAnalyzer(json.loads(LGD_PARAMS_CONFIG_JSON))
    
    if 'policy_engine' not in st.session_state:
        st.session_state.policy_engine = CreditPolicyEngine(POLICIES_CONFIG_JSON)
    
    if 'api_integrator' not in st.session_state:
        st.session_state.api_integrator = ExternalAPIIntegrator()

    # --- UI DA SIDEBAR ---
    st.sidebar.header("📝 Dados do Cliente")
    analysis_mode = st.sidebar.radio(
        "Modo de Análise",
        ["Análise Individual de Cliente", "Análise de Portfólio (Simulação)"],
        key="analysis_mode_selector",
        help="**Análise Individual:** Avalia um único cliente com dados manuais ou simulados. **Análise de Portfólio:** Simula a aplicação das políticas em um grande volume de clientes para análise estratégica de risco e performance."
    )

    # Campos de entrada para análise individual
    if analysis_mode == "Análise Individual de Cliente":
        client_name = st.sidebar.text_input("Nome do Cliente", "João Silva", help="Nome do cliente para identificação.")
        client_cpf = st.sidebar.text_input("CPF do Cliente", "123.456.789-00", help="CPF fictício do cliente para a análise.")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("🔑 **Chaves de API (Opcional - Futura Implementação)**")
        st.sidebar.text_input("Chave API Boa Vista", type="password", disabled=True, help="Campo para futura integração com API real.")
        st.sidebar.text_input("Chave API SPC", type="password", disabled=True, help="Campo para futura integração com API real.")
        st.sidebar.text_input("Chave API Serasa", type="password", disabled=True, help="Campo para futura integração com API real.")
        
        simulate_data = st.sidebar.checkbox("Simular dados do cliente?", value=True, help="Marque para usar dados gerados aleatoriamente; desmarque para inserção manual.")

        raw_client_data_dict = {}
        if simulate_data:
            st.sidebar.info("Dados de crédito serão simulados para este CPF.")
            raw_client_data_dict = st.session_state.api_integrator.mock_boa_vista_response(client_cpf).to_dict()
            raw_client_data_dict['name'] = client_name

        else:
            st.sidebar.warning("A integração com APIs reais **não** está implementada nesta versão. Por favor, use a simulação ou insira os dados manualmente.")
            raw_client_data_dict = { 
                'name': client_name, 'cpf': client_cpf, 'monthly_income': 5000.0, 'total_debt': 2000.0,
                'account_age_months': 24, 'restrictions': False, 'inquiries_30d': 1, 'age': 35,
                'employment_stability_months': 12, 'credit_utilization': 30, 'payment_history': 90,
                'historical_default_rate': 5, 'debt_regularization_speed': 70, 'protests': 0,
                'open_accounts': 5, 'bank_debt_concentration': 70, 'monthly_turnover': 8000.0,
                'bank_products_count': 3, 'banks_relationship_count': 2, 'has_salary_account': True,
                'education_level': 4, 'marital_status': 1, 'inquiries_90d': 3, 'self_inquiries': 0,
                'days_since_update': 30, 'data_consistency_score': 90, 'validated_phones': 1, 'address_confirmed': True,
                'average_monthly_transactions': 0.0, 'investment_balance': 0.0, 'utility_bill_on_time_payment_ratio': 1.0, 'default_event': False
            }

        st.sidebar.markdown("---")
        st.sidebar.header("📝 Entrada Manual de Dados (Sobrescreve Simulação)")
        for field_name, field_type in ClientData.__annotations__.items():
            if field_name in ['name', 'cpf', 'default_event']: continue

            display_name = field_name.replace('_', ' ').title()
            current_value = raw_client_data_dict.get(field_name)
            
            help_text = f"Valor de {display_name.lower()}."
            if 'income' in field_name: help_text = "Renda mensal declarada do cliente."
            elif 'debt' in field_name: help_text = "Dívida total atual do cliente."
            elif 'age' in field_name: help_text = "Idade do cliente."
            elif 'restrictions' in field_name: help_text = "Se o cliente possui restrições financeiras (SPC/Serasa)."
            elif 'credit_utilization' in field_name: help_text = "Percentual de utilização do limite do cartão de crédito."
            elif 'utility_bill_on_time_payment_ratio' in field_name: help_text = "Proporção de contas de consumo pagas em dia (simulando dados de Open Finance)."


            if field_name == 'education_level':
                edu_options = {1:'Fundamental', 2:'Médio', 3:'Superior Incompleto', 4:'Superior Completo', 5:'Pós-graduação'}
                current_value = st.sidebar.selectbox("Escolaridade", options=list(edu_options.keys()), format_func=lambda x: edu_options[x], index=list(edu_options.keys()).index(current_value), key=f"manual_{field_name}", help=help_text)
            elif field_name == 'marital_status':
                marital_options = {0:'Solteiro', 1:'Casado/União Estável'}
                current_value = st.sidebar.selectbox("Estado Civil", options=list(marital_options.keys()), format_func=lambda x: marital_options[x], index=list(marital_options.keys()).index(current_value), key=f"manual_{field_name}", help=help_text)
            elif field_type == bool:
                current_value = st.sidebar.checkbox(display_name, value=bool(current_value), key=f"manual_{field_name}", help=help_text)
            elif field_type == int:
                max_val = 120 if 'months' in field_name else (100 if ('rate' in field_name or 'concentration' in field_name or 'score' in field_name or 'utilization' in field_name or 'history' in field_name) else (90 if 'age' in field_name else 20))
                current_value = st.sidebar.slider(display_name, min_value=0, max_value=max_val, value=int(current_value), key=f"manual_{field_name}", help=help_text)
            elif field_type == float:
                prefix = "R$ " if 'income' in field_name or 'debt' in field_name or 'turnover' in field_name or 'balance' in field_name else ""
                format_str = prefix + "%.2f" # CORREÇÃO APLICADA AQUI

                step_val = 500.0 if 'income' in field_name or 'debt' in field_name or 'turnover' in field_name or 'balance' in field_name else 0.01

                current_value = st.sidebar.number_input(
                    display_name, 
                    min_value=0.0, 
                    value=float(current_value), 
                    step=step_val, 
                    format=format_str, # CORREÇÃO APLICADA AQUI
                    key=f"manual_{field_name}", 
                    help=help_text
                )
            
            raw_client_data_dict[field_name] = current_value
        
        try:
            client_data = ClientData(**raw_client_data_dict)
        except Exception as e:
            st.error(f"Erro na validação dos dados do cliente. Verifique se todos os campos estão preenchidos corretamente: {e}")
            logger.error(f"Erro na criação de ClientData: {e}", exc_info=True)
            st.stop()

            
    else: # analysis_mode == "Análise de Portfólio (Simulação)"
        st.sidebar.info("Modo de Análise de Portfólio: Os dados dos clientes serão simulados em lote para testar as políticas.")
        num_simulated_clients = st.sidebar.slider("Número de Clientes Simulados", min_value=100, max_value=10000, value=1000, step=100, help="Define o volume de clientes a serem simulados para a análise de portfólio.")
        
        if st.sidebar.button("Gerar Novo Portfólio Simulado", key="generate_portfolio_btn"):
            st.cache_data.clear()
            st.session_state.score_calculator = CreditScoreCalculator(json.loads(SCORE_WEIGHTS_CONFIG_JSON))
            st.session_state.risk_analyzer = RiskAnalyzer(json.loads(LGD_PARAMS_CONFIG_JSON))
            st.session_state.policy_engine = CreditPolicyEngine(POLICIES_CONFIG_JSON)
            st.session_state.api_integrator = ExternalAPIIntegrator()

            st.session_state['simulated_clients_df'] = generate_simulated_client_data_for_portfolio(num_simulated_clients)
            st.sidebar.success(f"{num_simulated_clients} clientes simulados gerados e configurações resetadas para o padrão.")
            st.rerun()

        if 'simulated_clients_df' not in st.session_state:
            st.session_state['simulated_clients_df'] = generate_simulated_client_data_for_portfolio(num_simulated_clients)
            
        st.sidebar.dataframe(st.session_state['simulated_clients_df'].head(), use_container_width=True, help="Amostra dos clientes simulados no portfólio.")


    st.sidebar.markdown("---")
    st.sidebar.write("⚙️ **Configurações Gerais**")
    segment_type = st.sidebar.selectbox(
        "Segmento do Cliente Padrão",
        ["pessoa_fisica", "pessoa_juridica", "microempresario"],
        format_func=lambda x: x.replace('_', ' ').title(),
        key="segment_type",
        help="Segmento padrão usado para ajustes de score se o cliente não tiver um segmentação explícita."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("💸 Ajustes de Perda por Inadimplência (LGD)")
    st.session_state.risk_analyzer.active_lgd['credito_pessoal'] = st.sidebar.slider(
        "LGD Crédito Pessoal (%)", 0.0, 1.0, st.session_state.risk_analyzer.active_lgd['credito_pessoal'], 0.01,
        format="%.2f", key="lgd_pessoal", help=CRITERIA_EXPLANATIONS['lgd_credito_pessoal']
    )
    st.session_state.risk_analyzer.active_lgd['cartao_credito'] = st.sidebar.slider(
        "LGD Cartão de Crédito (%)", 0.0, 1.0, st.session_state.risk_analyzer.active_lgd['cartao_credito'], 0.01,
        format="%.2f", key="lgd_cartao", help=CRITERIA_EXPLANATIONS['lgd_cartao_credito']
    )
    st.session_state.risk_analyzer.active_lgd['financiamento'] = st.sidebar.slider(
        "LGD Financiamento (%)", 0.0, 1.0, st.session_state.risk_analyzer.active_lgd['financiamento'], 0.01,
        format="%.2f", key="lgd_financiamento", help=CRITERIA_EXPLANATIONS['lgd_financiamento']
    )
    st.session_state.risk_analyzer.active_lgd['microcredito'] = st.sidebar.slider(
        "LGD Microcrédito (%)", 0.0, 1.0, st.session_state.risk_analyzer.active_lgd['microcredito'], 0.01,
        format="%.2f", key="lgd_microcredito", help=CRITERIA_EXPLANATIONS['lgd_microcredito']
    )

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Ajuste Granular de Políticas")
    
    policy_keys = list(st.session_state.policy_engine.policies.keys())
    policy_display_names = [st.session_state.policy_engine.policies[key]['nome'] for key in policy_keys]

    selected_policy_display_name = st.sidebar.selectbox(
        "Selecionar Política para Ajustar", 
        policy_display_names,
        key="policy_selector",
        help="Escolha uma política para visualizar e ajustar seus critérios internos."
    )
    selected_policy_key = policy_keys[policy_display_names.index(selected_policy_display_name)]


    with st.sidebar.expander(f"Ajustar Critérios de '{st.session_state.policy_engine.policies[selected_policy_key]['nome']}'", expanded=True):
        current_policy_details = st.session_state.policy_engine.policies[selected_policy_key]
        
        col_s_min, col_s_min_info = st.columns([0.8, 0.2])
        with col_s_min:
            current_policy_details['score_minimo'] = st.number_input(
                "Score Mínimo da Política",
                min_value=0, max_value=1000, 
                value=int(current_policy_details['score_minimo']),
                step=10, 
                key=f"policy_score_min_{selected_policy_key}"
            )
        with col_s_min_info:
            if st.button("ℹ️", key=f"info_score_min_{selected_policy_key}", help=CRITERIA_EXPLANATIONS['score_minimo']):
                st.info(CRITERIA_EXPLANATIONS['score_minimo'])

        col_test_type, col_test_type_info = st.columns([0.8, 0.2])
        with col_test_type:
            current_policy_details['tipo_teste'] = st.selectbox(
                "Tipo de Teste (Champion/Challenger)",
                options=[None, "Champion", "Challenger"],
                index=[None, "Champion", "Challenger"].index(current_policy_details.get('tipo_teste')),
                key=f"policy_test_type_{selected_policy_key}",
            )
        with col_test_type_info:
            if st.button("ℹ️", key=f"info_test_type_{selected_policy_key}", help=CRITERIA_EXPLANATIONS['tipo_teste']):
                st.info(CRITERIA_EXPLANATIONS['tipo_teste'])

        st.markdown("---")
        st.write("**Critérios da Política:**")
        for criterion_key, criterion_value in current_policy_details['criterios'].items():
            display_name = criterion_key.replace('_', ' ').title()
            
            col_crit, col_crit_info = st.columns([0.8, 0.2])
            with col_crit:
                if isinstance(criterion_value, bool):
                    current_policy_details['criterios'][criterion_key] = st.checkbox(
                        display_name, value=bool(criterion_value), key=f"crit_{criterion_key}_{selected_policy_key}"
                    )
                elif isinstance(criterion_value, (int, float)):
                    min_val, max_val, step = 0.0, 1000000.0, 1.0
                    if 'renda_minima' in criterion_key:
                        min_val, max_val, step = 0.0, 1000000.0, 500.0
                    elif 'comprometimento_maximo' in criterion_key or 'utilizacao_cartao_maxima' in criterion_key:
                        min_val, max_val, step = 0.0, 100.0, 1.0
                    elif 'tempo_relacionamento_minimo' in criterion_key or 'estabilidade_emprego_minima' in criterion_key:
                        min_val, max_val, step = 0.0, 120.0, 1.0
                    elif 'consultas_30d_max' in criterion_key:
                        min_val, max_val, step = 0.0, 10.0, 1.0
                    elif 'idade' in criterion_key:
                        min_val, max_val, step = 0.0, 100.0, 1.0
                    elif 'probabilidade_inadimplencia_max' in criterion_key:
                        min_val, max_val, step = 0.0, 1.0, 0.01

                    if max_val <= 100.0 and step == 1.0:
                        current_policy_details['criterios'][criterion_key] = st.slider(
                            display_name, min_value=int(min_val), max_value=int(max_val), value=int(criterion_value), key=f"crit_{criterion_key}_{selected_policy_key}"
                        )
                    elif max_val <= 1.0 and step == 0.01:
                         current_policy_details['criterios'][criterion_key] = st.slider(
                            display_name, min_value=min_val, max_value=max_val, value=float(criterion_value), step=step, format="%.2f", key=f"crit_{criterion_key}_{selected_policy_key}"
                        )
                    else:
                        current_policy_details['criterios'][criterion_key] = st.number_input(
                            display_name, min_value=min_val, max_value=max_val, value=float(criterion_value), step=step, key=f"crit_{criterion_key}_{selected_policy_key}"
                        )
                else:
                    current_policy_details['criterios'][criterion_key] = st.text_input(
                        display_name, value=str(criterion_value), key=f"crit_{criterion_key}_{selected_policy_key}"
                    )
            with col_crit_info:
                explanation = CRITERIA_EXPLANATIONS.get('comportamento_pagamento', {}).get('sub_criterios', {}).get(criterion_key) or \
                              CRITERIA_EXPLANATIONS.get('endividamento_atual', {}).get('sub_criterios', {}).get(criterion_key) or \
                              CRITERIA_EXPLANATIONS.get('relacionamento_bancario', {}).get('sub_criterios', {}).get(criterion_key) or \
                              CRITERIA_EXPLANATIONS.get('dados_demograficos', {}).get('sub_criterios', {}).get(criterion_key) or \
                              CRITERIA_EXPLANATIONS.get('consultas_recentes', {}).get('sub_criterios', {}).get(criterion_key) or \
                              CRITERIA_EXPLANATIONS.get('dados_cadastrais', {}).get('sub_criterios', {}).get(criterion_key) or \
                              CRITERIA_EXPLANATIONS.get(criterion_key)
                if explanation:
                    help_text = explanation['racional'] if isinstance(explanation, dict) else explanation
                    if st.button("ℹ️", key=f"info_crit_{criterion_key}_{selected_policy_key}", help=help_text):
                        if isinstance(explanation, dict):
                            st.info(f"**{explanation['nome']}**: {explanation['racional']}")
                            if explanation.get('sub_criterios'):
                                st.markdown("**Fatores internos detalhados:**")
                                for sc_key, sc_val in explanation['sub_criterios'].items():
                                    st.markdown(f"- **{sc_key.replace('_', ' ').title()}:** {sc_val}")
                        else:
                            st.info(explanation)
                else:
                    st.empty()


        st.markdown("---")
        st.write("**Limites de Crédito Máximos da Política:**")
        for limit_key, limit_value in current_policy_details['limites'].items():
            display_name = limit_key.replace('_', ' ').title()
            col_limit, col_limit_info = st.columns([0.8, 0.2])
            with col_limit:
                current_policy_details['limites'][limit_key] = st.number_input(
                    display_name, min_value=0.0, value=float(limit_value), step=1000.0, key=f"limit_{limit_key}_{selected_policy_key}"
                )
            with col_limit_info:
                if st.button("ℹ️", key=f"info_limit_{limit_key}_{selected_policy_key}", help=CRITERIA_EXPLANATIONS.get(limit_key, 'N/A')):
                    st.info(CRITERIA_EXPLANATIONS.get(limit_key, 'Explicação não disponível.'))
        
        st.markdown("---")
        st.write("**Condições da Política (Base para Ajuste Dinâmico):**")
        for condition_key, condition_value in current_policy_details['condicoes'].items():
            display_name = condition_key.replace('_', ' ').title()
            col_cond, col_cond_info = st.columns([0.8, 0.2])
            with col_cond:
                if isinstance(condition_value, bool):
                    current_policy_details['condicoes'][condition_key] = st.checkbox(
                        display_name, value=bool(condition_value), key=f"cond_{condition_key}_{selected_policy_key}"
                    )
                elif isinstance(condition_value, int):
                    current_policy_details['condicoes'][condition_key] = st.number_input(
                        display_name, value=int(condition_value), key=f"cond_{condition_key}_{selected_policy_key}"
                    )
                elif isinstance(condition_value, float):
                    current_policy_details['condicoes'][condition_key] = st.number_input(
                        display_name, min_value=0.0, max_value=20.0, value=float(condition_value), step=0.1, format="%.2f", key=f"cond_{condition_key}_{selected_policy_key}"
                    )
                else:
                    current_policy_details['condicoes'][condition_key] = st.text_input(
                        display_name, value=str(condition_value), key=f"cond_{condition_key}_{selected_policy_key}"
                    )
            with col_cond_info:
                if st.button("ℹ️", key=f"info_cond_{condition_key}_{selected_policy_key}", help=CRITERIA_EXPLANATIONS.get(condition_key, 'N/A')):
                    st.info(CRITERIA_EXPLANATIONS.get(condition_key, 'Explicação não disponível.'))

        if st.button(f"Resetar '{selected_policy_display_name}' para Padrão", key=f"reset_policy_button_{selected_policy_key}", help="Retorna todos os critérios desta política aos seus valores originais de fábrica."):
            if st.session_state.policy_engine.reset_policy_to_default(selected_policy_key):
                st.success(f"Política '{selected_policy_display_name}' resetada para os valores padrão.")
                st.rerun()
            else:
                st.error(f"Não foi possível resetar a política '{selected_policy_display_name}'.")

    st.markdown("---")
    
    if st.button("🔎 Realizar Análise de Crédito", type="primary"):
        if analysis_mode == "Análise Individual de Cliente":
            if not client_data.cpf:
                st.error("❌ **Erro:** Dados do cliente inválidos ou não preenchidos. Verifique as entradas.")
                return

            with st.spinner("Processando análise de crédito e determinando a melhor política..."):
                policy_result = st.session_state.policy_engine.find_best_policy(client_data)

                if policy_result.get('error') and not policy_result['aprovado']:
                    st.error(f"❌ **Erro Crítico na Análise Individual:** {policy_result.get('restricoes_violadas', ['Erro desconhecido'])[0]}")
                    st.info("Por favor, verifique os dados de entrada, os ajustes de política ou contate o suporte.")
                    return

                st.subheader(f"Resultado da Análise para **{client_data.name}** (CPF: {client_data.cpf})")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"**Score de Crédito Calculado:** <span class='metric-card'>{policy_result['score_calculado']}</span>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"**Probabilidade de Inadimplência (PD):** <span class='metric-card'>{policy_result['prob_inadimplencia']:.2%}</span>", unsafe_allow_html=True)
                with col3:
                    risk_level = st.session_state.risk_analyzer.calculate_risk_level(policy_result['score_calculado'])
                    st.markdown(f"**Nível de Risco:** <span class='risk-level-{risk_level}'>{risk_level}</span>", unsafe_allow_html=True)
                with col4:
                    if policy_result['aprovado'] and policy_result['expected_loss_total'] is not None:
                        st.markdown(f"**Expected Loss (Total):** <span class='metric-card-red'>R$ {policy_result['expected_loss_total']:.2f}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Expected Loss (Total):** <span class='metric-card-red'>R$ 0.00</span>", unsafe_allow_html=True)

                if policy_result.get('revisao_humana_sugerida', False):
                    st.warning("⚠️ **ATENÇÃO:** Este caso sugere **Revisão Humana**. A decisão automatizada pode ser limítrofe ou o sistema identificou um alerta que requer análise de um especialista.")


                st.write(f"---")
                st.markdown(f"### Política Aplicada: **{policy_result['politica']}**")

                if policy_result['aprovado']:
                    st.success("✅ **APROVADO** para esta política!")
                    st.write("---")
                    st.subheader("Limites Aprovados (Dinâmicos com base em Risco)")
                    if policy_result['limites_aprovados']:
                        for tipo, limite in policy_result['limites_aprovados'].items():
                            st.write(f"- **{tipo.replace('_', ' ').title()}:** R$ {limite:,.2f}")
                            el_prod_key = f'expected_loss_{tipo.replace("_max", "").replace("_", "")}'
                            el_prod = policy_result.get(el_prod_key, None)
                            if el_prod is not None:
                                st.write(f"  (Expected Loss para este produto: R$ {el_prod:,.2f})")
                    else:
                        st.info("Nenhum limite de crédito específico definido para esta aprovação.")

                    st.subheader("Condições Aplicadas (Dinâmicas com base em Risco)")
                    if policy_result['condicoes_aplicadas']:
                        for condicao, valor in policy_result['condicoes_aplicadas'].items():
                            if 'taxa_juros_base' in condicao:
                                st.write(f"- **{condicao.replace('_', ' ').title()}:** {valor:.2f}% ao mês")
                            elif 'prazo_maximo_meses' in condicao:
                                st.write(f"- **{condicao.replace('_', ' ').title()}:** {valor} meses")
                            else:
                                st.write(f"- **{condicao.replace('_', ' ').title()}:** {valor}")
                    else:
                        st.info("Nenhuma condição específica definida para esta aprovação.")

                else:
                    st.error("❌ **REPROVADO** para todas as políticas padrão!")
                    st.subheader("Motivos da Reprovação (Direito à Explicação - LGPD Art. 20):")
                    if policy_result['restricoes_violadas']:
                        for motivo in policy_result['restricoes_violadas']:
                            st.write(f"- {motivo}")
                    else:
                        st.info("Não foi possível determinar motivos específicos, mas o cliente não atendeu aos critérios gerais.")
                    
                    if policy_result['alertas']:
                        st.subheader("Alertas e Considerações:")
                        for alerta in policy_result['alertas']:
                            st.warning(f"- {alerta}")
                    
                    if policy_result['recomendacoes']:
                        st.subheader("Recomendações para Melhoria do Perfil:")
                        for rec in policy_result['recomendacoes']:
                            st.info(f"- {rec}")

                st.write("---")
                st.subheader("Comparativo de Avaliação por Política")
                st.info("Esta tabela mostra como o cliente se sairia em cada uma das políticas disponíveis, mesmo que não tenha sido aprovado na política mais vantajosa.")
                if policy_result.get('all_policy_evaluations'):
                    policy_comparison_data = []
                    for p_key, p_eval in policy_result['all_policy_evaluations'].items():
                        policy_comparison_data.append({
                            "Política": p_eval['politica'],
                            "Status": "✅ Aprovado" if p_eval['aprovado'] else "❌ Reprovado",
                            "Score Mínimo Exigido": st.session_state.policy_engine.policies[p_key]['score_minimo'],
                            "Prob. Inadimplência Máx. Exigida": st.session_state.policy_engine.policies[p_key]['criterios']['probabilidade_inadimplencia_max'],
                            "Motivos Reprovação": ", ".join(p_eval['restricoes_violadas']) if p_eval['restricoes_violadas'] else "N/A",
                            "Expected Loss Total (se aprovado)": p_eval['expected_loss_total'] if p_eval['aprovado'] else np.nan,
                            "Sugere Revisão Humana": "Sim" if p_eval.get('revisao_humana_sugerida', False) else "Não"
                        })
                    policy_comparison_df = pd.DataFrame(policy_comparison_data)
                    st.dataframe(policy_comparison_df.style.format({"Prob. Inadimplência Máx. Exigida": "{:.2%}", "Expected Loss Total (se aprovado)": "R$ {:,.2f}"}), use_container_width=True)
                else:
                    st.info("Não foi possível gerar um comparativo de políticas.")


                st.write("---")
                st.subheader("Detalhamento do Score (Pesos e Contribuições)")

                if policy_result['detalhamento_score']:
                    score_df = pd.DataFrame([
                        {'Categoria': k.replace('_', ' ').title(), 'Score': v}
                        for k, v in policy_result['detalhamento_score'].items()
                    ])
                    fig_score = px.bar(score_df, x='Categoria', y='Score',
                                    title='Score por Categoria (0-100)',
                                    color='Score', color_continuous_scale=px.colors.sequential.Plasma)
                    st.plotly_chart(fig_score, use_container_width=True)

                    st.markdown("---")
                    st.markdown("##### Entenda as Categorias de Score:")
                    for category_key, category_score in policy_result['detalhamento_score'].items():
                        explanation_data = CRITERIA_EXPLANATIONS.get(category_key)
                        if explanation_data:
                            with st.expander(f"Saiba mais sobre {explanation_data['nome']}", expanded=False):
                                st.write(explanation_data['racional'])
                                if explanation_data.get('sub_criterios'):
                                    st.markdown("**Fatores internos detalhados:**")
                                    for sc_key, sc_val in explanation_data['sub_criterios'].items():
                                        st.markdown(f"- **{sc_key.replace('_', ' ').title()}:** {sc_val}")
                        else:
                            st.info(f"Explicação não disponível para '{category_key.replace('_', ' ').title()}'.")

                else:
                    st.warning("Não foi possível gerar o detalhamento do score devido a dados indisponíveis.")

                if policy_result['contribuicao_pesos']:
                    contribuicao_df = pd.DataFrame([
                        {'Fator': k.replace('_', ' ').title(), 'Contribuição': v}
                        for k, v in policy_result['contribuicao_pesos'].items()
                    ])
                    fig_contrib = px.pie(contribuicao_df, names='Fator', values='Contribuição',
                                        title='Contribuição Ponderada para o Score Final',
                                        hole=.3)
                    st.plotly_chart(fig_contrib, use_container_width=True)
                else:
                    st.warning("Não foi possível gerar a contribuição dos pesos devido a dados indisponíveis.")
        
        else: # analysis_mode == "Análise de Portfólio (Simulação)"
            if 'simulated_clients_df' not in st.session_state or st.session_state['simulated_clients_df'].empty:
                st.warning("Por favor, gere um portfólio de clientes simulados primeiro.")
                return
            
            st.subheader("📊 Análise de Portfólio de Clientes Simulados")

            with st.spinner("Analisando performance das políticas no portfólio..."):
                results_df, metrics, policy_approval_rates_df = analyze_policy_performance(
                    st.session_state.policy_engine, 
                    st.session_state['simulated_clients_df']
                )
            
            st.markdown("---")
            st.markdown("### 📈 Principais Métricas de Performance e Risco do Portfólio")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.markdown(f"**Taxa de Aprovação Geral:** <span class='metric-card-green'>{metrics['Taxa de Aprovação Geral']:.2%}</span>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"**Score Médio (Aprovados):** <span class='metric-card'>{metrics['Score Médio (Aprovados)']:.0f}</span>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"**Prob. Inadimplência Média (Aprovados):** <span class='metric-card'>{metrics['Prob. Inadimplência Média (Aprovados)']:.2%}</span>", unsafe_allow_html=True)
            with col4:
                st.markdown(f"**Expected Loss Total (Aprovados):** <span class='metric-card-red'>R$ {metrics['Expected Loss Total (Aprovados)']:.2f}</span>", unsafe_allow_html=True)
            with col5:
                 st.markdown(f"**Taxa de Revisão Humana:** <span class='metric-card'>{metrics['Taxa de Revisão Humana']:.2%}</span>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### Desempenho e Risco por Política")
            if not policy_approval_rates_df.empty:
                st.dataframe(policy_approval_rates_df.style.format({
                    "Taxa de Aprovação": "{:.2%}", 
                    "EL Médio Aprovado": "R$ {:,.2f}"
                }), use_container_width=True, help="Taxas de aprovação e Expected Loss médio por política, permitindo comparar o desempenho estratégico de cada uma.")
                
                fig_policy_approval = px.bar(
                    policy_approval_rates_df.reset_index(), 
                    x='Política Aplicada', y='Taxa de Aprovação',
                    title='Taxa de Aprovação por Política',
                    color='Taxa de Aprovação',
                    color_continuous_scale=px.colors.sequential.Viridis,
                    text_auto=True
                )
                fig_policy_approval.update_layout(yaxis_tickformat=".0%")
                st.plotly_chart(fig_policy_approval, use_container_width=True)

                fig_el_policy = px.bar(
                    policy_approval_rates_df.reset_index().dropna(subset=['EL Médio Aprovado']),
                    x='Política Aplicada', y='EL Médio Aprovado',
                    title='Expected Loss Médio por Cliente Aprovado por Política',
                    color='EL Médio Aprovado',
                    color_continuous_scale=px.colors.sequential.Reds,
                    text_auto=".2s"
                )
                fig_el_policy.update_layout(yaxis_tickprefix="R$ ")
                st.plotly_chart(fig_el_policy, use_container_width=True)

            else:
                st.info("Nenhuma política aprovou clientes neste portfólio simulado.")

            st.markdown("---")
            st.markdown("### Análise da Distribuição de Risco e Score")
            col_dist1, col_dist2 = st.columns(2)
            with col_dist1:
                fig_score_dist = px.histogram(
                    results_df, 
                    x='Score Calculado', 
                    nbins=50, 
                    title='Distribuição de Scores Calculados',
                    color_discrete_sequence=['#636EFA'],
                    labels={'Score Calculado': 'Score de Crédito (0-1000)'},
                    hover_data={'Score Calculado':':.0f', 'Prob. Inadimplência':':.2%'}
                )
                st.plotly_chart(fig_score_dist, use_container_width=True)
            with col_dist2:
                fig_prob_dist = px.histogram(
                    results_df, 
                    x='Prob. Inadimplência', 
                    nbins=50, 
                    title='Distribuição de Probabilidade de Inadimplência (PD)',
                    color_discrete_sequence=['#EF553B'],
                    labels={'Prob. Inadimplência': 'Probabilidade de Inadimplência'},
                    hover_data={'Score Calculado':':.0f', 'Prob. Inadimplência':':.2%'}
                )
                fig_prob_dist.update_layout(xaxis_tickformat='.0%')
                st.plotly_chart(fig_prob_dist, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### Correlação Score vs. Probabilidade de Inadimplência (PD)")
            fig_scatter = px.scatter(
                results_df, x='Score Calculado', y='Prob. Inadimplência',
                color='Aprovado',
                hover_name='Nome', hover_data=['CPF', 'Expected Loss Total', 'Política Aplicada', 'Default Event (Simulado)'],
                title='Score de Crédito vs. Probabilidade de Inadimplência (PD)',
                labels={'Score Calculado': 'Score de Crédito (0-1000)', 'Prob. Inadimplência': 'Probabilidade de Inadimplência'},
                color_discrete_map={True: 'green', False: 'red'},
                height=500
            )
            fig_scatter.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig_scatter, use_container_width=True)


            st.markdown("---")
            st.markdown("### Detalhes do Portfólio Analisado")
            st.dataframe(results_df, use_container_width=True, help="Tabela completa com os resultados da análise de cada cliente simulado.")
            
            st.markdown("""
            ---
            ### 💡 **Insights de Nível FICO para Tomada de Decisão Estratégica:**
            -   **Otimização de Portfólio (Risco vs. Retorno):** Utilize as métricas de **Taxa de Aprovação**, **Score Médio**, **Probabilidade de Inadimplência Média** e, crucialmente, o **Expected Loss Total/Médio** para ajustar suas políticas.
                -   **Abaixar requisitos de uma política** pode aumentar a taxa de aprovação, mas observe o impacto no `Expected Loss`. Você está aprovando mais clientes, mas com um risco e rentabilidade aceitáveis?
                -   **Aumentar requisitos** pode reduzir o `Expected Loss`, mas pode estar perdendo clientes "bons" que seriam rentáveis e não representariam um risco excessivo.
            -   **Calibração de Limites e Taxas:** A análise mostra como os `Limites Aprovados` e as `Taxas de Juros` se ajustam dinamicamente por cliente. Isso reflete uma precificação de risco mais sofisticada, onde o custo do crédito é proporcional ao risco individual do tomador.
            -   **Sensibilidade ao LGD:** Ajuste os sliders de **LGD (Loss Given Default)** na barra lateral. Observe como isso muda o `Expected Loss` do seu portfólio. Isso é vital para a gestão de capital, provisões bancárias e para modelar o impacto de diferentes estratégias de recuperação de dívidas.
            -   **Identificação de Segmentos de Risco (PD):** Analise a `Distribuição de PD`. Há muitos clientes com PD alta que ainda são aprovados? Suas políticas estão capturando o risco adequadamente para cada segmento de cliente? O gráfico de dispersão Score vs. PD é útil para identificar anomalias.
            -   **Testes de Estresse (Simulados):** Altere os critérios das políticas (ex: renda mínima, score mínimo) e regenere o portfólio. Observe o impacto nas métricas gerais e na aprovação por política. Isso simula cenários econômicos adversos ou mudanças na estratégia de mercado.
            -   **Estratégias Champion/Challenger:** Monitore o desempenho de políticas experimentais (`Challenger`) em comparação com as políticas atuais (`Champion`) para identificar novas regras que podem otimizar o risco-retorno.
            ---
            ### **Considerações de IA Ética e Conformidade Regulatória (LGPD e PL de IA no Brasil):**
            -   **Direito à Explicação (Art. 20 LGPD / PL IA):** O painel fornece justificativas detalhadas para as decisões de score e política. Isso é fundamental para a transparência e para que o cliente compreenda o porquê de uma decisão desfavorável.
            -   **Monitoramento de Vieses:** Em um ambiente real, seria crucial monitorar as taxas de aprovação, PDs e ELs para diferentes grupos demográficos (idade, gênero, localização, etc.) para identificar e mitigar vieses algorítmicos. O `default_event` simulado é um placeholder para dados que seriam usados em um modelo de fairness.
            -   **Revisão Humana:** A sugestão de revisão humana para casos limítrofes ou complexos está implementada. Isso garante que a automação não seja cega e que a inteligência humana possa intervir em situações críticas, conforme a regulamentação emergente de IA.
            -   **Data Governance e Auditoria:** Todas as decisões e parâmetros de políticas ajustados devem ser logados e auditáveis. Os logs gerados por este sistema seriam a base para futuras auditorias e para garantir a rastreabilidade das decisões de crédito.
            """)


# Garante que o script é executado quando o arquivo é rodado
if __name__ == "__main__":
    main()