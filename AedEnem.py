import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt
from scipy import stats


# Definindo estilo
sns.set_style("darkgrid")

# Importando dados
dados = pd.read_csv('ENEM_2018.csv', sep = ';', encoding = 'cp1252')

# Verificando dados
dados.shape
dados.dtypes
dados.head()

# Criando amostra aleatória simples
amostra = dados.sample(n = 3000)

# Realizando saneamento dos dados, primeiro iremos excluir as colunas que não teram valor para nossa análise
amostra = amostra.drop([
    'NU_INSCRICAO', 'NU_ANO', 'CO_MUNICIPIO_RESIDENCIA', 'CO_UF_RESIDENCIA', 'CO_ESCOLA', 'CO_MUNICIPIO_ESC', 'CO_UF_ESC', 'TP_LOCALIZACAO_ESC',
    'CO_MUNICIPIO_NASCIMENTO', 'NO_MUNICIPIO_NASCIMENTO', 'CO_UF_NASCIMENTO', 'SG_UF_NASCIMENTO', 'TP_SIT_FUNC_ESC', 'IN_BAIXA_VISAO',
    'IN_CEGUEIRA', 'IN_SURDEZ', 'IN_DEFICIENCIA_AUDITIVA', 'IN_SURDO_CEGUEIRA', 'IN_DEFICIENCIA_FISICA', 'IN_DEFICIENCIA_MENTAL',
    'IN_DEFICIT_ATENCAO', 'IN_DISLEXIA', 'IN_DISCALCULIA', 'IN_AUTISMO', 'IN_VISAO_MONOCULAR', 'IN_OUTRA_DEF', 'IN_GESTANTE', 'IN_LACTANTE',
    'IN_IDOSO', 'IN_ESTUDA_CLASSE_HOSPITALAR', 'IN_SEM_RECURSO', 'IN_BRAILLE', 'IN_AMPLIADA_24', 'IN_AMPLIADA_18', 'IN_LEDOR', 'IN_ACESSO',
    'IN_TRANSCRICAO', 'IN_LIBRAS', 'IN_LEITURA_LABIAL', 'IN_MESA_CADEIRA_RODAS', 'IN_MESA_CADEIRA_SEPARADA', 'IN_APOIO_PERNA', 'IN_GUIA_INTERPRETE',
    'IN_COMPUTADOR', 'IN_CADEIRA_ESPECIAL', 'IN_CADEIRA_CANHOTO', 'IN_CADEIRA_ACOLCHOADA', 'IN_PROVA_DEITADO', 'IN_MOBILIARIO_OBESO',
    'IN_LAMINA_OVERLAY', 'IN_PROTETOR_AURICULAR', 'IN_MEDIDOR_GLICOSE', 'IN_MAQUINA_BRAILE', 'IN_SOROBAN', 'IN_MARCA_PASSO', 'IN_SONDA',
    'IN_MEDICAMENTOS', 'IN_SALA_INDIVIDUAL', 'IN_SALA_ESPECIAL', 'IN_SALA_ACOMPANHANTE', 'IN_MOBILIARIO_ESPECIFICO', 'IN_MATERIAL_ESPECIFICO',
    'IN_NOME_SOCIAL', 'CO_MUNICIPIO_PROVA', 'CO_UF_PROVA', 'CO_PROVA_CN', 'CO_PROVA_CH', 'CO_PROVA_LC', 'CO_PROVA_MT', 'TX_RESPOSTAS_CN',
    'TX_RESPOSTAS_CH', 'TX_RESPOSTAS_LC', 'TX_RESPOSTAS_MT', 'TX_GABARITO_CN', 'TX_GABARITO_CH', 'TX_GABARITO_LC', 'TX_GABARITO_MT',
    'Q003', 'Q004', 'Q007', 'Q008', 'Q009', 'Q010', 'Q011', 'Q012', 'Q013', 'Q014', 'Q015', 'Q016', 'Q017', 'Q018', 'Q019', 'Q020', 'Q021', 'Q022', 'Q023'
], axis = 1)

# Tratando variáveis qualitativas
amostra['Q025'] = amostra['Q025'].map({'A':'Não', 'B':'Sim'})
amostra['TP_SEXO'] = amostra['TP_SEXO'].map({'F':'Feminino', 'M':'Masculino'})
amostra['TP_COR_RACA'] = amostra['TP_COR_RACA'].map({0:'NA', 6:'NA', 1:'Branca', 2:'Preta', 3:'Parda', 4:'Amarela', 5:'Indígena'})
amostra['Q027'] = amostra['Q027'].map({'A':'Publica', 'B':'Pub e Pri', 'C':'Pub e Pri', 'D':'Privada', 'E':'Publica', 'F':'Nao Frequentou Escola'})
amostra['Q006'] = amostra['Q006'].map({
    'A':'0', 'B':'954', 'C':'1431', 'D':'1908', 'E':'2385', 'F':'2862', 'G':'3816', 'H':'4770', 'I':'5724',
    'J':'6678', 'K':'7632', 'L':'8586', 'M':'9540', 'N':'11448', 'O':'14310', 'P':'19080', 'Q':'19081'
})

amostra.Q027.value_counts()
amostra = amostra.loc[amostra.Q027 != 'Nao Frequentou Escola']


# ANÁLISE DAS VARIÁVEIS QUALITATIVAS
# Análisando sexo
amostra.TP_SEXO.value_counts()
amostra.TP_SEXO.value_counts() / amostra.shape[0]

sns.countplot(amostra.TP_SEXO).set_title("Sexo")
plt.xlabel("")
plt.ylabel("Frequência")
plt.show()

# Análisando cor da pele
amostra.TP_COR_RACA.value_counts()
amostra.TP_COR_RACA.value_counts() / amostra.shape[0]

sns.countplot(amostra.TP_COR_RACA).set_title("Cor da Pele")
plt.xlabel("")
plt.ylabel("Frequência")
plt.show()

# Análisando cor da pele x sexo
sns.countplot(amostra.TP_COR_RACA, hue = amostra.TP_SEXO).set_title("Cor da Pele x Sexo")
plt.xlabel("")
plt.ylabel("Frequência")
plt.show()

# Análisando UF
amostra.SG_UF_RESIDENCIA.value_counts()
amostra.SG_UF_RESIDENCIA.value_counts() / amostra.shape[0]

sns.countplot(amostra.SG_UF_RESIDENCIA).set_title("Estados")
plt.xlabel("")
plt.ylabel("Frequência")
plt.show()

# Análisando tipo de escolas
amostra.Q027.value_counts()
amostra.Q027.value_counts() / amostra.shape[0]

sns.countplot(amostra.Q027).set_title("Tipo de Escolas")
plt.xlabel("")
plt.ylabel("Frequência")
plt.show()

# Análisando tipo de escolas x cor da pele
sns.countplot(amostra.Q027, hue = amostra.TP_COR_RACA).set_title("Tipo de Escolas x Cor da Pele")
plt.xlabel("")
plt.ylabel("Frequência")
plt.show()

# Análisando tipo de escolas x acesso a internet
sns.countplot(amostra.Q027, hue = amostra.Q025).set_title("Tipo de Escolas x Acesso a Internet")
plt.xlabel("")
plt.ylabel("Frequência")
plt.show()


# ANÁLISE DAS VARIÁVEIS QUANTITATIVAS
amostra.dtypes
amostra.NU_IDADE
amostra.NU_NOTA_MT
amostra.NU_NOTA_REDACAO

# Convertendo renda mensal da familia para float
amostra['Q006'] = pd.to_numeric(amostra['Q006'], errors = 'coerce')

# Verificando valores nulos
amostra.isnull().sum().sort_values(ascending = False)

# Tratando valores nulos e nota 0 de NU_NOTA_MT, NU_NOTA_REDACAO
amostra['NU_NOTA_MT'].fillna(0, inplace = True)
amostra['NU_NOTA_REDACAO'].fillna(0, inplace = True)

# Análisando renda
amostra.Q006.describe()
amostra.groupby(['Q027', 'TP_SEXO']).agg({'Q006': 'mean'})

sns.boxplot(x = amostra.TP_SEXO, y = amostra.Q006, hue = amostra.Q027, showfliers = False).set_title("Sexo x Renda x Tp Escola")
plt.xlabel("")
plt.ylabel("Renda")
plt.show()

# Análisando nota da redação
amostra.NU_NOTA_REDACAO.describe()
amostra.groupby('TP_SEXO').agg({'NU_NOTA_REDACAO': 'mean'})
amostra.groupby(['Q027', 'TP_SEXO']).agg({'NU_NOTA_REDACAO': 'mean'})

sns.boxplot(x = amostra.TP_SEXO, y = amostra.NU_NOTA_REDACAO, showfliers = False).set_title("Sexo x Nota de Redação")
plt.xlabel("")
plt.ylabel("Nota")
plt.show()

sns.boxplot(x = amostra.TP_SEXO, y = amostra.NU_NOTA_REDACAO, hue = amostra.Q027, showfliers = False).set_title("Sexo x Nota de Redação x Tp Escola")
plt.xlabel("")
plt.ylabel("Nota")
plt.show()

# Análisando nota de matemática
amostra.NU_NOTA_MT.describe()
amostra.groupby('TP_SEXO').agg({'NU_NOTA_MT': 'mean'})
amostra.groupby(['Q027', 'TP_SEXO']).agg({'NU_NOTA_MT': 'mean'})

sns.boxplot(x = amostra.TP_SEXO, y = amostra.NU_NOTA_MT, showfliers = False).set_title("Sexo x Nota de Matemática")
plt.xlabel("")
plt.ylabel("Nota")
plt.show()

sns.boxplot(x = amostra.TP_SEXO, y = amostra.NU_NOTA_MT, hue = amostra.Q027, showfliers = False).set_title("Sexo x Nota de Matemática x Tp Escola")
plt.xlabel("")
plt.ylabel("Nota")
plt.show()

# Verificando a correlação de algumas variáveis quantitativas
feature = ['Q006', 'NU_IDADE', 'TP_ST_CONCLUSAO', 'TP_ESCOLA', 'TP_ENSINO', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']
corr = amostra[feature].corr()

plt.subplots(figsize = (11, 8))
sns.heatmap(corr, annot = True, cmap = 'YlGnBu',  annot_kws = {"size": 10})

# Verificando distribuição de notas matemática
amostra = amostra.loc[(amostra['NU_NOTA_MT'] != 0) & (amostra['NU_NOTA_REDACAO'] != 0)]

sns.distplot(amostra.loc[amostra.TP_SEXO == 'Masculino', 'NU_NOTA_MT'], hist = True)
sns.distplot(amostra.loc[amostra.TP_SEXO == 'Feminino', 'NU_NOTA_MT'], hist = True)
plt.title('Nota de Matemática')
plt.legend(labels = ['Mas','Fem'], ncol = 2, loc = 'upper left')
plt.show()

# Verificando distribuição de notas redação
sns.distplot(amostra.loc[amostra.TP_SEXO == 'Masculino', 'NU_NOTA_REDACAO'], hist = True)
sns.distplot(amostra.loc[amostra.TP_SEXO == 'Feminino', 'NU_NOTA_REDACAO'], hist = True)
plt.title('Nota de Redação')
plt.legend(labels = ['Mas','Fem'], ncol = 2, loc = 'upper left')
plt.show()

# Realizando teste de Shapiro Wilk
stats.shapiro(amostra.loc[amostra.TP_SEXO == 'Masculino', 'NU_NOTA_MT'])
stats.shapiro(amostra.loc[amostra.TP_SEXO == 'Feminino', 'NU_NOTA_MT'])

stats.shapiro(amostra.loc[amostra.TP_SEXO == 'Masculino', 'NU_NOTA_REDACAO'])
stats.shapiro(amostra.loc[amostra.TP_SEXO == 'Feminino', 'NU_NOTA_REDACAO'])

# Relizando teste de hipótese de Mann Whitney para notas de redação e matemática
stats.mannwhitneyu(
    amostra.loc[amostra.TP_SEXO == 'Masculino', 'NU_NOTA_MT'],
    amostra.loc[amostra.TP_SEXO == 'Feminino', 'NU_NOTA_MT']
)

stats.mannwhitneyu(
    amostra.loc[amostra.TP_SEXO == 'Masculino', 'NU_NOTA_REDACAO'],
    amostra.loc[amostra.TP_SEXO == 'Feminino', 'NU_NOTA_REDACAO']
)
