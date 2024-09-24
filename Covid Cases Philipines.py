#%%

# Análise de dados exploratória (EDA)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%

# Lembrar de fechar o arquivo se ele ainda estiver aberto
df = pd.read_excel('File_Path')

df.info()

df.describe()

df = df.drop(columns=['Case', 'Nationality'])

df.info()

#%%

sns.set(style="whitegrid")

# Gráfico da distribuição de idade
plt.figure(figsize=(10, 6))

sns.histplot(df['Age'], kde=True, color='blue')

plt.title('Distribuição de Idade dos Casos de COVID-19 nas Filipinas', fontsize=16)
plt.xlabel('Idade', fontsize=14)
plt.ylabel('Frequência', fontsize=14)

plt.show()


#%%


# Configurações gerais
plt.figure(figsize=(10, 6))

# Boxplot de Idade vs Status (Recuperado/Não Recuperado)
sns.boxplot(x='Status', y='Age', data=df[df['Status'].isin(['Recovered', 'Deceased'])])

# Título e rótulos
plt.title('Distribuição de Idade por Status de Recuperação', fontsize=16)
plt.xlabel('Status', fontsize=14)
plt.ylabel('Idade', fontsize=14)

# Exibir o gráfico
plt.show()

#%%

# Criar faixas etárias
df['Faixa_Etaria'] = pd.cut(df['Age'], bins=[0, 20, 40, 60, 80, 100], labels=['0-20', '21-40', '41-60', '61-80', '81+'])

# Calcular a proporção de recuperados por faixa etária
faixa_recuperacao = df[df['Status'] == 'Recovered'].groupby('Faixa_Etaria')['Status'].count() / df.groupby('Faixa_Etaria')['Status'].count()

# Plotar o gráfico de barras
faixa_recuperacao.plot(kind='bar', color='green', figsize=(10, 6))

plt.title('Proporção de Recuperação por Faixa Etária', fontsize=16)
plt.xlabel('Faixa Etária', fontsize=14)
plt.ylabel('Proporção de Recuperados', fontsize=14)
plt.show()

#%%

df = df.drop(columns=['Faixa_Etaria'])

# Converter colunas categóricas para numéricas
df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
df['Transmission'] = df['Transmission'].map({'Local': 1, 'Imported': 0})
df['Status'] = df['Status'].map({'Recovered': 1, 'Deceased': 0, 'Active': 2})  # Ajustar se houver mais status

#%%

# Gerar o mapa de calor de correlação
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Calor das Correlações', fontsize=16)
plt.show()