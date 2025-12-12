import pandas as pd

# %%
# Carregamento de dados
url= r"C:\Michel\Curso\dataScience\Exercicios\Exercicio_03\atividade3 - Dataset_sint_tico_de_voos.csv"

df_delays= pd.read_csv(url, sep=",", encoding="utf-8")

# %%
# Visão geral da estrutura
print("Shape:")
print(df_delays.shape,"\n")

print("Head:")
print(df_delays.head(),"\n")

print("Tail:")
print(df_delays.tail(),"\n")

print("Info:")
print(df_delays.info(),"\n")

# %%
# Quantificando nulos:
df_nulos= (df_delays.isnull().sum() / len(df_delays)*100)

print("Porcentagem de nulos por coluna:")
print(df_nulos,"\n")

# %%
# Tratando nulos
# Os valores nulos serão substituídos pela média da coluna
df_delays["DEP_DELAY"]= df_delays["DEP_DELAY"].fillna(df_delays["DEP_DELAY"].mean())
df_delays["ARR_DELAY"]= df_delays["ARR_DELAY"].fillna(df_delays["ARR_DELAY"].mean())

#%%
# Conversão da coluna FL_DATE para date time 
df_delays["FL_DATE"]= pd.to_datetime(df_delays["FL_DATE"], errors= 'coerce')

#%%
# Padronização de formatos
# Garantindo que as colunas CRS_DEP_TIME, DEP_DELAY, ARR_DELAY e DISTANCE estão em formato numérico
df_delays["DEP_DELAY"]= pd.to_numeric(df_delays["DEP_DELAY"])
df_delays["ARR_DELAY"]= pd.to_numeric(df_delays["ARR_DELAY"])
df_delays["DISTANCE"]= pd.to_numeric(df_delays["DISTANCE"])
df_delays["CRS_DEP_TIME"]= pd.to_numeric(df_delays["CRS_DEP_TIME"])

# Padronização de textos nas colunas OP_UNIQUE_CARRIER, ORIGIN e DEST
df_delays["OP_UNIQUE_CARRIER"]= df_delays["OP_UNIQUE_CARRIER"].str.lower().str.strip()
df_delays["ORIGIN"]= df_delays["ORIGIN"].str.lower().str.strip()
df_delays["DEST"]= df_delays["DEST"].str.lower().str.strip()

#%%
# Criando coluna binária de atraso absoluto > 15 min

for i in range (len(df_delays)):
    if df_delays.loc[i, "DEP_DELAY"] - df_delays.loc[i, "ARR_DELAY"] > 15:
        df_delays.loc[i, "ABS_DELAY_>_15MIN"]= True
        
    else:
        df_delays.loc[i, "ABS_DELAY_>_15MIN"]= False
        
print(df_delays)

#%%
# Criando dataframe filtrado de voos de origem "gig" que tiveram mais de 15 min de atraso absoluto

df_filtr= df_delays[
    (df_delays["ORIGIN"] == "gig") &
    (df_delays["ABS_DELAY_>_15MIN"] == True)
]

print(df_filtr.head())

#%%
# Estatísticas descritivas
import numpy as np

colunas_numericas= df_delays.select_dtypes(include= [np.number]).columns

for coluna in colunas_numericas:
    print(df_delays[coluna].describe())
    
#%%
# Identificação de outliers usando a regra IQR (1.5 × IQR)

# Adicionando colunas de flag para outliers em cada variável numérica
for coluna in colunas_numericas:
    Q1= df_delays[coluna].quantile(0.25)
    Q3= df_delays[coluna].quantile(0.75)
    IQR= Q1-Q3

    limite_inf= Q1-1.5*IQR
    limite_sup= Q3+1.5*IQR
    
    # Criando coluna de flag de outliers
    coluna_outlier= f"OUTLIER_{coluna}"
    df_delays[coluna_outlier]= ((df_delays[coluna] < limite_inf) | 
                                (df_delays[coluna] > limite_sup)) 
    
    
print(df_delays.head())

#%%
# Visualizações exploratórias
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# HISTPLOT Distribuição dos atrasos 


fig, axes = plt.subplots(2, 2, figsize=(15, 10))  

# Histplot para DEP_DELAY 
sns.histplot(data=df_delays, x='DEP_DELAY', kde=True, ax=axes[0, 0], bins=30)
axes[0, 0].axvline(df_delays['DEP_DELAY'].mean(), color='red', linestyle='--', 
                  label=f'Média: {df_delays["DEP_DELAY"].mean():.1f} min')
axes[0, 0].axvline(df_delays['DEP_DELAY'].median(), color='green', linestyle='--',
                  label=f'Mediana: {df_delays["DEP_DELAY"].median():.1f} min')
axes[0, 0].set_title('Distribuição do Atraso na Partida (DEP_DELAY)')
axes[0, 0].set_xlabel('Atraso (minutos)')
axes[0, 0].set_ylabel('Frequência')
axes[0, 0].legend()

# Histplot para ARR_DELAY 
sns.histplot(data=df_delays, x='ARR_DELAY', kde=True, ax=axes[0, 1], bins=30)
axes[0, 1].axvline(df_delays['ARR_DELAY'].mean(), color='red', linestyle='--',
                  label=f'Média: {df_delays["ARR_DELAY"].mean():.1f} min')
axes[0, 1].axvline(df_delays['ARR_DELAY'].median(), color='green', linestyle='--',
                  label=f'Mediana: {df_delays["ARR_DELAY"].median():.1f} min')
axes[0, 1].set_title('Distribuição do Atraso na Chegada (ARR_DELAY)')
axes[0, 1].set_xlabel('Atraso (minutos)')
axes[0, 1].set_ylabel('Frequência')
axes[0, 1].legend()

# Histplot para diferença entre atrasos 
df_delays['DIFF_DELAY'] = df_delays['DEP_DELAY'] - df_delays['ARR_DELAY']
sns.histplot(data=df_delays, x='DIFF_DELAY', kde=True, ax=axes[1, 0], bins=30)
axes[1, 0].axvline(df_delays['DIFF_DELAY'].mean(), color='red', linestyle='--',
                   label=f'Média: {df_delays["DIFF_DELAY"].mean():.1f} min')
axes[1, 0].axvline(15, color='orange', linestyle='-', linewidth=2,
                   label='Limite 15 min (ABS_DELAY)')
axes[1, 0].set_title('Diferença entre Atrasos (Partida - Chegada)')
axes[1, 0].set_xlabel('Diferença (minutos)')
axes[1, 0].set_ylabel('Frequência')
axes[1, 0].legend()

# Histplot para DISTANCE 
sns.histplot(data=df_delays, x='DISTANCE', kde=True, ax=axes[1, 1], bins=30)
axes[1, 1].axvline(df_delays['DISTANCE'].mean(), color='red', linestyle='--',
                   label=f'Média: {df_delays["DISTANCE"].mean():.1f} km')
axes[1, 1].set_title('Distribuição da Distância dos Voos')
axes[1, 1].set_xlabel('Distância (km)')
axes[1, 1].set_ylabel('Frequência')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# BOXPLOT Distribuição dos atrasos 

fig, axes = plt.subplots(1, 3, figsize=(18, 6))  

# Boxplot para DEP_DELAY e ARR_DELAY 
sns.boxplot(data=df_delays[['DEP_DELAY', 'ARR_DELAY']], ax=axes[0])
axes[0].set_title('Boxplot - Atrasos na Partida vs Chegada')
axes[0].set_ylabel('Atraso (minutos)')
axes[0].tick_params(axis='x', rotation=45)

# Boxplot por companhia aérea (top 10) 
top_carriers = df_delays['OP_UNIQUE_CARRIER'].value_counts().head(10).index
df_top_carriers = df_delays[df_delays['OP_UNIQUE_CARRIER'].isin(top_carriers)]
sns.boxplot(data=df_top_carriers, x='OP_UNIQUE_CARRIER', y='DEP_DELAY', ax=axes[1])
axes[1].set_title('Boxplot - Atraso na Partida por Companhia Aérea (Top 10)')
axes[1].set_xlabel('Companhia Aérea')
axes[1].set_ylabel('Atraso (minutos)')
axes[1].tick_params(axis='x', rotation=45)

# Boxplot por origem (top 10) 
top_origins = df_delays['ORIGIN'].value_counts().head(10).index
df_top_origins = df_delays[df_delays['ORIGIN'].isin(top_origins)]
sns.boxplot(data=df_top_origins, x='ORIGIN', y='DEP_DELAY', ax=axes[2])
axes[2].set_title('Boxplot - Atraso na Partida por Aeroporto de Origem (Top 10)')
axes[2].set_xlabel('Aeroporto de Origem')
axes[2].set_ylabel('Atraso (minutos)')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

#%%
# Heatmap de correlações para variáveis numéricas.

# Calcular matriz de correlação
colunas_numericas = df_delays.select_dtypes(include=[np.number]).columns
corr_matrix = df_delays[colunas_numericas].corr()

# Visualizar heatmap
plt.figure(figsize=(12, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Heatmap de Correlação - Variáveis Numéricas', fontsize=16, pad=20)
plt.tight_layout()
plt.show()

#%%
# Convertendo o df limpo para csv
df_delays.to_csv(r"C:\Michel\Curso\dataScience\Exercicios\Exercicio_03\flights_delay_limpo.csv", index= False, encoding="utf-8")