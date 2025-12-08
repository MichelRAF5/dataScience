import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#%%
# Estilo geral:

sns.set_theme(style="whitegrid")
# Configura o tema do seaborn para "whitegrid" (grade branca)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# Cria uma figura com 2 linhas e 3 colunas de subgráficos (total de 6 gráficos)
plt.subplots_adjust(hspace=0.4, wspace=0.3)
# Ajusta os espaçamentos entre os subgráficos

# Esta estrutura é muito útil para criar dashboards visuais ou comparar múltiplos gráficos relacionados em uma única figura
# Usando axes[linha,coluna] para configurar onde aparecerá cada grafico

#%%
# Gráfico de Linha:
# Utilizado para mostrar a evolução de um valor ao longo do tempo, sendo ideal para acompanhar séries temporais como temperatura, pressão,
# produção ou consumo de energia. Permite observar tendências, picos e quedas em processos contínuos.
 
dias = pd.date_range(start='2025-07-01', periods=7)
temperatura = [68, 70, 72, 74, 73, 75, 77]
axes[0, 0].plot(dias, temperatura, marker='o', color='teal')
axes[0, 0].set_title('Gráfico de Linha – Temperatura')
axes[0, 0].set_xlabel('Data')
axes[0, 0].set_ylabel('Temperatura (°C)')
axes[0, 0].tick_params(axis='x', rotation=45)
	 
#%%
# Gráfico de Dispersão:
# Esse tipo de gráfico é indicado para analisar a relação entre duas variáveis contínuas.
# Ele é muito usado em manutenção preditiva e controle de qualidade, como, por exemplo, 
# na comparação entre tempo de operação e falhas ocorridas. Facilita a visualização de padrões de correlação e anomalias.

pressao = np.random.normal(80, 5, 50)
tempo_uso = pressao * 0.4 + np.random.normal(0, 2, 50)
axes[0, 1].scatter(pressao, tempo_uso, alpha=0.7, color='tomato')
axes[0, 1].set_title('Gráfico de Dispersão – Pressão x Tempo')
axes[0, 1].set_xlabel('Pressão (bar)')
axes[0, 1].set_ylabel('Tempo (h)')

#%%	 
# Boxplot:
# Mostra a distribuição de dados com base em quartis. É útil para detectar outliers e entender a variabilidade de medições industriais, 
# como tempo de resposta de sensores ou consumo de equipamentos. Também permite comparar diferentes grupos (ex.: máquinas ou turnos).

df_box = pd.DataFrame({
	'Equipamento': ['A'] * 30 + ['B'] * 30,
	'Consumo': np.concatenate([np.random.normal(150, 10, 30), np.random.normal(160, 20, 30)])
})
sns.boxplot(x='Equipamento', y='Consumo', data=df_box, ax=axes[0, 2], palette='pastel')
axes[0, 2].set_title('Boxplot – Consumo por Equipamento')
axes[0, 2].set_xlabel('Equipamento')
axes[0, 2].set_ylabel('Consumo (kWh)')
	 
#%%
# Histograma:
# Exibe a frequência de ocorrência de valores em faixas (bins). 
# Excelente para analisar a distribuição de variáveis contínuas, como o tempo entre falhas ou o tempo de parada de máquinas. 
# Ajuda a identificar concentração, assimetria e dispersão dos dados.

tempo_falhas = np.random.normal(30, 5, 100)
axes[1, 0].hist(tempo_falhas, bins=10, color='slateblue', edgecolor='black')
axes[1, 0].set_title('Histograma – Tempo entre falhas')
axes[1, 0].set_xlabel('Tempo (min)')
axes[1, 0].set_ylabel('Frequência')
	 
#%%
# Gráfico de Área:
# Ideal para mostrar o acúmulo ou volume ao longo do tempo, como produção diária acumulada ou estoque de materiais. 
# A área preenchida sob a linha destaca a proporção total ao longo dos períodos e contribui para comparações visuais rápida.

dias = pd.date_range(start='2025-07-01', periods=7)
energia = [120, 130, 125, 140, 135, 145, 150]
axes[1, 1].fill_between(dias, energia, color='skyblue', alpha=0.5)
axes[1, 1].plot(dias, energia, color='blue')
axes[1, 1].set_title('Gráfico de Área – Consumo de Energia')
axes[1, 1].set_xlabel('Data')
axes[1, 1].set_ylabel('Energia (kWh)')
axes[1, 1].tick_params(axis='x', rotation=45)
	 
#%%
# Heatmap:
# Representa a densidade ou frequência de valores em uma matriz, com uso de cores. 
# Muito útil para cruzar variáveis categóricas e frequências (ex.: tipos de falhas por turno ou setor). 
# Ajuda a identificar padrões intensos em linhas de produção e setores específicos.

dados = np.random.randint(0, 10, size=(3, 4))
df_heat = pd.DataFrame(dados, columns=['Falha A', 'Falha B', 'Falha C', 'Falha D'],
                       index=['Turno 1', 'Turno 2', 'Turno 3'])
sns.heatmap(df_heat, annot=True, cmap='YlGnBu', fmt='d', ax=axes[1, 2])
axes[1, 2].set_title('Mapa de Calor – Falhas por Turno')
axes[1, 2].set_xlabel('Tipo de Falha')
axes[1, 2].set_ylabel('Turno')
	 
plt.suptitle('Exemplos de Gráficos', fontsize=16)
plt.show()
	 
#%%
# Gráfico de barras:
# Representa valores categóricos por meio da altura ou comprimento das barras.
# É ideal para comparar quantidades entre grupos ou categorias discretas, como setores, turnos ou linhas de produção. 
# No contexto industrial, permite visualizar facilmente quais áreas estão concentrando maior número de falhas, paradas ou alertas operacionais.

categorias = ['Linha A', 'Linha B', 'Linha C', 'Linha D']
paradas = [12, 9, 15, 7]
	 
plt.figure(figsize=(6, 4))
plt.bar(categorias, paradas, color='mediumseagreen', edgecolor='black')
plt.title('Gráfico de Barras – Paradas por linha de produção')
plt.xlabel('Linha de produção')
plt.ylabel('Número de paradas')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()