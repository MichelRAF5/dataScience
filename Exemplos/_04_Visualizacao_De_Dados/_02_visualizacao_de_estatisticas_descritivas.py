import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
	 
sns.set_theme(style="whitegrid")  # Estilo visual para gráficos Seaborn e Matplotlib

#%%
# Gráfico de linha com média destacada:
# Se a média da produção diária for 130 unidades, e um dia apresentar 90, isso pode indicar uma falha pontual.
# Uma linha horizontal marcando a média ajuda a visualizar desvios críticos.

dias = pd.date_range(start='2025-07-01', periods=7)
producao = [130, 125, 90, 140, 132, 128, 135]
media = sum(producao) / len(producao)
	 
df_producao = pd.DataFrame({'Data': dias, 'Produção': producao})
	 
plt.figure(figsize=(8, 4))
plt.plot(df_producao['Data'], df_producao['Produção'], marker='o', color='steelblue', label='Produção diária')
plt.axhline(media, color='red', linestyle='--', label=f'Média: {media:.1f}')
plt.title('Produção diária com média destacada')
plt.xlabel('Data')
plt.ylabel('Unidades produzidas')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
	 
#%%
# Boxplot com variabilidade de consumo por equipamento:
# O boxplot mostra a dispersão dos dados, mas o desvio-padrão ajuda a quantificar essa variação. 
# Juntos, eles ajudam a identificar se uma máquina está operando de forma instável.

consumo_a = np.random.normal(150, 5, 30)
consumo_b = np.random.normal(150, 15, 30)
	 
df_consumo = pd.DataFrame({
    'Equipamento': ['A'] * 30 + ['B'] * 30,
    'Consumo_kWh': np.concatenate([consumo_a, consumo_b])
})
	 
plt.figure(figsize=(6, 4))
sns.boxplot(x='Equipamento', y='Consumo_kWh', data=df_consumo, palette='pastel')
plt.title('Consumo por Equipamento (variabilidade)')
plt.ylabel('Consumo (kWh)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#%%	 
# Histograma com assimetria – Tempo de parada:
# Se o histograma estiver concentrado à esquerda com uma “cauda” à direita, temos um padrão assimétrico. 
# Isso pode indicar que a maioria das paradas é rápida, mas algumas são muito longas, um sinal de alerta.

paradas = np.concatenate([np.random.normal(15, 3, 80), np.random.normal(45, 5, 20)])
	 
plt.figure(figsize=(8, 4))
plt.hist(paradas, bins=15, color='slateblue', edgecolor='black')
plt.title('Distribuição do tempo de parada (assimétrica)')
plt.xlabel('Tempo de parada (min)')
plt.ylabel('Frequência')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
	 
#%%
# Gráfico de dispersão com linha de tendência – Temperatura x Falhas:
# Se os pontos estiverem alinhados em diagonal, pode haver uma relação direta entre duas variáveis. 
# Isso nem sempre exige um cálculo de correlação, o padrão salta aos olhos no gráfico.

temperatura = np.random.normal(70, 5, 50)
falhas = temperatura * 0.3 + np.random.normal(0, 1, 50)
	 
coef = np.polyfit(temperatura, falhas, 1)
reta = np.poly1d(coef)
	 
plt.figure(figsize=(8, 4))
plt.scatter(temperatura, falhas, color='darkorange', alpha=0.7, label='Dados observados')
plt.plot(np.sort(temperatura), reta(np.sort(temperatura)), color='black', linestyle='--', label='Tendência linear')
plt.title('Temperatura x Número de Falhas')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Falhas por dia')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()