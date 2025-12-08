import pandas as pd         # Manipulação e análise de dados (DataFrames, leitura de arquivos)
import seaborn as sns       # Visualização de dados estatísticos com gráficos
import matplotlib.pyplot as plt  # Criação de gráficos e personalização de visualizações
import numpy as np          # Operações matemáticas e manipulação de arrays numéricos
	 
# ---------- 1. Criando o DataFrame com variáveis industriais ----------
df = pd.DataFrame({
	'temperatura_motor': [68, 70, 72, 74, 73, 75, 77],
	'pressao_sistema':   [80, 79, 81, 78, 82, 77, 79],
	'vibracao':          [2.1, 2.3, 2.5, 2.7, 2.8, 3.0, 3.1],
	'energia_consumida': [150, 155, 160, 165, 163, 170, 172], 
	'eficiencia':        [0.91, 0.89, 0.87, 0.85, 0.84, 0.82, 0.81]  
})
	 
# ---------- 2. Calculando a matriz de correlação ----------
corr = df.corr(numeric_only=True)
	 
# ---------- 3. Plotando o heatmap da correlação ----------
plt.figure(figsize=(8, 4))  # Tamanho da figura
	 
sns.heatmap(
	corr,                      # Matriz de correlação
	annot=True,                # Mostra os números nas células
	cmap='RdBu_r',             # Azul para positivo, vermelho para negativo
	vmin=-1, vmax=1,           # Limites da escala de cores
	mask=np.triu(corr),        # Oculta metade superior da matriz
	linewidths=0.5,            # Linhas entre células
	fmt=".2f"                  # Formato dos números
)
	 
# ---------- 4. Personalizando título e eixos ----------
plt.title("Mapa de Correlação entre Variáveis Industriais")  # Define o título do gráfico
plt.xticks(rotation=45)                                      # Rotaciona os rótulos do eixo X em 45 graus para melhor visualização
plt.tight_layout()                                           # Ajusta automaticamente os elementos do gráfico para evitar sobreposição
plt.show()                                                   # Exibe o gráfico na tela