#%%
# No caso das variáveis numéricas, muitos algoritmos de aprendizado de máquina são sensíveis à escala.
# Isso significa  que variáveis com valores muito altos podem dominar outras menores, mesmo que não sejam mais importantes.
# Para lidar com isso, duas transformações são bastante utilizadas: 

#%%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
	 
# Criando um DataFrame com colunas numéricas e não numéricas
df = pd.DataFrame({
	'maquina': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7'],
	'status': ['ativa', 'manutenção', 'ativa', 'ativa', 'inativa', 'manutenção', 'ativa'],
	'horas_funcionamento': [200, 450, 300, 500, 120, 600, 350],
	'dias_desde_manutencao': [10, 45, 30, 60, 5, 80, 35],
	'temperatura': [78.2, 80.5, 77.0, 79.8, 75.5, 81.2, 76.0]
})
	 
print("DataFrame original:")
print(df)
	 
# Selecionando automaticamente apenas as colunas numéricas
dados = df.select_dtypes(include='number')
	 
# Normalização: escala entre 0 e 1
normalizador = MinMaxScaler()  
# Cria uma instância do normalizador Min-Max, que escala os dados para o intervalo [0, 1]
dados_normalizados = normalizador.fit_transform(dados)  
# Aplica o ajuste e transformação dos dados (normalização)
df_normalizado = pd.DataFrame(dados_normalizados, columns=dados.columns)  
# Converte o resultado para um DataFrame com os mesmos nomes de colunas do original
print("\nDados normalizados (0 a 1):")
print(df_normalizado)
	 
# Padronização: média 0, desvio padrão 1
padronizador = StandardScaler()  
# Cria uma instância do StandardScaler, que padroniza os dados com média 0 e desvio padrão 1
dados_padronizados = padronizador.fit_transform(dados)  
# Ajusta e transforma os dados, aplicando a padronização
df_padronizado = pd.DataFrame(dados_padronizados, columns=dados.columns)  
# Converte os dados padronizados para um DataFrame mantendo os nomes das colunas
	 
print("\nDados padronizados (média 0, desvio 1):")
print(df_padronizado)