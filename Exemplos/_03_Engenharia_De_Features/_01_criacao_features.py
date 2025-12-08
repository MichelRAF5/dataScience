#%%
# Aplicar engenharia de features significa transformar dados crus em informações úteis e interpretáveis,
# por exemplo, transformar "data de manutenção" em "dias desde a última manutenção", ou consolidar medições contínuas em médias por turno.

#%%
import pandas as pd 
from datetime import datetime # importa o método datetime da biblioteca datetime
	 
# Exemplo de DataFrame com datas de última manutenção
df = pd.DataFrame({
	'equipamento': ['A', 'A', 'B', 'B'],     
	'horas_funcionamento': [100, 200, 150, 300],
	'paradas': [2, 3, 1, 5],
	'data_ultima_manutencao': ['2024-12-01', '2024-12-20', '2024-11-30', '2025-01-10']
})
	 
# Converter datas para datetime
df['data_ultima_manutencao'] = pd.to_datetime(df['data_ultima_manutencao'])
	 
# Calcular dias desde a última manutenção (com base na data atual)
hoje = pd.to_datetime(datetime.now())
df['dias_desde_manutencao'] = (hoje - df['data_ultima_manutencao']).dt.days
# .dt.days: Extrai apenas a parte dos dias da diferença (ignorando horas/minutos)
	 
# Criar nova feature: taxa de parada por hora
df['taxa_parada'] = df['paradas'] / df['horas_funcionamento']
	 
# Exibir as 5 primeiras linhas do dataframe
print(df.head())