import pandas as pd
	 
# Criando um DataFrame com variações em texto e datas despadronizadas
df = pd.DataFrame({
	'resposta_checklist': ['Sim', ' sim', 'SIM', 'Sim ', 'sIm'],
	'data_leitura': ['2023-10-01', '01/10/2023', '10/01/23', 'erro', '']
})
	 
print("DataFrame original:")
print(df)
	 
# Padronização de textos: minúsculas + remoção de espaços extras
df['resposta_checklist'] = df['resposta_checklist'].str.lower().str.strip()
# str.strip() remove espaços em branco no início e fim
	 
# Padronização de datas: conversão para datetime
df['data_leitura'] = pd.to_datetime(df['data_leitura'], errors='coerce')
# to_datetime() converte a coluna em um formato de data-hora
	 
# Extração de partes da data
df['ano'] = df['data_leitura'].dt.year
df['mes'] = df['data_leitura'].dt.month
df['dia'] = df['data_leitura'].dt.day
# dt.year, dt.month e dt.day: extraem o ano, mês e dia, respectivamente, da coluna de data-hora.
	 
print("\nDataFrame após padronização:")
print(df)