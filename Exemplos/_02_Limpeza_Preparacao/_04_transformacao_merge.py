import pandas as pd
	 
# DataFrame principal com dados de sensores
df = pd.DataFrame({
	'maquina': ['M1', 'M2', 'M1', 'M3', 'M2', 'M1', 'M3', 'M2'],
	'setor': ['usinagem', 'pintura', 'usinagem', 'usinagem', 'usinagem', 'montagem', 'pintura', 'usinagem'],
	'temperatura': [85, 78, 90, 88, 95, 80, 75, 100],
	'tempo_operacao': [120, 110, 130, 115, 140, 125, 100, 135]
})
	 
# Segundo DataFrame com status das máquinas
df_status = pd.DataFrame({
	'maquina': ['M1', 'M2', 'M3'],
	'status': ['ativa', 'manutenção', 'ativa']
})
	 
print("DataFrame original:")
print(df)
	 
# 1. Filtros: setor = 'usinagem' e temperatura > 80
df_filtrado = df[(df['setor'] == 'usinagem') & (df['temperatura'] > 80)]
print("\nLinhas filtradas (setor = usinagem e temperatura > 80):")
print(df_filtrado)
	 
# 2. Agrupamento: tempo médio de operação por máquina
df_agrupado = df.groupby('maquina')['tempo_operacao'].mean().reset_index()
# df.groupby('maquina'): Agrupa os dados do DataFrame df pela coluna 'maquina'e cria grupos onde todos os registros com o mesmo valor na coluna 'maquina' ficam juntos
# .reset_index(): Converte o resultado (que é uma Series com índice 'maquina') de volta para um DataFrame
print("\nTempo médio de operação por máquina:")
print(df_agrupado)
	 
# 3. Junção: adicionando status das máquinas ao DataFrame principal
df_completo = pd.merge(df, df_status, on='maquina', how='left')
# pd.merge(): Função do pandas para combinar dois DataFrames
# on='maquina': Especifica qual coluna usar como chave para o merge. Ambas as tabelas devem ter uma coluna chamada 'maquina'
# Mantém todos os registros do DataFrame da esquerda (df). Adiciona informações do DataFrame da direita (df_status) apenas onde há correspondência. Se não houver correspondência, as colunas de df_status terão valores NaN (nulos)
print("\nDataFrame com status das máquinas (merge):")
print(df_completo)