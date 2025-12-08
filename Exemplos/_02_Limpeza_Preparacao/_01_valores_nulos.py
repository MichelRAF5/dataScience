import pandas as pd
import numpy as np
	 
# Criando um DataFrame com mais linhas e valores ausentes (NaN)
df = pd.DataFrame({
	'temperatura': [22.5, np.nan, 24.1, 23.0, np.nan, 25.3, 26.1],
	'pressao': [1.2, 1.5, np.nan, 1.4, 1.3, np.nan, np.nan],
	'vibracao': [0.5, 0.7, 0.6, np.nan, 0.9, 0.8, 1.0]
})
	 
print("DataFrame original:")
print(df)
	 
# Remover todas as linhas que tenham qualquer valor nulo
df_drop_linhas = df.dropna()
print("\nRemover linhas com qualquer valor nulo:")
print(df_drop_linhas)
	 
# Remover colunas que contenham qualquer valor nulo
df_drop_colunas = df.dropna(axis=1)
print("\nRemover colunas com qualquer valor nulo:")
print(df_drop_colunas)
	 
# Manter apenas as linhas onde a coluna 'temperatura' tem valor (não nulo)
df_filtrado = df[df['temperatura'].notnull()]
print("\nApenas linhas onde 'temperatura' tem valor:")
print(df_filtrado)
	 
# Preencher todos os valores nulos com 0
df_zero = df.fillna(0)
print("\nPreencher todos os nulos com 0:")
print(df_zero)
	 
# Preencher os nulos da coluna 'pressao' com a média da própria coluna
df_media = df.copy()
df_media['pressao'] = df_media['pressao'].fillna(df_media['pressao'].mean())
print("\nPreencher nulos da coluna 'pressao' com a média:")
print(df_media)