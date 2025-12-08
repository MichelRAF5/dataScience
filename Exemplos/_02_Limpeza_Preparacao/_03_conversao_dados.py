import pandas as pd
import numpy as np
	 
# Criando um DataFrame com problemas de tipo: texto em colunas numéricas e categoria textual
df = pd.DataFrame({
	'temperatura': ['22.5', '23.0', 'erro', '24.1', ''],
	'status_maquina': ['ativa', 'inativa', 'ativa', 'em manutenção', 'ativa'],
	'corrente': ['1.2', '--', '1.5', '1.3', '']
})
	 
print("DataFrame original:")
print(df)
	 
# Conversão de texto para número: transforma valores inválidos em NaN
df['temperatura'] = pd.to_numeric(df['temperatura'], errors='coerce')
# errors='coerce': os valores que podem ser conertidos para número são convertidos e os que não podem são transformados em NaN (Not a Number)
	 
# Conversão de colunas categóricas para economizar memória e organizar valores fixos
df['status_maquina'] = df['status_maquina'].astype('category')
# .astype('category') converte a coluna de strings/texto para o tipo categoria (valores que pertencem a um conjunto fixo e limitado de categorias)
	 
# Substituição de símbolos inválidos e conversão de texto para número na coluna 'corrente'
df['corrente'] = df['corrente'].replace("--", np.nan)
df['corrente'] = pd.to_numeric(df['corrente'], errors='coerce')
	 
print("\nDataFrame após conversões:")
print(df)